#!/usr/bin/env bash
set -euo pipefail

# Launch wrapper for MiniMax-M3-uncensored-NVFP4 on SGLang.
#
# This image serves exactly ONE model, whose identity is hard-coded below, so the
# container needs NO environment configuration to work: attach a writable volume
# at $MODEL_PATH and launch — it downloads the model on first boot and serves it.
# Operational knobs (TP size, context length, etc.) still accept env overrides.
# Defaults reproduce the model card's validated 4x RTX PRO 6000 Blackwell (TP=4).

# Fixed model identity — do not make this depend on the environment.
MODEL_REPO="ressl/MiniMax-M3-uncensored-NVFP4"
MODEL_SIZE_GB=260   # approximate download size, used only for progress reporting

# --- Where the model lives (auto-detected; no env var required) ---------------
# An explicit MODEL_PATH env var always wins. Otherwise prefer a mounted
# persistent volume so the ~260 GB download survives restarts. RunPod mounts a
# Pod volume at /workspace and a network volume at /runpod-volume by default; a
# plain `docker run -v host:/model` shows up as a mount at /model. If nothing is
# mounted, fall back to /model on the (ephemeral) container disk.
_is_mount() {
  local d=$1
  [[ -d $d ]] || return 1
  if command -v mountpoint >/dev/null 2>&1; then
    mountpoint -q "$d"
  else
    # Fallback without util-linux: a mount point sits on a different device than its parent.
    local a b
    a=$(stat -c %d "$d" 2>/dev/null) && b=$(stat -c %d "$d/.." 2>/dev/null) && [[ $a != "$b" ]]
  fi
}

MODEL_ON_VOLUME=0
if [[ -n "${MODEL_PATH:-}" ]]; then
  if _is_mount "$MODEL_PATH" || _is_mount "$(dirname "$MODEL_PATH")"; then MODEL_ON_VOLUME=1; fi
elif _is_mount /workspace; then
  MODEL_PATH=/workspace/model; MODEL_ON_VOLUME=1
elif _is_mount /runpod-volume; then
  MODEL_PATH=/runpod-volume/model; MODEL_ON_VOLUME=1
elif _is_mount /model; then
  MODEL_PATH=/model; MODEL_ON_VOLUME=1
else
  MODEL_PATH=/model
fi

if [[ $MODEL_ON_VOLUME == 1 ]]; then
  echo "[serve] MODEL_PATH=$MODEL_PATH (persistent volume detected)."
else
  echo "[serve] WARNING: MODEL_PATH=$MODEL_PATH is not on a mounted volume — the model will"
  echo "[serve]          re-download on every restart. Attach a persistent volume to avoid that."
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_REPO}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
DTYPE="${DTYPE:-auto}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
PAGE_SIZE="${PAGE_SIZE:-128}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-minimax-m3}"
REASONING_PARSER="${REASONING_PARSER:-minimax-m3}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_cutlass}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_cutlass}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"

# --- Model provisioning -------------------------------------------------------
# The weights live at $MODEL_PATH (attach a writable persistent volume there).
# Decide what to do WITHOUT relying on any env var:
#   .download_complete present  -> already fetched by us, serve as-is
#   .download_started present    -> our previous download was interrupted, resume it
#   config.json present (no markers) -> model was pre-provisioned by hand, serve as-is
#   otherwise                    -> fresh volume, download the model now
# .download_started/.download_complete live on the (persistent) volume; the
# /tmp provisioning flag is ephemeral so the healthcheck knows a download is live.
HF_REVISION="${HF_REVISION:-}"

# --- Readiness signal for the container healthcheck ---------------------------
# healthcheck.py reports healthy while /tmp/.model-starting exists, so the
# container is never marked unhealthy during the (possibly long) download or the
# multi-minute weight load that follows. A background watcher removes the flag
# only once SGLang's /health actually answers — after that the real probe takes
# over. This covers the whole startup without racing a fixed HEALTHCHECK window.
STARTING_FLAG=/tmp/.model-starting
: > "$STARTING_FLAG"
(
  set +e
  while [[ -e "$STARTING_FLAG" ]]; do
    sleep 15
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
      rm -f "$STARTING_FLAG"
    fi
  done
) &

NEED_DOWNLOAD=0
if [[ -e "$MODEL_PATH/.download_complete" ]]; then
  echo "[serve] Model already present at $MODEL_PATH — skipping download."
elif [[ -e "$MODEL_PATH/.download_started" ]]; then
  echo "[serve] Found an interrupted download at $MODEL_PATH — resuming."
  NEED_DOWNLOAD=1
elif [[ -e "$MODEL_PATH/config.json" ]]; then
  echo "[serve] Using pre-provisioned model at $MODEL_PATH — skipping download."
else
  echo "[serve] No model at $MODEL_PATH — downloading $MODEL_REPO (~${MODEL_SIZE_GB} GB)."
  NEED_DOWNLOAD=1
fi

if [[ "$NEED_DOWNLOAD" == 1 ]]; then
  if ! ( mkdir -p "$MODEL_PATH" && touch "$MODEL_PATH/.download_started" ) 2>/dev/null; then
    echo "[serve] ERROR: need to download the model but $MODEL_PATH is not writable." >&2
    echo "[serve]        Attach a writable persistent volume at $MODEL_PATH." >&2
    exit 1
  fi
  # Cache on the (persistent) volume, fast parallel transfer, unbuffered output so
  # progress is visible live in the container logs. HF_TOKEN is picked up from the
  # environment automatically if set (only needed for gated repos / rate limits).
  export HF_HOME="${HF_HOME:-$MODEL_PATH/.hf-cache}"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  export PYTHONUNBUFFERED=1

  # Heartbeat: report cumulative downloaded size every 30s. Guarantees visible
  # progress even where tqdm's live bar does not render (non-TTY log capture).
  ( set +e
    while true; do
      sleep 30
      bytes=$(du -scb "$MODEL_PATH"/*.safetensors 2>/dev/null | tail -1 | cut -f1)
      awk -v b="${bytes:-0}" -v t="$MODEL_SIZE_GB" \
        'BEGIN{g=b/1073741824; printf "[serve] download progress: %.1f GB / ~%d GB (%.0f%%)\n", g, t, (t>0? g/t*100 : 0)}'
    done ) &
  HEARTBEAT_PID=$!
  trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

  if command -v hf >/dev/null 2>&1; then DL=(hf download); else DL=(huggingface-cli download); fi
  echo "[serve] Downloading with: ${DL[*]} $MODEL_REPO --local-dir $MODEL_PATH (hf_transfer, progress below)"
  "${DL[@]}" "$MODEL_REPO" --local-dir "$MODEL_PATH" ${HF_REVISION:+--revision "$HF_REVISION"}

  kill "$HEARTBEAT_PID" 2>/dev/null || true
  trap - EXIT
  touch "$MODEL_PATH/.download_complete"
  echo "[serve] Download complete ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1) at $MODEL_PATH)."
fi

if [[ ! -e "$MODEL_PATH/config.json" ]]; then
  echo "[serve] ERROR: no usable model at $MODEL_PATH (missing config.json after provisioning)." >&2
  exit 1
fi

# EXTRA_ARGS is intentionally whitespace-split (globbing disabled so tokens with
# * or [ ] are not rewritten against the working dir). A single flag whose VALUE
# contains spaces (e.g. --json-model-override-args '{...}') cannot be carried this
# way — pass it as a positional container arg instead; $@ passes through verbatim.
# shellcheck disable=SC2206
set -f
EXTRA=(${EXTRA_ARGS:-})
set +f

set -x
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --quantization "$QUANTIZATION" \
  --trust-remote-code \
  --dtype "$DTYPE" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --page-size "$PAGE_SIZE" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --reasoning-parser "$REASONING_PARSER" \
  --moe-runner-backend "$MOE_RUNNER_BACKEND" \
  --fp4-gemm-backend "$FP4_GEMM_BACKEND" \
  --attention-backend "$ATTENTION_BACKEND" \
  --disable-flashinfer-autotune \
  --disable-prefill-cuda-graph \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  "${EXTRA[@]}" "$@"
