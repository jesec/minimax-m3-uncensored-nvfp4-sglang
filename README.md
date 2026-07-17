# MiniMax-M3-uncensored-NVFP4 · SGLang serving image

[![build](https://github.com/jesec/minimax-m3-uncensored-nvfp4-sglang/actions/workflows/build.yml/badge.svg)](https://github.com/jesec/minimax-m3-uncensored-nvfp4-sglang/actions/workflows/build.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![gpu](https://img.shields.io/badge/GPU-4%C3%97%20Blackwell%20(NVFP4)-76b900)

A ready-to-run container that serves [`ressl/MiniMax-M3-uncensored-NVFP4`](https://huggingface.co/ressl/MiniMax-M3-uncensored-NVFP4)
— the 428B-total / 23B-active NVFP4 MoE — with [SGLang](https://github.com/sgl-project/sglang).

The two SGLang compatibility patches that the model card mounts by hand (`-v`) are
**baked into the image**, and the model weights are **downloaded automatically on
first launch**. So deployment needs no code mounts and no configuration: point a
persistent volume at the container and start it.

> [!WARNING]
> This image serves a genuinely **uncensored / abliterated** model that will comply
> with requests a stock model refuses. It is intended for lawful security research,
> red-teaming, and analysis (as stated on the model card). You are responsible for
> how you use it and for complying with the MiniMax model license and applicable law.

## Features

- **Patches baked in** — no `-v` code mounts; the validated ModelOpt + FlashInfer
  fixes ship inside the image.
- **Zero-config model download** — the model identity is hard-coded and the volume
  path is auto-detected; a fresh volume is populated on first boot.
- **Resumable & self-healing** — an interrupted download resumes on the next start
  instead of re-pulling 260 GB or launching on half-downloaded weights.
- **Visible progress** — the multi-hour download reports `download progress: X GB /
  ~260 GB (N%)` to the container logs, even without a TTY.
- **Reproducible** — the base image is pinned by digest to the exact SGLang build
  the model was validated against.
- **Build in CI** — a GitHub Actions workflow builds and pushes the ~30 GB image so
  you never need a big local disk.

## Quick start

Requires **4× NVIDIA Blackwell GPUs** (SM120/SM100 — e.g. RTX PRO 6000 Blackwell or
B200). NVFP4 is a Blackwell hardware feature; Ada/Hopper cards cannot run it.

### RunPod (recommended)

Create a **Pod** template:

| Setting | Value |
|---|---|
| Container image | `jesec/minimax-m3-uncensored-nvfp4-sglang:latest` |
| Container disk | 60 GB |
| Volume disk | ≥ 350 GB |
| Volume mount path | `/workspace` *(RunPod default — auto-detected)* |
| HTTP port | `30000` |
| Start command | *(leave blank)* |
| Environment variables | *(none required — add `NCCL_P2P_DISABLE=1` if boot hangs at NCCL init; see [Troubleshooting](#troubleshooting))* |

Deploy it on **4× RTX PRO 6000 Blackwell**. First boot downloads the model into
`/workspace/model` on the volume (watch the logs for progress), then serves an
OpenAI-compatible API at `https://<POD_ID>-30000.proxy.runpod.net/v1/...`. Later
starts of the same pod skip the download.

### Docker

```bash
docker run --rm --runtime=nvidia --gpus all --ipc=host --shm-size 32g \
  -v /path/to/persistent/dir:/model \
  -p 30000:30000 \
  jesec/minimax-m3-uncensored-nvfp4-sglang:latest
```

Already have the weights? Mount them at `/model` (read-only is fine) and they're
served directly, with no download.

### Call it

```bash
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ressl/MiniMax-M3-uncensored-NVFP4",
  "messages": [{"role": "user", "content": "hello"}],
  "temperature": 1.0, "top_p": 0.95
}'
```

## How it works

**Baked-in patches.** The `Dockerfile` starts from the pinned SGLang `dev-minimax-m3`
image and `COPY`s the two files from `sglang_patch/` to the exact in-tree paths the
model card mounts over. They make SGLang's ModelOpt Cutlass path execute MiniMax-M3's
parameterized clamped-SwiGLU correctly. A build-time `py_compile` fails the build if a
patch is ever truncated. Those two files are vendored from
[Mapika/MiniMax-M3-NVFP4](https://huggingface.co/Mapika/MiniMax-M3-NVFP4) — see
[Credits & license](#credits--license) for their origin.

**Weight location (auto-detected).** `serve.sh` picks `MODEL_PATH` without any env var,
preferring a mounted persistent volume so the download survives restarts:

1. an explicit `MODEL_PATH`, if you set one; otherwise
2. `/workspace/model` — a volume at `/workspace` (RunPod Pod default);
3. `/runpod-volume/model` — a volume at `/runpod-volume` (RunPod network volume);
4. `/model` — a `docker run -v host:/model` mount, or an ephemeral fallback (with a
   warning that the model will re-download on every restart).

**Provisioning.** On first boot it downloads the model with `hf_transfer` (fast,
parallel) and writes a `.download_complete` marker on the volume. Restarts see the
marker and boot straight into serving; an interrupted download (no marker) resumes.
A pre-populated volume (`config.json` present, no markers) is served untouched.

**Launch.** `serve.sh` then execs `sglang.launch_server` with the model card's exact
validated flags (TP=4, `modelopt_fp4`, FlashInfer Cutlass MoE/FP4 backends,
`--page-size 128`, `--disable-shared-experts-fusion`, …). The container keeps the base
NVIDIA entrypoint, so GPU setup happens before the server starts.

## Configuration

Everything works with no configuration. Defaults reproduce the model card's validated
4× RTX PRO 6000 Blackwell (TP=4) setup; override any of these via environment variables:

| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | *auto* (`/workspace/model`, `/runpod-volume/model`, or `/model`) | where weights live / are downloaded |
| `SERVED_MODEL_NAME` | `ressl/MiniMax-M3-uncensored-NVFP4` | `--served-model-name` |
| `PORT` | `30000` | server port |
| `TP_SIZE` | *auto* (detected GPU count) | tensor-parallel size — the model supports 2, 4, or 8 |
| `CONTEXT_LENGTH` | `32768` | `--context-length` |
| `MEM_FRACTION_STATIC` | `0.90` | `--mem-fraction-static` |
| `MAX_RUNNING_REQUESTS` | `2` | `--max-running-requests` |
| `CHUNKED_PREFILL_SIZE` | `16384` | `--chunked-prefill-size` |
| `PAGE_SIZE` | `128` | `--page-size` (required by MiniMax Sparse Attention) |
| `HF_TOKEN` | *(unset)* | Hub auth for the download (gated repos / rate limits) |
| `HF_REVISION` | *(unset)* | pin a specific model revision to download |
| `DISABLE_FLASHINFER_AUTOTUNE` `DISABLE_PREFILL_CUDA_GRAPH` `DISABLE_SHARED_EXPERTS_FUSION` `DISABLE_CUSTOM_ALL_REDUCE` | `1` | the validated `--disable-*` stability flags; set one to `0` to drop that flag (e.g. `DISABLE_CUSTOM_ALL_REDUCE=0` on NVLink systems) |
| `EXTRA_ARGS` | *(unset)* | extra flags appended to the launch command (whitespace-split) |

The model repo is **hard-coded**, not an env var — this image serves
`ressl/MiniMax-M3-uncensored-NVFP4` only. Recommended sampling (from the base
checkpoint): temperature `1.0`, top-p `0.95`.

`EXTRA_ARGS` is whitespace-split, so it can only carry flags whose values contain no
spaces. For a value with spaces (e.g. `--json-model-override-args '{...}'`), pass it as
a positional container command argument instead — those are forwarded verbatim.

## Build & publish

### GitHub Actions (recommended)

The image is ~30 GB, which does not fit on many workstations. The included workflow
(`.github/workflows/build.yml`) builds it on GitHub's runners and pushes on every push
to `main`, on `v*` tags, and on manual dispatch.

- **GHCR** (`ghcr.io/<owner>/minimax-m3-uncensored-nvfp4-sglang`) — always, using the
  built-in `GITHUB_TOKEN`. After the first push, set the package visibility to **public**
  so RunPod can pull it without auth.
- **Docker Hub** (`<user>/minimax-m3-uncensored-nvfp4-sglang`) — additionally, if you add
  two repository secrets:
  - `DOCKERHUB_USERNAME` — e.g. `jesec`
  - `DOCKERHUB_TOKEN` — a Docker Hub access token

Trigger it from the Actions tab (**Run workflow**) or by pushing to `main`.

### Local (needs Docker + ~60 GB free disk)

```bash
./build.sh                                   # -> minimax-m3-uncensored-nvfp4-sglang:latest
printf '%s' "$DOCKERHUB_TOKEN" | docker login -u <user> --password-stdin
NAMESPACE=<user> ./publish.sh                # -> docker.io/<user>/<image>:latest
```

## Repository layout

```
Dockerfile                 base (pinned by digest) + patches + entrypoint
serve.sh                   volume detection, model download, launch wrapper
healthcheck.py             /health probe (stays green during the download)
sglang_patch/              the two baked-in SGLang compatibility patches
build.sh / publish.sh      local build & push helpers
.github/workflows/build.yml  CI: build and push the image
```

## GPU & sizing notes

- **GPUs:** 4× Blackwell (SM120/SM100), ~96 GB each, TP=4. Validated on 4× RTX PRO 6000
  Blackwell.
- **Disk:** ~30 GB for the image (container disk) + ~260 GB for the weights (volume).
- **Shared memory:** SGLang TP=4 wants large `/dev/shm` (`--ipc=host --shm-size 32g`).
  RunPod pods normally provide enough; `--disable-custom-all-reduce` is on to reduce IPC
  pressure. NCCL/shared-memory errors at load time point here. A silent *hang* right
  after the NCCL version line is a different problem — see [Troubleshooting](#troubleshooting).
- **Serverless:** not recommended — a 260 GB model makes cold starts impractical. Use a
  GPU Pod with a persistent volume.

## Other GPU configurations

The image is validated on 4× RTX PRO 6000 Blackwell (TP=4); other layouts need only
environment variables — and `TP_SIZE` now defaults to the detected GPU count, so most
of them are zero-config too. The model's 64 attention heads / 4 KV heads support
**TP of 2, 4, or 8 only** (6 GPUs cannot work: 64 % 6 ≠ 0 — a 6-GPU pod can at best
run `TP_SIZE=4` with two cards idle).

The model itself supports up to 1M positions (`max_position_embeddings`); the default
`CONTEXT_LENGTH=32768` is the KV budget of the validated setup, not a model limit.
KV costs ~30 KiB per token per rank at TP≥4 (~60 KiB at TP=2), plus ~25 % for the
sparse-attention indexer cache — that is what sizes the "context that fits" column:

| Config | Extra environment | Free VRAM for KV (per GPU) | Context that fits | Status |
|---|---|---|---|---|
| 4× RTX PRO 6000 | *(none)* | ~17 GB | 32k default, ~256k max | **validated** |
| 8× RTX PRO 6000 | `NCCL_P2P_DISABLE=1` on most hosts | ~48 GB | ~512k–1M | untested — index-head caveat |
| 2× B200 | *(none)* | ~43 GB | ~256–384k | should work — SM100 is FlashInfer's primary NVFP4 target |
| 4× B200 | *(none)* | ~104 GB | 1M | should work |
| 8× B200 | *(none)* | ~134 GB | 1M | untested — index-head caveat |
| 2× B300 | *(none)* | ~129 GB | 1M | untested — SM103 kernels unverified |
| 4× B300 | *(none)* | ~190 GB | 1M | untested — SM103 kernels unverified |

- **NVLink systems (B200/B300, any count).** Do **not** set `NCCL_P2P_DISABLE=1` —
  it would disable NVLink peer-to-peer. Consider `DISABLE_CUSTOM_ALL_REDUCE=0` to
  restore SGLang's custom all-reduce, which NVLink systems benefit from. PCIe-only
  RTX PRO 6000 hosts are the opposite: see [Troubleshooting](#troubleshooting).
- **TP=8 (any card).** The 4 KV heads are replicated across ranks (standard GQA
  handling), but whether this build also replicates the model's 4 sparse-attention
  index heads at TP=8 is unverified — a head-divisibility error at load means it
  does not. On 8× RTX PRO 6000 specifically, TP=8 all-reduce over PCIe also taxes
  every token; unless you need a single instance with >32k context, prefer **two
  TP=4 pods** — the validated config at double throughput. (Provision the volume
  with one pod first, then attach it to both: the download markers are not safe for
  two pods downloading into the same volume concurrently.)
- **B300 (SM103).** The pinned image predates broad B300 validation; if its kernels
  lack SM103 support, the first forward fails with "no kernel image is available",
  and fixing that means a newer base image plus re-validated patches.
- **Long context.** Raise `CONTEXT_LENGTH` (power-of-two-ish values like `262144`,
  `524288`, `1048576`) and drop `MAX_RUNNING_REQUESTS=1` for a single maximum-length
  request. Decode speed is nearly context-independent (MiniMax sparse attention),
  but prefilling a full 1M-token prompt still takes minutes.

## Troubleshooting

- **Hangs right after `sglang is using nccl==2.28.9` (GPU and CPU pinned at 100 %,
  VRAM flat).** NCCL peer-to-peer deadlock. RTX PRO 6000 Blackwell has no NVLink, so
  TP all-reduce runs over PCIe P2P — which deadlocks on many multi-tenant hosts
  (ACS/IOMMU), especially when `nvidia-smi topo -mp` shows GPU pairs connected via
  `SYS` (cross-NUMA). Fix: set the environment variable **`NCCL_P2P_DISABLE=1`** and
  restart — NCCL falls back to shared memory and distributed init completes in
  seconds. Known issue: [sgl-project/sglang#15181](https://github.com/sgl-project/sglang/issues/15181).
  Only set this on PCIe-only cards like the RTX PRO 6000: on NVLink systems (B200/B300)
  it would disable NVLink P2P too, where it is neither needed nor wanted.
- **Looks stuck after `Load weight begin` (GPU 0 %, `nvidia-smi` frozen for ~15 min).**
  Normal. The ~243 GB of weights stream from the (network) volume with little log
  output until `Multi-threaded loading of shards: 100%`; budget 10–20 minutes per
  boot when loading from a RunPod network volume.
- **`Tokenizer ... is still TokenizersBackend after retries with --trust-remote-code`.**
  Benign. MiniMax-M3 ships a plain `PreTrainedTokenizerFast` (there is no custom
  tokenizer class to find), so SGLang's retry has nothing to upgrade to. The chat
  template and the `minimax-m3` tool-call/reasoning parsers work regardless.

## Credits & license

- Model: [`ressl/MiniMax-M3-uncensored-NVFP4`](https://huggingface.co/ressl/MiniMax-M3-uncensored-NVFP4)
  by Robert Ressl, quantized from [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3).
- SGLang compatibility patches (`sglang_patch/`): vendored from
  [Mapika/MiniMax-M3-NVFP4](https://huggingface.co/Mapika/MiniMax-M3-NVFP4), revision
  `668435825700a0047399441720f430bdd8eca0ab` — `modelopt_quant.py` originates from the
  [vLLM project](https://github.com/vllm-project/vllm) and `flashinfer_trtllm.py` from
  [SGLang](https://github.com/sgl-project/sglang). Both are **Apache-2.0** and keep their upstream license.
- Engine: [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0).

This repository's own code (Dockerfile, scripts, CI) is licensed under **MIT** (see [LICENSE](LICENSE)).
The two vendored files under `sglang_patch/` are **not** covered by MIT — they keep their upstream
**Apache-2.0** license (a copy is included as [LICENSE-APACHE](LICENSE-APACHE)) and origin (above). The
model weights are **not** included and remain under the MiniMax model license — see the model card.
