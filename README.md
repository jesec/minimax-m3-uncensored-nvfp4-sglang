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
| Environment variables | *(none)* |

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
| `TP_SIZE` | `4` | tensor-parallel size (= GPU count) |
| `CONTEXT_LENGTH` | `32768` | `--context-length` |
| `MEM_FRACTION_STATIC` | `0.90` | `--mem-fraction-static` |
| `MAX_RUNNING_REQUESTS` | `2` | `--max-running-requests` |
| `CHUNKED_PREFILL_SIZE` | `16384` | `--chunked-prefill-size` |
| `PAGE_SIZE` | `128` | `--page-size` (required by MiniMax Sparse Attention) |
| `HF_TOKEN` | *(unset)* | Hub auth for the download (gated repos / rate limits) |
| `HF_REVISION` | *(unset)* | pin a specific model revision to download |
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
  pressure. NCCL/shared-memory errors at load time point here.
- **Serverless:** not recommended — a 260 GB model makes cold starts impractical. Use a
  GPU Pod with a persistent volume.

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
