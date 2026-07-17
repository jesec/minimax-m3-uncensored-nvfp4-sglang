#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Build the serving image. Run on a host with ~60 GB free disk and network
# access (base layer ~13 GB compressed, final image ~30 GB). No GPU needed.
# For most workstations, prefer the GitHub Actions workflow instead.
#
#   IMAGE=... TAG=... ENGINE=docker ./build.sh

IMAGE="${IMAGE:-minimax-m3-uncensored-nvfp4-sglang}"
TAG="${TAG:-latest}"
ENGINE="${ENGINE:-docker}"   # docker | podman | buildah

# podman/buildah default to OCI format, which drops the Dockerfile HEALTHCHECK.
# Force Docker format for them; `docker build` rejects --format, so keep it off there.
FMT=()
if [[ "$ENGINE" != docker ]]; then FMT=(--format docker); fi

echo ">> Building ${IMAGE}:${TAG} with ${ENGINE} (platform linux/amd64)"
"$ENGINE" build "${FMT[@]}" --platform linux/amd64 -t "${IMAGE}:${TAG}" .
echo ">> Built ${IMAGE}:${TAG}"
