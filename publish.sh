#!/usr/bin/env bash
set -euo pipefail

# Tag and push the built image to a public registry.
#
# Log in first, e.g.:
#   docker login                              # Docker Hub
#   echo "$GHCR_PAT" | docker login ghcr.io -u <user> --password-stdin
#
# Then:
#   REGISTRY=docker.io NAMESPACE=<your-user-or-org> ./publish.sh
#   REGISTRY=ghcr.io   NAMESPACE=<your-user-or-org> ./publish.sh

REGISTRY="${REGISTRY:-docker.io}"                 # docker.io | ghcr.io | quay.io ...
NAMESPACE="${NAMESPACE:?set NAMESPACE to your registry user/org}"
IMAGE="${IMAGE:-minimax-m3-uncensored-nvfp4-sglang}"
TAG="${TAG:-latest}"
ENGINE="${ENGINE:-docker}"

LOCAL="${IMAGE}:${TAG}"
REMOTE="${REGISTRY}/${NAMESPACE}/${IMAGE}:${TAG}"

echo ">> Tagging ${LOCAL} -> ${REMOTE}"
"$ENGINE" tag "$LOCAL" "$REMOTE"
echo ">> Pushing ${REMOTE}"
"$ENGINE" push "$REMOTE"
echo ">> Published ${REMOTE}"
echo ">> RunPod image field:  ${REMOTE}"
