#!/usr/bin/env bash
# Plain `docker build` of scripts/docker/Dockerfile -- the shared image used by BOTH Ray and the HPC .sif
# (nvcr isaacsim base + pip Isaac Lab, workspace code mounted + PYTHONPATH'd at job start; see entrypoint.sh).
set -euo pipefail
cd "$(dirname "$0")"

IMAGE_NAME="${HCRL_IMAGE_NAME:-hcrl-isaac}"
IMAGE_TAG="${HCRL_IMAGE_TAG:-latest}"
ISAACSIM_BASE_IMAGE="${ISAACSIM_BASE_IMAGE:-nvcr.io/nvidia/isaac-sim}"
# Default the Isaac Sim version to the pin in workspace.yaml (the isaacsim version field).
if [ -z "${ISAACSIM_VERSION:-}" ]; then
    ISAACSIM_VERSION="$(grep -E '^[[:space:]]*version:' ../../workspace.yaml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
fi
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"

echo "[build_image] building ${IMAGE_NAME}:${IMAGE_TAG} FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}"
docker build \
    --build-arg ISAACSIM_BASE_IMAGE="${ISAACSIM_BASE_IMAGE}" \
    --build-arg ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile .
echo "[build_image] done: ${IMAGE_NAME}:${IMAGE_TAG}"
