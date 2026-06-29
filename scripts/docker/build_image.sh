#!/usr/bin/env bash
# Build the shared Isaac image used by BOTH Ray and the HPC Apptainer (.sif) build.
#
# A plain `docker build` of scripts/docker/Dockerfile -- no IsaacLab container.py, no source COPY:
# isaacsim/Kit from the nvcr base, Isaac Lab from pip, workspace code mounted + editable-installed at
# job start (see entrypoint.sh). The resulting image is what `scripts/cluster.sh` converts to a .sif and
# what the Ray job config runs, so the two deploy paths share one image.
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
