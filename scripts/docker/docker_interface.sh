#!/usr/bin/env bash
# Docker interface for the shared Isaac image (reused by Ray + the HPC .sif). Decoupled from the
# IsaacLab source tree: nvcr isaacsim base + pip Isaac Lab, workspace code mounted at job start.
set -euo pipefail
cd "$(dirname "$0")"  # scripts/docker/

# Build the shared image from ./Dockerfile; Isaac Sim version defaults to the workspace.defaults.yaml pin.
build_image() {
    local image="${HCRL_IMAGE_NAME:-hcrl-isaac}:${HCRL_IMAGE_TAG:-latest}"
    local base="${ISAACSIM_BASE_IMAGE:-nvcr.io/nvidia/isaac-sim}"
    local version="${ISAACSIM_VERSION:-}"
    [ -n "$version" ] || version="$(grep -E '^[[:space:]]*version:' ../../workspace.defaults.yaml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
    version="${version:-5.1.0}"

    echo "[docker] building ${image} FROM ${base}:${version}"
    docker build \
        --build-arg ISAACSIM_BASE_IMAGE="${base}" \
        --build-arg ISAACSIM_VERSION="${version}" \
        -t "${image}" \
        -f Dockerfile .
    echo "[docker] done: ${image}"
}

case "${1:-build}" in
    build)          build_image ;;
    -h | --help | help) echo "usage: $(basename "$0") build   -- build the shared Isaac docker image (hcrl-isaac:latest)" ;;
    *)              echo "[ERROR] unknown command '${1}' (try: build)" >&2; exit 1 ;;
esac
