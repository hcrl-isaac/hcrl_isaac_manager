#!/usr/bin/env bash
# Build the HPC Apptainer (.sif) image from the shared Isaac docker image (hcrl-isaac:latest),
# decoupled from the IsaacLab source tree -- Apptainer just converts the same image Ray uses, so the
# two deploy paths share one image. Builds the docker image first if it is missing.
set -euo pipefail
cd "$(dirname "$0")/../.."  # manager root

IMAGE_NAME="${HCRL_IMAGE_NAME:-hcrl-isaac}"
IMAGE_TAG="${HCRL_IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
SIF_DIR="${HCRL_SIF_DIR:-scripts/cluster/exports}"
SIF_PATH="${SIF_DIR}/${IMAGE_NAME}.sif"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[build_sif] docker image $IMAGE not found; building it first"
    scripts/container.sh build
fi

mkdir -p "$SIF_DIR"
echo "[build_sif] apptainer build ${SIF_PATH} from docker-daemon://${IMAGE}"
apptainer build --force "$SIF_PATH" "docker-daemon://${IMAGE}"
echo "[build_sif] done: ${SIF_PATH}"
