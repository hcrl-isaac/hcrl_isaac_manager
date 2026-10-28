#!/usr/bin/env bash
# Docker interface for the shared Isaac image (reused by Ray + the HPC .sif). Ubuntu 24.04 + a uv venv from the
# Isaac Lab 3 lock; workspace code is mounted at job start.
set -euo pipefail
cd "$(dirname "$0")"  # scripts/docker/

# The Isaac Lab ref and uv extras the image installs: workspace.defaults.yaml's (the `isaaclab:` block's `ref: X` and
# `extras: [a, b]` lines) plus isaacsim (Kit), unless ISAACLAB_REF / ISAACLAB_EXTRAS are set. Sets ref and extras.
isaaclab_args() {
    local defaults=../../workspace.defaults.yaml
    ref="${ISAACLAB_REF:-}" extras="${ISAACLAB_EXTRAS:-}"
    [ -n "$ref" ] || ref="$(sed -nE 's/^[[:space:]]+ref:[[:space:]]*([^[:space:]#]+).*/\1/p' "$defaults" | head -1)"
    [ -n "$extras" ] || extras="isaacsim $(sed -nE 's/^[[:space:]]+extras:[[:space:]]*\[([^]]*)\].*/\1/p' "$defaults" | head -1 | tr ',' ' ')"
    [ -n "$ref" ] || { echo "[docker][ERROR] no isaaclab.ref in workspace.defaults.yaml" >&2; exit 1; }
    extras="$(echo $extras)"  # one space between names
}

# Build the shared image from ./Dockerfile.
build_image() {
    local image="${HCRL_IMAGE_NAME:-hcrl-isaac}:${HCRL_IMAGE_TAG:-latest}" ref extras
    isaaclab_args
    echo "[docker] building ${image}: Isaac Lab ${ref}, extras: ${extras}"
    docker build \
        ${BASE_IMAGE:+--build-arg BASE_IMAGE="${BASE_IMAGE}"} \
        --build-arg ISAACLAB_REF="${ref}" \
        --build-arg ISAACLAB_EXTRAS="${extras}" \
        -t "${image}" \
        -f Dockerfile .
    echo "[docker] done: ${image}"
}

case "${1:-build}" in
    build)          build_image ;;
    # the same Isaac Lab inputs as KEY=VALUE lines, for the .sif built on an arm64 cluster (hcrl-isaac.def)
    isaaclab-args)  isaaclab_args; printf 'ISAACLAB_REF=%s\nISAACLAB_EXTRAS=%s\n' "$ref" "$extras" ;;
    -h | --help | help)
        echo "usage: $(basename "$0") build          -- build the shared Isaac docker image (hcrl-isaac:latest)"
        echo "       $(basename "$0") isaaclab-args  -- print the Isaac Lab ref and uv extras the image installs" ;;
    *)              echo "[ERROR] unknown command '${1}' (try: build)" >&2; exit 1 ;;
esac
