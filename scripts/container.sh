#!/usr/bin/env bash
# Docker interface for the shared Isaac image (reused by Ray + the HPC .sif). Decoupled from the
# IsaacLab source tree: nvcr isaacsim base + pip Isaac Lab, workspace code mounted at job start.
set -euo pipefail
cd "$(dirname "$0")"

cmd="${1:-build}"
case "$cmd" in
    build)
        shift || true
        docker/build_image.sh "$@"
        ;;
    -h | --help | help)
        echo "usage: $(basename "$0") build   -- build the shared Isaac docker image (hcrl-isaac:latest)"
        ;;
    *)
        echo "[ERROR] unknown command '$cmd' (try: build)" >&2
        exit 1
        ;;
esac
