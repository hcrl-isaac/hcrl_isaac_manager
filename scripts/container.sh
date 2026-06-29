#!/usr/bin/env bash
# Docker interface for the shared Isaac image (reused by Ray and the HPC .sif).
#
# Replaces the old IsaacLab container.py-driven flow (which built FROM the IsaacLab source tree). The
# new image is decoupled from that source: isaacsim from the nvcr base + Isaac Lab from pip, workspace
# code mounted at job start. See scripts/docker/.
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
