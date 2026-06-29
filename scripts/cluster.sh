#!/usr/bin/env bash
# Cluster interface: build/push the shared Isaac .sif and submit jobs.
#
# Decoupled from the IsaacLab source tree -- the .sif is the SAME shared image Ray uses
# (scripts/docker/), Apptainer-converted, so the two deploy paths share one image and the cluster build
# no longer requires resources/IsaacLab. `CLUSTER=<name>` selects a cluster config under
# scripts/cluster/config/<name> (default: "default").
set -euo pipefail
[ -z "${CLUSTER:-}" ] && CLUSTER="default"
cd "$(dirname "$0")"  # scripts/

IMAGE_NAME="${HCRL_IMAGE_NAME:-hcrl-isaac}"
SIF_DIR="${HCRL_SIF_DIR:-cluster/exports}"
SIF_PATH="${SIF_DIR}/${IMAGE_NAME}.sif"

# Reuse the persistent SSH control master (opened by `cluster_dev.sh start`) so push needs no 2FA.
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${HOME}/.ssh/cm/%C" -o ControlPersist=48h -o ConnectTimeout=60)

load_cluster_env() {
    local env_file="cluster/config/${CLUSTER}/.env.cluster"
    [ -f "$env_file" ] || { echo "[ERROR] no $env_file -- run 'just add-cluster' first." >&2; exit 1; }
    # shellcheck disable=SC1090
    source "$env_file"
}

# Build the HPC Apptainer (.sif) from the shared docker image (building the image first if needed).
build_sif() {
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "[cluster] apptainer not found -- install it (see scripts/cluster/README.md) and retry." >&2
        exit 1
    fi
    local image="${IMAGE_NAME}:${HCRL_IMAGE_TAG:-latest}"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "[cluster] docker image $image not found; building it first"
        ./container.sh build
    fi
    mkdir -p "$SIF_DIR"
    echo "[cluster] apptainer build ${SIF_PATH} from docker-daemon://${image}"
    apptainer build --force "$SIF_PATH" "docker-daemon://${image}"
    echo "[cluster] done: ${SIF_PATH}"
}

# rsync the built .sif to the cluster (a .sif is a single compressed SquashFS file -- no tar/extract;
# rsync is resumable, which matters for a multi-GB file). Reuses the persistent SSH master (no 2FA).
push_sif() {
    load_cluster_env
    [ -f "$SIF_PATH" ] || { echo "[ERROR] $SIF_PATH not built -- run 'cluster.sh build' first." >&2; exit 1; }
    echo "[cluster] pushing ${SIF_PATH} -> ${CLUSTER_LOGIN}:${CLUSTER_SIF_PATH}/${IMAGE_NAME}.sif"
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "mkdir -p '${CLUSTER_SIF_PATH}'"
    rsync -rlptvh --info=progress2 -e "ssh ${SSH_OPTS[*]}" "$SIF_PATH" "${CLUSTER_LOGIN}:${CLUSTER_SIF_PATH}/"
    echo "[cluster] pushed."
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
    add-cluster)
        cluster/add_cluster.sh "$@"
        ;;
    build)
        # Build the shared .sif from the shared docker image (no push).
        build_sif
        ;;
    push | repush)
        # rsync the (already-built) .sif to the cluster. `build` first if it's missing.
        [ -f "$SIF_PATH" ] || build_sif
        push_sif
        ;;
    setup)
        # Build the shared .sif and push it to the cluster.
        build_sif
        push_sif
        ;;
    -h | --help | help)
        echo "usage: CLUSTER=<name> $(basename "$0") <command> [args]"
        echo "  setup         build the shared .sif and rsync it to the cluster"
        echo "  build         build the .sif from the shared docker image (no push)"
        echo "  push/repush   rsync the built .sif to the cluster (reuses the SSH master; no 2FA)"
        echo "  add-cluster   create a cluster config (scripts/cluster/config/<name>)"
        echo "  job [args]    submit a job (delegates to cluster_interface.sh)"
        echo "  develop ...   manage a persistent dev node (delegates to cluster_interface.sh)"
        ;;
    *)
        # Delegate job / develop / etc. to the existing cluster interface.
        CLUSTER="$CLUSTER" cluster/cluster_interface.sh "$cmd" "$@"
        ;;
esac
