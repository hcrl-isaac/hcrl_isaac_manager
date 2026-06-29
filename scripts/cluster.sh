#!/usr/bin/env bash
# Cluster interface: build/push the shared Isaac .sif and submit jobs.
#
# Decoupled from the IsaacLab source tree -- the .sif is the SAME shared image Ray uses
# (scripts/docker/), Apptainer-converted (scripts/cluster/build_sif.sh), so the two deploy paths share
# one image and the cluster build no longer requires resources/IsaacLab. `CLUSTER=<name>` selects a
# cluster config under scripts/cluster/<name>_config (default: "default").
set -euo pipefail
[ -z "${CLUSTER:-}" ] && CLUSTER="default"
cd "$(dirname "$0")"  # scripts/

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
    add-cluster)
        cluster/add_cluster.sh "$@"
        ;;
    build)
        # Build the shared .sif from the shared docker image (no push).
        cluster/build_sif.sh "$@"
        ;;
    setup)
        # Build the shared .sif, then push it to the cluster.
        CONFIG_DIR="cluster/${CLUSTER}_config"
        if [ ! -d "$CONFIG_DIR" ]; then
            echo "[ERROR] no cluster config $CONFIG_DIR -- run 'just add-cluster' first." >&2
            exit 1
        fi
        cluster/build_sif.sh
        # NOTE: the SSH push + on-node run path (cluster_interface.sh / run_singularity.sh) still needs
        # the layout decouple to use the new shared .sif (bind /workspace/ext + /isaac-sim/python.sh).
        # See scripts/cluster/README.md.
        CLUSTER="$CLUSTER" cluster/cluster_interface.sh repush "$@"
        ;;
    -h | --help | help)
        echo "usage: CLUSTER=<name> $(basename "$0") <command> [args]"
        echo "  setup         build the shared .sif and push it to the cluster"
        echo "  build         build the .sif from the shared docker image (no push)"
        echo "  add-cluster   create a cluster config (scripts/cluster/<name>_config)"
        echo "  job [args]    submit a job (delegates to cluster_interface.sh)"
        echo "  develop ...   manage a persistent dev node (delegates to cluster_interface.sh)"
        ;;
    *)
        # Delegate job / develop / push / repush / etc. to the existing cluster interface.
        CLUSTER="$CLUSTER" cluster/cluster_interface.sh "$cmd" "$@"
        ;;
esac
