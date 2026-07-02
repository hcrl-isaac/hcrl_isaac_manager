#!/usr/bin/env bash
# Single cluster entrypoint (used directly by `just cluster`). Builds/pushes the shared Isaac .sif,
# submits batch jobs, and drives the persistent dev node. CLUSTER=<name> selects config/<name>/.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CLUSTER="${CLUSTER:-default}"

IMAGE_NAME="${HCRL_IMAGE_NAME:-hcrl-isaac}"
SIF_DIR="${HCRL_SIF_DIR:-${SCRIPT_DIR}/exports}"
SIF_PATH="${SIF_DIR}/${IMAGE_NAME}.sif"
CLUSTER_ENV_FILE="${SCRIPT_DIR}/config/${CLUSTER}/.env.cluster"

# Reuse the persistent SSH control master (opened by `cluster_dev.sh start`) so push/job need no 2FA.
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${HOME}/.ssh/cm/%C" -o ControlPersist=48h -o ConnectTimeout=60)

source_cluster_env() {
    if [ ! -f "$CLUSTER_ENV_FILE" ]; then
        echo "[ERROR] Cluster config not found: $CLUSTER_ENV_FILE (run 'just cluster add'). Available:" \
            "$(ls "$SCRIPT_DIR/config" 2>/dev/null | paste -sd, -)." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CLUSTER_ENV_FILE"
}

ensure_ssh_master() {
    echo "[INFO] Opening SSH control master to $CLUSTER_LOGIN (enter 2FA once)"
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" true
}

# Build the HPC Apptainer (.sif) from the shared docker image (building the image first if needed).
build_sif() {
    command -v apptainer >/dev/null 2>&1 || { echo "[cluster] apptainer not found (see README)." >&2; exit 1; }
    local image="${IMAGE_NAME}:${HCRL_IMAGE_TAG:-latest}"
    docker image inspect "$image" >/dev/null 2>&1 || { echo "[cluster] image $image missing; building"; "${SCRIPT_DIR}/../docker/docker_interface.sh" build; }
    mkdir -p "$SIF_DIR"
    echo "[cluster] apptainer build ${SIF_PATH} from docker-daemon://${image}"
    apptainer build --force "$SIF_PATH" "docker-daemon://${image}"
}

# rsync the built .sif to the cluster (single compressed SquashFS file -- no tar/extract; resumable).
push_sif() {
    source_cluster_env
    [ -f "$SIF_PATH" ] || { echo "[ERROR] $SIF_PATH not built -- run 'build' first." >&2; exit 1; }
    echo "[cluster] pushing ${SIF_PATH} -> ${CLUSTER_LOGIN}:${CLUSTER_SIF_PATH}/"
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "mkdir -p '${CLUSTER_SIF_PATH}'"
    rsync -rlptvh --info=progress2 -e "ssh ${SSH_OPTS[*]}" "$SIF_PATH" "${CLUSTER_LOGIN}:${CLUSTER_SIF_PATH}/"
}

# Submit a batch job on the login node: the cluster's submit_job_slurm.sh holds its #SBATCH config and
# runs scripts/cluster/run_singularity.sh in the shared hcrl-isaac.sif.
submit_job() {
    case "$CLUSTER_JOB_SCHEDULER" in
        SLURM) job_script=submit_job_slurm.sh ;;
        PBS)   job_script=submit_job_pbs.sh ;;
        *) echo "[ERROR] Unsupported CLUSTER_JOB_SCHEDULER '$CLUSTER_JOB_SCHEDULER' (SLURM|PBS)" >&2; exit 1 ;;
    esac
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" \
        "cd $CLUSTER_ISAACLAB_DIR && bash scripts/cluster/config/${CLUSTER}/${job_script} \"$CLUSTER_ISAACLAB_DIR\" hcrl-isaac ${*}"
}

cmd_job() {
    source_cluster_env
    # Sync to a timestamped dir so concurrent jobs don't clobber each other's code copy.
    CLUSTER_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR}_$(date +"%Y%m%d_%H%M%S")"
    ensure_ssh_master
    echo "[INFO] Syncing workspace to ${CLUSTER_ISAACLAB_DIR}..."
    # Keep the source package .git dirs so W&B captures the commit + uncommitted diff.
    rsync -rvh -e "ssh ${SSH_OPTS[*]}" --rsync-path="mkdir -p $CLUSTER_ISAACLAB_DIR && rsync" \
        --include="resources/IsaacLab/source/*/.git/***" --exclude="*.git*" \
        --exclude="ilab/" --exclude="wandb/" --exclude="logs/" --exclude=".vscode/" --exclude="__pycache__" \
        --exclude="scripts/cluster/exports/" --exclude="*.sif" \
        "$SCRIPT_DIR/../.." "$CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR"
    # Stage THIS cluster's env over the synced workspace-level copy -- run_singularity.sh on the compute
    # node sources scripts/cluster/.env.cluster, which otherwise holds whatever cluster was set up last.
    echo "[INFO] Staging ${CLUSTER} env + W&B creds into the synced workspace..."
    rsync -vh -e "ssh ${SSH_OPTS[*]}" "$CLUSTER_ENV_FILE" \
        "$CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR/scripts/cluster/.env.cluster"
    if [ -f "$SCRIPT_DIR/../.env.wandb" ]; then
        rsync -vh -e "ssh ${SSH_OPTS[*]}" "$SCRIPT_DIR/../.env.wandb" \
            "$CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR/scripts/cluster/.env.wandb"
    else
        echo "[WARN] scripts/.env.wandb not found -- the job will run without W&B credentials."
    fi
    echo "[INFO] Submitting job..."
    submit_job "$@"
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
    add)         "${SCRIPT_DIR}/add_cluster.sh" "$@" ;;
    build)       build_sif ;;
    push | repush)
        [ -f "$SIF_PATH" ] || build_sif
        push_sif
        ;;
    setup)       build_sif; push_sif ;;
    job)         cmd_job "$@" ;;
    develop)     exec env CLUSTER="$CLUSTER" "${SCRIPT_DIR}/cluster_dev/cluster_dev.sh" "$@" ;;
    -h | --help | help)
        echo "usage: CLUSTER=<name> just cluster [<name>] <command> [args]"
        echo "  setup         build the shared .sif and rsync it to the cluster"
        echo "  build         build the .sif from the shared docker image (no push)"
        echo "  push/repush   rsync the built .sif to the cluster (reuses the SSH master; no 2FA)"
        echo "  add           create a cluster config (scripts/cluster/config/<name>)"
        echo "  job [args]    rsync the workspace + submit a batch job"
        echo "  develop ...   manage a persistent dev node (start/status/attach/exec/sync/stop)"
        ;;
    *) echo "[ERROR] unknown command '$cmd' (try: help)" >&2; exit 1 ;;
esac
