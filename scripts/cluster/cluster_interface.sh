#!/usr/bin/env bash
# Single cluster entrypoint (used directly by `pls cluster`). Builds/pushes the shared Isaac .sif,
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

source "${SCRIPT_DIR}/tools/restore_profiles.sh"

source_cluster_env() {
    if [ ! -f "$CLUSTER_ENV_FILE" ]; then
        echo "[ERROR] Cluster config not found: $CLUSTER_ENV_FILE (run 'pls cluster add'). Available:" \
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

# Build the .sif on the cluster (CLUSTER_ARCH=arm64): upload hcrl-isaac.def and its files next to the .sif, then
# `apptainer build --fakeroot` in a batch job with the profile's resources, replacing the .sif only once built.
build_remote_sif() {
    source_cluster_env
    local build="${CLUSTER_SIF_PATH}/build-${IMAGE_NAME}" submit="${SCRIPT_DIR}/config/${CLUSTER}/submit_job_slurm.sh"
    local dockerfile="${SCRIPT_DIR}/../docker/Dockerfile" base flags="" flag value free min="${CLUSTER_BUILD_MIN_FREE_GB:-40}"
    # the same base image as the docker build, from its ARGs
    base="$(sed -nE 's/^ARG ISAACSIM_BASE_IMAGE=(.+)/\1/p' "$dockerfile"):$(sed -nE 's/^ARG ISAACSIM_VERSION=(.+)/\1/p' "$dockerfile")"
    for flag in -p -A -q --reservation; do  # the profile's own submission flags, in any spelling
        value="$(python3 "${SCRIPT_DIR}/tools/merge_profile.py" get "$submit" "$flag")"
        [ -n "$value" ] && flags+=" ${flag} ${value}"
    done
    ensure_ssh_master
    # the base, the new .sif and the old one coexist under CLUSTER_SIF_PATH while building
    free="$(CLUSTER="$CLUSTER" bash "${SCRIPT_DIR}/cluster_dev/cluster_dev.sh" __free_gb "$CLUSTER_SIF_PATH" 2>/dev/null || true)"
    if [ -n "$free" ] && [ "$free" -lt "$min" ]; then
        echo "[ERROR] only ${free} GB free at ${CLUSTER_SIF_PATH}; the build needs ${min} (CLUSTER_BUILD_MIN_FREE_GB)" >&2
        exit 1
    fi
    echo "[cluster] uploading the ${CLUSTER_ARCH} recipe to ${CLUSTER_LOGIN}:${build}"
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "mkdir -p '${build}'"
    rsync -tv -e "ssh ${SSH_OPTS[*]}" "${SCRIPT_DIR}/hcrl-isaac.def" \
        "${SCRIPT_DIR}/../docker/requirements.workspace.txt" "${SCRIPT_DIR}/../docker/constraints.workspace.txt" \
        "${SCRIPT_DIR}/../docker/install-git-lfs.sh" "${SCRIPT_DIR}/../docker/entrypoint.sh" "${CLUSTER_LOGIN}:${build}/"
    # per-job names, so concurrent setups never share a file. Layers cache on scratch, the build unpacks on the node
    local job="#!/bin/bash
set -e
${CLUSTER_MODULE_LOAD:+module load ${CLUSTER_MODULE_LOAD}}
export APPTAINER_CACHEDIR=\"\${SCRATCH:-${build}}/apptainer-cache\" APPTAINER_TMPDIR=\"\${TMPDIR:-/tmp}\"
# site binds and preloads (TACC's XALT) target paths the image under construction lacks
unset SINGULARITY_BINDPATH APPTAINER_BINDPATH SINGULARITYENV_LD_PRELOAD APPTAINERENV_LD_PRELOAD
cd '${build}'
uname -m; apptainer --version
# a pulled base: a docker bootstrap loses the image's linked files when unprivileged
if [ \"\$(cat isaac-sim-base.ref 2>/dev/null)\" != '${base}' ]; then
    apptainer pull --force isaac-sim-base.sif.\$SLURM_JOB_ID 'docker://${base}'
    mv -f isaac-sim-base.sif.\$SLURM_JOB_ID isaac-sim-base.sif
    echo '${base}' > isaac-sim-base.ref
fi
apptainer build --fakeroot --force ${IMAGE_NAME}.sif.partial.\$SLURM_JOB_ID hcrl-isaac.def
mv -f ${IMAGE_NAME}.sif.partial.\$SLURM_JOB_ID '${CLUSTER_SIF_PATH}/${IMAGE_NAME}.sif'
echo \"[cluster] built ${CLUSTER_SIF_PATH}/${IMAGE_NAME}.sif\""
    local opts="-N 1 -t ${CLUSTER_BUILD_TIME:-02:00:00} -J ${IMAGE_NAME}-build -o '${build}/build-%j.log'${flags}"
    echo "[cluster] building in a batch job (sbatch ${opts}); it can take ~15 min"
    local rc=0 out id
    out="$(printf '%s\n' "$job" | ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "sbatch --parsable --wait ${opts}")" || rc=$?
    id="$(printf '%s\n' "$out" | grep -oE '^[0-9]+' | tail -1)"
    if [ -z "$id" ]; then
        printf '%s\n' "$out" >&2
        echo "[ERROR] sbatch did not accept the build job (exit ${rc})" >&2; exit "${rc/#0/1}"
    fi
    local log="${build}/build-${id}.log"
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "[ -f '${log}' ] && tail -5 '${log}' || echo '(no log at ${log})'" < /dev/null
    # sbatch --wait exits with the job's status, which can be 255 like a dropped ssh: ask whether the job still runs
    if [ "$rc" -ne 0 ] && [ -n "$(ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "squeue -h -t PENDING,RUNNING -j ${id}" < /dev/null 2>/dev/null)" ]; then
        echo "[ERROR] lost the connection while job ${id} runs; it may still finish: see ${log}" >&2; exit "$rc"
    fi
    [ "$rc" -eq 0 ] || { echo "[ERROR] build job ${id} failed (exit ${rc}); see ${log}" >&2; exit "$rc"; }
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

cmd_job() {  # job [--tree NAME] [args]: a batch job on a staged tree (default: the workspace, staged as `default`)
    source_cluster_env
    [ -f "$SCRIPT_DIR/../.env.wandb" ] || {
        echo "[ERROR] scripts/.env.wandb not found: a job without W&B credentials cannot log. Create it from" \
            "scripts/tools/.env.wandb.template." >&2
        exit 1
    }
    # the API key: owner-only here, which the copy below carries to the cluster
    chmod go-rwx "$SCRIPT_DIR/../.env.wandb"
    local tree_name="" dev="$SCRIPT_DIR/cluster_dev/cluster_dev.sh"
    if [ "${1:-}" = --tree ]; then
        [ -n "${2:-}" ] || { echo "[ERROR] --tree needs a tree name or <name>-<fingerprint>" >&2; exit 1; }
        tree_name="$2"; shift 2
    fi
    ensure_ssh_master
    if [ -z "$tree_name" ]; then
        CLUSTER="$CLUSTER" bash "$dev" stage || { echo "[ERROR] could not stage the workspace" >&2; exit 1; }
        tree_name=default
    fi
    # resolved once: the job runs this tree even if it waits in the queue while newer ones are staged
    local tree; tree="$(CLUSTER="$CLUSTER" bash "$dev" __resolve_tree "$tree_name")" || exit 1
    # one dir per submission: the tree's repos, this cluster's job scripts and the credentials run_singularity.sh reads
    local job_dir="${CLUSTER_ISAACLAB_DIR}/jobs/${CLUSTER}-$(date +"%Y%m%d_%H%M%S")-$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')"
    echo "[INFO] Job dir ${job_dir} on tree ${tree}"
    # a plain mkdir of the leaf: two submissions never share a dir (ln -s into an existing one would nest the link)
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "mkdir -p '${job_dir%/*}' && mkdir '${job_dir}' && \
        mkdir -p '${job_dir}/scripts/cluster/config/${CLUSTER}' && ln -s '${tree}/resources' '${job_dir}/resources'" < /dev/null || exit 1
    local f
    rsync -t -e "ssh ${SSH_OPTS[*]}" "$SCRIPT_DIR/run_singularity.sh" "$CLUSTER_LOGIN:${job_dir}/scripts/cluster/" || exit 1
    for f in "$SCRIPT_DIR/config/${CLUSTER}"/submit_job_*.sh; do
        [ -f "$f" ] && { rsync -t -e "ssh ${SSH_OPTS[*]}" "$f" "$CLUSTER_LOGIN:${job_dir}/scripts/cluster/config/${CLUSTER}/" || exit 1; }
    done
    rsync -t -e "ssh ${SSH_OPTS[*]}" "$CLUSTER_ENV_FILE" "$CLUSTER_LOGIN:${job_dir}/scripts/cluster/.env.cluster" || exit 1
    rsync -tp -e "ssh ${SSH_OPTS[*]}" "$SCRIPT_DIR/../.env.wandb" "$CLUSTER_LOGIN:${job_dir}/scripts/cluster/.env.wandb" || exit 1
    [ ! -f "$SCRIPT_DIR/../.env.base" ] || {
        rsync -t -e "ssh ${SSH_OPTS[*]}" "$SCRIPT_DIR/../.env.base" "$CLUSTER_LOGIN:${job_dir}/scripts/.env.base" || exit 1; }
    echo "[INFO] Submitting job..."
    local out id
    out="$(CLUSTER_ISAACLAB_DIR="$job_dir" submit_job "$@")" || { printf '%s\n' "$out"; exit 1; }
    printf '%s\n' "$out"
    # the tree stays while the job is queued or running: `trees rm` and pruning check squeue for this marker
    id="$(grep -oE 'Submitted batch job [0-9]+' <<< "$out" | grep -oE '[0-9]+$' | tail -1)"
    [ -z "$id" ] || ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "mkdir -p '${tree}/.in-use' && touch '${tree}/.in-use/${id}.nostep'" < /dev/null
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

# a profile with CLUSTER_ARCH other than amd64 builds its .sif on the cluster (`setup`), never from this machine
ARCH=""
[ -f "$CLUSTER_ENV_FILE" ] && ARCH="$(bash -c 'source "$1" >/dev/null 2>&1; printf %s "${CLUSTER_ARCH:-}"' _ "$CLUSTER_ENV_FILE")"
case "${ARCH:-amd64}" in
    amd64 | arm64) ;;
    *) echo "[ERROR] ${CLUSTER}: CLUSTER_ARCH must be amd64 or arm64, not '${ARCH}'" >&2; exit 1 ;;
esac
case "${cmd}:${ARCH:-amd64}" in
    *:amd64) ;;
    setup:arm64) build_remote_sif; exit ;;
    build:arm64 | push:arm64 | repush:arm64)
        echo "[ERROR] ${CLUSTER} is arm64: its .sif is built on the cluster by 'setup', not here" >&2; exit 1 ;;
esac

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
        echo "usage: pls cluster [<name>] <command> [args]"
        echo "  setup         build the shared .sif and rsync it to the cluster (CLUSTER_ARCH=arm64: build it there)"
        echo "  build         build the .sif from the shared docker image (no push)"
        echo "  push/repush   rsync the built .sif to the cluster (reuses the SSH master; no 2FA)"
        echo "  add [--update] [name]  create or regenerate your profile (scripts/cluster/config/<name>, gitignored)"
        echo "  job [--tree N] [args]  stage the workspace (or use tree N) + submit a batch job on it"
        echo "  develop ...   manage a persistent dev node (start/status/attach/exec/sync/kill/stop)"
        ;;
    *) echo "[ERROR] unknown command '$cmd' (try: help)" >&2; exit 1 ;;
esac
