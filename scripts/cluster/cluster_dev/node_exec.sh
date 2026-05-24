#!/usr/bin/env bash
# node_exec.sh — runs ON the Delta compute node (invoked by `cluster_dev.sh exec`).
# Stages the Apptainer SIF + Isaac Sim caches + code into node-local $TMPDIR once
# (cached across calls within the same job), then `apptainer exec`s the given command
# inside the container. Bind-mount list mirrors docker/cluster/run_singularity.sh.
#
# [VERIFY q5] assumes `apptainer` is on PATH on gpuA40x4 nodes (Delta config has no
#   `module load`); if not, add the needed `module load apptainer` here.
# [VERIFY q6] assumes ${CLUSTER_SIF_PATH}/isaac-lab-base.tar exists and is current.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../.env.cluster"
source "${SCRIPT_DIR}/../.env.wandb"  2>/dev/null || true
source "${SCRIPT_DIR}/../../.env.base" 2>/dev/null || true

PROFILE="${PROFILE:-isaac-lab-base}"
# Persistent per-job staging dir on node-local scratch (reused across exec calls).
STAGE="${TMPDIR:-/tmp}/cluster_dev_${SLURM_JOB_ID:-box}"
SIF="${STAGE}/${PROFILE}.sif"

stage_once() {
    [ -f "$SIF" ] && return 0
    echo "[node_exec] staging container + caches into ${STAGE} (first call only)…"
    mkdir -p "${STAGE}/tmp"
    cp -rn "$CLUSTER_ISAAC_SIM_CACHE_DIR" "$STAGE/" 2>/dev/null || true
    tar -xf "${CLUSTER_SIF_PATH}/${PROFILE}.tar" -C "$STAGE" || { echo "[node_exec] SIF extract failed"; exit 1; }
    echo "[node_exec] staged."
}

stage_once
cmd="$*"; [ -n "$cmd" ] || cmd="/isaac-sim/python.sh --version"
apptainer exec \
    -B ${STAGE}/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B ${STAGE}/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B ${STAGE}/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B ${STAGE}/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B ${STAGE}/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B ${CLUSTER_ISAACLAB_DIR}:/workspace/isaaclab:rw \
    -B ${STAGE}/tmp:/tmp:rw \
    --nv --writable --containall --no-home "$SIF" \
    bash -c "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} && export ISAACLAB_PATH=/workspace/isaaclab && export WANDB_USERNAME=${WANDB_USERNAME:-} && export WANDB_API_KEY=${WANDB_API_KEY:-} && cd /workspace/isaaclab && ${cmd}"
