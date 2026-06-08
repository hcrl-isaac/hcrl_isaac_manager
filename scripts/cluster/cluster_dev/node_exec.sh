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
    # The "SIF" is a sandbox *directory* (tar of an apptainer sandbox), not a single .sif file.
    # Use isaac-sim/python.sh (always in the SIF, never written at runtime) as the extract marker.
    [ -x "${SIF}/isaac-sim/python.sh" ] && return 0
    echo "[node_exec] staging container + caches into ${STAGE} (first call only)…"
    cp -rn "$CLUSTER_ISAAC_SIM_CACHE_DIR" "$STAGE/" 2>/dev/null || true
    tar -xf "${CLUSTER_SIF_PATH}/${PROFILE}.tar" -C "$STAGE" || { echo "[node_exec] SIF extract failed"; exit 1; }
    echo "[node_exec] staged."
}

stage_once
# Idempotent: ensure all bind-mount sources + sandbox bind targets exist on every call.
# (These can't live in stage_once because we want existing stages — pre-fix — to get the dirs
# without busting the 30+GB SIF cache.)
# Cache subdirs (ov/kit/pip/glcache/computecache) come from `cp -rn $CLUSTER_ISAAC_SIM_CACHE_DIR`
# in stage_once, but that's silenced with `|| true`; on a fresh node where the source cache
# doesn't exist yet, the cp is a no-op and the cache subdirs are missing. Pre-create them so
# the bind succeeds and Isaac Sim populates them on first use.
mkdir -p \
    "${STAGE}/tmp" \
    "${STAGE}/home" \
    "${STAGE}/docker-isaac-sim/cache/kit" \
    "${STAGE}/docker-isaac-sim/cache/ov" \
    "${STAGE}/docker-isaac-sim/cache/pip" \
    "${STAGE}/docker-isaac-sim/cache/glcache" \
    "${STAGE}/docker-isaac-sim/cache/computecache" \
    "${STAGE}/docker-isaac-sim/logs" \
    "${STAGE}/docker-isaac-sim/data" \
    "${STAGE}/docker-isaac-sim/documents" \
    "${SIF}/u/esturman"
cmd="$*"; [ -n "$cmd" ] || cmd="/isaac-sim/python.sh --version"
# Bind list mirrors docker/cluster/run_singularity.sh — with extra `-B …:/u/esturman` so HOME
# is writable inside the container. `--writable` is required (matches run_singularity.sh) so
# Kit can write /isaac-sim/kit/data/user.config.json; the SIF is a sandbox dir, not a file.
apptainer exec \
    -B ${STAGE}/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B ${STAGE}/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B ${STAGE}/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B ${STAGE}/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B ${STAGE}/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B ${STAGE}/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B ${STAGE}/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B ${STAGE}/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B ${STAGE}/home:/u/esturman:rw \
    -B ${CLUSTER_ISAACLAB_DIR}:/workspace/isaaclab:rw \
    -B ${STAGE}/tmp:/tmp:rw \
    --nv --writable --containall --no-home "$SIF" \
    bash -c "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} && export HOME=/u/esturman && export ISAACLAB_PATH=/workspace/isaaclab && export WANDB_USERNAME=${WANDB_USERNAME:-} && export WANDB_API_KEY=${WANDB_API_KEY:-} && cd /workspace/isaaclab && ${cmd}"
