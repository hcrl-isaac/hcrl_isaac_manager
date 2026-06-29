#!/usr/bin/env bash
# node_exec.sh -- runs ON the cluster compute node (invoked by `cluster_dev.sh exec`). Stages the .sif +
# Isaac Sim caches into node-local $TMPDIR once per job, then `apptainer exec`s the given command in the
# container. Bind list mirrors scripts/cluster/run_singularity.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../.env.cluster"
source "${SCRIPT_DIR}/../../.env.wandb"  2>/dev/null || true
# .env.base is only present in source mode; default the container paths so `set -u` doesn't trip.
source "${SCRIPT_DIR}/../../.env.base" 2>/dev/null || true
: "${DOCKER_ISAACSIM_ROOT_PATH:=/isaac-sim}"
: "${DOCKER_USER_HOME:=/root}"

# Some sites ship the container runtime as an Lmod module (e.g. TACC: tacc-apptainer)
# rather than on PATH. Load it if the cluster's .env.cluster sets CLUSTER_MODULE_LOAD.
if [ -n "${CLUSTER_MODULE_LOAD:-}" ]; then
    set +u
    if ! type module >/dev/null 2>&1; then
        for init in /etc/profile.d/z00_lmod.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
            [ -f "$init" ] && source "$init" && break
        done
    fi
    module load ${CLUSTER_MODULE_LOAD} || echo "[node_exec] WARNING: 'module load ${CLUSTER_MODULE_LOAD}' failed"
    set -u
fi

PROFILE="${PROFILE:-hcrl-isaac}"
# Persistent per-job staging dir on node-local scratch (reused across exec calls).
STAGE="${TMPDIR:-/tmp}/cluster_dev_${SLURM_JOB_ID:-box}"
SIF="${STAGE}/${PROFILE}.sif"

stage_once() {
    # Copy the .sif to node-local scratch once; re-copy if a newer one was pushed, so `cluster.sh
    # repush` takes effect without clearing the node cache. (Kit writes go to a --writable-tmpfs overlay.)
    local src="${CLUSTER_SIF_PATH}/${PROFILE}.sif"
    [ -f "$SIF" ] && [ ! "$src" -nt "$SIF" ] && return 0
    mkdir -p "$STAGE"
    echo "[node_exec] staging container + caches into ${STAGE} (sif new/updated)..."
    cp -rn "$CLUSTER_ISAAC_SIM_CACHE_DIR" "$STAGE/" 2>/dev/null || true
    cp "$src" "$SIF" || { echo "[node_exec] could not stage ${src}"; exit 1; }
    echo "[node_exec] staged."
}

stage_once
# Pre-create every bind source on each call (idempotent): on a fresh node the cache cp in stage_once
# is a no-op, so the cache subdirs would be missing and the binds would fail.
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
    "${STAGE}/docker-isaac-sim/documents"
cmd="$*"; [ -n "$cmd" ] || cmd="/isaac-sim/python.sh --version"
# Bind all workspace repos into /workspace/ext -- packages AND asset repos (e.g. hcrl_robots), since the
# in-repo resource symlinks need the asset repos mounted. The entrypoint PYTHONPATHs only the packages.
EXT_BINDS=""
for d in "${CLUSTER_ISAACLAB_DIR}"/resources/*/; do
    name="$(basename "$d")"
    [ "$name" = "IsaacLab" ] && continue   # handled by the source overlay below, not /workspace/ext
    EXT_BINDS="$EXT_BINDS -B ${d%/}:/workspace/ext/${name}:rw"
done
[ -d "${CLUSTER_ISAACLAB_DIR}/resources/IsaacLab/source" ] && \
    EXT_BINDS="$EXT_BINDS -B ${CLUSTER_ISAACLAB_DIR}/resources/IsaacLab/source:/workspace/isaaclab_source:rw"
# Bind list mirrors docker/cluster/run_singularity.sh -- with extra `-B ...:/u/esturman` so HOME is
# writable inside the container.
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
    ${EXT_BINDS} \
    -B ${STAGE}/tmp:/tmp:rw \
    --nv --writable-tmpfs --containall --no-home "$SIF" \
    bash -c "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} && export HOME=/u/esturman && export OMNI_KIT_ACCEPT_EULA=YES && export WANDB_USERNAME=${WANDB_USERNAME:-} && export WANDB_API_KEY=${WANDB_API_KEY:-} && cd /workspace/ext/hcrl_isaaclab && exec /usr/local/bin/hcrl-entrypoint ${cmd}"
