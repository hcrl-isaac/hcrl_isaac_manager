#!/usr/bin/env bash
# node_exec.sh -- runs ON the cluster compute node (invoked by `cluster_dev.sh exec`). Stages the .sif +
# Isaac Sim caches into node-local $TMPDIR once per job, then `apptainer exec`s the given command in the
# container. Bind list mirrors scripts/cluster/run_singularity.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ -n "${NODE_EXEC_ENV:-}" ] && [ -f "$NODE_EXEC_ENV" ]; then
    source "$NODE_EXEC_ENV"
else
    source "${SCRIPT_DIR}/../.env.cluster"
fi
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
# NODE_EXEC_ROOT (develop exec --root): run an Isaac Lab install root's venv in a stock bookworm image instead of
# the hcrl-isaac one, since the 3.x wheels need glibc >= 2.35 and the HPC hosts are older
ROOT_DIR="${NODE_EXEC_ROOT:-}"

root_sif() {  # <root>/bookworm.sif, pulled once; a concurrent pull lands under its own name and is renamed
    SIF="${ROOT_DIR}/bookworm.sif"
    [ -f "$SIF" ] && return 0
    mkdir -p "${ROOT_DIR}/tmp" "${ROOT_DIR}/apptainer-cache"
    echo "[node_exec] pulling python:3.12-bookworm into ${SIF}..."
    APPTAINER_CACHEDIR="${ROOT_DIR}/apptainer-cache" APPTAINER_TMPDIR="${ROOT_DIR}/tmp" \
        apptainer pull "${SIF}.partial.$$" docker://python:3.12-bookworm || { rm -f "${SIF}.partial.$$"; exit 1; }
    mv -f "${SIF}.partial.$$" "$SIF"
}

stage_once() {
    # Copy the .sif to node-local scratch once; re-copy if a newer one was pushed, so `pls setup --on <name>
    # setup --push-only` takes effect without clearing the node cache. (Kit writes go to a --writable-tmpfs overlay.)
    local src="${CLUSTER_SIF_PATH}/${PROFILE}.sif"
    [ -f "$SIF" ] && [ ! "$src" -nt "$SIF" ] && return 0
    mkdir -p "$STAGE"
    echo "[node_exec] staging container + caches into ${STAGE} (sif new/updated)..."
    cp -rn "$CLUSTER_ISAAC_SIM_CACHE_DIR" "$STAGE/" 2>/dev/null || true
    # temp + atomic rename: RUNNING containers loop-mount $SIF, and truncating it in place (plain cp)
    # corrupts their rootfs (observed: live recorders dying with ENOENT on /isaac-sim/*). A rename swaps
    # the dentry; running mounts keep the old (deleted-but-open) inode and stay intact.
    cp "$src" "${SIF}.staging.$$" || { rm -f "${SIF}.staging.$$"; echo "[node_exec] could not stage ${src}"; exit 1; }
    # carry the source mtime so the -nt freshness check stays false until a genuinely newer sif is pushed
    # (cp stamps 'now', which can leave -nt true forever -> a corrupting re-copy on EVERY exec)
    touch -r "$src" "${SIF}.staging.$$"
    mv -f "${SIF}.staging.$$" "$SIF"
    echo "[node_exec] staged."
}

if [ -n "$ROOT_DIR" ]; then
    [ -x "${ROOT_DIR}/venv/bin/python" ] || { echo "[node_exec] no Isaac Lab root venv at ${ROOT_DIR}/venv"; exit 97; }
    root_sif
else
    stage_once
fi
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
# one argument is a shell command string (run whole by bash -c), several are an argv passed through verbatim
if [ $# -eq 1 ]; then cmd="bash -c $(printf %q "$1")"; elif [ $# -gt 1 ]; then cmd="$(printf '%q ' "$@")"; else cmd=""; fi
[ -n "$cmd" ] || cmd="${ROOT_DIR:+${ROOT_DIR}/venv/bin/python --version}"
[ -n "$cmd" ] || cmd="/isaac-sim/python.sh --version"
# Bind all workspace repos into /workspace/ext -- packages AND asset repos (e.g. hcrl_robots), since the
# in-repo resource symlinks need the asset repos mounted. The entrypoint PYTHONPATHs only the packages.
# NODE_EXEC_RESOURCES (set by `exec --tree`) points at a staged tree's resources/ instead of the shared one.
RESOURCES="${NODE_EXEC_RESOURCES:-${CLUSTER_ISAACLAB_DIR}/resources}"
IN_USE="" OWN_TMP=""
cleanup() {  # one EXIT trap for both: the tree's in-use marker and a step's own /tmp
    [ -n "$IN_USE" ] && rm -f "$IN_USE"
    [ -n "$OWN_TMP" ] && rm -rf "$OWN_TMP"
    return 0
}
trap cleanup EXIT
# run logs and checkpoints; a profile points CLUSTER_LOGS_DIR off a quota'd home
LOGS_DIR="${CLUSTER_LOGS_DIR:-${CLUSTER_ISAACLAB_DIR}/resources/hcrl_isaaclab/logs}"
if [ -n "${NODE_EXEC_RESOURCES:-}" ]; then
    # `trees rm` refuses while this marker's step (or job, outside a step) is still in squeue
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        IN_USE="$(dirname "$RESOURCES")/.in-use/${SLURM_JOB_ID}.${SLURM_STEP_ID:-nostep}"
    else
        IN_USE="$(dirname "$RESOURCES")/.in-use/$(hostname -s).$$"
    fi
    mkdir -p "$(dirname "$IN_USE")" && touch "$IN_USE"
    trap 'exit 143' TERM INT HUP
fi
EXT_BINDS=""
for d in "${RESOURCES}"/*/; do
    name="$(basename "$d")"
    [ "$name" = "IsaacLab" ] && continue   # handled by the source overlay below, not /workspace/ext
    if [ -n "${NODE_EXEC_RESOURCES:-}" ] && [ ! -L "${d%/}" ]; then
        # the artifact root lives in the shared checkout
        [ "$name" = hcrl_isaaclab ] && mkdir -p "${CLUSTER_ISAACLAB_DIR}/resources/hcrl_isaaclab/.artifacts" &&
            EXT_BINDS="$EXT_BINDS -B ${CLUSTER_ISAACLAB_DIR}/resources/hcrl_isaaclab/.artifacts:/workspace/ext/hcrl_isaaclab/.artifacts:rw"
        # writable, since the artifact resolver re-links policy dests inside the repo; run outputs are redirected:
        # hcrl_isaaclab's logs to the shared logs dir, the rest node-local
        EXT_BINDS="$EXT_BINDS -B ${d%/}:/workspace/ext/${name}:rw"
        for rw in logs outputs wandb; do
            if [ "$name/$rw" = hcrl_isaaclab/logs ]; then
                out="$LOGS_DIR"
            else
                out="${STAGE}/tree-rw/$(basename "$(dirname "$RESOURCES")")/${name}/${rw}"
            fi
            mkdir -p "$out"
            EXT_BINDS="$EXT_BINDS -B ${out}:/workspace/ext/${name}/${rw}:rw"
        done
    else
        EXT_BINDS="$EXT_BINDS -B $(readlink -f "${d%/}"):/workspace/ext/${name}:rw"
        if [ "$name" = hcrl_isaaclab ] && [ -n "${CLUSTER_LOGS_DIR:-}" ]; then
            mkdir -p "$LOGS_DIR"
            EXT_BINDS="$EXT_BINDS -B ${LOGS_DIR}:/workspace/ext/hcrl_isaaclab/logs:rw"
        fi
    fi
done
[ -d "${RESOURCES}/IsaacLab/source" ] && \
    EXT_BINDS="$EXT_BINDS -B $(readlink -f "${RESOURCES}/IsaacLab/source"):/workspace/isaaclab_source:rw"
# isaaclab reads its kit file at <source>/../apps, which is /workspace/apps once the source is bound there
[ -d "${RESOURCES}/IsaacLab/apps" ] && \
    EXT_BINDS="$EXT_BINDS -B $(readlink -f "${RESOURCES}/IsaacLab/apps"):/workspace/apps:ro"
# artifacts/ holds cluster-only INPUT data no tree carries; it still has to be readable inside the container,
# which the resources/* glob misses.
[ -d "${CLUSTER_ISAACLAB_DIR}/artifacts" ] && \
    EXT_BINDS="$EXT_BINDS -B ${CLUSTER_ISAACLAB_DIR}/artifacts:/workspace/artifacts:rw"
# Bind list mirrors scripts/cluster/run_singularity.sh -- with the staged home bound over your home path on the node
# ($HOME there) so HOME is writable inside the container. CLUSTER_APPTAINER_FLAGS carries site quirks (TACC: --fakeroot, which
# unprivileged apptainer needs to create bind points like /root/Documents that don't exist in the image).
CONTAINER_HOME="${HOME:-/home/$(id -un)}"
# credentials go in through the environment (apptainer injects APPTAINERENV_*), never argv, which `ps` shows
export APPTAINERENV_WANDB_USERNAME="${WANDB_USERNAME:-}" APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:-}"
# the Vulkan clamp layer (scripts/vulkan/), when installed for this cluster: Isaac Sim's RTX renderer
# segfaults at startup on drivers that report a UINT64_MAX maxMemoryAllocationSize (595.71, 615.71)
VK_BINDS=""
VKCLAMP_DIR="${CLUSTER_VKCLAMP_DIR:-${CLUSTER_SIF_PATH}/vkclamp}"
if [ -f "${VKCLAMP_DIR}/libvkclamp.so" ] && [ "$(uname -m)" = x86_64 ]; then  # the layer is an x86 build
    VK_BINDS="-B ${VKCLAMP_DIR}:/opt/vkclamp:ro"
    export APPTAINERENV_XDG_CONFIG_DIRS="/opt/vkclamp/conf:/etc/xdg"
fi
# --nv binds the driver's Vulkan library but not its ICD file, which the Isaac Lab 3 image (plain Ubuntu) lacks
ICD_JSON="${STAGE}/nvidia_icd.json"
# a bare library name: --nv binds the driver libs on the container's search path, never at a host path
echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}' >"$ICD_JSON.$$"
mv -f "$ICD_JSON.$$" "$ICD_JSON"  # steps sharing the stage never read a half-written file
VK_BINDS="$VK_BINDS -B ${ICD_JSON}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
# NODE_EXEC_OWN_TMP=1 (develop exec --own-tmp): this step's /tmp is its own, not the job's shared one, so steps on
# one node never meet in Isaac Lab's hard-coded /tmp/IsaacLab (its URDF->USD conversions race); /tmp/cache stays the
# job's, so caches kept there persist across steps
STEP_TMP="${STAGE}/tmp" CACHE_BIND=""
if [ "${NODE_EXEC_OWN_TMP:-}" = 1 ]; then
    OWN_TMP="$(mktemp -d "${STAGE}/step-tmp.XXXXXX")" && STEP_TMP="$OWN_TMP"
    mkdir -p "${STAGE}/tmp/cache" && CACHE_BIND="-B ${STAGE}/tmp/cache:/tmp/cache:rw"
fi
run_container() {
apptainer exec ${CLUSTER_APPTAINER_FLAGS:-} \
    -B ${STAGE}/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B ${STAGE}/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B ${STAGE}/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B ${STAGE}/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B ${STAGE}/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B ${STAGE}/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B ${STAGE}/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B ${STAGE}/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B ${STAGE}/home:${CONTAINER_HOME}:rw \
    ${EXT_BINDS} ${VK_BINDS} \
    -B ${STEP_TMP}:/tmp:rw ${CACHE_BIND} \
    --nv --writable-tmpfs --containall --no-home "$SIF" \
    bash -c "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} && export HOME=${CONTAINER_HOME} && export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 && cd /workspace/ext/hcrl_isaaclab && exec /usr/local/bin/hcrl-entrypoint ${cmd}"
}

run_root() {  # the root's venv in the bookworm image, with the same /workspace/ext mounts as the hcrl-isaac image
    # the hcrl-isaac entrypoint is not there, so the package repos go on PYTHONPATH here, with any asset repo that is
    # itself a Python package (ssti_robots); pure asset and data repos do not
    local pkgs="" d name
    for d in "${RESOURCES}"/*/; do
        name="$(basename "$d")"
        case "$name" in IsaacLab | motion_datasets) continue ;; esac
        case "$name" in *_robots) [ -f "${d}pyproject.toml" ] || [ -f "${d}setup.py" ] || continue ;; esac
        # a repo whose package is not at its root imports from inside it: the twin of worktree_env.py SRC_LAYOUT
        # (this runs on the node, without the manager's python), so a new src-layout repo changes both
        case "$name" in holosoma) name="holosoma/src/holosoma_retargeting" ;; esac
        pkgs="${pkgs:+$pkgs:}/workspace/ext/${name}"
    done
    export APPTAINERENV_PYTHONPATH="$pkgs"
    # TACC's XALT preloads a library that needs glibc 2.38 through these (bookworm has 2.36), so every exec would die
    unset LD_PRELOAD SINGULARITYENV_LD_PRELOAD APPTAINERENV_LD_PRELOAD SINGULARITY_BINDPATH APPTAINER_BINDPATH
    apptainer exec \
        -B "${ROOT_DIR}" -B /dev/shm \
        -B ${STAGE}/home:${CONTAINER_HOME}:rw \
        ${EXT_BINDS} ${VK_BINDS} \
        -B ${STEP_TMP}:/tmp:rw ${CACHE_BIND} \
        --nv --containall --no-home "$SIF" \
        bash -c "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} HOME=${CONTAINER_HOME} OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 && cd /workspace/ext/hcrl_isaaclab && exec ${cmd}"
}
[ -n "$ROOT_DIR" ] && run_container() { run_root; }

# A segfault (exit 139) within KIT_BOOT_S of start is Kit crashing during boot (seen on Delta in libX11's getenv),
# before any run registers with W&B, so it is retried once.
KIT_BOOT_S=20
for attempt in 1 2; do
    start=$SECONDS
    run_container && rc=0 || rc=$?
    if [ "$rc" -ne 139 ] || [ $((SECONDS - start)) -ge "$KIT_BOOT_S" ] || [ "$attempt" -eq 2 ]; then
        exit "$rc"
    fi
    echo "[node_exec] the container segfaulted $((SECONDS - start)) s after start (exit 139, a Kit boot crash); retrying once" >&2
done
