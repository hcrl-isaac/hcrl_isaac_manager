#!/usr/bin/env bash

echo -e "(run_singularity.py): Called on compute node:"
echo -e "\tCurrent isaaclab directory: $1"
echo -e "\tContainer profile: $2"
echo -e "\tPython executable: $CLUSTER_PYTHON_EXECUTABLE ${@:3}"


#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
   "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents" \
   "${JOB_TMPDIR}/tmp"; do
   if [ ! -d "$dir" ]; then
  mkdir -p "$dir"
  echo "(run_singularity.py): Created directory: $dir"
   fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# use unique tmpdir for each run
JOB_TMPDIR="$TMPDIR/isaaclab_$SLURM_JOB_ID"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/.env.wandb
# .env.base is only present in source mode; default the container paths so `set -u` doesn't trip.
[ -f "$SCRIPT_DIR/../.env.base" ] && source "$SCRIPT_DIR/../.env.base"
# some clusters ship the container runtime as an Lmod module (e.g. TACC's tacc-apptainer)
if [ -n "${CLUSTER_MODULE_LOAD:-}" ] && command -v module >/dev/null 2>&1; then
    module load $CLUSTER_MODULE_LOAD
fi
: "${DOCKER_ISAACSIM_ROOT_PATH:=/isaac-sim}"
: "${DOCKER_USER_HOME:=/root}"

# make sure that all directories exists in cache directory
mkdir "$JOB_TMPDIR"
echo "(run_singularity.py): Created directory: $JOB_TMPDIR"
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $JOB_TMPDIR

# Stage a full container home from the cache layout and bind it over ${DOCKER_USER_HOME} as ONE mount.
# Binding the individual dot-dirs fails on clusters whose apptainer cannot create bind points that are
# missing from the image (e.g. TACC compute nodes); the home dir itself always exists in the image.
HOME_DIR="$JOB_TMPDIR/home"
mkdir -p "$HOME_DIR/.cache/nvidia" "$HOME_DIR/.nv" "$HOME_DIR/.nvidia-omniverse" "$HOME_DIR/.local/share/ov"
mv "$JOB_TMPDIR/docker-isaac-sim/cache/ov" "$HOME_DIR/.cache/ov"
mv "$JOB_TMPDIR/docker-isaac-sim/cache/pip" "$HOME_DIR/.cache/pip"
mv "$JOB_TMPDIR/docker-isaac-sim/cache/glcache" "$HOME_DIR/.cache/nvidia/GLCache"
mv "$JOB_TMPDIR/docker-isaac-sim/cache/computecache" "$HOME_DIR/.nv/ComputeCache"
mv "$JOB_TMPDIR/docker-isaac-sim/logs" "$HOME_DIR/.nvidia-omniverse/logs"
mv "$JOB_TMPDIR/docker-isaac-sim/data" "$HOME_DIR/.local/share/ov/data"

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $1 $JOB_TMPDIR
echo "(run_singularity.py) Copied $1 to $JOB_TMPDIR"
# Get the directory name
dir_name=$(basename "$1")

# copy the shared decoupled container (.sif, built by `just cluster setup`) to the compute node
SIF_SRC="$CLUSTER_SIF_PATH/$2.sif"
[ -f "$SIF_SRC" ] || SIF_SRC="$CLUSTER_SIF_PATH/hcrl-isaac.sif"
if [ -f "$SIF_SRC" ]; then
    cp "$SIF_SRC" "$JOB_TMPDIR/$2.sif"
    echo "(run_singularity.py) Using container $SIF_SRC"
elif [ -f "$CLUSTER_SIF_PATH/$2.tar" ]; then  # backwards-compat: an exported docker tar
    tar -xf "$CLUSTER_SIF_PATH/$2.tar" -C "$JOB_TMPDIR" || { echo "(run_singularity.py) Tar extraction failed!"; exit 1; }
else
    echo "(run_singularity.py) No container found at $CLUSTER_SIF_PATH ($2.sif / hcrl-isaac.sif / $2.tar)"; exit 1
fi

# Bind all workspace repos into /workspace/ext -- packages AND asset repos (for the in-repo resource
# symlinks); the entrypoint PYTHONPATHs only the packages. (+ IsaacLab source overlay in source mode.)
EXT_BINDS=""
SYNC_DIR="$JOB_TMPDIR/$dir_name"
for d in "$SYNC_DIR"/resources/*/; do
    name="$(basename "$d")"
    [ "$name" = "IsaacLab" ] && continue   # handled by the source overlay below, not /workspace/ext
    EXT_BINDS="$EXT_BINDS -B ${d%/}:/workspace/ext/${name}:rw"
done
[ -d "$SYNC_DIR/resources/IsaacLab/source" ] && EXT_BINDS="$EXT_BINDS -B $SYNC_DIR/resources/IsaacLab/source:/workspace/isaaclab_source:rw"

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
apptainer exec ${CLUSTER_APPTAINER_FLAGS:-} \
    -B $JOB_TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $HOME_DIR:${DOCKER_USER_HOME}:rw \
    $EXT_BINDS \
    -B $JOB_TMPDIR/tmp:/tmp:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/ext/hcrl_isaaclab/logs:rw \
    --nv --writable-tmpfs --containall --no-home $JOB_TMPDIR/$2.sif \
    bash -c "export HOME=${DOCKER_USER_HOME} && export OMP_NUM_THREADS=$OMP_NUM_THREADS && export OMNI_KIT_ACCEPT_EULA=YES && export WANDB_USERNAME=$WANDB_USERNAME && export WANDB_API_KEY=$WANDB_API_KEY && cd /workspace/ext/hcrl_isaaclab && /usr/local/bin/hcrl-entrypoint /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"

EXIT_CODE=$?

# copy resulting cache files back to host
# restore the persistent cache layout from the staged home before syncing it back
mv "$HOME_DIR/.cache/ov" "$JOB_TMPDIR/docker-isaac-sim/cache/ov"
mv "$HOME_DIR/.cache/pip" "$JOB_TMPDIR/docker-isaac-sim/cache/pip"
mv "$HOME_DIR/.cache/nvidia/GLCache" "$JOB_TMPDIR/docker-isaac-sim/cache/glcache"
mv "$HOME_DIR/.nv/ComputeCache" "$JOB_TMPDIR/docker-isaac-sim/cache/computecache"
mv "$HOME_DIR/.nvidia-omniverse/logs" "$JOB_TMPDIR/docker-isaac-sim/logs"
mv "$HOME_DIR/.local/share/ov/data" "$JOB_TMPDIR/docker-isaac-sim/data"
rsync -azPv $JOB_TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..
# clean up tmpdir
rm -rf $JOB_TMPDIR

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
# only remove folders if run finished successfully -- otherwise we want to keep the logs
if $REMOVE_CODE_COPY_AFTER_JOB && [ $EXIT_CODE -eq 0 ]; then
    rm -rf $1
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "(run_singularity.py): Success"
else
    echo "(run_singularity.py): Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
