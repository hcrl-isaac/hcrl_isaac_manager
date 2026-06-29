#!/usr/bin/env bash
# Runtime entrypoint for the shared Isaac image (Ray worker + HPC .sif).
#
# The image bakes isaacsim (base) + Isaac Lab (pip) but NOT the workspace code, so code syncs without a
# rebuild: the workspace packages are bound in and editable-installed here, at container start, then the
# requested command is exec'd. Bind mounts (set by the Ray job config / Apptainer run):
#   /workspace/ext/<name>      flat workspace packages (hcrl_isaaclab, robot_rl, *_tasks, packaged *_robots)
#   /workspace/isaaclab_source (source mode only) resources/IsaacLab/source -- its isaaclab* dirs override
#                              the baked pip isaaclab. Absent in pip mode, so the baked isaaclab is used.
set -e
ISAAC_PY="${ISAAC_PY:-/isaac-sim/python.sh}"

# Source-mode overlay first: editable-install the mounted IsaacLab source over the baked pip isaaclab.
# --no-deps: the dependency closure is already baked into the image; only re-point the editable packages.
if [ -d /workspace/isaaclab_source ]; then
    echo "[entrypoint] source-mode overlay: editable-installing mounted IsaacLab source"
    for d in /workspace/isaaclab_source/isaaclab*/; do
        [ -d "$d" ] && ${ISAAC_PY} -m pip install --no-deps -e "$d"
    done
fi

# Editable-install the mounted workspace packages (skip data-only repos with no setup.py/pyproject).
if [ -d /workspace/ext ]; then
    for d in /workspace/ext/*/; do
        if [ -d "$d" ] && { [ -f "${d}setup.py" ] || [ -f "${d}pyproject.toml" ]; }; then
            echo "[entrypoint] editable-installing $(basename "$d")"
            ${ISAAC_PY} -m pip install --no-deps -e "$d"
        fi
    done
fi

exec "$@"
