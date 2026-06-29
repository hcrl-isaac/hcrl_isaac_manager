#!/usr/bin/env bash
# Runtime entrypoint for the shared Isaac image (Ray worker + HPC .sif). The image bakes isaacsim + pip
# Isaac Lab but NOT the workspace code; this PYTHONPATHs the bound packages ahead of pip isaaclab so code
# syncs without a rebuild (the Apptainer fs is read-only, so no pip install). Bind mounts:
#   /workspace/ext/<name>       workspace package repos (hcrl_isaaclab, robot_rl, *_tasks, *_robots)
#   /workspace/isaaclab_source  source-mode IsaacLab (overrides pip); absent in pip mode
set -e
EXT_DIR="${HCRL_EXT_DIR:-/workspace/ext}"
SRC_DIR="${HCRL_ISAACLAB_SRC:-/workspace/isaaclab_source}"

new_pp=""
# Source-mode overlay first (highest precedence over the baked pip isaaclab).
if [ -d "$SRC_DIR" ]; then
    for d in "$SRC_DIR"/isaaclab*/; do [ -d "$d" ] && new_pp="${d%/}:${new_pp}"; done
fi
# Workspace package repo roots (so `import <name>` resolves the package subdir); skip data-only repos.
if [ -d "$EXT_DIR" ]; then
    for d in "$EXT_DIR"/*/; do
        if [ -d "$d" ] && { [ -f "${d}setup.py" ] || [ -f "${d}pyproject.toml" ]; }; then
            new_pp="${d%/}:${new_pp}"
        fi
    done
fi
export PYTHONPATH="${new_pp}${PYTHONPATH:-}"
[ -n "$new_pp" ] && echo "[entrypoint] PYTHONPATH += ${new_pp}"

exec "$@"
