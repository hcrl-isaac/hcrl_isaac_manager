#!/usr/bin/env bash
# Entrypoint for the shared Isaac image: PYTHONPATHs the bound workspace packages ahead of pip isaaclab.
# Binds: /workspace/ext/<name> (package repos), /workspace/isaaclab_source (source mode only).
set -e
EXT_DIR="${HCRL_EXT_DIR:-/workspace/ext}"
SRC_DIR="${HCRL_ISAACLAB_SRC:-/workspace/isaaclab_source}"

new_pp=""
# Source-mode IsaacLab takes precedence over the baked pip isaaclab.
if [ -d "$SRC_DIR" ]; then
    for d in "$SRC_DIR"/isaaclab*/; do [ -d "$d" ] && new_pp="${d%/}:${new_pp}"; done
fi
# Workspace package repo roots; data-only repos are skipped.
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
