#!/usr/bin/env bash
# Entrypoint for the shared Isaac image: PYTHONPATHs the bound workspace packages ahead of the baked Isaac Lab, and on
# aarch64 preloads the libgomp copies Isaac Sim's import requires.
# Binds: /workspace/ext/<name> (package repos), /workspace/isaaclab_source (source mode only).
set -e
EXT_DIR="${HCRL_EXT_DIR:-/workspace/ext}"
SRC_DIR="${HCRL_ISAACLAB_SRC:-/workspace/isaaclab_source}"

new_pp=""
# A bound IsaacLab source/ tree takes precedence over the baked /opt/IsaacLab clone.
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

# `import isaacsim` exits on aarch64 unless the system's and torch's libgomp are preloaded (isaacsim/__init__.py).
if [ "$(uname -m)" = aarch64 ]; then
    gomp=""
    for lib in /lib/*/libgomp.so.1 "${VIRTUAL_ENV:-/opt/isaaclab-venv}"/lib/python3.*/site-packages/torch.libs/libgomp*; do
        [ -f "$lib" ] && gomp="${gomp:+${gomp}:}${lib}"
    done
    [ -n "$gomp" ] && export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${gomp}"
fi

exec "$@"
