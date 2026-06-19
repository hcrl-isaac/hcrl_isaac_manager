#!/usr/bin/env bash
# Runs on a LARG box. Builds the `ilab` uv venv and installs Isaac Lab + our
# source packages, mirroring the justfile `setup` recipe minus the manager .venv
# and gitman (resources/ are rsynced).
#
# No sudo: cmake is preinstalled on these boxes, so isaaclab.sh -i won't apt.
#
# Usage (on remote):  bash ~/hcrl_isaac_manager/scripts/larg/remote_setup.sh
# Idempotent enough to re-run; pass FORCE=1 to wipe and rebuild the venv.

set -euo pipefail

MANAGER_DIR="${MANAGER_DIR:-$HOME/hcrl_isaac_manager}"
VENV_NAME="ilab"
ILAB_DIR="$MANAGER_DIR/resources/IsaacLab"
VENV_DIR="$ILAB_DIR/$VENV_NAME"   # this is a symlink -> $SCRATCH/ilab (see below)

# Put venv and uv cache on
# The NFS home is per-user quota-limited (~18-20G), too big for venv/uv cache
SCRATCH="${LARG_SCRATCH:-/var/local/$USER}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv_cache}"

# Accept the Isaac Sim EULA non-interactively
export ACCEPT_EULA=Y

export PATH="$HOME/.local/bin:$PATH"

# Write a clear terminal marker on failure so completion polling never has to
# guess from process state (pgrep -f self-matches and lies).
trap 'rc=$?; [ $rc -ne 0 ] && echo "[setup] FAILED rc=$rc on $(hostname)"' EXIT

echo "[setup] host=$(hostname) manager=$MANAGER_DIR"
echo "[setup] scratch=$SCRATCH  UV_CACHE_DIR=$UV_CACHE_DIR"
mkdir -p "$SCRATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

# Relocate any pre-existing home uv cache onto scratch (preserve downloads, free
# the home quota). Safe if it doesn't exist or is already moved.
if [ -d "$HOME/.cache/uv" ] && [ ! -L "$HOME/.cache/uv" ]; then
  echo "[setup] moving ~/.cache/uv -> $UV_CACHE_DIR (free home quota)"
  rm -rf "$UV_CACHE_DIR"
  mv "$HOME/.cache/uv" "$UV_CACHE_DIR"
fi

# install uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv (user-local, no sudo)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv: $(uv --version)"

# install just
if ! command -v just >/dev/null 2>&1; then
  echo "[setup] installing just via uv..."
  uv tool install rust-just
fi
echo "[setup] just: $(just --version 2>/dev/null || echo 'n/a')"

# setup ilab venv
REAL_VENV="$SCRATCH/$VENV_NAME"
if [ "${FORCE:-0}" = "1" ]; then
  echo "[setup] FORCE=1 -> removing existing venv $REAL_VENV"
  rm -rf "$REAL_VENV"
fi
# Drop any stale/partial venv that was created directly in the (quota-limited) tree.
[ -e "$VENV_DIR" ] && [ ! -L "$VENV_DIR" ] && { echo "[setup] removing in-tree partial venv $VENV_DIR"; rm -rf "$VENV_DIR"; }

if [ ! -d "$REAL_VENV" ]; then
  echo "[setup] creating venv (python 3.11) at $REAL_VENV"
  uv venv --python 3.11 "$REAL_VENV"
fi
ln -sfn "$REAL_VENV" "$VENV_DIR"   # resources/IsaacLab/ilab -> $SCRATCH/ilab

# logs dir on scratch (big disk), symlinked into the tree like the venv
mkdir -p "$SCRATCH/logs"
ln -sfn "$SCRATCH/logs" "$ILAB_DIR/logs"   # resources/IsaacLab/logs -> $SCRATCH/logs

cd "$ILAB_DIR"
# shellcheck disable=SC1090
source "$VENV_NAME/bin/activate"

echo "[setup] pip upgrade"
uv pip install --upgrade pip

echo "[setup] installing isaacsim 5.1.0 (this is the big download)..."
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "[setup] pinning torch 2.7.0 / torchvision 0.22.0 (cu128)..."
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "[setup] isaaclab.sh -u $VENV_NAME (register uv env)"
./isaaclab.sh -u "$VENV_NAME"

echo "[setup] isaaclab.sh -i rsl_rl (install all source/ extensions editable + rsl_rl)"
./isaaclab.sh -i rsl_rl

# sanity check
echo "[setup] AppLauncher smoke test (bootstraps Kit headless)..."
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES "$VENV_DIR/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab, isaaclab_rl, robot_rl, hcrl_isaaclab  # noqa: F401
print("ok: isaaclab + extensions import under Kit")
app.close()
print("SMOKE_OK")
PY

echo "[setup] DONE on $(hostname)"
