#!/bin/bash
# Build the Isaac Lab 3.0 (kit-less) venv for the synced trees under ROOT; every cache stays under ROOT.
# usage: remote_setup.sh ROOT   (EXTRA_EXTRAS="isaacsim video" adds Isaac Lab extras, e.g. for an RTX render box)
set -euo pipefail
ROOT=$(readlink -f "$1")
export UV_CACHE_DIR=$ROOT/uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/python UV_PROJECT_ENVIRONMENT=$ROOT/venv
export PATH=$HOME/.local/bin:$ROOT/bin:$PATH
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$ROOT/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
R=$ROOT/resources
uv sync -q --project $R/IsaacLab --frozen --extra rsl-rl --extra ovphysx --extra importers $(for e in ${EXTRA_EXTRAS:-}; do printf -- "--extra %s " "$e"; done)
# the lock resolves torch cu130 on aarch64 (GB200, DGX Spark); a cu128 backend here would swap it
BACKEND=cu128; [ "$(uname -m)" = aarch64 ] && BACKEND=cu130
PKGS=(-e $R/robot_rl -e $R/hcrl_isaaclab); [ -d $R/hhlm_tasks ] && PKGS+=(-e $R/hhlm_tasks)
# Isaac Lab EA's lock pins pydantic 2.14.0a1, whose models reject wandb's artifact-file responses
PKGS+=("pydantic==2.11.10")
uv pip install -q --python $ROOT/venv/bin/python --torch-backend $BACKEND "${PKGS[@]}"
cd $ROOT && $ROOT/venv/bin/python -c "import torch, hcrl_isaaclab, isaaclab_newton; print('SETUP OK', torch.__version__, torch.cuda.device_count(), 'gpus')"
