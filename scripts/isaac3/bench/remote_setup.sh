#!/bin/bash
# Build the Isaac Lab 3.0 (kit-less) venv for the synced trees under ROOT; every cache stays under ROOT.
# usage: remote_setup.sh ROOT
set -euo pipefail
ROOT=$(readlink -f "$1")
export UV_CACHE_DIR=$ROOT/uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/python UV_PROJECT_ENVIRONMENT=$ROOT/venv
export PATH=$HOME/.local/bin:$ROOT/bin:$PATH
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$ROOT/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
R=$ROOT/resources
uv sync -q --project $R/IsaacLab --frozen --extra rsl-rl --extra ovphysx --extra importers
uv pip install -q --python $ROOT/venv/bin/python --torch-backend cu128 -e $R/robot_rl -e $R/hcrl_isaaclab
cd $ROOT && $ROOT/venv/bin/python -c "import torch, hcrl_isaaclab, isaaclab_newton; print('SETUP OK', torch.__version__, torch.cuda.device_count(), 'gpus')"
