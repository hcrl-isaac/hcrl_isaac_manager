#!/usr/bin/env bash
# Build the `ilab` venv on a LARG box, from the workspace already rsynced to the shared home.
#
# Storage on LARG is split and it matters: /u/$USER is ONE NFS share that every box mounts, under a
# hard quota of a few tens of GB, while /var/local is per-box local disk with terabytes free. So code
# lives in the shared home and everything big -- venv, uv cache, Isaac's caches, logs -- goes to
# scratch. A venv therefore has to be built once per box even though the code is already there.
#
#   bash scripts/larg/remote_setup.sh          # build if missing
#   FORCE=1 bash scripts/larg/remote_setup.sh  # rebuild from scratch
set -euo pipefail

MANAGER_DIR="${MANAGER_DIR:-$HOME/hcrl_isaac_manager}"
SCRATCH="${LARG_SCRATCH:-/var/local/$USER}"
VENV="$SCRATCH/ilab"
PY="$VENV/bin/python"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv_cache}"
export UV_PROJECT_ENVIRONMENT="$VENV"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
export PATH="$HOME/.local/bin:$PATH"
# Kit writes GB-scale shader and extension caches; left on $HOME they blow the shared quota.
export OMNI_CACHE_DIR="$SCRATCH/omni_cache"
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"

trap 'rc=$?; [ $rc -ne 0 ] && echo "[setup] FAILED rc=$rc on $(hostname)"' EXIT

echo "[setup] host=$(hostname)  manager=$MANAGER_DIR  scratch=$SCRATCH"
[ -d "$MANAGER_DIR" ] || { echo "[setup] no workspace at $MANAGER_DIR -- rsync it first (scripts/larg/deploy.sh)"; exit 1; }
mkdir -p "$SCRATCH" "$OMNI_CACHE_DIR" "$XDG_CACHE_HOME" "$SCRATCH/logs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

command -v uv >/dev/null 2>&1 || { echo "[setup] installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
echo "[setup] uv $(uv --version)"

[ "${FORCE:-0}" = "1" ] && { echo "[setup] FORCE=1 -> removing $VENV"; rm -rf "$VENV"; }
[ -d "$VENV" ] || uv venv --relocatable --python 3.11 "$VENV"
# The symlink lives in the shared home but resolves to the local disk of whichever box reads it.
ln -sfn "$VENV" "$MANAGER_DIR/ilab"
ln -sfn "$SCRATCH/logs" "$MANAGER_DIR/logs"

cd "$MANAGER_DIR"
mode=$(grep -E '^[[:space:]]*mode:' workspace.yaml 2>/dev/null | head -1 | sed -E 's/.*mode:[[:space:]]*//; s/[[:space:]#].*//')
echo "[setup] IsaacLab mode: ${mode:-pip}"

echo "[setup] manager deps"
uv sync --frozen 2>/dev/null || uv pip install --python "$PY" -r pyproject.toml

echo "[setup] torch 2.7.0 / torchvision 0.22.0 (cu128 -- sm_120 needs it)"
uv pip install --python "$PY" --torch-backend cu128 torch==2.7.0 torchvision==0.22.0

echo "[setup] Isaac Lab + Isaac Sim"
if [ "$mode" = "source" ]; then
    uv pip install --python "$PY" "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    for d in resources/IsaacLab/source/isaaclab*/; do
        [ -d "$d" ] && uv pip install --python "$PY" --torch-backend cu128 -e "$d"
    done
    uv pip install --python "$PY" --torch-backend cu128 -e resources/hcrl_isaaclab
else
    # hcrl_isaaclab[isaacsim] pulls isaaclab[isaacsim]==2.3.2.post1, which brings isaacsim 5.1 with it
    uv pip install --python "$PY" --torch-backend cu128 --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match -e "resources/hcrl_isaaclab[isaacsim]"
fi
uv pip install --python "$PY" rsl_rl-lib

echo "[setup] workspace packages"
for d in resources/robot_rl resources/*_tasks resources/*_robots resources/holosoma/src/holosoma_retargeting; do
    if [ -d "$d" ] && { [ -f "$d/setup.py" ] || [ -f "$d/pyproject.toml" ]; }; then
        uv pip install --python "$PY" --torch-backend cu128 --extra-index-url https://pypi.nvidia.com -e "$d"
    elif [ -d "$d" ]; then
        echo "[setup]   skipping data repo $d"
    fi
done

# Isaac Lab hardcodes its URDF/MJCF conversion scratch to /tmp/IsaacLab. LARG boxes are shared, so that
# directory belongs to whichever user booted Isaac first and is mode 0700 -- every other user then dies
# with PermissionError before the sim starts. Make it honour TMPDIR, which runs point at local scratch.
CONV="$VENV/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/converters/asset_converter_base.py"
if [ -f "$CONV" ] && grep -q 'f"/tmp/IsaacLab/usd_' "$CONV"; then
    python3 - "$CONV" <<'PATCH'
import pathlib, sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = text.replace("import random\n", "import random\nimport tempfile\n", 1)
text = text.replace(
    'self._usd_dir = f"/tmp/IsaacLab/usd_{time_tag}_{random.randrange(10000)}"',
    'self._usd_dir = os.path.join(tempfile.gettempdir(), "IsaacLab", f"usd_{time_tag}_{random.randrange(10000)}")',
    1,
)
path.write_text(text)
print("[setup]   patched asset_converter_base to honour TMPDIR")
PATCH
fi

# LARG boxes are shared: pin the smoke test to the GPU this deploy is claiming, not to card 0.
export CUDA_VISIBLE_DEVICES="${LARG_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
echo "[setup] smoke test (boots Kit headless on GPU $CUDA_VISIBLE_DEVICES)"
"$PY" - <<'PY'
from importlib import metadata

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import isaaclab, isaaclab_rl, robot_rl, hcrl_isaaclab  # noqa: F401

print("isaaclab", metadata.version("isaaclab"))
app.close()
print("SMOKE_OK")
PY

echo "[setup] DONE on $(hostname) -- venv $VENV"
