#!/usr/bin/env bash
# Sync the manager + source code to a LARG box, excluding venvs, datasets, logs,
# docker, and other large artifacts that are rebuilt or unneeded on the remote.
#
# Usage: scripts/larg/sync.sh <host> [<host> ...]
#        scripts/larg/sync.sh mckennie hazard

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

# Exclude .venv and big motion datasets / FB-CPR checkpoints / docker images /
# local run logs (not needed for locomanipulation meta-RL).
EXCLUDES=(
  # virtualenvs (rebuilt on remote)
  --exclude='.venv/'
  --exclude='ilab/'
  # git metadata + lfs (code-state not needed on remote; store_code_state=False)
  --exclude='.git/'
  # local run outputs / logs / wandb
  --exclude='wandb/'
  --exclude='outputs/'
  --exclude='worktrees/'
  --exclude='resources/IsaacLab/logs/'
  --exclude='resources/IsaacLab/wandb/'
  --exclude='resources/IsaacLab/outputs/'
  --exclude='resources/IsaacLab/source/hcrl_isaaclab/logs/'
  # docker images (50G; venv install, no containers)
  --exclude='resources/IsaacLab/docker/'
  # large datasets for OTHER tasks (FB-CPR / GRAB / GigaHands / loco_mujoco)
  --exclude='resources/IsaacLab/source/hcrl_isaaclab/resources/motion_datasets/'
  --exclude='resources/gigahands/'
  --exclude='resources/gigahands_leap_csv/'
  --exclude='resources/grab/'
  --exclude='resources/body_models/'
  --exclude='resources/loco_mujoco_g1/'
  --exclude='resources/lafan1_lvhaidong/'
  --exclude='resources/robot_rl-cudagraph/'
  # not needed for locomanip; saves ~1G
  --exclude='resources/IsaacLab/source/hcrl_isaaclab/resources/ssti_robots/'
  # onnx duplicates of the .pt policies (training loads .pt only)
  --exclude='*.onnx'
  # python caches
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.pytest_cache/'
  --exclude='*.egg-info/'
)

[ $# -ge 1 ] || { echo "usage: $0 <host> [<host> ...]"; exit 1; }

for host in "$@"; do
  target="$(larg_target "$host")"
  echo "=== rsync -> ${target}:${LARG_REMOTE_DIR}/ ==="
  # -a archive, -z compress, -L copy-unsafe symlinks as files? No: keep symlinks
  # (the locomanip policy symlinks point within the synced tree, so they resolve).
  rsync -az --partial --info=stats1,progress2 \
    "${EXCLUDES[@]}" \
    -e "ssh -o ConnectTimeout=10" \
    "${LARG_LOCAL_DIR}/" \
    "${target}:${LARG_REMOTE_DIR}/"
  echo "=== done: ${host} ==="
done
