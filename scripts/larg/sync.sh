#!/usr/bin/env bash
# Sync the manager + source code to a LARG box, excluding venvs, datasets, logs,
# docker, and other large artifacts that are rebuilt or unneeded on the remote.
#
# Usage: scripts/larg/sync.sh <host> [<host> ...]
#        scripts/larg/sync.sh mckennie hazard

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

# LARG home is one quota'd NFS share across every box, so bulk data must stay excluded; keep the
# patterns matched to the flat resources/ layout (stale paths exclude nothing).
EXCLUDES=(
  # virtualenvs
  --exclude='.venv/'
  --exclude='ilab/'
  # git metadata
  --exclude='.git/'
  # local run outputs
  --exclude='wandb/'
  --exclude='outputs/'
  --exclude='logs/'
  --exclude='worktrees/'
  --exclude='artifacts/'
  # container images
  --exclude='scripts/cluster/'
  --exclude='resources/IsaacLab/docker/'
  # datasets other tasks use
  --exclude='resources/motion_datasets/'
  --exclude='resources/gigahands/'
  --exclude='resources/gigahands_leap_csv/'
  --exclude='resources/grab/'
  --exclude='resources/body_models/'
  --exclude='resources/loco_mujoco_g1/'
  --exclude='resources/lafan1_lvhaidong/'
  --exclude='resources/robot_rl-cudagraph/'
  # onnx duplicates of the .pt policies
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
  # Symlinks stay links: their targets are inside the synced tree.
  rsync -az --partial --mkpath --info=stats1,progress2 \
    "${EXCLUDES[@]}" \
    -e "ssh -o ConnectTimeout=10" \
    "${LARG_LOCAL_DIR}/" \
    "${target}:${LARG_REMOTE_DIR}/"
  echo "=== done: ${host} ==="
done
