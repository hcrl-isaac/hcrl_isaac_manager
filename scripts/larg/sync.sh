#!/usr/bin/env bash
# Sync the manager + source code to a LARG box, excluding venvs, datasets, logs,
# docker, and other large artifacts that are rebuilt or unneeded on the remote.
#
# Usage: scripts/larg/sync.sh <host> [<host> ...]
#        scripts/larg/sync.sh mckennie hazard

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

# LARG home is one NFS share across every box, under a hard quota of a few tens of GB -- a single
# unexcluded dataset dir overruns it and rsync dies mid-transfer with "Disk quota exceeded". Keep
# this list matched to the CURRENT flat resources/ layout: stale pre-reorg paths exclude nothing.
EXCLUDES=(
  # virtualenvs (rebuilt per box on /var/local scratch)
  --exclude='.venv/'
  --exclude='ilab/'
  # git metadata + lfs (code-state not needed on remote; store_code_state=False)
  --exclude='.git/'
  # local run outputs / logs / wandb
  --exclude='wandb/'
  --exclude='outputs/'
  --exclude='logs/'
  --exclude='worktrees/'
  --exclude='artifacts/'
  # container images: the cluster .sif alone is ~9G and LARG runs in a venv, not a container
  --exclude='scripts/cluster/'
  --exclude='resources/IsaacLab/docker/'
  # large datasets for OTHER tasks (FB-CPR motions / GRAB / GigaHands / loco_mujoco)
  --exclude='resources/motion_datasets/'
  --exclude='resources/gigahands/'
  --exclude='resources/gigahands_leap_csv/'
  --exclude='resources/grab/'
  --exclude='resources/body_models/'
  --exclude='resources/loco_mujoco_g1/'
  --exclude='resources/lafan1_lvhaidong/'
  --exclude='resources/robot_rl-cudagraph/'
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
  rsync -az --partial --mkpath --info=stats1,progress2 \
    "${EXCLUDES[@]}" \
    -e "ssh -o ConnectTimeout=10" \
    "${LARG_LOCAL_DIR}/" \
    "${target}:${LARG_REMOTE_DIR}/"
  echo "=== done: ${host} ==="
done
