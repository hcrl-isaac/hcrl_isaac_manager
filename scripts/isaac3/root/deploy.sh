#!/bin/bash
# Sync the minimal Isaac Lab 3.0 tree set (no venvs, no logs) to SSH_HOST:ROOT/resources.
# usage: deploy.sh SSH_HOST ROOT [TASK_REPO ...]   (task repos under resources/; default hhlm_tasks when present;
#        extra rsync/ssh args via RSYNC_RSH)
# then build its venv on the host: bash ROOT/resources/isaac3/root/remote_setup.sh ROOT. The copies of
# hcrl_isaaclab, robot_rl, the task repos and hcrl_robots/t1 exist only for the venv's editable installs (their
# dependencies); nothing refreshes them. Runs go through `pls run --root ROOT`, whose staged tree comes first on
# PYTHONPATH, so a run imports the staged code, never these copies.
set -euo pipefail
HOST=$1; ROOT=$2; shift 2
R=$(readlink -f "$(dirname "$0")/../../../resources")
TASKS=()
for t in "$@"; do
  [ -d "$R/$t" ] || { echo "[deploy] no repo resources/$t" >&2; exit 1; }
  TASKS+=("$R/$t")
done
[ $# -gt 0 ] || [ ! -d "$R/hhlm_tasks" ] || TASKS=("$R/hhlm_tasks")
ssh ${SSH_OPTS:-} $HOST "mkdir -p $ROOT/resources/hcrl_robots"
# a repo's worktrees and session state never ship: they can dwarf the repo
EX=(--exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=logs --exclude=runs --exclude=outputs --exclude='*.egg-info'
    --exclude='/*/worktrees' --exclude='/*/.claude')
rsync -az ${RSYNC_RSH:+-e "$RSYNC_RSH"} "${EX[@]}" $R/IsaacLab $R/robot_rl $R/hcrl_isaaclab "${TASKS[@]}" \
  $R/../scripts/isaac3 $HOST:$ROOT/resources/
rsync -azL ${RSYNC_RSH:+-e "$RSYNC_RSH"} "${EX[@]}" $R/hcrl_robots/t1 $HOST:$ROOT/resources/hcrl_robots/
echo "deployed to $HOST:$ROOT"
