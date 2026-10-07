#!/bin/bash
# Sync the minimal Isaac Lab 3.0 tree set (no venvs, no logs) to SSH_HOST:ROOT/resources.
# usage: deploy.sh SSH_HOST ROOT [extra rsync/ssh args via RSYNC_RSH]
set -euo pipefail
HOST=$1; ROOT=$2
R=$(readlink -f "$(dirname "$0")/../../../resources")
ssh ${SSH_OPTS:-} $HOST "mkdir -p $ROOT/resources/hcrl_robots"
EX=(--exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=logs --exclude=runs --exclude=outputs --exclude='*.egg-info')
rsync -az ${RSYNC_RSH:+-e "$RSYNC_RSH"} "${EX[@]}" $R/IsaacLab $R/robot_rl $R/hcrl_isaaclab $R/../scripts/isaac3 $HOST:$ROOT/resources/
rsync -azL ${RSYNC_RSH:+-e "$RSYNC_RSH"} "${EX[@]}" $R/hcrl_robots/t1 $HOST:$ROOT/resources/hcrl_robots/
echo "deployed to $HOST:$ROOT"
