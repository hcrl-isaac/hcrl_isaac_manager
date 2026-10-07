#!/bin/bash
# Kit-less smoke of a task on both physics backends (16 envs, 10 steps).
# usage: smoke_all.sh [TASK]
D=$(dirname "$(readlink -f "$0")")
TASK=${1:-hcrl/T1-Velocity-v0}
for ph in newton_mjwarp physx; do
  $D/ea_py.sh $D/probe/smoke_env.py "$TASK" physics=$ph env.scene.num_envs=16 --steps 10 2>&1 | grep -E "SMOKE|Traceback|Error:" | tail -2
done
