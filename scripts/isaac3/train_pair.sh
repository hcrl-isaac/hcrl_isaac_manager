#!/bin/bash
# Train T1 velocity on Newton (MJWarp) and PhysX (kit-less OV PhysX) side by side on one GPU.
# usage: train_pair.sh ITERS TAG
D=$(dirname "$(readlink -f "$0")"); R=$(readlink -f "$D/../../resources")
ITERS=${1:-3000}; TAG=${2:-hcrl2}
mkdir -p $R/../runs/isaac3 && cd $R/../runs/isaac3
/tmp/claude-1000/gpu_lock.sh acquire newton-port "T1 velocity Newton vs PhysX training ($ITERS it)" 4h $$
trap '/tmp/claude-1000/gpu_lock.sh release newton-port' EXIT
for phys in newton_mjwarp physx; do
  $D/ea_py.sh $R/hcrl_isaaclab/scripts/train.py --task T1-Velocity-v0 --max_iterations $ITERS \
    --logger wandb --run_name "isaac3-${phys}-${TAG}" --run_group isaac3-t1vel-compare --video off \
    physics=$phys > train_${phys}_${TAG}.log 2>&1 &
  echo "started $phys pid $!"
  sleep 60  # stagger Kit-less boots (shared asset conversion cache)
done
wait
echo "both finished"
