#!/bin/bash
# One T1 velocity training run on one GPU of a box set up by remote_setup.sh, logged to W&B.
# usage: train_remote.sh ROOT GPU PHYSICS TAG ITERS [NUM_ENVS]  (ITERS "scaled" = the task budget scaled to NUM_ENVS)
ROOT=$(readlink -f "$1"); GPU=$2; PHYS=$3; TAG=$4; ITERS=$5; ENVS=$6
EXTRA=(); [ "$ITERS" != scaled ] && EXTRA+=(--max_iterations $ITERS); [ -n "$ENVS" ] && EXTRA+=(--num_envs $ENVS)
mkdir -p $ROOT/train $ROOT/tmp/$TAG-$PHYS
export TMPDIR=$ROOT/tmp/$TAG-$PHYS CUDA_VISIBLE_DEVICES=$GPU XDG_CACHE_HOME=$ROOT/tmp/$TAG-$PHYS/cache WARP_CACHE_PATH=$ROOT/tmp/$TAG-$PHYS/warp
set -a; [ -f $ROOT/.env.wandb ] && source $ROOT/.env.wandb; set +a
cd $ROOT/train
exec $ROOT/venv/bin/python $ROOT/resources/hcrl_isaaclab/scripts/train.py --task T1-Velocity-v0 "${EXTRA[@]}" \
  --logger wandb --run_name "isaac3-${PHYS}-${TAG}" --run_group isaac3-t1vel-compare --video off physics=$PHYS \
  > $ROOT/train/${TAG}_${PHYS}.log 2>&1
