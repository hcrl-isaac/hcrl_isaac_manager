#!/bin/bash
# Same-GPU throughput of T1 velocity training on Newton (MJWarp) then PhysX (kit-less OV PhysX).
# usage: bench.sh ROOT GPU TAG [ITERS] [NUM_ENVS]   -> ROOT/bench/<TAG>_<physics>.log + results line in ROOT/bench/results.txt
ROOT=$(readlink -f "$1"); GPU=$2; TAG=$3; ITERS=${4:-40}; ENVS=${5:-4096}
mkdir -p $ROOT/bench $ROOT/tmp/$TAG
export TMPDIR=$ROOT/tmp/$TAG CUDA_VISIBLE_DEVICES=$GPU
export UV_CACHE_DIR=$ROOT/uv-cache XDG_CACHE_HOME=$ROOT/tmp/$TAG/cache WARP_CACHE_PATH=$ROOT/tmp/$TAG/warp
cd $ROOT/bench
for phys in newton_mjwarp physx; do
  log=$ROOT/bench/${TAG}_${phys}.log
  $ROOT/venv/bin/python $ROOT/resources/hcrl_isaaclab/scripts/train.py --task T1-Velocity-v0 --num_envs $ENVS --no_scale \
    --max_iterations $ITERS --logger tensorboard --video off physics=$phys > $log 2>&1
  echo "$TAG $phys exit $? $($ROOT/venv/bin/python $(dirname "$(readlink -f "$0")")/parse_bench.py $log $ENVS)" | tee -a $ROOT/bench/results.txt
done
