#!/bin/bash
# Record a checkpoint's fixed-command rollout on a box set up by remote_setup.sh.
# usage: record_remote.sh ROOT GPU CHECKPOINT OUT.npz PHYSICS
ROOT=$(readlink -f "$1"); GPU=$2; CKPT=$3; OUT=$4; PHYS=$5
mkdir -p $ROOT/tmp/record-$PHYS $(dirname "$OUT")
export TMPDIR=$ROOT/tmp/record-$PHYS CUDA_VISIBLE_DEVICES=$GPU XDG_CACHE_HOME=$ROOT/tmp/record-$PHYS/cache
cd $ROOT/tmp/record-$PHYS
$ROOT/venv/bin/python $(dirname "$(readlink -f "$0")")/../probe/record_rollout.py "$CKPT" "$OUT" physics=$PHYS --num_envs 4 2>&1 | grep -E "RECORDED|Traceback|Error" -A3
