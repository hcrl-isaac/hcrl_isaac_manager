#!/bin/bash
# Newton and PhysX T1 velocity training side by side, each in the bookworm container (old-glibc HPC hosts).
# usage: container_train_pair.sh ROOT NEWTON_GPU PHYSX_GPU TAG ITERS [NUM_ENVS]
ROOT=$(readlink -f "$1"); NG=$2; PG=$3; TAG=$4; ITERS=$5; ENVS=$6
D=$(dirname "$(readlink -f "$0")")
mkdir -p $ROOT/train
bash $D/container_run.sh $ROOT $D/train_remote.sh $ROOT $NG newton_mjwarp $TAG $ITERS $ENVS &
sleep 90  # stagger the two boots
bash $D/container_run.sh $ROOT $D/train_remote.sh $ROOT $PG physx $TAG $ITERS $ENVS &
wait
