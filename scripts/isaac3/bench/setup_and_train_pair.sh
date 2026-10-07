#!/bin/bash
# usage: setup_and_train_pair.sh ROOT NEWTON_GPU PHYSX_GPU TAG ITERS
ROOT=$1; NG=$2; PG=$3; TAG=$4; ITERS=$5
D=$(dirname "$(readlink -f "$0")")
mkdir -p $ROOT/train
[ -x $ROOT/venv/bin/python ] || bash $D/remote_setup.sh $ROOT > $ROOT/train/${TAG}_setup.log 2>&1 || exit 1
bash $D/train_remote.sh $ROOT $NG newton_mjwarp $TAG $ITERS &
sleep 90  # stagger the two boots
bash $D/train_remote.sh $ROOT $PG physx $TAG $ITERS &
wait
