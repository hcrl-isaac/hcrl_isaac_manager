#!/bin/bash
# Record each run's checkpoint under both physics backends (trained-on x evaluated-in), two at a time.
# usage: record_matrix.sh ROOT ITER NEWTON_RUN_DIR PHYSX_RUN_DIR OUT_DIR
ROOT=$1; IT=$2; NRUN=$3; PRUN=$4; OUT=$5
D=$(dirname "$(readlink -f "$0")")
mkdir -p $OUT
for evalphys in newton_mjwarp physx; do
  bash $D/record_remote.sh $ROOT 1 $NRUN/model_$IT.pt $OUT/trained-newton_eval-${evalphys}_$IT.npz $evalphys &
  bash $D/record_remote.sh $ROOT 3 $PRUN/model_$IT.pt $OUT/trained-physx_eval-${evalphys}_$IT.npz $evalphys &
  wait
done
ls $OUT/*_$IT.npz
