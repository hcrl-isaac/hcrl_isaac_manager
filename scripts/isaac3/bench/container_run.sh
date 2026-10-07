#!/bin/bash
# Run a bench script inside a stock Debian bookworm (python:3.12, has git) container: Isaac Lab 3.0 wheels need glibc >= 2.35,
# which the HPC hosts (Stampede3 2.34, Delta 2.28) lack. Driver comes in through --nv.
# usage: container_run.sh ROOT SCRIPT [args...]
ROOT=$(readlink -f "$1"); shift
command -v apptainer >/dev/null || export PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
SIF=$ROOT/bookworm.sif
if [ ! -f $SIF ]; then
  APPTAINER_CACHEDIR=$ROOT/apptainer-cache APPTAINER_TMPDIR=$ROOT/tmp apptainer pull $SIF docker://python:3.12-bookworm || exit 1
fi
# an empty CUDA_VISIBLE_DEVICES hides every GPU, so forward it only when set
ENVS=HOME=$ROOT/home${CUDA_VISIBLE_DEVICES:+,CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
exec apptainer exec --nv --cleanenv -B $ROOT -B /dev/shm --env $ENVS $SIF bash "$@"
