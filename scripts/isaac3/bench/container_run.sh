#!/bin/bash
# Run a bench script inside a stock Ubuntu 24.04 container: Isaac Lab 3.0 wheels need glibc >= 2.35,
# which the HPC hosts (Stampede3 2.34, Delta 2.28) lack. Driver comes in through --nv.
# usage: container_run.sh ROOT SCRIPT [args...]
ROOT=$(readlink -f "$1"); shift
command -v apptainer >/dev/null || export PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
SIF=$ROOT/ubuntu24.sif
if [ ! -f $SIF ]; then
  APPTAINER_CACHEDIR=$ROOT/apptainer-cache APPTAINER_TMPDIR=$ROOT/tmp apptainer pull $SIF docker://ubuntu:24.04 || exit 1
fi
exec apptainer exec --nv --cleanenv -B $ROOT -B /dev/shm --env HOME=$ROOT/home,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} $SIF bash "$@"
