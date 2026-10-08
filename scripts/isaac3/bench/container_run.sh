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
# TACC's XALT preloads a library through these that needs glibc 2.38 (bookworm has 2.36), so every exec would die
unset LD_PRELOAD SINGULARITYENV_LD_PRELOAD APPTAINERENV_LD_PRELOAD SINGULARITY_BINDPATH APPTAINER_BINDPATH
mkdir -p $ROOT/home
# an empty CUDA_VISIBLE_DEVICES hides every GPU, so forward it only when set. It goes through APPTAINERENV_, as --env
# splits its value on the commas of a multi-GPU list; HOME goes through --home, which --env cannot set.
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
exec apptainer exec --nv --cleanenv --home $ROOT/home -B $ROOT -B /dev/shm --env PYTHONUNBUFFERED=1 $SIF bash "$@"
