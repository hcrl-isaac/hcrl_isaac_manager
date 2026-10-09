#!/bin/bash
# Run a bench script inside a stock Debian bookworm (python:3.12, has git) container: Isaac Lab 3.0 wheels need glibc >= 2.35,
# which the HPC hosts (Stampede3 2.34, Delta 2.28) lack. Driver comes in through --nv.
# usage: container_run.sh ROOT SCRIPT [args...]
ROOT=$(readlink -f "$1"); shift
command -v apptainer >/dev/null || export PATH=/opt/apps/tacc-apptainer/1.4.1/bin:$PATH
SIF=$ROOT/bookworm.sif
if [ ! -f $SIF ]; then
  # a fresh root has neither dir, and the pull needs both. It lands under a per-process name and is renamed, so a
  # concurrent job never sees a half-written image
  mkdir -p $ROOT/tmp $ROOT/apptainer-cache
  APPTAINER_CACHEDIR=$ROOT/apptainer-cache APPTAINER_TMPDIR=$ROOT/tmp \
    apptainer pull $SIF.partial.$$ docker://python:3.12-bookworm || { rm -f $SIF.partial.$$; exit 1; }
  mv -f $SIF.partial.$$ $SIF
fi
# TACC's XALT preloads a library through these that needs glibc 2.38 (bookworm has 2.36), so every exec would die
unset LD_PRELOAD SINGULARITYENV_LD_PRELOAD APPTAINERENV_LD_PRELOAD SINGULARITY_BINDPATH APPTAINER_BINDPATH
mkdir -p $ROOT/home
# an empty CUDA_VISIBLE_DEVICES hides every GPU, so forward it only when set. It goes through APPTAINERENV_, as --env
# splits its value on the commas of a multi-GPU list; HOME goes through --home, which --env cannot set.
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# --cleanenv drops it, and a job's per-job dirs are named by it
[ -n "${SLURM_JOB_ID:-}" ] && export APPTAINERENV_SLURM_JOB_ID=$SLURM_JOB_ID
exec apptainer exec --nv --cleanenv --home $ROOT/home -B $ROOT -B /dev/shm --env PYTHONUNBUFFERED=1 $SIF bash "$@"
