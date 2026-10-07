#!/bin/bash
# setup_and_bench.sh for hosts whose glibc is too old: venv build and bench both run in the Ubuntu 24.04 container.
# usage: container_setup_and_bench.sh ROOT GPU TAG [ITERS] [NUM_ENVS]
ROOT=$(readlink -f "$1"); GPU=$2; TAG=$3
D=$(dirname "$(readlink -f "$0")")
mkdir -p $ROOT/bench $ROOT/home $ROOT/tmp
bash $D/container_run.sh $ROOT $D/remote_setup.sh $ROOT > $ROOT/bench/${TAG}_setup.log 2>&1 \
  || { echo "$TAG setup FAILED" >> $ROOT/bench/results.txt; exit 1; }
bash $D/container_run.sh $ROOT $D/bench.sh "$@"
echo "$TAG done" >> $ROOT/bench/results.txt
