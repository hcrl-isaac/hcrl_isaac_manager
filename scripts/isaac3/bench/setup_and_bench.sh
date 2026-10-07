#!/bin/bash
# usage: setup_and_bench.sh ROOT GPU TAG [ITERS] [NUM_ENVS]   (log: ROOT/bench/<TAG>_setup.log)
ROOT=$1; GPU=$2; TAG=$3
mkdir -p $ROOT/bench
D=$(dirname "$(readlink -f "$0")")
bash $D/remote_setup.sh $ROOT > $ROOT/bench/${TAG}_setup.log 2>&1 || { echo "$TAG setup FAILED" >> $ROOT/bench/results.txt; exit 1; }
bash $D/bench.sh "$@"
echo "$TAG done" >> $ROOT/bench/results.txt
