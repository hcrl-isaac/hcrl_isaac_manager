#!/usr/bin/env bash

if [ -z $CLUSTER ]; then
    CLUSTER="default"
fi;

# make the cwd the pwd
cd "$(dirname "$0")"

CONFIG_DIR="cluster/${CLUSTER}_config"
if [ ! -d $CONFIG_DIR ]; then
    echo "[ERROR] Config directory $CONFIG_DIR not found for cluster $CLUSTER. Create config from template with \`CLUSTER=$CLUSTER just add-cluster\`."
    exit 1
fi;

cp .env.wandb  ../resources/IsaacLab/docker/cluster/.env.wandb
cp cluster/cluster_interface.sh  ../resources/IsaacLab/docker/cluster/cluster_interface.sh
cp cluster/run_singularity.sh  ../resources/IsaacLab/docker/cluster/run_singularity.sh
cp $CONFIG_DIR/.env.cluster ../resources/IsaacLab/docker/cluster/.env.cluster
cp $CONFIG_DIR/submit_job_slurm.sh  ../resources/IsaacLab/docker/cluster/submit_job_slurm.sh
# Also mirror the cluster_dev/ subtree so `cluster_interface.sh develop` can find the dispatcher.
# We previously only auto-copied node_exec.sh; cluster_dev.sh and sentinel.sbatch lived only in
# scripts/cluster/cluster_dev/ and were missing at the destination, breaking `develop start`.
mkdir -p ../resources/IsaacLab/docker/cluster/cluster_dev
cp cluster/cluster_dev/cluster_dev.sh   ../resources/IsaacLab/docker/cluster/cluster_dev/cluster_dev.sh
cp cluster/cluster_dev/node_exec.sh     ../resources/IsaacLab/docker/cluster/cluster_dev/node_exec.sh
cp cluster/cluster_dev/sentinel.sbatch  ../resources/IsaacLab/docker/cluster/cluster_dev/sentinel.sbatch
# Mirror the per-cluster config dir too -- cluster_dev.sh resolves its env file relative to its
# own location (..//${CLUSTER}_config/.env.cluster), unlike cluster_interface.sh which jumps
# back to the canonical scripts/cluster/ path. Without this copy, `develop start` aborts with
# `CLUSTER_LOGIN not set (expected from ${CLUSTER}_config/.env.cluster)`.
mkdir -p ../resources/IsaacLab/docker/cluster/${CLUSTER}_config
cp $CONFIG_DIR/.env.cluster                ../resources/IsaacLab/docker/cluster/${CLUSTER}_config/.env.cluster
cp $CONFIG_DIR/submit_job_slurm.sh         ../resources/IsaacLab/docker/cluster/${CLUSTER}_config/submit_job_slurm.sh 2>/dev/null || true

cd ../resources/IsaacLab/docker/cluster
./cluster_interface.sh "${@:1}"
