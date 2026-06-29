#!/bin/bash
# Sentinel job: holds a compute node for the full walltime so we can ssh into it and develop
# interactively. Rendered by cluster_dev.sh via envsubst (which fills in / omits the resource
# directives per the selected cluster's .env.cluster), then sbatch'd on the cluster login node.
#
# Deliberately does NO heavy setup -- staging the SIF/caches is done on demand by
# node_exec.sh AFTER the node is up, so a staging bug can never waste the allocation.
#
# The ${CDEV_*_DIRECTIVE} lines expand to a full "#SBATCH ..." line or to nothing; an omitted
# directive (blank line) just lets SLURM use the queue default. Account/partition/gpus/cpus/
# mem/exclusive are all driven this way -- nothing here is cluster-specific.
#
${CDEV_ACCOUNT_DIRECTIVE}
${CDEV_PARTITION_DIRECTIVE}
#SBATCH -N 1
#SBATCH -n 1
${CDEV_GPUS_DIRECTIVE}
${CDEV_CPUS_DIRECTIVE}
#SBATCH --time=${CDEV_TIME}
${CDEV_EXCLUSIVE_DIRECTIVE}
${CDEV_MEM_DIRECTIVE}
#SBATCH --job-name=cluster-dev-box
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sturman@utexas.edu
#SBATCH --output=%x-%j.out

echo "cluster-dev sentinel started: job=$SLURM_JOB_ID node=$(hostname) at $(date -u +%FT%TZ)"
# Record identity cluster-side too (belt & suspenders; local watcher reads squeue directly).
mkdir -p "$HOME/.cluster_dev"
cat > "$HOME/.cluster_dev/current" <<EOF
JOBID=$SLURM_JOB_ID
NODE=$(hostname)
GPUS=${CDEV_GPUS_PER_NODE}
START_TS=$(date -u +%FT%TZ)
EOF
nvidia-smi -L >> "$HOME/.cluster_dev/current" 2>/dev/null || true

# Hold the allocation. Trap so scancel exits cleanly. Sleep just under the walltime.
trap 'echo "sentinel caught signal, exiting"; exit 0' TERM INT
sleep ${CDEV_TIME_SECONDS:-172800} &
wait $!
