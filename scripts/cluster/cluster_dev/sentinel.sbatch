#!/bin/bash
# Sentinel job: holds a Delta gpuA40x4 node for the full walltime so we can ssh into it
# and develop interactively. Rendered by cluster_dev.sh via envsubst, then sbatch'd on Delta.
#
# Deliberately does NO heavy setup — staging the SIF/caches is done on demand by
# node_exec.sh AFTER the node is up, so a staging bug can never waste the 48h allocation.
#
#SBATCH -A ${DELTA_ACCOUNT}
#SBATCH -p ${DELTA_PARTITION}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=${DELTA_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${DELTA_CPUS}
#SBATCH --time=${DELTA_TIME}
#SBATCH --exclusive
#SBATCH --mem=${DELTA_MEM}
#SBATCH --job-name=delta-dev-box
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sturman@utexas.edu
#SBATCH --output=%x-%j.out

echo "delta-dev sentinel started: job=$SLURM_JOB_ID node=$(hostname) at $(date -u +%FT%TZ)"
# Record identity Delta-side too (belt & suspenders; local watcher reads squeue directly).
mkdir -p "$HOME/.cluster_dev"
cat > "$HOME/.cluster_dev/current" <<EOF
JOBID=$SLURM_JOB_ID
NODE=$(hostname)
GPUS=${DELTA_GPUS_PER_NODE}
START_TS=$(date -u +%FT%TZ)
EOF
nvidia-smi -L >> "$HOME/.cluster_dev/current" 2>/dev/null || true

# Hold the allocation. Trap so scancel exits cleanly. Sleep just under the walltime.
trap 'echo "sentinel caught signal, exiting"; exit 0' TERM INT
sleep ${DELTA_TIME_SECONDS:-172800} &
wait $!
