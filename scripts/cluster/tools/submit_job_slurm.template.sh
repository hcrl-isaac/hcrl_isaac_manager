#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

$MODULE_LOADS

# create job script with compute demands
cat <<EOT > job.sh
#!/bin/bash

#SBATCH -p $QUEUE
$ACCOUNT_LINE
#SBATCH -N 1
#SBATCH -n $NUM_PROCS
#SBATCH --cpus-per-task=$NUM_CPUS
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H_%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/scripts/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
