#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

module load tacc-apptainer/1.4.1
module load nvidia/26.1

# create job script with compute demands
cat <<EOT > job.sh
#!/bin/bash

#SBATCH -p amd-rtx
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --mem-per-cpu=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sturman@utexas.edu
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H_%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/scripts/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
