#!/bin/bash
# Sentinel job: holds a compute node for the walltime so we can ssh in and develop interactively.
# Resource directives are injected from the cluster's submit_job_slurm.sh (same config as `job`);
# the sentinel adds its own name/output and does NO heavy setup (node_exec.sh stages on demand).
${SBATCH_DIRECTIVES}
#SBATCH --job-name=cluster-dev-box
#SBATCH --output=%x-%j.out

echo "cluster-dev sentinel started: job=$SLURM_JOB_ID node=$(hostname) at $(date -u +%FT%TZ)"
mkdir -p "$HOME/.cluster_dev"
cat > "$HOME/.cluster_dev/current" <<EOF
JOBID=$SLURM_JOB_ID
NODE=$(hostname)
START_TS=$(date -u +%FT%TZ)
EOF
nvidia-smi -L >> "$HOME/.cluster_dev/current" 2>/dev/null || true

# Hold the allocation until walltime (SLURM kills it then); trap so scancel exits cleanly.
trap 'echo "sentinel caught signal, exiting"; exit 0' TERM INT
sleep infinity & wait $!
