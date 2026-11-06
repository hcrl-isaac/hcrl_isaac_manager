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

# Detached steps start from here, inside the job, so no login-node process has to outlive the session that
# launched them: `exec --detach` drops <id>.req (the step's command) and its log path <id>.log into the spool.
SPOOL="$HOME/.cluster_dev/spool/$SLURM_JOB_ID"
mkdir -p "$SPOOL" && touch "$SPOOL/.ready"
# the job's end takes its steps with it, so its spool goes too (step logs live at their own paths)
trap 'echo "sentinel caught signal, exiting"; rm -rf "$SPOOL"; exit 0' TERM INT
# Hold the allocation until walltime (SLURM kills it then), starting each spooled step as it arrives.
while true; do
    for req in "$SPOOL"/*.req; do
        [ -e "$req" ] || continue
        id="$(basename "$req" .req)"
        mv "$req" "$SPOOL/$id.started" || continue
        log="$(cat "$SPOOL/$id.log" 2>/dev/null)"
        echo "$(date -u +%FT%TZ) starting step $id (log ${log:-none})"
        setsid bash "$SPOOL/$id.started" > "${log:-/dev/null}" 2>&1 < /dev/null &
    done
    sleep 5 & wait $!
done
