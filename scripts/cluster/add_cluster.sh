#!/usr/bin/env bash
# Create or regenerate a per-user cluster profile (scripts/cluster/config/<name>/, gitignored) from the
# templates. Invoked by `just cluster add [--update] [name]`.
set -euo pipefail
cd "$(dirname "$0")/.."  # scripts/
source cluster/tools/restore_profiles.sh

update=""
[ "${1:-}" = "--update" ] && { update=1; shift; }
cluster_name="${1:-}"
[ -z "$cluster_name" ] && read -r -p "Cluster Nickname (leave blank for default): " cluster_name
[ -z "$cluster_name" ] && cluster_name="default"
outdir="cluster/config/${cluster_name}"
env_file="$outdir/.env.cluster"
job_file="$outdir/submit_job_slurm.sh"

if [ -d "$outdir" ] && [ -z "$update" ]; then
    read -r -p "Profile '$cluster_name' exists. Regenerate it with its current values as defaults? [y/N] " yn
    case "$yn" in [yY]*) update=1 ;; *) echo "[INFO] Left $outdir unchanged."; exit 0 ;; esac
fi

# Current values of an existing profile become the prompt defaults.
env_get() { [ -f "$env_file" ] && sed -n "s/^$1=//p" "$env_file" | tail -1 | tr -d '"' || true; }
sbatch_get() { python3 cluster/tools/merge_profile.py get "$job_file" "$1"; }  # any spelling of the flag
ask() {  # ask VAR "Prompt" DEFAULT
    local reply
    read -r -p "$2${3:+ [$3]}: " reply
    printf -v "$1" '%s' "${reply:-$3}"
}

cur_dir="$(env_get CLUSTER_ISAACLAB_DIR)"
ask cluster_login "Cluster Login (username@address)" "$(env_get CLUSTER_LOGIN)"
ask workspace "Workspace Directory (large quota, e.g. \$WORK on TACC; job code copies and logs)" \
    "${cur_dir:+$(dirname "$cur_dir")}"
ask scratch "Scratch Directory (\$SCRATCH from cluster machine; .sif and Isaac Sim cache)" "$(env_get CLUSTER_SIF_PATH)"
ask email "Email (for job notifications)" "$(sbatch_get --mail-user)"
ask queue "Queue Name" "$(sbatch_get -p)"
ask account "Allocation/Account (#SBATCH -A, leave blank if not needed)" "$(sbatch_get -A)"
ask num_procs "GPUs per Node" "$(sbatch_get -n)"
ask num_cpus "CPUs per Task/GPU" "$(sbatch_get --cpus-per-task)"
cur_time="$(sbatch_get --time)"
ask walltime "Walltime (HH:MM:SS)" "${cur_time:-24:00:00}"
for required in cluster_login workspace scratch; do
    [ -n "${!required}" ] || { echo "[ERROR] $required is required; nothing written." >&2; exit 1; }
done
case "$workspace" in /* | \$*) ;; *) workspace="/$workspace" ;; esac  # keep ${VAR}/... as written
case "$scratch" in /* | \$*) ;; *) scratch="/$scratch" ;; esac
account_line=""
[ -n "$account" ] && account_line="#SBATCH -A $account"
# TACC ships apptainer only as an Lmod module on compute nodes
module_load=""
case "$cluster_login" in *.tacc.utexas.edu) module_load="tacc-apptainer" ;; esac
module_loads=""
[ -n "$module_load" ] && module_loads="module load $module_load"

backup=""
if [ -n "$update" ] && [ -d "$outdir" ]; then
    backup="$outdir/.backup"
    rm -rf "$backup" && mkdir -p "$backup"
    cp -a "$outdir"/.env.cluster "$outdir"/submit_job_slurm.sh "$backup"/ 2>/dev/null || true
fi
mkdir -p "$outdir"
echo "[INFO] Writing cluster env file..."
WORKSPACE="$workspace" SCRATCH="$scratch" CLUSTER_LOGIN="$cluster_login" NUM_CPUS="$num_cpus" \
    CLUSTER_MODULE_LOAD="$module_load" \
    envsubst '$WORKSPACE $SCRATCH $CLUSTER_LOGIN $NUM_CPUS $CLUSTER_MODULE_LOAD' \
    < cluster/tools/.env.cluster.template > "$env_file"
echo "[INFO] Writing SLURM job config file..."
EMAIL="$email" QUEUE="$queue" NUM_PROCS="$num_procs" NUM_CPUS="$num_cpus" WALLTIME="$walltime" \
    ACCOUNT_LINE="$account_line" MODULE_LOADS="$module_loads" \
    envsubst '$EMAIL $QUEUE $NUM_PROCS $NUM_CPUS $WALLTIME $ACCOUNT_LINE $MODULE_LOADS' \
    < cluster/tools/submit_job_slurm.template.sh > "$job_file"

if [ -n "$backup" ] && [ -f "$backup/.env.cluster" ]; then
    # keep hand-set values and lines the template doesn't emit
    python3 cluster/tools/merge_profile.py "$backup" "$outdir"
    echo "[INFO] Previous files saved in $backup. Changes:"
    diff -u "$backup/.env.cluster" "$env_file" || true
    diff -u "$backup/submit_job_slurm.sh" "$job_file" || true
fi
restore_profiles  # snapshot the profile just written, so .backup/ never lags it
echo "[INFO] Wrote cluster profile $outdir (per-user, not tracked by git)."
