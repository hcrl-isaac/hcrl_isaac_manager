#!/usr/bin/env bash
# Create a cluster config (scripts/cluster/config/<name>/) from the templates. Invoked by
# `just cluster add`.
set -euo pipefail
cd "$(dirname "$0")/.."  # scripts/

read -p "Cluster Nickname (leave blank for default): " cluster_name
[ -z "$cluster_name" ] && cluster_name="default"
outdir="cluster/config/${cluster_name}"
if [ -d "$outdir" ]; then
    echo "[ERROR] Cluster config '$cluster_name' already exists. Delete it, edit it directly, or pick a different name." >&2
    exit 1
fi

read -p "Cluster Login (username@address): " cluster_login
read -p "Home Directory (\$HOME from cluster machine): " home
read -p "Scratch Directory (\$SCRATCH from cluster machine): " scratch
read -p "Email (for job notifications): " email
read -p "Queue Name: " queue
read -p "GPUs per Node: " num_procs
read -p "CPUs per Task/GPU: " num_cpus
case "$home" in /*) ;; *) home="/$home" ;; esac
case "$scratch" in /*) ;; *) scratch="/$scratch" ;; esac

mkdir -p "$outdir"
echo "[INFO] Writing cluster env file..."
HOME="$home" SCRATCH="$scratch" CLUSTER_LOGIN="$cluster_login" NUM_PROCS="$num_procs" NUM_CPUS="$num_cpus" \
    envsubst < cluster/tools/.env.cluster.template > "$outdir/.env.cluster"
echo "[INFO] Writing SLURM job config file..."
EMAIL="$email" QUEUE="$queue" NUM_PROCS="$num_procs" NUM_CPUS="$num_cpus" \
    envsubst < cluster/tools/submit_job_slurm.template.sh > "$outdir/submit_job_slurm.sh"
echo "[INFO] Created cluster config in $outdir."
