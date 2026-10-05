#!/usr/bin/env bash
# Install the Vulkan clamp layer (scripts/vulkan/) for a cluster profile into ${CLUSTER_SIF_PATH}/vkclamp, which
# node_exec.sh and run_singularity.sh bind into the container when present. Needs the profile's SSH master.
#
# Usage: scripts/cluster/tools/install_vkclamp.sh <cluster>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cluster="${1:?usage: install_vkclamp.sh <cluster>}"
env_file="$HERE/../config/$cluster/.env.cluster"
[ -f "$env_file" ] || { echo "[vkclamp] no profile $env_file"; exit 1; }
source "$env_file"
dest="${CLUSTER_VKCLAMP_DIR:-${CLUSTER_SIF_PATH}/vkclamp}"
ssh_opts=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm/%C" -o ControlPersist=48h -o ConnectTimeout=60)
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
bash "$HERE/../../vulkan/install_clamp_layer.sh" "$build" /opt/vkclamp >/dev/null
ssh "${ssh_opts[@]}" "$CLUSTER_LOGIN" "mkdir -p '$dest'"
rsync -rq -e "ssh ${ssh_opts[*]}" "$build/" "$CLUSTER_LOGIN:$dest/"
echo "[vkclamp] installed at $CLUSTER_LOGIN:$dest (bound at /opt/vkclamp by the next node_exec/run_singularity run)"
