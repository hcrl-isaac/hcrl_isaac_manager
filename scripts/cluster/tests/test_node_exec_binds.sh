#!/usr/bin/env bash
# node_exec.sh in tree mode, against a stub apptainer that prints its arguments: staged repos are mounted writable
# with writable logs/outputs/wandb, linked shared repos stay read-write, and the shared mode is unchanged.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/cluster_dev" "$T/shared/resources/hcrl_isaaclab" "$T/shared/resources/hcrl_robots" \
    "$T/sif" "$T/cache/docker-isaac-sim" "$T/tmp"
cp "$REPO/scripts/cluster/cluster_dev/node_exec.sh" "$T/scripts/cluster/cluster_dev/"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$@"\nls "%s/shared/trees/t-0123456789/.in-use" 2>/dev/null | sed "s/^/marker /"\necho "envkey $APPTAINERENV_WANDB_API_KEY"\n' "$T" > "$T/bin/apptainer"
chmod +x "$T/bin/apptainer"
touch "$T/sif/hcrl-isaac.sif"
cat > "$T/env" <<EOF
CLUSTER_SIF_PATH=$T/sif
CLUSTER_ISAAC_SIM_CACHE_DIR=$T/cache/docker-isaac-sim
CLUSTER_ISAACLAB_DIR=$T/shared
EOF
TREE="$T/shared/trees/t-0123456789"
mkdir -p "$TREE/resources/hcrl_isaaclab/logs" "$TREE/resources/hcrl_isaaclab/outputs" "$TREE/resources/hcrl_isaaclab/wandb"
ln -s "$T/shared/resources/hcrl_robots" "$TREE/resources/hcrl_robots"
run() { PATH="$T/bin:$PATH" TMPDIR="$T/tmp" SLURM_JOB_ID=7 SLURM_STEP_ID=3 WANDB_API_KEY=sekret-key NODE_EXEC_ENV="$T/env" bash "$T/scripts/cluster/cluster_dev/node_exec.sh" true; }

fails=0
check() {
    if grep -qxF -- "$2" "$T/out"; then echo "PASS $1"; else echo "FAIL $1 (missing: $2)"; fails=$((fails + 1)); fi
}

NODE_EXEC_RESOURCES="$TREE/resources" run > "$T/out" 2>&1
check "staged repo is writable (the artifact resolver re-links inside it)" "$TREE/resources/hcrl_isaaclab:/workspace/ext/hcrl_isaaclab:rw"
check "its logs go to the shared logs dir" "$T/shared/resources/hcrl_isaaclab/logs:/workspace/ext/hcrl_isaaclab/logs:rw"
check "its outputs are writable node-locally" "$T/tmp/cluster_dev_7/tree-rw/t-0123456789/hcrl_isaaclab/outputs:/workspace/ext/hcrl_isaaclab/outputs:rw"
check "its wandb dir is writable node-locally" "$T/tmp/cluster_dev_7/tree-rw/t-0123456789/hcrl_isaaclab/wandb:/workspace/ext/hcrl_isaaclab/wandb:rw"
check "the W&B key reaches the container through its environment" "envkey sekret-key"
if grep -v '^envkey ' "$T/out" | grep -q sekret-key; then
    echo "FAIL the W&B key is not on apptainer's command line"; fails=$((fails + 1))
else
    echo "PASS the W&B key is not on apptainer's command line"
fi
check "a linked shared repo stays read-write" "$T/shared/resources/hcrl_robots:/workspace/ext/hcrl_robots:rw"
check "the run holds an in-use marker for its step" "marker 7.3"
check "the shared artifact root is bound writable" \
    "$T/shared/resources/hcrl_isaaclab/.artifacts:/workspace/ext/hcrl_isaaclab/.artifacts:rw"
if [ -z "$(ls -A "$TREE/.in-use")" ]; then echo "PASS the marker is removed on exit"; else echo "FAIL the marker is removed on exit"; fails=$((fails + 1)); fi

PATH="$T/bin:$PATH" TMPDIR="$T/tmp" SLURM_JOB_ID=7 NODE_EXEC_ENV="$T/env" NODE_EXEC_RESOURCES="$TREE/resources" \
    bash "$T/scripts/cluster/cluster_dev/node_exec.sh" true > "$T/out" 2>&1
check "outside a step the marker names the job" "marker 7.nostep"
printf '#!/usr/bin/env bash\nsleep 30 & wait\n' > "$T/bin/apptainer"
PATH="$T/bin:$PATH" TMPDIR="$T/tmp" SLURM_JOB_ID=7 SLURM_STEP_ID=4 NODE_EXEC_ENV="$T/env" NODE_EXEC_RESOURCES="$TREE/resources" \
    bash "$T/scripts/cluster/cluster_dev/node_exec.sh" true > /dev/null 2>&1 &
pid=$!
for _ in $(seq 50); do [ -e "$TREE/.in-use/7.4" ] && break; sleep 0.1; done
kill -TERM "$pid"
wait "$pid" 2>/dev/null
if [ ! -e "$TREE/.in-use/7.4" ]; then echo "PASS SIGTERM removes the marker"; else echo "FAIL SIGTERM removes the marker"; fails=$((fails + 1)); fi
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$@"\n' > "$T/bin/apptainer"

run > "$T/out" 2>&1
check "shared mode binds the shared repo read-write" "$T/shared/resources/hcrl_isaaclab:/workspace/ext/hcrl_isaaclab:rw"
if grep -q ":ro$" "$T/out"; then echo "FAIL shared mode has no read-only binds"; fails=$((fails + 1)); else echo "PASS shared mode has no read-only binds"; fi

if [ "$fails" -ne 0 ]; then
    echo "--- apptainer args"; cat "$T/out"
    exit 1
fi
echo "all checks passed"
