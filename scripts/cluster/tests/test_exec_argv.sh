#!/usr/bin/env bash
# `develop exec` keeps its argv intact through every hop (login shell, srun's bash -lc, the detached wrapper, and
# node_exec.sh's container bash -c), so quotes, ';' and '&&' never escape onto the bare node. Stubs only.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote/trees/default-0123456789/scripts/cluster/cluster_dev" "$T/.cluster_dev/zz" "$T/tmp"
touch "$T/remote/trees/default-0123456789/.complete"  # exec runs the newest `default` tree
cp -r "$REPO/scripts/cluster/cluster_dev" "$REPO/scripts/cluster/tools" "$T/scripts/cluster/"
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "-O" ] && exit 0; done
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-J|-i|-p|-l|-F|-E|-c|-m|-L|-R|-D|-W|-b|-e|-S|-w|-Q|-B) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
exec bash -c "$*"
EOF
printf '#!/usr/bin/env bash\necho "RUNNING n1"\n' > "$T/bin/squeue"
cat > "$T/bin/srun" <<'EOF'
#!/usr/bin/env bash
while [ $# -gt 0 ] && [ "$1" != bash ]; do shift; done
exec "$@"
EOF
# the tree's node_exec.sh records the argv it was handed
printf '#!/usr/bin/env bash\nprintf "ARG<%%s>\\n" "$@"\n' > "$T/remote/trees/default-0123456789/scripts/cluster/cluster_dev/node_exec.sh"
chmod +x "$T/bin/"*
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
dev() { PATH="$T/bin:$PATH" HOME="$T" USER=me CLUSTER=zz DEV_JOBID=900 bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}
want="$(printf 'ARG<%s>\n' bash -lc "ls -d /isaac-sim; echo 'q \"x\"' && echo \$HOME" 'a b' '*')"

dev exec -- bash -lc "ls -d /isaac-sim; echo 'q \"x\"' && echo \$HOME" 'a b' '*' > "$T/out1" 2>&1
check "srun mode delivers the argv unchanged" "[ \"\$(grep '^ARG<' '$T/out1')\" = \"\$want\" ]"
check "nothing ran outside node_exec" "! grep -v -e '^ARG<' -e '^\[cluster_dev\]' '$T/out1' | grep -q ."

dev exec --detach --log "$T/run.log" -- bash -lc "ls -d /isaac-sim; echo 'q \"x\"' && echo \$HOME" 'a b' '*' > "$T/out2" 2>&1
for _ in $(seq 50); do [ -s "$T/run.log" ] && break; sleep 0.1; done
sleep 0.3
check "detached mode delivers the argv unchanged" "[ \"\$(grep '^ARG<' '$T/run.log')\" = \"\$want\" ]"

# node_exec.sh itself: the container's bash -c string must re-parse to the same argv
mkdir -p "$T/n/scripts/cluster/cluster_dev" "$T/n/shared/resources/hcrl_isaaclab" "$T/n/sif" "$T/n/cache/docker-isaac-sim"
cp "$REPO/scripts/cluster/cluster_dev/node_exec.sh" "$T/n/scripts/cluster/cluster_dev/"
printf '#!/usr/bin/env bash\nprintf "%%s" "${@: -1}" > "%s/n/cmd"\n' "$T" > "$T/bin/apptainer"
chmod +x "$T/bin/apptainer"
touch "$T/n/sif/hcrl-isaac.sif"
printf 'CLUSTER_SIF_PATH=%s\nCLUSTER_ISAAC_SIM_CACHE_DIR=%s\nCLUSTER_ISAACLAB_DIR=%s\n' \
    "$T/n/sif" "$T/n/cache/docker-isaac-sim" "$T/n/shared" > "$T/n/env"
node() { PATH="$T/bin:$PATH" TMPDIR="$T/tmp" SLURM_JOB_ID=7 NODE_EXEC_ENV="$T/n/env" bash "$T/n/scripts/cluster/cluster_dev/node_exec.sh" "$@"; }
reparse() { local s; s="$(cat "$1")"; bash -c "set -- ${s##*hcrl-entrypoint }; printf 'ARG<%s>\n' \"\$@\""; }

node bash -lc "ls; echo 'q'" 'a b' > "$T/n/out" 2>&1
check "node_exec passes several arguments as an argv" \
    "[ \"\$(reparse '$T/n/cmd')\" = \"\$(printf 'ARG<%s>\n' bash -lc \"ls; echo 'q'\" 'a b')\" ]"
node "python scripts/train.py --task T1-CubeLift-v0" > /dev/null 2>&1
check "node_exec runs one argument as a command string under bash -c" \
    "[ \"\$(reparse '$T/n/cmd')\" = \"\$(printf 'ARG<%s>\n' bash -c 'python scripts/train.py --task T1-CubeLift-v0')\" ]"
node "cd x && python a.py; python b.py | tee l" > /dev/null 2>&1
check "a compound command string stays whole inside the entrypoint" \
    "[ \"\$(reparse '$T/n/cmd')\" = \"\$(printf 'ARG<%s>\n' bash -c 'cd x && python a.py; python b.py | tee l')\" ]"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out* "$T/run.log" "$T/n/out"; do echo "--- $f"; cat "$f" 2>/dev/null | tail -12; done
    exit 1
fi
echo "all checks passed"
