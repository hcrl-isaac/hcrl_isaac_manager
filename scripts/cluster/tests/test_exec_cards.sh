#!/usr/bin/env bash
# `develop exec --cards N` asks srun for N cards' CPUs, but never more than the sentinel job holds: a whole-node
# sentinel (one task with every CPU) gets one step with all of them. Stubs only.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote/trees/default-0123456789/scripts/cluster/cluster_dev" "$T/.cluster_dev/zz" "$T/tmp"
touch "$T/remote/trees/default-0123456789/.complete"
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
# the job's CPU count comes from $JOB_CPUS ("" = squeue prints nothing for it); every other query sees it running
cat > "$T/bin/squeue" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "%C" ] && { [ -n "${JOB_CPUS:-}" ] && echo "  ${JOB_CPUS}"; exit 0; }; done
echo "RUNNING n1"
EOF
cat > "$T/bin/srun" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in --cpus-per-task=*) echo "STEP ${a#--cpus-per-task=}" ;; esac; done
EOF
printf '#!/usr/bin/env bash\n' > "$T/remote/trees/default-0123456789/scripts/cluster/cluster_dev/node_exec.sh"
chmod +x "$T/bin/"*
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n#SBATCH --cpus-per-task=128\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
step() {  # step <job cpus> <cards>: the --cpus-per-task the step asked for
    JOB_CPUS="$1" PATH="$T/bin:$PATH" HOME="$T" USER=me CLUSTER=zz DEV_JOBID=900 \
        bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" exec --cards "$2" -- true 2>/dev/null | sed -n 's/^STEP //p'
}

fails=0
check() {
    if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1 (got '$2', want '$3')"; fails=$((fails + 1)); fi
}
check "a whole-node sentinel's 8-card step asks for the job's 128 CPUs, not 1024" "$(step 128 8)" 128
check "a job holding every card's CPUs gets them all" "$(step 1024 8)" 1024
check "a step never asks for more than its cards' share" "$(step 2048 8)" 1024
check "when squeue gives no CPU count, the cards' share is asked for" "$(step "" 8)" 1024
check "a one-card step keeps the sentinel's own share" "$(step 128 1)" 128

if [ "$fails" -ne 0 ]; then exit 1; fi
echo "all checks passed"
