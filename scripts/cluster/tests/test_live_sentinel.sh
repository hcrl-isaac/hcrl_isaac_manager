#!/usr/bin/env bash
# `develop exec`/`status` when the state file tracks a job that has ended: the user's only RUNNING sentinel is
# used, several are listed for DEV_JOBID, and none keeps the original error. Runs through ssh/squeue stubs.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote" "$T/.cluster_dev/zz"
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
cat > "$T/bin/squeue" <<EOF
#!/usr/bin/env bash
case " \$* " in
    *" -n cluster-dev-box "*) cat "$T/live" 2>/dev/null ;;
    *" -j "*) for a in "\$@"; do case "\$a" in 9*) grep "^\$a " "$T/live" | awk '{print "RUNNING", \$2}' ;; esac; done ;;
esac
exit 0
EOF
printf '#!/usr/bin/env bash\necho TIMEOUT\n' > "$T/bin/sacct"
printf '#!/usr/bin/env bash\necho "SRUN $*"\n' > "$T/bin/srun"
chmod +x "$T/bin/"*
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
printf 'JOBID=111\nJOB_STATE=RUNNING\nNODE=old\n' > "$T/.cluster_dev/zz/state"
dev() { PATH="$T/bin:$PATH" HOME="$T" USER=me CLUSTER=zz bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

echo "900 gpub065 12:00:00" > "$T/live"
dev exec -- echo hi > "$T/out1" 2>&1
check "an ended tracked job falls back to the only live sentinel" "grep -q 'using the only RUNNING sentinel 900' '$T/out1' && grep -q 'SRUN.*--jobid[= ]900' '$T/out1'"
check "the ended job's state is recorded" "grep -qx 'JOB_STATE=TIMEOUT' '$T/.cluster_dev/zz/state'"

printf '901 gpub001 1:00:00\n900 gpub065 12:00:00\n' > "$T/live"
dev exec -- echo hi > "$T/out2" 2>&1
check "several live sentinels are refused and listed" "[ $? -ne 0 ] && grep -q 'DEV_JOBID' '$T/out2' && grep -q '901 gpub001' '$T/out2' && ! grep -q SRUN '$T/out2'"
DEV_JOBID=901 dev exec -- echo hi > "$T/out3" 2>&1
check "DEV_JOBID picks one of them" "grep -q 'SRUN.*--jobid[= ]901' '$T/out3'"

: > "$T/live"
dev exec -- echo hi > "$T/out4" 2>&1
check "no live sentinel keeps the original error" "[ $? -ne 0 ] && grep -q 'No running job yet' '$T/out4'"

echo "900 gpub065 12:00:00" > "$T/live"
dev status > "$T/out5" 2>&1
check "status lists the live sentinels" "grep -q 'RUNNING sentinels' '$T/out5' && grep -q '900 gpub065' '$T/out5'"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -8 "$f"; done
    exit 1
fi
echo "all checks passed"
