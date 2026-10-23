#!/usr/bin/env bash
# `develop stop` cancels only the named sentinel (argument, DEV_JOBID, or this user's only one) and never closes the
# SSH master other sessions share; `develop close` does. Runs through ssh/squeue/scancel stubs.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote" "$T/.cluster_dev/zz"
cp -r "$REPO/scripts/cluster/cluster_dev" "$REPO/scripts/cluster/tools" "$T/scripts/cluster/"
cat > "$T/bin/ssh" <<EOF
#!/usr/bin/env bash
for a in "\$@"; do
    case "\$a" in -O) for b in "\$@"; do [ "\$b" = exit ] && echo closed >> "$T/master"; done; exit 0 ;; esac
done
while [ \$# -gt 0 ]; do
    case "\$1" in
        -o|-J|-i|-p|-l|-F|-E|-c|-m|-L|-R|-D|-W|-b|-e|-S|-w|-Q|-B) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
exec bash -c "\$*"
EOF
printf '#!/usr/bin/env bash\ncat "%s/live" 2>/dev/null\n' "$T" > "$T/bin/squeue"
printf '#!/usr/bin/env bash\necho "$*" >> "%s/cancelled"\n' "$T" > "$T/bin/scancel"
chmod +x "$T/bin/"*
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
printf 'JOBID=901\nJOB_STATE=RUNNING\nNODE=a\n' > "$T/.cluster_dev/zz/state"
dev() { PATH="$T/bin:$PATH" HOME="$T" USER=me CLUSTER=zz bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

printf '901 RUNNING a\n902 RUNNING b\n' > "$T/live"
dev stop > "$T/out1" 2>&1
check "with several sentinels, an unnamed stop is refused and lists them" \
    "[ $? -ne 0 ] && grep -q '902 RUNNING b' '$T/out1' && [ ! -s '$T/cancelled' ]"

dev stop 902 > "$T/out2" 2>&1
check "stop JOBID cancels only that sentinel" "grep -qx '902' '$T/cancelled'"
check "another session's job leaves the tracked state alone" "grep -qx 'JOB_STATE=RUNNING' '$T/.cluster_dev/zz/state'"
check "stop never closes the shared master" "[ ! -s '$T/master' ]"

: > "$T/cancelled"
DEV_JOBID=901 dev stop > "$T/out3" 2>&1
check "DEV_JOBID names the sentinel" "grep -qx '901' '$T/cancelled'"
check "the tracked job's state is marked stopped" "grep -qx 'JOB_STATE=STOPPED' '$T/.cluster_dev/zz/state'"

: > "$T/cancelled"
dev stop 12345 > "$T/out4" 2>&1
check "a job that is not one of this user's sentinels is refused" "[ $? -ne 0 ] && [ ! -s '$T/cancelled' ]"

printf '903 RUNNING c\n' > "$T/live"
dev stop > "$T/out5" 2>&1
check "the only sentinel is stopped when none is named" "grep -qx '903' '$T/cancelled'"

dev close > "$T/out6" 2>&1
check "close closes the SSH master" "grep -q closed '$T/master'"

[ "$fails" -eq 0 ] && echo "all develop stop tests passed"
exit "$fails"
