#!/usr/bin/env bash
# `list`, `logs` and `stop` (pls cluster <name> list|logs|stop) on a SLURM profile. Stubs only: ssh runs here, and
# squeue/scontrol/sacct/scancel record their arguments or answer from fixtures.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts" "$T/remote/jobdir"
cp -r "$REPO/scripts/cluster" "$T/scripts/"
rm -rf "$T/scripts/cluster/config" && mkdir -p "$T/scripts/cluster/config/zz"
printf 'CLUSTER_LOGIN=fake@host\nCLUSTER_ISAACLAB_DIR=%s\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p gpu-a100\n#SBATCH -o logs/%%x-%%j.out\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
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
printf '%s ' "\$@" > "$T/squeue_args"
EOF
# job 7 is still known to scontrol; job 8 has left the queue and is found through sacct's WorkDir and the -o pattern
cat > "$T/bin/scontrol" <<EOF
#!/usr/bin/env bash
[ "\$3" = 7 ] && echo "   StdOut=$T/remote/live.out"
EOF
cat > "$T/bin/sacct" <<EOF
#!/usr/bin/env bash
echo "$T/remote/jobdir"
EOF
cat > "$T/bin/scancel" <<EOF
#!/usr/bin/env bash
printf '%s ' "\$@" > "$T/scancel_args"
EOF
chmod +x "$T/bin/"*
mkdir -p "$T/remote/jobdir/logs"
seq 1 300 > "$T/remote/live.out"
echo "finished job output" > "$T/remote/jobdir/logs/train-8.out"
ci() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz bash "$T/scripts/cluster/cluster_interface.sh" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

ci list > "$T/out1" 2>&1; rc=$?
check "list shows our jobs on the profile's partition" "[ $rc = 0 ] && grep -q -- '--me -p gpu-a100' '$T/squeue_args'"

ci logs 7 > "$T/out2" 2>&1; rc=$?
check "logs tails a running job's StdOut (last 100 lines)" "[ $rc = 0 ] && grep -qx 300 '$T/out2' && ! grep -qx 200 '$T/out2'"
ci logs 7 -n 2 > "$T/out3" 2>&1
check "logs passes tail's arguments" "[ \"\$(grep -cx '[0-9]*' '$T/out3')\" = 2 ]"
ci logs 8 > "$T/out4" 2>&1; rc=$?
check "a finished job's log comes from its work dir and the -o pattern" "[ $rc = 0 ] && grep -q 'finished job output' '$T/out4'"

ci stop 7 8 > "$T/out5" 2>&1; rc=$?
check "stop cancels the named jobs" "[ $rc = 0 ] && grep -q '7 8' '$T/scancel_args'"
rm -f "$T/scancel_args"
ci stop '7; rm -rf /' > "$T/out6" 2>&1; rc=$?
check "stop refuses anything but job ids" "[ $rc != 0 ] && [ ! -e '$T/scancel_args' ] && grep -q 'expected a job id' '$T/out6'"
ci logs > "$T/out7" 2>&1; rc=$?
check "logs needs a job id" "[ $rc != 0 ]"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -6 "$f"; done
    exit 1
fi
echo "all checks passed"
