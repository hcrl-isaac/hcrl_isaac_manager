#!/usr/bin/env bash
# `develop start --no-sync` submits the sentinel without mirroring the local checkout; plain `start` still syncs.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
cleanup() {
    [ -f "$T/.cluster_dev/zz/watch.pid" ] && kill "$(cat "$T/.cluster_dev/zz/watch.pid")" 2>/dev/null
    rm -rf "$T"
}
trap cleanup EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote" "$T/local/resources/hcrl_isaaclab"
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
cd "$HOME" && exec bash -c "$*"  # remote commands run from the remote home
EOF
printf '#!/usr/bin/env bash\necho 4242\n' > "$T/bin/sbatch"
printf '#!/usr/bin/env bash\necho PENDING\n' > "$T/bin/squeue"
chmod +x "$T/bin/"*
echo code > "$T/local/resources/hcrl_isaaclab/a.py"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
dev() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$T/local" POLL_SECONDS=3600 \
    bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }
stop_watcher() { [ -f "$T/.cluster_dev/zz/watch.pid" ] && kill "$(cat "$T/.cluster_dev/zz/watch.pid")" 2>/dev/null; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

dev start --no-sync > "$T/out1" 2>&1
stop_watcher
check "--no-sync submits the sentinel" "grep -qx 'JOBID=4242' '$T/.cluster_dev/zz/state'"
check "--no-sync ships no code" "[ ! -e '$T/remote/resources/hcrl_isaaclab/a.py' ]"
check "--no-sync says so" "grep -q 'Skipping the code sync' '$T/out1'"

check "--no-sync warns that the remote node_exec.sh differs" "grep -q 'remote shared node_exec.sh differs' '$T/out1'"
dev start --nosync > "$T/out3" 2>&1
check "a misspelled flag is refused before anything runs" \
    "[ $? -ne 0 ] && grep -q \"unknown argument '--nosync'\" '$T/out3' && [ ! -e '$T/remote/resources/hcrl_isaaclab/a.py' ]"
check "nothing lands in the caller's directory" "[ ! -e '$REPO/.cluster_dev_sentinel.sbatch' ] && [ -e '$T/.cluster_dev_sentinel.sbatch' ]"

dev start > "$T/out2" 2>&1
stop_watcher
check "plain start still syncs" "[ -f '$T/remote/resources/hcrl_isaaclab/a.py' ]"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -10 "$f"; done
    exit 1
fi
echo "all checks passed"
