#!/usr/bin/env bash
# `develop exec` onto a job named by DEV_JOBID steps with the job's own partition, account and GPU request; a manager
# worktree (no profiles of its own) borrows the main checkout's profile for exec but refuses start/sync/stop. Stubs only.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
M="$T/main"
mkdir -p "$T/bin" "$M/scripts/cluster" "$T/remote/scripts/cluster/cluster_dev" "$T/.cluster_dev"
cp -r "$REPO/scripts/cluster/cluster_dev" "$REPO/scripts/cluster/tools" "$M/scripts/cluster/"
(cd "$M" && git init -q -b main . && git config user.email t@t && git config user.name t &&
    git add -A && git commit -qm init)
# the profile is per user and untracked, so a worktree of this checkout has none
mkdir -p "$M/scripts/cluster/config/zz"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$T/remote" \
    > "$M/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$M/scripts/cluster/config/zz/submit_job_slurm.sh"
git -C "$M" worktree add -q "$T/wt"
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
printf '#!/usr/bin/env bash\necho "$SQ_ROW"\n' > "$T/bin/squeue"
cat > "$T/bin/srun" <<EOF
#!/usr/bin/env bash
printf '%s ' "\$@" > "$T/srun_args"
while [ \$# -gt 0 ] && [ "\$1" != bash ]; do shift; done
exec "\$@"
EOF
printf '#!/usr/bin/env bash\nprintf "ARG<%%s>\\n" "$@"\n' > "$T/remote/scripts/cluster/cluster_dev/node_exec.sh"
chmod +x "$T/bin/"*
dev() {  # dev CHECKOUT ARGS... : cluster_dev.sh of that checkout, aimed at job 900
    local co="$1"; shift
    PATH="$T/bin:$PATH" HOME="$T" USER=me CLUSTER=zz DEV_JOBID=900 bash "$co/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"
}

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

SQ_ROW="RUNNING n1 amd-rtx IRI26004 gres/gpu:4" dev "$M" exec -- echo hi > "$T/out1" 2>&1
check "the step takes the job's partition" "grep -q -- '-p amd-rtx ' '$T/srun_args' && ! grep -q -- '-p test' '$T/srun_args'"
check "and its account" "grep -q -- '-A IRI26004 ' '$T/srun_args'"
check "and its GPU request" "grep -q -- '--gres=gpu:4 ' '$T/srun_args'"
check "the command reaches node_exec" "grep -q '^ARG<hi>' '$T/out1'"

SQ_ROW="RUNNING n1 skx (null) N/A" dev "$M" exec -- echo hi > "$T/out2" 2>&1
check "no account and no GPU request add none" "grep -q -- '-p skx ' '$T/srun_args' && ! grep -q -- ' -A ' '$T/srun_args' && ! grep -q -- '--gres' '$T/srun_args'"

SQ_ROW="RUNNING n1 amd-rtx IRI26004 N/A" dev "$T/wt" exec -- echo hi > "$T/out3" 2>&1
check "a worktree's exec uses the main checkout's profile" "grep -q '^ARG<hi>' '$T/out3' && grep -q \"main checkout's profile\" '$T/out3'"
for cmd in sync stop start kill; do
    dev "$T/wt" "$cmd" > "$T/out4" 2>&1
    check "a worktree refuses $cmd" "[ \$? -ne 0 ] && grep -q 'run it from the main checkout' '$T/out4'"
done

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -5 "$f"; done
    echo "--- srun args"; cat "$T/srun_args" 2>/dev/null
    exit 1
fi
echo "all checks passed"
