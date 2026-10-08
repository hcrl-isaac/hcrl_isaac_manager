#!/usr/bin/env bash
# Runs the real `cluster_dev.sh sync` through an ssh stub against throwaway local/remote trees and checks
# what --delete removed and what it protected. Needs only bash and rsync.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

# The script under test reads config/ next to its own dir, so it runs from a copy inside $T.
mkdir -p "$T/bin" "$T/scripts/cluster/config"
cp -r "$REPO/scripts/cluster/cluster_dev" "$REPO/scripts/cluster/tools" "$T/scripts/cluster/"

cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# answers `-O check`; otherwise runs the remote command locally
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
chmod +x "$T/bin/ssh"

L="$T/local"
R="$T/remote"
mkdir -p "$L/resources/hcrl_isaaclab/worktrees/wtlocal" "$L/resources/motion_datasets" "$L/scripts/cluster"
echo a > "$L/resources/hcrl_isaaclab/a.py"
echo wl > "$L/resources/hcrl_isaaclab/worktrees/wtlocal/x.py"
echo b > "$L/resources/motion_datasets/bundle_local.pt"
echo n > "$L/resources/motion_datasets/notes.txt"
echo LOCAL_SLOT > "$L/scripts/cluster/.env.cluster"
echo WANDB_API_KEY=k > "$L/scripts/.env.wandb"
chmod 664 "$L/scripts/.env.wandb"

mkdir -p "$R/resources/hcrl_isaaclab/worktrees/wtremote" "$R/resources/hcrl_isaaclab/protected_dir" \
    "$R/resources/motion_datasets" "$R/resources/fork_pbfm" "$R/scripts/cluster" "$R/artifacts"
echo old > "$R/resources/hcrl_isaaclab/a.py"
echo stale > "$R/resources/hcrl_isaaclab/stale.py"
echo wr > "$R/resources/hcrl_isaaclab/worktrees/wtremote/y.py"
echo p > "$R/resources/hcrl_isaaclab/protected_dir/keep.txt"
echo ro > "$R/resources/motion_datasets/bundle_remote_only.pt"
echo f > "$R/resources/fork_pbfm/z.py"
echo art > "$R/artifacts/keep.txt"
mkdir -p "$R/resources/hcrl_isaaclab/outputs/run1" "$R/resources/hcrl_isaaclab/.claude" "$R/resources/retired/pkg" \
    "$L/resources/retired/worktrees/wt"
echo o > "$R/resources/hcrl_isaaclab/outputs/run1/hydra.log"
echo doc > "$R/resources/hcrl_isaaclab/.claude/infra.md"
echo r > "$R/resources/retired/pkg/mod.py"
echo REMOTE_SLOT > "$R/scripts/cluster/.env.cluster"

# zz-sib shares the destination (written through a variable) and is the only one protecting protected_dir.
CFG="$T/scripts/cluster/config"
mkdir -p "$CFG/zz" "$CFG/zz-sib"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$R" > "$CFG/zz/.env.cluster"
printf 'BASE=%s\nCLUSTER_ISAACLAB_DIR="${BASE}"\nCLUSTER_LOGIN=fake@host\n' "$R" > "$CFG/zz-sib/.env.cluster"
echo "resources/hcrl_isaaclab/protected_dir/" > "$CFG/zz-sib/.rsync-exclude"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n#SBATCH --time=01:00:00\n' > "$CFG/zz/submit_job_slurm.sh"

run_sync() {
    PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$L" \
        bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" sync "$@"
}

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

run_sync --dry-run > "$T/dry.log" 2>&1
check "dry run exits 0" "[ $? -eq 0 ]"
check "dry run lists the stale file" "grep -q '^\*deleting *resources/hcrl_isaaclab/stale.py' '$T/dry.log'"
check "dry run deletes nothing" "[ -e '$R/resources/hcrl_isaaclab/stale.py' ]"
check "dry run pushes no env" "[ ! -e '$R/scripts/cluster/config/zz/.env.cluster' ]"

run_sync > "$T/sync.log" 2>&1
check "sync exits 0" "[ $? -eq 0 ]"
check "stale file deleted" "[ ! -e '$R/resources/hcrl_isaaclab/stale.py' ]"
check "changed file updated" "grep -qx a '$R/resources/hcrl_isaaclab/a.py'"
check "remote-only worktree kept" "[ -e '$R/resources/hcrl_isaaclab/worktrees/wtremote/y.py' ]"
check "local worktree not shipped" "[ ! -e '$R/resources/hcrl_isaaclab/worktrees/wtlocal' ]"
check "remote-only motion bundle kept" "[ -e '$R/resources/motion_datasets/bundle_remote_only.pt' ]"
check "allowlisted bundle shipped" "[ -e '$R/resources/motion_datasets/bundle_local.pt' ]"
check "non-allowlisted file not shipped" "[ ! -e '$R/resources/motion_datasets/notes.txt' ]"
check "remote-only repo kept" "[ -e '$R/resources/fork_pbfm/z.py' ]"
check "remote artifacts kept" "[ -e '$R/artifacts/keep.txt' ]"
check "remote hydra outputs kept" "[ -e '$R/resources/hcrl_isaaclab/outputs/run1/hydra.log' ]"
check "remote repo .claude kept" "[ -e '$R/resources/hcrl_isaaclab/.claude/infra.md' ]"
check "repo retired locally (worktrees only) kept" "[ -e '$R/resources/retired/pkg/mod.py' ]"
check "sibling-config protection honored" "[ -e '$R/resources/hcrl_isaaclab/protected_dir/keep.txt' ]"
check "remote shared slot untouched" "grep -q REMOTE_SLOT '$R/scripts/cluster/.env.cluster'"
check "config env pushed to its config dir" "grep -q CLUSTER_LOGIN=fake@host '$R/scripts/cluster/config/zz/.env.cluster'"
check "local slot not rewritten" "grep -q LOCAL_SLOT '$L/scripts/cluster/.env.cluster'"
check "the W&B key lands owner-only" "[ \"\$(stat -c %a '$R/scripts/.env.wandb')\" = 600 ]"

# a fresh profile whose workspace and its parent do not exist on the remote yet
FRESH="$T/fresh/parent/isaaclab"
mkdir -p "$CFG/zz-new"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$FRESH" \
    > "$CFG/zz-new/.env.cluster"
cp "$CFG/zz/submit_job_slurm.sh" "$CFG/zz-new/"
sync_new() {
    PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz-new LOCAL_ISAACLAB_DIR="$L" \
        bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" sync "$@"
}
sync_new --dry-run > "$T/fresh_dry.log" 2>&1
check "a dry run leaves a missing workspace missing" "[ ! -e '$T/fresh' ]"
sync_new > "$T/fresh.log" 2>&1
check "a sync creates a missing workspace and its parent" "[ \$? -eq 0 ] && [ -e '$FRESH/resources/hcrl_isaaclab/a.py' ]"
check "and names the directory it created" "grep -q 'created $FRESH' '$T/fresh.log'"

if [ "$fails" -ne 0 ]; then
    echo "--- sync log"; tail -20 "$T/sync.log"
    echo "--- fresh log"; tail -10 "$T/fresh.log"
    exit 1
fi
echo "all checks passed"
