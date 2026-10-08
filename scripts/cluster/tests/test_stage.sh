#!/usr/bin/env bash
# `develop stage` / `trees` through an ssh stub: named repos at named refs land in a new tree, a bare `stage` ships the
# workspace as on disk as `default` (the base of later partial trees), bad input touches nothing, and trees resolve,
# prune and are removed safely.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz"
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
printf '#!/usr/bin/env bash\n[ -f "%s/squeue_err" ] && { cat "%s/squeue_err" >&2; exit 1; }\ncat "%s/running_jobs" 2>/dev/null\n' \
    "$T" "$T" "$T" > "$T/bin/squeue"
chmod +x "$T/bin/ssh" "$T/bin/squeue"

L="$T/local"
R="$T/remote"
G="$L/resources/hcrl_isaaclab"
mkdir -p "$G" "$R/resources/hcrl_isaaclab" "$R/resources/hcrl_robots"
(
    cd "$G" && git init -q -b main . && git config user.email t@t && git config user.name t
    printf 'logs/\n' > .gitignore
    echo main > code.py && head -c 100000 /dev/urandom > big.bin && git add -A && git commit -qm main
    git switch -q -c feat && echo feat > code.py && git commit -qam feat && git switch -q main
)
cp "$G/code.py" "$G/big.bin" "$R/resources/hcrl_isaaclab/"
touch -d 2020-01-01 "$R/resources/hcrl_isaaclab/big.bin"  # an old copy: linking must go by content, not mtime
echo asset > "$R/resources/hcrl_robots/t1.urdf"
mkdir -p "$R/resources/hcrl_isaaclab/pol"
ln -s /workspace/ext/hcrl_isaaclab/.artifacts/abc "$R/resources/hcrl_isaaclab/pol/bfmzero_x"
mkdir -p "$T/art/.artifacts/k/v0" "$T/art/.artifacts/k/v1"
echo w > "$T/art/.artifacts/k/v0/w.pt" && echo w > "$T/art/.artifacts/k/v1/w.pt"
ln -s "$T/art/.artifacts/k/v0" "$R/resources/hcrl_isaaclab/pol/bfmzero_y"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$R" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
dev() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$L" bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }
ntrees() { ls -d "$R"/trees/"$1"-* 2>/dev/null | grep -c . ; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

dev stage t1 hcrl_isaaclab=feat > "$T/out1" 2>&1
check "stage exits 0" "[ $? -eq 0 ]"
tree="$(ls -d "$R"/trees/t1-* 2>/dev/null | head -1)"
check "tree created and complete" "[ -n '$tree' ] && [ -f '$tree/.complete' ]"
check "tree has the ref's content" "grep -qx feat '$tree/resources/hcrl_isaaclab/code.py'"
check "manifest records the commit" "grep -q \"hcrl_isaaclab feat \$(git -C '$G' rev-parse feat) local-ref\" '$tree/MANIFEST'"
check "manifest lists shared repos as live" "grep -q '^hcrl_robots shared-live ' '$tree/MANIFEST'"
check "unstaged repos link to the shared checkout" "[ -L '$tree/resources/hcrl_robots' ] && [ -f '$tree/resources/hcrl_robots/t1.urdf' ]"
check "staged repo has writable mount points" "[ -d '$tree/resources/hcrl_isaaclab/logs' ] && [ -d '$tree/resources/hcrl_isaaclab/outputs' ] && [ -d '$tree/resources/hcrl_isaaclab/wandb' ]"
check "staged files are read-only" "[ ! -w '$tree/resources/hcrl_isaaclab/code.py' ] && ! (echo x >> '$tree/resources/hcrl_isaaclab/code.py') 2>/dev/null"
check "its directories stay writable" "touch '$tree/resources/hcrl_isaaclab/new_file' && rm '$tree/resources/hcrl_isaaclab/new_file'"
check "artifact links from the shared checkout are carried" \
    "[ \"\$(readlink '$tree/resources/hcrl_isaaclab/pol/bfmzero_x')\" = /workspace/ext/hcrl_isaaclab/.artifacts/abc ]"
ARTIFACTS_PY="${HCRL_ISAACLAB_DIR:-$(cd "$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)/resources/hcrl_isaaclab}/hcrl_isaaclab/utils/artifacts.py"
if [ -f "$ARTIFACTS_PY" ]; then
    HCRL_ARTIFACT_ROOT="$T/art/.artifacts" python3 "$REPO/scripts/cluster/tests/resolve_in_tree.py" "$ARTIFACTS_PY" \
        "$tree/resources" hcrl_isaaclab/pol/bfmzero_y "$T/art/.artifacts/k/v1" > "$T/out_resolve" 2>&1
    check "the artifact resolver re-links a dest inside the tree" "grep -qx '$T/art/.artifacts/k/v1' '$T/out_resolve'"
    check "and leaves the shared checkout's link alone" "[ \"\$(readlink '$R/resources/hcrl_isaaclab/pol/bfmzero_y')\" = '$T/art/.artifacts/k/v0' ]"
    rm -rf "$tree/resources/hcrl_isaaclab/hcrl_isaaclab"
else
    echo "SKIP artifact resolver checks (no hcrl_isaaclab checkout; set HCRL_ISAACLAB_DIR)"
fi
check "hcrl_isaaclab has an artifact mount point" "[ -d '$tree/resources/hcrl_isaaclab/.artifacts' ]"
check "tree carries its own node_exec.sh" "[ -f '$tree/scripts/cluster/cluster_dev/node_exec.sh' ]"
check "shared checkout untouched" "grep -qx main '$R/resources/hcrl_isaaclab/code.py'"
check "no inode is shared with the mutable shared checkout" \
    "[ \"\$(stat -c %i '$tree/resources/hcrl_isaaclab/big.bin')\" != \"\$(stat -c %i '$R/resources/hcrl_isaaclab/big.bin')\" ]"
check "temporary checkout cleaned up" "[ \"\$(git -C '$G' worktree list | wc -l)\" -eq 1 ]"
check "run hint names the full tree id" "grep -q \"exec --tree \$(basename '$tree')\" '$T/out1'"

dev stage t1 hcrl_isaaclab=feat > "$T/out2" 2>&1
check "same refs reuse the tree" "grep -q 'already staged' '$T/out2' && [ \$(ntrees t1) -eq 1 ]"

rm "$R/resources/hcrl_isaaclab/big.bin"
dev stage t3 hcrl_isaaclab=feat > "$T/out_t3" 2>&1
check "earlier trees serve as hardlink sources too" \
    "[ \"\$(stat -c %i \"\$(ls -d '$R'/trees/t3-*)/resources/hcrl_isaaclab/big.bin\")\" = \"\$(stat -c %i '$tree/resources/hcrl_isaaclab/big.bin')\" ]"

# names are validated before anything is touched
mkdir -p "$L/resources/hcrl_robots" "$L/resources/IsaacLab"
for spec in "../../../resources/hcrl_isaaclab=feat" "=feat" "hcrl_isaacla=feat" "hcrl_isaaclab=feat hcrl_isaaclab=main" \
    "motion_datasets=main" "hcrl_isaaclab=$G/sub"; do
    # shellcheck disable=SC2086
    dev stage bad $spec > "$T/out_b1" 2>&1
    check "rejects spec '$spec'" "[ $? -ne 0 ] && [ \$(ntrees bad) -eq 0 ]"
done
dev stage "../x" hcrl_isaaclab=feat > "$T/out_b1" 2>&1
check "rejects a tree name with /" "[ $? -ne 0 ]"
mkdir -p "$G/sub"
check "bad input left the shared checkout alone" "grep -qx main '$R/resources/hcrl_isaaclab/code.py' && [ ! -e '$R/resources/hcrl_isaaclab/hcrl_isaaclab' ]"

# untracked files are uploaded and fingerprinted
echo v1 > "$G/new_mod.py"
dev stage wt "hcrl_isaaclab=$G" > "$T/out_b2" 2>&1
echo v2 > "$G/new_mod.py"
dev stage wt "hcrl_isaaclab=$G" > "$T/out_b2b" 2>&1
check "an untracked edit is a new tree" "[ \$(ntrees wt) -eq 2 ]"
check "the newest tree carries it" "grep -qx v2 \"\$(ls -td '$R'/trees/wt-*/ | head -1)resources/hcrl_isaaclab/new_mod.py\""
check "ignored files are not uploaded" "mkdir -p '$G/logs' && touch '$G/logs/run.txt' && dev stage wt2 'hcrl_isaaclab=$G' >/dev/null 2>&1 && \
    [ ! -e \"\$(ls -d '$R'/trees/wt2-* | head -1)/resources/hcrl_isaaclab/logs/run.txt\" ]"
check "manifest records the absolute worktree path" "grep -q '^hcrl_isaaclab $G ' \"\$(ls -td '$R'/trees/wt-*/ | head -1)MANIFEST\""
mkdir -p "$G/pkg_a" "$G/pkg_b" && echo a > "$G/pkg_a/f" && echo b > "$G/pkg_b/f" && ln -s pkg_a "$G/link_dir"
dev stage sl "hcrl_isaaclab=$G" > "$T/out_sl1" 2>&1
check "a symlink to a directory stages" "[ $? -eq 0 ] && [ -L \"\$(ls -d '$R'/trees/sl-* | head -1)/resources/hcrl_isaaclab/link_dir\" ]"
ln -sfn pkg_b "$G/link_dir"
dev stage sl "hcrl_isaaclab=$G" > /dev/null 2>&1
check "retargeting a symlink is a new tree" "[ \$(ntrees sl) -eq 2 ]"
chmod +x "$G/pkg_a/f"
dev stage sl "hcrl_isaaclab=$G" > /dev/null 2>&1
check "a mode change is a new tree" "[ \$(ntrees sl) -eq 3 ]"
rm -rf "$G/new_mod.py" "$G/pkg_a" "$G/pkg_b" "$G/link_dir" "$G/sub"

# resolution is exact, and a bare name means its newest tree
dev stage r hcrl_isaaclab=main > /dev/null 2>&1
dev stage r-x hcrl_isaaclab=main > /dev/null 2>&1
check "one tree named r resolves" "dev __resolve_tree r 2>/dev/null | grep -q '/trees/r-[0-9a-f]\{10\}$'"
dev stage r hcrl_isaaclab=feat > /dev/null 2>&1
check "of two trees named r the newest resolves" "grep -qx feat \"\$(dev __resolve_tree r 2>/dev/null)/resources/hcrl_isaaclab/code.py\""
check "the full id still resolves" "dev __resolve_tree \"\$(basename \"\$(ls -d '$R'/trees/r-* | grep -v r-x | head -1)\")\" > /dev/null 2>&1"
check "a prefix-sharing name is not matched" "dev __resolve_tree r-x 2>/dev/null | grep -q '/trees/r-x-'"

# failures clean up and never report success
dev stage multi hcrl_isaaclab=feat hcrl_robots=nope > "$T/out_b5" 2>&1
check "a failing spec fails the stage" "[ $? -ne 0 ] && [ \$(ntrees multi) -eq 0 ]"
check "it leaves no temporary checkout" "[ \"\$(git -C '$G' worktree list | wc -l)\" -eq 1 ]"
check "it leaves no partial" "! ls -d '$R'/trees/*.partial.* 2>/dev/null"
mkdir -p "$T/lfsbin"
printf '#!/usr/bin/env bash\n[ "$1" = ls-files ] && echo "0123abcd - big.bin"\nexit 0\n' > "$T/lfsbin/git-lfs"
chmod +x "$T/lfsbin/git-lfs"
PATH="$T/lfsbin:$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$L" \
    bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" stage lfs hcrl_isaaclab=feat > "$T/out_lfs" 2>&1
check "missing LFS objects fail the stage" "[ $? -ne 0 ] && grep -q 'LFS objects missing' '$T/out_lfs' && [ \$(ntrees lfs) -eq 0 ]"
check "and leave no worktree registered" "[ \"\$(git -C '$G' worktree list | wc -l)\" -eq 1 ]"
chmod 555 "$R/trees"
dev stage ro hcrl_isaaclab=feat > "$T/out_b5b" 2>&1
rc=$?
chmod 755 "$R/trees"
check "an upload failure is not reported as staged" "[ $rc -ne 0 ] && ! grep -q '^\[cluster_dev\] Staged' '$T/out_b5b'"

# concurrent stages of the same refs end in one complete tree
dev stage conc hcrl_isaaclab=feat > "$T/out_c1" 2>&1 &
dev stage conc hcrl_isaaclab=feat > "$T/out_c2" 2>&1 &
wait
check "concurrent stages give one tree" "[ \$(ntrees conc) -eq 1 ] && [ -f \"\$(ls -d '$R'/trees/conc-*)/.complete\" ] && ! ls -d '$R'/trees/conc-*.partial.* 2>/dev/null"

dev trees > "$T/out5" 2>&1
check "trees lists them" "[ \"\$(grep -c '^t1-' '$T/out5')\" -eq 1 ] && ! grep -q '^ *#' '$T/out5'"

# trees rm: refused while a job that used the tree is running
id="$(basename "$tree")"
mkdir -p "$tree/.in-use" && touch "$tree/.in-use/4242.0"
printf '4242.0\n4242.1\n' > "$T/running_jobs"
dev trees rm "$id" > "$T/out6" 2>&1
check "rm refuses a tree whose step still runs" "[ $? -ne 0 ] && [ -d '$tree' ]"
echo 'Socket timed out on send/recv operation' > "$T/squeue_err"
dev trees rm "$id" > "$T/out6" 2>&1
check "a failing squeue refuses instead of removing" "[ $? -ne 0 ] && [ -d '$tree' ] && grep -q 'cannot check job 4242' '$T/out6'"
rm "$T/squeue_err"
echo 4242.1 > "$T/running_jobs"
touch "$tree/.in-use/4242.nostep"
dev trees rm "$id" > "$T/out6" 2>&1
check "a no-step marker holds while its job runs" "[ $? -ne 0 ] && grep -q '4242.nostep' '$T/out6'"
rm "$tree/.in-use/4242.nostep"
touch "$tree/.in-use/node7.99"
dev trees rm "$id" > "$T/out6" 2>&1
check "an ended step on a live job does not hold it, a non-slurm marker does" "[ $? -ne 0 ] && grep -q 'node7.99' '$T/out6' && ! grep -q '4242.0' '$T/out6'"
rm "$tree/.in-use/node7.99"
echo 'slurm_load_jobs error: Invalid job id specified' > "$T/squeue_err"
dev trees rm "$id" > "$T/out6" 2>&1
check "rm removes it once the step ended" "[ $? -eq 0 ] && [ ! -e '$tree' ]"
rm -f "$T/squeue_err"
check "rm needs the full id" "! dev trees rm r > /dev/null 2>&1"
mkdir -p "$R/trees/old.partial.1/a" "$R/trees/new.partial.2" "$R/trees/busy.partial.3/a"
touch -d '2 hours ago' "$R/trees/old.partial.1/a" "$R/trees/old.partial.1" "$R/trees/busy.partial.3"
dev trees rm --partials > /dev/null 2>&1
check "rm --partials removes only partials idle for an hour" \
    "[ ! -e '$R/trees/old.partial.1' ] && [ -e '$R/trees/new.partial.2' ] && [ -e '$R/trees/busy.partial.3' ]"

# an asset repo stages at a ref like a code repo, but writable and with files of its own: the run's URDF -> USD
# conversion rewrites files beside the URDF, which must touch neither another tree nor the shared checkout
A="$L/resources/ssti_robots"
mkdir -p "$A" "$R/resources/ssti_robots"
(
    cd "$A" && git init -q -b main . && git config user.email t@t && git config user.name t
    echo "mass 1.0" > shank.urdf && git add -A && git commit -qm main
    git switch -q -c heavy && echo "mass 1.82" > shank.urdf && git commit -qam heavy && git switch -q main
)
echo "mass 1.0" > "$R/resources/ssti_robots/shank.urdf"
dev stage assets1 ssti_robots=heavy hcrl_isaaclab=feat > "$T/out_a1" 2>&1
check "an asset repo stages at a ref" "[ $? -eq 0 ]"
at="$(ls -d "$R"/trees/assets1-* 2>/dev/null | head -1)"
check "with the ref's content" "grep -qx 'mass 1.82' '$at/resources/ssti_robots/shank.urdf'"
check "its files stay writable for the run's conversions" "[ -w '$at/resources/ssti_robots/shank.urdf' ]"
check "while code repos stay read-only" "[ ! -w '$at/resources/hcrl_isaaclab/code.py' ]"
dev stage assets2 ssti_robots=heavy > "$T/out_a2" 2>&1
at2="$(ls -d "$R"/trees/assets2-* 2>/dev/null | head -1)"
check "and no file is shared with another tree" \
    "[ \"\$(stat -c %i '$at/resources/ssti_robots/shank.urdf')\" != \"\$(stat -c %i '$at2/resources/ssti_robots/shank.urdf')\" ]"
echo "usd" > "$at/resources/ssti_robots/shank.usd" && echo "mass 9" > "$at/resources/ssti_robots/shank.urdf"
check "so a write in one tree stays there" \
    "grep -qx 'mass 1.82' '$at2/resources/ssti_robots/shank.urdf' && grep -qx 'mass 1.0' '$R/resources/ssti_robots/shank.urdf'"

# a bare `stage` is the whole workspace as on disk -> `default`: every git repo under resources/ but the shared data
# and the repos this cluster excludes; the data repos' training files are added to the shared copy
mkr() {  # mkr NAME FILE CONTENT: a one-commit git repo under the local resources/
    mkdir -p "$L/resources/$1" && (cd "$L/resources/$1" && git init -q -b main . && git config user.email t@t && \
        git config user.name t && echo "$3" > "$2" && git add -A && git commit -qm init)
}
mkr robot_rl r.py rl
mkr ssti_tasks t.py excluded
echo ssti_tasks > "$T/scripts/cluster/config/zz/.rsync-exclude"
mkr motion_datasets README.md data
mkdir -p "$L/resources/motion_datasets/clips" "$R/resources/motion_datasets"
echo pt > "$L/resources/motion_datasets/clips/walk.pt" && echo m > "$L/resources/motion_datasets/clips/walk.manifest.json"
echo raw > "$L/resources/motion_datasets/clips/walk.npz" && echo keep > "$R/resources/motion_datasets/remote_only.pt"
echo uncommitted > "$G/whole_new.py"
mkdir -p "$G/worktrees/other/pkg" "$G/.claude" && echo big > "$G/worktrees/other/pkg/huge.bin" && echo s > "$G/.claude/notes.md"
dev stage > "$T/out_w1" 2>&1
check "a bare stage exits 0" "[ $? -eq 0 ]"
dt="$(ls -d "$R"/trees/default-* 2>/dev/null | head -1)"
check "as the tree default" "[ -n '$dt' ] && [ -f '$dt/.complete' ]"
check "with uncommitted work" "grep -qx uncommitted '$dt/resources/hcrl_isaaclab/whole_new.py'"
check "but never the checkout's worktrees or session notes" \
    "[ ! -e '$dt/resources/hcrl_isaaclab/worktrees/other' ] && [ ! -e '$dt/resources/hcrl_isaaclab/.claude' ]"
check "every git repo staged" "[ -f '$dt/resources/robot_rl/r.py' ] && [ ! -L '$dt/resources/ssti_robots' ]"
check "but an excluded repo" "[ ! -e '$dt/resources/ssti_tasks' ]"
check "a non-repo dir stays a shared link" "[ -L '$dt/resources/hcrl_robots' ]"
check "the data repo is a link to the shared copy" "[ \"\$(readlink '$dt/resources/motion_datasets')\" = '$R/resources/motion_datasets' ]"
check "its training files are added there" "[ -f '$R/resources/motion_datasets/clips/walk.pt' ] && [ -f '$R/resources/motion_datasets/clips/walk.manifest.json' ]"
check "other files are not" "[ ! -e '$R/resources/motion_datasets/clips/walk.npz' ] && [ ! -e '$R/resources/motion_datasets/README.md' ]"
check "and nothing there is deleted" "[ -f '$R/resources/motion_datasets/remote_only.pt' ]"
check "nothing lands on the shared workspace's code" "[ ! -e '$R/resources/robot_rl' ]"

# a partial tree takes its other repos from the newest default, never the shared checkout
dev stage p1 hcrl_isaaclab=feat > "$T/out_p1" 2>&1
pt="$(ls -d "$R"/trees/p1-* 2>/dev/null | head -1)"
check "a partial stage exits 0" "[ -n '$pt' ] && [ -f '$pt/.complete' ]"
check "its named repo is at the ref" "grep -qx feat '$pt/resources/hcrl_isaaclab/code.py'"
check "a code repo of the base is hardlinked" "[ \"\$(stat -c %i '$pt/resources/robot_rl/r.py')\" = \"\$(stat -c %i '$dt/resources/robot_rl/r.py')\" ]"
check "an asset repo of the base is its own writable copy" \
    "[ \"\$(stat -c %i '$pt/resources/ssti_robots/shank.urdf')\" != \"\$(stat -c %i '$dt/resources/ssti_robots/shank.urdf')\" ] && [ -w '$pt/resources/ssti_robots/shank.urdf' ]"
check "the data stays a shared link" "[ -L '$pt/resources/motion_datasets' ]"
check "the manifest names the base" "grep -q \"^robot_rl base \$(basename '$dt')\" '$pt/MANIFEST' && grep -q \"^base \$(basename '$dt')\" '$pt/MANIFEST'"

# a nearly full Lustre metadata target refuses the stage
printf '#!/usr/bin/env bash\ncase "$1" in getstripe) echo 0 ;; df) echo "fs-MDT0000_UUID 100 99 1 99%% /fs[MDT:0]" ;; esac\n' > "$T/bin/lfs"
chmod +x "$T/bin/lfs"
dev stage full hcrl_isaaclab=feat > "$T/out_i1" 2>&1
check "99% inodes refuses" "[ $? -ne 0 ] && grep -q 'out of inodes' '$T/out_i1' && [ \$(ntrees full) -eq 0 ]"
dev stage --no-space-check full hcrl_isaaclab=feat > /dev/null 2>&1
check "--no-space-check stages anyway" "[ \$(ntrees full) -eq 1 ]"
rm "$T/bin/lfs"

# each stage keeps the newest 5 trees of its name, and any a run still uses
mkdir -p "$dt/.in-use" && touch "$dt/.in-use/node7.99"
for i in 1 2 3 4 5 6; do echo "v$i" > "$G/whole_new.py"; dev stage > /dev/null 2>&1; done
check "older trees are pruned" "[ \$(ntrees default) -eq 6 ]"
check "the one in use is kept" "[ -d '$dt' ]"
check "the newest is what default resolves to" "grep -qx v6 \"\$(dev __resolve_tree default 2>/dev/null)/resources/hcrl_isaaclab/whole_new.py\""
rm -f "$G/whole_new.py"

if [ "$fails" -ne 0 ]; then
    ls -la "$R/trees"
    for f in "$T"/out*; do echo "--- $f"; tail -15 "$f"; done
    exit 1
fi
echo "all checks passed"
