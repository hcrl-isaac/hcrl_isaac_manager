#!/usr/bin/env bash
# Branch switches against the gitignored profiles, in a throwaway repo: the cluster interface must refuse on a branch
# that still tracks scripts/cluster/config/ and, back on the new branch, restore the hand-edited profile exactly.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
cd "$T" || exit 1
git init -q -b main . && git config user.email t@t && git config user.name t
mkdir -p scripts/cluster/config/zz scripts/cluster/tools
cp "$REPO/scripts/cluster/cluster_interface.sh" scripts/cluster/
cp "$REPO/scripts/cluster/tools/restore_profiles.sh" scripts/cluster/tools/
printf '#!/usr/bin/env bash\n#SBATCH -p old\n' > scripts/cluster/config/zz/submit_job_slurm.sh
printf 'CLUSTER_LOGIN=u@h\n' > scripts/cluster/config/zz/.env.cluster
printf '.env.*\n' > .gitignore
git add -A && git commit -qm "profiles tracked" && git branch old

git rm -rq --cached scripts/cluster/config && printf '/scripts/cluster/config/\n' >> .gitignore
git add .gitignore && git commit -qm "profiles per-user"
printf '#!/usr/bin/env bash\n#SBATCH -p mine\n#SBATCH -A hand-edit\n' > scripts/cluster/config/zz/submit_job_slurm.sh
cp scripts/cluster/config/zz/submit_job_slurm.sh "$T/hand_edited"

run() { CLUSTER=zz bash scripts/cluster/cluster_interface.sh help > "$T/out" 2>&1; }
fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

run
check "runs on the new branch" "[ $? -eq 0 ]"
check "snapshot taken" "cmp -s scripts/cluster/config/zz/submit_job_slurm.sh scripts/cluster/config/zz/.backup/submit_job_slurm.sh"

git checkout -q old 2>/dev/null
check "old branch overwrote the profile (the hazard)" "grep -q 'SBATCH -p old' scripts/cluster/config/zz/submit_job_slurm.sh"
run
check "refuses on a branch that tracks the profiles" "[ $? -ne 0 ] && grep -q 'Merge main' '$T/out'"
check "refusal leaves the snapshot alone" "cmp -s '$T/hand_edited' scripts/cluster/config/zz/.backup/submit_job_slurm.sh"

git checkout -q main 2>/dev/null
check "switching back deleted the profile's submit script" "[ ! -e scripts/cluster/config/zz/submit_job_slurm.sh ]"
check ".backup/ and .env.cluster survive both switches" "[ -d scripts/cluster/config/zz/.backup ] && [ -f scripts/cluster/config/zz/.env.cluster ]"
run
check "runs again on the new branch" "[ $? -eq 0 ]"
check "restored byte-identical from .backup/" "cmp -s '$T/hand_edited' scripts/cluster/config/zz/submit_job_slurm.sh"

rm -rf scripts/cluster/config/zz/.backup scripts/cluster/config/zz/submit_job_slurm.sh
run
check "without a backup, restores from git history" "grep -q 'SBATCH -p old' scripts/cluster/config/zz/submit_job_slurm.sh"

if [ "$fails" -ne 0 ]; then
    echo "--- last output"; cat "$T/out"
    exit 1
fi
echo "all checks passed"
