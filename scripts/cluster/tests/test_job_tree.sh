#!/usr/bin/env bash
# `cluster job` runs on a staged tree: it stages the workspace as `default` (or takes --tree), submits from a per-job
# dir holding the tree's repos, this cluster's job scripts and the credentials, and marks the tree in use by the job.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote"
cp -r "$REPO/scripts/cluster/cluster_dev" "$REPO/scripts/cluster/tools" "$T/scripts/cluster/"
cp "$REPO/scripts/cluster/cluster_interface.sh" "$REPO/scripts/cluster/run_singularity.sh" "$T/scripts/cluster/"
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
cd "$HOME" && exec bash -c "$*"
EOF
chmod +x "$T/bin/ssh"
G="$T/resources/hcrl_isaaclab"
mkdir -p "$G" && (
    cd "$G" && git init -q -b main . && git config user.email t@t && git config user.name t
    echo main > code.py && git add -A && git commit -qm main && git switch -q -c feat && echo feat > code.py &&
        git commit -qam feat && git switch -q main
)
echo "WANDB_API_KEY=k" > "$T/scripts/.env.wandb"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\nCLUSTER_JOB_SCHEDULER=SLURM\n' \
    "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
# the profile's submit script: record where and with what it ran, and answer like sbatch
cat > "$T/scripts/cluster/config/zz/submit_job_slurm.sh" <<EOF
#!/usr/bin/env bash
#SBATCH -p test
{ echo "cwd \$PWD"; echo "args \$*"; } > "$T/submitted"
echo "Submitted batch job 777"
EOF
job() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz bash "$T/scripts/cluster/cluster_interface.sh" job "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

job --task T > "$T/out1" 2>&1
check "job exits 0" "[ $? -eq 0 ]"
tree="$(ls -d "$T"/remote/trees/default-* 2>/dev/null | head -1)"
check "it stages the workspace as default" "[ -n '$tree' ] && grep -qx main '$tree/resources/hcrl_isaaclab/code.py'"
jd="$(ls -d "$T"/remote/jobs/zz-* 2>/dev/null | head -1)"
check "a job dir links the tree's repos" "[ -n '$jd' ] && [ \"\$(readlink '$jd/resources')\" = '$tree/resources' ]"
check "with the job scripts and this cluster's env" \
    "[ -f '$jd/scripts/cluster/run_singularity.sh' ] && [ -f '$jd/scripts/cluster/config/zz/submit_job_slurm.sh' ] && grep -q CLUSTER_LOGIN '$jd/scripts/cluster/.env.cluster'"
check "and owner-only W&B credentials" "[ \"\$(stat -c %a '$jd/scripts/cluster/.env.wandb')\" = 600 ]"
check "submitted from the job dir with its path first" "grep -qx 'cwd $jd' '$T/submitted' && grep -qx 'args $jd hcrl-isaac --task T' '$T/submitted'"
check "the tree is marked in use by the job" "[ -f '$tree/.in-use/777.nostep' ]"
check "no workspace copy on the cluster" "[ ! -e '$T/remote/resources' ] && [ -z \"\$(ls -d '$T'/remote_* 2>/dev/null)\" ]"

PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" stage pin hcrl_isaaclab=feat > /dev/null 2>&1
rm -f "$T/submitted"
job --tree pin --task T > "$T/out2" 2>&1
pin="$(ls -d "$T"/remote/trees/pin-* 2>/dev/null | head -1)"
check "--tree runs that tree" "[ $? -eq 0 ] && [ \"\$(readlink \"\$(ls -td '$T'/remote/jobs/zz-*/ | head -1)resources\")\" = '$pin/resources' ]"
check "without staging a new default" "[ \$(ls -d '$T'/remote/trees/default-* | wc -l) -eq 1 ]"
rm -f "$T/submitted"
job --tree nope --task T > "$T/out3" 2>&1
check "an unknown tree fails before submitting" "[ $? -ne 0 ] && [ ! -e '$T/submitted' ]"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -15 "$f"; done
    exit 1
fi
echo "all checks passed"
