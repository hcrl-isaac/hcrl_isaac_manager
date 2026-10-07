#!/usr/bin/env bash
# `setup` on a CLUSTER_ARCH=arm64 profile builds the .sif in a batch job on the cluster. Stubs only: ssh runs here,
# sbatch runs the job script here, apptainer fakes pull and build. A failed pull or a rejected job never replaces
# the .sif, and the profile's resources reach sbatch.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts" "$T/remote/sif" "$T/remote/ws"
cp -r "$REPO/scripts/cluster" "$REPO/scripts/docker" "$T/scripts/"
rm -rf "$T/scripts/cluster/config" && mkdir -p "$T/scripts/cluster/config/zz"
printf 'CLUSTER_LOGIN=fake@host\nCLUSTER_ISAACLAB_DIR=%s\nCLUSTER_SIF_PATH=%s\nCLUSTER_ARCH=arm64\nCLUSTER_MIN_FREE_GB=0\n' \
    "$T/remote/ws" "$T/remote/sif" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -pdebug\n#SBATCH --account=PROJ\n#SBATCH -q normal\n#SBATCH -N 1\n' \
    > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
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
# sbatch: record the flags, run the job script here as job 4242 (as --wait would), print the id (--parsable)
cat > "$T/bin/sbatch" <<EOF
#!/usr/bin/env bash
printf '%s ' "\$@" > "$T/sbatch_args"
[ -n "\${SBATCH_REJECT:-}" ] && { echo "sbatch: error: invalid partition"; exit 1; }
cat > "$T/job.sh"
log=""; while [ \$# -gt 0 ]; do [ "\$1" = -o ] && log="\${2//%j/4242}"; shift; done
echo 4242
SLURM_JOB_ID=4242 bash "$T/job.sh" > "\${log:-/dev/null}" 2>&1
EOF
cat > "$T/bin/apptainer" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    --version) echo "apptainer version stub" ;;
    pull) [ -n "${STUB_PULL_FAIL:-}" ] && { echo "FATAL: pull failed"; exit 1; }; touch "$3" ;;
    build) echo "new" > "${@: -2:1}" ;;
esac
EOF
printf '#!/usr/bin/env bash\necho "Filesystem 1024-blocks Used Available Capacity Mounted"\necho "fake 999999999 0 999999999 0%% /"\n' > "$T/bin/df"
chmod +x "$T/bin/"*
ci() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz bash "$T/scripts/cluster/cluster_interface.sh" "$@"; }
sif="$T/remote/sif/hcrl-isaac.sif"

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

ci setup > "$T/out1" 2>&1; rc=$?
check "an arm64 setup builds the .sif in a batch job" "[ $rc = 0 ] && [ \"\$(cat '$sif')\" = new ]"
check "with the profile's partition, account and qos" \
    "grep -q -- '-p debug' '$T/sbatch_args' && grep -q -- '-A PROJ' '$T/sbatch_args' && grep -q -- '-q normal' '$T/sbatch_args'"
check "submitted --parsable --wait" "grep -q -- '--parsable --wait' '$T/sbatch_args'"
check "the base pull is recorded for reuse" "grep -q 'isaac-sim:' '$T/remote/sif/build-hcrl-isaac/isaac-sim-base.ref'"

echo old > "$sif"; rm -f "$T/remote/sif/build-hcrl-isaac/isaac-sim-base.ref"
STUB_PULL_FAIL=1 ci setup > "$T/out2" 2>&1; rc=$?
check "a failed base pull fails the setup" "[ $rc != 0 ] && grep -q 'build job 4242 failed' '$T/out2'"
check "and leaves the old .sif in place" "[ \"\$(cat '$sif')\" = old ]"

SBATCH_REJECT=1 ci setup > "$T/out3" 2>&1; rc=$?
check "a rejected job is reported as such" "[ $rc != 0 ] && grep -q 'did not accept' '$T/out3' && [ \"\$(cat '$sif')\" = old ]"

rm -f "$T/sbatch_args"
CLUSTER_BUILD_MIN_FREE_GB=99999999 ci setup > "$T/out4" 2>&1; rc=$?
check "too little free space refuses before submitting" "[ $rc != 0 ] && grep -q 'GB free' '$T/out4' && [ ! -e '$T/sbatch_args' ]"

for cmd in build push; do
    ci "$cmd" > "$T/out5" 2>&1; rc=$?
    check "an arm64 profile refuses $cmd" "[ $rc != 0 ] && grep -q 'built on the cluster' '$T/out5'"
done

sed -i 's/^CLUSTER_ARCH=.*/CLUSTER_ARCH=aarch64/' "$T/scripts/cluster/config/zz/.env.cluster"
ci setup > "$T/out6" 2>&1; rc=$?
check "an unknown CLUSTER_ARCH is refused" "[ $rc != 0 ] && grep -q 'must be amd64 or arm64' '$T/out6'"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -6 "$f"; done
    exit 1
fi
echo "all checks passed"
