#!/usr/bin/env bash
# scripts/larg/train.sh through an ssh stub against a fake box: the flat layout's train.py under the workspace's
# ilab python, torchrun only for several GPUs, argv quoting, the GPU pin, and per-run/per-GPU scratch dirs.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/ws/ilab/bin" "$T/ws/scripts" "$T/ws/resources/hcrl_isaaclab/scripts"
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-J|-i|-p|-l|-F) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
exec bash -c "$*"
EOF
# the box's python records what it was asked to run and the environment it got
cat > "$T/ws/ilab/bin/python" <<EOF
#!/usr/bin/env bash
{ printf 'ARG<%s>\n' "\$@"; echo "CVD=\${CUDA_VISIBLE_DEVICES:-unset}"; echo "TMPDIR=\$TMPDIR"; echo "OMNI=\$OMNI_CACHE_DIR"; } > "$T/ran"
EOF
chmod +x "$T/bin/ssh" "$T/ws/ilab/bin/python"
launch() { env PATH="$T/bin:$PATH" LARG_REMOTE_DIR="$T/ws" LARG_SCRATCH="$T/scratch" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}
wait_ran() { for _ in $(seq 50); do [ -s "$T/ran" ] && return; sleep 0.1; done; }

rm -f "$T/ran"
launch CUDA_VISIBLE_DEVICES=2 bash "$REPO/scripts/larg/train.sh" pogba hhlm/T1-CubeLift-v0 "rate 0.5 seed 1" "cube lift" 4096 \
    -- --rate 0.5 --headless > "$T/out1" 2>&1
wait_ran
check "one GPU runs the flat layout's train.py directly" "grep -qx 'ARG<resources/hcrl_isaaclab/scripts/train.py>' '$T/ran' && ! grep -q 'torch.distributed' '$T/ran'"
check "run name and group keep their spaces" "grep -qx 'ARG<rate 0.5 seed 1>' '$T/ran' && grep -qx 'ARG<cube lift>' '$T/ran'"
check "extra args and num_envs pass through" "grep -qx 'ARG<--headless>' '$T/ran' && grep -qx 'ARG<4096>' '$T/ran' && grep -qx 'ARG<--video>' '$T/ran'"
check "the GPU pin is exported" "grep -qx 'CVD=2' '$T/ran'"
check "TMPDIR is per run, Kit caches per GPU, both on scratch" \
    "grep -q '^TMPDIR=$T/scratch/larg-runs/hhlm_T1-CubeLift-v0_.*_gpu2/tmp$' '$T/ran' && grep -qx 'OMNI=$T/scratch/kit-cache/gpu2/omni' '$T/ran'"
check "the log lands in the run dir" "grep -q 'log: $T/scratch/larg-runs/.*_gpu2/train.log' '$T/out1'"

rm -f "$T/ran"
launch LARG_NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash "$REPO/scripts/larg/train.sh" pogba hhlm/T1-CubeLift-v0 r > "$T/out2" 2>&1
wait_ran
check "several GPUs run under torchrun with --distributed" \
    "grep -qx 'ARG<torch.distributed.run>' '$T/ran' && grep -qx 'ARG<--nproc_per_node=2>' '$T/ran' && grep -qx 'ARG<--distributed>' '$T/ran'"

check "an A40 host records video in-process" "sed -n '/^ARG<--video>$/{n;p}' '$T/ran' | grep -qx 'ARG<on>'"
check "an unleased launch says so" "grep -q 'LARG_HOLDER unset' '$T/out2'"

rm -f "$T/ran"
launch CUDA_VISIBLE_DEVICES=3 bash "$REPO/scripts/larg/train.sh" hazard hhlm/T1-CubeLift-v0 r > "$T/out3" 2>&1
wait_ran
check "an A100 host passes --video async" "sed -n '/^ARG<--video>$/{n;p}' '$T/ran' | grep -qx 'ARG<async>'"

# LARG_HOLDER leases the pinned cards through `just res claim` first; a refused claim launches nothing
printf '#!/usr/bin/env bash\necho "$@" > "%s/claimed"\nexit "${CLAIM_RC:-0}"\n' "$T" > "$T/bin/python3"
chmod +x "$T/bin/python3"
rm -f "$T/ran"
launch LARG_HOLDER=me CUDA_VISIBLE_DEVICES=0,2 bash "$REPO/scripts/larg/train.sh" pogba hhlm/T1-CubeLift-v0 r > "$T/out4" 2>&1
wait_ran
check "LARG_HOLDER claims every pinned card" "grep -q 'claim pogba:0 pogba:2 --holder me --note r' '$T/claimed'"
rm -f "$T/ran"
launch CLAIM_RC=1 LARG_HOLDER=me CUDA_VISIBLE_DEVICES=1 bash "$REPO/scripts/larg/train.sh" pogba hhlm/T1-CubeLift-v0 r > "$T/out5" 2>&1
sleep 1
check "a refused claim launches nothing" "[ ! -e '$T/ran' ]"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out* "$T/ran"; do echo "--- $f"; cat "$f" 2>/dev/null | tail -20; done
    exit 1
fi
echo "all checks passed"
