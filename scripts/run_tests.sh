#!/usr/bin/env bash
# Run every manager script test: scripts/<area>/tests/test_*.py (unittest) and test_*.sh (bash). Tests use stubs
# only (no cluster, ssh, Ray or GPU). Fails if any test fails or if none is found.
set -u
cd "$(dirname "$0")/.."

failed=() found=0
while IFS= read -r dir; do
    if compgen -G "$dir/test_*.py" > /dev/null; then
        found=$((found + 1))
        echo "== python: $dir"
        python3 -m unittest discover -s "$dir" -p 'test_*.py' || failed+=("$dir (python)")
    fi
    for test in "$dir"/test_*.sh; do
        [ -f "$test" ] || continue
        found=$((found + 1))
        echo "== shell: $test"
        bash "$test" || failed+=("$test")
    done
done < <(find scripts -type d -name tests -not -path '*/__pycache__/*' | sort)

if [ "$found" -eq 0 ]; then
    echo "[run_tests] no tests found under scripts/*/tests" >&2
    exit 1
fi
if [ "${#failed[@]}" -gt 0 ]; then
    printf '[run_tests] FAILED: %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "[run_tests] all $found test group(s) passed"
