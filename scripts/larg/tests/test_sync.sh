#!/usr/bin/env bash
# scripts/larg/sync.sh's package registration through ssh/uv stubs: a venv inside the tree or one already serving it
# gets the editable installs; a venv that links into another tree is left alone with a warning.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/.local/bin" "$T/local/resources/hcrl_isaaclab" "$T/local/resources/hhlm_tasks" "$T/remote"
touch "$T/local/resources/hcrl_isaaclab/pyproject.toml" "$T/local/resources/hhlm_tasks/pyproject.toml"
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-J|-i|-p|-l|-F|-e) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
exec bash -c "$*"
EOF
printf '#!/usr/bin/env bash\necho "$@" >> "%s/uv_calls"\n' "$T" > "$T/.local/bin/uv"
chmod +x "$T/bin/ssh" "$T/.local/bin/uv"
venv() {  # venv DIR SERVES: a fake venv whose hcrl_isaaclab resolves to SERVES
    mkdir -p "$1/bin"
    printf '#!/usr/bin/env bash\necho %q\n' "$2" > "$1/bin/python"
    chmod +x "$1/bin/python"
}
sync() { rm -f "$T/uv_calls"; env PATH="$T/bin:$PATH" HOME="$T" LARG_LOCAL_DIR="$T/local" LARG_REMOTE_DIR="$T/remote" \
    bash "$REPO/scripts/larg/sync.sh" box > "$T/out" 2>&1; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

venv "$T/remote/ilab" "$T/remote/resources/hcrl_isaaclab/hcrl_isaaclab/__init__.py"
sync
check "a venv inside the tree gets the editable installs" "grep -q -- '-e resources/hhlm_tasks' '$T/uv_calls' && grep -q -- '--no-deps' '$T/uv_calls'"

rm -rf "$T/remote/ilab"
venv "$T/other/ilab" "$T/other/resources/hcrl_isaaclab/hcrl_isaaclab/__init__.py"
ln -s "$T/other/ilab" "$T/remote/ilab"
sync
check "a venv linked into another tree is left alone" "[ ! -e '$T/uv_calls' ] && grep -q 'WARNING: ilab resolves to $T/other/ilab' '$T/out'"

rm "$T/remote/ilab"
venv "$T/venv" "$T/remote/resources/hcrl_isaaclab/hcrl_isaaclab/__init__.py"
ln -s "$T/venv" "$T/remote/ilab"
sync
check "a linked venv that already serves this tree gets the installs" "grep -q -- '-e resources/hhlm_tasks' '$T/uv_calls'"

if [ "$fails" -ne 0 ]; then
    cat "$T/out"
    exit 1
fi
echo "all checks passed"
