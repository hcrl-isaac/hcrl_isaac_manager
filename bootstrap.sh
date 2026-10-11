#!/bin/sh
# Fresh machine: install uv if missing, then run `pls setup` (or `./bootstrap.sh <verb> ...`) before pls is installed.
# Needs only sh + curl; `pls deps` (part of setup) creates the ilab venv that puts `pls` on PATH.
set -eu
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
fi
[ "$#" -gt 0 ] || set -- setup
export PYTHONPATH="$PWD/src"
exec uv run --no-project --python 3.11 python -m hcrl_cli "$@"
