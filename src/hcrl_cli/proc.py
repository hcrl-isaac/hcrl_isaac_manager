"""Paths and process helpers shared by the `pls` commands. Every command runs from the manager root."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/hcrl_cli/proc.py -> the manager root
VENV = "ilab"
VENV_PY = f"{VENV}/bin/python"  # relative to ROOT, the cwd of every command
BASH_UTILS = ROOT / "scripts" / "utils.sh"
RC_FILE = Path.home() / ".bashrc"
CALLER_CWD = os.getcwd()  # where `pls` was started; set again by main() before it moves to ROOT
# The single uv venv at ./ilab (`uv sync`/`uv run` read the first); a profile-activated venv must not leak in.
BASE_ENV = {"UV_PROJECT_ENVIRONMENT": VENV, "VIRTUAL_ENV": ""}


class Failed(SystemExit):
    """A step exited non-zero; `pls` exits with the same code."""

    def __init__(self, cmd: Sequence[str], code: int) -> None:
        print(f"[pls] failed (exit {code}): {shlex.join(cmd)}", file=sys.stderr)
        super().__init__(code)


def _env(env: Mapping[str, str] | None) -> dict[str, str]:
    return {**os.environ, **BASE_ENV, **(env or {})}


def run(
    cmd: Sequence[str],
    *,
    echo: bool = False,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run one step; ``echo`` prints it first, ``check`` stops `pls` on a non-zero exit, ``capture`` keeps stdout."""
    if echo:
        print(shlex.join(cmd), file=sys.stderr)
    result = subprocess.run(
        list(cmd), env=_env(env), cwd=cwd, stdout=subprocess.PIPE if capture else None, text=True, check=False
    )
    if check and result.returncode != 0:
        raise Failed(cmd, result.returncode)
    return result


def handoff(cmd: Sequence[str], *, echo: bool = False, env: Mapping[str, str] | None = None) -> None:
    """Replace `pls` with the command's last step, so its exit code, signals and terminal are the command's own."""
    if echo:
        print(shlex.join(cmd), file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(cmd[0], list(cmd), _env(env))


def ask_select(prompt: str, choices: Sequence[str]) -> str:
    """The interactive picker (scripts/tools/ask.py); a cancelled pick stops `pls`."""
    return run([VENV_PY, "scripts/tools/ask.py", "select", prompt, *choices], capture=True).stdout.strip()
