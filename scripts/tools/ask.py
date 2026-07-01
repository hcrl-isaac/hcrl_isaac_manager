#!/usr/bin/env python3
"""Small questionary-backed prompts shared by the justfile (pretty, TTY-aware, non-TTY fallbacks).

Subcommands:
    select <title> <option>...     Arrow-key pick one option; prints the chosen value to stdout.
    confirm <message>               Yes/no prompt; exit 0 on yes, 1 on no (defaults to no with no TTY).
    wandb-env <template> <out>      Prompt W&B username + (masked) API key; render <template> to <out>.

Each falls back to plain stdin (or a safe default) when there is no TTY, so CI / piped use never hangs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _select(title: str, options: list[str]) -> str:
    if not options:
        sys.exit("[ask] select needs at least one option")
    if not sys.stdin.isatty():
        return options[0]
    import questionary

    choice = questionary.select(title, choices=options).ask()
    if choice is None:
        sys.exit(130)
    return choice


def _confirm(message: str, default: bool = False) -> bool:
    if not sys.stdin.isatty():
        return default
    import questionary

    return bool(questionary.confirm(message, default=default).ask())


def _wandb_env(template: Path, out: Path) -> None:
    tty = sys.stdin.isatty()
    if tty:
        import questionary

        username = questionary.text("W&B username:").ask()
        api_key = questionary.password("W&B API key:").ask()
        if username is None or api_key is None:
            sys.exit(130)
    else:  # no TTY -> plain fallback
        import getpass

        username = input("W&B username: ").strip()
        api_key = getpass.getpass("W&B API key: ").strip()

    rendered = template.read_text()
    for key, val in {"WANDB_USERNAME": username, "WANDB_API_KEY": api_key}.items():
        rendered = rendered.replace(f"${key}", val)
    out.write_text(rendered)
    print(f"[ask] wrote {out}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: ask.py <select|wandb-env> ...")
    cmd, rest = args[0], args[1:]
    if cmd == "select":
        print(_select(rest[0], rest[1:]))
    elif cmd == "confirm":
        sys.exit(0 if _confirm(rest[0]) else 1)
    elif cmd == "wandb-env":
        _wandb_env(Path(rest[0]), Path(rest[1]))
    else:
        sys.exit(f"[ask] unknown subcommand {cmd!r}")


if __name__ == "__main__":
    main()
