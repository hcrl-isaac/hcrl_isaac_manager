#!/usr/bin/env python3
"""Rich output helpers for the justfile: styled section headers + a spinner around opaque steps.

Subcommands:
    section <title>             Print a styled rule to delineate a phase of a multi-step recipe.
    spin <label> -- <cmd>...     Run <cmd> under an animated spinner, hiding its output on success and
                                 dumping it on failure. With no TTY (CI/pipes) it runs transparently so
                                 the full output is preserved. Exits with the command's status.

Only wrap genuinely opaque/quiet steps (e.g. the headless Isaac Sim boot) -- tools that stream their
own progress (uv, gitman, docker) should keep their native output.
"""

from __future__ import annotations

import subprocess
import sys

from rich.console import Console

console = Console()


def _spin(label: str, cmd: list[str]) -> int:
    if not cmd:
        console.print("[red][ui] spin needs a command after `--`[/]")
        return 2
    if not (sys.stdout.isatty() and sys.stderr.isatty()):
        console.print(f"[cyan]▶[/] {label}")
        return subprocess.run(cmd).returncode

    with console.status(f"[cyan]{label}…[/]", spinner="dots"):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        console.print(f"[green]✓[/] {label}")
    else:
        console.print(f"[red]✗[/] {label} (exit {proc.returncode})")
        for stream in (proc.stdout, proc.stderr):
            if stream and stream.strip():
                console.print(stream.rstrip())
    return proc.returncode


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: ui.py <section|spin> ...")
    cmd, rest = args[0], args[1:]
    if cmd == "section":
        console.rule(f"[bold cyan]{' '.join(rest)}", align="left")
    elif cmd == "spin":
        if "--" not in rest:
            sys.exit("usage: ui.py spin <label> -- <cmd>...")
        sep = rest.index("--")
        sys.exit(_spin(" ".join(rest[:sep]), rest[sep + 1:]))
    else:
        sys.exit(f"[ui] unknown subcommand {cmd!r}")


if __name__ == "__main__":
    main()
