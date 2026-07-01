#!/usr/bin/env python3
"""Interactively select which projects + IsaacLab mode this workspace composes.

Reads the committed catalog (``workspace.defaults.yaml``), prompts the user (arrow keys + space to
toggle projects, enter to confirm), and writes the per-user selection to a gitignored
``workspace.yaml`` (just ``projects`` + ``isaaclab.source``). ``resolve_workspace.py`` merges the two.

Modes:
    (default, "ensure")  Write a selection from the catalog defaults if ``workspace.yaml`` is absent;
                         otherwise leave it untouched. Never prompts -- safe for ``just resolve``/CI.
    --interactive        On a TTY, always (re)open the picker pre-filled with the current selection (so a
                         plain ``just setup`` reconfigures). With no TTY, keep the existing selection, or
                         write catalog defaults if none exists. Used by ``just setup``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

MANAGER_DIR = Path(__file__).resolve().parent.parent
DEFAULTS = MANAGER_DIR / "workspace.defaults.yaml"
OUT = MANAGER_DIR / "workspace.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {} if path.is_file() else {}


def _default_selection(defaults: dict) -> dict:
    """The selection used with no user input: catalog ``default: true`` projects + default IsaacLab mode."""
    projects = [p["name"] for p in defaults.get("available_projects", []) if p.get("default")]
    return {"projects": projects, "isaaclab": {"source": bool(defaults.get("isaaclab", {}).get("source", False))}}


def _write(selection: dict) -> None:
    header = (
        "# Per-user workspace selection (gitignored). Written by `just setup` (or `just setup reconfigure`).\n"
        "# Merged with the committed workspace.defaults.yaml by resolve_workspace.py.\n"
    )
    OUT.write_text(header + yaml.safe_dump(selection, sort_keys=False))
    proj = ", ".join(selection["projects"]) or "(none)"
    mode = "source" if selection["isaaclab"]["source"] else "pip"
    print(f"[configure] wrote {OUT.name}: projects=[{proj}], isaaclab={mode}")


def _prompt(defaults: dict, current: dict) -> dict:
    """Arrow-key/space selector for projects + IsaacLab mode, pre-filling ``current``."""
    import questionary

    catalog = defaults.get("available_projects", [])
    cur_projects = set(current.get("projects", [p["name"] for p in catalog if p.get("default")]))
    choices = [
        questionary.Choice(
            title=f"{p['name']:<8} {p.get('description', '')}".rstrip(),
            value=p["name"],
            checked=p["name"] in cur_projects,
        )
        for p in catalog
    ]
    projects = questionary.checkbox("Select projects to install (space to toggle, enter to confirm):", choices=choices).ask()
    if projects is None:  # Ctrl-C / EOF
        sys.exit("[configure] cancelled")

    cur_source = current.get("isaaclab", {}).get("source", defaults.get("isaaclab", {}).get("source", False))
    source = questionary.select(
        "Install IsaacLab from:",
        choices=[
            questionary.Choice("pip     (isaacsim + isaaclab wheels)", value=False),
            questionary.Choice("source  (clone IsaacLab under resources/)", value=True),
        ],
        default=cur_source,
    ).ask()
    if source is None:
        sys.exit("[configure] cancelled")

    return {"projects": projects, "isaaclab": {"source": bool(source)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interactive", action="store_true",
                    help="On a TTY, (re)open the picker pre-filled with the current selection.")
    args = ap.parse_args()

    defaults = _load(DEFAULTS)
    if not defaults:
        sys.exit(f"[configure] missing catalog: {DEFAULTS}")
    current = _load(OUT)

    if args.interactive and sys.stdin.isatty():  # setup on a TTY -> always (re)prompt, pre-filled
        _write(_prompt(defaults, current or _default_selection(defaults)))
    elif OUT.is_file():  # ensure, or no TTY -> keep the existing selection untouched
        if args.interactive:
            print(f"[configure] no TTY; keeping existing {OUT.name}.")
    else:  # no selection yet -> catalog defaults
        print(f"[configure] {OUT.name} absent; writing catalog defaults.")
        _write(_default_selection(defaults))


if __name__ == "__main__":
    main()
