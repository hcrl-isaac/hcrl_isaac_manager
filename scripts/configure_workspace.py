#!/usr/bin/env python3
"""Interactively select which projects + IsaacLab mode this workspace composes.

Reads the committed catalog (``workspace.defaults.yaml``) and writes the per-user selection to a gitignored
``workspace.yaml`` (``projects`` + ``isaaclab.mode``: source/none), which ``resolve_workspace.py`` merges. The picker
keeps the file's other keys and its own-repo ``projects`` entries (``{name, git, ref?}``).

Modes:
    (default)      Write the catalog defaults this user can read if ``workspace.yaml`` is absent, else leave it.
                   Never prompts.
    --interactive  On a TTY, open the picker pre-filled with the current selection. With no TTY, keep the
                   existing selection, or write catalog defaults if none exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from resolve_workspace import RENAMED_PROJECTS, _git_url, project_name, reachable

MANAGER_DIR = Path(__file__).resolve().parent.parent
DEFAULTS = MANAGER_DIR / "workspace.defaults.yaml"
OUT = MANAGER_DIR / "workspace.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {} if path.is_file() else {}


# Isaac Lab 3 has no pip release: a source install, or none at all
MODES = ("source", "none")


def _default_mode(defaults: dict) -> str:
    """Default IsaacLab mode from the catalog: its ``mode`` when it is one of MODES, else ``source``."""
    mode = defaults.get("isaaclab", {}).get("mode")
    return mode if mode in MODES else "source"


def _current_mode(current: dict, defaults: dict) -> str:
    """The user's current IsaacLab mode, else the default. A retired mode (pip, or the legacy ``source: bool`` form)
    reads as ``source``."""
    il = current.get("isaaclab", {})
    if "mode" in il or "source" in il:
        return il["mode"] if il.get("mode") in MODES else "source"
    return _default_mode(defaults)


def _readable_defaults(defaults: dict) -> list[str]:
    """The catalog's ``default: true`` projects whose task repo this user can read, warning about each other one."""
    org = defaults.get("org", "hcrl-isaac")
    readable = []
    for p in defaults.get("available_projects", []):
        if not p.get("default"):
            continue
        url = _git_url(f"{p['name']}_tasks", org, None)
        if reachable(url, "main"):
            readable.append(p["name"])
        else:
            print(f"[configure] WARNING: leaving out default project {p['name']!r}: no read access to {url}")
    return readable


def _default_selection(defaults: dict) -> dict:
    """The selection used with no user input: readable catalog ``default: true`` projects + default IsaacLab mode."""
    return {"projects": _readable_defaults(defaults), "isaaclab": {"mode": _default_mode(defaults)}}


_HEADER = (
    "# Per-user workspace selection (gitignored). Written by `pls setup` (or `pls setup reconfigure`).\n"
    "# Merged with the committed workspace.defaults.yaml by resolve_workspace.py.\n"
)


def _write(selection: dict) -> None:
    OUT.write_text(_HEADER + yaml.safe_dump(selection, sort_keys=False))
    proj = ", ".join(project_name(p) for p in selection["projects"]) or "(none)"
    print(f"[configure] wrote {OUT.name}: projects=[{proj}], isaaclab={selection['isaaclab']['mode']}")


def _migrate_renamed(current: dict) -> dict:
    """Rewrite renamed project names and ``<old>_tasks`` ref pins in a saved selection, saving it if changed."""
    if not current:
        return current
    projects = current.get("projects") or []
    projects = [projects] if isinstance(projects, str) else projects
    refs = current.get("refs") or {}
    renamed = [p for p in projects if isinstance(p, str) and p in RENAMED_PROJECTS]  # own-repo entries keep theirs
    notes = [f"project {p!r} -> {RENAMED_PROJECTS[p]!r}" for p in renamed]
    notes += [f"refs pin {o}_tasks -> {n}_tasks" for o, n in RENAMED_PROJECTS.items() if f"{o}_tasks" in refs]
    if not notes:
        return current
    current["projects"] = [RENAMED_PROJECTS[p] if p in renamed else p for p in projects]
    if refs:
        current["refs"] = {_renamed_repo(k): v for k, v in refs.items()}
    OUT.write_text(_HEADER + yaml.safe_dump(current, sort_keys=False))
    print(f"[configure] renamed in {OUT.name}: {'; '.join(notes)}")
    return current


def _renamed_repo(name: str) -> str:
    """The current name of a ``<project>_tasks`` repo."""
    old = name.removesuffix("_tasks")
    return f"{RENAMED_PROJECTS[old]}_tasks" if name.endswith("_tasks") and old in RENAMED_PROJECTS else name


def _prompt(defaults: dict, current: dict) -> dict:
    """Arrow-key/space selector for projects + IsaacLab mode, pre-filling ``current`` and keeping its other keys."""
    import questionary

    catalog = defaults.get("available_projects", [])
    saved = current.get("projects", [p["name"] for p in catalog if p.get("default")])
    saved = [saved] if isinstance(saved, str) else saved
    cur_projects = {project_name(p) for p in saved}
    # own-repo entries are offered (checked) with their URL, in place of a catalog project of the same name
    own = {p["name"]: p for p in saved if isinstance(p, dict) and p.get("git")}
    choices = [
        questionary.Choice(
            title=f"{p['name']:<8} {p.get('description', '')}".rstrip(),
            value=p["name"],
            checked=p["name"] in cur_projects,
        )
        for p in catalog
        if p["name"] not in own
    ]
    choices += [questionary.Choice(title=f"{n:<8} {p['git']}", value=n, checked=True) for n, p in own.items()]
    projects = questionary.checkbox(
        "Select projects to install (space to toggle, enter to confirm):", choices=choices
    ).ask()
    if projects is None:  # Ctrl-C / EOF
        sys.exit("[configure] cancelled")

    mode = questionary.select(
        "IsaacLab:",
        choices=[
            questionary.Choice("source  (clone IsaacLab under resources/ and install editable)", value="source"),
            questionary.Choice("none    (don't install IsaacLab; just fetch the project repos)", value="none"),
        ],
        default=_current_mode(current, defaults),
    ).ask()
    if mode is None:
        sys.exit("[configure] cancelled")

    isaaclab = {k: v for k, v in current.get("isaaclab", {}).items() if k != "source"}  # `source:` is retired
    return {**current, "projects": [own.get(n, n) for n in projects], "isaaclab": {**isaaclab, "mode": mode}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="On a TTY, (re)open the picker pre-filled with the current selection.",
    )
    args = ap.parse_args()

    defaults = _load(DEFAULTS)
    if not defaults:
        sys.exit(f"[configure] missing catalog: {DEFAULTS}")
    current = _migrate_renamed(_load(OUT))

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
