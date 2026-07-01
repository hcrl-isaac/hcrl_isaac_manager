#!/usr/bin/env python3
"""Resolve a per-project workspace into a flat, deduped gitman config, then materialize it.

Each repo (core ``hcrl_isaaclab``, ``robot_rl``, shared ``hcrl_robots``, per-project ``*_tasks`` /
``*_robots``) declares only its *direct* deps in a ``dependencies.yaml``; this resolves them into one
flat ``gitman.yaml`` (all repos siblings under ``resources/``) for ``gitman update`` to fetch.

The dedup-by-name pass is the reason this exists rather than plain gitman: gitman alone vendors a
nested copy of a shared dep per consumer, so a dep two projects share would be checked out twice.

The effective manifest is the committed ``workspace.defaults.yaml`` (org, always, refs, isaaclab
version, and the ``available_projects`` catalog) overlaid with a per-user, gitignored
``workspace.yaml`` (just ``projects`` + ``isaaclab.source``, written by ``configure_workspace.py``).
When no per-user file exists, the catalog's ``default: true`` projects are used.

Defaults (``workspace.defaults.yaml``)::

    org: hcrl-isaac                      # default GitHub org for bare repo names
    isaaclab:
      source: false                      # true -> check out IsaacLab source under resources/IsaacLab
      version: "5.1.0"                    # pin for pip mode (informational here)
    always: [hcrl_isaaclab, robot_rl]     # repos always present (the core + RL package)
    available_projects:                  # selectable <name>_tasks repos
      - {name: ssti, default: true}

``dependencies.yaml`` (in each repo)::

    deps:
      - name: hcrl_isaaclab
        git: git@github.com:hcrl-isaac/hcrl_isaaclab.git  # optional; derived from org + name if omitted
        ref: main
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

MANAGER_DIR = Path(__file__).resolve().parent.parent
RESOURCES = MANAGER_DIR / "resources"
DEFAULTS = MANAGER_DIR / "workspace.defaults.yaml"


def load_manifest(overrides_path: Path) -> dict:
    """Merge the committed defaults with a per-user selection into one flat manifest.

    Args:
        overrides_path: Per-user ``workspace.yaml`` (``projects`` + ``isaaclab.source``); may be absent.

    Returns:
        A manifest with ``org``, ``always``, ``refs``, ``isaaclab`` and a flat ``projects`` name list --
        the shape ``resolve()``/``to_gitman()`` consume.
    """
    defaults = yaml.safe_load(DEFAULTS.read_text()) if DEFAULTS.is_file() else {}
    defaults = defaults or {}
    overrides = (yaml.safe_load(overrides_path.read_text()) or {}) if overrides_path.is_file() else {}

    catalog = defaults.get("available_projects", [])
    projects = overrides.get("projects")
    if projects is None:  # no per-user selection -> catalog defaults
        projects = [p["name"] for p in catalog if p.get("default")]

    isaaclab = {**defaults.get("isaaclab", {}), **overrides.get("isaaclab", {})}
    return {
        "org": overrides.get("org", defaults.get("org", "hcrl-isaac")),
        "always": defaults.get("always", ["hcrl_isaaclab", "robot_rl"]),
        "refs": {**defaults.get("refs", {}), **overrides.get("refs", {})},
        "isaaclab": isaaclab,
        "projects": projects,
    }


def _git_url(name: str, org: str, explicit: str | None) -> str:
    """Return the git URL for a dep: the explicit one, else ``git@github.com:<org>/<name>.git``."""
    return explicit or f"git@github.com:{org}/{name}.git"


def _read_deps(repo_dir: Path) -> list[dict]:
    """Read a repo's ``dependencies.yaml`` ``deps`` list (empty if the file or key is absent)."""
    f = repo_dir / "dependencies.yaml"
    if not f.is_file():
        return []
    data = yaml.safe_load(f.read_text()) or {}
    return data.get("deps", []) or []


def resolve(manifest: dict) -> dict[str, dict]:
    """Resolve the workspace manifest to a flat ``{name: {git, ref}}`` map (deduped, conflict-checked).

    Args:
        manifest: The parsed ``workspace.yaml``.

    Returns:
        A name-keyed map of every repo to check out, with its git URL and ref.

    Raises:
        SystemExit: Two repos request the same dependency name at different refs (unresolved conflict).
    """
    org = manifest.get("org", "hcrl-isaac")
    # Per-repo ref overrides (workspace.yaml `refs: {name: ref}`) win over the default "main" and over
    # any ref a dependency declares -- so e.g. `refs: {hcrl_isaaclab: feature/reorg-core}` pins the core
    # without tripping the conflict check when a dep requests it at main.
    refs = manifest.get("refs", {}) or {}
    resolved: dict[str, dict] = {}

    # seed roots: the always-present repos + each selected project's task repo
    roots = list(manifest.get("always", ["hcrl_isaaclab", "robot_rl"]))
    roots += [f"{p}_tasks" for p in manifest.get("projects", [])]

    queue = [{"name": n, "git": None, "ref": refs.get(n, "main")} for n in roots]
    while queue:
        dep = queue.pop(0)
        name = dep["name"]
        url = _git_url(name, org, dep.get("git"))
        ref = refs.get(name, dep.get("ref", "main"))
        if name in resolved:
            if resolved[name]["ref"] != ref:
                sys.exit(
                    f"[resolve] dependency conflict: {name!r} requested at both "
                    f"{resolved[name]['ref']!r} and {ref!r}. Pin one ref in workspace.yaml."
                )
            continue
        resolved[name] = {"git": url, "ref": ref}
        # recurse into this repo's declared deps (if it is already checked out)
        for sub in _read_deps(RESOURCES / name):
            queue.append(sub)
    return resolved


def to_gitman(resolved: dict[str, dict], manifest: dict) -> dict:
    """Render the resolved map as a flat gitman config (everything a sibling under ``resources/``)."""
    sources = []
    if manifest.get("isaaclab", {}).get("source"):
        sources.append({"repo": "https://github.com/isaac-sim/IsaacLab.git", "name": "IsaacLab", "rev": "main"})
    for name, info in sorted(resolved.items()):
        sources.append({"repo": info["git"], "name": name, "rev": info["ref"]})
    return {"location": "resources", "sources": sources, "default_group": "", "groups": []}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANAGER_DIR / "workspace.yaml"),
                    help="Per-user selection overlaid on workspace.defaults.yaml (may be absent).")
    ap.add_argument("--update", action="store_true", help="Run `gitman update --skip-changes` after writing.")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    out = MANAGER_DIR / "gitman.yaml"
    (MANAGER_DIR / "gitman.yml").unlink(missing_ok=True)  # drop a stale .yml so it can't shadow .yaml
    is_source = bool(manifest.get("isaaclab", {}).get("source"))

    # In pip mode, remove an orphaned source clone left by a previous source-mode resolve so
    # downstream source-vs-pip detection (and editable installs) don't pick it up.
    if not is_source:
        isaaclab_dir = RESOURCES / "IsaacLab"
        if isaaclab_dir.exists():
            print(f"[resolve] pip mode: removing orphaned IsaacLab source clone at {isaaclab_dir}")
            shutil.rmtree(isaaclab_dir)

    # A repo's transitive deps are in its own dependencies.yaml, readable only once it's checked out --
    # so resolve -> gitman update -> re-resolve to a fixpoint. Without --update, write the first pass + stop.
    prev_names: set[str] | None = None
    while True:
        resolved = resolve(manifest)
        out.write_text(yaml.safe_dump(to_gitman(resolved, manifest), sort_keys=False))
        names = set(resolved)
        print(f"[resolve] wrote {out} with {len(resolved)} repos: {', '.join(sorted(resolved))}")
        if not args.update or names == prev_names:
            break
        subprocess.run(["gitman", "update", "--skip-changes"], cwd=MANAGER_DIR, check=True)
        prev_names = names

    if is_source:
        print("[resolve] IsaacLab: source mode (cloned into resources/IsaacLab)")
    else:
        print(f"[resolve] IsaacLab: pip mode (pin {manifest.get('isaaclab', {}).get('version', '?')})")


if __name__ == "__main__":
    main()
