#!/usr/bin/env python3
"""Resolve a per-project workspace into a flat, deduped gitman config, then materialize it.

The hcrl Isaac Lab workspace is composed from independent repos: a shared core (``hcrl_isaaclab``),
the RL package (``robot_rl``), shared robots (``hcrl_robots``), and per-project task + robot repos
(``ssti_tasks``/``ssti_robots``, ``umrl_tasks``, …). Each repo declares only its *direct* deps in a
``dependencies.yaml``; this script walks that graph **transitively**, **dedups by name** (so a dep
shared by two projects is checked out once), detects ref conflicts, and writes a single flat
``resources/gitman.yml`` (everything a sibling under ``resources/``). ``gitman update`` then fetches.

This is the "west-style flat/name-keyed/dedup resolver, gitman as the fetch backend" from the reorg
design: gitman alone would vendor a nested copy of a shared dep per consumer; the dedup pass here is
what guarantees one copy.

Manifest (``workspace.yaml`` at the manager root)::

    org: Creampelt                       # default GitHub org for bare repo names
    isaaclab:
      source: false                      # true → check out IsaacLab source under resources/IsaacLab
      version: "5.1.0"                    # pin for pip mode (informational here)
    projects: [ssti, umrl]               # which project task repos to include (-> <name>_tasks)
    always: [hcrl_isaaclab, robot_rl]     # repos always present (the core + RL package)

``dependencies.yaml`` (in each repo)::

    deps:
      - name: hcrl_isaaclab
        git: git@github.com:Creampelt/hcrl_isaaclab.git   # optional; derived from org + name if omitted
        ref: main
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

MANAGER_DIR = Path(__file__).resolve().parent.parent
RESOURCES = MANAGER_DIR / "resources"


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
    org = manifest.get("org", "Creampelt")
    resolved: dict[str, dict] = {}

    # seed roots: the always-present repos + each selected project's task repo
    roots = list(manifest.get("always", ["hcrl_isaaclab", "robot_rl"]))
    roots += [f"{p}_tasks" for p in manifest.get("projects", [])]

    queue = [{"name": n, "git": None, "ref": "main"} for n in roots]
    while queue:
        dep = queue.pop(0)
        name = dep["name"]
        url = _git_url(name, org, dep.get("git"))
        ref = dep.get("ref", "main")
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
    ap.add_argument("--manifest", default=str(MANAGER_DIR / "workspace.yaml"))
    ap.add_argument("--update", action="store_true", help="Run `gitman update --skip-changes` after writing.")
    args = ap.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    resolved = resolve(manifest)
    gitman_cfg = to_gitman(resolved, manifest)
    out = MANAGER_DIR / "gitman.yml"
    out.write_text(yaml.safe_dump(gitman_cfg, sort_keys=False))
    print(f"[resolve] wrote {out} with {len(resolved)} repos: {', '.join(sorted(resolved))}")
    if manifest.get("isaaclab", {}).get("source"):
        print("[resolve] IsaacLab: source mode (cloned into resources/IsaacLab)")
    else:
        print(f"[resolve] IsaacLab: pip mode (pin {manifest.get('isaaclab', {}).get('version', '?')})")
    if args.update:
        subprocess.run(["gitman", "update", "--skip-changes"], cwd=MANAGER_DIR, check=True)


if __name__ == "__main__":
    main()
