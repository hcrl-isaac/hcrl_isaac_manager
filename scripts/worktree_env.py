"""Resolve a named worktree set to per-repo paths.

Each resource repo may keep worktrees under ``resources/<repo>/worktrees/<name>``. A worktree SET is
every repo's ``<name>`` worktree where one exists, with the main checkout as the per-repo fallback --
so one name selects a coherent cross-repo feature line. Prints shell exports:

  WT_PYTHONPATH  colon list of the worktree'd package roots (prepend to PYTHONPATH: it precedes the
                 editable installs of the main checkouts, so only diverging repos are overridden)
  WT_CORE        the hcrl_isaaclab root whose scripts/ should be used

With no name (or an empty one) both fall back to the main checkouts, so callers can eval it
unconditionally.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

MANAGER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESOURCES_DIR = os.path.join(MANAGER_DIR, "resources")


def workspace_repos(resources: str = RESOURCES_DIR) -> list[str]:
    """Name the repos that form a worktree set, in a fixed order: core, RL package, ``*_tasks``, ``*_robots``.

    Args:
        resources: The workspace ``resources/`` directory to glob the per-project repos from.

    Returns:
        Repo directory names (not checked for existence), limited to the ``gitman.yaml`` sources when it exists.
    """
    listed = _gitman_names(os.path.join(os.path.dirname(os.path.abspath(resources)), "gitman.yaml"))
    repos = ["hcrl_isaaclab", "robot_rl"]
    for pattern in ("*_tasks", "*_robots"):
        found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(resources, pattern)))
        for r in found:
            if listed is None or r in listed:
                repos.append(r)
            else:
                print(
                    f"[worktree_env] skipping resources/{r}: not in gitman.yaml (run `just resolve`?)", file=sys.stderr
                )
    return repos


def _gitman_names(path: str) -> set[str] | None:
    """Repo names in the resolved ``gitman.yaml`` (a retired checkout left in resources/ is unlisted), or None."""
    try:
        with open(path) as f:
            lines = [line.strip().removeprefix("- ") for line in f]
        return {line.split(":", 1)[1].strip() for line in lines if line.startswith("name:")}
    except OSError:
        return None


def resolve(name: str) -> tuple[dict[str, str], list[str]]:
    """Map each workspace repo to its resolved root for the named worktree set.

    Args:
        name: Worktree-set name, or empty for main checkouts only.

    Returns:
        ``(paths, overridden)``: repo name -> resolved root, and the repos taken from a worktree.
    """
    paths, overridden = {}, []
    for repo in workspace_repos():
        main = os.path.join(RESOURCES_DIR, repo)
        if not os.path.isdir(main):
            continue
        wt = os.path.join(main, "worktrees", name) if name else ""
        if name and os.path.isdir(wt):
            paths[repo] = wt
            overridden.append(repo)
        else:
            paths[repo] = main
    return paths, overridden


def main() -> None:
    """Print eval-able exports for the requested worktree set."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "name",
        nargs="?",
        default="",
        help="Worktree-set name (empty = main checkouts).",
    )
    args = ap.parse_args()
    paths, overridden = resolve(args.name)
    if args.name and not overridden:
        raise SystemExit(f"[worktree] no repo has worktrees/{args.name}; nothing to select")
    pypath = ":".join(paths[r] for r in overridden)
    print(f'export WT_PYTHONPATH="{pypath}"')
    print(f'export WT_CORE="{paths["hcrl_isaaclab"]}"')
    if overridden:
        print(f'echo "[worktree] {args.name}: {", ".join(overridden)} (others from main checkouts)" >&2')


if __name__ == "__main__":
    main()
