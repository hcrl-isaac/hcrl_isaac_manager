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

MANAGER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve(name: str) -> tuple[dict[str, str], list[str]]:
    """Map each workspace repo to its resolved root for the named worktree set.

    Args:
        name: Worktree-set name, or empty for main checkouts only.

    Returns:
        ``(paths, overridden)``: repo name -> resolved root, and the repos taken from a worktree.
    """
    resources = os.path.join(MANAGER_DIR, "resources")
    repos = ["hcrl_isaaclab", "robot_rl"]
    repos += sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(resources, "*_tasks"))
    )
    paths, overridden = {}, []
    for repo in repos:
        main = os.path.join(resources, repo)
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
        raise SystemExit(
            f"[worktree] no repo has worktrees/{args.name}; nothing to select"
        )
    pypath = ":".join(paths[r] for r in overridden)
    print(f'export WT_PYTHONPATH="{pypath}"')
    print(f'export WT_CORE="{paths["hcrl_isaaclab"]}"')
    if overridden:
        print(
            f'echo "[worktree] {args.name}: {", ".join(overridden)} (others from main checkouts)" >&2'
        )


if __name__ == "__main__":
    main()
