"""Print the Ray ``file_mounts`` YAML block for the resolved workspace package repos.

Each workspace Python package under ``resources/`` maps to ``/workspace/ext/<name>`` in the shared Isaac
image; data-only repos (no setup.py/pyproject) are skipped. In IsaacLab source mode the
``resources/IsaacLab/source/isaaclab*`` packages and ``apps/`` are mounted too, under ``/workspace/IsaacLab`` as in
the checkout: Isaac Lab finds its root only as ``<root>/source/isaaclab/isaaclab`` (its ``paths.py``).

Output is a YAML flow mapping of absolute local paths, injected into the job-config templates via
``envsubst`` (``file_mounts: ${WORKSPACE_FILE_MOUNTS}``).
"""

from __future__ import annotations

import glob
import os
import sys

MANAGER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# per-user state (resources/, workspace.yaml): the main checkout's when run from a manager worktree
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR") or MANAGER_DIR
CONTAINER_EXT = "/workspace/ext"
CONTAINER_ISAACLAB = "/workspace/IsaacLab"

sys.path.insert(0, os.path.join(MANAGER_DIR, "scripts"))
from worktree_env import workspace_repos


def _is_package(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "setup.py")) or os.path.isfile(os.path.join(path, "pyproject.toml"))


def _source_mode() -> bool:
    manifest = os.path.join(WORKSPACE_DIR, "workspace.yaml")
    try:
        with open(manifest) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("mode:"):
                    return stripped.split("#", 1)[0].split(":", 1)[1].strip() == "source"
                if stripped.startswith("source:"):  # legacy `source: bool` form
                    return stripped.split("#", 1)[0].split(":", 1)[1].strip() == "true"
    except OSError:
        pass
    return False


def main() -> None:
    resources = os.path.join(WORKSPACE_DIR, "resources")
    # the same repo list `pls run --wt` selects, so a Ray job ships the same worktree set
    candidates = [os.path.join(resources, repo) for repo in workspace_repos(resources)]
    # WT=<name>: ship the named worktree instead of the main checkout for any repo that has one
    wt = os.environ.get("WT", "")
    if wt:
        # (source path, repo name): the container path and dedup key are the repo name, since every
        # repo's worktree shares the worktree-set basename
        resolved = []
        for c in candidates:
            wdir = os.path.join(c, "worktrees", wt)
            resolved.append((wdir if os.path.isdir(wdir) else c, os.path.basename(c)))
        if all(src == c for (src, _), c in zip(resolved, candidates, strict=False)):
            raise SystemExit(f"[file_mounts] WT={wt!r} matches no resources/<repo>/worktrees/{wt}")
        candidates = resolved
    else:
        candidates = [(c, os.path.basename(c)) for c in candidates]
    seen: set[str] = set()
    lines = ["{"]
    for path, name in candidates:
        if name in seen or not os.path.isdir(path) or not _is_package(path):
            continue
        seen.add(name)
        lines.append(f'  "{path}": "{CONTAINER_EXT}/{name}",')
    # source mode: IsaacLab's source packages (over the baked pip isaaclab) and its apps/, in the checkout's layout
    if _source_mode():
        checkout = os.path.join(resources, "IsaacLab")
        lines.extend(
            f'  "{path}": "{CONTAINER_ISAACLAB}/source/{os.path.basename(path)}",'
            for path in sorted(glob.glob(os.path.join(checkout, "source", "isaaclab*")))
            if os.path.isdir(path) and _is_package(path)
        )
        if os.path.isdir(os.path.join(checkout, "apps")):
            lines.append(f'  "{os.path.join(checkout, "apps")}": "{CONTAINER_ISAACLAB}/apps",')
    lines.append("}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
