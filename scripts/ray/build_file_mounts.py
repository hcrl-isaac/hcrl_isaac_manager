"""Print the Ray ``file_mounts`` YAML block for the resolved workspace package repos.

Each workspace Python package under ``resources/`` (hcrl_isaaclab, robot_rl, the ``*_tasks`` and any
packaged ``*_robots``) is mounted into the cluster container's Isaac Lab source tree at
``/workspace/isaaclab/source/<name>`` (the container is always source-layout, regardless of whether
IsaacLab is installed from source or pip locally). Data-only repos (no setup.py/pyproject, e.g.
``hcrl_robots``) are skipped -- their bulk files are fetched at runtime as W&B artifacts.

Emitted as a YAML flow mapping with absolute local paths so it can be injected verbatim into the
job-config templates via ``envsubst`` (``file_mounts: ${WORKSPACE_FILE_MOUNTS}``).
"""

from __future__ import annotations

import glob
import os

MANAGER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTAINER_SRC = "/workspace/isaaclab/source"


def _is_package(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "setup.py")) or os.path.isfile(os.path.join(path, "pyproject.toml"))


def main() -> None:
    resources = os.path.join(MANAGER_DIR, "resources")
    # Deterministic order: core, RL package, then task/robot packages.
    candidates = [
        os.path.join(resources, "hcrl_isaaclab"),
        os.path.join(resources, "robot_rl"),
        *sorted(glob.glob(os.path.join(resources, "*_tasks"))),
        *sorted(glob.glob(os.path.join(resources, "*_robots"))),
    ]
    seen: set[str] = set()
    lines = ["{"]
    for path in candidates:
        name = os.path.basename(path)
        if name in seen or not os.path.isdir(path) or not _is_package(path):
            continue
        seen.add(name)
        lines.append(f'  "{path}": "{CONTAINER_SRC}/{name}",')
    lines.append("}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
