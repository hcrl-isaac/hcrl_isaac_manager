"""Print the Ray ``file_mounts`` YAML block for the resolved workspace package repos.

Each workspace Python package under ``resources/`` (hcrl_isaaclab, robot_rl, the ``*_tasks`` and any
packaged ``*_robots``) is uploaded to the Ray cluster as a py_module and placed at ``/workspace/ext/<name>``
inside the shared Isaac image, then editable/PYTHONPATH'd there. The image bakes isaacsim (nvcr base) +
Isaac Lab (pip), so nothing from the IsaacLab source tree is required. Data-only repos (no setup.py/
pyproject, e.g. ``hcrl_robots``) are skipped -- their bulk files are fetched at runtime as W&B artifacts.

Source-mode overlay: when ``workspace.yaml`` has ``isaaclab.source: true``, the local IsaacLab source
packages (``resources/IsaacLab/source/isaaclab*``) are mounted alongside so they land on the worker
PYTHONPATH ahead of the baked pip isaaclab, overriding it. Pip mode emits none of these, so the baked
pip isaaclab is used.

Emitted as a YAML flow mapping with absolute local paths so it can be injected verbatim into the
job-config templates via ``envsubst`` (``file_mounts: ${WORKSPACE_FILE_MOUNTS}``).
"""

from __future__ import annotations

import glob
import os

MANAGER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTAINER_EXT = "/workspace/ext"


def _is_package(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "setup.py")) or os.path.isfile(os.path.join(path, "pyproject.toml"))


def _source_mode() -> bool:
    manifest = os.path.join(MANAGER_DIR, "workspace.yaml")
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
    resources = os.path.join(MANAGER_DIR, "resources")
    # Deterministic order: core, RL package, then task/robot packages.
    candidates = [
        os.path.join(resources, "hcrl_isaaclab"),
        os.path.join(resources, "robot_rl"),
        *sorted(glob.glob(os.path.join(resources, "*_tasks"))),
        *sorted(glob.glob(os.path.join(resources, "*_robots"))),
    ]
    # Source-mode overlay: editable IsaacLab source packages override the baked pip isaaclab.
    if _source_mode():
        candidates += sorted(glob.glob(os.path.join(resources, "IsaacLab", "source", "isaaclab*")))

    seen: set[str] = set()
    lines = ["{"]
    for path in candidates:
        name = os.path.basename(path)
        if name in seen or not os.path.isdir(path) or not _is_package(path):
            continue
        seen.add(name)
        lines.append(f'  "{path}": "{CONTAINER_EXT}/{name}",')
    lines.append("}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
