"""Generate ``.vscode/settings.json`` for the workspace (run from ``just vscode`` / end of ``just setup``).

Writes the three search-root families Pylance cannot discover from the interpreter alone, as
``python.analysis.extraPaths``:

1. **Isaac Sim extensions.** ``python -m isaaclab --generate-vscode-settings`` does ``import
   omni.kit_app``, which is not importable under a pip-installed isaacsim (Kit loads it dynamically).
   So we boot a headless ``SimulationApp`` -- the path that *does* work in both source and pip mode --
   and snapshot the extension dirs. This is the slow part; ``--no-kit`` reuses the last snapshot.
2. **pip-isaaclab packages.** ``site-packages/isaaclab`` is a Kit-bootstrapping shim with no
   submodules; the importable trees are under its ``source/``.
3. **Editable workspace repos.** ``uv pip install -e`` installs them as PEP 660 import-hook finders,
   which only run at import time and so are invisible to static analysis.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import sysconfig
from pathlib import Path

MANAGER_DIR = Path(__file__).resolve().parents[2]


def _kit_ext_paths() -> tuple[list[str], object]:
    """Boot headless Kit and snapshot its extension dirs. Returns the app too -- closing it exits."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaacsim import SimulationApp  # noqa: PLC0415 -- import after the EULA env is set

    # Boot Kit so the extension layout is materialized, then glob the ext dirs Pylance needs (Kit loads
    # most extensions via its own finder, not sys.path, so a sys.path snapshot would miss them).
    app = SimulationApp({"headless": True})
    import isaacsim  # noqa: PLC0415

    isaacsim_path = Path(isaacsim.__file__).resolve().parent
    kit_path = isaacsim_path / "kit"

    ext_dirs: set[str] = set()
    for base in (isaacsim_path, kit_path):
        for sub in ("exts", "extscore", "extscache", "extsDeprecated", "extsUser"):
            d = base / sub
            if d.is_dir():
                ext_dirs.update(str(c) for c in d.iterdir() if c.is_dir())
    kernel_py = kit_path / "kernel" / "py"
    if kernel_py.is_dir():
        ext_dirs.add(str(kernel_py))
    return sorted(ext_dirs), app


def _package_paths() -> list[str]:
    """Roots for pip-isaaclab's real package trees and every editable workspace install."""
    site_packages = Path(sysconfig.get_paths()["purelib"])
    paths: set[str] = set()

    # source/<name>/<name>/ -- the parent is the search root that makes `import <name>` resolve
    for src in (site_packages / "isaaclab" / "source").glob("*"):
        if (src / src.name).is_dir():
            paths.add(str(src))

    for finder in site_packages.glob("__editable___*_finder.py"):
        match = re.search(r"MAPPING[^=]*=\s*(\{[^}]*\})", finder.read_text())
        if match:
            paths.update(str(Path(target).parent) for target in ast.literal_eval(match.group(1)).values())
    return sorted(paths)


def _cached_kit_paths(settings_path: Path) -> list[str]:
    """Kit dirs from a previous run, so --no-kit can refresh the cheap paths without a 1-min boot."""
    if not settings_path.is_file():
        raise SystemExit(f"[vscode] --no-kit needs an existing {settings_path}; run `just vscode` once first")
    text = re.sub(r"^\s*//.*$", "", settings_path.read_text(), flags=re.M)
    stale = (str(Path(sysconfig.get_paths()["purelib"]) / "isaaclab" / "source"), str(MANAGER_DIR / "resources"))
    return [p for p in json.loads(text).get("python.analysis.extraPaths", []) if not p.startswith(stale)]


def main() -> None:
    settings_path = MANAGER_DIR / ".vscode" / "settings.json"
    app = None
    if "--no-kit" in sys.argv[1:]:
        ext_paths = _cached_kit_paths(settings_path)
    else:
        ext_paths, app = _kit_ext_paths()
    extra_paths = _package_paths() + ext_paths

    # NOTE: SimulationApp.close() terminates the process, so write the file *before* closing.
    venv_python = MANAGER_DIR / "ilab" / "bin" / "python"
    settings = {
        "editor.rulers": [120],
        "python.languageServer": "Pylance",
        "python.defaultInterpreterPath": str(venv_python),
        "python.analysis.extraPaths": extra_paths,
        "python.analysis.typeCheckingMode": "basic",
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length", "120"],
        # Index all of resources/ (gitman subrepos, gitignored by the manager) in Ctrl-P / search.
        "search.useIgnoreFiles": False,
        "search.exclude": {
            "**/.git": True, "**/ilab": True, "**/.venv": True, "**/__pycache__": True, "**/*.pyc": True,
            "**/logs": True, "**/wandb": True, "**/outputs": True, "**/.pytest_cache": True,
            "**/*.sif": True, "**/scripts/cluster/exports": True,
        },
        "files.watcherExclude": {
            "**/ilab/**": True, "**/.venv/**": True, "**/.git/**": True, "**/logs/**": True, "**/wandb/**": True,
        },
        # Detect each gitman subrepo under resources/ (depth 2) so SCM shows per-subrepo git status.
        "git.autoRepositoryDetection": True,
        "git.repositoryScanMaxDepth": 3,
        "git.detectSubmodules": False,
    }

    settings_path.parent.mkdir(exist_ok=True)
    header = (
        "// Auto-generated by scripts/tools/setup_vscode.py (just vscode). Do not edit by hand;\n"
        "// re-run `just vscode` to refresh the Isaac Sim extension paths, or `just vscode --no-kit`\n"
        "// to refresh only the workspace package paths (no Isaac Sim boot).\n"
    )
    settings_path.write_text(header + json.dumps(settings, indent=4) + "\n")
    print(f"[vscode] wrote {settings_path} with {len(extra_paths)} search paths")

    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
