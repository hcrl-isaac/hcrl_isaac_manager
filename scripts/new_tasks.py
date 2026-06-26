#!/usr/bin/env python3
"""Scaffold a new ``<name>_tasks`` extension repo for the hcrl Isaac Lab workspace.

Produces a self-contained task package (mirroring Isaac Lab's external-extension template). Its
``register_task`` calls auto-derive the source from the package name, so ``foo_tasks`` namespaces
under ``foo/`` against the shared ``hcrl_isaaclab`` core with no extra wiring.

    python scripts/new_tasks.py foo            # -> resources/foo_tasks/

Add ``foo`` to ``workspace.yaml``'s ``projects:`` and run ``resolve_workspace.py --update`` to install.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MANAGER_DIR = Path(__file__).resolve().parent.parent


def _files(name: str, org: str) -> dict[str, str]:
    """Return the {relative_path: content} scaffold for a ``<name>_tasks`` repo."""
    pkg = f"{name}_tasks"
    Pkg = "".join(p.capitalize() for p in name.split("_"))
    return {
        "pyproject.toml": (
            "[build-system]\n"
            'requires = ["setuptools>=61"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            "[project]\n"
            f'name = "{pkg}"\n'
            'version = "0.1.0"\n'
            f'description = "{name} task extension for hcrl Isaac Lab"\n'
            'requires-python = ">=3.11"\n'
            'dependencies = ["hcrl_isaaclab"]\n\n'
            "[tool.setuptools.packages.find]\n"
            f'include = ["{pkg}*"]\n'
        ),
        "dependencies.yaml": (
            "# Direct deps of this repo; the manager resolver hoists + dedups these workspace-wide.\n"
            "deps:\n"
            "  - name: hcrl_isaaclab\n"
            "    ref: main\n"
            f"  # - name: {name}_robots   # uncomment if this project has its own robot assets\n"
            "  #   ref: main\n"
        ),
        f"{pkg}/__init__.py": (
            f'"""The {pkg} extension: {name}-project tasks, namespaced under ``{name}/``."""\n\n'
            "from isaaclab_tasks.utils import import_packages\n\n"
            "# import every sub-package so its register_task() calls run (skip helpers/mdp)\n"
            '_BLACKLIST_PKGS = ["utils", ".mdp"]\n'
            "import_packages(__name__, _BLACKLIST_PKGS)\n"
        ),
        f"{pkg}/example/__init__.py": (
            "import gymnasium as gym  # noqa: F401\n\n"
            "from hcrl_isaaclab.tasks_registry import register_task\n\n"
            "# Registers as '" + name + "/Example-v0' (source auto-derived from the package name).\n"
            "register_task(\n"
            '    id="Example-v0",\n'
            '    entry_point="hcrl_isaaclab.envs:ManagerBasedRLEnv",\n'
            "    disable_env_checker=True,\n"
            "    kwargs={\n"
            f'        "env_cfg_entry_point": f"{{__name__}}.example_cfg:{Pkg}ExampleTaskCfg",\n'
            f'        "rsl_rl_cfg_entry_point": f"{{__name__}}.example_cfg:{Pkg}ExamplePPORunnerCfg",\n'
            "    },\n"
            ")\n"
        ),
        f"{pkg}/example/example_cfg.py": (
            '"""Minimal example task — subclass a core base and override per-robot bits here.\n\n'
            "Replace this with real configs; it exists so a freshly-generated repo registers + loads.\n"
            '"""\n\n'
            "# from hcrl_isaaclab.tasks.locomotion.locomotion_env_cfg import LocomotionEnvCfg\n"
            "# from hcrl_isaaclab.tasks.locomotion.g1.ppo_cfg import G1VelocityPPORunnerCfg\n"
            "#\n"
            "# @configclass\n"
            f"# class {Pkg}ExampleTaskCfg(LocomotionEnvCfg): ...\n"
            f"# class {Pkg}ExamplePPORunnerCfg(G1VelocityPPORunnerCfg): ...\n"
        ),
        "README.md": (
            f"# {pkg}\n\n"
            f"`{name}`-project task extension for hcrl Isaac Lab. Tasks register under the `{name}/` "
            "namespace against the shared `hcrl_isaaclab` core.\n\n"
            f"Install via the manager: add `{name}` to `workspace.yaml` `projects:` and run "
            "`python scripts/resolve_workspace.py --update`.\n"
        ),
        ".gitignore": "__pycache__/\n*.pyc\n*.egg-info/\nlogs/\noutputs/\n.artifacts/\n",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="Project name (without the _tasks suffix), e.g. 'foo' -> foo_tasks.")
    ap.add_argument("--org", default="Creampelt")
    ap.add_argument("--dest", default=None, help="Output dir (default: resources/<name>_tasks).")
    args = ap.parse_args()

    name = args.name.removesuffix("_tasks")
    dest = Path(args.dest) if args.dest else MANAGER_DIR / "resources" / f"{name}_tasks"
    if dest.exists():
        raise SystemExit(f"[new-tasks] {dest} already exists; refusing to overwrite.")

    for rel, content in _files(name, args.org).items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    print(f"[new-tasks] scaffolded {dest} ({name}/ namespace).")
    print(f"[new-tasks] next: git init it, push to {args.org}/{name}_tasks, add '{name}' to workspace.yaml projects.")


if __name__ == "__main__":
    main()
