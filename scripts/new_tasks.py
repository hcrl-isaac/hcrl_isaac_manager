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
            '"""Minimal example task -- subclass a core base and override per-robot bits here.\n\n'
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
            f"`{name}` task extension for the hcrl Isaac Lab workspace. Tasks register under the "
            f"`{name}/` source namespace (via `hcrl_isaaclab.tasks_registry.register_task`) and run "
            "against the shared `hcrl_isaaclab` core, coexisting with the other project task packages.\n\n"
            "## Domains\n\n"
            "- `example/` -- replace with this project's task domains.\n\n"
            "## Install\n\n"
            f"Normally installed as part of the workspace via the manager: add `{name}` to "
            "`workspace.yaml`'s `projects:` and run `just setup`. Standalone, with core already in the "
            "env: `pip install -e .`.\n\n"
            "## Run\n\n"
            "```bash\n"
            f"python scripts/train.py --task <Task-id> --source {name}   # --source only needed on a shared id\n"
            "```\n\n"
            "## Dependencies\n\n"
            f"Core `hcrl_isaaclab` (and `{name}_robots` if this project has its own robots) -- see "
            "`dependencies.yaml`.\n"
        ),
        ".github/PULL_REQUEST_TEMPLATE.md": (
            "## Summary\n\n"
            "<!-- What changed and why. -->\n\n"
            "## Checklist\n\n"
            f"- [ ] Changes fit within this repo's scope -- {name} tasks under the `{name}/` namespace. "
            "Shared infra/robots belong upstream (`hcrl_isaaclab` / `hcrl_robots`).\n"
            "- [ ] Ran the GPU test suite locally (`pytest -m gpu`). CI runs only the CPU build tests + lint.\n"
            "- [ ] Added tests for any new functionality not already covered by the registration/build "
            "smoke (usually unnecessary -- the smoke tests cover every registered task automatically).\n"
        ),
        "pytest.ini": "[pytest]\ntestpaths = tests\nmarkers =\n    gpu: heavier check that instantiates env cfgs (needs a GPU)\n",
        "tests/conftest.py": (
            '"""Pytest fixtures: launch Isaac Sim once per session (skip when absent)."""\n\n'
            "import pytest\n\n"
            "from hcrl_isaaclab import testing\n\n\n"
            "def pytest_configure(config):\n"
            '    config.addinivalue_line("markers", "gpu: heavier check that instantiates env cfgs (needs a GPU)")\n\n\n'
            '@pytest.fixture(scope="session")\n'
            "def sim_app():\n"
            "    if not testing.isaac_sim_available():\n"
            '        pytest.skip("Isaac Sim not installed")\n'
            "    return testing.launch_app(headless=True)\n"
        ),
        "tests/test_smoke.py": (
            f'"""Smoke tests for {pkg} (CPU: register + build; GPU: create + reset)."""\n\n'
            "import pytest\n\n"
            "from hcrl_isaaclab import testing\n\n"
            f'PACKAGE = "{pkg}"\n'
            f'SOURCE = "{name}"\n\n\n'
            "def test_tasks_register(sim_app):\n"
            "    testing.run_registration_smoke(PACKAGE, SOURCE, min_expected=1)\n\n\n"
            "def test_tasks_build(sim_app):\n"
            "    testing.run_build_smoke(PACKAGE, SOURCE)\n\n\n"
            "@pytest.mark.gpu\n"
            "def test_tasks_run(sim_app):\n"
            "    testing.run_gpu_smoke(PACKAGE, SOURCE)\n"
        ),
        ".github/workflows/ci.yml": (
            "name: ci\n\n"
            "on:\n  pull_request:\n  push:\n    branches: [main]\n\n"
            "jobs:\n"
            "  lint:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            '        with:\n          python-version: "3.11"\n'
            "      - run: python -m pip install pre-commit\n"
            "      - run: pre-commit run --all-files --show-diff-on-failure\n\n"
            "  build:\n"
            "    # CPU tier; importing needs the Isaac Sim env, so run on the self-hosted `isaac` runner.\n"
            "    name: build smoke (CPU)\n"
            "    runs-on: [self-hosted, isaac]\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - name: Install + run CPU smoke\n"
            "        run: |\n"
            "          ${ILAB_PYTHON:-python} -m pip install -e .\n"
            '          ${ILAB_PYTHON:-python} -m pytest -m "not gpu" -q\n'
        ),
        ".gitignore": "__pycache__/\n*.pyc\n*.egg-info/\nlogs/\noutputs/\n.artifacts/\n",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="Project name (without the _tasks suffix), e.g. 'foo' -> foo_tasks.")
    ap.add_argument("--org", default="hcrl-isaac")
    ap.add_argument("--dest", default=None, help="Output dir (default: resources/<name>_tasks).")
    args = ap.parse_args()

    name = args.name.removesuffix("_tasks")
    dest = Path(args.dest) if args.dest else MANAGER_DIR / "resources" / f"{name}_tasks"
    if dest.exists():
        raise SystemExit(f"[new] {dest} already exists; refusing to overwrite.")

    for rel, content in _files(name, args.org).items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    print(f"[new] scaffolded {dest} ({name}/ namespace).")
    print(f"[new] next: git init it, push to {args.org}/{name}_tasks, add '{name}' to workspace.yaml projects.")


if __name__ == "__main__":
    main()
