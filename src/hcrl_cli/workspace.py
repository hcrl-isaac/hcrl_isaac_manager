"""Local install and workspace commands: deps, setup, resolve, vscode, clean, new, docker, tests."""

from __future__ import annotations

import glob
import os
import re
import shutil
import sys
from pathlib import Path

from hcrl_cli.proc import BASH_UTILS, RC_FILE, ROOT, VENV, VENV_PY, handoff, run

CU = "cu128"
ISAACLAB_PYTHON = "3.12"  # Isaac Lab 3's interpreter, so the ilab venv's
WANDB_ENV = Path("scripts/.env.wandb")
RC_LINE = f"VENV_NAME={VENV} source {BASH_UTILS}"
LFS_INSTALLERS = (
    ("brew", "brew install git-lfs"),
    ("apt-get", "sudo apt-get update -qq && sudo apt-get install -y git-lfs"),
    ("dnf", "sudo dnf install -y git-lfs"),
    ("pacman", "sudo pacman -S --noconfirm git-lfs"),
)
DOCKER_INSTALL = (
    "curl -fsSL https://get.docker.com -o get-docker.sh; sudo sh get-docker.sh; sudo groupadd docker; "
    "sudo usermod -aG docker $USER; newgrp docker"
)


def _rc_has_hook() -> bool:
    return RC_FILE.is_file() and f"source {BASH_UTILS}" in RC_FILE.read_text()


def deps() -> None:
    """Manager base env: the `ilab` venv (with `pls`) plus uv, gitman, git-lfs, W&B creds and the `ilab` alias."""
    if not shutil.which("uv"):
        run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], echo=True)
        os.environ["PATH"] = f"{Path.home()}/.local/bin{os.pathsep}{os.environ['PATH']}"  # where the installer puts uv
    if os.access(VENV_PY, os.X_OK):
        # An existing ilab holds the Isaac stack + editable packages: add pls, never remove (`uv sync` is exact).
        owner = (ROOT / VENV).resolve().parent
        if owner != ROOT.resolve():  # a worktree's linked ilab: -e . would repoint the shared venv here
            sys.exit(f"[pls] deps: {VENV} belongs to {owner}; run `pls deps` from that checkout.")
        run(["uv", "pip", "install", "--python", VENV_PY, "-e", "."], echo=True)
    else:
        # --relocatable: console scripts derive the interpreter from their own path (survives a dir rename).
        run(["uv", "venv", "--relocatable", "--python", ISAACLAB_PYTHON, VENV], echo=True)
        run(["uv", "sync"], echo=True)
    # git-lfs must materialize LFS assets as real files -- the Ray mount ships pointers as-is otherwise.
    if not shutil.which("git-lfs"):
        print("[deps] git-lfs not found; attempting install...")
        installer = next((cmd for tool, cmd in LFS_INSTALLERS if shutil.which(tool)), None)
        if installer:
            run(["bash", "-c", installer])
    if shutil.which("git-lfs"):
        run(["git", "lfs", "install"])
    else:
        print(
            "[deps][WARN] git-lfs missing -- LFS assets (e.g. crab/MCP policies) will be pointers and fail on the"
            " cluster; install git-lfs and re-run."
        )
    # Install the gitman tool only; fetching subrepos (+ generating gitman.yaml) is `pls resolve`'s job.
    run(["uv", "tool", "install", "gitman"], echo=True)
    if not WANDB_ENV.exists():
        run([VENV_PY, "scripts/tools/ask.py", "wandb-env", "scripts/tools/.env.wandb.template", str(WANDB_ENV)])
    if not _rc_has_hook():
        print(f"[INFO] Adding {BASH_UTILS.name} to {RC_FILE.name}")
        with RC_FILE.open("a") as rc:
            rc.write(RC_LINE + "\n")
        print(f"[INFO] Successfully added {BASH_UTILS.name} to {RC_FILE.name}\n")
        print("\t\tRun  ilab  to activate the venv and enter the manager dir.\n")


def _isaaclab_mode() -> str:
    """`mode:` from workspace.yaml (first match, comments stripped), as `pls setup` gates the installs on it."""
    for line in Path("workspace.yaml").read_text().splitlines():
        if m := re.match(r"\s*mode:\s*([^\s#]*)", line):
            return m.group(1)
    return ""


def _pip(*args: str) -> bool:
    return run(["uv", "pip", "install", "--python", VENV_PY, *args], echo=True, check=False).returncode == 0


def _section(title: str) -> None:
    run([VENV_PY, "scripts/tools/ui.py", "section", title], check=False)


def setup() -> None:
    """Full local install into the `ilab` venv: manager deps + Isaac Lab 3 + every workspace package."""
    deps()
    # Re-open the project/IsaacLab picker pre-filled (plain `pls setup` reconfigures); no TTY keeps it.
    run([VENV_PY, "scripts/configure_workspace.py", "--interactive"], echo=True)
    resolve([])  # merge selection + defaults -> gitman.yaml + fetch repos under resources/
    if install():
        vscode([])


def install() -> bool:
    """Layer Isaac Lab 3 (the uv-locked source clone, which owns torch) + every workspace package onto the ilab venv;
    returns False when the IsaacLab mode is 'none' and nothing was installed."""
    mode = _isaaclab_mode()
    if mode == "none":
        print("[install] IsaacLab mode 'none': repos fetched under resources/; skipping IsaacLab + package installs.")
        return False
    if mode != "source":
        sys.exit("[install][ERROR] Isaac Lab 3 has no pip release; set isaaclab.mode: source (pls setup).")
    version = f"import sys; sys.exit('%d.%d' % sys.version_info[:2] != '{ISAACLAB_PYTHON}')"
    if run([VENV_PY, "-c", version], check=False).returncode:
        sys.exit(
            f"[install][ERROR] Isaac Lab 3 needs a Python {ISAACLAB_PYTHON} venv; rebuild it: pls clean && pls setup"
        )
    # A failed install does not stop the rest; the failures are listed at the end.
    ok: dict[str, bool] = {}
    _section("Isaac Lab (source, uv-locked)")
    extras = run(
        [VENV_PY, "scripts/resolve_workspace.py", "--isaaclab-extras"], capture=True, check=False
    ).stdout.split()
    # an absolute venv path: a relative UV_PROJECT_ENVIRONMENT would land inside resources/IsaacLab
    sync = ["uv", "sync", "--project", "resources/IsaacLab", "--frozen", "--inexact", *extras]
    ok["IsaacLab"] = run(sync, echo=True, check=False, env={"UV_PROJECT_ENVIRONMENT": str(ROOT / VENV)}).returncode == 0
    _section("Workspace packages")
    retired = run([VENV_PY, "scripts/resolve_workspace.py", "--retire-renamed"], capture=True, check=False)
    retired = set(retired.stdout.split())
    packages = [
        "resources/hcrl_isaaclab",
        "resources/robot_rl",
        "resources/hcrl_sim2real",
        *sorted(glob.glob("resources/*_tasks")),
        *sorted(glob.glob("resources/*_robots")),
        "resources/holosoma/src/holosoma_retargeting",
    ]
    for d in packages:
        if d in retired:
            print(f"[install] skipping pre-rename checkout: {d}")
        elif Path(d, "setup.py").is_file() or Path(d, "pyproject.toml").is_file():
            ok[d] = _pip("--torch-backend", CU, "-e", d)
        elif Path(d).is_dir():
            print(f"[install] skipping non-package data repo: {d}")
    if failed := [name for name, good in ok.items() if not good]:
        print(f"[install][WARN] these installs failed (see above): {', '.join(failed)}", file=sys.stderr)
    return True


def vscode(args: list[str]) -> None:
    """Generate .vscode/settings.json; `no-kit` reuses the cached Kit paths instead of booting Isaac Sim."""
    if args in (["no-kit"], ["--no-kit"]):
        handoff([VENV_PY, "scripts/tools/setup_vscode.py", "--no-kit"])
    spin = [
        VENV_PY,
        "scripts/tools/ui.py",
        "spin",
        "Booting headless Isaac Sim to snapshot VS Code extension paths (~1 min)",
    ]
    if run(
        [*spin, "--", VENV_PY, "scripts/tools/setup_vscode.py"], env={"OMNI_KIT_ACCEPT_EULA": "YES"}, check=False
    ).returncode:
        print("[vscode][WARN] settings generation failed")


def clean() -> None:
    """Remove the ilab venv, generated workspace config (workspace.yaml/gitman.yaml) and the shell alias (asks first)."""
    prompt = "Remove the ilab venv, generated workspace config, and shell alias?"
    if (
        os.access(VENV_PY, os.X_OK)
        and run([VENV_PY, "scripts/tools/ask.py", "confirm", prompt], check=False).returncode
    ):
        print("[clean] aborted.")
        return
    venv_dir = ROOT / VENV
    if venv_dir.is_dir():
        print(f"[INFO] Removing virtual environment at {venv_dir}.")
        shutil.rmtree(venv_dir)
    for f in ("workspace.yaml", "gitman.yaml", "gitman.yml"):
        if Path(f).is_file():
            print(f"[INFO] Removing generated {f}.")
            Path(f).unlink()
    if _rc_has_hook():
        print(f"[INFO] Removing {BASH_UTILS.name} from {RC_FILE.name}")
        lines = RC_FILE.read_text().splitlines(keepends=True)
        RC_FILE.write_text("".join(line for line in lines if f"source {BASH_UTILS}" not in line))
    print("[INFO] Successfully cleaned up environment.")


def resolve(args: list[str]) -> None:
    """Merge selection + committed defaults -> gitman.yaml, fetch every repo under resources/, then pull LFS files."""
    run([VENV_PY, "scripts/configure_workspace.py"], echo=True)  # ensure a workspace.yaml exists (no prompt)
    run([VENV_PY, "scripts/resolve_workspace.py", "--manifest", "workspace.yaml", "--update", *args], echo=True)
    # materialize LFS in every fetched repo (repos cloned before git-lfs was active hold pointers)
    if not shutil.which("git-lfs"):
        print("[resolve][WARN] git-lfs missing -- LFS files stay as pointers; run 'pls deps' to install it.")
        return
    for d in sorted(glob.glob("resources/*/")):
        if Path(d, ".git").is_dir():
            print(f"[resolve] git lfs pull {d}")
            run(["git", "-C", d, "lfs", "pull"], check=False)


def new(name: str) -> None:
    """Scaffold a new <name>_tasks extension repo under resources/ (registers under the <name>/ namespace)."""
    handoff([VENV_PY, "scripts/new_tasks.py", name], echo=True)


def docker_deps() -> None:
    """Regenerate the image's workspace-dependency list from each repo's pyproject/setup.py."""
    handoff([VENV_PY, "scripts/docker/collect_workspace_deps.py"], echo=True)


def docker(args: list[str]) -> None:
    """Build the shared Isaac image (scripts/docker/); refuses a stale committed dependency list."""
    if run([VENV_PY, "scripts/docker/collect_workspace_deps.py", "--check"], check=False).returncode:
        sys.exit("[docker] run 'pls docker-deps' and commit the result")
    if not shutil.which("docker"):
        run(["bash", "-c", DOCKER_INSTALL], echo=True)
    handoff(["scripts/docker/docker_interface.sh", *args], echo=True)


def test_scripts() -> None:
    """The manager's own script tests (scripts/<area>/tests; stubs only, no cluster or GPU). CI runs these."""
    handoff(["bash", "scripts/run_tests.sh"])


def test(repo: str, args: list[str]) -> None:
    """A workspace repo's CPU smoke tests (skips `-m gpu`); launches Isaac Sim, so it needs the venv and a GPU."""
    os.chdir(ROOT / "resources" / repo)
    cmd = [str(ROOT / VENV_PY), "-m", "pytest", "-m", "not gpu", *args]
    handoff(cmd, echo=True, env={"OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONPATH": ""})
