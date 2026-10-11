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
NVIDIA_INDEX = "https://pypi.nvidia.com"
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
    # --relocatable: console scripts derive the interpreter from their own path (survives a dir rename).
    run(["uv", "venv", "--relocatable", "--python", "3.11", VENV], echo=True)
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
    """Full local install into the `ilab` venv: manager deps + Isaac Lab / Isaac Sim + every workspace package."""
    deps()
    # Re-open the project/IsaacLab picker pre-filled (plain `pls setup` reconfigures); no TTY keeps it.
    run([VENV_PY, "scripts/configure_workspace.py", "--interactive"], echo=True)
    resolve([])  # merge selection + defaults -> gitman.yaml + fetch repos under resources/
    mode = _isaaclab_mode()
    if mode == "none":
        print("[setup] IsaacLab mode 'none': repos fetched under resources/; skipping IsaacLab + package installs.")
        return
    # A failed install does not stop the rest (as before); the failures are listed at the end.
    ok: dict[str, bool] = {}
    _section("PyTorch (CUDA 12.8)")
    ok["torch"] = _pip("--torch-backend", CU, "torch==2.7.0", "torchvision==0.22.0")
    _section("Isaac Lab / Isaac Sim")
    if mode == "source":
        print(
            "[setup] IsaacLab source mode -> editable install (+ explicit isaacsim; source isaaclab has no isaacsim extra)"
        )
        ok["isaacsim"] = _pip("isaacsim[all,extscache]==5.1.0", "--extra-index-url", NVIDIA_INDEX)
        for d in sorted(glob.glob("resources/IsaacLab/source/isaaclab*/")):
            ok[d] = _pip("--torch-backend", CU, "-e", d)
        ok["hcrl_isaaclab"] = _pip("--torch-backend", CU, "-e", "resources/hcrl_isaaclab")
    else:
        print(
            "[setup] IsaacLab via pip -> hcrl_isaaclab[isaacsim] pulls isaaclab[isaacsim]==2.3.2.post1 (isaacsim 5.1 + torch)"
        )
        ok["hcrl_isaaclab"] = _pip(
            "--torch-backend", CU, "--extra-index-url", NVIDIA_INDEX, "--index-strategy", "unsafe-best-match",
            "-e", "resources/hcrl_isaaclab[isaacsim]",
        )  # fmt: skip
    ok["rsl_rl-lib"] = _pip("rsl_rl-lib")
    _section("Workspace packages")
    retired = run([VENV_PY, "scripts/resolve_workspace.py", "--retire-renamed"], capture=True, check=False)
    retired = set(retired.stdout.split())
    packages = [
        "resources/robot_rl",
        "resources/hcrl_sim2real",
        *sorted(glob.glob("resources/*_tasks")),
        *sorted(glob.glob("resources/*_robots")),
        "resources/holosoma/src/holosoma_retargeting",
    ]
    for d in packages:
        if d in retired:
            print(f"[setup] skipping pre-rename checkout: {d}")
        elif Path(d, "setup.py").is_file() or Path(d, "pyproject.toml").is_file():
            ok[d] = _pip("--torch-backend", CU, "--extra-index-url", NVIDIA_INDEX, "-e", d)
        elif Path(d).is_dir():
            print(f"[setup] skipping non-package data repo: {d}")
    if failed := [name for name, good in ok.items() if not good]:
        print(f"[setup][WARN] these installs failed (see above): {', '.join(failed)}", file=sys.stderr)
    vscode([])


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
