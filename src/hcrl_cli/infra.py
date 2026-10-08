"""Compute commands: cluster, res, ray, sync, upload-artifacts and run. The scripts they call are unchanged."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from hcrl_cli.proc import ROOT, VENV_PY, ask_select, handoff

CLUSTER_CONFIGS = Path("scripts/cluster/config")
CLUSTER_VERBS = ("setup", "job", "develop", "repush", "build", "add")
CLUSTER_TARGETED = ("setup", "job", "develop", "repush")  # verbs that ask for a target when several clusters exist
RAY_VERBS = ("setup", "bench", "push", "list", "logs", "stop")
RAY_RUNS = ("job", "run")  # `pls run --on ray` submits these


def cluster(args: list[str]) -> None:
    """Cluster interface (scripts/cluster/); a leading config name selects config/<name>, bare args show a picker."""
    os.environ.pop("CLUSTER", None)  # the leading name (or the picker) is the only cluster selector
    name = ""
    if args and args[0] and (CLUSTER_CONFIGS / args[0]).is_dir():
        name, args = args[0], args[1:]
    if not args or not args[0]:
        args = [ask_select("Cluster subcommand:", CLUSTER_VERBS)]
    if not name and args[0] in CLUSTER_TARGETED:
        configs = sorted(p.name for p in CLUSTER_CONFIGS.iterdir() if p.is_dir()) if CLUSTER_CONFIGS.is_dir() else []
        if len(configs) > 1:
            name = ask_select("Target cluster:", configs)
    handoff(["scripts/cluster/cluster_interface.sh", *args], env={"CLUSTER": name} if name else None)


def res(args: list[str]) -> None:
    """Compute resources (scripts/cluster/res/): `status` probes every GPU on every pool; claims and leases."""
    handoff(["python3", "scripts/cluster/res/res.py", *args])


def ray(args: list[str]) -> None:
    """Ray interface (scripts/ray/): setup, bench, push, list, logs, stop; bare picks. Runs go through `pls run --on ray`."""
    if not args or not args[0]:
        args = [ask_select("Ray subcommand:", RAY_VERBS)]
    if args[0] in RAY_RUNS:
        sys.exit(
            f"[pls] Ray runs and training jobs go through `pls run --on ray -- <script|train> [args]`, not ray {args[0]}"
        )
    handoff(["scripts/ray/ray_interface.sh", *args])


def sync(host: str, args: list[str]) -> None:
    """Push this workspace + Claude sessions to a peer box (git-aware code sync, transcripts, venv update)."""
    handoff(["python3", "scripts/sync_machine.py", host, *args], echo=True)


def upload_artifacts(args: list[str]) -> None:
    """Upload managed large-file resources to W&B as artifacts (W&B creds from scripts/.env.wandb, never argv)."""
    if not Path("scripts/.env.wandb").is_file():
        sys.exit("[ERROR] scripts/.env.wandb not found; run 'pls deps' first.")
    upload = "resources/hcrl_isaaclab/scripts/tools/upload_artifacts.py"
    handoff(["bash", "-c", 'set -a; source scripts/.env.wandb; set +a; exec "$0" "$@"', VENV_PY, upload, *args])


def run_script(script: str, args: list[str], wt: str = "") -> None:
    """Run hcrl_isaaclab/scripts/<script>.py here with the ilab venv; ``wt`` names a worktree set."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from worktree_env import select

    pythonpath, core, overridden = select(wt)
    if overridden:
        print(f"[worktree] {wt}: {', '.join(overridden)} (others from main checkouts)", file=sys.stderr)
    # The worktree'd roots precede the editable installs of the main checkouts, so only diverging repos change.
    pythonpath = ":".join(p for p in (pythonpath, os.environ.get("PYTHONPATH", "")) if p)
    env = {"PYTHONPATH": pythonpath, "OMNI_KIT_ACCEPT_EULA": "YES"}
    handoff([VENV_PY, f"{core}/scripts/{script}.py", *args], env=env)
