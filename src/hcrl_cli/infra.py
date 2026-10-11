"""Compute commands: cluster (SLURM profiles and Ray), res, sync and upload-artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hcrl_cli.proc import VENV_PY, ask_select, handoff

CLUSTER_CONFIGS = Path("scripts/cluster/config")
SLURM_BACKEND = "scripts/cluster/cluster_interface.sh"
RAY_BACKEND = "scripts/ray/ray_interface.sh"
SHARED_VERBS = ("setup", "list", "logs", "stop", "status")
BACKEND_VERBS = {"develop": "slurm", "add": "slurm", "bench": "ray"}  # verbs only one kind of cluster has
JOB_VERBS = ("job", "run")  # launches go through `pls run`


def _profile_dir() -> Path:
    """This checkout's cluster profiles, or the main checkout's when this worktree has none (they are gitignored)."""
    if any(CLUSTER_CONFIGS.glob("*/.env.cluster")):
        return CLUSTER_CONFIGS
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, check=False
    ).stdout.strip()
    main = Path(common).parent / CLUSTER_CONFIGS if common else CLUSTER_CONFIGS
    return main if any(main.glob("*/.env.cluster")) else CLUSTER_CONFIGS


def profiles() -> list[str]:
    """SLURM cluster profiles: scripts/cluster/config/<name>/.env.cluster (gitignored, made by `pls cluster add`)."""
    return sorted(p.parent.name for p in _profile_dir().glob("*/.env.cluster"))


def cluster(args: list[str]) -> None:
    """`pls cluster <name> <verb> [args]`: <name> is a SLURM profile or `ray`; bare arguments show pickers."""
    os.environ.pop("CLUSTER", None)  # the name is the only cluster selector
    if args and args[0] == "add":  # a new profile has no name yet
        return handoff([SLURM_BACKEND, "add", *args[1:]])
    names = [*profiles(), "ray"]
    name, args = (args[0], args[1:]) if args and args[0] else (ask_select("Cluster:", names), [])
    if name not in names:
        sys.exit(f"[pls] cluster: no cluster {name!r} (profiles: {', '.join(names[:-1]) or 'none'}; or ray)."
                 " `pls cluster add` creates a profile.")  # fmt: skip
    kind = "ray" if name == "ray" else "slurm"
    verbs = [*SHARED_VERBS, *(v for v, k in BACKEND_VERBS.items() if k == kind)]
    verb, rest = (args[0], args[1:]) if args and args[0] else (ask_select(f"{name}:", verbs), [])
    if verb in JOB_VERBS:
        batch = "-- <script> [args]" if kind == "ray" else "--batch [--tree N] -- train [args]"
        sys.exit(f"[pls] runs go through `pls run --on {name} {batch}`, not `pls cluster {name} {verb}`")
    if verb in BACKEND_VERBS and BACKEND_VERBS[verb] != kind:
        sys.exit(f"[pls] cluster: {verb} is for {BACKEND_VERBS[verb].upper()} clusters only; {name} is {kind}")
    if verb not in verbs:
        sys.exit(f"[pls] cluster {name}: unknown verb {verb!r} ({', '.join(verbs)})")
    if verb == "status":  # the cluster's cards, from the resource probe
        return handoff(["python3", "scripts/cluster/res/res.py", "status", "--pool", name, *rest])
    if verb == "add":
        return handoff([SLURM_BACKEND, "add", *rest, name])
    if kind == "ray":
        return handoff([RAY_BACKEND, verb, *rest])
    handoff([SLURM_BACKEND, verb, *rest], env={"CLUSTER": name})


def res(args: list[str]) -> None:
    """Compute resources (scripts/cluster/res/): `status` probes every GPU on every pool; claims and leases."""
    handoff(["python3", "scripts/cluster/res/res.py", *args])


def sync(host: str, args: list[str]) -> None:
    """Push this workspace + Claude sessions to a peer box (git-aware code sync, transcripts, venv update)."""
    handoff(["python3", "scripts/sync_machine.py", host, *args], echo=True)


def upload_artifacts(args: list[str]) -> None:
    """Upload managed large-file resources to W&B as artifacts (W&B creds from scripts/.env.wandb, never argv)."""
    if not Path("scripts/.env.wandb").is_file():
        sys.exit("[ERROR] scripts/.env.wandb not found; run 'pls deps' first.")
    upload = "resources/hcrl_isaaclab/scripts/tools/upload_artifacts.py"
    handoff(["bash", "-c", 'set -a; source scripts/.env.wandb; set +a; exec "$0" "$@"', VENV_PY, upload, *args])
