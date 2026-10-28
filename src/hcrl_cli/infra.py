"""Compute commands: connect/disconnect, shell, stage and jobs on a cluster, res (cards and held nodes), sync and
upload-artifacts. Every cluster command names its cluster with `--on <name>` (a SLURM profile, or `ray`)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hcrl_cli.proc import VENV_PY, ask_select, handoff

CLUSTER_CONFIGS = Path("scripts/cluster/config")
SLURM_BACKEND = "scripts/cluster/cluster_interface.sh"
DEV_BACKEND = "scripts/cluster/cluster_dev/cluster_dev.sh"
RAY_BACKEND = "scripts/ray/ray_interface.sh"
RES = ("python3", "scripts/cluster/res/res.py")
INVENTORY = ("python3", "scripts/cluster/res/inventory.py")


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
    """SLURM cluster profiles: scripts/cluster/config/<name>/.env.cluster (gitignored, made by `pls setup --add`)."""
    return sorted(p.parent.name for p in _profile_dir().glob("*/.env.cluster"))


def _usage(verb: str, usage: str, text: str) -> None:
    print(f"usage: pls {verb} {usage}\n\n{text}")
    sys.exit(0)


def _on(verb: str, args: list[str], ray: bool = True) -> tuple[str, list[str]]:
    """Take ``--on <cluster>`` out of ``args`` (a picker when it is missing): the cluster and the other arguments."""
    names = [*profiles(), *(["ray"] if ray else [])]
    on, rest, i = "", [], 0
    while i < len(args):
        if args[i] == "--on" and i + 1 < len(args):
            on, i = args[i + 1], i + 2
        elif args[i].startswith("--on="):
            on, i = args[i].removeprefix("--on="), i + 1
        else:
            rest.append(args[i])
            i += 1
    on = on or ask_select(f"pls {verb} --on:", names)
    if on not in names:
        kinds = "a SLURM cluster profile" + (" or ray" if ray else "")
        sys.exit(f"[pls] {verb}: --on takes {kinds} ({', '.join(names) or 'none'}), not {on!r}."
                 " `pls setup --add <name>` creates a profile.")  # fmt: skip
    return on, rest


def _env(cluster: str) -> dict[str, str]:
    return {"CLUSTER": cluster}


def _connected(cluster: str, verb: str) -> None:
    """Open the cluster's SSH master when it is down: a 2FA prompt here, or a pointer to `pls connect` when no one is
    at the terminal (an agent session)."""
    up = subprocess.run(["bash", DEV_BACKEND, "__master_up"], env={**os.environ, **_env(cluster)}, check=False)
    if up.returncode == 0:
        return
    if not sys.stdin.isatty():
        sys.exit(f"[pls] {verb}: not connected to {cluster}, which needs 2FA: ask the user to run"
                 f" `pls connect --on {cluster}`")  # fmt: skip
    if subprocess.run(["bash", DEV_BACKEND, "open"], env={**os.environ, **_env(cluster)}, check=False).returncode:
        sys.exit(f"[pls] {verb}: could not connect to {cluster}")


def connect(args: list[str]) -> None:
    if set(args) & {"-h", "--help"}:
        _usage("connect", "--on <cluster>", "Open the cluster's shared SSH connection (2FA once). Every session and\n"
               "command reuses it until `pls disconnect`. Other commands connect on their own when run at a terminal.")  # fmt: skip
    cluster, rest = _on("connect", args, ray=False)
    handoff(["bash", DEV_BACKEND, "open", *rest], env=_env(cluster))


def disconnect(args: list[str]) -> None:
    if set(args) & {"-h", "--help"}:
        _usage("disconnect", "--on <cluster>", "Close the cluster's SSH connection, for every session using it.")
    cluster, rest = _on("disconnect", args, ray=False)
    handoff(["bash", DEV_BACKEND, "close", *rest], env=_env(cluster))


def shell(args: list[str]) -> None:
    if set(args) & {"-h", "--help"}:
        _usage("shell", "--on <cluster>", "An interactive shell on the cluster's held node (`pls res hold`).")
    cluster, rest = _on("shell", args, ray=False)
    _connected(cluster, "shell")
    handoff(["bash", DEV_BACKEND, "attach", *rest], env=_env(cluster))


STAGE_HELP = """Code reaches a cluster only as staged code trees; `pls run` stages what it runs by itself.
  pls stage --on <cluster>                          the workspace as it is on disk, as tree `default`
  pls stage --on <cluster> <name> <repo>=<ref|path>  a named tree with these repos pinned (the rest from the workspace)
  pls stage --on <cluster> list                     the staged trees and what each holds
  pls stage --on <cluster> rm <tree> [--force]      remove a tree (rm --partials: abandoned uploads)
Run a staged tree with `pls run --on <cluster> --tree <name> -- <script>`."""


def stage(args: list[str]) -> None:
    if set(args) & {"-h", "--help"}:
        _usage("stage", "--on <cluster> [list | rm <tree> | <name> <repo>=<ref|path>...]", STAGE_HELP)
    cluster, rest = _on("stage", args, ray=False)
    _connected(cluster, "stage")
    if rest[:1] == ["list"]:
        handoff(["bash", DEV_BACKEND, "trees", *rest[1:]], env=_env(cluster))
    if rest[:1] == ["rm"]:
        handoff(["bash", DEV_BACKEND, "trees", *rest], env=_env(cluster))
    handoff(["bash", DEV_BACKEND, "stage", *rest], env=_env(cluster))


JOBS_HELP = """Batch jobs on a cluster (`pls run --on <cluster> --batch`) and Ray jobs (`pls run --on ray`).
  pls jobs --on <cluster>                 queued and running jobs (`list`; extra args go to squeue or Ray)
  pls jobs --on <cluster> logs <job>      a job's output (SLURM: the last 100 lines, -f follows)
  pls jobs --on <cluster> stop <job>...   cancel jobs (a held node is `pls res drop`)
  pls jobs --on <cluster> kill [--all | STEP...]   cancel run steps on the held node, keeping the node"""


def jobs(args: list[str]) -> None:
    if set(args) & {"-h", "--help"}:
        _usage("jobs", "--on <cluster> [list | logs <job> | stop <job>... | kill ...]", JOBS_HELP)
    cluster, rest = _on("jobs", args)
    sub, rest = (rest[0], rest[1:]) if rest and rest[0] in ("list", "logs", "stop", "kill") else ("list", rest)
    if cluster == "ray":
        if sub == "kill":
            sys.exit("[pls] jobs --on ray: kill is for a SLURM held node; stop a Ray job with `pls jobs --on ray stop`")
        handoff([RAY_BACKEND, sub, *rest])
    _connected(cluster, "jobs")
    if sub == "kill":
        handoff(["bash", DEV_BACKEND, "kill", *rest], env=_env(cluster))
    handoff([SLURM_BACKEND, sub, *rest], env=_env(cluster))


def res(args: list[str]) -> None:
    """Compute resources: probe every GPU, claim/release/transfer leases, and hold or drop a cluster's node."""
    sub = args[0] if args else ""
    if sub == "hold":
        cluster, rest = _on("res hold", args[1:], ray=False)
        _connected(cluster, "res hold")
        handoff(["bash", DEV_BACKEND, "start", *rest], env=_env(cluster))
    if sub == "drop":
        cluster, rest = _on("res drop", args[1:], ray=False)
        if not rest:
            sys.exit(f"[pls] res drop: name the held node's job id (`pls res status --on {cluster}` shows it)")
        _connected(cluster, "res drop")
        handoff(["bash", DEV_BACKEND, "stop", *rest], env=_env(cluster))
    if sub == "status" and any(a == "--on" or a.startswith("--on=") for a in args):
        cluster, rest = _on("res status", args[1:])
        if cluster != "ray":  # the connection and held node first; an unreachable cluster still shows its cards
            subprocess.run(["bash", DEV_BACKEND, "status"], env={**os.environ, **_env(cluster)}, check=False)
        handoff([*RES, "status", "--pool", cluster, *rest])
    handoff([*RES, *args])


def setup_on(cluster: str, args: list[str]) -> None:
    """`pls setup --on <cluster>`: build the shared image and push it to a SLURM cluster, or set up Ray."""
    if cluster == "ray":
        handoff([RAY_BACKEND, "setup", *args])
    handoff([SLURM_BACKEND, "setup", *args], env=_env(cluster))


ADD_KINDS = ("slurm", "ssh", "ray")


def add_cluster(name: str, args: list[str]) -> None:
    """`pls setup --add <name> [--kind slurm|ssh|ray]`: create (or with --update, regenerate) a SLURM cluster profile,
    or write an ssh or ray pool into the gitignored compute.local.toml."""
    kind, rest, i = "slurm", [], 0
    while i < len(args):
        if args[i] == "--kind" and i + 1 < len(args):
            kind, i = args[i + 1], i + 2
        elif args[i].startswith("--kind="):
            kind, i = args[i].removeprefix("--kind="), i + 1
        else:
            rest.append(args[i])
            i += 1
    if kind not in ADD_KINDS:
        sys.exit(f"[pls] setup --add: --kind takes {', '.join(ADD_KINDS)}, not {kind!r}")
    if kind == "slurm":
        return handoff([SLURM_BACKEND, "add", *rest, *([name] if name else [])])
    if not name:
        sys.exit(f"[pls] setup --add --kind {kind}: name the pool (`pls setup --add <name> --kind {kind} ...`)")
    handoff([*INVENTORY, "add", name, "--kind", kind, *rest])


MOVED = {  # the retired `pls cluster <name> ...` forms -> what replaces each
    "develop open": "pls connect --on {c}",
    "develop close": "pls disconnect --on {c}",
    "develop start": "pls res hold --on {c}",
    "develop stop": "pls res drop --on {c} <job>",
    "develop status": "pls res status --on {c}",
    "develop attach": "pls shell --on {c}",
    "develop stage": "pls stage --on {c} [<name> <repo>=<ref|path>...]",
    "develop trees": "pls stage --on {c} list",
    "develop exec": "pls run --on {c} --cmd -- <command>",
    "develop kill": "pls jobs --on {c} kill [--all | STEP...]",
    "develop tail": "pls jobs --on {c} logs <job>",
    "setup": "pls setup --on {c}",
    "list": "pls jobs --on {c}",
    "logs": "pls jobs --on {c} logs <job>",
    "stop": "pls jobs --on {c} stop <job>",
    "status": "pls res status --on {c}",
    "bench": "pls run --on ray -- bench",
}


def cluster(args: list[str]) -> None:
    """`pls cluster` is retired: name the command that replaces the form given."""
    if args[:1] == ["add"]:
        sys.exit(f"[pls] `pls cluster add` is now `pls setup --add {' '.join(args[1:]) or '<name>'}`")
    c = args[0] if args else "<cluster>"
    key = " ".join(args[1:3]) if args[1:2] == ["develop"] else " ".join(args[1:2])
    new = MOVED.get(key)
    lines = [f"  {k:<15} -> {v.format(c='<cluster>')}" for k, v in MOVED.items()]
    hint = (
        f"use `{new.format(c=c)}`"
        if new
        else "its commands are now verb-first with --on <cluster>:\n" + "\n".join(lines)
    )
    sys.exit(f"[pls] `pls cluster` is retired; {hint}")


def sync(host: str, args: list[str]) -> None:
    """Push this workspace + Claude sessions to a peer box (git-aware code sync, transcripts, venv update)."""
    handoff(["python3", "scripts/sync_machine.py", host, *args], echo=True)


def upload_artifacts(args: list[str]) -> None:
    """Upload managed large-file resources to W&B as artifacts (W&B creds from scripts/.env.wandb, never argv)."""
    if not Path("scripts/.env.wandb").is_file():
        sys.exit("[ERROR] scripts/.env.wandb not found; run 'pls deps' first.")
    upload = "resources/hcrl_isaaclab/scripts/tools/upload_artifacts.py"
    handoff(["bash", "-c", 'set -a; source scripts/.env.wandb; set +a; exec "$0" "$@"', VENV_PY, upload, *args])
