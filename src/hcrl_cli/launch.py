"""`pls run`: run a script or a command here, on a leased card (local, ssh, a cluster's held job), as a cluster
batch job, or on Ray."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hcrl_cli import proc
from hcrl_cli.proc import ROOT, VENV, VENV_PY, handoff

CARD_CMD = ("python3", "scripts/cluster/res/evaluate.py")  # leases the card, ships code + checkpoints, runs
RAY_BACKEND = "scripts/ray/ray_interface.sh"
SLURM_BACKEND = "scripts/cluster/cluster_interface.sh"
DEV_BACKEND = "scripts/cluster/cluster_dev/cluster_dev.sh"
CORE = "hcrl_isaaclab"
HELP = ("-h", "--help")

USAGE = """pls run <script> [args]
       pls run [--on TARGET] [--wt NAME] [card options] -- <script> [args]
       pls run [--on TARGET] [--wt NAME] [card options] --cmd -- <command> [args]
       pls run --on CLUSTER --batch [--tree N | --wt NAME] -- train [args]"""
EPILOG = """targets (--on):
  (none)           this machine, with the ilab venv
  host:gpu         that card (host:job:gpu on a SLURM node running several jobs; local:<gpu> = this machine)
  any              any free card on the boxes
  <pool>           any free card of that pool (`pls res pools`)
  lease:<id>       a card you already lease (left leased afterwards)
  ray              the Ray cluster (queued until a GPU frees); `train` submits a training job, `--distributed`
                   one spanning a sub-job per GPU node
  <cluster> --batch  a batch job on that SLURM profile (`pls cluster`), running train: the workspace is staged
                   as tree `default` (or --wt NAME's worktree set as tree NAME) and the job runs that tree as
                   resolved now, even if newer trees are staged while it queues; --tree N runs a staged tree as is

scripts (the same on every target):
  <name>           hcrl_isaaclab/scripts/<name>.py
  <repo>/<path>    a file of a workspace repo; --wt picks its worktree, and a card or Ray ships it with the run
  <repo>:<path>    the same
  <path>           any other path is a local file, relative to where you are (not on Ray)

--cmd runs the words after -- as a command instead (e.g. `--cmd -- nvidia-smi`, `--cmd -- python -m pkg`): here from
where you are, on a card from the box workspace (a SLURM card: the run's work dir), with the ilab venv first on PATH
(the container's python on SLURM). Ray runs python scripts only.

card options (--holder is required): --holder, --note, --pool, --min-free-gb, --checkpoint, --env, --timeout,
--stall, --detach; `pls run --on any --help` describes them."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pls run",
        usage=USAGE,
        add_help=False,  # -h is handled before parsing: a card target shows the card backend's options
        description="Run a script or a command here, on a leased GPU, or on the Ray cluster.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--on", default="", metavar="TARGET", help="where to run (default: this machine)")
    p.add_argument(
        "--wt", default="", metavar="NAME", help="run this machine's worktree set resources/<repo>/worktrees/<NAME>"
    )
    p.add_argument("--cmd", action="store_true", help="the words after -- are a command, not a script")
    p.add_argument("--batch", action="store_true", help="submit a batch job on SLURM cluster TARGET (train only)")
    p.add_argument("--tree", default="", metavar="N", help="(--batch) run staged tree N instead of staging")
    p.add_argument("--distributed", action="store_true", help="(--on ray, train) a sub-job per GPU node")
    return p


def _help(opts: list[str]) -> None:
    """`pls run [--on <card>] --help`: the card backend's options for a card target, else this command's."""
    ns, _ = parser().parse_known_args([o for o in opts if o not in HELP])
    if ns.on and ns.on != "ray" and not ns.batch:
        handoff([*CARD_CMD, "--help"])
    parser().print_help()
    sys.exit(0)


def _split(argv: list[str]) -> tuple[list[str], list[str]]:
    """Options, then the script and its arguments; without options the first word is the script."""
    if not argv:
        parser().print_help()
        sys.exit(2)
    if not argv[0].startswith("-"):
        return [], argv
    head = argv[: argv.index("--")] if "--" in argv else argv
    if any(h in head for h in HELP):
        _help(head)
    if "--" not in argv:
        sys.exit("[pls] run: options go before `-- <script> [args]`")
    i = argv.index("--")
    if i + 1 == len(argv):
        sys.exit("[pls] run: missing <script> after --")
    return argv[:i], argv[i + 1 :]


def _is_name(script: str) -> bool:
    return "/" not in script and ":" not in script and not script.endswith(".py")


def _repo_path(script: str) -> tuple[str, str] | None:
    """``(repo, path)`` for ``<repo>/<path>`` or ``<repo>:<path>`` naming a workspace repo, else None."""
    repo, sep, path = script.partition(":") if ":" in script else script.partition("/")
    if sep and path and repo and not repo.startswith(".") and (ROOT / "resources" / repo).is_dir():
        return repo, path
    return None


def _local_file(script: str) -> str:
    """A local script file, relative to where `pls` was started, as an absolute path."""
    path = Path(proc.CALLER_CWD, script)
    if not path.is_file():
        sys.exit(f"[pls] run: no script {script} (not a <name>, a <repo>/<path> or a local file)")
    return str(path.resolve())


def _here(argv: list[str], wt: str, command: bool) -> None:
    """This machine, with the ilab venv; ``wt`` routes PYTHONPATH (and repo files) to a worktree set."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from worktree_env import resolve, select

    pythonpath, core, overridden = select(wt)
    if overridden:
        print(f"[worktree] {wt}: {', '.join(overridden)} (others from main checkouts)", file=sys.stderr)
    # The worktree'd roots precede the editable installs of the main checkouts, so only diverging repos change.
    env = {
        "PYTHONPATH": ":".join(p for p in (pythonpath, os.environ.get("PYTHONPATH", "")) if p),
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    if command:
        env["PATH"] = f"{ROOT / VENV}/bin{os.pathsep}{os.environ.get('PATH', '')}"
        os.chdir(proc.CALLER_CWD)
        handoff(argv, env=env)
        return
    script, *args = argv
    if _is_name(script):
        path = f"{core}/scripts/{script}.py"
    elif named := _repo_path(script):
        repo, rel = named
        path = f"{resolve(wt)[0].get(repo, str(ROOT / 'resources' / repo))}/{rel}"
    else:
        path = _local_file(script)
    handoff([VENV_PY, path, *args], env=env)


def _card_script(script: str) -> str:
    if _is_name(script):
        return f"{CORE}:scripts/{script}.py"
    named = _repo_path(script)
    return f"{named[0]}:{named[1]}" if named else _local_file(script)


def _ray_script(script: str) -> str:
    if _is_name(script):
        return f"{CORE}/scripts/{script}.py"
    named = _repo_path(script)
    if not named:
        sys.exit(f"[pls] run --on ray ships a workspace repo's file (<repo>/<path>.py); {script} is not one")
    return f"{named[0]}/{named[1]}"


def _card_flags(on: str) -> list[str]:
    if on == "any":
        return ["--any"]
    if on.startswith("lease:"):
        return ["--lease", on.removeprefix("lease:")]
    if ":" in on:
        return ["--on", on]
    return ["--any", "--pool", on]


def _batch(on: str, wt: str, tree: str, run: list[str]) -> None:
    """A batch job on a SLURM profile: stage the code (the workspace, or the worktree set ``wt``), then submit."""
    from hcrl_cli.infra import profiles

    if on not in profiles():
        sys.exit(f"[pls] run --batch: --on names a SLURM cluster profile ({', '.join(profiles()) or 'none'}), not {on}")
    script, *args = run
    if script != "train":
        sys.exit(f"[pls] run --batch runs train; run {script} on a card instead (pls run --on {on} -- {script} ...)")
    if tree and wt:
        sys.exit("[pls] run --batch: --tree runs a staged tree as is; --wt stages a worktree set, so pass one")
    if wt:  # the worktree set's repos over the newest `default`, as tree <wt>
        sys.path.insert(0, str(ROOT / "scripts"))
        from worktree_env import resolve

        paths, overridden = resolve(wt)
        if not overridden:
            sys.exit(f"[pls] run --batch: no repo has a worktree set {wt}")
        proc.run(
            ["bash", DEV_BACKEND, "stage", wt, *(f"{repo}={paths[repo]}" for repo in overridden)], env={"CLUSTER": on}
        )
        tree = wt
    handoff([SLURM_BACKEND, "job", *(["--tree", tree] if tree else []), *args], env={"CLUSTER": on})


def main(argv: list[str]) -> None:
    opts, run = _split(argv)
    ns, extra = parser().parse_known_args(opts)
    os.environ.pop("WT", None)  # --wt is the only worktree selector
    if ns.batch:
        if extra or ns.cmd:
            sys.exit(f"[pls] run --batch: {' '.join([*extra, *(['--cmd'] if ns.cmd else [])])} only apply on a card")
        _batch(ns.on, ns.wt, ns.tree, run)
        return
    if ns.tree:
        sys.exit("[pls] run: --tree only applies with --batch")
    if ns.distributed and (ns.on != "ray" or run[0] != "train"):
        sys.exit("[pls] run: --distributed only applies to train on Ray (--on ray -- train)")
    if not ns.on:
        if extra:
            sys.exit(f"[pls] run: {' '.join(extra)} only apply with --on")
        _here(run, ns.wt, ns.cmd)
    elif ns.on == "ray":
        if extra or ns.cmd:
            sys.exit(f"[pls] run --on ray: {' '.join([*extra, *(['--cmd'] if ns.cmd else [])])} only apply on a card")
        script, *args = run
        env = {"WT": ns.wt} if ns.wt else None
        if script == "train":  # a training job: wrap_resources runs train.py (sweeps, aggregate jobs)
            handoff([RAY_BACKEND, "job_distributed" if ns.distributed else "job", *args], env=env)
        else:
            handoff([RAY_BACKEND, "run", _ray_script(script), *args], env=env)
    else:
        script, *args = run
        wt = ["--wt", ns.wt] if ns.wt else []
        target = ["--cmd", script] if ns.cmd else [_card_script(script)]
        handoff([*CARD_CMD, *extra, *_card_flags(ns.on), *wt, *target, "--", *args])
