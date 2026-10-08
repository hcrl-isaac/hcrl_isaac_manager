"""`pls run`: run a script here, on a leased card (local, ssh, a cluster's held job) or on the Ray cluster."""

from __future__ import annotations

import argparse
import os
import sys

from hcrl_cli import infra
from hcrl_cli.proc import ROOT, handoff

CARD_CMD = ("python3", "scripts/cluster/res/evaluate.py")  # leases the card, ships code + checkpoints, runs
RAY_BACKEND = "scripts/ray/ray_interface.sh"
CORE = "hcrl_isaaclab"
HELP = ("-h", "--help")

USAGE = """pls run <script> [args]
       pls run [--on TARGET] [--wt NAME] [card options] -- <script> [args]"""
EPILOG = """targets (--on):
  (none)           this machine, with the ilab venv
  host:gpu         that card (host:job:gpu on a SLURM node running several jobs; local:<gpu> = this machine)
  any              any free card on the boxes
  <pool>           any free card of that pool (`pls res pools`)
  lease:<id>       a card you already lease (left leased afterwards)
  ray              the Ray cluster (queued until a GPU frees); `train` submits a training job

scripts:
  <name>           hcrl_isaaclab/scripts/<name>.py
  <repo>/<path>.py a file of a workspace repo (shipped with the run; --wt picks its worktree)
  <repo>:<path>    the same, as res names it; any other path is a local file (cards only)

card options (--holder is required): --holder, --note, --pool, --min-free-gb, --checkpoint, --env, --timeout,
--stall, --detach; `pls run --on any --help` describes them."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pls run",
        usage=USAGE,
        add_help=False,  # -h is handled before parsing: a card target shows the card backend's options
        description="Run a script here, on a leased GPU, or on the Ray cluster.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--on", default="", metavar="TARGET", help="where to run (default: this machine)")
    p.add_argument(
        "--wt", default="", metavar="NAME", help="run this machine's worktree set resources/<repo>/worktrees/<NAME>"
    )
    return p


def _help(opts: list[str]) -> None:
    """`pls run [--on <card>] --help`: the card backend's options for a card target, else this command's."""
    ns, _ = parser().parse_known_args([o for o in opts if o not in HELP])
    if ns.on and ns.on != "ray":
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


def _repo_path(script: str) -> tuple[str, str] | None:
    """``(repo, path)`` for ``<repo>/<path>`` or ``<repo>:<path>`` naming a workspace repo, else None."""
    repo, sep, path = script.partition(":") if ":" in script else script.partition("/")
    if sep and path and repo and not repo.startswith(".") and (ROOT / "resources" / repo).is_dir():
        return repo, path
    return None


def _card_script(script: str) -> str:
    if "/" not in script and ":" not in script and not script.endswith(".py"):
        return f"{CORE}:scripts/{script}.py"
    named = _repo_path(script)
    return f"{named[0]}:{named[1]}" if named else script


def _ray_script(script: str) -> str:
    if "/" not in script and ":" not in script and not script.endswith(".py"):
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


def main(argv: list[str]) -> None:
    opts, (script, *args) = _split(argv)
    ns, extra = parser().parse_known_args(opts)
    os.environ.pop("WT", None)  # --wt is the only worktree selector
    if not ns.on:
        if extra:
            sys.exit(f"[pls] run: {' '.join(extra)} only apply with --on")
        infra.run_script(script, args, ns.wt)
    elif ns.on == "ray":
        if extra:
            sys.exit(f"[pls] run --on ray: {' '.join(extra)} only apply on a card")
        env = {"WT": ns.wt} if ns.wt else None
        if script == "train":  # a training job: wrap_resources runs train.py (sweeps, aggregate jobs)
            handoff([RAY_BACKEND, "job", *args], env=env)
        else:
            handoff([RAY_BACKEND, "run", _ray_script(script), *args], env=env)
    else:
        wt = ["--wt", ns.wt] if ns.wt else []
        handoff([*CARD_CMD, *extra, *_card_flags(ns.on), *wt, _card_script(script), "--", *args])
