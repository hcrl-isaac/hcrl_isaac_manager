"""`pls run`: run a script or a command here, on a leased card (local, ssh, a cluster's held job), as a cluster
batch job, or on Ray."""

from __future__ import annotations

import argparse
import os
import shutil
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
MIN_FREE_GB, LOW_FREE_GB = 10, 50  # a run here is refused below the first and warned below the second

USAGE = """pls run <script> [args]
       pls run [--on TARGET] [--wt NAME] [card options] -- <script> [args]
       pls run [--on TARGET] [--wt NAME] [card options] --cmd -- <command> [args]
       pls run --on CLUSTER --batch [--tree N | --wt NAME] -- <script> [args]
       pls run --stop LEASE --holder YOU    (stop a --detach run on that lease and release the lease)"""
EPILOG = """targets (--on):
  (none)           this machine, with the ilab venv
  host:gpu         that card (host:job:gpu on a SLURM node running several jobs; local:<gpu> = this machine)
  any              any free card on the boxes
  <pool>           any free card of that pool (`pls res pools`)
  lease:<id>       a card you already lease (left leased afterwards)
  ray              the Ray cluster (queued until a GPU frees); `train` submits a training job, `--distributed`
                   one spanning a sub-job per GPU node
  <cluster> --batch  a batch job of its own on that SLURM cluster (queued, no held node needed): the workspace is
                   staged as tree `default` (or --wt NAME's worktree set as tree NAME) and the job runs that tree
                   as resolved now, even if newer trees are staged while it queues; --tree N runs a staged tree
                   as is. Any <name> or <repo>/<path> script; `pls jobs --on <cluster>` lists, follows and stops it

scripts (the same on every target):
  <name>           hcrl_isaaclab/scripts/<name>.py
  <repo>/<path>    a file of a workspace repo; --wt picks its worktree, and a card or Ray ships it with the run
  <repo>:<path>    the same
  <path>           any other path is a local file, relative to where you are (here or on a card; not on Ray
                   or in a batch job, which run only workspace repos' files)

--cmd runs the words after -- as a command instead (e.g. `--cmd -- nvidia-smi`, `--cmd -- python -m pkg`): here from
where you are, on a card from the box workspace (a SLURM card: the run's work dir), with the ilab venv first on PATH
(the container's python on SLURM). The box workspace's resources/ are that box's own checkouts, not the code the run
shipped: $PLS_RESOURCES is the dir holding the staged repos, e.g. `--cmd -- bash -c 'cd "$PLS_RESOURCES/ssti_tasks" &&
python -m pytest'`. Ray runs python scripts only.

On a card, {DEVICE} in the script's arguments becomes the leased card's torch device (`--device {DEVICE}`): boxes
pin cards differently, so a hand-written cuda:N that is not the card is refused.

card options (--holder is required): --holder, --note, --pool, --min-free-gb, --checkpoint, --env, --timeout,
--stall, --detach, --gpus N (N cards of one held SLURM node, one torchrun rank each; needs --on <node>), --cpus N,
--venv, --root (an Isaac Lab install root's venv, e.g. an ssh box's 3.x root), --with, --jobs/--out
(a job list over lease:A,B,...); `pls run --on any --help`
describes them.

CPU-only runs (--cpus N, --on <ssh host>): no card is leased; the run sees no GPU (CUDA_VISIBLE_DEVICES=""), is
pinned to N of the box's cores at nice 10 with OMP/MKL/OPENBLAS/NUMEXPR threads N, and its lease (<host>:cpu) lasts
as long as the run. Your CPU runs on a box hold at most half its cores (or the pool's cpu_budget). {DEVICE} is
refused; SLURM and Ray allocate CPUs themselves and refuse it. E.g.
  pls run --on <host> --cpus 24 --root <root> --detach --cmd -- python -m holosoma_retargeting ..."""


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
        "--wt",
        default="",
        metavar="NAME",
        help="run this machine's worktree set resources/<repo>/worktrees/<NAME>; a holosoma worktree in the set"
        " ships too, imported from its src/holosoma_retargeting",
    )
    p.add_argument("--cmd", action="store_true", help="the words after -- are a command, not a script")
    p.add_argument("--batch", action="store_true", help="submit a batch job on SLURM cluster TARGET (train only)")
    p.add_argument(
        "--tree", default="", metavar="N", help="(--batch or a SLURM card) run staged tree N instead of staging"
    )
    p.add_argument("--distributed", action="store_true", help="(--on ray, train) a sub-job per GPU node")
    p.add_argument(
        "--wt-data-ok",
        nargs="+",
        default=[],
        metavar="REL",
        help="(--on ray --wt) these artifact-only entries, as the preflight names them, may differ from W&B: the job"
        " trains on the published copy",
    )
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


def _disk_guard() -> None:
    """A run on this machine needs room: a full root disk kills every job and session writing to it."""
    free = shutil.disk_usage(ROOT).free / 2**30
    where = "free space first: du -xsh ~/tmp ~/.claude/jobs/*/tmp resources/*/worktrees/*"
    if free < MIN_FREE_GB:
        sys.exit(f"[pls] run: {free:.0f} GB free on this machine's disk (< {MIN_FREE_GB}); {where}")
    if free < LOW_FREE_GB:
        print(f"[pls] run: only {free:.0f} GB free on this machine's disk; {where}", file=sys.stderr)


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
        # Kit forks at startup; with scipy loaded first, OpenBLAS's atfork shutdown of its worker threads crashes it
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
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


def _repo_script(script: str, where: str) -> str:
    """``<repo>/<path>`` of a named script or a workspace repo's file: Ray and batch jobs run only those."""
    if _is_name(script):
        return f"{CORE}/scripts/{script}.py"
    named = _repo_path(script)
    if not named:
        sys.exit(f"[pls] run {where} runs a workspace repo's file (<name> or <repo>/<path>.py); {script} is not one")
    return f"{named[0]}/{named[1]}"


def _card_flags(on: str, cpus: bool = False) -> list[str]:
    """The card backend's selector for ``--on``; a ``--cpus`` run names a box, never a pool of cards."""
    if on == "any":
        return ["--any"]
    if on.startswith("lease:"):
        return ["--lease", on.removeprefix("lease:")]
    if ":" in on or cpus:
        return ["--on", on]
    return ["--any", "--pool", on]


def _has_cpus(extra: list[str]) -> bool:
    return any(a == "--cpus" or a.startswith("--cpus=") for a in extra)


NO_CPUS = "[pls] run {where}: --cpus runs on an ssh box's cores; {why}"


def _batch(on: str, wt: str, tree: str, run: list[str]) -> None:
    """A batch job on a SLURM profile: stage the code (the workspace, or the worktree set ``wt``), then submit."""
    from hcrl_cli.infra import _connected, profiles

    if on not in profiles():
        sys.exit(f"[pls] run --batch: --on names a SLURM cluster profile ({', '.join(profiles()) or 'none'}), not {on}")
    script, *args = run
    rel = _repo_script(script, "--batch")
    # the container starts in hcrl_isaaclab; the profile's own train.py stays as is, any other script replaces it
    job_script = [] if rel == f"{CORE}/scripts/train.py" else ["--script", f"/workspace/ext/{rel}"]
    _connected(on, "run --batch")
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
    handoff([SLURM_BACKEND, "job", *(["--tree", tree] if tree else []), *job_script, *args], env={"CLUSTER": on})


def main(argv: list[str]) -> None:
    if "--stop" in (argv[: argv.index("--")] if "--" in argv else argv):  # stop a detached run: no script
        handoff([*CARD_CMD, *argv])
        return
    opts, run = _split(argv)
    ns, extra = parser().parse_known_args(opts)
    os.environ.pop("WT", None)  # --wt is the only worktree selector
    if _has_cpus(extra) and (ns.batch or ns.on == "ray"):
        why = "Ray schedules a job's CPUs itself" if ns.on == "ray" else "a SLURM job's CPUs come from its allocation"
        sys.exit(NO_CPUS.format(where="--batch" if ns.batch else "--on ray", why=why))
    if ns.batch:
        if extra or ns.cmd:
            sys.exit(f"[pls] run --batch: {' '.join([*extra, *(['--cmd'] if ns.cmd else [])])} only apply on a card")
        _batch(ns.on, ns.wt, ns.tree, run)
        return
    if ns.tree and (not ns.on or ns.on == "ray" or ns.wt):
        sys.exit("[pls] run: --tree runs a staged tree as is, with --batch or on a SLURM card; not with --wt")
    if ns.distributed and (ns.on != "ray" or run[0] != "train"):
        sys.exit("[pls] run: --distributed only applies to train on Ray (--on ray -- train)")
    if ns.wt_data_ok and (ns.on != "ray" or not ns.wt):
        sys.exit("[pls] run: --wt-data-ok only applies to a worktree set on Ray (--on ray --wt NAME)")
    if not ns.on:
        if extra:
            sys.exit(f"[pls] run: {' '.join(extra)} only apply with --on")
        _disk_guard()
        _here(run, ns.wt, ns.cmd)
    elif ns.on == "ray":
        if extra or ns.cmd:
            sys.exit(f"[pls] run --on ray: {' '.join([*extra, *(['--cmd'] if ns.cmd else [])])} only apply on a card")
        script, *args = run
        env = {"WT": ns.wt} if ns.wt else {}
        if ns.wt_data_ok:
            env["WT_DATA_OK"] = os.pathsep.join(ns.wt_data_ok)
        env = env or None
        if script == "train":  # a training job: wrap_resources runs train.py (sweeps, aggregate jobs)
            handoff([RAY_BACKEND, "job_distributed" if ns.distributed else "job", *args], env=env)
        elif script == "bench":  # the throughput benchmark: one sub-job per GPU through wrap_resources
            handoff([RAY_BACKEND, "bench", *args], env=env)
        else:
            handoff([RAY_BACKEND, "run", _repo_script(script, "--on ray"), *args], env=env)
    else:
        script, *args = run
        wt = ["--wt", ns.wt] if ns.wt else ["--tree", ns.tree] if ns.tree else []
        if ns.on.startswith("local:"):
            _disk_guard()
        target = ["--cmd", script] if ns.cmd else [_card_script(script)]
        handoff([*CARD_CMD, *extra, *_card_flags(ns.on, _has_cpus(extra)), *wt, *target, "--", *args])
