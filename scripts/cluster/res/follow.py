"""What `pls run --detach` does once a train run starts: confirm it trains, arm its async recorder, then watch it.

The launch stays in the foreground on the watch (scripts/tools/watch_run.sh), so a session that runs it in the
background is woken when the run finishes, stalls, dies, fails or goes unreadable. Stopping the watch leaves the run.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from evaluate import Stage

HERE = Path(__file__).resolve().parent
WATCH = HERE.parents[1] / "tools" / "watch_run.sh"
ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
WANDB_RUN = re.compile(r"https://\S+?/([^\s/]+)/([^\s/]+)/runs/(\w+)")
START_S = 2400  # a first .sif copy, Kit's boot and asset imports all come before the first iteration
POLL_S = 20
RECORDER_GB = 12  # free memory one recorder's render worker (one env, RTX) needs on a shared card
RECORDER = "hcrl_isaaclab:scripts/video_recorder.py"
RECORDER_KIT = "--kit_args=--/renderer/multiGpu/enabled=false"


def _probe(stage: Stage, cmd: str) -> str:
    """What a shell command prints on the stage's host ("" when it fails or the host does not answer)."""
    if stage.t.kind == "local":
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    else:
        out = subprocess.run(["ssh", *stage.t.ssh_opts, stage.t.ssh, cmd], capture_output=True, text=True)
    return out.stdout


def started(
    stage: Stage,
    deadline_s: float = START_S,
    poll_s: float = POLL_S,
    clock: Callable = time.monotonic,
    sleep: Callable = time.sleep,
) -> tuple[str, int] | None:
    """The run's W&B run (``entity/project/id``, "" without one) and final iteration once it logs a first iteration.

    Returns:
        None when the run ended (a ``status``, a dead runner, a SLURM heartbeat 2 min old) or logged no iteration
        within ``deadline_s``.
    """
    d = shlex.quote(stage.dir)
    # a preflight refusal exits before any status is written, so a runner that is gone counts as ended too
    if stage.t.kind == "slurm":
        gone = f"[ -e {d}/heartbeat ] && [ $(( $(date +%s) - $(stat -c %Y {d}/heartbeat) )) -gt 120 ]"
    else:
        gone = f'p=$(cat {d}/pid 2>/dev/null) && [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null'
    cmd = (
        f"grep -m1 -oE 'Learning iteration [0-9]+/[0-9]+' {d}/log; grep -m1 -oE 'https://[^ ]+/runs/[A-Za-z0-9]+'"
        f" {d}/log; {{ [ -e {d}/status ] || {{ {gone}; }}; }} && echo __ENDED__"
    )
    end = clock() + deadline_s
    while True:
        out = _probe(stage, cmd)
        if it := ITERATION.search(out):
            run = WANDB_RUN.search(out)
            return ("/".join(run.groups()) if run else ""), int(it.group(2))
        if "__ENDED__" in out or clock() >= end:
            return None
        sleep(poll_s)


def eval_mode(script_args: list[str]) -> str:
    """How train.py records its evals and videos: ``--eval``, else ``--video``, else its default ``on``."""

    def value(flag: str) -> str | None:
        for i, a in enumerate(script_args):
            if a == flag and i + 1 < len(script_args):
                return script_args[i + 1]
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
        return None

    return value("--eval") or value("--video") or "on"


def arm_recorder(run: str, holder: str, wt: str, pools: list, launch: Callable = subprocess.run) -> str:
    """Start an async recorder that follows ``run`` into W&B on a shared card of a pool with ``recorder_root``.

    Returns:
        The pool it started on, or "" when no pool could start one.
    """
    entity, project, run_id = run.split("/")
    for pool in (p for p in pools if p.settings.get("recorder_root")):
        cmd = [sys.executable, str(HERE / "evaluate.py"), "--any", "--pool", pool.name, "--share", "--min-free-gb"]
        cmd += [str(RECORDER_GB), "--root", pool.settings["recorder_root"], "--holder", holder, "--detach"]
        cmd += [*(["--wt", wt] if wt else []), RECORDER, "--", "--mode", "async", "--follow", "--only_run", run_id]
        cmd += ["--wandb_project", f"{entity}/{project}", "--device", "{DEVICE}", RECORDER_KIT]
        if launch(cmd).returncode == 0:
            return pool.name
    return ""


def _control_path(ssh_opts: list[str]) -> str:
    """The ControlPath a SLURM login is reached over, so the watch reuses its master instead of a 2FA login."""
    return next((o.split("=", 1)[1] for o in ssh_opts if o.startswith("ControlPath=")), "")


def watch(stage: Stage, run: str, target: int, label: str, launch: Callable = subprocess.run) -> int:
    """Watch the run's log to its end with watch_run.sh and return its exit status (0 only once it finished)."""
    log = f"{stage.dir}/log" if stage.t.kind == "local" else f"{stage.t.ssh}:{stage.dir}/log"
    env = dict(os.environ)
    if path := _control_path(stage.t.ssh_opts if stage.t.kind == "slurm" else []):
        env["WATCH_RUN_CONTROL_PATH"] = path
    cmd = ["bash", str(WATCH), *(["--wandb", run] if run else []), label, log, str(target)]
    return launch(cmd, env=env).returncode


def _run_name(script_args: list[str], fallback: str) -> str:
    for i, a in enumerate(script_args):
        if a == "--run_name" and i + 1 < len(script_args):
            return script_args[i + 1]
        if a.startswith("--run_name="):
            return a.split("=", 1)[1]
    return fallback


def after_launch(stage: Stage, args: argparse.Namespace, script_args: list[str], pools: list) -> int:
    """Confirm the detached train run trains, arm its recorder when it records async, and watch it to its end.

    Returns:
        The launch's exit status: watch_run.sh's, or 1 when the run never logged an iteration.
    """
    where = stage.t.ssh or stage.t.host
    print("[res] waiting for its first training iteration (--no-watch returns at launch)", file=sys.stderr)
    got = started(stage)
    if got is None:
        print(f"[res] FAILED: the run logged no training iteration; log at {where}:{stage.dir}/log", file=sys.stderr)
        sys.stderr.write(_probe(stage, f"tail -n 8 {shlex.quote(stage.dir)}/log"))
        return 1
    run, target = got
    print(f"[res] training toward iteration {target}" + (f" (W&B {run})" if run else ""), file=sys.stderr)
    if eval_mode(script_args) == "async" and not getattr(args, "no_recorder", False):
        on = arm_recorder(run, args.holder, args.wt, pools) if run else ""
        if on:
            print(f"[res] armed an async recorder for {run} on {on}", file=sys.stderr)
        else:
            why = "the run logs to no W&B run" if not run else "no pool with recorder_root could start one"
            print(f"[res] NO RECORDER: {why}; its evals and videos stay unrecorded until one runs", file=sys.stderr)
    print("[res] watching it to its end (stopping this watch leaves the run running)", file=sys.stderr)
    return watch(stage, run, target, _run_name(script_args, os.path.basename(stage.dir)))
