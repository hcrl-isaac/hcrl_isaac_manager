"""Resume a detached SLURM train run its hold's end cut short (walltime, node failure, preemption).

The run's watch calls :func:`after_cut` once the run stops logging. When sacct says the hold ended under it, the
cut hold gets one replacement hold (queued by the first of its runs to notice), and each cut run relaunches on the
same card of the new node from its newest W&B checkpoint, with its recorded launch, code and remaining budget, as
`pls run --move` relaunches. A run stopped on purpose, finished, or crashed on a live hold is never resumed.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import evaluate as ev
import leases as ls
import move
import res
from probe import MASTER_OPTS, SSH_OPTS
from sshslot import SSH

CUT_STATES = ("TIMEOUT", "NODE_FAIL", "PREEMPTED")  # sacct states a hold ends in without anyone asking
MAX_RESUMES = 3  # per run chain, so a run its node keeps killing cannot loop
PENDING_NOTE_S = 6 * 3600  # a replacement hold still queued this long gets one note with its queue reason
POLL_S = 60.0
VERIFY_S = 2400.0  # a relaunch boots Kit and loads its checkpoint before its first iteration
EVENTS = ls.STORE.parent / "events.jsonl"
REPLACEMENTS = ls.STORE.parent / "replacements.json"
NO_RESUME = "--no-resume"
SUBMITTED = re.compile(r"Submitted sentinel job (\d+)")  # cluster_dev.sh's line for the hold it queued


def event(**row: object) -> None:
    """Append one line to the event log `pls res events` prints."""
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as f:
        f.write(json.dumps({"ts": int(time.time()), **row}) + "\n")


@contextmanager
def _locked() -> Iterator[dict]:
    """``holds`` (cut job id -> replacement job id) and ``pending`` (stage dir -> card), under a file lock."""
    REPLACEMENTS.parent.mkdir(parents=True, exist_ok=True)
    with (REPLACEMENTS.parent / "replacements.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = json.loads(REPLACEMENTS.read_text()) if REPLACEMENTS.is_file() else {}
        table.setdefault("holds", {})
        table.setdefault("pending", {})
        table.setdefault("resumed", {})
        yield table
        REPLACEMENTS.write_text(json.dumps(table))


def pending_stages(holder: str, pools: list) -> list[tuple[str, ev.Stage]]:
    """``holder``'s cut runs whose resume a watch began and never ended, as (label, stage) for `pls run --watch`."""
    with _locked() as table:
        rows = [(d, r) for d, r in table["pending"].items() if r["holder"] == holder]
    by_name = {p.name: p for p in pools}
    out = []
    for d, r in rows:
        if r["pool"] in by_name:
            stage = ev.Stage(ev.make_target(by_name[r["pool"]], r["host"], r["gpu"], r["job"], r["uuid"]))
            stage.dir = d
            out.append((f"resume:{r['job']}", stage))
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(stage: ev.Stage, holder: str) -> str:
    """Claim the resume of ``stage``'s run for this watch; "" when claimed, else why another watch has it.

    Two watches on one run (a re-arm beside a live watch) would each relaunch it: the first to claim resumes it, a
    claim whose watch died passes to the next, and a run already relaunched is never relaunched again.
    """
    t = stage.t
    with _locked() as table:
        if done := table["resumed"].get(stage.dir):
            return f"already resumed as {done}"
        row = table["pending"].get(stage.dir) or {}
        pid = int(row.get("pid") or 0)
        if pid and pid != os.getpid() and _alive(pid):
            return f"another watch (pid {pid}) is resuming it"
        table["pending"][stage.dir] = {"holder": holder, "pool": t.pool, "host": t.host, "gpu": t.gpu,
                                       "job": t.job, "uuid": t.uuid, "pid": os.getpid()}  # fmt: skip
    return ""


def _release(stage: ev.Stage) -> None:
    """Drop ``stage``'s pending row once this watch decided it; a row another live watch claimed stays theirs."""
    with _locked() as table:
        row = table["pending"].get(stage.dir) or {}
        pid = int(row.get("pid") or 0)
        if not row.get("failed") and (not pid or pid == os.getpid() or not _alive(pid)):
            table["pending"].pop(stage.dir, None)


def _leave_pending(stage: ev.Stage, new_job: str) -> None:
    """Keep ``stage``'s row after its resume failed, unclaimed, so `pls run --watch --holder` adopts the run."""
    with _locked() as table:
        if row := table["pending"].get(stage.dir):
            row.update(pid=0, failed=new_job)


def _mark_resumed(old_dir: str, new_dir: str) -> None:
    with _locked() as table:
        table["resumed"][old_dir] = new_dir


def _login(t: ev.Target, cmd: str, timeout: float = 60) -> str:
    """What ``cmd`` prints on the cluster's login node ("" when it fails)."""
    try:
        out = subprocess.run(
            [SSH, *MASTER_OPTS, *SSH_OPTS, t.ssh, cmd], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return ""
    return out.stdout if out.returncode == 0 else ""


def hold_end(t: ev.Target) -> dict[str, str]:
    """The hold job's sacct record: State, Partition, Timelimit, NodeList (empty when sacct cannot say)."""
    out = _login(t, f"sacct -j {shlex.quote(t.job)} -X -n -P -o State,Partition,Timelimit,NodeList")
    line = next((x for x in out.splitlines() if x.strip()), "")
    keys = ("state", "partition", "timelimit", "nodes")
    vals = line.split("|") + [""] * len(keys)
    rec = dict(zip(keys, vals[: len(keys)], strict=True))
    rec["state"] = rec["state"].split()[0] if rec["state"] else ""  # "CANCELLED by 123" -> CANCELLED
    return rec


def verdict(stage_files: dict[str, str | None], hold_state: str) -> str:
    """``cut`` when the hold's end killed a run still short of its target, else why the run is left alone.

    Args:
        stage_files: The stage's ``status``, ``stop`` and ``pls_argv.json`` contents (None when absent).
        hold_state: The hold job's sacct state.
    """
    if stage_files.get("stop") is not None:
        return "stopped on purpose"
    status = (stage_files.get("status") or "").strip()
    if status == "0":
        return "finished"
    if hold_state not in CUT_STATES:
        return f"not cut (hold {hold_state or 'state unknown'})"
    argv = stage_files.get("pls_argv.json")
    if argv is None:
        return "launched before --move: no recorded launch to resume"
    if NO_RESUME in json.loads(argv):
        return "launched with --no-resume"
    return "cut"


def replacement(t: ev.Target, rec: dict[str, str], holder: str) -> str:
    """The job id of the hold that replaces ``t.job``: queued here unless another cut run queued it first."""
    with _locked() as table:
        if t.job in table["holds"]:
            return table["holds"][t.job]
        args = ["start", "--no-stage"]
        if rec.get("partition"):
            args += ["--partition", rec["partition"]]
        if rec.get("timelimit"):
            args += ["--time", rec["timelimit"]]
        cmd, env = ev.cluster_dev_cmd(t.pool, "", *args)
        # a hold changes the shared sentinel state, which only the main checkout's cluster_dev.sh may write
        cmd[1] = str(Path(ev.local_workspace()) / "scripts" / "cluster" / "cluster_dev" / "cluster_dev.sh")
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stderr.write(out.stderr)
        found = SUBMITTED.search(out.stdout + out.stderr)
        if out.returncode != 0 or not found:
            raise SystemExit(f"[resume] could not queue a hold to replace job {t.job} on {t.pool}")
        table["holds"][t.job] = found[1]
        event(kind="hold_queued", cut_job=t.job, new_job=found[1], cluster=t.pool, holder=holder, **rec)
        print(f"[resume] queued hold {found[1]} on {t.pool} to replace job {t.job}", file=sys.stderr)
        return found[1]


def forget(cut_job: str) -> None:
    """Drop the cut job's replacement record, so the next resume of its runs queues a fresh hold."""
    with _locked() as table:
        table["holds"].pop(cut_job, None)


SLURM_LIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED", "REQUEUE_HOLD", "RESIZING"}
"""sacct states of a job that has not ended: squeue failing to list one says nothing about it."""


def wait_running(t: ev.Target, job: str, holder: str, sleep: Callable = time.sleep) -> str:
    """The node of ``job`` once it runs; notes a long queue once, and exits if the job ends without running."""
    start, noted = time.monotonic(), False
    while True:
        out = _login(t, f"squeue -j {shlex.quote(job)} -h -o '%T|%N|%r'").strip()
        state, node, reason = ([*out.split("|"), "", ""])[:3] if out else ("", "", "")
        if state == "RUNNING" and node:
            return node
        # an empty squeue is also a failed call: only sacct's record of an end means the hold is gone
        acct = _login(t, f"sacct -j {shlex.quote(job)} -X -n -o State").split() if not out else []
        if acct and acct[0] not in SLURM_LIVE:
            raise SystemExit(f"[resume] replacement hold {job} ended {acct[0]} before it ran")
        if not noted and time.monotonic() - start > PENDING_NOTE_S:
            note = (f"[resume] hold {job} still {state or 'queued'} after 6 h ({reason}); a shorter --time or another"
                    " partition or cluster (pls res status) would start sooner")  # fmt: skip
            print(note, file=sys.stderr)
            event(kind="hold_pending", job=job, cluster=t.pool, reason=reason, holder=holder)
            noted = True
        sleep(POLL_S)


def chain_count(stage: ev.Stage) -> int:
    """How many times this run's chain was already resumed (the stage's ``resumes`` file)."""
    raw = move.read_stage(stage, "resumes")
    return int(raw.strip()) if raw and raw.strip().isdigit() else 0


def after_cut(stage: ev.Stage, holder: str, label: str) -> ev.Stage | None:
    """Resume ``stage``'s run if its hold's end cut it; returns the resumed run's stage, else None (said why)."""
    try:
        return _after_cut(stage, holder, label)
    finally:
        _release(stage)  # decided; a watch that dies mid-resume never gets here, so its claim stays


def _after_cut(stage: ev.Stage, holder: str, label: str) -> ev.Stage | None:
    rec = hold_end(stage.t)
    files = {name: move.read_stage(stage, name) for name in ("status", "stop", "pls_argv.json")}
    why = verdict(files, rec.get("state", ""))
    if why != "cut":
        print(f"[resume] {label}: not resumed: {why}", file=sys.stderr)
        return None
    done = chain_count(stage)
    if done >= MAX_RESUMES:
        print(f"[resume] {label}: not resumed: already resumed {done} times", file=sys.stderr)
        event(kind="resume_refused", label=label, holder=holder, why=f"{done} resumes")
        return None
    opts, script, args = move.split_launch(json.loads(files["pls_argv.json"] or "[]"))
    state = move.progress(move.read_stage(stage, "log") or "")
    if state is None:
        print(f"[resume] {label}: not resumed: the run logged no W&B run and iteration to resume from", file=sys.stderr)
        return None
    url, _, total = state
    have = move.wandb_checkpoints(url)
    if not have:
        print(f"[resume] {label}: not resumed: no checkpoint on {url} yet", file=sys.stderr)
        return None
    ckpt = max(have)
    print(f"[resume] {label}: hold {stage.t.job} ended {rec['state']} at checkpoint {ckpt}/{total}", file=sys.stderr)
    # the claim is left in place by a watch that dies mid-resume, for `pls run --watch` to pick up
    if taken := claim(stage, holder):
        print(f"[resume] {label}: not resumed here: {taken}", file=sys.stderr)
        return None
    return _resume(stage, holder, label, rec, done, (opts, script, args), state, ckpt)


def _resume(
    stage: ev.Stage,
    holder: str,
    label: str,
    rec: dict[str, str],
    done: int,
    launch: tuple[list[str], str, list[str]],
    state: tuple[str, int, int],
    ckpt: int,
) -> ev.Stage | None:
    opts = launch[0]
    new_job = replacement(stage.t, rec, holder)
    try:
        node = wait_running(stage.t, new_job, holder)
    except SystemExit as e:
        forget(stage.t.job)
        _leave_pending(stage, new_job)
        event(kind="resume_failed", label=label, holder=holder, cut_job=stage.t.job, new_job=new_job, why=str(e))
        print(f"{e}; `pls run --watch --holder {holder}` on the run queues a fresh one", file=sys.stderr)
        return None
    # a hold can queue for hours: a run stopped meanwhile (`pls run --stop`) stays stopped
    if move.read_stage(stage, "stop") is not None:
        print(f"[resume] {label}: not resumed: stopped on purpose while hold {new_job} queued", file=sys.stderr)
        event(kind="resume_refused", label=label, holder=holder, new_job=new_job, why="stopped while its hold queued")
        return None
    gpus = int(move.launch_value(opts, "--gpus") or 1)
    on = f"{node}:{new_job}" if gpus > 1 else f"{node}:{new_job}:{stage.t.gpu}"
    wt, made = move.pin_code(stage, f"resume-{stage.t.job}-{int(time.time())}")
    try:  # the pinned code stays until the resumed run's recorder is armed from it
        return _relaunch(stage, holder, label, rec, done, launch, state, ckpt, new_job, on, gpus, wt)
    finally:
        move.unpin(made)


def _relaunch(
    stage: ev.Stage,
    holder: str,
    label: str,
    rec: dict[str, str],
    done: int,
    launch: tuple[list[str], str, list[str]],
    state: tuple[str, int, int],
    ckpt: int,
    new_job: str,
    on: str,
    gpus: int,
    wt: str,
) -> ev.Stage | None:
    opts, script, args = launch
    url, _, total = state
    drop = f"pls res drop --on {stage.t.pool} {new_job}"
    resume = ["--load_run", url, "--checkpoint", f"model_{ckpt}.pt", "--max_iterations", str(total - ckpt)]
    where = ["--on", on, *(["--gpus", str(gpus), "--wait", "10m"] if gpus > 1 else [])]
    kept = [*move.drop_card_selection(_drop_holder(opts)), *(["--wt", wt] if wt else [])]
    argv = [*kept, *where, "--holder", holder]
    argv += ["--detach", "--no-watch", script, "--", *move.drop_resume(args), *resume]
    out = subprocess.run([sys.executable, ev.__file__, *argv], capture_output=True, text=True)
    sys.stderr.write(out.stderr)
    found = move.STARTED.findall(out.stderr)
    if out.returncode != 0 or not found:
        event(kind="resume_failed", label=label, holder=holder, cut_job=stage.t.job, new_job=new_job, ckpt=ckpt,
              drop=drop)  # fmt: skip
        print(f"[resume] {label}: the relaunch on hold {new_job} failed (its output above); hold {new_job} stays held"
              f" to relaunch it by hand from model_{ckpt}.pt (`{drop}` releases it)", file=sys.stderr)  # fmt: skip
        return None
    new = ev.Stage(ev.make_target(*_card(found[-1], holder)))
    new.dir = found[-1]
    _mark_resumed(stage.dir, new.dir)  # relaunched, confirmed or not: no other watch relaunches it again
    _write(new, "resumes", str(done + 1))
    verdict_new = move.confirm(new, ckpt, time.monotonic() + VERIFY_S)
    moved = move.progress(move.read_stage(new, "log") or "")
    new_url = moved[0] if moved else ""
    event(kind="resumed" if verdict_new == "ok" else "resume_unconfirmed", label=label, holder=holder,
          cut_job=stage.t.job, cut_state=rec["state"], new_job=new_job, card=on, ckpt=ckpt, url=new_url,
          stage=new.dir)  # fmt: skip
    print(f"[resume] {label}: cut at iter {ckpt} (hold {stage.t.job} {rec['state']}); resumed on hold {new_job},"
          f" card {on}" + (f", as {new_url}" if new_url else "") + (f" ({verdict_new})" if verdict_new != "ok" else ""),
          file=sys.stderr)  # fmt: skip
    if verdict_new != "ok":
        return None
    move.arm_moved_recorder(new, holder, wt, opts, args, res.load_pools())  # its new W&B run, as its launch's had
    return new


def _drop_holder(opts: list[str]) -> list[str]:
    """The launch options without ``--holder``, which the resume passes as the watching session's."""
    out, skip = [], False
    for a in opts:
        if skip:
            skip = False
        elif a == "--holder":
            skip = True
        elif not a.startswith("--holder="):
            out.append(a)
    return out


def _card(stage_dir: str, holder: str) -> tuple:
    """The make_target arguments of the card the relaunch leased: (pool, host, gpu, job, uuid)."""
    return ev._lease_card(move.recording_lease(stage_dir), holder, res.load_pools())


def _write(stage: ev.Stage, rel: str, text: str) -> None:
    move.follow._probe(stage, f"echo {shlex.quote(text)} > {shlex.quote(stage.dir)}/{rel}")


def print_events(follow: bool = False, sleep: Callable = time.sleep) -> None:
    """`pls res events [--follow]`: every resume and replacement hold, one JSON line each."""
    seen = 0
    while True:
        lines = EVENTS.read_text().splitlines() if EVENTS.is_file() else []
        for line in lines[seen:]:
            print(line, flush=True)
        seen = len(lines)
        if not follow:
            return
        sleep(5)
