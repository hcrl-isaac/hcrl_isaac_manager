"""`pls run --on lease:A,B --jobs FILE --out DIR -- <script> [args]`: one script over a job list, one worker per card.

Each job is a normal card run on one of the leases (code shipped, checkpoints fetched, the lease left held), so a
batch behaves exactly like the runs it replaces. Results land in DIR as ``<id>.log`` and ``<id>.status``; a rerun
with the same DIR skips the jobs whose status is ``ok``.
"""

from __future__ import annotations

import contextlib
import json
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

EVALUATE = Path(__file__).resolve().parent / "evaluate.py"
MAX_FAILS_IN_A_ROW = 3  # a card that fails this many jobs in a row is broken, not the jobs: its worker stops
# backend options a batch decides per worker or per job, so they are not passed through to every job
OWN = {"--jobs": 1, "--out": 1, "--lease": 1, "--on": 1, "--any": 0, "--detach": 0}


@dataclass
class Job:
    """One run of the batch's script: its extra arguments, checkpoints and environment."""

    id: str
    args: list[str] = field(default_factory=list)
    checkpoint: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


def load_jobs(path: str) -> list[Job]:
    """Jobs from a JSON array or JSON lines of ``{"id", "args", "checkpoint", "env"}``.

    Args:
        path: The job file.

    Returns:
        The jobs, in file order; exits on a malformed entry or a repeated id.
    """
    text = Path(path).read_text()
    try:
        raw = json.loads(text)
        raw = raw if isinstance(raw, list) else [raw]
    except json.JSONDecodeError:
        try:
            raw = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            sys.exit(f"[res] --jobs {path}: neither a JSON array nor JSON lines ({exc})")
    jobs, seen = [], set()
    for i, entry in enumerate(raw):
        where = f"--jobs {path} entry {i}"
        if not isinstance(entry, dict) or set(entry) - {"id", "args", "checkpoint", "env"}:
            sys.exit(f"[res] {where}: a job is {{id, args, checkpoint, env}}; got {str(entry)[:120]}")
        job = Job(str(entry.get("id", "")), entry.get("args", []), entry.get("checkpoint", {}), entry.get("env", {}))
        if not re.fullmatch(r"[\w.+-]{1,120}", job.id):
            sys.exit(f"[res] {where}: id {job.id!r} must be 1-120 of letters, digits, . _ + - (it names files)")
        if job.id in seen:
            sys.exit(f"[res] {where}: id {job.id} repeats")
        if not (isinstance(job.args, list) and all(isinstance(a, str | int | float) for a in job.args)):
            sys.exit(f"[res] {where}: args is a list of strings")
        if not all(isinstance(d, dict) for d in (job.checkpoint, job.env)):
            sys.exit(f"[res] {where}: checkpoint and env map names to strings")
        job.args = [str(a) for a in job.args]
        seen.add(job.id)
        jobs.append(job)
    return jobs


def shared_options(argv: list[str]) -> list[str]:
    """The backend options before ``--`` that every job gets: all but the batch's own (``OWN``)."""
    out, i = [], 0
    while i < len(argv):
        word = argv[i]
        name = word.split("=", 1)[0]
        if name in OWN:
            i += 1 + (OWN[name] if "=" not in word else 0)
            continue
        out.append(word)
        i += 1
    return out


def job_argv(job: Job, lease: str, shared: list[str], script: str, common: list[str]) -> list[str]:
    """The `evaluate.py` command line that runs one job on one leased card."""
    flags = [f for name, ref in job.checkpoint.items() for f in ("--checkpoint", f"{name}={ref}")]
    flags += [f for key, val in job.env.items() for f in ("--env", f"{key}={val}")]
    return [sys.executable, str(EVALUATE), *shared, "--lease", lease, *flags, script, "--", *common, *job.args]


def status(out: Path, job: Job) -> str:
    try:
        return (out / f"{job.id}.status").read_text().strip()
    except OSError:
        return ""


def run_batch(argv: list[str], leases: list[str], jobs_file: str, out_dir: str, script: str, common: list[str]) -> int:
    """Run every job not yet ``ok`` in ``out_dir``, one worker per lease taking the next job when it is free.

    Args:
        argv: The backend arguments before ``--`` (the shared options are taken from them).
        leases: Lease ids, one card each; every job leaves its lease held, as ``--lease`` does.
        jobs_file: The job list.
        out_dir: Where each job's ``<id>.log`` and ``<id>.status`` go.
        script: The script (or command, with ``--cmd``) every job runs.
        common: Arguments every job's script gets before the job's own.

    Returns:
        0 if every job ended ``ok``, else 1.
    """
    if len(set(leases)) != len(leases):
        sys.exit("[res] --jobs: a lease is named twice; one worker per card")
    jobs = load_jobs(jobs_file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if status(out, j) != "ok"]
    print(
        f"[res] batch: {len(todo)} of {len(jobs)} jobs to run on {len(leases)} card(s); results in {out}",
        file=sys.stderr,
    )
    pending: queue.Queue[Job] = queue.Queue()
    for job in todo:
        pending.put(job)
    shared = shared_options(argv)
    lock, procs, stopping = threading.Lock(), {}, threading.Event()
    counts = {"ok": 0, "failed": 0}

    def worker(lease: str) -> None:
        fails = 0
        while not stopping.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            (out / f"{job.id}.status").write_text(f"running on lease {lease}\n")
            start = time.time()
            with open(out / f"{job.id}.log", "w") as log:
                proc = subprocess.Popen(
                    job_argv(job, lease, shared, script, common), stdout=log, stderr=subprocess.STDOUT
                )
                with lock:
                    procs[lease] = proc
                rc = proc.wait()
            minutes = (time.time() - start) / 60
            result = "ok" if rc == 0 else f"failed rc={rc}"
            if stopping.is_set() and rc != 0:
                result = "stopped"
            (out / f"{job.id}.status").write_text(result + "\n")
            with lock:
                procs.pop(lease, None)
                counts["ok" if rc == 0 else "failed"] += 1
                done = counts["ok"] + counts["failed"]
            print(f"[res] batch {done}/{len(todo)}: {job.id} {result} on lease {lease} ({minutes:.1f} min)",
                  file=sys.stderr)  # fmt: skip
            fails = 0 if rc == 0 else fails + 1
            if fails >= MAX_FAILS_IN_A_ROW:
                print(f"[res] batch: lease {lease} failed {fails} jobs in a row; its worker stops", file=sys.stderr)
                return

    def stop(signum: int, _frame: object) -> None:
        stopping.set()
        with lock:
            for proc in procs.values():  # each job's own handler kills its run and cleans its stage
                with contextlib.suppress(OSError):
                    proc.send_signal(signal.SIGTERM)

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        threads = [threading.Thread(target=worker, args=(lease,), daemon=True) for lease in leases]
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():
                thread.join(1.0)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    left = [j.id for j in jobs if status(out, j) != "ok"]
    print(f"[res] batch: {len(jobs) - len(left)} of {len(jobs)} ok" + (f"; not ok: {', '.join(left)}" if left else ""),
          file=sys.stderr)  # fmt: skip
    if left:
        print(f"[res] rerun the same command to run only those (status files in {out})", file=sys.stderr)
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit("[res] batch.py runs through `pls run --on lease:A,B --jobs FILE --out DIR -- <script>`")
