"""Refuse a Ray submit while another of this user's jobs is running or pending on the cluster.

The Ray cluster is one node, and every job copies its code into, and on exit deletes, the same ``/workspace/ext``, so
two jobs at once can delete each other's code mid-run. ``pls run --on ray --alongside`` submits anyway.

usage: busy.py ADDRESS USER_ID
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import urllib.request

LIVE = ("RUNNING", "PENDING")


def live_jobs(jobs: list[dict], user_id: str) -> list[dict]:
    """This user's jobs that are running or waiting for a card, oldest first."""
    mine = [j for j in jobs if (j.get("metadata") or {}).get("user_id") == user_id and j.get("status") in LIVE]
    return sorted(mine, key=lambda j: j.get("start_time") or 0)


def describe(job: dict) -> str:
    """One line naming a job: its id, state, start, worktree set and task."""
    start = job.get("start_time")
    when = datetime.datetime.fromtimestamp(start / 1000).strftime("%m-%d %H:%M") if start else "?"
    task = re.search(r"--task[ =](\S+)", job.get("entrypoint") or "")
    wt = (job.get("metadata") or {}).get("wt") or "main"
    return f"{job.get('submission_id')} {job.get('status')} since {when}, --wt {wt}, task {task[1] if task else '?'}"


def main() -> None:
    address, user_id = sys.argv[1], sys.argv[2]
    try:
        with urllib.request.urlopen(f"{address.rstrip('/')}/api/jobs/", timeout=30) as r:
            jobs = json.load(r)
    except Exception as exc:  # cannot tell: submitting blind is no worse than before this check
        print(f"[WARN] could not list the Ray cluster's jobs ({exc}); not checking for a running one", file=sys.stderr)
        return
    live = live_jobs(jobs, user_id)
    if not live or os.environ.get("RAY_ALONGSIDE"):
        return
    print("[ERROR] another Ray job of yours is on the cluster:", file=sys.stderr)
    for job in live:
        print(f"  {describe(job)}", file=sys.stderr)
    print(
        "  The cluster is one node whose jobs share /workspace/ext, and a job's exit deletes the other's code. Wait"
        " for it, or pass --alongside (pls run --on ray --alongside ...) to submit anyway.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
