"""Turn the launcher's argument list into one shell-quoted command per Ray job."""

from __future__ import annotations

import shlex


def split_jobs(tokens: list[str]) -> list[str]:
    """Group ``tokens`` into jobs at standalone ``*`` tokens and quote each job for the Ray entrypoint shell.

    Args:
        tokens: The launcher's arguments, e.g. ``["ray/wrap_resources.py", "--run_group", "push foot"]``.

    Returns:
        One command string per job; an argument with spaces or quotes stays one argument when the shell parses it.
    """
    jobs: list[list[str]] = [[]]
    for tok in tokens:
        if tok == "*":
            jobs.append([])
        else:
            jobs[-1].append(tok)
    return [shlex.join(job) for job in jobs if job]
