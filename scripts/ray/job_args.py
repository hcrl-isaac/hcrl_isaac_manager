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


def ext_script(spec: str) -> str:
    """The worker path of a script named as ``<repo>/<path inside the repo>``, as the job mounts it.

    Args:
        spec: E.g. ``hcrl_isaaclab/scripts/video_logger.py``.

    Returns:
        ``/workspace/ext/<spec>``.

    Raises:
        ValueError: For an absolute path, one that climbs out with ``..``, one without a repo, or a non-``.py`` file.
    """
    parts = spec.split("/")
    if spec.startswith("/") or ".." in parts or len(parts) < 2 or not spec.endswith(".py") or "" in parts:
        raise ValueError(f"script must be <repo>/<path>.py inside a shipped repo, got {spec!r}")
    return f"/workspace/ext/{spec}"
