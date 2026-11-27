"""`pls run --on A,B,C --seeds 42,43,44 --detach -- train ...`: one detached run per (card, seed), launched at once.

Each run is an ordinary `pls run --on <card> --detach` launch of the same script with ``--seed <n>``. Every launch's
output streams live, prefixed with its seed and card, so each run's start (or failure) is reported as it happens.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable

import evaluate as ev


def _pop(head: list[str], flag: str) -> tuple[list[str], str]:
    """``head`` without ``flag`` and its value, and the value ("" when absent)."""
    out, value, i = [], "", 0
    while i < len(head):
        a = head[i]
        if a == flag and i + 1 < len(head):
            value, i = head[i + 1], i + 2
            continue
        if a.startswith(flag + "="):
            value = a.split("=", 1)[1]
        else:
            out.append(a)
        i += 1
    return out, value


def seeded(args: list[str], seed: int) -> list[str]:
    """The script's arguments with ``--seed <seed>``; the run name stays the change, the seed is in the W&B config."""
    args, _ = _pop(args, "--seed")
    return [*args, "--seed", str(seed)]


def plan(argv: list[str]) -> list[tuple[str, int, list[str]]]:
    """One (where, seed, argv) per run; exits on a launch `--seeds` cannot split.

    The cards come from ``--on A,B`` (one per seed), ``--lease a,b`` (one lease per seed), or ``--any`` (with or
    without ``--pool``), which each run claims a free card of on its own.
    """
    if "--" not in argv:
        sys.exit("[seeds] name the script after the options: pls run --on A,B --seeds 1,2 --detach -- train ...")
    cut = argv.index("--")
    head, script_args = argv[:cut], argv[cut + 1 :]
    head, raw = _pop(head, "--seeds")
    head, on = _pop(head, "--on")
    head, lease = _pop(head, "--lease")
    try:
        seeds = [int(s) for s in raw.split(",") if s]
    except ValueError:
        sys.exit(f"[seeds] --seeds takes comma-separated integers, not {raw!r}")
    if not seeds or len(set(seeds)) != len(seeds):
        sys.exit("[seeds] --seeds takes distinct seeds, e.g. --seeds 42,43,44")
    if "--detach" not in head:
        sys.exit("[seeds] --seeds launches detached runs: add --detach")
    for flag in ("--gpus", "--jobs", "--cpus", "--cmd"):
        if flag in head or any(a.startswith(flag + "=") for a in head):
            sys.exit(f"[seeds] --seeds runs one card per seed; it does not combine with {flag}")
    if on or lease:
        where = [w for w in (on or lease).split(",") if w]
        if len(where) != len(seeds):
            sys.exit(f"[seeds] {len(where)} card(s) for {len(seeds)} seed(s): give one card (or lease) per seed")
        pick = [["--on", w] if on else ["--lease", w] for w in where]
    elif "--any" in head:
        where, pick = ["any free card"] * len(seeds), [[]] * len(seeds)  # --any (and --pool) stay in the options
    else:
        sys.exit("[seeds] pick the cards with --on A,B,C (one per seed), --on lease:a,b, or a pool (any free cards)")
    *opts, script = head  # the script is the last word before --
    return [
        (w, s, [*opts, *p, script, "--", *seeded(script_args, s)]) for w, s, p in zip(where, seeds, pick, strict=True)
    ]


def main(argv: list[str], popen: Callable = subprocess.Popen) -> int:
    """Launch every (card, seed) run at once and stream their output; the worst launch status is returned."""
    runs = plan(argv)
    codes: dict[int, int] = {}
    lock = threading.Lock()

    def one(where: str, seed: int, run_argv: list[str]) -> None:
        tag = f"[seed={seed}]" if where == "any free card" else f"[seed={seed} {where}]"
        proc = popen([sys.executable, ev.__file__, *run_argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)  # fmt: skip
        for line in proc.stdout:
            with lock:
                print(f"{tag} {line}", end="", file=sys.stderr, flush=True)
        codes[seed] = proc.wait()

    print(f"[seeds] launching {len(runs)} detached runs: " + ", ".join(f"seed {s} on {c}" for c, s, _ in runs),
          file=sys.stderr)  # fmt: skip
    threads = [threading.Thread(target=one, args=r, daemon=True) for r in runs]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    done = "launched" if "--no-watch" in argv[: argv.index("--")] else "finished"  # a watched launch ends with its run
    for where, seed, _ in runs:
        rc = codes.get(seed, 1)
        print(f"[seeds] seed {seed} ({where}): {done if rc == 0 else f'exit {rc}'}", file=sys.stderr)
    return max(codes.values(), default=1)
