"""`pls run --move <lease>`: carry a detached train.py run to another card from its next checkpoint.

The run must have been launched by a `pls run` that records its argv in the stage (``pls_argv.json``). The relaunch
trains the code the run ran: worktrees at its recorded commits, whatever the checkouts hold now. The move claims the
target first, waits for a checkpoint newer than any already on W&B, stops the run (keeping the lease when the card
stays the same), relaunches it detached from that checkpoint with the remaining iteration budget, and confirms it
trains past it.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import evaluate as ev
import follow
import leases as ls
from inventory import Pool

TRAIN = "scripts/train.py"
RESUME_FLAGS = ("--load_run", "--checkpoint", "--max_iterations", "--resume")
POLL_S = 30.0
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# any W&B host (wandb.ai, or a dedicated one such as forge.coreweave.com/wandb): .../<entity>/<project>/runs/<id>
WANDB_URL = re.compile(r"https://\S+?/[^\s/]+/[^\s/]+/runs/\w+")
ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
STARTED = re.compile(r"started detached on .* \(stage (\S+)\)")
FAILURES = (
    "Traceback",
    "There was an error running python",
    "Could not override",
    "CUDA out of memory",
    "Segmentation fault",
    "Killed",
    "srun: error",
)


def read_stage(stage: ev.Stage, rel: str) -> str | None:
    """A text file of the stage, or None when it is missing."""
    if stage.t.kind == "local":
        path = Path(stage.dir, rel)
        return path.read_text() if path.is_file() else None
    out = stage._ssh(f"cat {shlex.quote(f'{stage.dir}/{rel}')} 2>/dev/null", capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def split_launch(argv: list[str]) -> tuple[list[str], str, list[str]]:
    """The backend options, the script (the last word before ``--``, where `pls run` puts it) and its arguments."""
    i = argv.index("--") if "--" in argv else len(argv)
    if i == 0:
        raise ValueError("the recorded launch names no script")
    return argv[: i - 1], argv[i - 1], argv[i + 1 :]


def drop_card_selection(opts: list[str]) -> list[str]:
    """The launch options without the card selection, --wt and --detach, which the move sets itself."""
    sel = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ("--on", "--lease", "--min-free-gb", "--gpus", "--wt"):
        sel.add_argument(flag)
    sel.add_argument("--pool", action="append")
    sel.add_argument("--wait", nargs="?", const="")
    for flag in ("--any", "--share", "--detach"):
        sel.add_argument(flag, action="store_true")
    return sel.parse_known_args(opts)[1]


def launch_value(opts: list[str], flag: str) -> str:
    """The value the recorded launch gave ``flag`` ("" when it gave none)."""
    for i, a in enumerate(opts):
        if a == flag and i + 1 < len(opts):
            return opts[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return ""


def drop_resume(args: list[str]) -> list[str]:
    """The script arguments without a previous resume (each of RESUME_FLAGS takes a value)."""
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        flag, eq, _ = arg.partition("=")
        if flag in RESUME_FLAGS:
            skip = not eq
            continue
        out.append(arg)
    return out


def progress(log: str) -> tuple[str, int, int] | None:
    """The run's W&B URL and its last ``Learning iteration k/N`` (k, N), or None before it trains."""
    text = ANSI.sub("", log)
    urls, its = WANDB_URL.findall(text), ITERATION.findall(text)
    if not urls or not its:
        return None
    return urls[-1], int(its[-1][0]), int(its[-1][1])


def unpin(made: list[tuple[str, str]]) -> None:
    """Remove the worktrees :func:`pin_code` made."""
    for main, path in made:
        subprocess.run(["git", "-C", main, "worktree", "remove", "--force", path], capture_output=True)


def pin_code(stage: ev.Stage, tag: str) -> tuple[str, list[tuple[str, str]]]:
    """A worktree set named ``tag`` at the commits the run ran ("" for a --tree run, which reuses its tree).

    Returns:
        The set's name and the (main checkout, worktree) pairs made; exits when the run shipped uncommitted changes
        or a commit is not here, since then its code cannot be rebuilt.
    """
    raw = read_stage(stage, "wandb_config.yaml")
    if raw is None:
        sys.exit("[move] the run recorded no code refs (wandb_config.yaml), so its code cannot be rebuilt")
    refs = json.loads(raw)["pls_refs"]["value"]
    if refs.get("tree"):
        return "", []
    workspace = ev.local_workspace()
    code = ev.local_code(workspace, refs.get("wt") or "")
    made: list[tuple[str, str]] = []
    for repo, ran in refs.get("repos", {}).items():
        if repo not in code:  # data and asset repos ship as any launch ships them
            continue
        main = str(Path(workspace, "resources", repo))
        path = str(Path(main, "worktrees", tag))
        if ran.get("dirty"):
            unpin(made)
            sys.exit(f"[move] the run shipped uncommitted changes in {repo}, so its code cannot be rebuilt")
        out = subprocess.run(["git", "-C", main, "worktree", "add", "--detach", path, str(ran.get("commit"))],
                             capture_output=True, text=True)  # fmt: skip
        if out.returncode != 0:
            unpin(made)
            sys.exit(f"[move] cannot check out {repo} at {str(ran.get('commit'))[:8]}: {out.stderr.strip()}")
        made.append((main, path))
    return tag, made


def wandb_checkpoints(url: str) -> dict[int, int]:
    """Iteration -> size of each ``model_<it>.pt`` the W&B run holds."""
    import wandb

    head, _, run_id = url.partition("/runs/")
    entity, project = head.rstrip("/").split("/")[-2:]
    run = wandb.Api().run(f"{entity}/{project}/{run_id.split('/')[0]}")
    found = {}
    for f in run.files():
        if m := re.fullmatch(r"model_(\d+)\.pt", f.name):
            found[int(m.group(1))] = int(f.size or 0)
    return found


def resume_warmup(url: str) -> int:
    """Iterations an off-policy resume of the run collects before it updates (its cfg's resume warm-up), else 0."""
    import wandb

    head, _, run_id = url.partition("/runs/")
    entity, project = head.rstrip("/").split("/")[-2:]
    cfg = wandb.Api().run(f"{entity}/{project}/{run_id.split('/')[0]}").config.get("train_cfg", {})
    if not str(cfg.get("class_name", "")).endswith("OffPolicyRunner"):
        return 0
    return int(cfg.get("resume_warmup_steps_per_env") or 0)


def wait_checkpoint(url: str, after: int, deadline: float, poll_s: float = POLL_S) -> int | None:
    """The first checkpoint past ``after`` whose upload finished (its size held over one poll), or None."""
    seen: dict[int, int] = {}
    while time.monotonic() < deadline:
        newer = {k: s for k, s in wandb_checkpoints(url).items() if k > after and s > 0}
        for k in sorted(newer, reverse=True):
            if seen.get(k) == newer[k]:
                return k
        seen = newer
        time.sleep(poll_s)
    return None


def target_lease(to: str, lease_id: str, holder: str, pools: list[Pool]) -> str:
    """The lease the run moves onto: its own (``same``), one the holder has (``lease:<id>``), or a new claim."""
    if to == "same":
        return lease_id
    if to.startswith("lease:"):
        ev._lease_card(to.removeprefix("lease:"), holder, pools)  # exits unless the holder has it
        return to.removeprefix("lease:")
    import res

    ns = argparse.Namespace(
        cards=[to], any=False, count=1, min_free_gb=0, pool=None, holder=holder,
        note=f"move target of {lease_id}", for_=0.0,
    )  # fmt: skip
    (lease, _, _), *_ = res.claim(ns, pools)[0]
    print(f"[move] leased {lease.id}: {lease.card} for the move", file=sys.stderr)
    return lease.id


def target_node(to: str, node: str, gpus: int, lease_id: str, holder: str, pools: list[Pool]) -> str:
    """The node a ``--gpus`` run moves onto: its own (``same``, whose leases the stop keeps) or another one held,
    whose ``gpus`` cards are claimed first so the run never stops without somewhere to go."""
    if to == "same":
        return node
    if to.startswith("lease:"):
        sys.exit(f"[move] a --gpus {gpus} run moves to a node (host:job, as --on takes it), not to one lease")
    import res

    ns = argparse.Namespace(
        cards=[], any=True, count=gpus, within=ev.resolve_on(to, pools), one_host=True, min_free_gb=0, share=False,
        pool=None, holder=holder, note=f"move target of {lease_id}", for_=0.0, wait=None,
    )  # fmt: skip
    for lease, _, _ in res.claim(ns, [p for p in pools if p.kind in ev.SUPPORTED])[0]:
        print(f"[move] leased {lease.id}: {lease.card} for the move", file=sys.stderr)
    return to


def recording_lease(stage_dir: str) -> str:
    """A lease that records ``stage_dir`` (each card of a ``--gpus`` run's does)."""
    with ls.locked_store() as leases:
        return next((x.id for x in leases if stage_dir in x.stages or x.stage == stage_dir), "")


def relaunch(
    opts: list[str], script: str, args: list[str], lease_id: str, wt: str, on: str = "", gpus: int = 1
) -> tuple[int, str]:
    """Start the run detached on ``lease_id``, or a ``--gpus`` run on ``on``'s cards (from worktree set ``wt``),
    through `pls run`'s own path; its status and new stage dir."""
    pinned = ["--wt", wt] if wt else []
    # the stopped run's cards may still be draining: a --gpus claim waits for them to free
    where = ["--on", on, "--gpus", str(gpus), "--wait", "10m"] if gpus > 1 else ["--lease", lease_id]
    argv = [*drop_card_selection(opts), *pinned, *where, "--detach", "--no-watch", script, "--", *args]
    out = subprocess.run([sys.executable, ev.__file__, *argv], capture_output=True, text=True)
    sys.stderr.write(out.stderr)
    found = STARTED.findall(out.stderr)
    return out.returncode, found[-1] if found else ""


def confirm(stage: ev.Stage, past: int, deadline: float, poll_s: float = POLL_S) -> str:
    """``ok`` once the relaunched run logs an iteration past ``past``, else the failure or ``timeout``."""
    while time.monotonic() < deadline:
        log = ANSI.sub("", read_stage(stage, "log") or "")
        if any(int(k) > past for k, _ in ITERATION.findall(log)):
            return "ok"
        if bad := next((f for f in FAILURES if f in log), None):
            return bad
        time.sleep(poll_s)
    return "timeout"


def cmd_move(
    lease_id: str, holder: str, to: str, wait_s: float, verify_s: float, pools: list[Pool], watch: bool = True
) -> int:
    """Move the detached train.py run on ``lease_id`` to ``to`` from its next checkpoint, then watch it there.

    Args:
        lease_id: The lease the run holds (its id, or ``lease:<id>``).
        holder: The caller, who must hold it.
        to: ``same`` (the run's own card or cards), ``lease:<id>`` (a card the holder holds) or a card to claim; for
            a ``--gpus`` run, ``same`` or a held node (``host:job``) whose cards are claimed.
        wait_s: How long to wait for the next checkpoint before giving up (the run keeps going).
        verify_s: How long the relaunch has to train past that checkpoint.
        pools: Configured pools.
        watch: Watch the moved run to its end, as a `pls run --detach` launch does.

    Returns:
        The watch's verdict (0 once the moved run finished), or with no watch 0 once the relaunch trains past the
        checkpoint; 1 when it does not (each step says what was left).
    """
    lease_id = lease_id.removeprefix("lease:")
    pool, host, gpu, job, uuid_ = ev._lease_card(lease_id, holder, pools)
    with ls.locked_store() as leases:
        lease = next(x for x in leases if x.id == lease_id)
        recorded = {x.id: list(x.stages) or ([x.stage] if x.stage else []) for x in leases}
        stages = recorded[lease_id]
        group = [i for i, s in recorded.items() if s == stages]  # a --gpus run's cards all record its one stage
        others = [i for i, s in recorded.items() if i not in group and set(s) & set(stages)]
    if len(stages) != 1 or others or ls.is_cpu(lease):
        sys.exit("[move] --move takes a lease with one detached run (not --cpus, or several runs on its cards)")
    stage = ev.Stage(ev.make_target(pool, host, gpu, job, uuid_))
    stage.dir = stages[0]
    raw = read_stage(stage, "pls_argv.json")
    if raw is None:
        sys.exit("[move] the run records no launch (pls_argv.json): it predates --move; move it by hand this once")
    opts, script, args = split_launch(json.loads(raw))
    if not script.replace(":", "/").endswith(TRAIN):
        sys.exit(f"[move] --move moves hcrl_isaaclab's train.py runs; this run is {script}")
    gpus = int(launch_value(opts, "--gpus") or 1)
    if gpus != len(group):
        sys.exit(f"[move] the run was launched with --gpus {gpus}, but {len(group)} lease(s) record it")
    node = launch_value(opts, "--on") if gpus > 1 else ""
    state = progress(read_stage(stage, "log") or "")
    if state is None:
        sys.exit("[move] the run has not logged a W&B run and a learning iteration yet; let it start first")
    url, _, total = state
    wt, made = pin_code(stage, f"move-{lease_id}-{int(time.time())}")
    try:
        new = _move(lease_id, holder, to, wait_s, verify_s, pools, opts, script, args, url, total, wt, node, gpus)
    finally:
        unpin(made)
    if new is None:
        return 1
    moved = progress(read_stage(new, "log") or "")
    if not watch or moved is None:
        return 0
    run = follow.WANDB_RUN.search(moved[0])
    print("[move] watching the moved run to its end (stopping this watch leaves the run running)", file=sys.stderr)
    return follow.watch(new, "/".join(run.groups()) if run else "", moved[2], follow._run_name(args, new.dir))


def _move(
    lease_id: str, holder: str, to: str, wait_s: float, verify_s: float, pools: list[Pool],
    opts: list[str], script: str, args: list[str], url: str, total: int, wt: str, node: str = "", gpus: int = 1,
) -> ev.Stage | None:  # fmt: skip
    """The move once the run is known movable and its code pinned: claim, wait, stop, relaunch, confirm, recorder.

    Returns:
        The relaunched run's stage once it trains past the checkpoint, else None.
    """
    have = wandb_checkpoints(url)
    # a --gpus run relaunches on a node's cards (claimed, or its own kept); a one-card run on one lease
    on = target_node(to, node, gpus, lease_id, holder, pools) if gpus > 1 else ""
    target = "" if on else target_lease(to, lease_id, holder, pools)
    keep = to == "same" if on else target == lease_id
    where = f"node {on}" if on else f"lease {target}"
    print(f"[move] waiting up to {wait_s / 60:g} min for a checkpoint past {max(have, default=-1)} on {url}",
          file=sys.stderr)  # fmt: skip
    ckpt = wait_checkpoint(url, max(have, default=-1), time.monotonic() + wait_s)
    if ckpt is None:
        sys.exit(f"[move] no new checkpoint in {wait_s / 60:g} min; the run keeps going and {where} stays held")
    if ckpt >= total:
        sys.exit(f"[move] the run reached its last checkpoint ({ckpt}/{total}); nothing left to move")
    # an off-policy resume refills its replay buffer for `warmup` iterations on top of --max_iterations, so the run
    # keeps every training update it had left and ends `warmup` iterations later
    warmup = resume_warmup(url)
    left = total - ckpt
    print(f"[move] checkpoint model_{ckpt}.pt is on W&B; stopping the run", file=sys.stderr)
    if ev.cmd_stop(lease_id, holder, pools, keep_lease=keep) != 0:
        sys.exit(f"[move] the run did not stop cleanly; nothing relaunched ({where} stays held)")
    resume = ["--load_run", url, "--checkpoint", f"model_{ckpt}.pt", "--max_iterations", str(left)]
    rc, new_dir = relaunch(opts, script, [*drop_resume(args), *resume], target, wt, on=on, gpus=gpus)
    if rc != 0 or not new_dir:
        sys.exit(f"[move] the relaunch failed (status {rc}); resume by hand from model_{ckpt}.pt of {url}")
    t_pool, t_host, t_gpu, t_job, t_uuid = ev._lease_card(target or recording_lease(new_dir), holder, pools)
    new = ev.Stage(ev.make_target(t_pool, t_host, t_gpu, t_job, t_uuid))
    new.dir = new_dir
    verdict = confirm(new, ckpt, time.monotonic() + verify_s)
    if verdict != "ok":
        print(f"[move] the relaunch has not trained past {ckpt} ({verdict}); check {new_dir}/log", file=sys.stderr)
        return None
    print(f"[move] moved: training past iteration {ckpt} on {where} (stage {new_dir})", file=sys.stderr)
    if warmup:
        end = total + warmup
        print(
            f"[move] it refills its replay buffer first: updates resume at {ckpt + warmup}, and it ends at {end}",
            file=sys.stderr,
        )
    print(f"[move] the run continues as a new W&B run; its history up to {ckpt} stays in {url}", file=sys.stderr)
    arm_moved_recorder(new, holder, wt, opts, args, pools)
    return new


def arm_moved_recorder(stage: ev.Stage, holder: str, wt: str, opts: list[str], args: list[str], pools: list) -> None:
    """The new W&B run of a moved async run gets its own recorder, as its launch's did (none with --no-recorder)."""
    if follow.eval_mode(args) != "async" or "--no-recorder" in opts:
        return
    moved = progress(read_stage(stage, "log") or "")
    run = follow.WANDB_RUN.search(moved[0]) if moved else None
    on = follow.arm_recorder("/".join(run.groups()), holder, wt, pools) if run else ""
    if on:
        print(f"[move] armed an async recorder for the new W&B run on {on}", file=sys.stderr)
    else:
        print("[move] NO RECORDER for the new W&B run: arm one by hand", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    """`pls run --move`'s options."""
    p = argparse.ArgumentParser(prog="pls run --move", description=__doc__.split("\n", 1)[0])
    p.add_argument("--move", required=True, metavar="LEASE", help="the lease the detached train.py run holds")
    p.add_argument("--holder", required=True, help="your session name (must match the lease)")
    p.add_argument(
        "--to",
        required=True,
        help="same (its own card or cards), lease:<id> (a card you hold), a card (host:gpu, host:job:gpu), or for a"
        " --gpus run another held node (host:job)",
    )
    p.add_argument("--wait", type=ev._duration, default=1800.0, help="how long to wait for the next checkpoint")
    p.add_argument("--verify", type=ev._duration, default=1800.0, help="how long the relaunch has to train past it")
    p.add_argument("--no-watch", action="store_true", help="return once the moved run trains, with no watch")
    return p


def main(argv: list[str], pools: list[Pool]) -> int:
    """Parse `pls run --move` and move the run."""
    a: Any = parser().parse_args(argv)
    return cmd_move(a.move, a.holder, a.to, a.wait, a.verify, pools, watch=not a.no_watch)
