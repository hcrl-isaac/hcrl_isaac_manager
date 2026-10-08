#!/usr/bin/env python3
"""`just res`: one view of every GPU on every compute pool, probed live (never declared), plus card leases."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import evaluate
import leases as ls
from inventory import Pool, load_config, load_pools
from probe import HELD_UTIL, Card, Report, probe_local, probe_ray, probe_slurm_login, probe_ssh_host

CAUTION = {"free": 0, "held": 1, "busy": 2, "unknown": 3}
WAIT_POLL_S = 60.0  # how often --wait probes again for a card to free


class Busy(SystemExit):
    """A claim refused only because the cards it wants are taken right now, which --wait retries."""


def _guarded(fn: Callable[..., list[Report]], label: str, kind: str, *args: Any) -> list[Report]:
    """Run one probe; an unexpected exception becomes that probe's UNKNOWN report instead of ending the run."""
    try:
        return fn(*args)
    except Exception as exc:  # a probe bug or missing tool must not hide the other pools
        return [Report(label, kind, error=f"probe failed: {type(exc).__name__}: {exc}")]


def _error(message: str) -> list[Report]:
    raise RuntimeError(message)


def _probe_one_host(pool: Pool, host: str) -> list[Report]:
    return [probe_ssh_host(pool, host)]


def probe_all(pools: list[Pool]) -> list[Report]:
    """Probe every pool in parallel; SLURM profiles that share a login are probed once.

    Args:
        pools: Pools to probe.

    Returns:
        Reports in pool order, with cards seen by two pools (same uuid) kept only in the first.
    """
    tasks = []
    by_login: dict[str, list[Pool]] = {}
    for pool in pools:
        if pool.kind == "local":
            tasks.append((probe_local, pool.name, pool.kind, pool))
        elif pool.kind == "ssh":
            hosts = pool.settings.get("hosts", [])
            tasks.extend((_probe_one_host, f"{pool.name}/{h}", pool.kind, pool, h) for h in hosts)
        elif pool.kind == "ray":
            tasks.append((probe_ray, pool.name, pool.kind, pool))
        elif pool.kind == "slurm":
            if pool.settings.get("login"):
                by_login.setdefault(pool.settings["login"], []).append(pool)
            else:
                tasks.append((_error, pool.name, pool.kind, f"pool {pool.name} has no login"))
        else:
            tasks.append((lambda: [], pool.name, pool.kind))
            print(f"[res] unknown kind '{pool.kind}' for pool {pool.name}", file=sys.stderr)
    tasks.extend((probe_slurm_login, login, "slurm", login, group) for login, group in by_login.items())
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda t: _guarded(*t), tasks))
    return merge_duplicates([r for group in results for r in group])


def merge_duplicates(reports: list[Report]) -> list[Report]:
    """List a card seen by two pools (same uuid) once, with the more cautious state and both process lists."""
    first: dict[str, Card] = {}
    for rep in reports:
        kept = []
        for card in rep.cards:
            other = first.get(card.uuid) if card.uuid else None
            if other is None:
                if card.uuid:
                    first[card.uuid] = card
                kept.append(card)
                continue
            if CAUTION[card.state] > CAUTION[other.state]:
                other.state = card.state
            other.note = "; ".join(n for n in dict.fromkeys([other.note, card.note]) if n)
            for proc in card.procs:
                if proc.pid not in {p.pid for p in other.procs}:
                    other.procs.append(proc)
        rep.cards = kept
    return reports


def _ago(ts: float) -> str:
    mins = int((time.time() - ts) // 60)
    return f"{mins // 60}h{mins % 60:02d}m" if mins >= 60 else f"{mins}m"


def _lease_text(lease: ls.Lease, card: Card) -> str:
    tag = "CONFLICT " if lease.conflict else ""
    if card.state == "held" and not card.procs and card.util < HELD_UTIL:
        tag += "held, no process; "
    return f"[{tag}{lease.holder}: {lease.note or lease.id}, {_ago(lease.created)}]"


def render(reports: list[Report], held: dict[str, ls.Lease]) -> tuple[str, Counter]:
    """Format reports as a per-pool table, marking leased cards.

    Args:
        reports: Probe results.
        held: Current leases by card key.

    Returns:
        The text and the per-state card counts (plus `unknown` pools).
    """
    lines = []
    head = (
        f"{'HOST/JOB':<22} {'GPU':>3} {'MODEL':<16} {'MEM (MiB)':>15} {'UTIL':>5}  {'STATE':<7} {'WHO / WHAT':<44} LEFT"
    )
    totals = Counter()
    for rep in reports:
        lines.append(f"\n== {rep.pool} [{rep.kind}]")
        if rep.error:
            lines.append(f"   UNKNOWN: {rep.error}")
            totals["unknown"] += 1
            continue
        if rep.cards:
            lines.append("   " + head)
        for c in rep.cards:
            lease = held.get(ls.card_key(c))
            state = "leased" if lease and c.state == "free" else c.state
            totals[state] += 1
            who = "; ".join(f"{p.user} {p.cmd or p.pid} {p.elapsed}".strip() for p in c.procs) or c.note
            if c.state == "held" and not who:
                who = "in use, no visible process"
            if lease:
                who = f"{_lease_text(lease, c)} {who}".strip()
            where = f"{c.host} {c.job}".strip()
            model = c.model.replace("NVIDIA ", "").replace("GeForce ", "")[:16]
            mem = f"{c.mem_used}/{c.mem_total}" if c.mem_used >= 0 else f"?/{c.mem_total}"
            row = f"{where:<22} {c.index:>3} {model:<16} {mem:>15} {c.util:>4}%  {state:<7} {who:<44.44} {c.wall_left}"
            lines.append("   " + row)
        lines.extend(f"   - {note}" for note in rep.notes)
    parts = [f"{totals[k]} {k}" for k in ("free", "leased", "busy", "held") if totals[k]]
    if totals["unknown"]:
        parts.append(f"{totals['unknown']} card(s)/pool(s) unknown")
    lines.append(f"\ncards: {', '.join(parts) or 'none found'}")
    return "\n".join(lines), totals


def select_pools(pools: list[Pool], prefixes: list[str] | None) -> list[Pool]:
    """Pools whose name starts with one of `prefixes` (all pools when none are given)."""
    chosen = [p for p in pools if not prefixes or any(p.name.startswith(sel) for sel in prefixes)]
    if not chosen:
        sys.exit(f"[res] no pool matches {', '.join(prefixes or [])} (see: just res pools)")
    return chosen


def lease_windows() -> dict[str, float]:
    """grace_min / idle_min from compute.toml's [leases] table; each must be a finite number of minutes > 0."""
    cfg = load_config().get("leases", {})
    windows = {}
    for key, default in (("grace_min", ls.GRACE_MIN), ("idle_min", ls.IDLE_MIN)):
        val = cfg.get(key, default)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
            sys.exit(f"[res] compute.toml [leases] {key} must be a positive number of minutes, not {val!r}")
        windows[key] = float(val)
    return windows


def _report_released(released: list[tuple[ls.Lease, str]]) -> None:
    for lease, why in released:
        print(f"[res] released {lease.id} ({lease.card}, {lease.holder}): {why}", file=sys.stderr)


def cmd_status(args: argparse.Namespace, pools: list[Pool]) -> None:
    """Probe, reconcile leases and print every card; a lease store problem only hides the lease column."""
    reports = probe_all(select_pools(pools, args.pool))
    windows = lease_windows()
    try:
        with ls.locked_store() as leases:
            _report_released(ls.reconcile(leases, reports, **windows))
            held = {lease.key: lease for lease in leases}
    except Exception as exc:  # the card table must still print when the lease store is broken
        print(f"[res] WARNING: leases not shown: {type(exc).__name__}: {exc}", file=sys.stderr)
        held = {}
    if args.free:
        for rep in reports:
            rep.cards = [
                c for c in rep.cards if c.state == "unknown" or (c.state == "free" and ls.card_key(c) not in held)
            ]
    if args.json:
        out = {"reports": [r.to_dict() for r in reports], "leases": [vars(x) for x in held.values()]}
        print(json.dumps(out, indent=1))
    else:
        print(render(reports, held)[0])
    if reports and all(r.error or (r.cards and all(c.state == "unknown" for c in r.cards)) for r in reports):
        sys.exit(2)


def _named(spec: str, seen: list[tuple[Card, Report]]) -> tuple[Card, Report]:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        sys.exit(f"[res] {spec}: expected host:gpu or host:job:gpu; nothing claimed")
    host, job, gpu = (parts[0], None, parts[1]) if len(parts) == 2 else parts
    hits = [(c, r) for c, r in seen if c.host == host and str(c.index) == gpu and (job is None or c.job == job)]
    if not hits:
        sys.exit(f"[res] {spec}: not found or its pool is unreachable; nothing claimed")
    if len(hits) > 1:
        jobs = ", ".join(f"{host}:{c.job}:{gpu}" for c, _ in hits)
        sys.exit(f"[res] {spec} is ambiguous ({jobs}); nothing claimed")
    return hits[0]


def _choose(seen: list[tuple[Card, Report]], taken: set[str], args: argparse.Namespace) -> list[tuple[Card, Report]]:
    """The cards a claim takes: all named ones, or --any free ones packed onto partly used hosts first."""
    adopt = getattr(args, "adopt", False)
    if args.any and adopt:
        sys.exit("[res] --adopt takes over named cards; name them (host:gpu ...)")
    if args.any:
        explicit_ray = {
            r.pool for _, r in seen if r.kind == "ray" and any(r.pool.startswith(s) for s in args.pool or [])
        }
        free = [
            (c, r)
            for c, r in seen
            if c.state == "free"
            and ls.card_key(c) not in taken
            and c.mem_total - c.mem_used >= args.min_free_gb * 1024
            and (r.kind != "ray" or r.pool in explicit_ray)  # Ray schedules onto its own free cards
        ]
        in_use = {c.host for c, _ in seen if c.state != "free" or ls.card_key(c) in taken}
        per_host = Counter(c.host for c, _ in free)

        def pack(cr: tuple[Card, Report]) -> tuple:
            c = cr[0]
            return (c.host not in in_use, per_host[c.host], -(c.mem_total - c.mem_used), c.host, c.index)

        free.sort(key=pack)
        if len(free) < args.count:
            raise Busy(f"[res] only {len(free)} free card(s) match; nothing claimed")
        return free[: args.count]
    chosen = []
    for spec in dict.fromkeys(args.cards):
        card, rep = _named(spec, seen)
        key = ls.card_key(card)
        if key in taken:
            raise Busy(
                f"[res] {spec} is leased; move it with `just res transfer {spec} --to <holder>`; nothing claimed"
            )
        if card.state != "free" and not (adopt and card.state in ("busy", "held")):
            hint = "; pass --adopt to take over the run on it" if card.state in ("busy", "held") else ""
            raise Busy(f"[res] {spec} is {card.state}{hint}; nothing claimed")
        if key in {ls.card_key(c) for c, _ in chosen}:
            continue
        chosen.append((card, rep))
    return chosen


def claim(args: argparse.Namespace, pools: list[Pool]) -> tuple[list[tuple[ls.Lease, Card, Report]], dict]:
    """Lease named cards, or --any free ones, after a fresh probe; all or nothing (exits when refused).

    With ``args.wait`` set (seconds, 0 for no limit) a refusal only because the cards are taken right now is retried
    every ``WAIT_POLL_S`` until they free or the wait runs out.

    Args:
        args: cards, any, count, min_free_gb, pool, holder, note and for_ as `just res claim` takes them, plus
            optional adopt, run and wait.
        pools: Pools to probe.

    Returns:
        The new leases with their cards and reports, and the lease windows.
    """
    wait = getattr(args, "wait", None)
    deadline = time.monotonic() + wait if wait else None
    announced = False
    while True:
        try:
            return _claim_once(args, pools)
        except Busy as exc:
            if wait is None:
                raise
            if deadline is not None and time.monotonic() + WAIT_POLL_S > deadline:
                raise SystemExit(f"{exc.code}; gave up waiting after {wait / 60:g} min") from None
            if not announced:
                print(f"{exc.code}; waiting for one to free (--wait)", file=sys.stderr, flush=True)
                announced = True
            time.sleep(WAIT_POLL_S)


def _claim_once(args: argparse.Namespace, pools: list[Pool]) -> tuple[list[tuple[ls.Lease, Card, Report]], dict]:
    """One probe-and-lease attempt of :func:`claim`."""
    reports = probe_all(select_pools(pools, args.pool))
    seen = [(c, r) for r in reports for c in r.cards]
    windows = lease_windows()
    refused = None
    try:
        with ls.locked_store() as leases:
            _report_released(ls.reconcile(leases, reports, **windows))
            try:
                chosen = _choose(seen, {lease.key for lease in leases}, args)
            except SystemExit as exc:  # keep the reconcile result even when the claim is refused
                refused = exc
            else:
                run = getattr(args, "run", "")
                new = [ls.new_lease(c, r, args.holder, args.note, args.for_, run=run) for c, r in chosen]
                leases.extend(new)
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] nothing claimed: {exc}")
    if refused is not None:
        raise refused
    return [(lease, c, r) for lease, (c, r) in zip(new, chosen, strict=True)], windows


def cmd_claim(args: argparse.Namespace, pools: list[Pool]) -> None:
    """Lease named cards, or --any free ones, after a fresh probe; all or nothing."""
    if not args.cards and not args.any:
        sys.exit("[res] name cards (host:gpu ...) or pass --any")
    taken, windows = claim(args, pools)
    for lease, card, rep in taken:
        box = f", time box {int(args.for_ // 60)} min" if args.for_ else ""
        verb = "adopted" if card.state != "free" else "claimed"
        run = f", run {lease.run}" if lease.run else ""
        print(f"{verb} {lease.id}: {lease.card} ({rep.pool}{box}{run}) for {lease.holder}")
    print(
        f"A lease ends after {windows['grace_min']:g} min unused or {windows['idle_min']:g} min idle, measured from "
        "the first idle observation. Release early with: just res release <id> --holder <you>"
    )


def _slurm_label(target: str) -> str:
    """`host:job:gpu` as the lease label `host:gpu (job job)`; other targets unchanged."""
    parts = target.split(":")
    return f"{parts[0]}:{parts[2]} (job {parts[1]})" if len(parts) == 3 else target


def cmd_release(args: argparse.Namespace, _pools: list[Pool]) -> None:
    """Drop leases by id or card; only the holder may, unless --force."""
    try:
        with ls.locked_store() as leases:
            targets = set(args.targets) | {_slurm_label(t) for t in args.targets}
            gone = [x for x in leases if x.id in targets or x.card in targets or x.key in targets]
            foreign = [x for x in gone if x.holder != args.holder]
            if foreign and not args.force:
                names = ", ".join(f"{x.id} ({x.holder})" for x in foreign)
                sys.exit(f"[res] held by someone else: {names}; pass --force to release anyway")
            for lease in gone:
                leases.remove(lease)
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] nothing released: {exc}")
    for lease in gone:
        print(f"released {lease.id}: {lease.card} ({lease.holder})")
    found = {x.id for x in gone} | {x.card for x in gone} | {x.key for x in gone}
    missing = {t for t in args.targets if t not in found and _slurm_label(t) not in found}
    if missing:
        sys.exit(f"[res] no lease for: {', '.join(sorted(missing))}")


def cmd_transfer(args: argparse.Namespace, _pools: list[Pool]) -> None:
    """Hand leases (by id or card) to another holder; only the holder may, unless --force."""
    try:
        with ls.locked_store() as leases:
            targets = set(args.targets) | {_slurm_label(t) for t in args.targets}
            moving = [x for x in leases if x.id in targets or x.card in targets or x.key in targets]
            found = {x.id for x in moving} | {x.card for x in moving} | {x.key for x in moving}
            missing = {t for t in args.targets if t not in found and _slurm_label(t) not in found}
            if missing:
                sys.exit(f"[res] no lease for: {', '.join(sorted(missing))}; nothing transferred")
            foreign = [x for x in moving if x.holder != args.holder]
            if foreign and not args.force:
                names = ", ".join(f"{x.id} ({x.holder})" for x in foreign)
                sys.exit(f"[res] held by someone else: {names}; pass --force to transfer anyway")
            for lease in moving:
                lease.previous, lease.holder = lease.holder, args.to
                lease.note = args.note or lease.note
                lease.run = args.run or lease.run
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] nothing transferred: {exc}")
    for lease in moving:
        print(f"transferred {lease.id}: {lease.card} {lease.previous} -> {lease.holder}")


def cmd_leases(_args: argparse.Namespace, _pools: list[Pool]) -> None:
    """List the stored leases without probing (activity as of the last res call)."""
    try:
        with ls.locked_store() as leases:
            rows = list(leases)
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] {exc}")
    if not rows:
        print("no leases")
    for x in rows:
        act = f"active {_ago(x.last_active)} ago" if x.last_active else f"claimed {_ago(x.created)} ago, not used yet"
        if x.idle_since:
            act += f", idle since {_ago(x.idle_since)} ago"
        box = f", time box ends in {int((x.expires - time.time()) // 60)}m" if x.expires else ""
        flag = " CONFLICT" if x.conflict else ""
        extra = "".join(f" [{k} {v}]" for k, v in (("run", x.run), ("from", x.previous)) if v)
        print(f"{x.id}  {x.card:<22} {x.report:<28} {x.holder:<24} {act}{box}{flag}  {x.note}{extra}")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _duration(text: str) -> float:
    try:
        return ls.parse_duration(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    """Parse the CLI and run a command."""
    parser = argparse.ArgumentParser(prog="just res", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    st = sub.add_parser("status", help="probe every pool and show each card's state and lease")
    st.add_argument("--pool", action="append", help="only these pools (repeatable; prefix match)")
    st.add_argument("--free", action="store_true", help="only list free, unleased cards (and unknown ones)")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    cl = sub.add_parser("claim", help="lease cards (host:gpu ...) or --any free ones")
    cl.add_argument("cards", nargs="*", help="cards as host:gpu, or host:job:gpu on a SLURM node")
    cl.add_argument("--any", action="store_true", help="pick free cards instead of naming them")
    cl.add_argument("--count", type=_positive_int, default=1, help="with --any: how many cards")
    cl.add_argument("--min-free-gb", type=float, default=0, help="with --any: free memory each card needs")
    cl.add_argument("--pool", action="append", help="only probe/pick from these pools (prefix match)")
    cl.add_argument("--holder", required=True, help="who holds it (your session name)")
    cl.add_argument("--note", default="", help="what it is for")
    cl.add_argument("--for", dest="for_", type=_duration, default=0.0, help="hard time box, e.g. 90m or 2h")
    cl.add_argument("--adopt", action="store_true", help="take over named busy cards whose run you are taking on")
    cl.add_argument("--run", default="", help="the run the lease covers (W&B id, job.step)")
    cl.add_argument(
        "--wait",
        nargs="?",
        const=0.0,
        type=_duration,
        help="when the cards are taken, wait for them to free (optionally at most this long, e.g. 2h)",
    )
    rl = sub.add_parser("release", help="release your leases by id or host:gpu")
    rl.add_argument("targets", nargs="+")
    rl.add_argument("--holder", required=True, help="your session name (must match the lease)")
    rl.add_argument("--force", action="store_true", help="release someone else's lease")
    tr = sub.add_parser("transfer", help="hand your leases (id or host:gpu) to another holder")
    tr.add_argument("targets", nargs="+")
    tr.add_argument("--holder", required=True, help="the current holder (must match the lease)")
    tr.add_argument("--to", required=True, help="the new holder")
    tr.add_argument("--note", default="", help="new note (default: keep)")
    tr.add_argument("--run", default="", help="the run the lease covers (default: keep)")
    tr.add_argument("--force", action="store_true", help="transfer someone else's lease")
    sub.add_parser("leases", help="list leases (no probe)")
    sub.add_parser("pools", help="list the configured pools")
    evaluate.add_parser(sub)
    argv, script_args = evaluate.split_script_args(sys.argv[1:] or ["status"])
    args = parser.parse_args(argv)
    args.script_args = script_args

    try:
        pools = load_pools()
    except Exception as exc:  # a broken inventory must say so, not print a traceback mid-table
        sys.exit(f"[res] cannot read the compute inventory: {type(exc).__name__}: {exc}")
    if args.cmd == "pools":
        for p in pools:
            detail = p.settings.get("hosts") or p.settings.get("login") or p.settings.get("address") or ""
            print(f"{p.name:<16} {p.kind:<6} {detail}")
        return
    if args.cmd == "eval":
        evaluate.cmd_eval(args, pools, claim)
        return
    commands = {
        "status": cmd_status,
        "claim": cmd_claim,
        "release": cmd_release,
        "transfer": cmd_transfer,
        "leases": cmd_leases,
    }
    commands[args.cmd](args, pools)


if __name__ == "__main__":
    main()
