#!/usr/bin/env python3
"""`pls res`: one view of every GPU on every compute pool, probed live (never declared), plus card leases."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import leases as ls
from inventory import Pool, load_config, load_pools, ssh_target
from probe import (
    HELD_UTIL,
    Card,
    Report,
    box_cpu_state,
    probe_local,
    probe_ray,
    probe_slurm_login,
    probe_ssh_host,
)

CAUTION = {"free": 0, "held": 1, "busy": 2, "unknown": 3}
WAIT_POLL_S = 60.0  # how often --wait probes again for a card to free


class Busy(SystemExit):
    """A claim refused only because the cards it wants are taken right now, which --wait retries."""

    reports: list[Report] = []  # the probe that refused it


_report_pools = ls.report_pools


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


def _reservation_lines(reservations: list[ls.Reservation], leases: list[ls.Lease]) -> list[str]:
    lines = []
    for r in reservations:
        until = time.strftime("%m-%d %H:%M", time.localtime(r.expires))
        yielding = [x for x in leases if x.yield_to == r.id]
        who = ", ".join(f"{x.holder} {x.card}" for x in yielding) or "none"
        lines.append(f"{r.id}  {r.holder}: {r.scope()} needed {ls.eta_text(r)}, until {until}  {r.note}")
        lines.append(f"        yielding: {who}")
    return lines


def render(
    reports: list[Report], held: dict[str, ls.Lease], reservations: list[ls.Reservation] | None = None
) -> tuple[str, Counter]:
    """Format reports as a per-pool table, marking leased and reserved cards.

    Args:
        reports: Probe results.
        held: Current leases by card key.
        reservations: Current reservations (listed after the table, and marked on the cards they cover).

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
            r = ls.covering(reservations or [], ls.card_key(c), rep.pool)
            if r and not (lease and lease.holder == r.holder):
                who = f"[RESERVED {r.holder} {ls.eta_text(r)}] {who}".strip()
            where = f"{c.host} {c.job}".strip()
            model = c.model.replace("NVIDIA ", "").replace("GeForce ", "")[:16]
            mem = f"{c.mem_used}/{c.mem_total}" if c.mem_used >= 0 else f"?/{c.mem_total}"
            row = f"{where:<22} {c.index:>3} {model:<16} {mem:>15} {c.util:>4}%  {state:<7} {who:<44.44} {c.wall_left}"
            lines.append("   " + row)
        lines.extend(
            f"   CPU run [{x.holder}: {x.note or x.id}, {_ago(x.created)}] {x.cpus} cores ({x.cores}), no GPU"
            for x in held.values()
            if ls.is_cpu(x) and x.report == rep.pool
        )
        lines.extend(f"   - {note}" for note in rep.notes)
    parts = [f"{totals[k]} {k}" for k in ("free", "leased", "busy", "held") if totals[k]]
    if totals["unknown"]:
        parts.append(f"{totals['unknown']} card(s)/pool(s) unknown")
    lines.append(f"\ncards: {', '.join(parts) or 'none found'}")
    if reservations:
        lines += ["\nreservations (priority; leases on these cards yield when the holder claims them):"]
        lines += ["  " + line for line in _reservation_lines(reservations, list(held.values()))]
    return "\n".join(lines), totals


def select_pools(pools: list[Pool], prefixes: list[str] | None) -> list[Pool]:
    """Pools whose name starts with one of `prefixes` (all pools when none are given)."""
    chosen = [p for p in pools if not prefixes or any(p.name.startswith(sel) for sel in prefixes)]
    if not chosen:
        sys.exit(f"[res] no pool matches {', '.join(prefixes or [])} (see: pls res pools)")
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


def _settle(leases: list[ls.Lease], reservations: list[ls.Reservation]) -> None:
    """End finished reservations (and say so), then mark the leases on the remaining ones' cards as yielding."""
    for r, why in ls.settle(reservations, leases):
        print(f"[res] reservation {r.id} ({r.holder}, {r.scope()}) ended: {why}", file=sys.stderr)
    for r in reservations:
        ls.mark_yields(r, leases)


def cpu_run_states(leases: list[ls.Lease], pools: list[Pool]) -> dict[str, set[str]]:
    """For each box holding CPU leases, the stage dirs whose run is gone there (one ssh per box, in parallel).

    Args:
        leases: The leases (only CPU leases of the given pools' ssh boxes are looked at).
        pools: Pools whose boxes to check.

    Returns:
        Report label (``<pool>/<host>``) -> ended stage dirs; a box that could not be read is left out.
    """
    by_label: dict[str, list[str]] = {}
    for lease in leases:
        if ls.is_cpu(lease):
            by_label.setdefault(lease.report, []).append(lease.stage)
    ssh_pools = {p.name: p for p in pools if p.kind == "ssh"}
    tasks = []
    for label, stages in by_label.items():
        name, _, host = label.partition("/")
        pool = ssh_pools.get(name)
        if pool is not None and host in pool.settings.get("hosts", []) and pool.settings.get("user"):
            tasks.append((label, ssh_target(pool, host), [s for s in stages if s]))
    if not tasks:
        return {}
    start_min = ls.CPU_START_S // 60
    with ThreadPoolExecutor(max_workers=16) as ex:
        states = list(ex.map(lambda t: box_cpu_state(t[1], t[2], start_min), tasks))
    return {label: state[1] for (label, _, _), state in zip(tasks, states, strict=True) if state is not None}


def cmd_status(args: argparse.Namespace, pools: list[Pool]) -> None:
    """Probe, reconcile leases and print every card; a lease store problem only hides the lease column."""
    chosen = select_pools(pools, args.pool)
    reports = probe_all(chosen)
    windows = lease_windows()
    try:
        with ls.locked_store() as leases:
            cpu_leases = [x for x in leases if ls.is_cpu(x)]
        ended = cpu_run_states(cpu_leases, chosen)  # outside the lock: it reaches the boxes
        with ls.locked_state() as (leases, reservations):
            _report_released(ls.reconcile(leases, reports, **windows))
            for label, stages in ended.items():
                _report_released(ls.reap_cpu(leases, label, stages))
            _settle(leases, reservations)
            held = {lease.key: lease for lease in leases}
            reserved = list(reservations)
    except Exception as exc:  # the card table must still print when the lease store is broken
        print(f"[res] WARNING: leases not shown: {type(exc).__name__}: {exc}", file=sys.stderr)
        held, reserved = {}, []
    if args.free:
        for rep in reports:
            rep.cards = [
                c for c in rep.cards if c.state == "unknown" or (c.state == "free" and ls.card_key(c) not in held)
            ]
    if args.json:
        out = {
            "reports": [r.to_dict() for r in reports],
            "leases": [vars(x) for x in held.values()],
            "reservations": [vars(x) for x in reserved],
        }
        print(json.dumps(out, indent=1))
    else:
        print(render(reports, held, reserved)[0])
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


def _choose(
    seen: list[tuple[Card, Report]],
    taken: dict[str, ls.Lease],
    args: argparse.Namespace,
    yielding: frozenset[str] = frozenset(),
    rank: Callable[[Card, Report], int] = lambda c, r: 0,
) -> list[tuple[Card, Report]]:
    """The cards a claim takes: all named ones, or --any free ones packed onto partly used hosts first.

    ``taken`` are the leases by card key. ``yielding`` are cards whose leases yield to the caller's reservation: the
    caller may take them although they are leased and in use. ``rank`` orders --any candidates before the packing
    (lower first).
    """
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
            if (ls.card_key(c) in yielding)
            or (
                c.state == "free"
                and ls.card_key(c) not in taken
                and c.mem_total - c.mem_used >= args.min_free_gb * 1024
                and (r.kind != "ray" or r.pool in explicit_ray)  # Ray schedules onto its own free cards
            )
        ]
        in_use = {c.host for c, _ in seen if c.state != "free" or ls.card_key(c) in taken}
        per_host = Counter(c.host for c, _ in free)

        def pack(cr: tuple[Card, Report]) -> tuple:
            c = cr[0]
            return (rank(*cr), c.host not in in_use, per_host[c.host], -(c.mem_total - c.mem_used), c.host, c.index)

        within = getattr(args, "within", "")
        if within:  # host or host:job: only that node's cards
            host, _, job = within.partition(":")
            free = [(c, r) for c, r in free if c.host == host and (not job or c.job == job.split(":")[0])]
        free.sort(key=pack)
        if getattr(args, "one_host", False):  # all on one node, for a multi-GPU run
            nodes: dict[tuple[str, str], list] = {}
            for c, r in free:
                nodes.setdefault((c.host, c.job), []).append((c, r))
            for cards in nodes.values():
                if len(cards) >= args.count:
                    return cards[: args.count]
            raise Busy(f"[res] no node has {args.count} free cards that match; nothing claimed")
        if len(free) < args.count:
            raise Busy(f"[res] only {len(free)} free card(s) match; nothing claimed")
        return free[: args.count]
    chosen = []
    for spec in dict.fromkeys(args.cards):
        card, rep = _named(spec, seen)
        key = ls.card_key(card)
        if key in yielding:
            if key not in {ls.card_key(c) for c, _ in chosen}:
                chosen.append((card, rep))
            continue
        if key in taken and taken[key].holder == args.holder:  # never frees by waiting: say how to use it
            sys.exit(
                f"[res] {spec} is already yours (lease {taken[key].id}); run on it with `--on lease:{taken[key].id}`"
            )
        if key in taken:
            raise Busy(f"[res] {spec} is leased; move it with `pls res transfer {spec} --to <holder>`; nothing claimed")
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
        args: cards, any, count, min_free_gb, pool, holder, note and for_ as `pls res claim` takes them, plus
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
            if args.cards and not announced:
                # named cards: later probes need only the pools they were found in, not every login and box
                hosts = {spec.split(":")[0] for spec in args.cards}
                found = set().union(
                    *(_report_pools(r.pool) for r in exc.reports if any(c.host in hosts for c in r.cards))
                )
                pools = [p for p in pools if p.name in found] or pools
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
        with ls.locked_state() as (leases, reservations):
            _report_released(ls.reconcile(leases, reports, **windows))
            _settle(leases, reservations)
            mine = {r.id for r in reservations if r.holder == args.holder}
            by_key = {lease.key: lease for lease in leases}
            yielding = frozenset(k for k, x in by_key.items() if x.yield_to in mine)

            def rank(c: Card, r: Report) -> int:  # a priority holder takes its reserved cards first, others last
                res_ = ls.covering(reservations, ls.card_key(c), r.pool)
                if res_ is None:
                    return 1
                return 0 if res_.holder == args.holder else 2

            try:
                chosen = _choose(seen, {k: x for k, x in by_key.items() if k not in yielding}, args, yielding, rank)
            except SystemExit as exc:  # keep the reconcile result even when the claim is refused
                refused = exc
            else:
                run = getattr(args, "run", "")
                new, notices = [], []
                for c, r in chosen:
                    key = ls.card_key(c)
                    res_ = ls.covering(reservations, key, r.pool)
                    if key in yielding:  # preempt: the yielding lease moves to the priority holder
                        lease = by_key[key]
                        notices.append(
                            f"[res] TOOK {lease.card} from {lease.holder} (lease {lease.id}) under your reservation: "
                            f"tell {lease.holder} to stop its run there now, and wait for the card to drain "
                            "(pls res status) before you launch on it"
                        )
                        lease.previous, lease.holder, lease.yield_to = lease.holder, args.holder, ""
                        lease.note, lease.run = args.note or lease.note, run or lease.run
                        now = time.time()
                        lease.created, lease.idle_since = now, 0.0
                        lease.expires = now + args.for_ if args.for_ else 0.0
                    else:
                        lease = ls.new_lease(c, r, args.holder, args.note, args.for_, run=run)
                        leases.append(lease)
                        if res_ is not None and res_.holder != args.holder:
                            lease.yield_to = res_.id
                            notices.append(
                                f"[res] {lease.card} is reserved by {res_.holder}, needed {ls.eta_text(res_)} "
                                f"({res_.note or res_.id}): your lease yields to it; checkpoint and be ready to stop"
                            )
                    if res_ is not None and res_.holder == args.holder:
                        res_.used = True
                    new.append(lease)
                for line in notices:
                    print(line, file=sys.stderr)
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] nothing claimed: {exc}")
    if refused is not None:
        if isinstance(refused, Busy):
            refused.reports = reports
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
        verb = f"took over (from {lease.previous})" if lease.previous and card.state != "free" else verb
        run = f", run {lease.run}" if lease.run else ""
        print(f"{verb} {lease.id}: {lease.card} ({rep.pool}{box}{run}) for {lease.holder}")
    print(
        f"A lease ends after {windows['grace_min']:g} min unused or {windows['idle_min']:g} min idle, measured from "
        "the first idle observation. Release early with: pls res release <id> --holder <you>"
    )


def _slurm_label(target: str) -> str:
    """`host:job:gpu` as the lease label `host:gpu (job job)`; other targets unchanged."""
    parts = target.split(":")
    return f"{parts[0]}:{parts[2]} (job {parts[1]})" if len(parts) == 3 else target


def cmd_release(args: argparse.Namespace, _pools: list[Pool]) -> None:
    """Drop leases by id or card; only the holder may, unless --force."""
    try:
        with ls.locked_state() as (leases, reservations):
            targets = set(args.targets) | {_slurm_label(t) for t in args.targets}
            gone = [x for x in leases if x.id in targets or x.card in targets or x.key in targets]
            foreign = [x for x in gone if x.holder != args.holder]
            if foreign and not args.force:
                names = ", ".join(f"{x.id} ({x.holder})" for x in foreign)
                sys.exit(f"[res] held by someone else: {names}; pass --force to release anyway")
            for lease in gone:
                leases.remove(lease)
            _settle(leases, reservations)  # a used reservation whose cards are all released is done
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


def cmd_reserve(args: argparse.Namespace, pools: list[Pool]) -> None:
    """Give the holder priority on named cards or N of a pool from an ETA; others' leases there yield to it."""
    if args.end is not None:
        try:
            with ls.locked_state() as (leases, reservations):
                gone = [r for r in reservations if r.holder == args.holder and args.end in ("", r.id)]
                for r in gone:
                    reservations.remove(r)
                    for lease in leases:
                        if lease.yield_to == r.id:
                            lease.yield_to = ""
        except ls.LeaseStoreError as exc:
            sys.exit(f"[res] nothing ended: {exc}")
        if not gone:
            sys.exit(f"[res] {args.holder} holds no reservation {args.end}".rstrip())
        for r in gone:
            print(f"ended reservation {r.id}: {r.scope()} ({r.holder})")
        return
    if bool(args.cards) == bool(args.pool):
        sys.exit("[res] reserve named cards (host:gpu ...) or --pool NAME --count N, not both")
    keys, labels = "", ""
    if args.cards:
        seen = [(c, r) for r in probe_all(pools) for c in r.cards]
        named = [_named(spec, seen) for spec in dict.fromkeys(args.cards)]
        keys = ",".join(ls.card_key(c) for c, _ in named)
        labels = ",".join(ls.card_label(c) for c, _ in named)
    elif not any(p.name.startswith(args.pool) for p in pools):
        sys.exit(f"[res] no pool matches {args.pool} (see: pls res pools)")
    try:
        lo, hi = ls.parse_eta(args.eta)
    except ValueError as exc:
        sys.exit(f"[res] --eta: {exc}")
    now = time.time()
    try:
        with ls.locked_state() as (leases, reservations):
            _settle(leases, reservations)
            same = [
                r for r in reservations if r.holder == args.holder and (r.pool, r.keys) == ((args.pool or ""), keys)
            ]
            if same:
                r = same[0]
                verb = "updated"
            else:
                r = ls.Reservation(id=uuid.uuid4().hex[:6], holder=args.holder, keys=keys, cards=labels,
                                   pool=args.pool or "", count=args.count, created=now)  # fmt: skip
                reservations.append(r)
                verb = "reserved"
            r.eta_lo, r.eta_hi, r.expires = lo, hi, now + args.for_
            r.note, r.count = args.note or r.note, args.count if args.pool else r.count
            yielding = ls.mark_yields(r, leases)
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] nothing reserved: {exc}")
    until = time.strftime("%m-%d %H:%M", time.localtime(r.expires))
    print(f"{verb} {r.id}: {r.scope()} for {r.holder}, needed {ls.eta_text(r)}, priority until {until}")
    if yielding:
        holders: dict[str, list[str]] = {}
        for x in yielding:
            holders.setdefault(x.holder, []).append(f"{x.card} (lease {x.id})")
        print("These leases now yield to it. Message each holder with the ETA so they checkpoint and stop in time:")
        for holder, cards in holders.items():
            print(f"  {holder}: {', '.join(cards)}")
    else:
        print("No one else holds these cards now; anyone who leases them meanwhile is told they yield.")
    print(f"Update the ETA with: pls res reserve ... --eta <when> --holder {r.holder}; "
          f"claim with pls res claim (yielding leases move to you); end early with --end {r.id}.")  # fmt: skip


def cmd_leases(_args: argparse.Namespace, _pools: list[Pool]) -> None:
    """List the stored leases and reservations without probing (activity as of the last res call)."""
    try:
        with ls.locked_state() as (leases, reservations):
            _settle(leases, reservations)
            rows, reserved = list(leases), list(reservations)
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] {exc}")
    if not rows:
        print("no leases")
    for x in rows:
        act = f"active {_ago(x.last_active)} ago" if x.last_active else f"claimed {_ago(x.created)} ago, not used yet"
        if ls.is_cpu(x):  # no card to watch: it ends with its run
            act = f"{x.cpus} cores ({x.cores}) since {_ago(x.created)} ago, held while its run lives"
        if x.idle_since:
            act += f", idle since {_ago(x.idle_since)} ago"
        box = f", time box ends in {int((x.expires - time.time()) // 60)}m" if x.expires else ""
        flag = " CONFLICT" if x.conflict else ""
        flag += f" YIELDS to {x.yield_to}" if x.yield_to else ""
        extra = "".join(f" [{k} {v}]" for k, v in (("run", x.run), ("from", x.previous)) if v)
        print(f"{x.id}  {x.card:<22} {x.report:<28} {x.holder:<24} {act}{box}{flag}  {x.note}{extra}")
    if reserved:
        print("\nreservations:")
        for line in _reservation_lines(reserved, rows):
            print(f"  {line}")


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
    parser = argparse.ArgumentParser(
        prog="pls res",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="held nodes (a cluster's long-lived job whose cards `pls run --on` leases), handled by `pls res`:\n"
        "  pls res hold --on <cluster>        submit one (it may queue for hours)\n"
        "  pls res drop --on <cluster> <job>  cancel it (the cluster connection stays up)\n"
        "  pls res status --on <cluster>      the connection and held node, then that cluster's cards",
    )
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
    rv = sub.add_parser("reserve", help="priority on cards from an ETA: others may use them and yield to you")
    rv.add_argument("cards", nargs="*", help="cards as host:gpu (or host:job:gpu); or use --pool and --count")
    rv.add_argument("--pool", help="reserve cards of this pool (prefix match) instead of naming them")
    rv.add_argument("--count", type=_positive_int, default=1, help="with --pool: how many cards you need")
    rv.add_argument("--holder", required=True, help="who has priority (your session name)")
    rv.add_argument("--eta", default="now", help="when you need them: now, 30m, or a window like 10m-60m")
    rv.add_argument("--for", dest="for_", type=_duration, default=24 * 3600.0,
                    help="how long the priority lasts at most (default 24h); it also ends once you used the cards"
                    " and released them")  # fmt: skip
    rv.add_argument("--note", default="", help="what it is for, and who granted the priority")
    rv.add_argument("--end", nargs="?", const="", help="end your reservation (by id, or all of yours)")
    sub.add_parser("leases", help="list leases and reservations (no probe)")
    sub.add_parser("pools", help="list the configured pools")
    args = parser.parse_args(sys.argv[1:] or ["status"])

    try:
        pools = load_pools()
    except Exception as exc:  # a broken inventory must say so, not print a traceback mid-table
        sys.exit(f"[res] cannot read the compute inventory: {type(exc).__name__}: {exc}")
    if args.cmd == "pools":
        for p in pools:
            detail = p.settings.get("hosts") or p.settings.get("login") or p.settings.get("address") or ""
            print(f"{p.name:<16} {p.kind:<6} {detail}")
        return
    commands = {
        "status": cmd_status,
        "claim": cmd_claim,
        "release": cmd_release,
        "transfer": cmd_transfer,
        "reserve": cmd_reserve,
        "leases": cmd_leases,
    }
    commands[args.cmd](args, pools)


if __name__ == "__main__":
    main()
