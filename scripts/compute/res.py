#!/usr/bin/env python3
"""`just res`: one view of every GPU on every compute pool, probed live (never declared), plus card leases."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import leases as ls
from inventory import Pool, load_pools
from probe import Card, Report, probe_local, probe_ray, probe_slurm_login, probe_ssh_host


def _guarded(fn: Callable[..., list[Report]], label: str, kind: str, *args: Any) -> list[Report]:
    """Run one probe; an unexpected exception becomes that probe's UNKNOWN report instead of ending the run."""
    try:
        return fn(*args)
    except Exception as exc:  # a probe bug or missing tool must not hide the other pools
        return [Report(label, kind, error=f"probe failed: {type(exc).__name__}: {exc}")]


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
            by_login.setdefault(pool.settings["login"], []).append(pool)
        else:
            tasks.append((lambda: [], pool.name, pool.kind))
            print(f"[res] unknown kind '{pool.kind}' for pool {pool.name}", file=sys.stderr)
    tasks.extend((probe_slurm_login, login, "slurm", login, group) for login, group in by_login.items())
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda t: _guarded(*t), tasks))
    reports = [r for group in results for r in group]
    seen = set()
    for rep in reports:
        rep.cards = [c for c in rep.cards if not c.uuid or c.uuid not in seen]
        seen.update(c.uuid for c in rep.cards if c.uuid)
    return reports


def _ago(ts: float) -> str:
    mins = int((time.time() - ts) // 60)
    return f"{mins // 60}h{mins % 60:02d}m" if mins >= 60 else f"{mins}m"


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
                who = f"[{lease.holder}: {lease.note or lease.id}] {who}".strip()
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
    return [p for p in pools if not prefixes or any(p.name.startswith(sel) for sel in prefixes)]


def reconcile(reports: list[Report]) -> dict[str, ls.Lease]:
    """Apply the probe to the lease store and return the remaining leases by card."""
    with ls.locked_store() as leases:
        for lease, why in ls.reconcile(leases, reports):
            print(f"[res] released {lease.id} ({lease.card}, {lease.holder}): {why}", file=sys.stderr)
        return {lease.card: lease for lease in leases}


def cmd_status(args: argparse.Namespace, pools: list[Pool]) -> None:
    """Probe, reconcile leases and print every card."""
    reports = probe_all(select_pools(pools, args.pool))
    held = reconcile(reports)
    if args.free:
        for rep in reports:
            rep.cards = [c for c in rep.cards if c.state == "free" and ls.card_key(c) not in held]
    if args.json:
        out = {"reports": [r.to_dict() for r in reports], "leases": [vars(x) for x in held.values()]}
        print(json.dumps(out, indent=1))
        return
    text, _ = render(reports, held)
    print(text)
    if reports and all(r.error for r in reports):
        sys.exit(2)


def _choose(cards: dict[str, Card], taken: set[str], args: argparse.Namespace) -> list[Card]:
    if args.any:
        free = [c for c in cards.values() if c.state == "free" and ls.card_key(c) not in taken]
        free = [c for c in free if c.mem_total - c.mem_used >= args.min_free_gb * 1024]
        free.sort(key=lambda c: (-(c.mem_total - c.mem_used), c.host, c.index))
        if len(free) < args.count:
            sys.exit(f"[res] only {len(free)} free card(s) match; nothing claimed")
        return free[: args.count]
    chosen = []
    for key in args.cards:
        card = cards.get(key)
        if card is None:
            sys.exit(f"[res] {key}: not found or its pool is unreachable; nothing claimed")
        if card.state != "free" or key in taken:
            sys.exit(f"[res] {key} is {'leased' if key in taken else card.state}; nothing claimed")
        chosen.append(card)
    return chosen


def cmd_claim(args: argparse.Namespace, pools: list[Pool]) -> None:
    """Lease named cards, or --any free ones, after a fresh probe."""
    if not args.cards and not args.any:
        sys.exit("[res] name cards (host:gpu ...) or pass --any")
    reports = probe_all(select_pools(pools, args.pool))
    cards = {ls.card_key(c): c for r in reports for c in r.cards}
    with ls.locked_store() as leases:
        ls.reconcile(leases, reports)
        chosen = _choose(cards, {lease.card for lease in leases}, args)
        new = [ls.new_lease(c, args.holder, args.note, ls.parse_duration(args.for_)) for c in chosen]
        leases.extend(new)
    for lease, card in zip(new, chosen, strict=True):
        where = f"{card.pool}, job {card.job}" if card.job else card.pool
        box = f", time box {args.for_}" if args.for_ else ""
        print(f"claimed {lease.id}: {lease.card} ({where}{box}) for {lease.holder}")
    print(
        f"Start using the card(s) within {ls.GRACE_S // 60} min; a lease is released after "
        f"{ls.IDLE_S // 60} min without activity. Release early with: just res release <id>"
    )


def cmd_release(args: argparse.Namespace, _pools: list[Pool]) -> None:
    """Drop leases by id or card."""
    with ls.locked_store() as leases:
        gone = [x for x in leases if x.id in args.targets or x.card in args.targets]
        for lease in gone:
            leases.remove(lease)
    for lease in gone:
        print(f"released {lease.id}: {lease.card} ({lease.holder})")
    missing = set(args.targets) - {x.id for x in gone} - {x.card for x in gone}
    if missing:
        sys.exit(f"[res] no lease for: {', '.join(sorted(missing))}")


def cmd_leases(_args: argparse.Namespace, _pools: list[Pool]) -> None:
    """List the stored leases without probing."""
    with ls.locked_store() as leases:
        rows = list(leases)
    if not rows:
        print("no leases")
    for x in rows:
        act = f"active {_ago(x.last_active)} ago" if x.last_active else f"claimed {_ago(x.created)} ago, not used yet"
        box = f", time box ends in {int((x.expires - time.time()) // 60)}m" if x.expires else ""
        print(f"{x.id}  {x.card:<18} {x.pool:<28} {x.holder:<24} {act}{box}  {x.note}")


def main() -> None:
    """Parse the CLI and run a command."""
    parser = argparse.ArgumentParser(prog="just res", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    st = sub.add_parser("status", help="probe every pool and show each card's state and lease")
    st.add_argument("--pool", action="append", help="only these pools (repeatable; prefix match)")
    st.add_argument("--free", action="store_true", help="only list free, unleased cards")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    cl = sub.add_parser("claim", help="lease cards (host:gpu ...) or --any free ones")
    cl.add_argument("cards", nargs="*", help="cards as host:gpu, e.g. mckennie:1 gpub065:2")
    cl.add_argument("--any", action="store_true", help="pick free cards instead of naming them")
    cl.add_argument("--count", type=int, default=1, help="with --any: how many cards")
    cl.add_argument("--min-free-gb", type=float, default=0, help="with --any: free memory each card needs")
    cl.add_argument("--pool", action="append", help="only probe/pick from these pools (prefix match)")
    cl.add_argument("--holder", default=os.environ.get("USER", "?"), help="who holds it (your session name)")
    cl.add_argument("--note", default="", help="what it is for")
    cl.add_argument("--for", dest="for_", default="", help="hard time box, e.g. 90m or 2h (interactive work)")
    rl = sub.add_parser("release", help="release leases by id or host:gpu")
    rl.add_argument("targets", nargs="+")
    sub.add_parser("leases", help="list leases (no probe)")
    sub.add_parser("pools", help="list the configured pools")
    args = parser.parse_args(sys.argv[1:] or ["status"])

    pools = load_pools()
    if args.cmd == "pools":
        for p in pools:
            detail = p.settings.get("hosts") or p.settings.get("login") or p.settings.get("address") or ""
            print(f"{p.name:<16} {p.kind:<6} {detail}")
        return
    commands = {"status": cmd_status, "claim": cmd_claim, "release": cmd_release, "leases": cmd_leases}
    commands[args.cmd](args, pools)


if __name__ == "__main__":
    main()
