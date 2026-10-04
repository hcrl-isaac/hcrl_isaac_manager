#!/usr/bin/env python3
"""`just res`: one view of every GPU on every compute pool, probed live (never declared)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from inventory import Pool, load_pools
from probe import Card, Report, probe_local, probe_ray, probe_slurm_login, probe_ssh_host

CAUTION = {"free": 0, "held": 1, "busy": 2, "unknown": 3}


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
                other.state, other.note = card.state, card.note or other.note
            other.procs.extend(card.procs)
        rep.cards = kept
    return reports


def render(reports: list[Report]) -> tuple[str, Counter]:
    """Format reports as a per-pool table.

    Args:
        reports: Probe results.

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
            totals[c.state] += 1
            who = "; ".join(f"{p.user} {p.cmd or p.pid} {p.elapsed}".strip() for p in c.procs) or c.note
            if c.state == "held" and not who:
                who = "in use, no visible process"
            where = f"{c.host} {c.job}".strip()
            model = c.model.replace("NVIDIA ", "").replace("GeForce ", "")[:16]
            mem = f"{c.mem_used}/{c.mem_total}" if c.mem_used >= 0 else f"?/{c.mem_total}"
            row = (
                f"{where:<22} {c.index:>3} {model:<16} {mem:>15} {c.util:>4}%  {c.state:<7} {who:<44.44} {c.wall_left}"
            )
            lines.append("   " + row)
        lines.extend(f"   - {note}" for note in rep.notes)
    parts = [f"{totals[k]} {k}" for k in ("free", "busy", "held") if totals[k]]
    if totals["unknown"]:
        parts.append(f"{totals['unknown']} card(s)/pool(s) unknown")
    lines.append(f"\ncards: {', '.join(parts) or 'none found'}")
    return "\n".join(lines), totals


def main() -> None:
    """Parse the CLI and run a command."""
    parser = argparse.ArgumentParser(prog="just res", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    st = sub.add_parser("status", help="probe every pool and show each card's state")
    st.add_argument("--pool", action="append", help="only these pools (repeatable; prefix match)")
    st.add_argument("--free", action="store_true", help="only list free cards")
    st.add_argument("--json", action="store_true", help="machine-readable output")
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
    if args.pool:
        pools = [p for p in pools if any(p.name.startswith(sel) for sel in args.pool)]
        if not pools:
            sys.exit(f"[res] no pool matches {', '.join(args.pool)} (see: just res pools)")
    reports = probe_all(pools)
    if args.free:
        for rep in reports:
            rep.cards = [c for c in rep.cards if c.state == "free"]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=1))
    else:
        print(render(reports)[0])
    if reports and all(r.error for r in reports):
        sys.exit(2)


if __name__ == "__main__":
    main()
