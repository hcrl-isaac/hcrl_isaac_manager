#!/usr/bin/env python3
"""`just res`: one view of every GPU on every compute pool, probed live (never declared)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from inventory import Pool, load_pools
from probe import Report, probe_local, probe_ray, probe_slurm_login, probe_ssh_host


def probe_all(pools: list[Pool]) -> list[Report]:
    """Probe every pool in parallel; SLURM profiles that share a login are probed once."""
    tasks = []
    by_login: dict[str, list[Pool]] = {}
    for pool in pools:
        if pool.kind == "local":
            tasks.append((probe_local, (pool,)))
        elif pool.kind == "ssh":
            tasks.extend((lambda p, h: [probe_ssh_host(p, h)], (pool, h)) for h in pool.settings.get("hosts", []))
        elif pool.kind == "ray":
            tasks.append((probe_ray, (pool,)))
        elif pool.kind == "slurm":
            by_login.setdefault(pool.settings["login"], []).append(pool)
        else:
            print(f"[res] unknown kind '{pool.kind}' for pool {pool.name}", file=sys.stderr)
    tasks.extend((probe_slurm_login, (login, group)) for login, group in by_login.items())
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda t: t[0](*t[1]), tasks))
    return [r for group in results for r in group]


def render(reports: list[Report]) -> str:
    lines = []
    head = f"{'HOST/JOB':<22} {'GPU':>3} {'MODEL':<16} {'MEM (MiB)':>15} {'UTIL':>5}  {'STATE':<6} {'WHO / WHAT':<44} LEFT"
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
            who = "; ".join(f"{p.user} {p.cmd or p.pid} {p.elapsed}".strip() for p in c.procs)
            if c.state == "held":
                who = "memory held, no process"
            where = f"{c.host} {c.job}".strip()
            model = c.model.replace("NVIDIA ", "").replace("GeForce ", "")[:16]
            mem = f"{c.mem_used}/{c.mem_total}"
            lines.append(f"   {where:<22} {c.index:>3} {model:<16} {mem:>15} {c.util:>4}%  {c.state:<6} {who:<44.44} {c.wall_left}")
        for note in rep.notes:
            lines.append(f"   - {note}")
    summary = ", ".join(f"{totals[k]} {k}" for k in ("free", "busy", "held") if totals[k])
    if totals["unknown"]:
        summary += f"; {totals['unknown']} pool(s)/host(s) unknown"
    lines.append(f"\ncards: {summary or 'none found'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="just res", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    st = sub.add_parser("status", help="probe every pool and show each card's state")
    st.add_argument("--pool", action="append", help="only these pools (repeatable; prefix match)")
    st.add_argument("--free", action="store_true", help="only list free cards")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    sub.add_parser("pools", help="list the configured pools")
    args = parser.parse_args(sys.argv[1:] or ["status"])

    pools = load_pools()
    if args.cmd == "pools":
        for p in pools:
            detail = p.settings.get("hosts") or p.settings.get("login") or p.settings.get("address") or ""
            print(f"{p.name:<16} {p.kind:<6} {detail}")
        return
    if args.pool:
        pools = [p for p in pools if any(p.name.startswith(sel) for sel in args.pool)]
    reports = probe_all(pools)
    if args.free:
        for rep in reports:
            rep.cards = [c for c in rep.cards if c.state == "free"]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=1))
    else:
        print(render(reports))


if __name__ == "__main__":
    main()
