"""Card leases: who holds which GPU, kept alive by its holder's activity on the card and released when it stops.

Reservations give one holder priority on cards (named ones, or N of a pool) from an ETA: others may lease the cards
meanwhile, and those leases yield (move to the priority holder) when it claims them. A reservation ends at its time
box, or once its holder has claimed under it and holds none of its cards any more (the prioritized run is done).
"""

from __future__ import annotations

import errno
import fcntl
import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from probe import HELD_UTIL, Card, Report

# One store per OS user and machine; flock needs a local filesystem (an NFS home may not honour it).
STORE = Path.home() / ".local" / "state" / "hcrl_res" / "leases.json"
# Beside the leases, in its own file, so a res that predates reservations still reads the lease store.
RESERVATIONS = "reservations.json"
LOCK_WAIT_S = 10
# Defaults for compute.toml's [leases] grace_min / idle_min. 30 min covers the ~15 min stall watchdog plus a
# relaunch and Kit boot, so a crashed run's relaunch keeps its card; a design choice, not a measurement.
GRACE_MIN = 30
IDLE_MIN = 30


class LeaseStoreError(Exception):
    """The lease store could not be read or locked; callers must not act on an empty lease list."""


@dataclass
class Lease:
    """A claim on one card.

    Args:
        id: Short lease id.
        key: Unique card key (uuid, or host:job:index when the card has none).
        card: Card shown to people, `host:gpu` (plus `job` on SLURM).
        report: Probe report the card was seen in (pool, host or SLURM login group).
        holder: Who holds it (session name).
        owner: OS user whose processes count as the holder's ("" = cannot attribute, e.g. Ray).
        job: SLURM job id of the card's allocation ("" elsewhere).
        note: What it is for.
        created: Unix time of the claim.
        last_active: Last time the holder's activity was seen on the card (0 = never).
        idle_since: First idle observation since the last activity (0 = not idle).
        expires: Hard end for time-boxed leases (0 = none).
        conflict: The card's processes belong to someone else.
        missing_since: First probe that no longer listed the card (0 = listed).
        run: The run the lease covers (W&B id, job.step, ...), when known.
        previous: Holder the lease came from (transfer) or the run was adopted from.
        yield_to: Id of the reservation this lease gives way to ("" = none).
    """

    id: str
    key: str
    card: str
    report: str
    holder: str
    owner: str = ""
    job: str = ""
    note: str = ""
    created: float = field(default_factory=time.time)
    last_active: float = 0.0
    idle_since: float = 0.0
    expires: float = 0.0
    conflict: bool = False
    missing_since: float = 0.0
    run: str = ""
    previous: str = ""
    yield_to: str = ""


@dataclass
class Reservation:
    """Priority on cards for one holder, from an ETA window until its time box.

    Args:
        id: Short reservation id.
        holder: Who has priority (session name).
        keys: Comma-separated card keys of the named cards ("" for a pool reservation).
        cards: The named cards as people read them (``host:gpu``, comma-separated).
        pool: Pool-name prefix of a pool reservation ("" for named cards).
        count: Cards the holder needs in the pool (pool reservations).
        eta_lo: Earliest the holder needs the cards (Unix time).
        eta_hi: Latest the holder expects to need them.
        expires: Hard end of the priority (Unix time).
        note: What it is for (and who granted it).
        created: Unix time of the reservation.
        used: The holder has claimed cards under it; holding none of them again ends it.
    """

    id: str
    holder: str
    keys: str = ""
    cards: str = ""
    pool: str = ""
    count: int = 0
    eta_lo: float = 0.0
    eta_hi: float = 0.0
    expires: float = 0.0
    note: str = ""
    created: float = field(default_factory=time.time)
    used: bool = False

    def scope(self) -> str:
        return f"{self.count} of {self.pool}" if self.pool else self.cards


def report_pools(label: str) -> set[str]:
    """The pool names behind a report label: ``pool/host``, or ``site (pool, pool)`` for a SLURM login."""
    if "(" in label:
        return {name.strip() for name in label.split("(", 1)[1].split(")", 1)[0].split(",")}
    return {label.split("/", 1)[0]}


def covers(res: Reservation, key: str, report_label: str) -> bool:
    """True if the reservation covers the card ``key``, seen in the report ``report_label``."""
    if res.pool:
        return any(name.startswith(res.pool) for name in report_pools(report_label))
    return key in res.keys.split(",")


def covering(reservations: list[Reservation], key: str, report_label: str) -> Reservation | None:
    """The reservation covering a card (the soonest needed if several do)."""
    hits = [r for r in reservations if covers(r, key, report_label)]
    return min(hits, key=lambda r: r.eta_lo) if hits else None


def parse_eta(text: str, now: float | None = None) -> tuple[float, float]:
    """``now``, ``30m`` or ``10m-60m`` as an absolute (earliest, latest) window; durations as parse_duration reads.

    Args:
        text: The ETA from now.
        now: Current Unix time (tests inject it).

    Returns:
        (earliest, latest) Unix times; raises ValueError for anything else.
    """
    now = time.time() if now is None else now
    if text == "now":
        return now, now
    lo, _, hi = text.partition("-")
    lo_s, hi_s = parse_duration(lo), parse_duration(hi or lo)
    if hi_s < lo_s:
        raise ValueError(f"ETA window ends before it starts: {text}")
    return now + lo_s, now + hi_s


def eta_text(res: Reservation, now: float | None = None) -> str:
    """When the holder needs the cards, from now: ``now``, ``in 25m`` or ``in 10-60m``."""
    now = time.time() if now is None else now
    lo, hi = max(0, round((res.eta_lo - now) / 60)), max(0, round((res.eta_hi - now) / 60))
    if hi == 0:
        return "now"
    return f"in {lo}m" if lo == hi else f"in {lo}-{hi}m"


def mark_yields(res: Reservation, leases: list[Lease]) -> list[Lease]:
    """Mark other holders' leases on the reservation's cards as yielding to it; return every yielding lease."""
    out = []
    for lease in leases:
        if lease.holder != res.holder and covers(res, lease.key, lease.report):
            lease.yield_to = lease.yield_to or res.id
            if lease.yield_to == res.id:
                out.append(lease)
    return out


def settle(
    reservations: list[Reservation], leases: list[Lease], now: float | None = None
) -> list[tuple[Reservation, str]]:
    """End reservations past their time box, or used ones whose holder holds none of their cards; their leases stop
    yielding.

    Args:
        reservations: Reservations from locked_state(), changed in place.
        leases: The current leases, whose yield marks are cleared for ended reservations.
        now: Current Unix time (tests inject it).

    Returns:
        The ended reservations with the reason for each.
    """
    now = time.time() if now is None else now
    ended = []
    for res in list(reservations):
        if now >= res.expires:
            ended.append((res, "time box ended"))
        elif res.used and not any(x.holder == res.holder and covers(res, x.key, x.report) for x in leases):
            ended.append((res, "its holder used the cards and released them"))
    for res, _ in ended:
        reservations.remove(res)
        for lease in leases:
            if lease.yield_to == res.id:
                lease.yield_to = ""
    return ended


def card_key(card: Card) -> str:
    """Unique key of a card: its uuid, or host:job:index (SLURM jobs on one node can share an index)."""
    return card.uuid or f"{card.host}:{card.job}:{card.index}"


def card_label(card: Card) -> str:
    """`host:gpu` for people, with the SLURM job when there is one."""
    return f"{card.host}:{card.index}" + (f" (job {card.job})" if card.job else "")


_TYPES = {"str": str, "float": (int, float), "bool": bool, "int": int}


def _lease(d: dict, cls: type = Lease) -> Lease:
    """Lease (or ``cls``) from stored data; unknown fields are ignored, wrong types raise TypeError."""
    values = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val, want = d[f.name], _TYPES[f.type]
        if not isinstance(val, want) or (f.type == "float" and isinstance(val, bool)):
            raise TypeError(f"{f.name} has type {type(val).__name__}")
        values[f.name] = val
    return cls(**values)


def _load(path: Path, cls: type = Lease) -> list:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise TypeError("not a list of leases")
        return [_lease(d, cls) for d in data]
    except (ValueError, TypeError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        aside = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
        path.rename(aside)
        raise LeaseStoreError(f"lease store was unreadable ({exc}); moved to {aside}") from exc


@contextmanager
def locked_store() -> Iterator[list[Lease]]:
    """Yield the lease list under an exclusive lock; write it back only if the block changed it and finished.

    Raises LeaseStoreError when the lock is not free within LOCK_WAIT_S or the store cannot be read or written.
    """
    with locked_state() as (leases, _):
        yield leases


def _write(path: Path, rows: list) -> None:
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        tmp.replace(path)
    except OSError as exc:
        raise LeaseStoreError(f"cannot write the lease store ({exc})") from exc


@contextmanager
def locked_state() -> Iterator[tuple[list[Lease], list[Reservation]]]:
    """Yield the leases and the reservations under the store's lock; each is written back only if changed."""
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        lock = open(STORE.with_suffix(".lock"), "w")  # noqa: SIM115 - closed by the with below
    except OSError as exc:
        raise LeaseStoreError(f"cannot open the lease store ({exc})") from exc
    with lock:
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES) or time.monotonic() > deadline:
                    raise LeaseStoreError(f"lease store is locked by another res call ({exc})") from exc
                time.sleep(0.1)
        rpath = STORE.with_name(RESERVATIONS)
        try:
            leases, reservations = _load(STORE), _load(rpath, Reservation)
        except OSError as exc:
            raise LeaseStoreError(f"cannot read the lease store ({exc})") from exc
        before = [asdict(x) for x in leases], [asdict(x) for x in reservations]
        yield leases, reservations
        after = [asdict(x) for x in leases], [asdict(x) for x in reservations]
        if after[0] != before[0]:
            _write(STORE, after[0])
        if after[1] != before[1]:
            _write(rpath, after[1])


def _uncertain(reports: list[Report]) -> tuple[set[str], set[tuple[str, str]]]:
    """Reports that listed all their cards, and (report, job) pairs a failed SLURM sub-report left unknown."""
    ok = {r.pool for r in reports if not r.error and not r.partial}
    jobs = set()
    for r in reports:
        if r.error and " job " in r.pool:
            label, _, job = r.pool.rpartition(" job ")
            jobs.add((label, job))
    return ok, jobs


def _activity(lease: Lease, card: Card) -> bool | None:
    """True if the holder's activity shows on the card, False if it is idle, None if it cannot be told."""
    if card.state == "unknown":
        return None
    if card.procs:
        # only attributable processes can show a conflict; unattributed ones (Ray, "?") are not evidence
        users = {p.user for p in card.procs} - {"?"}
        lease.conflict = bool(lease.owner) and bool(users) and lease.owner not in users
        return not lease.conflict
    lease.conflict = False
    # busy without a listed process: in use but unattributable (Ray). Retained memory with no process and no
    # utilization is a finished run holding VRAM, not work.
    return card.state == "busy" or (card.state == "held" and card.util >= HELD_UTIL)


def reconcile(
    leases: list[Lease],
    reports: list[Report],
    now: float | None = None,
    grace_min: float = GRACE_MIN,
    idle_min: float = IDLE_MIN,
) -> list[tuple[Lease, str]]:
    """Apply one probe to the leases: renew on the holder's activity, release time-boxed, vanished or idle ones.

    Idle time runs from the first idle observation, so a lease is only released after two observations at least
    the window apart (grace before any activity was seen, idle after). Cards that could not be read are left alone.

    Args:
        leases: Lease list from locked_store(), changed in place.
        reports: Probe results.
        now: Current Unix time (tests inject it).
        grace_min: Window for a lease whose activity was never seen.
        idle_min: Window for a lease that was active.

    Returns:
        The released leases with the reason for each.
    """
    now = time.time() if now is None else now
    cards = {card_key(c): c for r in reports for c in r.cards}
    ok, uncertain_jobs = _uncertain(reports)
    released = []
    for lease in list(leases):
        card = cards.get(lease.key)
        if lease.expires and now >= lease.expires:
            released.append((lease, "time box ended"))
            continue
        if card is None:
            if lease.report in ok and (lease.report, lease.job) not in uncertain_jobs:
                # like idle: gone on two complete probes at least the idle window apart
                if not lease.missing_since:
                    lease.missing_since = now
                elif now - lease.missing_since >= idle_min * 60:
                    released.append((lease, "card no longer exists"))
            continue
        lease.missing_since = 0.0
        active = _activity(lease, card)
        if active is None:
            continue
        if active:
            lease.last_active, lease.idle_since = now, 0.0
            continue
        if not lease.idle_since:
            lease.idle_since = now
            continue
        window = (idle_min if lease.last_active else grace_min) * 60
        if now - lease.idle_since >= window:
            what = "idle" if lease.last_active else "unused"
            released.append((lease, f"{what} for {int((now - lease.idle_since) // 60)} min"))
    for lease, _ in released:
        leases.remove(lease)
    return released


def new_lease(
    card: Card, report: Report, holder: str, note: str, for_s: float, run: str = "", previous: str = ""
) -> Lease:
    """Lease on a card, held by `holder` (attributed to the report's OS user); a card already in use (adopted) starts
    active, so its run renews the lease and its end releases it."""
    now = time.time()
    return Lease(
        id=uuid.uuid4().hex[:6],
        key=card_key(card),
        card=card_label(card),
        report=report.pool,
        holder=holder,
        owner=report.owner,
        job=card.job,
        note=note,
        created=now,
        expires=now + for_s if for_s else 0.0,
        last_active=now if card.state in ("busy", "held") else 0.0,
        run=run,
        previous=previous,
    )


def parse_duration(text: str) -> float:
    """Parse a positive `90s`, `30m`, `2h`, `1d` (bare numbers are minutes) into seconds.

    Args:
        text: The duration.

    Returns:
        Seconds; raises ValueError for anything else.
    """
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(text[-1:])
    seconds = float(text[:-1]) * unit if unit else float(text) * 60
    if seconds <= 0:
        raise ValueError(f"duration must be positive: {text}")
    return seconds
