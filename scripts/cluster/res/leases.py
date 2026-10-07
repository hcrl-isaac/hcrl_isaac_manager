"""Card leases: who holds which GPU, kept alive by its holder's activity on the card and released when it stops."""

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


def card_key(card: Card) -> str:
    """Unique key of a card: its uuid, or host:job:index (SLURM jobs on one node can share an index)."""
    return card.uuid or f"{card.host}:{card.job}:{card.index}"


def card_label(card: Card) -> str:
    """`host:gpu` for people, with the SLURM job when there is one."""
    return f"{card.host}:{card.index}" + (f" (job {card.job})" if card.job else "")


_TYPES = {"str": str, "float": (int, float), "bool": bool}


def _lease(d: dict) -> Lease:
    """Lease from stored data; unknown fields are ignored, wrong types raise TypeError."""
    values = {}
    for f in fields(Lease):
        if f.name not in d:
            continue
        val, want = d[f.name], _TYPES[f.type]
        if not isinstance(val, want) or (f.type == "float" and isinstance(val, bool)):
            raise TypeError(f"{f.name} has type {type(val).__name__}")
        values[f.name] = val
    return Lease(**values)


def _load(path: Path) -> list[Lease]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise TypeError("not a list of leases")
        return [_lease(d) for d in data]
    except (ValueError, TypeError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        aside = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
        path.rename(aside)
        raise LeaseStoreError(f"lease store was unreadable ({exc}); moved to {aside}") from exc


@contextmanager
def locked_store() -> Iterator[list[Lease]]:
    """Yield the lease list under an exclusive lock; write it back only if the block changed it and finished.

    Raises LeaseStoreError when the lock is not free within LOCK_WAIT_S or the store cannot be read or written.
    """
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
        try:
            leases = _load(STORE)
        except OSError as exc:
            raise LeaseStoreError(f"cannot read the lease store ({exc})") from exc
        before = [asdict(lease) for lease in leases]
        yield leases
        after = [asdict(lease) for lease in leases]
        if after != before:
            try:
                tmp = STORE.with_suffix(".tmp")
                tmp.write_text(json.dumps(after, indent=1))
                tmp.replace(STORE)
            except OSError as exc:
                raise LeaseStoreError(f"cannot write the lease store ({exc})") from exc


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
