"""Card leases: who holds which GPU, kept alive by activity on the card and released when it stops."""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from probe import Card, Report

STORE = Path.home() / ".local" / "state" / "hcrl_res" / "leases.json"
GRACE_S = 20 * 60  # a new lease must show activity within this window
IDLE_S = 15 * 60  # an active lease is released after this long without activity


@dataclass
class Lease:
    """A claim on one card.

    Args:
        id: Short lease id.
        card: `host:gpu` key of the card.
        pool: Pool (or SLURM login group) the card was probed in.
        holder: Who holds it (session name or user).
        note: What it is for.
        created: Unix time of the claim.
        last_active: Unix time the card last showed activity (0 = never).
        expires: Hard expiry for time-boxed leases (0 = none).
    """

    id: str
    card: str
    pool: str
    holder: str
    note: str = ""
    created: float = field(default_factory=time.time)
    last_active: float = 0.0
    expires: float = 0.0


def card_key(card: Card) -> str:
    return f"{card.host}:{card.index}"


@contextmanager
def locked_store() -> Iterator[list[Lease]]:
    """Yield the lease list under an exclusive lock; write it back only if the block finished without error."""
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        leases = [Lease(**d) for d in json.loads(STORE.read_text())] if STORE.is_file() else []
        try:
            yield leases
        except BaseException:
            raise
        else:
            tmp = STORE.with_suffix(".tmp")
            tmp.write_text(json.dumps([asdict(lease) for lease in leases], indent=1))
            tmp.replace(STORE)


def active(card: Card) -> bool:
    """A card shows activity when something holds memory on it (containers can hide the process)."""
    return card.state in ("busy", "held")


def reconcile(leases: list[Lease], reports: list[Report], now: float | None = None) -> list[tuple[Lease, str]]:
    """Renew leases whose card is active and drop expired or idle ones; returns (lease, reason) released.

    Leases on cards that were not probed (pool filtered out or unreachable) are left untouched.
    """
    now = now or time.time()
    cards = {card_key(c): c for r in reports for c in r.cards}
    released = []
    for lease in list(leases):
        card = cards.get(lease.card)
        if lease.expires and now >= lease.expires:
            released.append((lease, "time box ended"))
        elif card is None:
            continue
        elif active(card):
            lease.last_active = now
        elif lease.last_active and now - lease.last_active >= IDLE_S:
            released.append((lease, f"idle {int((now - lease.last_active) // 60)} min"))
        elif not lease.last_active and now - lease.created >= GRACE_S:
            released.append((lease, f"no activity within {GRACE_S // 60} min of the claim"))
    for lease, _ in released:
        leases.remove(lease)
    return released


def new_lease(card: Card, holder: str, note: str, for_s: float) -> Lease:
    now = time.time()
    return Lease(uuid.uuid4().hex[:6], card_key(card), card.pool, holder, note, now, 0.0, now + for_s if for_s else 0.0)


def parse_duration(text: str) -> float:
    """Parse `90s`, `30m`, `2h`, `1d` (bare numbers are minutes) into seconds."""
    if not text:
        return 0.0
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(text[-1])
    return float(text[:-1]) * unit if unit else float(text) * 60
