"""Unit tests for lease reconciliation (run: python3 -m unittest discover -s scripts/compute)."""

import tempfile
import unittest
from pathlib import Path

import leases as ls
from probe import Card, Proc, Report

T0 = 1_000_000.0


def card(host: str = "mckennie", index: int = 1, state: str = "free") -> Card:
    procs = [Proc(1, 4000, "sturman")] if state == "busy" else []
    used = 4000 if state in ("busy", "held") else 0
    return Card("larg", host, index, "A100", used, 81920, 0, state, procs=procs)


def report(*cards: Card) -> list[Report]:
    return [Report("larg", "ssh", list(cards))]


def lease(**kw: object) -> ls.Lease:
    base = {"id": "abc123", "card": "mckennie:1", "pool": "larg", "holder": "s", "created": T0}
    base.update(kw)
    return ls.Lease(**base)


class ReconcileTest(unittest.TestCase):
    def test_activity_renews(self) -> None:
        leases = [lease()]
        self.assertEqual(ls.reconcile(leases, report(card(state="busy")), now=T0 + 60), [])
        self.assertEqual(leases[0].last_active, T0 + 60)

    def test_held_memory_counts_as_activity(self) -> None:
        leases = [lease()]
        ls.reconcile(leases, report(card(state="held")), now=T0 + 60)
        self.assertEqual(leases[0].last_active, T0 + 60)

    def test_unused_lease_released_after_grace(self) -> None:
        leases = [lease()]
        self.assertEqual(ls.reconcile(leases, report(card()), now=T0 + ls.GRACE_S - 1), [])
        released = ls.reconcile(leases, report(card()), now=T0 + ls.GRACE_S)
        self.assertEqual(len(released), 1)
        self.assertEqual(leases, [])

    def test_idle_lease_released(self) -> None:
        leases = [lease(last_active=T0)]
        self.assertEqual(ls.reconcile(leases, report(card()), now=T0 + ls.IDLE_S - 1), [])
        self.assertEqual(len(ls.reconcile(leases, report(card()), now=T0 + ls.IDLE_S)), 1)

    def test_time_box_ends_even_when_active(self) -> None:
        leases = [lease(expires=T0 + 100)]
        self.assertEqual(len(ls.reconcile(leases, report(card(state="busy")), now=T0 + 100)), 1)

    def test_unprobed_card_left_alone(self) -> None:
        leases = [lease(created=T0 - 10 * ls.GRACE_S)]
        self.assertEqual(ls.reconcile(leases, report(card(host="hazard")), now=T0), [])
        self.assertEqual(len(leases), 1)


class StoreTest(unittest.TestCase):
    def test_roundtrip_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            saved, ls.STORE = ls.STORE, Path(d) / "leases.json"
            try:
                with ls.locked_store() as leases:
                    leases.append(ls.new_lease(card(), "s", "probe", 0))
                with ls.locked_store() as leases:
                    self.assertEqual([x.card for x in leases], ["mckennie:1"])
            finally:
                ls.STORE = saved

    def test_parse_duration(self) -> None:
        self.assertEqual(
            [ls.parse_duration(x) for x in ("90s", "30m", "2h", "1d", "5", "")], [90, 1800, 7200, 86400, 300, 0]
        )


if __name__ == "__main__":
    unittest.main()
