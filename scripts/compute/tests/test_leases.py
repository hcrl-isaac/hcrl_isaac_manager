"""Unit tests for card leases (run all script tests: just test-scripts). Every test uses a temporary store."""

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

COMPUTE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPUTE))
import leases as ls
import res
from inventory import Pool
from probe import Card, Proc, Report

T0 = 1_000_000.0
MIN = 60.0


def card(host: str = "mckennie", index: int = 1, state: str = "free", user: str = "me", job: str = "") -> Card:
    procs = [Proc(1, 4000, user)] if state == "busy" else []
    used = 4000 if state in ("busy", "held") else 0
    return Card(
        "larg", host, index, "A100", used, 81920, 0, state, job=job, uuid=f"GPU-{host}-{job}-{index}", procs=procs
    )


def report(*cards: Card, label: str = "larg/mckennie", error: str = "") -> Report:
    return Report(label, "ssh", list(cards), error=error, owner="me")


def lease_on(c: Card, **kw: object) -> ls.Lease:
    lease = ls.new_lease(c, report(c), "s", "", 0)
    lease.created = T0
    for key, val in kw.items():
        setattr(lease, key, val)
    return lease


class TempStore(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.saved = ls.STORE
        ls.STORE = Path(self.dir.name) / "leases.json"

    def tearDown(self) -> None:
        ls.STORE = self.saved
        self.dir.cleanup()


class ReconcileTest(unittest.TestCase):
    def run_at(self, lease: ls.Lease, rep: list[Report], minute: float) -> list:
        self.leases = [lease] if not hasattr(self, "leases") else self.leases
        return ls.reconcile(self.leases, rep, now=T0 + minute * MIN, grace_min=30, idle_min=30)

    def test_activity_renews_and_clears_idle(self) -> None:
        c = card(state="busy")
        self.leases = [lease_on(c, idle_since=T0)]
        self.assertEqual(self.run_at(None, [report(c)], 5), [])
        self.assertEqual((self.leases[0].last_active, self.leases[0].idle_since), (T0 + 5 * MIN, 0.0))

    def test_unknown_card_leaves_the_lease_alone(self) -> None:
        c = card()
        self.leases = [lease_on(c, idle_since=T0)]
        unknown = card(state="unknown")
        self.assertEqual(self.run_at(None, [report(unknown)], 500), [])
        self.assertEqual(self.leases[0].idle_since, T0)

    def test_errored_report_leaves_the_lease_alone(self) -> None:
        c = card()
        self.leases = [lease_on(c, idle_since=T0)]
        self.assertEqual(self.run_at(None, [report(error="unreachable")], 500), [])

    def test_vanished_card_is_released_when_its_report_probed_cleanly(self) -> None:
        self.leases = [lease_on(card())]
        released = self.run_at(None, [report(card(index=3))], 1)
        self.assertEqual([why for _, why in released], ["card no longer exists"])

    def test_vanished_slurm_card_kept_while_its_job_is_uncertain(self) -> None:
        c = card(job="42")
        lease = ls.new_lease(c, Report("delta (delta)", "slurm", [c]), "s", "", 0)
        self.leases = [lease]
        reports = [Report("delta (delta)", "slurm"), Report("delta (delta) job 42", "slurm", error="saw 1 of 4")]
        self.assertEqual(self.run_at(None, reports, 1), [])

    def test_sparse_sampling_needs_two_idle_observations_a_window_apart(self) -> None:
        c_busy, c_free = card(state="busy"), card()
        self.leases = [lease_on(c_busy)]
        self.run_at(None, [report(c_busy)], 1)
        self.assertEqual(self.run_at(None, [report(c_free)], 179), [])  # first idle observation
        self.assertEqual(self.run_at(None, [report(c_free)], 179 + 29), [])
        released = self.run_at(None, [report(c_free)], 179 + 30)
        self.assertEqual([why for _, why in released], ["idle for 30 min"])

    def test_unused_lease_uses_the_grace_window(self) -> None:
        c = card()
        self.leases = [lease_on(c)]
        self.run_at(None, [report(c)], 26)
        self.assertEqual(len(self.run_at(None, [report(c)], 26 + 30)), 1)

    def test_someone_elses_process_does_not_renew(self) -> None:
        c = card(state="busy", user="squatter")
        self.leases = [lease_on(c, last_active=T0)]
        self.run_at(None, [report(c)], 1)
        self.assertTrue(self.leases[0].conflict)
        self.assertEqual(self.leases[0].last_active, T0)
        self.assertEqual(len(self.run_at(None, [report(c)], 31)), 1)

    def test_retained_memory_without_process_does_not_renew(self) -> None:
        c = card(state="held")
        self.leases = [lease_on(c, last_active=T0)]
        self.run_at(None, [report(c)], 1)
        self.assertEqual(self.leases[0].last_active, T0)

    def test_time_box_ends_even_when_active(self) -> None:
        c = card(state="busy")
        self.leases = [lease_on(c, expires=T0 + 100)]
        self.assertEqual(len(self.run_at(None, [report(c)], 2)), 1)


class StoreTest(TempStore):
    def test_roundtrip_and_unknown_fields_ignored(self) -> None:
        with ls.locked_store() as leases:
            leases.append(lease_on(card()))
        data = json.loads(ls.STORE.read_text())
        data[0]["future_field"] = 1
        ls.STORE.write_text(json.dumps(data))
        with ls.locked_store() as leases:
            self.assertEqual([x.card for x in leases], ["mckennie:1"])

    def test_corrupt_store_is_moved_aside(self) -> None:
        for bad in ("", "{not json", "{}", '[{"id": "x"}]'):
            ls.STORE.write_text(bad)
            with self.assertRaises(ls.LeaseStoreError), ls.locked_store():
                pass
            self.assertFalse(ls.STORE.exists())

    def test_no_write_when_nothing_changed(self) -> None:
        with ls.locked_store() as leases:
            leases.append(lease_on(card()))
        mtime = ls.STORE.stat().st_mtime_ns
        with ls.locked_store():
            pass
        self.assertEqual(ls.STORE.stat().st_mtime_ns, mtime)

    def test_lock_timeout(self) -> None:
        saved, ls.LOCK_WAIT_S = ls.LOCK_WAIT_S, 0.2
        try:
            with ls.locked_store(), self.assertRaises(ls.LeaseStoreError), ls.locked_store():
                pass
        finally:
            ls.LOCK_WAIT_S = saved

    def test_concurrent_writers_lose_nothing(self) -> None:
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; import leases as ls\n"
            "ls.STORE = Path(sys.argv[2])\n"
            "for i in range(25):\n"
            "    with ls.locked_store() as leases:\n"
            "        leases.append(ls.Lease(id=f'{sys.argv[3]}{i}', key='k', card='c', report='r', holder='h'))\n"
        )
        procs = [
            subprocess.Popen([sys.executable, "-c", code, str(COMPUTE), str(ls.STORE), f"p{n}-"]) for n in range(4)
        ]
        self.assertEqual([p.wait() for p in procs], [0, 0, 0, 0])
        with ls.locked_store() as leases:
            self.assertEqual(len({x.id for x in leases}), 100)

    def test_parse_duration(self) -> None:
        self.assertEqual([ls.parse_duration(x) for x in ("90s", "30m", "2h", "1d", "5")], [90, 1800, 7200, 86400, 300])
        for bad in ("abc", "-5m", "0", ""):
            with self.assertRaises(ValueError):
                ls.parse_duration(bad)


class CommandTest(TempStore):
    def setUp(self) -> None:
        super().setUp()
        self.saved_probe = res.probe_all
        self.cards = [card(index=0), card(index=1), card(index=2, state="busy")]
        res.probe_all = lambda pools: [report(*self.cards)]
        self.pools = [Pool("larg-a100", "ssh", {})]

    def tearDown(self) -> None:
        res.probe_all = self.saved_probe
        super().tearDown()

    def claim(self, *cards: str, **kw: object) -> None:
        args = {
            "cards": list(cards),
            "any": False,
            "count": 1,
            "min_free_gb": 0,
            "pool": None,
            "holder": "s",
            "note": "",
            "for_": 0.0,
        }
        args.update(kw)
        res.cmd_claim(argparse.Namespace(**args), self.pools)

    def stored(self) -> list[str]:
        with ls.locked_store() as leases:
            return sorted(x.card for x in leases)

    def test_duplicate_names_give_one_lease(self) -> None:
        self.claim("mckennie:0", "mckennie:0")
        self.assertEqual(self.stored(), ["mckennie:0"])

    def test_all_or_nothing(self) -> None:
        with self.assertRaises(SystemExit):
            self.claim("mckennie:0", "mckennie:2")
        self.assertEqual(self.stored(), [])

    def test_ambiguous_slurm_card_is_refused(self) -> None:
        self.cards = [card(host="gpub1", index=0, job="1"), card(host="gpub1", index=0, job="2")]
        with self.assertRaises(SystemExit):
            self.claim("gpub1:0")
        self.claim("gpub1:2:0")
        self.assertEqual(self.stored(), ["gpub1:0 (job 2)"])

    def test_any_respects_count(self) -> None:
        self.claim(any=True, count=2)
        self.assertEqual(self.stored(), ["mckennie:0", "mckennie:1"])
        with self.assertRaises(SystemExit):
            self.claim(any=True, count=1)

    def test_bad_count_and_duration_rejected_by_the_parser(self) -> None:
        saved = sys.argv
        bad = (["--any", "--count=0"], ["--any", "--count=-1"], ["a:0", "--for=-5m"], ["a:0", "--for=abc"])
        try:
            for argv in bad:
                sys.argv = ["res", "claim", *argv, "--holder", "s"]
                with self.assertRaises(SystemExit) as ctx:
                    res.main()
                self.assertEqual(ctx.exception.code, 2, argv)
        finally:
            sys.argv = saved
        self.assertEqual(self.stored(), [])

    def test_release_needs_the_holder_or_force(self) -> None:
        self.claim("mckennie:0")
        with self.assertRaises(SystemExit):
            res.cmd_release(argparse.Namespace(targets=["mckennie:0"], holder="other", force=False), self.pools)
        self.assertEqual(self.stored(), ["mckennie:0"])
        res.cmd_release(argparse.Namespace(targets=["mckennie:0"], holder="other", force=True), self.pools)
        self.assertEqual(self.stored(), [])


if __name__ == "__main__":
    unittest.main()
