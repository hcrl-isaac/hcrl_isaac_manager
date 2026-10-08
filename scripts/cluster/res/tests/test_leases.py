"""Unit tests for card leases (run all script tests: just test-scripts). Every test uses a temporary store."""

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RES))
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

    def test_vanished_card_needs_two_complete_probes_a_window_apart(self) -> None:
        self.leases = [lease_on(card())]
        self.assertEqual(self.run_at(None, [report(card(index=3))], 1), [])
        self.assertEqual(self.run_at(None, [report(card(index=3))], 30), [])
        released = self.run_at(None, [report(card(index=3))], 31)
        self.assertEqual([why for _, why in released], ["card no longer exists"])

    def test_reappearing_card_resets_the_vanish_clock(self) -> None:
        self.leases = [lease_on(card())]
        self.run_at(None, [report(card(index=3))], 1)
        self.run_at(None, [report(card())], 2)
        self.assertEqual(self.run_at(None, [report(card(index=3))], 40), [])

    def test_partial_report_proves_nothing_missing(self) -> None:
        self.leases = [lease_on(card())]
        partial = report(card(index=3))
        partial.partial = True
        for minute in (1, 100, 200):
            self.assertEqual(self.run_at(None, [partial], minute), [])

    def test_non_running_slurm_job_proves_nothing_missing(self) -> None:
        c = card(job="42")
        self.leases = [ls.new_lease(c, Report("delta (delta)", "slurm", [c]), "s", "", 0)]
        reports = [Report("delta (delta)", "slurm"), Report("delta (delta) job 42", "slurm", error="job is COMPLETING")]
        for minute in (1, 100):
            self.assertEqual(self.run_at(None, reports, minute), [])

    def test_unattributed_process_does_not_hide_a_squatter(self) -> None:
        c = card(state="busy", user="squatter")
        c.procs.append(Proc(9, 100, "?"))
        self.leases = [lease_on(c, last_active=T0)]
        self.run_at(None, [report(c)], 1)
        self.assertTrue(self.leases[0].conflict)

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
        procs = [subprocess.Popen([sys.executable, "-c", code, str(RES), str(ls.STORE), f"p{n}-"]) for n in range(4)]
        self.assertEqual([p.wait() for p in procs], [0, 0, 0, 0])
        with ls.locked_store() as leases:
            self.assertEqual(len({x.id for x in leases}), 100)

    def test_wrong_types_and_bad_bytes_are_moved_aside(self) -> None:
        good = json.loads(json.dumps([vars(lease_on(card()))]))
        for patch in ({"created": "yesterday"}, {"created": None}, {"conflict": 1}, {"holder": 5}):
            ls.STORE.write_text(json.dumps([dict(good[0], **patch)]))
            with self.assertRaises(ls.LeaseStoreError), ls.locked_store():
                pass
            self.assertFalse(ls.STORE.exists(), patch)
        ls.STORE.write_bytes(b"\xff\xfe[")
        with self.assertRaises(ls.LeaseStoreError), ls.locked_store():
            pass
        self.assertEqual(len(list(ls.STORE.parent.glob("leases.json.corrupt-*"))), 5)

    def test_unwritable_store_is_a_store_error(self) -> None:
        ls.STORE = Path(self.dir.name) / "ro" / "leases.json"
        ls.STORE.parent.mkdir()
        ls.STORE.parent.chmod(0o500)
        try:
            with self.assertRaises(ls.LeaseStoreError), ls.locked_store():
                pass
        finally:
            ls.STORE.parent.chmod(0o700)

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

    def test_refused_claim_still_persists_releases(self) -> None:
        with ls.locked_store() as leases:
            leases.append(lease_on(card(index=0), expires=T0))  # time box long over
        with self.assertRaises(SystemExit):
            self.claim("mckennie:2")  # busy card: refused
        self.assertEqual(self.stored(), [])

    def _probes(self, *rounds: list) -> list[int]:
        """Each probe returns the next round's cards (the last repeats); returns a probe counter."""
        calls = [0]

        def probe(pools: list) -> list:
            cards = rounds[min(calls[0], len(rounds) - 1)]
            calls[0] += 1
            return [report(*cards)]

        res.probe_all = probe
        return calls

    def test_wait_leases_the_card_once_it_frees(self) -> None:
        calls = self._probes([card(index=2, state="busy")], [card(index=2, state="busy")], [card(index=2)])
        with mock.patch.object(res, "WAIT_POLL_S", 0.01), contextlib.redirect_stderr(io.StringIO()) as err:
            self.claim("mckennie:2", wait=0.0)
        self.assertEqual(self.stored(), ["mckennie:2"])
        self.assertEqual(calls[0], 3)
        self.assertEqual(err.getvalue().count("waiting for one to free"), 1, "it says so once, not every poll")

    def test_wait_takes_the_first_free_any_match(self) -> None:
        self._probes([card(index=0, state="busy")], [card(index=0), card(index=1)])
        with mock.patch.object(res, "WAIT_POLL_S", 0.01), contextlib.redirect_stderr(io.StringIO()):
            self.claim(any=True, wait=0.0)
        self.assertEqual(len(self.stored()), 1)

    def test_wait_gives_up_at_its_limit(self) -> None:
        self._probes([card(index=2, state="busy")])
        with (
            mock.patch.object(res, "WAIT_POLL_S", 0.05),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as ctx,
        ):
            self.claim("mckennie:2", wait=0.12)
        self.assertIn("gave up waiting", str(ctx.exception.code))
        self.assertEqual(self.stored(), [])

    def test_without_wait_a_taken_card_is_refused_at_once(self) -> None:
        calls = self._probes([card(index=2, state="busy")])
        with self.assertRaises(SystemExit):
            self.claim("mckennie:2")
        self.assertEqual(calls[0], 1)

    def test_wait_never_retries_a_card_that_does_not_exist(self) -> None:
        calls = self._probes([card(index=0)])
        with mock.patch.object(res, "WAIT_POLL_S", 0.01), self.assertRaises(SystemExit) as ctx:
            self.claim("nosuch:0", wait=0.0)
        self.assertIn("not found", str(ctx.exception.code))
        self.assertEqual(calls[0], 1)

    def test_any_packs_onto_hosts_already_in_use(self) -> None:
        self.cards = [card(host="solo", index=0), card(index=0), card(index=1, state="busy")]
        self.claim(any=True, count=1)
        self.assertEqual(self.stored(), ["mckennie:0"])

    def test_release_accepts_host_job_gpu(self) -> None:
        self.cards = [card(host="gpub1", index=0, job="7")]
        self.claim("gpub1:7:0")
        res.cmd_release(argparse.Namespace(targets=["gpub1:7:0"], holder="s", force=False), self.pools)
        self.assertEqual(self.stored(), [])

    def test_bad_lease_windows_are_rejected(self) -> None:
        saved = res.load_config
        try:
            for bad in ("soon", -5, 0, True, float("nan")):
                res.load_config = lambda bad=bad: {"leases": {"grace_min": bad}}
                with self.assertRaises(SystemExit):
                    res.lease_windows()
        finally:
            res.load_config = saved

    def leases(self) -> list[ls.Lease]:
        with ls.locked_store() as leases:
            return list(leases)

    def transfer(self, *targets: str, **kw: object) -> None:
        args = {"targets": list(targets), "holder": "s", "to": "t", "note": "", "run": "", "force": False}
        args.update(kw)
        res.cmd_transfer(argparse.Namespace(**args), self.pools)

    def test_a_busy_card_needs_adopt_and_says_so(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.claim("mckennie:2")
        self.assertIn("--adopt", str(ctx.exception.code))
        self.assertEqual(self.stored(), [])

    def test_adopt_takes_a_busy_card_and_its_run_renews_it(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.claim("mckennie:2", adopt=True, run="ni2cb9af", holder="b")
        [lease] = self.leases()
        self.assertIn("adopted", out.getvalue())
        self.assertEqual((lease.holder, lease.run), ("b", "ni2cb9af"))
        self.assertGreater(lease.last_active, 0, "an adopted lease starts active")
        idle = card(index=2)  # the run ended: the card is free again
        later = lease.last_active + 31 * MIN
        ls.reconcile([lease], [report(idle)], now=later)
        released = ls.reconcile([lease], [report(idle)], now=later + 31 * MIN)
        self.assertEqual([x.id for x, _ in released], [lease.id])

    def test_adopt_refuses_any_leased_and_unknown_cards(self) -> None:
        with self.assertRaises(SystemExit):
            self.claim(any=True, adopt=True)
        self.claim("mckennie:0")
        with self.assertRaises(SystemExit) as ctx:
            self.claim("mckennie:0", adopt=True, holder="b")
        self.assertIn("just res transfer", str(ctx.exception.code))
        self.cards = [card(index=3, state="unknown")]
        with self.assertRaises(SystemExit):
            self.claim("mckennie:3", adopt=True)
        self.assertEqual(self.stored(), ["mckennie:0"])

    def test_transfer_moves_the_lease_and_records_the_previous_holder(self) -> None:
        self.claim("mckennie:0", note="seed 1")
        with contextlib.redirect_stdout(io.StringIO()):
            self.transfer("mckennie:0", run="u1ae5im1")
        [lease] = self.leases()
        self.assertEqual((lease.holder, lease.previous, lease.note, lease.run), ("t", "s", "seed 1", "u1ae5im1"))

    def test_transfer_needs_the_holder_or_force_and_an_existing_lease(self) -> None:
        self.claim("mckennie:0")
        with self.assertRaises(SystemExit):
            self.transfer("mckennie:0", holder="other")
        with self.assertRaises(SystemExit):
            self.transfer("mckennie:1")
        self.assertEqual([x.holder for x in self.leases()], ["s"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.transfer("mckennie:0", holder="other", force=True)
        self.assertEqual([x.holder for x in self.leases()], ["t"])

    def test_release_needs_the_holder_or_force(self) -> None:
        self.claim("mckennie:0")
        with self.assertRaises(SystemExit):
            res.cmd_release(argparse.Namespace(targets=["mckennie:0"], holder="other", force=False), self.pools)
        self.assertEqual(self.stored(), ["mckennie:0"])
        res.cmd_release(argparse.Namespace(targets=["mckennie:0"], holder="other", force=True), self.pools)
        self.assertEqual(self.stored(), [])


if __name__ == "__main__":
    unittest.main()
