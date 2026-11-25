"""`pls run --stop --stage <id> --on <cluster>`: stop a detached run no lease records."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev
from inventory import Pool

POOLS = [Pool("horizon", "slurm", {"login": "u@login", "scratch": "/s"})]


class StopStageTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.argv = json.dumps(["--holder", "Latent", "--on", "c1:5:0", "train.py", "--", "--task", "T"])
        self.ended = 1  # `[ -f status ]` fails: the run is live
        self.killed: list[str] = []
        patches = [
            mock.patch.object(ev.ls, "STORE", Path(tmp.name) / "leases.json"),
            mock.patch.object(ev.follow, "_probe", lambda st, cmd: self.argv),
            mock.patch.object(ev.Stage, "sh", lambda st, cmd: self.ended),
            mock.patch.object(ev.Stage, "kill", lambda st: self.killed.append(st.dir) or True),
            mock.patch("sys.stderr"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_the_holders_own_unleased_run_is_stopped(self) -> None:
        self.assertEqual(ev.cmd_stop_stage("horizon", "e1a90305", "Latent", POOLS), 0)
        self.assertEqual(len(self.killed), 1)
        self.assertTrue(self.killed[0].endswith("/res-eval/e1a90305"))

    def test_another_holders_run_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ev.cmd_stop_stage("horizon", "e1a90305", "Kick", POOLS)
        self.assertEqual(self.killed, [])

    def test_a_run_a_lease_still_records_goes_through_its_lease(self) -> None:
        with ev.ls.locked_store() as leases:
            leases.append(ev.ls.Lease("a1", "k", "c1:0", "horizon", "Latent", stage="/s/res-eval/e1a90305"))
        with self.assertRaises(SystemExit) as cm:
            ev.cmd_stop_stage("horizon", "e1a90305", "Latent", POOLS)
        self.assertIn("lease:a1", str(cm.exception.code))
        self.assertEqual(self.killed, [])

    def test_an_ended_run_has_nothing_to_stop(self) -> None:
        self.ended = 0
        self.assertEqual(ev.cmd_stop_stage("horizon", "e1a90305", "Latent", POOLS), 0)
        self.assertEqual(self.killed, [])


if __name__ == "__main__":
    unittest.main()
