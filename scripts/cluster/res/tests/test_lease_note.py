"""A detached launch on a lease replaces its note, so `pls res status` names the run now on the card."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev


class LeaseNoteTest(unittest.TestCase):
    def test_the_note_is_the_launches_note_else_its_script_and_run_name(self) -> None:
        self.assertEqual(ev.launch_note("Kick seed 44", "/r/train.py", ["--run_name", "x"]), "Kick seed 44")
        self.assertEqual(ev.launch_note("", "/r/train.py", ["--task", "T", "--run_name", "gaze=0.05"]),
                         "train.py gaze=0.05")  # fmt: skip
        self.assertEqual(ev.launch_note("", "/r/train.py", ["--run_name=gaze=0.1"]), "train.py gaze=0.1")
        self.assertEqual(ev.launch_note("", "/r/train.py", ["--task", "T"]), "train.py")

    def test_a_relaunch_replaces_the_first_runs_note_and_adds_its_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ev.ls, "STORE", Path(tmp) / "leases.json"):
            with ev.ls.locked_store() as leases:
                leases.append(ev.ls.Lease("a1", "k", "c1:0", "horizon", "Latent", note="old probe", stage="/s/1",
                                          stages=["/s/1"]))  # fmt: skip
            ev._record_stage("a1", "/s/2", "train.py gaze=0.05")
            ev._record_stage("a1", "/s/2")  # a call with no note keeps the one set
            with ev.ls.locked_store() as leases:
                [x] = leases
        self.assertEqual((x.note, x.stage, x.stages), ("train.py gaze=0.05", "/s/2", ["/s/1", "/s/2"]))


if __name__ == "__main__":
    unittest.main()
