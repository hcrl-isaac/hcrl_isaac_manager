"""`pls run --jobs`: job files, the options every job shares, and the per-card workers."""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batch

# stands in for evaluate.py: logs its argv, fails when an argument contains "bad"
FAKE = """import json, sys
print(json.dumps(sys.argv[1:]))
sys.exit(1 if any("bad" in a for a in sys.argv) else 0)
"""


class JobFileTest(unittest.TestCase):
    def write(self, text: str) -> str:
        path = Path(tempfile.mkdtemp()) / "jobs.json"
        path.write_text(text)
        return str(path)

    def test_an_array_and_json_lines_read_the_same(self) -> None:
        entries = [{"id": "a", "args": ["--rate", 12.5], "checkpoint": {"CKPT": "e/p/r"}}, {"id": "b"}]
        for text in (json.dumps(entries), "\n".join(json.dumps(e) for e in entries)):
            jobs = batch.load_jobs(self.write(text))
            self.assertEqual([(j.id, j.args, j.checkpoint) for j in jobs],
                             [("a", ["--rate", "12.5"], {"CKPT": "e/p/r"}), ("b", [], {})])  # fmt: skip

    def test_bad_entries_are_refused(self) -> None:
        for entries in (
            [{"id": "a"}, {"id": "a"}],
            [{"id": "../x"}],
            [{"id": "a", "task": "t"}],
            [{"id": "a", "args": "x"}],
        ):
            with self.assertRaises(SystemExit):
                batch.load_jobs(self.write(json.dumps(entries)))


class ArgvTest(unittest.TestCase):
    def test_the_batch_options_are_not_passed_to_jobs(self) -> None:
        argv = ["--holder", "B", "--lease", "a,b", "--jobs", "j", "--out=o", "--wt", "legacy", "--cmd"]
        self.assertEqual(batch.shared_options(argv), ["--holder", "B", "--wt", "legacy", "--cmd"])

    def test_a_job_runs_on_its_lease_with_its_checkpoints_and_args_last(self) -> None:
        job = batch.Job("x", ["--measure", "kick"], {"CKPT": "e/p/r"}, {"A": "1"})
        argv = batch.job_argv(job, "L1", ["--holder", "B"], "s.py", ["--checkpoint", "{CKPT}"])
        self.assertEqual(argv[2:], ["--holder", "B", "--lease", "L1", "--checkpoint", "CKPT=e/p/r", "--env", "A=1",
                                    "s.py", "--", "--checkpoint", "{CKPT}", "--measure", "kick"])  # fmt: skip


class RunBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        fake = self.dir / "fake_evaluate.py"
        fake.write_text(FAKE)
        self.patch = mock.patch.object(batch, "EVALUATE", fake)
        self.patch.start()
        self.out = self.dir / "out"

    def tearDown(self) -> None:
        self.patch.stop()

    def run_jobs(self, ids: list[str], leases: list[str]) -> int:
        jobs = self.dir / "jobs.jsonl"
        jobs.write_text("\n".join(json.dumps({"id": i, "args": [f"--tag={i}"]}) for i in ids))
        with contextlib.redirect_stderr(io.StringIO()):
            return batch.run_batch(["--holder", "B"], leases, str(jobs), str(self.out), "s.py", [])

    def status(self, job: str) -> str:
        return (self.out / f"{job}.status").read_text().strip()

    def test_every_job_runs_once_with_a_log_and_status(self) -> None:
        self.assertEqual(self.run_jobs(["j1", "j2", "j3"], ["L1", "L2"]), 0)
        self.assertEqual([self.status(j) for j in ("j1", "j2", "j3")], ["ok"] * 3)
        logged = json.loads((self.out / "j2.log").read_text())
        self.assertIn("--tag=j2", logged)
        self.assertIn(logged[logged.index("--lease") + 1], ("L1", "L2"))

    def test_a_rerun_runs_only_what_is_not_ok(self) -> None:
        self.assertEqual(self.run_jobs(["j1", "bad1"], ["L1"]), 1)
        self.assertEqual((self.status("j1"), self.status("bad1")), ("ok", "failed rc=1"))
        (self.out / "j1.log").write_text("kept")
        self.run_jobs(["j1", "bad1"], ["L1"])
        self.assertEqual((self.out / "j1.log").read_text(), "kept", "an ok job is not run again")

    def test_a_sigint_to_the_batch_gives_a_running_job_exactly_one_signal(self) -> None:
        signals = self.dir / "signals"
        started = self.dir / "started"
        (self.dir / "fake_evaluate.py").write_text(f"""import pathlib, signal, sys, time
def note(signum, _frame):
    with open({str(signals)!r}, "a") as f:
        f.write(signal.Signals(signum).name + "\\n")
signal.signal(signal.SIGINT, note)
signal.signal(signal.SIGTERM, note)
pathlib.Path({str(started)!r}).touch()
end = time.time() + 20
while time.time() < end and not pathlib.Path({str(signals)!r}).exists():
    time.sleep(0.05)
time.sleep(1.0)  # a second signal would land here, during cleanup
sys.exit(1)
""")

        jobs = self.dir / "jobs.jsonl"
        jobs.write_text(json.dumps({"id": "j1"}))
        res_dir = str(Path(batch.__file__).parent)
        driver = (  # the batch as a terminal's foreground job: its own process group, the stand-in as its runner
            f"import sys; sys.path.insert(0, {res_dir!r}); import batch, pathlib; "
            f"batch.EVALUATE = pathlib.Path({str(self.dir / 'fake_evaluate.py')!r}); "
            f"sys.exit(batch.run_batch([], ['L1'], {str(jobs)!r}, {str(self.out)!r}, 's.py', []))"
        )
        proc = subprocess.Popen([sys.executable, "-c", driver], start_new_session=True, stderr=subprocess.DEVNULL)
        try:
            end = time.time() + 20
            while not started.exists() and time.time() < end:
                time.sleep(0.05)
            os.killpg(proc.pid, signal.SIGINT)  # what a terminal Ctrl-C does: the whole foreground group
            self.assertEqual(proc.wait(timeout=30), 1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        self.assertEqual(signals.read_text().split(), ["SIGTERM"])
        self.assertEqual(self.status("j1"), "stopped")

    def test_a_card_failing_three_jobs_in_a_row_stops_taking_jobs(self) -> None:
        self.assertEqual(self.run_jobs(["bad1", "bad2", "bad3", "j4"], ["L1"]), 1)
        self.assertFalse((self.out / "j4.status").exists(), "the broken card's worker stopped before j4")


if __name__ == "__main__":
    unittest.main()
