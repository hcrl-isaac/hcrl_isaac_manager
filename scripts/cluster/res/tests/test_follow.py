"""`pls run --detach -- train ...` follow-up: start confirmation, the async recorder and the watch, on a local stage."""

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import follow
from inventory import Pool

URL = "wandb: View run at https://forge.coreweave.com/wandb/hcrl/kick/runs/ab12cd34"


def _stage(kind: str = "local") -> SimpleNamespace:
    t = SimpleNamespace(kind=kind, host="box", ssh="" if kind == "local" else "u@login", ssh_opts=[])
    return SimpleNamespace(t=t, dir=tempfile.mkdtemp())


class StartedTest(unittest.TestCase):
    def test_the_first_iteration_gives_the_run_and_its_final_iteration(self) -> None:
        stage = _stage()
        Path(stage.dir, "log").write_text(f"boot\n{URL}\n[default0]: Learning iteration 1200/6000\n")
        self.assertEqual(follow.started(stage, 60, 0), ("hcrl/kick/ab12cd34", 6000))

    def test_a_run_without_wandb_still_starts(self) -> None:
        stage = _stage()
        Path(stage.dir, "log").write_text("Learning iteration 0/2\n")
        self.assertEqual(follow.started(stage, 60, 0), ("", 2))

    def test_a_run_that_ended_before_training_never_started(self) -> None:
        stage = _stage()
        Path(stage.dir, "log").write_text(f"{URL}\nTraceback (most recent call last):\n")
        Path(stage.dir, "status").write_text("1")
        self.assertIsNone(follow.started(stage, 60, 0))

    def test_no_iteration_by_the_deadline_is_no_start(self) -> None:
        stage = _stage()
        Path(stage.dir, "log").write_text("booting Kit\n")
        ticks = iter(range(0, 1000, 30))
        self.assertIsNone(follow.started(stage, 60, 0, clock=lambda: next(ticks), sleep=lambda s: None))


class EvalModeTest(unittest.TestCase):
    def test_eval_wins_over_video_and_on_is_the_default(self) -> None:
        self.assertEqual(follow.eval_mode(["--task", "T"]), "on")
        self.assertEqual(follow.eval_mode(["--video", "async"]), "async")
        self.assertEqual(follow.eval_mode(["--video=off", "--eval", "async"]), "async")
        self.assertEqual(follow.eval_mode(["--eval=on", "--video", "async"]), "on")


class RecorderTest(unittest.TestCase):
    POOLS = [
        Pool("larg-a100", "ssh", {}),
        Pool("larg-a40", "ssh", {"recorder_root": "/var/local/u/isaac3", "render": True}),
        Pool("ebuntu", "ssh", {"recorder_root": "/home/e/isaac3"}),
    ]

    def test_it_follows_the_run_on_a_shared_card_of_a_recorder_pool(self) -> None:
        calls = []
        on = follow.arm_recorder("hcrl/kick/ab12cd34", "Kick", "", self.POOLS, lambda c: calls.append(c) or _rc(0))
        self.assertEqual((on, len(calls)), ("larg-a40", 1), "pools without recorder_root are skipped")
        cmd = calls[0]
        for want in (["--pool", "larg-a40"], ["--root", "/var/local/u/isaac3"], ["--only_run", "ab12cd34"]):
            self.assertIn(want, [cmd[i : i + 2] for i in range(len(cmd) - 1)])
        self.assertIn("--share", cmd)
        self.assertIn("--follow", cmd)
        self.assertEqual(cmd[cmd.index("--wandb_project") + 1], "hcrl/kick")
        self.assertNotIn("--wt", cmd)

    def test_a_pool_that_cannot_start_one_falls_through_to_the_next(self) -> None:
        codes = iter([1, 0])
        calls = []
        on = follow.arm_recorder("e/p/r", "S", "lift", self.POOLS, lambda c: calls.append(c) or _rc(next(codes)))
        self.assertEqual(on, "ebuntu")
        self.assertEqual(calls[1][calls[1].index("--wt") + 1], "lift", "the recorder renders the run's own code")

    def test_no_pool_starts_one(self) -> None:
        self.assertEqual(follow.arm_recorder("e/p/r", "S", "", self.POOLS, lambda c: _rc(1)), "")


class WatchTest(unittest.TestCase):
    def test_a_slurm_watch_reads_over_the_login_master(self) -> None:
        stage = _stage("slurm")
        stage.t.ssh_opts = ["-o", "ControlMaster=no", "-o", "ControlPath=/h/.ssh/cm/%C"]
        seen = {}

        def launch(cmd: list[str], env: dict) -> subprocess.CompletedProcess:
            seen.update(cmd=cmd, cp=env.get("WATCH_RUN_CONTROL_PATH"))
            return _rc(0)

        self.assertEqual(follow.watch(stage, "e/p/r", 600, "lbl", launch), 0)
        self.assertEqual(seen["cp"], "/h/.ssh/cm/%C")
        self.assertEqual(seen["cmd"][2:], ["--wandb", "e/p/r", "lbl", f"u@login:{stage.dir}/log", "600"])

    def test_watch_run_exits_with_its_verdict(self) -> None:
        stage = _stage()
        Path(stage.dir, "log").write_text("Learning iteration 1/2\nLearning iteration 2/2\n")
        Path(stage.dir, "status").write_text("0")
        out = subprocess.run(["bash", str(follow.WATCH), "t", f"{stage.dir}/log", "2", "1", "0"], capture_output=True)
        self.assertEqual(out.returncode, 0, out.stdout)


class AfterLaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = _stage()
        self.args = argparse.Namespace(holder="Kick", wt="", no_recorder=False)
        self.armed: list = []
        self.watched: list = []
        patches = [
            mock.patch.object(follow, "started", lambda st: ("hcrl/kick/ab12cd34", 6000)),
            mock.patch.object(follow, "arm_recorder", lambda *a: self.armed.append(a) or "larg-a40"),
            mock.patch.object(follow, "watch", lambda *a: self.watched.append(a) or 3),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_an_async_run_gets_a_recorder_and_its_watch_verdict_is_the_exit(self) -> None:
        rc = follow.after_launch(self.stage, self.args, ["--video", "async", "--run_name", "kp=40"], [])
        self.assertEqual((rc, len(self.armed)), (3, 1))
        self.assertEqual(self.watched[0][1:], ("hcrl/kick/ab12cd34", 6000, "kp=40"))

    def test_an_in_process_video_run_and_an_opt_out_get_none(self) -> None:
        follow.after_launch(self.stage, self.args, ["--video", "on"], [])
        self.args.no_recorder = True
        follow.after_launch(self.stage, self.args, ["--video", "async"], [])
        self.assertEqual((self.armed, len(self.watched)), ([], 2))

    def test_a_run_that_never_trains_fails_without_a_watch(self) -> None:
        with mock.patch.object(follow, "started", lambda st: None):
            self.assertEqual(follow.after_launch(self.stage, self.args, ["--video", "async"], []), 1)
        self.assertEqual((self.armed, self.watched), ([], []))


def _rc(code: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code)


if __name__ == "__main__":
    unittest.main()
