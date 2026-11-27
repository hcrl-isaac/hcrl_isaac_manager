"""`pls run --detach -- train ...` follow-up: start confirmation, the async recorder and the watch, on a local stage."""

import argparse
import os
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

    def test_a_runner_that_is_gone_without_a_status_never_started(self) -> None:
        stage = _stage()
        Path(stage.dir, "log").write_text("[res] no Isaac Lab root venv at /r/venv/bin/python on box\n")
        dead = subprocess.Popen(["true"])
        dead.wait()
        Path(stage.dir, "pid").write_text(str(dead.pid))
        self.assertIsNone(follow.started(stage, 60, 0), "a preflight refusal writes no status")

    def test_a_live_runner_still_booting_is_waited_on(self) -> None:
        stage = _stage()
        Path(stage.dir, "log").write_text("booting Kit\n")
        Path(stage.dir, "pid").write_text(str(os.getpid()))
        ticks = iter([0, 10, 20, 70])
        sleeps = []
        self.assertIsNone(follow.started(stage, 60, 0, clock=lambda: next(ticks), sleep=sleeps.append))
        self.assertEqual(len(sleeps), 2, "it polled until the deadline")

    def test_a_slurm_heartbeat_gone_stale_is_an_end(self) -> None:
        stage = _stage()
        stage.t.kind = "slurm"
        with mock.patch.object(follow, "_probe", lambda st, cmd: "__ENDED__\n" if "heartbeat" in cmd else ""):
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
        Pool("larg-a40", "ssh", {"user": "u", "recorder_root": "/var/local/u/isaac3", "render": True}),
        Pool("ebuntu", "ssh", {"user": "u", "recorder_root": "/home/e/isaac3", "render": True}),
        Pool("horizon-like", "ssh", {"recorder_root": "/r"}),  # a root, but no RTX render: never picked
    ]

    def arm(self, wt: str, codes: list[int], verdicts: list[str] | None = None, calls: list | None = None) -> tuple:
        rcs, said = iter(codes), iter(verdicts or ["ok"] * len(codes))
        calls = [] if calls is None else calls

        def launch(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
            calls.append(cmd)
            rc = next(rcs)
            err = "[res] started detached on pepi:gpu3 (stage /var/rec/ab)\n" if rc == 0 else "[res] no free card\n"
            return subprocess.CompletedProcess(cmd, rc, "", err)

        return follow.arm_recorder("e/p/r", "S", wt, self.POOLS, launch, alive=lambda st: next(said))

    def test_it_follows_the_run_on_a_shared_card_of_a_recorder_pool(self) -> None:
        calls = []
        on = self.arm("", [0], calls=calls)
        self.assertEqual((on, len(calls)), (("larg-a40", "/var/rec/ab"), 1), "pools without recorder_root are skipped")
        cmd = calls[0]
        for want in (["--pool", "larg-a40"], ["--root", "/var/local/u/isaac3"], ["--only_run", "r"]):
            self.assertIn(want, [cmd[i : i + 2] for i in range(len(cmd) - 1)])
        self.assertIn("--share-mine", cmd)
        self.assertIn("--follow", cmd)
        self.assertEqual(cmd[cmd.index("--wandb_project") + 1], "e/p")
        self.assertNotIn("--wt", cmd)

    def test_a_pool_that_cannot_start_one_falls_through_to_the_next(self) -> None:
        calls = []
        self.assertEqual(self.arm("lift", [1, 0], calls=calls), ("ebuntu", "/var/rec/ab"))
        self.assertEqual(calls[1][calls[1].index("--wt") + 1], "lift", "the recorder renders the run's own code")

    def test_a_recorder_that_dies_before_its_first_sweep_is_not_armed(self) -> None:
        on, why = self.arm("", [0, 0], ["it exited before its first sweep:\nTraceback", "ok"])
        self.assertEqual(on, "ebuntu", "the next render pool is tried")
        on, why = self.arm("", [0, 1], ["it exited before its first sweep:\nCUDA numbers pepi:gpu3 cuda:0"])
        self.assertEqual(on, "")
        self.assertIn("pepi:gpu3", why)
        self.assertIn("died", why)

    def test_only_render_pools_are_tried(self) -> None:
        calls = []
        self.arm("", [1, 1], calls=calls)
        self.assertEqual([c[c.index("--pool") + 1] for c in calls], ["larg-a40", "ebuntu"])

    def test_no_pool_starts_one(self) -> None:
        self.assertEqual(self.arm("", [1, 1])[0], "")

    def test_a_recorder_is_alive_once_it_sweeps_and_dead_once_it_exits_first(self) -> None:
        stage = _stage()
        log = Path(stage.dir, "log")
        log.write_text("[recorder] Sweeping 1 tagged run(s): ['r']\n")
        self.assertEqual(follow.recorder_alive(stage, 5, 0.01), "ok")
        log.write_text("Traceback (most recent call last):\nRuntimeError: boom\n")
        Path(stage.dir, "status").write_text("1")
        self.assertIn("RuntimeError: boom", follow.recorder_alive(stage, 5, 0.01))

    def test_arming_notes_the_recorder_in_the_runs_stage(self) -> None:
        stage = _stage()
        Path(stage.dir, follow.RECORDER_FILE).write_text("pending\n")
        self.assertTrue(follow.wants_recorder(stage))
        with mock.patch.object(follow, "arm_recorder", lambda *a: ("larg-a40", "/var/rec/ab")):
            self.assertTrue(follow.arm_for(stage, "e/p/r", "S", "", []))
        self.assertEqual(Path(stage.dir, follow.RECORDER_FILE).read_text().strip(), "larg-a40 /var/rec/ab")
        self.assertFalse(follow.wants_recorder(stage))
        with mock.patch.object(follow, "arm_recorder", lambda *a: ("", "no render pool")):
            self.assertFalse(follow.arm_for(stage, "e/p/r", "S", "", []))


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
        self.assertEqual(
            seen["cmd"][2:],
            ["--wandb", "e/p/r", "--status", f"{stage.dir}/status", "lbl", f"u@login:{stage.dir}/log", "600"],
        )

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
            mock.patch.object(follow, "arm_recorder", lambda *a: self.armed.append(a) or ("larg-a40", "/rec")),
            mock.patch.object(follow, "watch", lambda *a, **kw: self.watched.append(a) or 3),
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


class RewatchTest(unittest.TestCase):
    def test_a_holders_detached_runs_once_each_oldest_first(self) -> None:
        def lease(i: str, holder: str, created: float, stages: list[str]) -> SimpleNamespace:
            return SimpleNamespace(
                id=i, holder=holder, created=created, stages=stages, stage=stages[-1] if stages else ""
            )

        leases = [
            lease("b", "me", 2, ["/s/run2"]),
            lease("a", "me", 1, ["/s/run1"]),
            lease("a2", "me", 1.5, ["/s/run1"]),  # another card of the same --gpus run
            lease("c", "other", 0, ["/s/theirs"]),
            lease("d", "me", 3, []),
        ]
        self.assertEqual(follow.detached_stages(leases, "me", []), [("a", "/s/run1"), ("b", "/s/run2")])
        self.assertEqual(follow.detached_stages(leases, "me", ["b"]), [("b", "/s/run2")])

    def test_every_live_run_is_watched_from_one_call(self) -> None:
        live, ended = _stage(), _stage()
        got = {live.dir: ("e/p/run1", 4000), ended.dir: None}
        launched = []
        with (
            mock.patch.object(follow, "REWATCH_STAGGER_S", 0),
            mock.patch.object(follow, "started", lambda st, deadline_s=0: got[st.dir]),
            mock.patch("sys.stderr"),
        ):
            rc = follow.rewatch([("L1", live), ("L2", ended)], launch=lambda cmd, env: launched.append(cmd) or _rc(0))
        self.assertEqual(len(launched), 1, "an ended run is reported, not watched")
        self.assertIn("e/p/run1", launched[0])
        self.assertEqual(rc, 1, "the ended run makes the call non-zero")

    def test_a_run_whose_launch_armed_no_recorder_gets_one_from_the_watch(self) -> None:
        pending, armed_already = _stage(), _stage()
        Path(pending.dir, follow.RECORDER_FILE).write_text("pending\n")
        Path(armed_already.dir, follow.RECORDER_FILE).write_text("larg-a40 /var/rec/x\n")
        armed = []
        with (
            mock.patch.object(follow, "REWATCH_STAGGER_S", 0),
            mock.patch.object(follow, "started", lambda st, deadline_s=0: ("e/p/" + Path(st.dir).name, 10)),
            mock.patch("sys.stderr"),
        ):
            follow.rewatch(
                [("L1", pending), ("L2", armed_already)],
                launch=lambda cmd, env: _rc(0),
                arm=lambda st, run: armed.append(run),
            )
        self.assertEqual(armed, ["e/p/" + Path(pending.dir).name], "only the run still waiting on its recorder")


if __name__ == "__main__":
    unittest.main()
