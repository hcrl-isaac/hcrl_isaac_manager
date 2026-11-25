"""Auto-resume of a detached SLURM run its hold's end cut short."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev
import follow
import resume
from inventory import Pool

POOL = Pool("horizon", "slurm", {"login": "u@login"})


def _stage(job: str = "100", gpu: int = 2) -> ev.Stage:
    t = ev.Target("horizon", "slurm", "c101-001", gpu, "u@login", "/w", "/w/artifacts", job=job, uuid="GPU-a")
    st = ev.Stage(t)
    st.dir = "/w/artifacts/res-eval/abc"
    return st


NEW_LOG = "https://wandb.ai/e/p/runs/xyz Learning iteration 1460/2000"
ARGV = ["--wt", "kick", "--on", "horizon", "hcrl_isaaclab:scripts/train.py", "--", "--task", "T", "--seed", "3"]


class VerdictTest(unittest.TestCase):
    def test_only_a_hold_that_ended_under_an_unfinished_run_is_a_cut(self) -> None:
        argv = json.dumps(ARGV)
        cases = [
            ({"status": None, "stop": None, "pls_argv.json": argv}, "TIMEOUT", "cut"),
            ({"status": None, "stop": None, "pls_argv.json": argv}, "NODE_FAIL", "cut"),
            ({"status": None, "stop": None, "pls_argv.json": argv}, "PREEMPTED", "cut"),
            ({"status": None, "stop": "1", "pls_argv.json": argv}, "TIMEOUT", "stopped on purpose"),
            ({"status": "0\n", "stop": None, "pls_argv.json": argv}, "TIMEOUT", "finished"),
            ({"status": "1", "stop": None, "pls_argv.json": argv}, "RUNNING", "not cut (hold RUNNING)"),
            ({"status": None, "stop": None, "pls_argv.json": argv}, "CANCELLED", "not cut (hold CANCELLED)"),
            ({"status": None, "stop": None, "pls_argv.json": None}, "TIMEOUT", "launched before --move"),
            (
                {"status": None, "stop": None, "pls_argv.json": json.dumps([*ARGV[:4], "--no-resume", *ARGV[4:]])},
                "TIMEOUT",
                "launched with --no-resume",
            ),
        ]
        for files, state, want in cases:
            with self.subTest(state=state, want=want):
                self.assertTrue(resume.verdict(files, state).startswith(want))


class StateTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        for name, path in (("EVENTS", d / "events.jsonl"), ("REPLACEMENTS", d / "replacements.json")):
            p = mock.patch.object(resume, name, path)
            p.start()
            self.addCleanup(p.stop)
        self.d = d

    def events(self) -> list[dict]:
        return [json.loads(x) for x in (self.d / "events.jsonl").read_text().splitlines()]


class ReplacementTest(StateTest):
    def test_the_first_cut_run_queues_one_hold_and_the_rest_share_it(self) -> None:
        calls = []

        def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
            if "start" in cmd:  # the hold's sbatch; cluster_dev_cmd's own workspace lookup runs too
                calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "[cluster_dev] Submitted sentinel job 105901.\n")

        rec = {"state": "TIMEOUT", "partition": "gpu", "timelimit": "2-00:00:00", "nodes": "c101-001"}
        with mock.patch.object(resume.subprocess, "run", run):
            first = resume.replacement(_stage().t, rec, "Kick")
            again = resume.replacement(_stage(gpu=3).t, rec, "Kick")
        self.assertEqual((first, again, len(calls)), ("105901", "105901", 1))
        self.assertIn("--partition", calls[0])
        self.assertEqual(calls[0][calls[0].index("--time") + 1], "2-00:00:00")
        self.assertEqual(self.events()[0]["kind"], "hold_queued")

    def test_a_long_queue_is_noted_once(self) -> None:
        answers = iter(["PENDING|(None)|Resources"] * 3 + ["RUNNING|c102-007|None"])
        clock = iter([0, 1, 7 * 3600, 8 * 3600, 9 * 3600, 10 * 3600])
        with (
            mock.patch.object(resume, "_login", lambda t, cmd, timeout=60: next(answers) if "squeue" in cmd else ""),
            mock.patch.object(resume.time, "monotonic", lambda: next(clock)),
            mock.patch("sys.stderr"),
        ):
            node = resume.wait_running(_stage().t, "105901", "Kick", sleep=lambda s: None)
        self.assertEqual(node, "c102-007")
        self.assertEqual([e["kind"] for e in self.events()], ["hold_pending"])


class AfterCutTest(StateTest):
    def run_cut(self, resumes: str | None = None, confirm: str = "ok") -> tuple[ev.Stage | None, list[list[str]]]:
        files = {"status": None, "stop": None, "pls_argv.json": json.dumps(ARGV), "resumes": resumes,
                 "log": "https://wandb.ai/e/p/runs/abc Learning iteration 1450/2000"}  # fmt: skip
        launched = []

        def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "[res] started detached on c102-007:gpu2 (stage /w/new)\n")

        written = []
        patches = [
            mock.patch.object(
                resume, "hold_end", lambda t: {"state": "TIMEOUT", "partition": "gpu", "timelimit": "2:00:00"}
            ),
            mock.patch.object(
                resume.move, "read_stage", lambda st, rel: NEW_LOG if st.dir == "/w/new" else files.get(rel)
            ),
            mock.patch.object(resume.move, "wandb_checkpoints", lambda url: {1000: 1, 1400: 1}),
            mock.patch.object(resume, "replacement", lambda t, rec, holder: "105901"),
            mock.patch.object(resume, "wait_running", lambda t, job, holder: "c102-007"),
            mock.patch.object(resume.move, "pin_code", lambda st, tag: ("kick", [])),
            mock.patch.object(resume.move, "confirm", lambda st, past, deadline: confirm),
            mock.patch.object(resume, "_card", lambda stage_dir, holder: (POOL, "c102-007", 2, "105901", "GPU-z")),
            mock.patch.object(resume.ev, "make_target", lambda *a: _stage(job="105901").t),
            mock.patch.object(resume, "_write", lambda st, rel, text: written.append((rel, text))),
            mock.patch.object(resume.subprocess, "run", run),
            mock.patch("sys.stderr"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        new = resume.after_cut(_stage(), "Kick", "lease/abc")
        self.written = written
        return new, launched

    def test_a_cut_run_relaunches_on_the_same_card_of_the_new_hold_from_its_newest_checkpoint(self) -> None:
        new, [cmd] = self.run_cut()
        self.assertEqual(new.dir, "/w/new")
        args = cmd[cmd.index("--") + 1 :]
        head = cmd[: cmd.index("--")]
        self.assertEqual(head[head.index("--on") + 1], "c102-007:105901:2", "same card index on the new node")
        self.assertEqual(head[head.index("--wt") + 1], "kick", "the run's own code, pinned")
        self.assertIn("--detach", head)
        self.assertEqual(args[args.index("--checkpoint") + 1], "model_1400.pt")
        self.assertEqual(args[args.index("--max_iterations") + 1], "600", "the budget the run had left")
        self.assertEqual(args[args.index("--seed") + 1], "3", "its own arguments kept")
        self.assertEqual(self.written, [("resumes", "1")])
        [ev_row] = self.events()
        self.assertEqual((ev_row["kind"], ev_row["new_job"], ev_row["ckpt"]), ("resumed", "105901", 1400))

    def test_a_chain_resumed_three_times_is_left_cut(self) -> None:
        new, launched = self.run_cut(resumes="3")
        self.assertIsNone(new)
        self.assertEqual(launched, [])
        self.assertEqual(self.events()[0]["kind"], "resume_refused")

    def test_a_relaunch_that_never_trains_is_reported_not_watched(self) -> None:
        new, _ = self.run_cut(confirm="timeout")
        self.assertIsNone(new)
        self.assertEqual(self.events()[0]["kind"], "resume_unconfirmed")


class WatchTest(unittest.TestCase):
    def test_a_cut_slurm_run_is_watched_on_after_its_resume(self) -> None:
        old, new = _stage(), _stage(job="105901")
        new.dir = "/w/new"
        codes = iter([1, 0])
        watched = []

        def once(stage: ev.Stage, run: str, target: int, label: str, launch: object) -> int:
            watched.append(stage.dir)
            return next(codes)

        with (
            mock.patch.object(follow, "_watch_once", once),
            mock.patch.object(resume, "after_cut", lambda st, holder, label: new),
            mock.patch.object(follow, "started", lambda st: ("e/p/xyz", 2000)),
        ):
            rc = follow.watch(old, "e/p/abc", 2000, "lbl", holder="Kick")
        self.assertEqual((rc, watched), (0, [old.dir, "/w/new"]))

    def test_without_a_holder_or_off_slurm_a_dead_run_is_just_reported(self) -> None:
        with mock.patch.object(follow, "_watch_once", lambda *a: 1), mock.patch.object(resume, "after_cut") as cut:
            self.assertEqual(follow.watch(_stage(), "r", 10, "l"), 1)
            cut.assert_not_called()


class EventsTest(StateTest):
    def test_events_print_one_json_line_each(self) -> None:
        resume.event(kind="resumed", label="x")
        resume.event(kind="hold_queued", new_job="1")
        with mock.patch("builtins.print") as out:
            resume.print_events()
        self.assertEqual([json.loads(c.args[0])["kind"] for c in out.call_args_list], ["resumed", "hold_queued"])


if __name__ == "__main__":
    unittest.main()
