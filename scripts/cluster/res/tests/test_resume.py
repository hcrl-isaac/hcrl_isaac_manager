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
ARGV = [
    "--holder",
    "Kick",
    "--wt",
    "kick",
    "--on",
    "horizon",
    "hcrl_isaaclab:scripts/train.py",
    "--",
    "--task",
    "T",
    "--seed",
    "3",
]


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

    def test_the_job_id_is_cluster_devs_own_line_not_an_earlier_number(self) -> None:
        banner = "Welcome to Horizon. Allocation balance: 1234567 SU, 20261125.\n"

        def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(cmd, 0, banner, "[cluster_dev] Submitted sentinel job 105901 (x).\n")

        with mock.patch.object(resume.subprocess, "run", run), mock.patch("sys.stderr"):
            self.assertEqual(resume.replacement(_stage().t, {"state": "TIMEOUT"}, "Kick"), "105901")

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
    def run_cut(
        self, resumes: str | None = None, confirm: str = "ok", wait: object = None
    ) -> tuple[ev.Stage | None, list[list[str]]]:
        files = {"status": None, "stop": None, "pls_argv.json": json.dumps(ARGV), "resumes": resumes,
                 "log": "https://wandb.ai/e/p/runs/abc Learning iteration 1450/2000"}  # fmt: skip
        self.files = files
        launched, self.armed = [], []

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
            mock.patch.object(resume, "wait_running", wait or (lambda t, job, holder: "c102-007")),
            mock.patch.object(resume.move, "pin_code", lambda st, tag: ("kick", [])),
            mock.patch.object(resume.move, "confirm", lambda st, past, deadline: confirm),
            mock.patch.object(resume, "_card", lambda stage_dir, holder: (POOL, "c102-007", 2, "105901", "GPU-z")),
            mock.patch.object(resume.ev, "make_target", lambda *a: _stage(job="105901").t),
            mock.patch.object(resume, "_write", lambda st, rel, text: written.append((rel, text))),
            mock.patch.object(resume.move, "arm_moved_recorder", lambda st, *a: self.armed.append(st.dir)),
            mock.patch.object(resume.res, "load_pools", lambda: [POOL]),
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
        self.assertEqual(head.count("--holder"), 1, "the recorded holder is replaced, not repeated")
        self.assertEqual(args[args.index("--checkpoint") + 1], "model_1400.pt")
        self.assertEqual(args[args.index("--max_iterations") + 1], "600", "the budget the run had left")
        self.assertEqual(args[args.index("--seed") + 1], "3", "its own arguments kept")
        self.assertEqual(self.written, [("resumes", "1")])
        self.assertEqual(self.armed, ["/w/new"], "the resumed run's new W&B run gets its recorder")
        [ev_row] = self.events()
        self.assertEqual((ev_row["kind"], ev_row["new_job"], ev_row["ckpt"]), ("resumed", "105901", 1400))

    def test_a_decided_stage_leaves_the_pending_table(self) -> None:
        with resume._locked() as table:  # a released lease's run: no watch has claimed it
            table["pending"][_stage().dir] = {"holder": "Kick", "pool": "horizon", "host": "c102-007", "gpu": 2,
                                              "job": "100", "uuid": ""}  # fmt: skip
        new, launched = self.run_cut(resumes="3")
        self.assertEqual((new, launched), (None, []))
        with resume._locked() as table:
            self.assertEqual(table["pending"], {}, "a later watch does not decide it again")

    def test_a_released_leases_run_is_resumed_by_one_watch_and_the_next_stands_down(self) -> None:
        with resume._locked() as table:  # keep_cut_runs' row: no pid, so the first watch to decide it claims it
            table["pending"][_stage().dir] = {"holder": "Kick", "pool": "horizon", "host": "c102-007", "gpu": 2,
                                              "job": "100", "uuid": ""}  # fmt: skip
        first, launched = self.run_cut()
        self.assertEqual((first.dir, len(launched)), ("/w/new", 1))
        self.assertIsNone(resume.after_cut(_stage(), "Kick", "lease/abc"), "a second watch on the same run")
        self.assertEqual(len(launched), 1, "relaunched once")
        with resume._locked() as table:
            self.assertEqual((table["pending"], table["resumed"]), ({}, {_stage().dir: "/w/new"}))

    def test_a_chain_resumed_three_times_is_left_cut(self) -> None:
        new, launched = self.run_cut(resumes="3")
        self.assertIsNone(new)
        self.assertEqual(launched, [])
        self.assertEqual(self.events()[0]["kind"], "resume_refused")

    def test_a_replacement_that_ends_unused_is_forgotten_and_reported(self) -> None:
        with resume._locked() as table:
            table["holds"]["100"] = "105901"

        def ended(t: ev.Target, job: str, holder: str) -> str:
            raise SystemExit("[resume] replacement hold 105901 ended before it ran")

        new, launched = self.run_cut(wait=ended)
        self.assertEqual((new, launched), (None, []))
        self.assertEqual(self.events()[0]["kind"], "resume_failed")
        with resume._locked() as table:
            self.assertEqual(table, {"holds": {}, "pending": {}, "resumed": {}}, "a re-armed watch queues a fresh hold")

    def test_a_run_stopped_while_its_hold_queued_is_not_relaunched(self) -> None:
        def stopped_meanwhile(t: ev.Target, job: str, holder: str) -> str:
            self.files["stop"] = ""  # `pls run --stop` touches an empty stop file
            return "c102-007"

        new, launched = self.run_cut(wait=stopped_meanwhile)
        self.assertEqual((new, launched), (None, []))
        [row] = self.events()
        self.assertEqual((row["kind"], row["new_job"]), ("resume_refused", "105901"))

    def test_a_resume_is_pending_while_it_runs_and_cleared_after(self) -> None:
        seen = []

        def wait(t: ev.Target, job: str, holder: str) -> str:
            with resume._locked() as table:
                rows = dict(table["pending"])
            seen.append((rows, resume.pending_stages("Kick", [POOL])))
            return "c102-007"

        self.run_cut(wait=wait)
        [(rows, [(label, st)])] = seen
        row = rows["/w/artifacts/res-eval/abc"]
        self.assertEqual((label, st.dir), ("resume:100", "/w/artifacts/res-eval/abc"))
        self.assertEqual((row["holder"], row["job"], row["gpu"], row["host"]), ("Kick", "100", 2, "c101-001"))
        self.assertEqual(resume.pending_stages("Kick", [POOL]), [], "cleared once the resume ended")
        self.assertEqual(resume.pending_stages("Latent", [POOL]), [])

    def test_a_relaunch_that_never_trains_is_reported_not_watched(self) -> None:
        new, _ = self.run_cut(confirm="timeout")
        self.assertIsNone(new)
        self.assertEqual(self.armed, [])
        self.assertEqual(self.events()[0]["kind"], "resume_unconfirmed")


class ClaimTest(StateTest):
    def test_a_second_watch_stands_down_while_the_first_resumes_the_run(self) -> None:
        with resume._locked() as table:
            table["pending"]["/w/artifacts/res-eval/abc"] = {"holder": "Kick", "pid": 1}  # pid 1 is always alive
        self.assertIn("another watch (pid 1)", resume.claim(_stage(), "Kick"))

    def test_a_dead_watchs_claim_passes_on_and_a_resumed_run_is_never_relaunched(self) -> None:
        with resume._locked() as table:
            table["pending"]["/w/artifacts/res-eval/abc"] = {"holder": "Kick", "pid": 2**22 + 7}  # no such process
        self.assertEqual(resume.claim(_stage(), "Kick"), "", "the claim of a watch that died is taken over")
        resume._mark_resumed("/w/artifacts/res-eval/abc", "/w/new")
        self.assertEqual(resume.claim(_stage(), "Kick"), "already resumed as /w/new")

    def test_after_cut_does_not_relaunch_a_run_another_watch_is_resuming(self) -> None:
        with resume._locked() as table:
            table["pending"]["/w/artifacts/res-eval/abc"] = {"holder": "Kick", "pid": 1}
        files = {"status": None, "stop": None, "pls_argv.json": json.dumps(ARGV), "resumes": None,
                 "log": "https://wandb.ai/e/p/runs/abc Learning iteration 1450/2000"}  # fmt: skip
        with (
            mock.patch.object(resume, "hold_end", lambda t: {"state": "TIMEOUT", "partition": "gpu", "timelimit": "1"}),
            mock.patch.object(resume.move, "read_stage", lambda st, rel: files.get(rel)),
            mock.patch.object(resume.move, "wandb_checkpoints", lambda url: {1400: 1}),
            mock.patch.object(resume, "replacement", side_effect=AssertionError("queued a second hold")),
            mock.patch("sys.stderr"),
        ):
            self.assertIsNone(resume.after_cut(_stage(), "Kick", "lease/abc"))
        with resume._locked() as table:
            self.assertEqual(table["pending"]["/w/artifacts/res-eval/abc"]["pid"], 1, "its claim stays the other's")


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

    def test_a_rewatched_run_already_cut_is_resumed(self) -> None:
        new = _stage(job="105901")
        new.dir = "/w/new"
        got = iter([None, ("e/p/xyz", 2000)])
        with (
            mock.patch.object(follow, "_watch_once", lambda *a: 0),
            mock.patch.object(resume, "after_cut", lambda st, holder, label: new),
            mock.patch.object(follow, "started", lambda st: next(got)),
            mock.patch.object(follow, "REWATCH_STAGGER_S", 0),
            mock.patch("sys.stderr"),
        ):
            self.assertEqual(follow.rewatch([("resume:100", _stage())], holder="Kick"), 0)

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


class ReleasedLeaseTest(StateTest):
    """A lease `pls res status` releases at its hold's end hands its runs to the table `pls run --watch` reads."""

    def test_a_released_slurm_leases_runs_stay_pending_for_the_watch(self) -> None:
        p = mock.patch.object(resume.ls, "STORE", self.d / "leases.json")
        p.start()
        self.addCleanup(p.stop)
        # a lease as res.claim records one on a SLURM card
        cut = ev.ls.Lease("9136af", "GPU-0b4c", "c103-021:3 (job 107937)", "horizon (horizon)", "Latent", job="107937")
        cut.stages, cut.stage = ["/w/res-eval/old", "/w/res-eval/0e4c"], "/w/res-eval/0e4c"
        box = ev.ls.Lease("b2", "pepi:1", "pepi:1", "larg", "Latent")
        box.stage = "/w/box"
        ev.ls.keep_cut_runs([(cut, "job 107937 ended"), (box, "idle for 30 min")])
        target = lambda pool, host, gpu, job, uuid: ev.Target(pool.name, "slurm", host, gpu, job=job, uuid=uuid)  # noqa: E731
        with mock.patch.object(resume.ev, "make_target", target):
            found = resume.pending_stages("Latent", [POOL])
        self.assertEqual(
            sorted(st.dir for _, st in found), ["/w/res-eval/0e4c", "/w/res-eval/old"], "a box run is not one"
        )
        t = found[0][1].t
        self.assertEqual((t.pool, t.host, t.gpu, t.job, t.uuid), ("horizon", "c103-021", 3, "107937", "GPU-0b4c"))


class SlurmProfileTest(unittest.TestCase):
    """A lease on a hold that already ended still maps to its cluster profile, so `pls run --watch` can resume it."""

    def test_a_single_profile_cluster_needs_no_lookup(self) -> None:
        with mock.patch.object(ev.subprocess, "run", side_effect=AssertionError("looked up a job")):
            self.assertEqual(ev.slurm_profile("horizon (horizon)", "105795", [POOL]).name, "horizon")

    def test_an_ended_job_falls_back_instead_of_ending_the_watch(self) -> None:
        pools = [Pool("amd-rtx", "slurm", {"login": "u@s3"}), Pool("rtx-small", "slurm", {"login": "u@s3"})]
        gone = subprocess.CompletedProcess([], 0, "", "slurm_load_jobs error: Invalid job id specified")
        with mock.patch.object(ev.subprocess, "run", lambda *a, **k: gone), mock.patch("sys.stderr"):
            self.assertEqual(ev.slurm_profile("stampede3 (amd-rtx, rtx-small)", "1", pools).name, "amd-rtx")
        sacct = subprocess.CompletedProcess([], 0, "rtx-small\n", "")
        with mock.patch.object(ev.subprocess, "run", lambda *a, **k: sacct):
            self.assertEqual(ev.slurm_profile("stampede3 (amd-rtx, rtx-small)", "1", pools).name, "rtx-small")


if __name__ == "__main__":
    unittest.main()
