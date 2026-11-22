"""Unit tests for `pls run --move` (run all script tests: pls test-scripts)."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

RES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RES))
import move

LAUNCH = [
    "--holder", "s", "--on", "c112-022:3", "--wait", "--detach", "--stall", "45m", "--wt", "agile",
    "hcrl_isaaclab:scripts/train.py", "--", "--task", "Crab-Agile-v0", "--run_name", "slot-sgs",
    "--max_iterations", "3000",
]  # fmt: skip
LOG = "\x1b[1m wandb: View run at https://wandb.ai/ent/proj/runs/ab12cd34\n Learning iteration 1200/3000\n"


class HelperTest(unittest.TestCase):
    def test_split_launch_finds_the_script_before_the_dashes(self) -> None:
        opts, script, args = move.split_launch(LAUNCH)
        self.assertEqual(script, "hcrl_isaaclab:scripts/train.py")
        self.assertEqual(args[:2], ["--task", "Crab-Agile-v0"])
        self.assertEqual(opts[-2:], ["--wt", "agile"])

    def test_card_selection_wt_and_detach_go_the_rest_stays_in_order(self) -> None:
        opts, _, _ = move.split_launch(LAUNCH)
        self.assertEqual(move.drop_card_selection(opts), ["--holder", "s", "--stall", "45m"])
        self.assertEqual(move.drop_card_selection(["--any", "--pool", "a", "--wait", "4h", "--x"]), ["--x"])

    def test_previous_resume_arguments_are_dropped_in_both_forms(self) -> None:
        args = [
            "--task",
            "T",
            "--load_run",
            "u",
            "--checkpoint=model_5.pt",
            "--resume",
            "True",
            "--max_iterations",
            "9",
        ]
        self.assertEqual(move.drop_resume(args), ["--task", "T"])

    def test_progress_reads_the_last_url_and_iteration_through_color_codes(self) -> None:
        self.assertEqual(move.progress(LOG), ("https://wandb.ai/ent/proj/runs/ab12cd34", 1200, 3000))
        self.assertIsNone(move.progress("Learning iteration 3/10"), "no W&B run yet")
        forge = (
            "wandb: View run at https://forge.coreweave.com/wandb/ent/proj/runs/ug5vpkfc\n Learning iteration 159/600\n"
        )
        self.assertEqual(move.progress(forge), ("https://forge.coreweave.com/wandb/ent/proj/runs/ug5vpkfc", 159, 600))

    def test_a_checkpoint_counts_once_its_size_holds_over_a_poll(self) -> None:
        sizes = iter([{1000: 5}, {1000: 5, 1200: 10}, {1000: 5, 1200: 20}, {1000: 5, 1200: 20}])
        with (
            mock.patch.object(move, "wandb_checkpoints", lambda url: next(sizes)),
            mock.patch.object(move.time, "sleep"),
        ):
            self.assertEqual(move.wait_checkpoint("u", 1000, move.time.monotonic() + 60), 1200)

    def test_no_new_checkpoint_before_the_deadline_gives_none(self) -> None:
        with mock.patch.object(move, "wandb_checkpoints", lambda url: {1000: 5}), mock.patch.object(move.time, "sleep"):
            self.assertIsNone(move.wait_checkpoint("u", 1000, move.time.monotonic() - 1))


class PinTest(unittest.TestCase):
    def test_pin_checks_out_the_recorded_commit_not_the_current_head(self) -> None:
        import subprocess
        import tempfile

        root = Path(tempfile.mkdtemp())
        repo = root / "resources" / "ssti_tasks"
        repo.mkdir(parents=True)

        def git(*a: str) -> str:
            env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                   "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}  # fmt: skip
            return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                                  env=env).stdout.strip()  # fmt: skip

        git("init", "-q")
        (repo / "f").write_text("ran")
        git("add", "f")
        git("commit", "-qm", "ran")
        ran = git("rev-parse", "HEAD")
        (repo / "f").write_text("later")
        git("commit", "-qam", "later")
        refs = {"pls_refs": {"value": {"repos": {"ssti_tasks": {"commit": ran, "dirty": 0}}, "tree": "", "wt": ""}}}
        with (
            mock.patch.object(move, "read_stage", lambda st, rel: json.dumps(refs)),
            mock.patch.object(move.ev, "local_workspace", lambda: str(root)),
            mock.patch.object(move.ev, "local_code", lambda ws, wt: {"ssti_tasks": str(repo)}),
        ):
            wt, made = move.pin_code(None, "move-t")
        pinned = repo / "worktrees" / "move-t"
        self.assertEqual((wt, (pinned / "f").read_text()), ("move-t", "ran"), "the code the run ran, not HEAD")
        move.unpin(made)
        self.assertFalse(pinned.exists())

    def test_a_run_with_uncommitted_changes_cannot_be_pinned(self) -> None:
        refs = {"pls_refs": {"value": {"repos": {"ssti_tasks": {"commit": "abc", "dirty": 2}}, "tree": "", "wt": ""}}}
        with (
            mock.patch.object(move, "read_stage", lambda st, rel: json.dumps(refs)),
            mock.patch.object(move.ev, "local_workspace", lambda: "/nowhere"),
            mock.patch.object(move.ev, "local_code", lambda ws, wt: {"ssti_tasks": "/nowhere/resources/ssti_tasks"}),
            self.assertRaises(SystemExit) as ctx,
        ):
            move.pin_code(None, "move-t")
        self.assertIn("uncommitted", str(ctx.exception.code))


class MoveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lease = SimpleNamespace(id="old", stages=["/s/a1"], stage="/s/a1", cpus=0)
        self.warmup = 0
        self.files = {"pls_argv.json": json.dumps(LAUNCH), "log": LOG}
        patches = [
            mock.patch.object(move.ev, "_lease_card", lambda i, h, p: ("pool", "host", 0, "", "uuid")),
            mock.patch.object(move.ev, "make_target", lambda *a: SimpleNamespace(kind="ssh")),
            mock.patch.object(move.ev, "Stage", lambda t: SimpleNamespace(t=t, dir="")),
            mock.patch.object(move, "read_stage", lambda st, rel: self.files.get(rel)),
            mock.patch.object(move, "pin_code", lambda st, tag: ("move-x", [])),
            mock.patch.object(move, "wandb_checkpoints", lambda url: {1000: 5}),
            mock.patch.object(move, "wait_checkpoint", lambda url, after, dl: 1250),
            mock.patch.object(move, "resume_warmup", lambda url: self.warmup),
            mock.patch.object(move, "confirm", lambda st, past, dl: "ok"),
            mock.patch.object(move.ls, "is_cpu", lambda x: False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        store = mock.MagicMock()
        store.__enter__.return_value = [self.lease]
        lock = mock.patch.object(move.ls, "locked_store", lambda: store)
        lock.start()
        self.addCleanup(lock.stop)
        self.stops, self.launches = [], []
        stop = mock.patch.object(
            move.ev, "cmd_stop", lambda i, h, p, keep_lease=False: self.stops.append(keep_lease) or 0
        )
        go = mock.patch.object(
            move, "relaunch", lambda o, s, a, lid, wt: self.launches.append((o, s, a, lid, wt)) or (0, "/s/b2")
        )
        for p in (stop, go):
            p.start()
            self.addCleanup(p.stop)

    def test_a_same_card_move_keeps_the_lease_and_resumes_with_the_remaining_budget(self) -> None:
        self.assertEqual(move.cmd_move("old", "s", "same", 60, 60, []), 0)
        self.assertEqual(self.stops, [True], "the card stays leased between stop and relaunch")
        [(_, script, args, lease_id, wt)] = self.launches
        self.assertEqual((script, lease_id, wt), ("hcrl_isaaclab:scripts/train.py", "old", "move-x"))
        self.assertEqual(args[-6:], ["--load_run", "https://wandb.ai/ent/proj/runs/ab12cd34", "--checkpoint",
                                     "model_1250.pt", "--max_iterations", "1750"])  # fmt: skip
        self.assertEqual(args.count("--max_iterations"), 1, "the launch's own budget is replaced, not doubled")

    def test_a_move_elsewhere_claims_the_target_before_stopping_and_releases_the_old_card(self) -> None:
        order = []
        with mock.patch.object(move, "target_lease", lambda to, i, h, p: order.append("claim") or "new"):
            self.stops.clear()
            self.assertEqual(move.cmd_move("old", "s", "c107-026:1", 60, 60, []), 0)
        self.assertEqual((order, self.stops, self.launches[0][3]), (["claim"], [False], "new"))

    def test_an_off_policy_resume_keeps_every_update_it_had_left(self) -> None:
        self.warmup = 500
        self.assertEqual(move.cmd_move("old", "s", "same", 60, 60, []), 0)
        args = self.launches[0][2]
        self.assertEqual(args[args.index("--max_iterations") + 1], "1750", "3000 - 1250; the warm-up comes on top")

    def test_refusals_leave_the_run_alone(self) -> None:
        cases = {
            "predates --move": lambda: self.files.pop("pls_argv.json"),
            "train.py": lambda: self.files.update({"pls_argv.json": json.dumps(["--holder", "s", "probe.py", "--"])}),
            "not logged": lambda: self.files.update({"log": "booting"}),
        }
        for says, make in cases.items():
            with self.subTest(says=says):
                saved = dict(self.files)
                make()
                with self.assertRaises(SystemExit) as ctx:
                    move.cmd_move("old", "s", "same", 60, 60, [])
                self.assertIn(says, str(ctx.exception.code))
                self.assertEqual((self.stops, self.launches), ([], []))
                self.files.clear()
                self.files.update(saved)

    def test_code_that_cannot_be_rebuilt_is_refused_before_anything_stops(self) -> None:
        def dirty(st: object, tag: str) -> None:
            raise SystemExit("[move] the run shipped uncommitted changes in ssti_tasks, so its code cannot be rebuilt")

        with mock.patch.object(move, "pin_code", dirty), self.assertRaises(SystemExit) as ctx:
            move.cmd_move("old", "s", "same", 60, 60, [])
        self.assertIn("cannot be rebuilt", str(ctx.exception.code))
        self.assertEqual((self.stops, self.launches), ([], []))

    def test_the_pinned_worktrees_go_even_when_the_move_fails(self) -> None:
        removed = []
        with (
            mock.patch.object(move, "pin_code", lambda st, tag: ("move-x", [("main", "wt")])),
            mock.patch.object(move, "unpin", removed.extend),
            mock.patch.object(move, "wait_checkpoint", lambda url, after, dl: None),
            self.assertRaises(SystemExit),
        ):
            move.cmd_move("old", "s", "same", 60, 60, [])
        self.assertEqual(removed, [("main", "wt")])

    def test_no_checkpoint_in_time_stops_nothing(self) -> None:
        with mock.patch.object(move, "wait_checkpoint", lambda url, after, dl: None), self.assertRaises(SystemExit):
            move.cmd_move("old", "s", "same", 60, 60, [])
        self.assertEqual((self.stops, self.launches), ([], []))


if __name__ == "__main__":
    unittest.main()
