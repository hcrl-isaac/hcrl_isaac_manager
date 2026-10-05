"""`just res eval` with stubs and no GPU: argument checks, pool refusal, checkpoints, lease release and exit status."""

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

COMPUTE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPUTE))
import checkpoints as ck
import evaluate as ev
import leases as ls
import res
from inventory import Pool
from probe import Card, Report

SCRIPT = """import os, sys
print("ARGV", sys.argv[1:])
for k in ("CKPT_A", "CHECKPOINT", "MODE", "CUDA_VISIBLE_DEVICES"):
    print(k, os.environ.get(k, "-"))
print("TMPDIR", os.environ["TMPDIR"])
print("OMNI", os.environ["OMNI_CACHE_DIR"])
print("PP", os.environ["PYTHONPATH"])
if os.environ.get("CKPT_A"):
    print("CKPT_A_CONTENT", open(os.environ["CKPT_A"]).read())
mode = os.environ.get("MODE", "")
if mode == "fail":
    sys.exit(3)
if mode == "traceback":
    print("Traceback (most recent call last):")
"""


class CheckpointRefTest(unittest.TestCase):
    def test_wandb_url_with_iteration(self) -> None:
        ref = ck.parse("https://wandb.ai/ent/proj/runs/d1tafrx7@5999")
        self.assertEqual((ref.name, ref.run_path, ref.model), ("CHECKPOINT", "ent/proj/d1tafrx7", "model_5999.pt"))

    def test_named_run_path_takes_latest(self) -> None:
        ref = ck.parse("REORIENT_CKPT=ent/proj/d1tafrx7")
        self.assertEqual((ref.name, ref.run_path, ref.model), ("REORIENT_CKPT", "ent/proj/d1tafrx7", ""))

    def test_local_path(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            self.assertEqual(ck.parse(f"A={f.name}").local, f.name)

    def test_nonsense_is_refused(self) -> None:
        for bad in ("no/such", "A=", "ent/proj/run@x", "./missing.pt"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ck.parse(bad)

    def test_duplicate_names_are_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ev.parse_checkpoints(["ent/proj/a", "ent/proj/b"])

    def test_bad_env_pair_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ev.parse_env(["1BAD=x"])

    def test_wandb_ref_downloads_through_the_ilab_helper(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            done = mock.Mock(returncode=0, stdout=f"progress\n{f.name}\n", stderr="")
            with (
                mock.patch.object(ev.subprocess, "run", return_value=done) as run,
                mock.patch.object(ev, "wandb_env", return_value={}),
            ):
                path = ev.fetch_checkpoint(ck.parse("ent/proj/run1@7"))
            self.assertEqual(path, f.name)
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[-2:], ["ent/proj/run1", "model_7.pt"])
            self.assertTrue(cmd[1].endswith("checkpoints.py"))


class PoolRefusalTest(unittest.TestCase):
    def test_ray_and_slurm_refuse_with_a_pointer(self) -> None:
        for kind, hint in (("ray", "just ray job"), ("slurm", "develop exec")):
            with self.subTest(kind=kind), self.assertRaises(SystemExit) as cm:
                ev.make_target(Pool("p", kind, {}), "h", 0)
            self.assertIn(hint, str(cm.exception.code))

    def test_ssh_target_uses_pool_settings(self) -> None:
        t = ev.make_target(Pool("larg", "ssh", {"user": "u", "domain": "cs.x"}), "hazard", 2)
        want = ("u@hazard.cs.x", "/var/local/u/hcrl_isaac_manager", "/var/local/u")
        self.assertEqual((t.ssh, t.workspace, t.scratch), want)

    def test_ssh_staging_never_deletes(self) -> None:
        t = ev.make_target(Pool("larg", "ssh", {"user": "u"}), "hazard", 0)
        with mock.patch.object(ev.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            ev.Stage(t).put(__file__, "x.py")
        self.assertFalse(any("--delete" in " ".join(c.args[0]) for c in run.call_args_list))


class CodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        for repo in ("hcrl_isaaclab", "robot_rl", "ssti_tasks", "ssti_robots"):
            (self.tmp / "resources" / repo).mkdir(parents=True)
        (self.tmp / "resources" / "ssti_tasks" / "worktrees" / "wt").mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_packages_only_with_worktree_overrides(self) -> None:
        code = ev.local_code(str(self.tmp), "wt")
        self.assertEqual(sorted(code), ["hcrl_isaaclab", "robot_rl", "ssti_tasks"])
        self.assertTrue(code["ssti_tasks"].endswith("ssti_tasks/worktrees/wt"))
        self.assertTrue(code["robot_rl"].endswith("resources/robot_rl"))

    def test_unknown_worktree_set_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ev.local_code(str(self.tmp), "nope")

    def test_ssh_sync_sends_listed_files_without_delete_and_links_assets(self) -> None:
        t = ev.make_target(Pool("larg", "ssh", {"user": "u"}), "hazard", 3)
        ls_out = mock.Mock(returncode=0, stdout=b"a.py\0gone.py\0")
        (self.tmp / "resources" / "robot_rl" / "a.py").write_text("x")
        calls = []

        def fake_run(cmd: list, **kw: object) -> mock.Mock:
            calls.append((cmd, kw))
            return ls_out if cmd[:1] == ["git"] else mock.Mock(returncode=0)

        with mock.patch.object(ev.subprocess, "run", side_effect=fake_run):
            pp = ev.Stage(t).sync_code({"robot_rl": str(self.tmp / "resources" / "robot_rl")})
        self.assertEqual(pp, ["/var/local/u/res-eval/code/hazard-gpu3/robot_rl"])
        rsync = [(c, kw) for c, kw in calls if c[:1] == ["rsync"]]
        self.assertEqual(len(rsync), 1)
        self.assertNotIn("--delete", " ".join(rsync[0][0]))
        self.assertEqual(rsync[0][1]["input"], b"a.py", "only files present on disk are sent")
        self.assertTrue(any("ln -s" in " ".join(c) for c, _ in calls))


class EvalRunTest(unittest.TestCase):
    """End to end on a stub local pool: the workspace's ilab python is this interpreter."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        ws = self.tmp / "ws"
        (ws / "ilab" / "bin").mkdir(parents=True)
        (ws / "ilab" / "bin" / "python").symlink_to(sys.executable)
        (ws / "scripts").mkdir()
        shutil.copy(COMPUTE.parent / "worktree_env.py", ws / "scripts")
        for repo in ("hcrl_isaaclab", "robot_rl"):
            (ws / "resources" / repo).mkdir(parents=True)
        self.pool = Pool("local", "local", {"workspace": str(ws), "scratch": str(self.tmp / "scratch")})
        self.script = self.tmp / "probe_script.py"
        self.script.write_text(SCRIPT)
        self.ckpt = self.tmp / "model_5.pt"
        self.ckpt.write_text("weights")
        self.patches = [
            mock.patch.object(ls, "STORE", self.tmp / "leases.json"),
            mock.patch.object(ev, "wandb_env", return_value={}),
        ]
        for p in self.patches:
            p.start()
        self.card = Card("local", "box", 1, "RTX", 0, 32000, 0, "free", uuid="GPU-1")
        self.report = Report("local", "local", [self.card], owner="me")

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def _claim(self, args: argparse.Namespace, pools: list) -> tuple:
        with mock.patch.object(res, "probe_all", return_value=[self.report]):
            return res.claim(args, pools)

    def _eval(self, *argv: str) -> tuple[int, str]:
        parser = argparse.ArgumentParser()
        ev.add_parser(parser.add_subparsers(dest="cmd"))
        head, tail = ev.split_script_args(["eval", str(self.script), "--holder", "me", *argv])
        args = parser.parse_args(head)
        args.script_args = tail
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            ev.cmd_eval(args, [self.pool], self._claim)
        return cm.exception.code, out.getvalue() + err.getvalue()

    def _leases(self) -> list:
        with ls.locked_store() as leases:
            return list(leases)

    def test_success_passes_args_env_and_checkpoint_then_releases(self) -> None:
        rc, out = self._eval(
            "--any", "--checkpoint", f"CKPT_A={self.ckpt}", "--env", "MODE=ok", "--", "--n", "a b", "*"
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("ARGV ['--n', 'a b', '*']", out)
        self.assertIn("CKPT_A_CONTENT weights", out)
        self.assertIn("MODE ok", out)
        self.assertIn("CUDA_VISIBLE_DEVICES 1", out)
        self.assertIn(str(self.tmp / "ws" / "resources" / "hcrl_isaaclab"), out)
        self.assertEqual(self._leases(), [])
        left = [p.name for p in (self.tmp / "scratch" / "res-eval").iterdir() if p.name != "cache"]
        self.assertEqual(left, [], "stage dir not removed")

    def test_failure_status_propagates_and_still_releases(self) -> None:
        rc, out = self._eval("--any", "--env", "MODE=fail")
        self.assertEqual(rc, 3, out)
        self.assertIn("FAILED with status 3", out)
        self.assertEqual(self._leases(), [])

    def test_traceback_with_exit_zero_is_failure(self) -> None:
        rc, out = self._eval("--any", "--env", "MODE=traceback")
        self.assertEqual(rc, 1, out)

    def test_each_run_gets_its_own_tmpdir(self) -> None:
        tmpdirs = [line for _ in range(2) for line in self._eval("--any")[1].splitlines() if line.startswith("TMPDIR")]
        self.assertEqual(len(set(tmpdirs)), 2)

    def test_a_passed_lease_is_left_alone(self) -> None:
        [(lease, _, _)], _ = self._claim(
            argparse.Namespace(
                cards=["box:1"], any=False, count=1, min_free_gb=0, pool=None, holder="me", note="", for_=0.0
            ),
            [self.pool],
        )
        rc, out = self._eval("--lease", lease.id, "--env", "MODE=fail")
        self.assertEqual(rc, 3, out)
        self.assertEqual([x.id for x in self._leases()], [lease.id])

    def test_someone_elses_lease_is_refused(self) -> None:
        [(lease, _, _)], _ = self._claim(
            argparse.Namespace(
                cards=["box:1"], any=False, count=1, min_free_gb=0, pool=None, holder="other", note="", for_=0.0
            ),
            [self.pool],
        )
        rc, _ = self._eval("--lease", lease.id)
        self.assertIn("held by other", str(rc))

    def test_exactly_one_card_selector(self) -> None:
        rc, _ = self._eval("--any", "--on", "box:1")
        self.assertIn("exactly one", str(rc))


if __name__ == "__main__":
    unittest.main()
