"""`just res eval` with stubs and no GPU: argument checks, pool refusal, checkpoints, code snapshots, lease release,
timeouts and exit status. ssh pools run against a stub `ssh` that executes locally, so rsync and the remote shell
commands are real."""

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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

SCRIPT = """import os, sys, time
print("ARGV", sys.argv[1:])
for k in ("CKPT_A", "MODE", "CUDA_VISIBLE_DEVICES", "RES_EVAL_DEVICE"):
    print(k, os.environ.get(k, "-"))
print("TMPDIR", os.environ["TMPDIR"])
print("PP", os.environ["PYTHONPATH"])
print("PID", os.getpid(), flush=True)
if os.environ.get("CKPT_A"):
    print("CKPT_A_CONTENT", open(os.environ["CKPT_A"]).read())
mode = os.environ.get("MODE", "")
if mode == "fail":
    sys.exit(3)
if mode == "traceback":
    print("Traceback (most recent call last):")
if mode == "hang":
    time.sleep(120)
"""

SSH_STUB = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-J|-i|-p|-l|-F|-E|-c|-m|-L|-R|-D|-W|-b|-e|-S|-w|-Q|-B) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
exec bash -c "$*"
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(path), "config", k, v], check=True)
    (path / "mod.py").write_text("x = 1\n")
    (path / "old_mod.py").write_text("y = 1\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)


class Isolated(unittest.TestCase):
    """Every test gets its own checkpoint cache and lease store."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [
            mock.patch.object(ev, "CKPT_CACHE", self.tmp / "cache" / "checkpoints"),
            mock.patch.object(ls, "STORE", self.tmp / "leases.json"),
            mock.patch.object(ev, "wandb_env", return_value={}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)


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


class TargetTest(unittest.TestCase):
    def test_ray_and_slurm_refuse_with_a_pointer(self) -> None:
        for kind, hint in (("ray", "just ray job"), ("slurm", "develop exec")):
            with self.subTest(kind=kind), self.assertRaises(SystemExit) as cm:
                ev.make_target(Pool("p", kind, {}), "h", 0)
            self.assertIn(hint, str(cm.exception.code))

    def test_ssh_target_uses_pool_settings(self) -> None:
        t = ev.make_target(Pool("larg", "ssh", {"user": "u", "domain": "cs.x"}), "hazard", 2)
        want = ("u@hazard.cs.x", "/var/local/u/hcrl_isaac_manager", "/var/local/u", "cvd")
        self.assertEqual((t.ssh, t.workspace, t.scratch, t.pin), want)

    def test_device_pinning_leaves_the_mask_off(self) -> None:
        t = ev.make_target(Pool("a40", "ssh", {"user": "u", "pin": "device"}), "pepi", 2)
        runner = ev.runner_script(t, "/s", "x.py", [], [])
        self.assertIn("unset CUDA_VISIBLE_DEVICES; export RES_EVAL_DEVICE=cuda:2", runner)
        cvd = ev.runner_script(ev.make_target(Pool("a", "ssh", {"user": "u"}), "h", 2), "/s", "x.py", [], [])
        self.assertIn("export CUDA_VISIBLE_DEVICES=2 RES_EVAL_DEVICE=cuda:0", cvd)

    def test_a_bad_pin_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ev.make_target(Pool("a", "ssh", {"user": "u", "pin": "maybe"}), "h", 0)

    def test_on_a_cluster_node_names_the_backend(self) -> None:
        pools = [Pool("larg", "ssh", {"hosts": ["pepi"]}), Pool("delta", "slurm", {})]
        with self.assertRaises(SystemExit) as cm:
            ev._check_on("gpub065:0", pools)
        self.assertIn("slurm", str(cm.exception.code))


class LocalCodeTest(unittest.TestCase):
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


class SshStageTest(Isolated):
    """The ssh code path against a stub ssh: snapshots, links and the shared checkout are real directories."""

    def setUp(self) -> None:
        super().setUp()
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "ssh").write_text(SSH_STUB)
        (bin_dir / "ssh").chmod(0o755)
        self.patches.append(mock.patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ['PATH']}"}))
        self.patches[-1].start()
        self.ws = self.tmp / "remote_ws"
        for repo in ("robot_rl", "hcrl_robots"):
            (self.ws / "resources" / repo).mkdir(parents=True)
        self.shared_file = self.ws / "resources" / "robot_rl" / "mod.py"
        self.shared_file.write_text("shared = True\n")
        self.src = self.tmp / "local" / "robot_rl"
        _git_repo(self.src)
        pool = Pool("larg", "ssh", {"user": "u", "workspace": str(self.ws), "scratch": str(self.tmp / "scratch")})
        self.t = ev.make_target(pool, "box", 0)

    def _stage(self) -> ev.Stage:
        stage = ev.Stage(self.t)
        stage.make()
        with contextlib.redirect_stderr(io.StringIO()):
            stage.pp, stage.manifest = stage.sync_code({"robot_rl": str(self.src)})
        return stage

    def test_code_lands_in_a_snapshot_never_in_the_shared_checkout(self) -> None:
        stage = self._stage()
        self.assertEqual(self.shared_file.read_text(), "shared = True\n")
        snap = Path(os.path.realpath(f"{stage.dir}/resources/robot_rl"))
        self.assertTrue(snap.name.startswith("robot_rl-") and (snap / ".complete").is_file())
        self.assertEqual((snap / "mod.py").read_text(), "x = 1\n")
        self.assertEqual(stage.pp, [f"{stage.dir}/resources/robot_rl"])
        self.assertEqual(os.path.realpath(f"{stage.dir}/resources/hcrl_robots"), str(self.ws / "resources/hcrl_robots"))

    def test_a_repo_first_linked_then_shipped_never_writes_through_the_link(self) -> None:
        stage = self._stage()
        self.assertTrue(os.path.islink(f"{stage.dir}/resources/robot_rl"))
        (self.src / "mod.py").write_text("x = 2\n")
        self._stage()
        self.assertEqual(self.shared_file.read_text(), "shared = True\n")

    def test_a_deleted_file_is_not_importable_after_the_next_snapshot(self) -> None:
        first = self._stage()
        (self.src / "old_mod.py").unlink()
        second = self._stage()
        a, b = (os.path.realpath(f"{s.dir}/resources/robot_rl") for s in (first, second))
        self.assertNotEqual(a, b)
        self.assertTrue(os.path.exists(f"{a}/old_mod.py"))
        self.assertFalse(os.path.exists(f"{b}/old_mod.py"))
        self.assertIn("dirty=1", second.manifest[0])

    def test_concurrent_stages_share_one_complete_snapshot(self) -> None:
        stages = [ev.Stage(self.t) for _ in range(3)]
        for s in stages:
            s.make()

        def sync(s: ev.Stage) -> None:
            with contextlib.redirect_stderr(io.StringIO()):
                s.sync_code({"robot_rl": str(self.src)})

        threads = [threading.Thread(target=sync, args=(s,)) for s in stages]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        snaps = {os.path.realpath(f"{s.dir}/resources/robot_rl") for s in stages}
        self.assertEqual(len(snaps), 1)
        self.assertEqual(list((self.tmp / "scratch" / "res-eval" / "code").glob("*.partial.*")), [])
        self.assertEqual(len({s.dir for s in stages}), 3)

    def test_staging_never_deletes(self) -> None:
        with mock.patch.object(ev.subprocess, "run", wraps=subprocess.run) as run:
            stage = self._stage()
            stage.put(__file__, "x.py")
        self.assertFalse(any("--delete" in " ".join(map(str, c.args[0])) for c in run.call_args_list))

    def test_a_symlinked_checkpoint_lands_as_its_content(self) -> None:
        real = self.tmp / "real.pt"
        real.write_text("weights")
        link = self.tmp / "link.pt"
        link.symlink_to(real)
        stage = self._stage()
        dest = stage.link_checkpoint(str(link), "A")
        self.assertEqual(Path(dest).read_text(), "weights")
        self.assertFalse(os.path.islink(dest))


class EvalRunTest(Isolated):
    """End to end on a stub local pool: the workspace's ilab python is this interpreter."""

    def setUp(self) -> None:
        super().setUp()
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
        self.card = Card("local", "box", 1, "RTX", 0, 32000, 0, "free", uuid="GPU-1")
        self.report = Report("local", "local", [self.card], owner="me")

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

    def _stages(self) -> list[Path]:
        root = self.tmp / "scratch" / "res-eval"
        return [p for p in root.iterdir() if p.name not in ("cache", "artifacts", "code")] if root.is_dir() else []

    def _pid(self, out: str) -> int:
        return int(next(line.split()[1] for line in out.splitlines() if line.startswith("PID ")))

    def test_success_passes_args_env_and_checkpoint_then_cleans_up(self) -> None:
        rc, out = self._eval("--any", "--checkpoint", f"CKPT_A={self.ckpt}", "--env", "MODE=ok", "--", "--n", "a b", "*")
        self.assertEqual(rc, 0, out)
        self.assertIn("ARGV ['--n', 'a b', '*']", out)
        self.assertIn("CKPT_A_CONTENT weights", out)
        self.assertIn("MODE ok", out)
        self.assertIn("CUDA_VISIBLE_DEVICES 1", out)
        self.assertIn("RES_EVAL_DEVICE cuda:0", out)
        self.assertIn("robot_rl " + str(self.tmp / "ws" / "resources" / "robot_rl"), out, "MANIFEST is printed")
        self.assertEqual(self._leases(), [])
        self.assertEqual(self._stages(), [], "stage dir not removed")

    def test_failure_status_propagates_keeps_the_log_and_releases(self) -> None:
        rc, out = self._eval("--any", "--env", "MODE=fail")
        self.assertEqual(rc, 3, out)
        self.assertEqual(self._leases(), [])
        [stage] = self._stages()
        self.assertIn("MODE fail", (stage / "log").read_text())
        self.assertFalse((stage / "env").exists(), "credentials file removed even on failure")

    def test_traceback_with_exit_zero_is_failure(self) -> None:
        rc, out = self._eval("--any", "--env", "MODE=traceback")
        self.assertEqual(rc, 1, out)

    def test_a_timeout_kills_the_process_group_and_releases(self) -> None:
        start = time.monotonic()
        rc, out = self._eval("--any", "--env", "MODE=hang", "--timeout", "2s", "--stall", "0s")
        self.assertEqual(rc, ev.TIMED_OUT, out)
        self.assertLess(time.monotonic() - start, 60)
        self.assertIn("killed the run (gone)", out)
        self.assertFalse(_alive(self._pid(out)))
        self.assertEqual(self._leases(), [])

    def test_a_stall_kills_the_run(self) -> None:
        rc, out = self._eval("--any", "--env", "MODE=hang", "--stall", "2s")
        self.assertEqual(rc, ev.TIMED_OUT, out)
        self.assertIn("stalled", out)
        self.assertFalse(_alive(self._pid(out)))

    def test_an_interrupt_kills_the_run_and_releases(self) -> None:
        real_run = ev.run

        def interrupted(stage: ev.Stage, argv: list, timeout: float, stall: float) -> int:
            stage.start(argv)
            time.sleep(3)
            raise KeyboardInterrupt

        with mock.patch.object(ev, "run", side_effect=interrupted):
            rc, out = self._eval("--any", "--env", "MODE=hang")
        self.assertIs(ev.run, real_run)
        self.assertEqual(rc, 130, out)
        self.assertIn("interrupted; killed the run (gone)", out)
        self.assertEqual(self._leases(), [])

    def test_an_error_while_staging_still_releases(self) -> None:
        with mock.patch.object(ev, "fetch_checkpoint", side_effect=SystemExit("[res] could not fetch")):
            rc, _ = self._eval("--any", "--checkpoint", "ent/proj/run")
        self.assertIn("could not fetch", str(rc))
        self.assertEqual(self._leases(), [])

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
