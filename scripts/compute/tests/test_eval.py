"""`just res eval` with stubs and no GPU: argument checks, pool refusal, checkpoints, code snapshots, lease release,
timeouts and exit status. ssh pools run against a stub `ssh` that executes locally, so rsync and the remote shell
commands are real."""

import argparse
import contextlib
import io
import os
import shutil
import signal
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
if mode == "chatty":
    while True:
        print("tick", flush=True)
        time.sleep(0.1)
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
        subprocess.run(["chmod", "-R", "u+w", str(self.tmp)], check=False)  # snapshots are read-only
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
    def test_ray_refuses_with_a_pointer(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            ev.make_target(Pool("p", "ray", {}), "h", 0)
        self.assertIn("just ray job", str(cm.exception.code))

    def test_a_slurm_card_needs_its_job(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            ev.make_target(Pool("amd-rtx", "slurm", {"login": "u@login"}), "c571-003", 3)
        self.assertIn("running job", str(cm.exception.code))

    def test_slurm_target_stages_under_the_cluster_artifacts(self) -> None:
        with mock.patch.object(ev, "profile_value", return_value="/work/u/isaaclab"):
            t = ev.make_target(Pool("amd-rtx", "slurm", {"login": "u@login"}), "c571-003", 3, "3557743", "GPU-ab")
        self.assertEqual((t.ssh, t.workspace, t.scratch, t.job, t.uuid), (
            "u@login", "/work/u/isaaclab", "/work/u/isaaclab/artifacts", "3557743", "GPU-ab"
        ))  # fmt: skip
        self.assertIn("ControlPath", " ".join(t.ssh_opts), "a TACC login is reached over its master only")
        self.assertEqual(ev.container_path(t, "/work/u/isaaclab/artifacts/res-eval/x/ckpt/A/m.pt"),
                         "/workspace/artifacts/res-eval/x/ckpt/A/m.pt")  # fmt: skip

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

    def test_on_an_unknown_host_without_clusters_names_the_backend(self) -> None:
        pools = [Pool("larg", "ssh", {"hosts": ["pepi"]}), Pool("ray", "ray", {})]
        with self.assertRaises(SystemExit) as cm:
            ev.resolve_on("gpub065:0", pools)
        self.assertIn("ray", str(cm.exception.code))

    def test_on_a_cluster_node_is_left_for_the_claim(self) -> None:
        pools = [Pool("larg", "ssh", {"hosts": ["pepi"]}), Pool("delta", "slurm", {})]
        self.assertEqual(ev.resolve_on("gpub065:22681023:0", pools), "gpub065:22681023:0")

    def test_local_names_this_machine(self) -> None:
        pools = [Pool("local", "local", {})]
        self.assertEqual(ev.resolve_on("local:0", pools), f"{os.uname().nodename.split('.')[0]}:0")


class SlurmTest(unittest.TestCase):
    """The SLURM pieces that run here: profile choice and the container runner."""

    def test_the_profile_named_after_the_partition_wins(self) -> None:
        pools = [Pool(n, "slurm", {"login": "u@login"}) for n in ("amd-rtx", "rtx-small", "stampede")]
        label = "stampede3 (amd-rtx, rtx-small, stampede)"
        for partition, want in (("rtx-small\n", "rtx-small"), ("skx\n", "amd-rtx")):
            with self.subTest(partition=partition):
                done = mock.Mock(returncode=0, stdout=partition, stderr="")
                with mock.patch.object(ev.subprocess, "run", return_value=done):
                    self.assertEqual(ev.slurm_profile(label, "3566414", pools).name, want)
        # a job squeue cannot read is refused rather than guessed
        gone = mock.Mock(returncode=1, stdout="", stderr="slurm_load_jobs error: Invalid job id")
        with mock.patch.object(ev.subprocess, "run", return_value=gone), self.assertRaises(SystemExit) as cm:
            ev.slurm_profile(label, "3566414", pools)
        self.assertIn("Invalid job id", str(cm.exception.code))

    def test_runner_pins_by_uuid_and_stops_on_request(self) -> None:
        t = ev.Target("amd-rtx", "slurm", "c571-003", 3, "u@login", "/w", "/w/artifacts", job="35", uuid="GPU-ab")
        runner = ev.slurm_runner_script(t, "/workspace/artifacts/res-eval/x", "/workspace/ext/r/s.py", ["A"], "/c")
        self.assertIn("nvidia-smi -i GPU-ab", runner)
        self.assertIn("export CUDA_VISIBLE_DEVICES=GPU-ab RES_EVAL_DEVICE=cuda:0", runner)
        self.assertIn("/isaac-sim/python.sh /workspace/ext/r/s.py", runner)
        self.assertIn("/workspace/artifacts/res-eval/x/stop", runner)
        self.assertIn("> /workspace/artifacts/res-eval/x/status", runner)

    def test_a_staged_tree_id_is_read_from_develop_stage(self) -> None:
        out = "  robot_rl /p abc worktree 123\nRun with: just cluster amd-rtx develop exec --tree res-eval-0123456789 -- <cmd>\n"
        done = mock.Mock(returncode=0, stdout=out)
        t = ev.Target("amd-rtx", "slurm", "c571-003", 3, "u@login", "/w", "/w/artifacts", job="35", uuid="GPU-ab")
        with (
            mock.patch.object(ev.subprocess, "run", return_value=done) as run,
            mock.patch.object(ev, "profile_value", return_value=""),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            tree = ev.stage_tree(t, {"robot_rl": "/p", "hcrl_robots": "/q", "IsaacLab": "/i"})
        self.assertEqual(tree, "res-eval-0123456789")
        [stage_call] = [c for c in run.call_args_list if c.args[0][2:3] == ["stage"]]
        self.assertEqual(stage_call.args[0][2:], ["stage", "res-eval", "robot_rl=/p"], "asset repos stay shared")
        self.assertEqual(stage_call.kwargs["env"]["CLUSTER"], "amd-rtx")
        prune = run.call_args_list[-1].args[0][-1]
        self.assertIn("/w/trees/res-eval-*", prune)
        self.assertIn("res-eval-0123456789", prune, "the tree just staged is never pruned")
        self.assertTrue(prune.startswith("touch /w/trees/res-eval-0123456789;"), "a reused tree counts as just used")
        self.assertIn(".in-use", prune, "a tree a running step uses is never pruned")


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

    def test_a_mode_change_on_a_changed_file_is_a_new_fingerprint(self) -> None:
        (self.src / "new.py").write_text("z = 1\n")
        before = ev.code_fingerprint(str(self.src))[0]
        (self.src / "new.py").chmod(0o755)
        self.assertNotEqual(ev.code_fingerprint(str(self.src))[0], before)

    def test_a_staged_rename_counts_once(self) -> None:
        subprocess.run(["git", "-C", str(self.src), "mv", "mod.py", "renamed.py"], check=True)
        self.assertTrue(ev.code_fingerprint(str(self.src))[1].endswith("dirty=1"))

    def test_snapshots_are_read_only(self) -> None:
        stage = self._stage()
        snap = Path(os.path.realpath(f"{stage.dir}/resources/robot_rl"))
        self.assertFalse(os.access(snap / "mod.py", os.W_OK))
        self.assertFalse(os.access(snap, os.W_OK))

    def test_an_interrupted_upload_leaves_no_partial(self) -> None:
        stage = ev.Stage(self.t)
        stage.make()
        real = subprocess.run

        def interrupt_rsync(cmd: list, **kw: object) -> subprocess.CompletedProcess:
            if cmd[:1] == ["rsync"]:
                raise KeyboardInterrupt
            return real(cmd, **kw)

        with (
            mock.patch.object(ev.subprocess, "run", side_effect=interrupt_rsync),
            self.assertRaises(KeyboardInterrupt),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            stage.sync_code({"robot_rl": str(self.src)})
        self.assertEqual(list((self.tmp / "scratch" / "res-eval" / "code").glob("*.partial.*")), [])

    def test_the_env_file_is_never_readable_by_others(self) -> None:
        stage = self._stage()
        modes = []
        real_put = ev.Stage.put

        def spy(self_: ev.Stage, src: str, rel: str, mode: int = 0o600) -> str:
            modes.append(os.stat(src).st_mode & 0o777)
            return real_put(self_, src, rel, mode)

        with mock.patch.object(ev.Stage, "put", spy):
            dest = stage.write("WANDB_API_KEY=x\n", "env")
        self.assertEqual(modes, [0o600])
        self.assertEqual(os.stat(dest).st_mode & 0o777, 0o600)

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


CLUSTER_DEV_STUB = """#!/usr/bin/env bash
# records its call, then runs the command after -- here: in the background with --detach, else in the foreground
{ echo "JOB=${DEV_JOBID:-} CLUSTER=${CLUSTER:-}"; printf '%s\\n' "$@"; } > "$CALLS"
[ -n "${FAIL_START:-}" ] && exit 1
detach=""; log=/dev/null
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in --detach) detach=1 ;; --log) log="$2"; shift ;; esac
    shift
done
shift
[ -n "$detach" ] && { setsid nohup "$@" > "$log" 2>&1 < /dev/null & exit 0; }
exec "$@"
"""

# the card a step can see is GPU-ab; no process is left on it after a run
NVIDIA_SMI_STUB = """#!/usr/bin/env bash
case " $* " in *" --query-compute-apps"*) exit 0 ;; esac
[ "$2" = GPU-ab ]
"""

SLURM_SCRIPT = """import os, sys, time
for k in ("CUDA_VISIBLE_DEVICES", "RES_EVAL_DEVICE", "TMPDIR", "N"):
    print(k, os.environ.get(k, "-"))
print("CWD", os.getcwd())
print("PID", os.getpid(), flush=True)
if os.environ.get("CKPT_A"):
    print("CKPT_A_CONTENT", open(os.environ["CKPT_A"]).read())
mode = os.environ.get("MODE", "")
if mode == "traceback":
    print("Traceback (most recent call last):")
if mode == "hang":
    time.sleep(120)
"""


class SlurmEvalTest(Isolated):
    """A run on a held sentinel's card, executed here: stub ssh (the login is this machine), stub cluster_dev.sh,
    stub nvidia-smi, and container paths equal to the stage's own."""

    def setUp(self) -> None:
        super().setUp()
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        for name, text in (("ssh", SSH_STUB), ("nvidia-smi", NVIDIA_SMI_STUB), ("squeue", "#!/bin/sh\necho amd-rtx\n")):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        stub = self.tmp / "cluster_dev.sh"
        stub.write_text(CLUSTER_DEV_STUB)
        self.calls = self.tmp / "calls"
        self.remote = self.tmp / "cluster"
        ws = self.tmp / "ws"
        self.main_repo = ws / "resources" / "hcrl_isaaclab"
        self.main_repo.mkdir(parents=True)
        self.src = self.tmp / "local" / "robot_rl" / "worktrees" / "wt"
        _git_repo(self.src)
        self.code = {"robot_rl": str(self.src), "hcrl_isaaclab": str(self.main_repo)}
        self.env_patch = mock.patch.dict(
            os.environ, {"PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(self.calls)}
        )
        for p in (
            self.env_patch,
            mock.patch.object(ev, "CLUSTER_DEV", stub),
            mock.patch.object(ev, "CONTAINER_ARTIFACTS", str(self.remote / "artifacts")),
            mock.patch.object(ev, "CONTAINER_PYTHON", sys.executable),
            mock.patch.object(ev, "CONTAINER_TMP", str(self.tmp / "node_tmp")),
            mock.patch.object(ev, "START_S", 20),
            mock.patch.object(ev, "profile_value", return_value=str(self.remote)),
            mock.patch.object(ev, "stage_tree", return_value="res-eval-0123456789"),
            mock.patch.object(ev, "local_workspace", return_value=str(ws)),
            mock.patch.object(ev, "local_code", side_effect=lambda *_: dict(self.code)),
        ):
            p.start()
            self.patches.append(p)
        self.pool = Pool("amd-rtx", "slurm", {"login": "u@login"})
        self.script = self.tmp / "census.py"
        self.script.write_text(SLURM_SCRIPT)
        self.ckpt = self.tmp / "model_5.pt"
        self.ckpt.write_text("weights")
        self.uuid = "GPU-ab"

    def _claim(self, args: argparse.Namespace, pools: list) -> tuple:
        card = Card("stampede3 (amd-rtx)", "c571-003", 3, "RTX", 0, 97000, 0, "free", job="3557743", uuid=self.uuid)
        with mock.patch.object(res, "probe_all", return_value=[Report("stampede3 (amd-rtx)", "slurm", [card])]):
            return res.claim(args, pools)

    def _eval(self, *argv: str) -> tuple[int, str]:
        parser = argparse.ArgumentParser()
        ev.add_parser(parser.add_subparsers(dest="cmd"))
        base = ["eval", str(self.script), "--holder", "me", "--on", "c571-003:3"]
        head, tail = ev.split_script_args([*base, "--checkpoint", f"CKPT_A={self.ckpt}", *argv])
        args = parser.parse_args(head)
        args.script_args = tail
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            ev.cmd_eval(args, [self.pool], self._claim)
        return cm.exception.code, out.getvalue() + err.getvalue()

    def _leases(self) -> list:
        with ls.locked_store() as leases:
            return list(leases)

    def _stage(self) -> Path:
        root = self.remote / "artifacts" / "res-eval"
        [stage] = [p for p in root.iterdir() if p.name not in ("cache", "work")]
        return stage

    def _status(self, stage: Path) -> str:
        for _ in range(200):
            if (stage / "status").exists():
                return (stage / "status").read_text().strip()
            time.sleep(0.1)
        self.fail(f"no status in {stage}: {(stage / 'log').read_text() if (stage / 'log').exists() else '-'}")

    def test_detached_run_executes_on_the_card_and_keeps_its_lease(self) -> None:
        rc, out = self._eval("--detach", "--env", "N=2", "--", "-x")
        self.assertEqual(rc, 0, out)
        stage = self._stage()
        self.assertEqual(self._status(stage), "0")
        log = (stage / "log").read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES GPU-ab", log)
        self.assertIn("CKPT_A_CONTENT weights", log)
        node_tmp = self.tmp / "node_tmp" / stage.name
        self.assertIn(f"TMPDIR {node_tmp}", log, "a per-run TMPDIR inside the node-local /tmp")
        self.assertFalse(node_tmp.exists(), "the run's TMPDIR goes with it")
        self.assertIn(f"CWD {stage}", log)
        self.assertFalse((stage / "env").exists(), "the runner removes the credentials file")
        self.assertTrue((stage / "heartbeat").exists())
        self.assertEqual([x.card for x in self._leases()], ["c571-003:3 (job 3557743)"], "a detached run keeps it")
        calls = self.calls.read_text().splitlines()
        self.assertEqual(calls[0], "JOB=3557743 CLUSTER=amd-rtx")
        self.assertEqual(calls[1:6], ["exec", "--tree", "res-eval-0123456789", "--detach", "--log"])
        self.assertIn("ControlPath", out, "the printed stop command goes over the master")
        ev.stage_tree.assert_called_once()
        self.assertEqual(ev.stage_tree.call_args.args[1], {"robot_rl": str(self.src)}, "only the --wt repo is staged")

    def test_without_worktrees_the_run_uses_the_shared_checkout(self) -> None:
        self.code = {"hcrl_isaaclab": str(self.main_repo)}
        rc, out = self._eval("--detach")
        self.assertEqual(rc, 0, out)
        self._status(self._stage())
        ev.stage_tree.assert_not_called()
        self.assertNotIn("--tree", self.calls.read_text().splitlines())
        self.assertIn("(shared checkout)", (self._stage() / "MANIFEST").read_text())

    def test_foreground_run_streams_releases_and_cleans_up(self) -> None:
        rc, out = self._eval("--env", "N=7")
        self.assertEqual(rc, 0, out)
        self.assertIn("CUDA_VISIBLE_DEVICES GPU-ab", out)
        self.assertIn("N 7", out)
        self.assertEqual(self._leases(), [])
        self.assertEqual([p for p in (self.remote / "artifacts" / "res-eval").iterdir() if p.name != "cache"], [])

    def test_a_timeout_stops_the_run_through_its_stop_file(self) -> None:
        rc, out = self._eval("--env", "MODE=hang", "--timeout", "3s", "--stall", "0s")
        self.assertEqual(rc, ev.TIMED_OUT, out)
        self.assertIn("killed the run (gone)", out)
        pid = int(next(line.split()[1] for line in out.splitlines() if line.startswith("PID ")))
        self.assertFalse(_alive(pid), "the script's process group is gone")
        self.assertEqual(self._leases(), [])

    def test_a_traceback_with_exit_zero_is_a_failed_status(self) -> None:
        rc, out = self._eval("--detach", "--env", "MODE=traceback")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self._status(self._stage()), "1")

    def test_a_run_stopped_before_its_step_starts_never_launches(self) -> None:
        t = ev.Target("amd-rtx", "slurm", "c571-003", 3, "u@login", str(self.remote), str(self.remote / "artifacts"),
                      job="3557743", uuid=self.uuid)  # fmt: skip
        stage = ev.Stage(t)
        stage.make()
        Path(stage.dir, "env").write_text("SECRET=x\n")
        Path(stage.dir, "MANIFEST").write_text("")
        Path(stage.dir, "stop").touch()
        runner = ev.slurm_runner_script(t, stage.dir, str(self.script), [], stage.dir)
        Path(stage.dir, "run.sh").write_text(runner)
        rc = subprocess.run(["bash", f"{stage.dir}/run.sh"], capture_output=True, text=True)
        self.assertEqual(rc.returncode, 130)
        self.assertEqual(Path(stage.dir, "status").read_text().strip(), "130")
        self.assertNotIn("PID", Path(stage.dir, "log").read_text(), "the script never ran")
        self.assertFalse(Path(stage.dir, "env").exists(), "and its credentials are gone")

    def test_a_card_the_step_cannot_see_fails_fast(self) -> None:
        self.uuid = "GPU-elsewhere"
        rc, out = self._eval()
        self.assertEqual(rc, 98, out)
        self.assertIn("is not visible in this step", out)
        self.assertEqual(self._leases(), [])

    def test_a_run_that_never_starts_releases_and_drops_the_credentials(self) -> None:
        with mock.patch.dict(os.environ, {"FAIL_START": "1"}), mock.patch.object(ev, "START_S", 1):
            rc, _ = self._eval("--detach")
        self.assertIn("did not start", str(rc))
        self.assertEqual(self._leases(), [])
        stage = self._stage()
        self.assertFalse((stage / "env").exists(), "credentials removed when the step never ran")
        self.assertTrue((stage / "stop").exists(), "a late start ends at once")


class PruneStagesTest(unittest.TestCase):
    def test_old_credentials_and_dead_stages_go_live_ones_stay(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        hour, week = time.time() - 3600, time.time() - 10 * 86400

        def stage(name: str, files: dict[str, float], age: float) -> Path:
            d = root / name
            d.mkdir(parents=True)
            for f, when in files.items():
                (d / f).write_text("x")
                os.utime(d / f, (when, when))
            os.utime(d, (age, age))
            return d

        unstarted = stage("unstarted", {"env": hour}, hour)
        finished = stage("finished", {"status": week, "heartbeat": week}, week)
        live = stage("live", {"heartbeat": time.time()}, week)
        old_work = stage("work/abc", {}, week)
        subprocess.run(["bash", "-c", ev._prune_stages(str(root))], check=True)
        self.assertTrue(unstarted.is_dir() and not (unstarted / "env").exists(), "credentials of a run that never ran")
        self.assertFalse(finished.exists())
        self.assertTrue(live.is_dir(), "a run with a fresh heartbeat is never pruned")
        self.assertFalse(old_work.exists())


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
        rc, out = self._eval(
            "--any", "--checkpoint", f"CKPT_A={self.ckpt}", "--env", "MODE=ok", "--", "--n", "a b", "*"
        )
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

    def test_a_timeout_fires_on_a_script_that_prints_often(self) -> None:
        start = time.monotonic()
        rc, out = self._eval("--any", "--env", "MODE=chatty", "--timeout", "3s", "--stall", "0")
        self.assertEqual(rc, ev.TIMED_OUT, out[-300:])
        self.assertLess(time.monotonic() - start, 20)
        self.assertFalse(_alive(self._pid(out)))

    def test_sigterm_to_res_eval_kills_the_run_and_releases(self) -> None:
        timer = threading.Timer(3, os.kill, (os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            rc, out = self._eval("--any", "--env", "MODE=hang")
        finally:
            timer.cancel()
        self.assertEqual(rc, 130, out)
        self.assertIn("interrupted (SIGTERM)", out)
        self.assertFalse(_alive(self._pid(out)))
        self.assertEqual(self._leases(), [])
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_any_exception_after_the_start_kills_the_run(self) -> None:
        pids = []

        def broken(stage: ev.Stage, argv: list, timeout: float, stall: float) -> int:
            proc = stage.start(argv)
            pids.append(int(next(x for x in proc.stdout if x.startswith("PID ")).split()[1]))
            raise BrokenPipeError

        parser = argparse.ArgumentParser()
        ev.add_parser(parser.add_subparsers(dest="cmd"))
        args = parser.parse_args(["eval", str(self.script), "--holder", "me", "--any"])
        args.script_args = []
        with (
            mock.patch.object(ev, "run", side_effect=broken),
            mock.patch.dict(os.environ, {"MODE": "hang"}),
            self.assertRaises(BrokenPipeError),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            ev.cmd_eval(args, [self.pool], self._claim)
        self.assertFalse(_alive(pids[0]))
        self.assertEqual(self._leases(), [])

    def test_a_run_that_survives_the_kill_keeps_its_lease(self) -> None:
        def stuck(stage: ev.Stage, argv: list, timeout: float, stall: float) -> int:
            stage.start(argv)
            raise KeyboardInterrupt

        with mock.patch.object(ev, "run", side_effect=stuck), mock.patch.object(ev.Stage, "kill", return_value=False):
            _, out = self._eval("--any", "--env", "MODE=hang")
        self.assertIn("KEPT lease", out)
        self.assertEqual(len(self._leases()), 1)
        [stage] = self._stages()
        with contextlib.suppress(ProcessLookupError):  # the kill this test stubbed out
            os.killpg(int((stage / "pid").read_text()), signal.SIGKILL)

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
        self.assertIn("interrupted", out)
        self.assertIn("stopped the run (gone)", out)
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

    def test_a_repo_script_runs_from_its_repo_with_sibling_imports(self) -> None:
        scripts = self.tmp / "ws" / "resources" / "robot_rl" / "scripts"
        scripts.mkdir()
        (scripts / "helper.py").write_text("NAME = 'helper ok'\n")
        (scripts / "train.py").write_text(
            "import os, sys\nsys.path.insert(0, os.path.dirname(__file__))\nimport helper\n"
            "print('CWD', os.getcwd())\nprint('HELPER', helper.NAME)\n"
        )
        self.script = "robot_rl:scripts/train.py"
        with mock.patch.object(ev, "local_workspace", return_value=str(self.tmp / "ws")):
            rc, out = self._eval("--any")
            _, out2 = self._eval("--any")
        self.assertEqual(rc, 0, out)
        self.assertIn("HELPER helper ok", out)
        cwd = Path(next(line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("CWD ")))
        self.assertEqual(cwd.parent, self.tmp / "scratch" / "res-eval" / "work")
        self.assertTrue(cwd.is_dir(), "the work dir is kept after the run")
        cwd2 = next(line.split(" ", 1)[1] for line in out2.splitlines() if line.startswith("CWD "))
        self.assertNotEqual(str(cwd), cwd2, "each run gets its own work dir")

    def test_an_unshipped_repo_script_is_refused_before_a_lease(self) -> None:
        self.script = "nope:scripts/train.py"
        with mock.patch.object(ev, "local_workspace", return_value=str(self.tmp / "ws")):
            rc, _ = self._eval("--any")
        self.assertIn("not a shipped repo", str(rc))
        self.assertEqual(self._leases(), [])

    def test_detach_returns_at_once_and_keeps_the_lease(self) -> None:
        start = time.monotonic()
        rc, out = self._eval("--any", "--detach", "--env", "MODE=hang")
        self.assertEqual(rc, 0, out)
        self.assertLess(time.monotonic() - start, 15)
        self.assertEqual(len(self._leases()), 1)
        self.assertIn("lease", out)
        [stage] = self._stages()
        for _ in range(50):
            if "PID " in (stage / "log").read_text() if (stage / "log").exists() else False:
                break
            time.sleep(0.2)
        pid = self._pid((stage / "log").read_text())
        self.assertTrue(_alive(pid))
        self.assertFalse((stage / "env").exists(), "credentials removed once the run read them")
        stop = next(line.split("stop:", 1)[1].strip() for line in out.splitlines() if "stop:" in line)
        subprocess.run(["bash", "-c", stop], check=True)
        for _ in range(50):
            if not _alive(pid):
                break
            time.sleep(0.2)
        self.assertFalse(_alive(pid))

    def test_exactly_one_card_selector(self) -> None:
        rc, _ = self._eval("--any", "--on", "box:1")
        self.assertIn("exactly one", str(rc))


if __name__ == "__main__":
    unittest.main()
