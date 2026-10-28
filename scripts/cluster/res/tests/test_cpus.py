"""`pls run --on <host> --cpus N`: CPU-only runs on an ssh box, with stubs and no GPU. The box is this machine behind a
stub `ssh` that runs the remote command locally, so taskset, nice and the stage dirs are real."""

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

RES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RES))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev
import leases as ls
import res
from inventory import Pool
from probe import Card, Report
from test_eval import SSH_STUB, _alive, _git_repo

T0 = 1_000_000.0
# what the run sees: its cores, its GPU mask, its thread pools and its niceness
PROBE = (
    "import os; print('AFFINITY', sorted(os.sched_getaffinity(0))); print('CVD', repr(os.environ.get('CUDA_VISIBLE_DEVICES')));"
    " print('OMP', os.environ.get('OMP_NUM_THREADS'), os.environ.get('MKL_NUM_THREADS'),"
    " os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('NUMEXPR_NUM_THREADS'));"
    " print('DEV', os.environ.get('RES_EVAL_DEVICE', '-')); print('NICE', os.nice(0)); print('PID', os.getpid(), flush=True)"
)


def _no_card_claim(*_args: object) -> None:
    raise AssertionError("a --cpus run must not claim a card")


def cpu_lease(host: str = "box", cpus: int = 2, cores: str = "6-7", stage: str = "", owner: str = "u") -> ls.Lease:
    lease = ls.new_cpu_lease(f"larg/{host}", host, owner, "me", "", cpus, cores)
    lease.stage = stage
    return lease


class CoresTest(unittest.TestCase):
    def test_cores_round_trip_as_taskset_lists(self) -> None:
        self.assertEqual(ls.format_cores([8, 0, 1, 2, 5]), "0-2,5,8")
        self.assertEqual(ls.parse_cores("0-2,5,8"), {0, 1, 2, 5, 8})

    def test_the_highest_cores_no_other_cpu_run_holds(self) -> None:
        self.assertEqual(ls.pick_cores(64, 24, set()), "40-63")
        self.assertEqual(ls.pick_cores(64, 8, set(range(40, 64))), "32-39")
        self.assertEqual(ls.pick_cores(8, 4, set(range(0, 7))), "4-7", "too few unpinned: the last N")

    def test_budget_defaults_to_half_the_box(self) -> None:
        self.assertEqual(ls.cpu_budget(None, 64), 32)
        self.assertEqual(ls.cpu_budget(48, 64), 48)
        for bad in (0, -1, "32", True, 1.5):
            with self.assertRaises(ValueError):
                ls.cpu_budget(bad, 64)


class CpuLeaseTest(unittest.TestCase):
    """GPU-card logic leaves CPU leases alone: no idle lapse, no missing-card release, no reservation yield."""

    def test_reconcile_never_releases_a_cpu_lease_for_its_missing_card(self) -> None:
        lease = cpu_lease()
        lease.created = T0
        complete = Report("larg/box", "ssh", [Card("larg", "box", 0, "A40", 0, 46000, 0)], owner="u")
        leases = [lease]
        for minute in (0, 60, 600):
            self.assertEqual(ls.reconcile(leases, [complete], now=T0 + minute * 60), [])
        self.assertEqual(leases, [lease])

    def test_a_time_box_still_ends_it(self) -> None:
        lease = cpu_lease()
        lease.expires = T0
        self.assertEqual(len(ls.reconcile([lease], [], now=T0 + 1)), 1)

    def test_a_pool_reservation_does_not_cover_it(self) -> None:
        lease = cpu_lease()
        r = ls.Reservation(id="r1", holder="other", pool="larg", count=1, expires=T0 + 3600)
        self.assertEqual(ls.mark_yields(r, [lease]), [])
        self.assertEqual(lease.yield_to, "")

    def test_a_card_on_the_same_box_is_still_claimable(self) -> None:
        c = Card("larg", "box", 0, "A40", 0, 46000, 0, uuid="GPU-0")
        args = argparse.Namespace(any=True, cards=[], count=1, min_free_gb=0, pool=None, holder="x")
        chosen = res._choose([(c, Report("larg/box", "ssh", [c]))], {cpu_lease().key: cpu_lease()}, args)
        self.assertEqual([x[0] for x in chosen], [c])

    def test_reap_releases_ended_and_never_started_runs_only(self) -> None:
        ended, alive, unstarted, fresh = (cpu_lease(stage=s) for s in ("/s/a", "/s/b", "", ""))
        unstarted.created, fresh.created = T0 - ls.CPU_START_S, T0
        other_box = cpu_lease(host="pepi", stage="/s/a")
        leases = [ended, alive, unstarted, fresh, other_box]
        gone = ls.reap_cpu(leases, "larg/box", {"/s/a"}, now=T0)
        self.assertEqual([x for x, _ in gone], [ended, unstarted])
        self.assertEqual(leases, [alive, fresh, other_box])


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.t = ev.make_target(Pool("larg", "ssh", {"user": "u"}), "box", -1)
        self.t.cores = "40-63"

    def test_no_cuda_pinned_cores_nice_and_thread_pools(self) -> None:
        run = ev.runner_script(self.t, "/s/st", "x.py", [], [])
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', run)
        self.assertIn("unset RES_EVAL_DEVICE", run)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertIn(f'{var}="${{{var}:-24}}"', run)
        self.assertIn('taskset -c 40-63 nice -n 10 "$py" x.py "$@"', run)
        self.assertIn("XDG_CACHE_HOME=/s/st/cache/xdg", run, "a CPU run's caches are its own")
        self.assertNotIn("gpu-1", run)

    def test_a_command_runs_pinned_too(self) -> None:
        run = ev.runner_script(self.t, "/s/st", "", [], [], command=True)
        self.assertIn('export PATH="${py%/*}:$PATH"; taskset -c 40-63 nice -n 10 "$@"', run)

    def test_a_card_run_is_unchanged(self) -> None:
        run = ev.runner_script(ev.make_target(Pool("larg", "ssh", {"user": "u"}), "box", 2), "/s", "x.py", [], [])
        self.assertIn("export CUDA_VISIBLE_DEVICES=2 RES_EVAL_DEVICE=cuda:0", run)
        self.assertNotIn("taskset", run)


class CpuRunTest(unittest.TestCase):
    """End to end on a stub ssh pool whose box is this machine."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "ssh").write_text(SSH_STUB)
        (bin_dir / "ssh").chmod(0o755)
        ws = self.tmp / "remote_ws"
        (ws / "ilab" / "bin").mkdir(parents=True)
        (ws / "ilab" / "bin" / "python").symlink_to(sys.executable)
        (ws / "resources" / "robot_rl").mkdir(parents=True)
        src = self.tmp / "local" / "robot_rl"
        _git_repo(src)
        local_ws = self.tmp / "local_ws"
        (local_ws / "resources").mkdir(parents=True)
        settings = {"hosts": ["box"], "user": "u", "workspace": str(ws), "scratch": str(self.tmp / "scratch")}
        self.pool = Pool("larg", "ssh", settings)
        self.patches = [
            mock.patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ['PATH']}"}),
            mock.patch.object(ls, "STORE", self.tmp / "leases.json"),
            mock.patch.object(ev, "CKPT_CACHE", self.tmp / "cache" / "checkpoints"),
            mock.patch.object(ev, "wandb_env", return_value={}),
            mock.patch.object(ev, "local_workspace", return_value=str(local_ws)),
            mock.patch.object(ev, "local_code", return_value={"robot_rl": str(src)}),
        ]
        for p in self.patches:
            p.start()
        self.nproc = os.cpu_count() or 1

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        subprocess.run(["chmod", "-R", "u+w", str(self.tmp)], check=False)
        shutil.rmtree(self.tmp)

    def _eval(self, *argv: str, pools: list | None = None) -> tuple[object, str]:
        args = ev.parse_args(["--cmd", "python", "--holder", "me", *argv])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out), self.assertRaises(SystemExit) as cm:
            ev.cmd_eval(args, pools or [self.pool], _no_card_claim)
        return cm.exception.code, out.getvalue()

    def _leases(self) -> list:
        with ls.locked_store() as leases:
            return list(leases)

    def test_runs_on_its_cores_with_no_gpu_and_no_card_lease(self) -> None:
        rc, out = self._eval("--on", "box", "--cpus", "1", "--", "-c", PROBE)
        self.assertEqual(rc, 0, out)
        self.assertIn(f"AFFINITY [{self.nproc - 1}]", out, "pinned to the box's last core")
        self.assertIn("CVD ''", out)
        self.assertIn("OMP 1 1 1 1", out)
        self.assertIn("DEV -", out)
        self.assertGreaterEqual(int(next(x.split()[1] for x in out.splitlines() if x.startswith("NICE "))), 10)
        self.assertIn("no GPU", out)
        self.assertEqual(self._leases(), [], "the CPU lease is released when the run ends")

    def test_detached_holds_a_cpu_lease_until_stopped(self) -> None:
        rc, out = self._eval("--on", "box", "--cpus", "1", "--detach", "--", "-c", "import time; time.sleep(120)")
        self.assertEqual(rc, 0, out)
        [lease] = self._leases()
        self.assertTrue(ls.is_cpu(lease))
        self.assertEqual((lease.card, lease.cpus, lease.report, lease.owner), ("box:cpu", 1, "larg/box", "u"))
        self.assertTrue(lease.stage, "the lease records the run's stage")
        self.assertIn(f"pls run --stop {lease.id} --holder me", out)
        pid = int(Path(lease.stage, "pid").read_text())
        for _ in range(50):
            if res.cpu_run_states([lease], [self.pool]) == {"larg/box": set()}:
                break
            time.sleep(0.1)
        self.assertEqual(res.cpu_run_states([lease], [self.pool]), {"larg/box": set()}, "alive on the box")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(ev.cmd_stop(lease.id, "me", [self.pool]), 0, err.getvalue())
        self.assertFalse(_alive(pid))
        self.assertEqual(self._leases(), [])

    def test_status_releases_a_detached_run_that_ended(self) -> None:
        rc, out = self._eval("--on", "box", "--cpus", "1", "--detach", "--", "-c", "print('done')")
        self.assertEqual(rc, 0, out)
        [lease] = self._leases()
        for _ in range(100):
            if res.cpu_run_states([lease], [self.pool]).get("larg/box"):
                break
            time.sleep(0.1)
        args = argparse.Namespace(pool=None, free=False, json=False)
        with (
            mock.patch.object(res, "probe_all", return_value=[Report("larg/box", "ssh", [], owner="u")]),
            mock.patch.object(res, "lease_windows", return_value={"grace_min": 30, "idle_min": 30}),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            res.cmd_status(args, [self.pool])
        self.assertIn("its run ended", err.getvalue())
        self.assertEqual(self._leases(), [])

    def test_budget_counts_your_cpu_leases_on_the_box(self) -> None:
        with ls.locked_store() as leases:
            leases.append(cpu_lease(cpus=3, cores="5-7", stage=str(self.tmp / "live")))
        with mock.patch.object(ev, "box_cpu_state", return_value=(8, set())):
            rc, _ = self._eval("--on", "box", "--cpus", "2", "--", "-c", "pass")
            self.assertIn("over its budget of 4 (half its 8 cores", str(rc))
            self.assertEqual(len(self._leases()), 1, "nothing claimed")
            self.pool.settings["cpu_budget"] = 5
            ns = argparse.Namespace(on="box", cpus=2, holder="me", note="")
            _, _, lease = ev.claim_cpus(ns, [self.pool])
        self.assertEqual(lease.cores, "3-4", "the cores the other run holds are skipped")

    def test_an_ended_run_frees_its_budget(self) -> None:
        with ls.locked_store() as leases:
            leases.append(cpu_lease(cpus=4, cores="4-7", stage="/gone"))
        with (
            mock.patch.object(ev, "box_cpu_state", return_value=(8, {"/gone"})),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            _, _, lease = ev.claim_cpus(argparse.Namespace(on="box", cpus=4, holder="me", note=""), [self.pool])
        self.assertEqual([x.id for x in self._leases()], [lease.id])

    def test_device_placeholder_is_refused_before_any_claim(self) -> None:
        with mock.patch.object(ev, "claim_cpus") as claim:
            rc, _ = self._eval("--on", "box", "--cpus", "1", "--", "--device", "{DEVICE}")
        self.assertIn("{DEVICE}", str(rc))
        claim.assert_not_called()
        self.assertEqual(self._leases(), [])

    def test_slurm_and_ray_refuse(self) -> None:
        slurm = Pool("delta", "slurm", {"login": "l"})
        rc, _ = self._eval("--on", "gh001", "--cpus", "4", "--", "-c", "pass", pools=[self.pool, slurm])
        self.assertIn("on SLURM a job gets its CPUs from its own allocation", str(rc))
        rc, _ = self._eval("--on", "delta", "--cpus", "4", "--", "-c", "pass", pools=[self.pool, slurm])
        self.assertIn("SLURM", str(rc))
        ray = Pool("ray", "ray", {"address": "http://h:8265"})
        rc, _ = self._eval("--on", "ray", "--cpus", "4", "--", "-c", "pass", pools=[self.pool, ray])
        self.assertIn("Ray schedules", str(rc))
        self.assertEqual(self._leases(), [])

    def test_a_card_or_a_card_selector_is_refused(self) -> None:
        for argv in (("--on", "box:0"), ("--any",), ("--on", "box", "--gpus", "2")):
            rc, _ = self._eval(*argv, "--cpus", "1", "--", "-c", "pass")
            self.assertIn("--cpus", str(rc))
        self.assertEqual(self._leases(), [])

    def test_a_cpu_lease_is_not_a_card_to_run_on(self) -> None:
        with ls.locked_store() as leases:
            leases.append(cpu_lease(stage="/s"))
            lease_id = leases[0].id
        rc, _ = self._eval("--lease", lease_id, "--", "-c", "pass")
        self.assertIn("holds cores, not a card", str(rc))


if __name__ == "__main__":
    unittest.main()
