"""`pls` (src/hcrl_cli): verb dispatch, argument pass-through, pickers, setup gating, packaging and bootstrap.sh.

Every process step is stubbed: nothing is installed, probed or launched.
"""

import os
import re
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hcrl_cli import cli, infra, launch, workspace
from hcrl_cli.proc import VENV_PY


class HandoffError(Exception):
    """Raised by the stubbed handoff so a test sees the final command instead of an exec."""

    def __init__(self, cmd: Sequence[str], env: dict[str, str] | None) -> None:
        super().__init__(cmd)
        self.cmd, self.env = list(cmd), env or {}


def _handoff(cmd: Sequence[str], *, echo: bool = False, env: dict[str, str] | None = None) -> None:
    raise HandoffError(cmd, env)


def _done(code: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout=stdout)


class PassThroughTest(unittest.TestCase):
    def _pls(self, *argv: str) -> tuple[HandoffError, mock.Mock]:
        with (
            mock.patch.object(infra, "handoff", _handoff),
            mock.patch.object(workspace, "handoff", _handoff),
            mock.patch("os.chdir") as chdir,
            self.assertRaises(HandoffError) as ctx,
        ):
            cli.main(list(argv))
        return ctx.exception, chdir

    def test_verb_args_reach_the_script_verbatim_including_a_literal_double_dash(self) -> None:
        for argv in (["--json"], ["--help"], ["claim", "--any", "--holder", "me", "--", "x"]):
            h, chdir = self._pls("res", *argv)
            self.assertEqual(h.cmd, ["python3", "scripts/cluster/res/res.py", *argv])
            chdir.assert_called_once_with(cli.ROOT)

    def test_sync_and_new_take_their_positional(self) -> None:
        self.assertEqual(self._pls("sync", "hcrl2", "--dry-run")[0].cmd[2:], ["hcrl2", "--dry-run"])
        self.assertEqual(self._pls("new", "foo")[0].cmd, [VENV_PY, "scripts/new_tasks.py", "foo"])

    def test_missing_positional_and_extra_args_are_refused(self) -> None:
        for argv in (["sync"], ["new"], ["new", "a", "b"], ["setup", "x"]):
            with self.subTest(argv=argv), mock.patch("os.chdir"), self.assertRaises(SystemExit) as ctx:
                cli.main(argv)
            self.assertNotEqual(ctx.exception.code, 0)

    def test_test_defaults_to_the_core_repo_and_clears_pythonpath(self) -> None:
        h, _ = self._pls("test")
        self.assertEqual(h.cmd[1:], ["-m", "pytest", "-m", "not gpu"])
        self.assertEqual(h.env["PYTHONPATH"], "")
        h, _ = self._pls("test", "ssti_tasks", "-k", "walk")
        self.assertEqual(h.cmd[-2:], ["-k", "walk"])

    def test_vscode_no_kit_skips_the_isaac_boot(self) -> None:
        for flag in ("no-kit", "--no-kit"):
            self.assertEqual(self._pls("vscode", flag)[0].cmd, [VENV_PY, "scripts/tools/setup_vscode.py", "--no-kit"])

    def test_no_verb_prints_help_and_an_unknown_verb_fails(self) -> None:
        for argv in ([], ["frobnicate"]):
            with self.subTest(argv=argv), mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as ctx:
                    cli.main(argv)
                self.assertEqual(ctx.exception.code, 2)

    def test_every_justfile_recipe_has_a_verb(self) -> None:
        recipes = re.findall(r"^([a-z][\w-]*)(?:\s[^:]*)?:(?!=)", (ROOT / "justfile").read_text(), re.M)
        self.assertTrue(recipes)
        self.assertEqual(sorted(set(recipes) - set(cli.VERBS)), [])


class ClusterTest(unittest.TestCase):
    """`pls cluster <name> <verb>`: SLURM profiles and ray, shared and backend-only verbs, pickers."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        configs = Path(self.tmp.name, "scripts/cluster/config")
        for name in ("delta", "horizon"):
            (configs / name).mkdir(parents=True)
            (configs / name / ".env.cluster").touch()
        (configs / "half-made").mkdir()  # no .env.cluster: not a profile
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CLUSTER", None)

    def _cluster(self, args: list[str], picks: Sequence[str] = ()) -> tuple[HandoffError, list[str]]:
        picks = list(picks)
        with (
            mock.patch.object(infra, "handoff", _handoff),
            mock.patch.object(infra, "ask_select", side_effect=lambda prompt, choices: picks.pop(0)) as ask,
            self.assertRaises(HandoffError) as ctx,
        ):
            infra.cluster(args)
        return ctx.exception, [c.args[0] for c in ask.call_args_list]

    def _refused(self, args: list[str], says: str) -> None:
        with mock.patch.object(infra, "handoff", _handoff), self.assertRaises(SystemExit) as ctx:
            infra.cluster(args)
        self.assertIn(says, str(ctx.exception.code))

    def test_a_profile_and_its_verb_reach_the_slurm_backend(self) -> None:
        h, asked = self._cluster(["delta", "develop", "start"])
        self.assertEqual((h.cmd, h.env, asked), ([infra.SLURM_BACKEND, "develop", "start"], {"CLUSTER": "delta"}, []))
        h, _ = self._cluster(["horizon", "logs", "123", "-f"])
        self.assertEqual((h.cmd[1:], h.env), (["logs", "123", "-f"], {"CLUSTER": "horizon"}))

    def test_ray_takes_the_shared_verbs_from_its_own_backend(self) -> None:
        for verb in ("setup", "list", "logs", "stop", "bench"):
            with self.subTest(verb=verb):
                h, _ = self._cluster(["ray", verb, "x"])
                self.assertEqual((h.cmd, h.env), ([infra.RAY_BACKEND, verb, "x"], {}))

    def test_status_is_the_resource_probe_of_that_cluster(self) -> None:
        for name in ("horizon", "ray"):
            h, _ = self._cluster([name, "status", "--json"])
            self.assertEqual(h.cmd, ["python3", "scripts/cluster/res/res.py", "status", "--pool", name, "--json"])

    def test_bare_cluster_asks_for_the_cluster_then_the_verb(self) -> None:
        h, asked = self._cluster([], picks=["horizon", "list"])
        self.assertEqual((h.cmd[1:], h.env, asked), (["list"], {"CLUSTER": "horizon"}, ["Cluster:", "horizon:"]))

    def test_an_exported_cluster_is_ignored(self) -> None:
        os.environ["CLUSTER"] = "delta"
        h, _ = self._cluster(["horizon", "list"])
        self.assertEqual(h.env, {"CLUSTER": "horizon"})

    def test_add_creates_or_updates_a_profile(self) -> None:
        h, _ = self._cluster(["add", "x"])
        self.assertEqual((h.cmd, h.env), ([infra.SLURM_BACKEND, "add", "x"], {}))
        h, _ = self._cluster(["delta", "add", "--update"])
        self.assertEqual(h.cmd, [infra.SLURM_BACKEND, "add", "--update", "delta"])

    def test_backend_only_verbs_and_launches_are_refused_with_the_way(self) -> None:
        self._refused(["ray", "develop"], "SLURM clusters only")
        self._refused(["ray", "add"], "SLURM clusters only")
        self._refused(["delta", "bench"], "RAY clusters only")
        self._refused(["delta", "job", "--task", "T"], "pls run --on delta --batch")
        self._refused(["ray", "run", "x.py"], "pls run --on ray --")
        self._refused(["delta", "frobnicate"], "unknown verb")
        self._refused(["nope", "list"], "no cluster 'nope'")
        self._refused(["half-made", "list"], "no cluster 'half-made'")

    def test_a_worktree_without_profiles_uses_the_main_checkouts(self) -> None:
        main = Path(self.tmp.name, "main")
        (main / "scripts/cluster/config/stampede").mkdir(parents=True)
        (main / "scripts/cluster/config/stampede/.env.cluster").touch()
        Path(self.tmp.name, "wt").mkdir()
        os.chdir(Path(self.tmp.name, "wt"))
        found = subprocess.CompletedProcess([], 0, stdout=f"{main}/.git\n")
        with mock.patch.object(infra.subprocess, "run", return_value=found):
            self.assertEqual(infra.profiles(), ["stampede"])


class BatchTest(unittest.TestCase):
    """`pls run --on <cluster> --batch`: a SLURM batch job on a staged tree."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        Path(self.tmp.name, "scripts/cluster/config/delta").mkdir(parents=True)
        Path(self.tmp.name, "scripts/cluster/config/delta/.env.cluster").touch()
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)
        fake = types.ModuleType("worktree_env")
        fake.resolve = mock.Mock(
            return_value=({"robot_rl": "/w/robot_rl", "hhlm_tasks": "/m/hhlm_tasks"}, ["robot_rl"])
        )
        self.run = mock.Mock(return_value=_done())
        for patch in (
            mock.patch.dict(sys.modules, {"worktree_env": fake}),
            mock.patch.dict(os.environ, {"WT": "stale"}),
            mock.patch.object(launch, "handoff", _handoff),
            mock.patch.object(launch.proc, "run", self.run),
            mock.patch("os.chdir"),
            mock.patch("sys.stderr"),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _pls(self, *argv: str) -> HandoffError:
        with self.assertRaises(HandoffError) as ctx:
            cli.main(["run", *argv])
        return ctx.exception

    def _refused(self, *argv: str) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["run", *argv])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_batch_submits_train_on_the_profile(self) -> None:
        h = self._pls("--on", "delta", "--batch", "--", "train", "--task", "T")
        self.assertEqual((h.cmd, h.env), ([launch.SLURM_BACKEND, "job", "--task", "T"], {"CLUSTER": "delta"}))
        h = self._pls("--on", "delta", "--batch", "--tree", "nightly", "--", "train")
        self.assertEqual(h.cmd, [launch.SLURM_BACKEND, "job", "--tree", "nightly"])
        self.run.assert_not_called()

    def test_a_worktree_set_is_staged_as_its_own_tree_first(self) -> None:
        h = self._pls("--on", "delta", "--batch", "--wt", "feat", "--", "train", "--task", "T")
        self.run.assert_called_once_with(
            ["bash", launch.DEV_BACKEND, "stage", "feat", "robot_rl=/w/robot_rl"], env={"CLUSTER": "delta"}
        )
        self.assertEqual(h.cmd, [launch.SLURM_BACKEND, "job", "--tree", "feat", "--task", "T"])

    def test_batch_needs_a_profile_train_and_no_card_options(self) -> None:
        self._refused("--on", "ray", "--batch", "--", "train")
        self._refused("--on", "nope", "--batch", "--", "train")
        self._refused("--on", "delta", "--batch", "--", "census")
        self._refused("--on", "delta", "--batch", "--cmd", "--", "nvidia-smi")
        self._refused("--on", "delta", "--batch", "--holder", "me", "--", "train")
        self._refused("--on", "delta", "--batch", "--tree", "t", "--wt", "feat", "--", "train")
        self._refused("--on", "delta", "--tree", "t", "--", "train")


class RunTest(unittest.TestCase):
    """`pls run`: local runs, --wt, and the card and Ray targets."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        Path(self.tmp.name, "resources", "robot_rl").mkdir(parents=True)
        root = mock.patch.object(launch, "ROOT", Path(self.tmp.name))
        root.start()
        self.addCleanup(root.stop)
        self.select = mock.Mock(return_value=("", "/m/resources/hcrl_isaaclab", []))
        fake = types.ModuleType("worktree_env")
        fake.select = self.select
        fake.resolve = mock.Mock(return_value=({"robot_rl": "/w/robot_rl"}, ["robot_rl"]))
        Path(self.tmp.name, "probe.py").touch()
        cwd = os.getcwd()
        os.chdir(self.tmp.name)  # where `pls` is started: local files resolve against it
        self.addCleanup(os.chdir, cwd)
        for patch in (
            mock.patch.dict(sys.modules, {"worktree_env": fake}),
            mock.patch.dict(os.environ, {"PYTHONPATH": "/x", "WT": "stale"}),
            mock.patch.object(infra, "handoff", _handoff),
            mock.patch.object(launch, "handoff", _handoff),
            mock.patch("sys.stderr"),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.chdir = mock.patch("os.chdir").start()
        self.addCleanup(mock.patch.stopall)

    def _pls(self, *argv: str) -> HandoffError:
        with self.assertRaises(HandoffError) as ctx:
            cli.main(["run", *argv])
        return ctx.exception

    def _refused(self, *argv: str) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["run", *argv])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_a_bare_script_runs_here_on_the_main_checkouts(self) -> None:
        h = self._pls("train", "--task", "T", "--", "x")
        self.assertEqual(h.cmd, [VENV_PY, "/m/resources/hcrl_isaaclab/scripts/train.py", "--task", "T", "--", "x"])
        self.assertEqual((h.env["PYTHONPATH"], h.env["OMNI_KIT_ACCEPT_EULA"]), ("/x", "YES"))
        self.select.assert_called_once_with("")  # an exported WT selects nothing

    def test_wt_flag_selects_the_set_ahead_of_the_existing_pythonpath(self) -> None:
        self.select.return_value = ("/w/a:/w/b", "/w/core", ["a", "b"])
        h = self._pls("--wt", "feat", "--", "train")
        self.select.assert_called_once_with("feat")
        self.assertEqual((h.env["PYTHONPATH"], h.cmd[1]), ("/w/a:/w/b:/x", "/w/core/scripts/train.py"))

    def test_an_unknown_set_stops_before_launch(self) -> None:
        self.select.side_effect = SystemExit("[worktree] no repo has worktrees/nope; nothing to select")
        self._refused("--wt", "nope", "--", "train")

    def test_options_need_the_double_dash_and_card_options_need_a_card(self) -> None:
        self._refused("--wt", "feat", "train")
        self._refused("--holder", "me", "--", "train")
        self._refused("--on", "ray", "--holder", "me", "--", "train")
        self._refused("--on", "ray", "--cmd", "--", "nvidia-smi")
        self._refused("--on", "any", "--")

    def test_repo_and_local_files_run_here_as_on_a_card(self) -> None:
        h = self._pls("--wt", "feat", "--", "robot_rl/scripts/census.py", "--n", "1")
        self.assertEqual(h.cmd, [VENV_PY, "/w/robot_rl/scripts/census.py", "--n", "1"])
        self.assertEqual(self._pls("--", "robot_rl:scripts/x.py").cmd[1], "/w/robot_rl/scripts/x.py")
        self.assertEqual(
            self._pls("./probe.py", "a").cmd, [VENV_PY, str(Path(self.tmp.name, "probe.py").resolve()), "a"]
        )
        self._refused("missing.py")

    def test_a_command_runs_here_from_the_callers_cwd_with_the_venv_first(self) -> None:
        h = self._pls("--wt", "feat", "--cmd", "--", "nvidia-smi", "-L")
        self.assertEqual(h.cmd, ["nvidia-smi", "-L"])
        self.assertTrue(h.env["PATH"].startswith(f"{launch.ROOT / 'ilab'}/bin:"))
        self.chdir.assert_called_with(os.getcwd())
        self.select.assert_called_once_with("feat")

    def test_a_command_on_a_card_is_marked_for_the_backend(self) -> None:
        h = self._pls("--on", "any", "--holder", "me", "--cmd", "--", "bash", "-c", "nvidia-smi")
        self.assertEqual(
            h.cmd, [*launch.CARD_CMD, "--holder", "me", "--any", "--cmd", "bash", "--", "-c", "nvidia-smi"]
        )

    def test_card_targets_become_the_card_backends_selectors(self) -> None:
        for on, flags in (
            ("c571-003:3", ["--on", "c571-003:3"]),
            ("local:0", ["--on", "local:0"]),
            ("any", ["--any"]),
            ("lease:ab12cd", ["--lease", "ab12cd"]),
            ("delta", ["--any", "--pool", "delta"]),
        ):
            with self.subTest(on=on):
                h = self._pls("--on", on, "--holder", "me", "--checkpoint", "A=x", "--", "train", "--task", "T")
                expect = [*launch.CARD_CMD, "--holder", "me", "--checkpoint", "A=x", *flags]
                self.assertEqual(h.cmd, [*expect, "hcrl_isaaclab:scripts/train.py", "--", "--task", "T"])

    def test_card_scripts_name_a_repo_file_or_a_local_one(self) -> None:
        for script, shipped in (
            ("robot_rl/scripts/census.py", "robot_rl:scripts/census.py"),
            ("robot_rl:scripts/census.py", "robot_rl:scripts/census.py"),
            ("./probe.py", str(Path(self.tmp.name, "probe.py").resolve())),
            ("probe.py", str(Path(self.tmp.name, "probe.py").resolve())),
        ):
            with self.subTest(script=script):
                h = self._pls("--on", "any", "--holder", "me", "--wt", "feat", "--", script)
                self.assertEqual(h.cmd[-4:], ["--wt", "feat", shipped, "--"])

    def test_ray_trains_through_job_and_ships_other_scripts_through_run(self) -> None:
        h = self._pls("--on", "ray", "--", "train", "--task", "T")
        self.assertEqual((h.cmd, h.env), ([launch.RAY_BACKEND, "job", "--task", "T"], {}))
        h = self._pls("--on", "ray", "--wt", "feat", "--", "census", "--n", "2")
        self.assertEqual(h.cmd, [launch.RAY_BACKEND, "run", "hcrl_isaaclab/scripts/census.py", "--n", "2"])
        self.assertEqual(h.env, {"WT": "feat"})
        h = self._pls("--on", "ray", "--", "robot_rl:scripts/census.py")
        self.assertEqual(h.cmd[1:3], ["run", "robot_rl/scripts/census.py"])
        self._refused("--on", "ray", "--", "./probe.py")
        h = self._pls("--on", "ray", "--distributed", "--", "train", "--task", "T")
        self.assertEqual(h.cmd, [launch.RAY_BACKEND, "job_distributed", "--task", "T"])
        self._refused("--on", "ray", "--distributed", "--", "census")
        self._refused("--on", "any", "--holder", "me", "--distributed", "--", "train")

    def test_help_before_the_double_dash_is_ours_after_it_the_scripts(self) -> None:
        self.assertEqual(self._pls("--on", "any", "--help").cmd, [*launch.CARD_CMD, "--help"])
        with mock.patch("sys.stdout"), self.assertRaises(SystemExit) as ctx:
            cli.main(["run", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(self._pls("train", "--help").cmd[-1], "--help")

    def test_pls_ray_is_gone(self) -> None:
        with mock.patch("sys.stdout"), self.assertRaises(SystemExit) as ctx:
            cli.main(["ray", "list"])
        self.assertEqual(ctx.exception.code, 2)


class WorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.tmp)], check=True))
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)

    def test_isaaclab_mode_reads_the_first_mode_line(self) -> None:
        Path("workspace.yaml").write_text("isaaclab:\n  # mode: none\n  mode: source  # editable\n  mode: pip\n")
        self.assertEqual(workspace._isaaclab_mode(), "source")

    def _setup(self, mode: str, failing: Sequence[str] = ()) -> tuple[list[list[str]], mock.Mock, str]:
        Path("workspace.yaml").write_text(f"isaaclab:\n  mode: {mode}\n")
        Path("resources/robot_rl").mkdir(parents=True)
        Path("resources/robot_rl/pyproject.toml").touch()
        calls = []

        def run(cmd: Sequence[str], **kw: object) -> subprocess.CompletedProcess:
            calls.append(list(cmd))
            failed = any(f in " ".join(cmd) for f in failing)
            return _done(1 if failed else 0, stdout="")

        with (
            mock.patch.object(workspace, "deps"),
            mock.patch.object(workspace, "resolve"),
            mock.patch.object(workspace, "vscode") as vscode,
            mock.patch.object(workspace, "run", side_effect=run),
            mock.patch("sys.stderr") as err,
            mock.patch("sys.stdout"),
        ):
            workspace.setup()
        installs = [c for c in calls if c[:3] == ["uv", "pip", "install"]]
        return installs, vscode, "".join(str(c) for c in err.write.call_args_list)

    def test_mode_none_fetches_repos_and_installs_nothing(self) -> None:
        installs, vscode, _ = self._setup("none")
        self.assertEqual(installs, [])
        vscode.assert_not_called()

    def test_a_failed_install_does_not_stop_the_rest_and_is_reported(self) -> None:
        installs, vscode, err = self._setup("pip", failing=("torch==",))
        self.assertIn("resources/robot_rl", installs[-1])
        vscode.assert_called_once()
        self.assertIn("torch", err)

    def test_rc_hook_is_added_once_and_clean_removes_it(self) -> None:
        rc = self.tmp / ".bashrc"
        rc.write_text("export A=1\n")
        with (
            mock.patch.object(workspace, "RC_FILE", rc),
            mock.patch.object(workspace, "run", return_value=_done()),
            mock.patch.object(workspace.shutil, "which", return_value="/usr/bin/x"),
            mock.patch.object(workspace, "WANDB_ENV", self.tmp / "env"),
            mock.patch("sys.stdout"),
        ):
            (self.tmp / "env").touch()
            workspace.deps()
            workspace.deps()
            self.assertEqual(rc.read_text().count(workspace.RC_LINE), 1)
            with mock.patch.object(workspace, "ROOT", self.tmp):
                workspace.clean()
        self.assertEqual(rc.read_text(), "export A=1\n")

    def test_upload_artifacts_needs_the_wandb_env_and_keeps_keys_off_argv(self) -> None:
        with self.assertRaises(SystemExit):
            infra.upload_artifacts(["--list"])
        Path("scripts").mkdir()
        Path("scripts/.env.wandb").write_text("WANDB_API_KEY=secret\n")
        with mock.patch.object(infra, "handoff", _handoff), self.assertRaises(HandoffError) as ctx:
            infra.upload_artifacts(["--list"])
        self.assertEqual(ctx.exception.cmd[0:2], ["bash", "-c"])
        self.assertNotIn("secret", " ".join(ctx.exception.cmd))
        self.assertEqual(ctx.exception.cmd[-1], "--list")


class Sim2realTest(unittest.TestCase):
    def _modules(self, fn: Callable) -> dict[str, types.ModuleType]:
        pkg = types.ModuleType("hcrl_sim2real")
        cli_mod = types.ModuleType("hcrl_sim2real.cli")
        cli_mod.__doc__ = "sim2real commands"
        cli_mod.COMMANDS = {"fit": ("hcrl_sim2real.fit_mod", "main")}
        target = types.ModuleType("hcrl_sim2real.fit_mod")
        target.main = fn
        return {"hcrl_sim2real": pkg, "hcrl_sim2real.cli": cli_mod, "hcrl_sim2real.fit_mod": target}

    def test_a_command_runs_from_the_registry_in_the_callers_cwd(self) -> None:
        fn = mock.Mock()
        with mock.patch.dict(sys.modules, self._modules(fn)), mock.patch("os.chdir") as chdir:
            cli.main(["sim2real", "fit", "runs/a", "--plot"])
        fn.assert_called_once_with(["runs/a", "--plot"])
        chdir.assert_not_called()

    def test_unknown_or_help_prints_the_registry_doc(self) -> None:
        with mock.patch.dict(sys.modules, self._modules(mock.Mock())), mock.patch("sys.stdout"):
            for argv, code in ((["--help"], 0), (["nope"], 1), ([], 1)):
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as ctx:
                    cli.main(["sim2real", *argv])
                self.assertEqual(ctx.exception.code, code)

    def test_missing_package_says_how_to_install_it(self) -> None:
        with mock.patch.dict(sys.modules, {"hcrl_sim2real": None}), self.assertRaises(SystemExit) as ctx:
            cli.main(["sim2real", "fit"])
        self.assertIn("pls setup", str(ctx.exception.code))


class PackagingTest(unittest.TestCase):
    def test_pls_is_the_console_script_of_the_src_package(self) -> None:
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(meta["project"]["scripts"], {"pls": "hcrl_cli.cli:main"})
        self.assertEqual(meta["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"], ["src/hcrl_cli"])

    def test_bootstrap_hands_off_to_the_uninstalled_package(self) -> None:
        with tempfile.TemporaryDirectory() as stub:
            uv = Path(stub, "uv")
            uv.write_text('#!/bin/sh\necho "PYTHONPATH=$PYTHONPATH"\necho "$@"\n')
            uv.chmod(0o755)
            env = {"PATH": f"{stub}:/usr/bin:/bin", "HOME": stub}
            for args, verb in (([], "setup"), (["res", "pools"], "res pools")):
                out = subprocess.run(
                    ["sh", str(ROOT / "bootstrap.sh"), *args], env=env, capture_output=True, text=True, check=True
                ).stdout.splitlines()
                self.assertEqual(out[0], f"PYTHONPATH={ROOT}/src")
                self.assertEqual(out[1], f"run --no-project --python 3.11 python -m hcrl_cli {verb}")


if __name__ == "__main__":
    unittest.main()
