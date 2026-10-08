"""`pls` (src/hcrl_cli): verb dispatch, argument pass-through, pickers, setup gating, packaging and bootstrap.sh.

Every process step is stubbed: nothing is installed, probed or launched.
"""

import os
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


class PickerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        configs = Path(self.tmp.name, "scripts/cluster/config")
        for name in ("delta", "horizon"):
            (configs / name).mkdir(parents=True)
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

    def test_a_leading_config_name_selects_the_cluster(self) -> None:
        h, asked = self._cluster(["delta", "develop"])
        self.assertEqual((h.cmd[1:], h.env, asked), (["develop"], {"CLUSTER": "delta"}, []))

    def test_bare_cluster_asks_for_the_verb_then_the_target(self) -> None:
        h, asked = self._cluster([], picks=["job", "horizon"])
        self.assertEqual((h.cmd[1:], h.env), (["job"], {"CLUSTER": "horizon"}))
        self.assertEqual(asked, ["Cluster subcommand:", "Target cluster:"])

    def test_an_exported_cluster_is_ignored_and_untargeted_verbs_ask_nothing(self) -> None:
        os.environ["CLUSTER"] = "delta"
        h, asked = self._cluster(["job"], picks=["horizon"])
        self.assertEqual((h.env, asked), ({"CLUSTER": "horizon"}, ["Target cluster:"]))
        h, asked = self._cluster(["add", "x"])
        self.assertEqual((h.cmd[1:], h.env, asked), (["add", "x"], {}, []))

    def test_bare_ray_asks_for_the_verb(self) -> None:
        with (
            mock.patch.object(infra, "handoff", _handoff),
            mock.patch.object(infra, "ask_select", return_value="list"),
            self.assertRaises(HandoffError) as ctx,
        ):
            infra.ray([])
        self.assertEqual(ctx.exception.cmd, ["scripts/ray/ray_interface.sh", "list"])


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

    def test_help_before_the_double_dash_is_ours_after_it_the_scripts(self) -> None:
        self.assertEqual(self._pls("--on", "any", "--help").cmd, [*launch.CARD_CMD, "--help"])
        with mock.patch("sys.stdout"), self.assertRaises(SystemExit) as ctx:
            cli.main(["run", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(self._pls("train", "--help").cmd[-1], "--help")

    def test_ray_job_and_run_are_refused(self) -> None:
        for verb in ("job", "run"):
            with self.subTest(verb=verb), self.assertRaises(SystemExit) as ctx:
                cli.main(["ray", verb, "--task", "T"])
            self.assertIn("pls run --on ray", str(ctx.exception.code))


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
        fn = mock.Mock(side_effect=lambda _: self.assertEqual(sys.argv[0], "pls sim2real fit", "names its usage"))
        with (
            mock.patch.dict(sys.modules, self._modules(fn)),
            mock.patch("os.chdir") as chdir,
            mock.patch.object(sys, "argv", ["pls"]),
        ):
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
