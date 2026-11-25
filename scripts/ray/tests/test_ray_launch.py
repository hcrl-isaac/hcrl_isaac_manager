"""Ray launcher: argument quoting and the artifact pre-flight (no Ray, no W&B)."""

import contextlib
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import preflight
from job_args import split_jobs
from worktree_env import workspace_repos


class SplitJobsTest(unittest.TestCase):
    def test_argument_with_spaces_stays_one_argument(self) -> None:
        tokens = ["ray/wrap_resources.py", "--task", "T1-Kick-v0", "--run_group", "push foot contact"]
        (job,) = split_jobs(tokens)
        self.assertEqual(shlex.split(job), tokens)

    def test_quotes_and_shell_characters_survive(self) -> None:
        tokens = ["train.py", "--run_name", 'a\'b "c" $HOME;rm']
        self.assertEqual(shlex.split(split_jobs(tokens)[0]), tokens)

    def test_standalone_star_separates_jobs(self) -> None:
        jobs = split_jobs(["a.py", "--x", "1", "*", "b.py", "--y", "a*b"])
        self.assertEqual([shlex.split(j) for j in jobs], [["a.py", "--x", "1"], ["b.py", "--y", "a*b"]])

    def test_empty(self) -> None:
        self.assertEqual(split_jobs([]), [])


class WorkspaceReposTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.resources = Path(self.dir.name) / "resources"
        for repo in ("hhlm_tasks", "umrl_tasks", "hcrl_robots"):
            (self.resources / repo).mkdir(parents=True)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_checkouts_missing_from_gitman_are_not_mounted(self) -> None:
        names = ["hcrl_isaaclab", "robot_rl", "hhlm_tasks", "hcrl_robots"]
        (self.resources.parent / "gitman.yaml").write_text("".join(f"- name: {n}\n  rev: main\n" for n in names))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(workspace_repos(str(self.resources)), names)
        self.assertIn("umrl_tasks", err.getvalue())

    def test_without_gitman_every_checkout_counts(self) -> None:
        self.assertIn("umrl_tasks", workspace_repos(str(self.resources)))


class FileMountsTest(unittest.TestCase):
    def test_isaaclab_source_mounts_keep_the_checkout_layout(self) -> None:
        """Isaac Lab finds its root only as <root>/source/isaaclab/isaaclab, so its packages mount under one root."""
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            (ws / "workspace.yaml").write_text("isaaclab:\n  mode: source\n")
            for pkg in ("isaaclab", "isaaclab_rl"):
                (ws / "resources" / "IsaacLab" / "source" / pkg).mkdir(parents=True)
                (ws / "resources" / "IsaacLab" / "source" / pkg / "setup.py").write_text("")
            (ws / "resources" / "IsaacLab" / "apps").mkdir()
            (ws / "resources" / "robot_rl").mkdir()
            (ws / "resources" / "robot_rl" / "setup.py").write_text("")
            script = Path(__file__).resolve().parents[1] / "build_file_mounts.py"
            out = subprocess.run(
                [sys.executable, str(script)], env={**os.environ, "WORKSPACE_DIR": d}, capture_output=True, text=True
            ).stdout
        self.assertIn('/resources/IsaacLab/source/isaaclab": "/workspace/IsaacLab/source/isaaclab"', out)
        self.assertIn('/resources/IsaacLab/source/isaaclab_rl": "/workspace/IsaacLab/source/isaaclab_rl"', out)
        self.assertIn('/resources/IsaacLab/apps": "/workspace/IsaacLab/apps"', out)
        self.assertIn('/resources/robot_rl": "/workspace/ext/robot_rl"', out)
        self.assertNotIn("/workspace/ext/isaaclab", out)


class SubmitConfigTest(unittest.TestCase):
    def test_each_submit_reads_its_own_rendered_config(self) -> None:
        """A shared rendered config can be re-rendered by a concurrent submit (another WT) before it is read."""
        import re

        sh = (Path(__file__).resolve().parents[1] / "ray_interface.sh").read_text()
        self.assertIn('CONFIG_DIR="$(mktemp -d', sh)
        reads = re.findall(r"job_config=([^\s]+)|--job_config (\S+)", sh)
        paths = [a or b for a, b in reads if (a or b) != "$job_config"]
        self.assertTrue(paths)
        for path in paths:
            self.assertIn("CONFIG_DIR", path, f"{path} reads the shared rendered config")


class BusyTest(unittest.TestCase):
    JOBS = [
        {"submission_id": "raysubmit_a", "status": "RUNNING", "start_time": 1_700_000_000_000,
         "metadata": {"user_id": "me", "wt": "bl-fz-kick"}, "entrypoint": "x --task hhlm/T1-Kick-v0 --seed 1"},
        {"submission_id": "raysubmit_b", "status": "SUCCEEDED", "metadata": {"user_id": "me"}},
        {"submission_id": "raysubmit_c", "status": "PENDING", "metadata": {"user_id": "someone"}},
    ]  # fmt: skip

    def _main(self, env: dict) -> tuple[int, str]:
        import busy

        resp = mock.MagicMock()
        resp.__enter__.return_value = io.StringIO(json.dumps(self.JOBS))
        err = io.StringIO()
        with (
            mock.patch.object(busy.urllib.request, "urlopen", return_value=resp),
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(sys, "argv", ["busy.py", "http://ray:8265", "me"]),
            contextlib.redirect_stderr(err),
        ):
            try:
                busy.main()
            except SystemExit as exc:
                return int(exc.code or 0), err.getvalue()
        return 0, err.getvalue()

    def test_a_running_job_of_this_user_refuses_the_submit_naming_it(self) -> None:
        os.environ.pop("RAY_ALONGSIDE", None)
        rc, err = self._main({})
        self.assertEqual(rc, 1)
        self.assertIn("raysubmit_a RUNNING", err)
        self.assertIn("--wt bl-fz-kick, task hhlm/T1-Kick-v0", err)
        self.assertNotIn("raysubmit_b", err, "a finished job is no obstacle")
        self.assertNotIn("raysubmit_c", err, "another user's job is not this check's business")

    def test_alongside_submits_anyway(self) -> None:
        self.assertEqual(self._main({"RAY_ALONGSIDE": "1"})[0], 0)

    def test_an_unreadable_cluster_warns_and_lets_the_submit_through(self) -> None:
        import busy

        err = io.StringIO()
        with (
            mock.patch.object(busy.urllib.request, "urlopen", side_effect=OSError("refused")),
            mock.patch.object(sys, "argv", ["busy.py", "http://ray:8265", "me"]),
            contextlib.redirect_stderr(err),
        ):
            busy.main()
        self.assertIn("not checking", err.getvalue())


class TemplateExcludesTest(unittest.TestCase):
    def test_every_job_template_excludes_what_preflight_checks(self) -> None:
        tools = Path(__file__).resolve().parents[1] / "tools"
        templates = sorted(tools.glob("*job_config*.template.yaml"))
        self.assertGreaterEqual(len(templates), 3)
        for path in templates:
            excludes = {line.strip()[3:-1] for line in path.read_text().splitlines() if line.startswith('  - "')}
            with self.subTest(template=path.name):
                self.assertTrue(set(preflight.ARTIFACT_EXCLUDES) <= excludes, sorted(excludes))
                self.assertTrue(set(preflight.UPLOAD_EXCLUDES) <= excludes, sorted(excludes))
                self.assertFalse(any(e.startswith("**/.git/") for e in excludes), "a partial .git still ships")
                self.assertFalse(any("bfmzero" in e for e in excludes))


class PreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name) / "hhlm_tasks"
        for d in (
            "hhlm_tasks/policies/fbcpr/t1/bfmzero_a",
            "hhlm_tasks/policies/fbcpr/t1/bfmzero_a/nested",
            "hhlm_tasks/policies/cvae/t1/cvae_bfm_x",
            "hhlm_tasks/agile/crab/style_data",
            "worktrees/wt/hhlm_tasks/policies/fbcpr/t1/bfmzero_wt",
            "scripts/bfmzero_not_a_policy",
        ):
            (root / d).mkdir(parents=True)
        (root / "hhlm_tasks/policies/fbcpr/g1").mkdir(parents=True)
        (root / "hhlm_tasks/policies/fbcpr/g1/buffer.hdf5").write_bytes(b"x")
        (root / "hhlm_tasks/mcp/policies").mkdir(parents=True)
        (root / "hhlm_tasks/mcp/policies/approach.pt").write_bytes(b"x")  # shallow: ships with the code
        self.root = root
        self.all = [
            "hhlm_tasks/hhlm_tasks/agile/crab/style_data",
            "hhlm_tasks/hhlm_tasks/policies/cvae/t1/cvae_bfm_x",
            "hhlm_tasks/hhlm_tasks/policies/fbcpr/g1/buffer.hdf5",
            "hhlm_tasks/hhlm_tasks/policies/fbcpr/t1/bfmzero_a",
        ]
        self.saved = preflight.mounted_sources, preflight._published_rel_paths

    def tearDown(self) -> None:
        preflight.mounted_sources, preflight._published_rel_paths = self.saved
        self.dir.cleanup()

    def test_finds_every_exported_policy_and_style_data(self) -> None:
        self.assertEqual(preflight.artifact_only_dirs(str(self.root), "hhlm_tasks"), self.all)

    def run_main(self, published: set[str]) -> int:
        preflight.mounted_sources = lambda wt: [(str(self.root), "hhlm_tasks")]
        preflight._published_rel_paths = lambda: published
        try:
            preflight.main()
        except SystemExit as exc:
            return exc.code
        return 0

    def test_passes_when_all_published(self) -> None:
        self.assertEqual(self.run_main(set(self.all)), 0)

    def test_fails_naming_the_missing_dir(self) -> None:
        self.assertEqual(self.run_main(set(self.all[:-1])), 1)

    def test_unreadable_wandb_warns_and_passes(self) -> None:
        preflight.mounted_sources = lambda wt: [(str(self.root), "hhlm_tasks")]

        def boom() -> set[str]:
            raise RuntimeError("no network")

        preflight._published_rel_paths = boom
        preflight.main()


class WorktreeDataTest(unittest.TestCase):
    """A --wt job whose artifact-only data differs from W&B's would train on W&B's copy: refused before submit."""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.wt = Path(self.dir.name) / "ssti_tasks" / "worktrees" / "momonly"
        data = self.wt / "ssti_tasks" / "agile" / "style_data"
        data.mkdir(parents=True)
        (data / "traverse.pt").write_bytes(b"140 columns")
        self.rel = "ssti_tasks/ssti_tasks/agile/style_data"
        self.saved = (preflight.mounted_sources, preflight._published_rel_paths, preflight._published_digests)
        preflight.mounted_sources = lambda wt: [(str(self.wt), "ssti_tasks")]
        preflight._published_rel_paths = lambda: {self.rel}

    def tearDown(self) -> None:
        preflight.mounted_sources, preflight._published_rel_paths, preflight._published_digests = self.saved
        self.dir.cleanup()

    def run_main(self, published_bytes: bytes, accepted: str = "") -> tuple[int, str]:
        digest = preflight.local_digests(self._file(published_bytes))
        preflight._published_digests = lambda rels: {r: ("traverse:v3", digest) for r in rels}
        err = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"WT": "momonly", "WT_DATA_OK": accepted}),
            contextlib.redirect_stderr(err),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                preflight.main()
            except SystemExit as exc:
                return exc.code, err.getvalue()
        return 0, err.getvalue()

    def _file(self, content: bytes) -> str:
        path = Path(self.dir.name) / "published.pt"
        path.write_bytes(content)
        return str(path)

    def test_differing_worktree_data_is_refused_naming_it(self) -> None:
        code, err = self.run_main(b"134 columns")
        self.assertEqual(code, 1)
        self.assertIn(self.rel, err)
        self.assertIn("traverse:v3", err)

    def test_matching_worktree_data_passes(self) -> None:
        self.assertEqual(self.run_main(b"140 columns")[0], 0)

    def test_lfs_pointers_are_refused_with_the_pull(self) -> None:
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n"
        (self.wt / "ssti_tasks" / "agile" / "style_data" / "traverse.pt").write_bytes(pointer)
        code, err = self.run_main(b"134 columns")
        self.assertEqual(code, 1)
        self.assertIn("git lfs pull", err)

    def test_a_differing_readme_alone_is_not_a_difference(self) -> None:
        data = self.wt / "ssti_tasks" / "agile" / "style_data"
        (data / "README.md").write_text("an older description")  # docs never reach the comparison
        self.assertEqual(self.run_main(b"140 columns")[0], 0)

    def test_the_refusal_names_the_opt_out(self) -> None:
        self.assertIn(f"--wt-data-ok {self.rel}", self.run_main(b"134 columns")[1])

    def test_an_accepted_entry_warns_and_passes(self) -> None:
        code, err = self.run_main(b"134 columns", accepted=self.rel)
        self.assertEqual(code, 0)
        self.assertIn("--wt-data-ok accepts it", err)

    def test_mounts_are_printed(self) -> None:
        out = io.StringIO()
        preflight._published_digests = lambda rels: {}
        with mock.patch.dict(os.environ, {"WT": "momonly"}), contextlib.redirect_stdout(out):
            preflight.main()
        self.assertIn("WT=momonly mounts ssti_tasks from worktrees/momonly", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class ExtScriptTest(unittest.TestCase):
    def test_a_repo_script_maps_to_its_worker_mount(self) -> None:
        from job_args import ext_script

        self.assertEqual(
            ext_script("hcrl_isaaclab/scripts/video_logger.py"), "/workspace/ext/hcrl_isaaclab/scripts/video_logger.py"
        )

    def test_paths_outside_a_shipped_repo_are_refused(self) -> None:
        from job_args import ext_script

        for bad in ("/abs/x.py", "hcrl_isaaclab/../x.py", "video_logger.py", "hcrl_isaaclab/scripts/run.sh", "a//b.py"):
            with self.assertRaises(ValueError, msg=bad):
                ext_script(bad)


class SubmitPreparationTest(unittest.TestCase):
    def test_every_submitting_subcommand_renders_and_syncs_first(self) -> None:
        """A subcommand that submits without re-rendering ships whatever job config another session left behind."""
        import re

        text = (Path(__file__).resolve().parents[1] / "ray_interface.sh").read_text()
        prepare = re.search(r"^\s+([\w|]+)\)\n\s+prepare_submit", text, re.M)
        self.assertIsNotNone(prepare)
        prepared = set(prepare.group(1).split("|"))
        submitting = set()
        for m in re.finditer(r"^    ([\w|]+)\)\n(.*?)^        ;;", text, re.M | re.S):
            if "submit_job.py" in m.group(2):
                submitting.update(m.group(1).split("|"))
        self.assertTrue(submitting)
        self.assertEqual(submitting - prepared, set())
