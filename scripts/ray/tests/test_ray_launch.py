"""Ray launcher: argument quoting and the artifact pre-flight (no Ray, no W&B)."""

import contextlib
import io
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

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


class TemplateExcludesTest(unittest.TestCase):
    def test_every_job_template_excludes_what_preflight_checks(self) -> None:
        tools = Path(__file__).resolve().parents[1] / "tools"
        templates = sorted(tools.glob("*job_config*.template.yaml"))
        self.assertGreaterEqual(len(templates), 3)
        for path in templates:
            excludes = {line.strip()[3:-1] for line in path.read_text().splitlines() if line.startswith('  - "')}
            with self.subTest(template=path.name):
                self.assertTrue(set(preflight.ARTIFACT_EXCLUDES) <= excludes, sorted(excludes))
                self.assertIn("**/.claude/**", excludes, "gitignored session docs ship otherwise")
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


if __name__ == "__main__":
    unittest.main()
