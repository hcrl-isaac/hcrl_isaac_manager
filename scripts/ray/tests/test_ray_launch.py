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
        self.root = root
        self.saved = preflight.mounted_sources, preflight._published_rel_paths

    def tearDown(self) -> None:
        preflight.mounted_sources, preflight._published_rel_paths = self.saved
        self.dir.cleanup()

    def test_finds_only_artifact_only_dirs(self) -> None:
        self.assertEqual(
            preflight.artifact_only_dirs(str(self.root), "hhlm_tasks"),
            ["hhlm_tasks/hhlm_tasks/agile/crab/style_data", "hhlm_tasks/hhlm_tasks/policies/fbcpr/t1/bfmzero_a"],
        )

    def run_main(self, published: set[str]) -> int:
        preflight.mounted_sources = lambda wt: [(str(self.root), "hhlm_tasks")]
        preflight._published_rel_paths = lambda: published
        try:
            preflight.main()
        except SystemExit as exc:
            return exc.code
        return 0

    def test_passes_when_all_published(self) -> None:
        published = {"hhlm_tasks/hhlm_tasks/agile/crab/style_data", "hhlm_tasks/hhlm_tasks/policies/fbcpr/t1/bfmzero_a"}
        self.assertEqual(self.run_main(published), 0)

    def test_fails_naming_the_missing_dir(self) -> None:
        self.assertEqual(self.run_main({"hhlm_tasks/hhlm_tasks/agile/crab/style_data"}), 1)

    def test_unreadable_wandb_warns_and_passes(self) -> None:
        preflight.mounted_sources = lambda wt: [(str(self.root), "hhlm_tasks")]

        def boom() -> set[str]:
            raise RuntimeError("no network")

        preflight._published_rel_paths = boom
        preflight.main()


if __name__ == "__main__":
    unittest.main()
