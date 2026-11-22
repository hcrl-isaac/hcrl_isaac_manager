"""`pls new <name>`: the scaffolded task repo is its own git repo and carries the task repos' test harness."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "new_tasks.py"


class NewTasksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dest = Path(tempfile.mkdtemp()) / "demo_tasks"
        subprocess.run([sys.executable, str(SCRIPT), "demo", "--dest", str(self.dest)], check=True, capture_output=True)

    def test_it_is_a_git_repo_of_its_own_on_main(self) -> None:
        top = subprocess.run(
            ["git", "-C", str(self.dest), "rev-parse", "--show-toplevel"], capture_output=True, text=True
        )
        self.assertEqual(Path(top.stdout.strip()), self.dest, "git inside it must not reach an enclosing checkout")
        self.assertEqual((self.dest / ".git" / "HEAD").read_text().strip(), "ref: refs/heads/main")

    def test_its_gpu_smoke_runs_on_the_device_kit_launched_on(self) -> None:
        conftest = (self.dest / "tests" / "conftest.py").read_text()
        for want in ("def pytest_addoption", '"--device"', "def device():", "device=DEVICE", "_ISAACSIM.origin"):
            self.assertIn(want, conftest)
        smoke = (self.dest / "tests" / "test_smoke.py").read_text()
        self.assertIn("def test_tasks_run(sim_app, device):", smoke)
        self.assertIn("run_gpu_smoke(PACKAGE, SOURCE, device=device)", smoke)
        compile(conftest, "conftest.py", "exec")
        compile(smoke, "test_smoke.py", "exec")

    def test_its_conftest_is_the_one_every_checked_out_task_repo_carries(self) -> None:
        common = subprocess.run(
            ["git", "-C", str(SCRIPT.parent), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        resources = Path(common).parent / "resources"  # the main checkout's, also from a worktree
        repos = sorted(resources.glob("*_tasks/tests/conftest.py")) if common else []
        if not repos:
            self.skipTest("no task repo checked out (a hosted CI runner)")
        scaffolded = (self.dest / "tests" / "conftest.py").read_text()
        for conftest in repos:
            with self.subTest(repo=conftest.parents[1].name):
                self.assertEqual(scaffolded, conftest.read_text(), "update CONFTEST in new_tasks.py to match")

    def test_an_existing_dest_is_refused(self) -> None:
        out = subprocess.run([sys.executable, str(SCRIPT), "demo", "--dest", str(self.dest)], capture_output=True)
        self.assertNotEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main()
