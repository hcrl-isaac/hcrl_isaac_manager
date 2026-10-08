"""Unit tests for the pool inventory's config files (run all script tests: just test-scripts)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inventory as inv

BASE = '[compute.box]\nkind = "local"\n'


class LocalOverridesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.res, self.main = tmp / "wt" / "res", tmp / "main"
        self.res.mkdir(parents=True)
        (self.main / "scripts" / "cluster" / "res").mkdir(parents=True)
        (self.res / "compute.toml").write_text(BASE)
        self.main_local = self.main / "scripts" / "cluster" / "res" / "compute.local.toml"
        self.main_local.write_text('[compute.larg]\nkind = "ssh"\nuser = "me"\n')
        for p in (
            mock.patch.object(inv, "RES_DIR", self.res),
            mock.patch.object(inv, "_main_checkout", lambda: self.main),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_a_worktree_without_one_uses_the_main_checkouts(self) -> None:
        """compute.local.toml is gitignored, so a worktree has none and would otherwise lose its ssh logins."""
        cfg = inv.load_config()
        self.assertEqual(cfg["compute"]["larg"]["user"], "me")

    def test_this_checkouts_own_wins(self) -> None:
        (self.res / "compute.local.toml").write_text('[compute.larg]\nkind = "ssh"\nuser = "mine"\n')
        self.assertEqual(inv.load_config()["compute"]["larg"]["user"], "mine")


if __name__ == "__main__":
    unittest.main()
