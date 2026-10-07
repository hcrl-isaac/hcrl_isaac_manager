"""Unit tests for the pool inventory's config files (run all script tests: just test-scripts)."""

import contextlib
import io
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
        self.res, self.old = tmp / "res", tmp / "compute"
        self.res.mkdir()
        self.old.mkdir()
        (self.res / "compute.toml").write_text(BASE)
        patches = [
            mock.patch.object(inv, "RES_DIR", self.res),
            mock.patch.object(inv, "OLD_LOCAL_TOML", self.old / "compute.local.toml"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def load(self) -> tuple[dict, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cfg = inv.load_config()
        return cfg, err.getvalue()

    def test_a_leftover_override_is_read_and_named(self) -> None:
        """A compute.local.toml left at scripts/compute keeps its pools, with a message to move it."""
        (self.old / "compute.local.toml").write_text('[compute.mine]\nkind = "ssh"\n')
        cfg, err = self.load()
        self.assertIn("mine", cfg["compute"])
        self.assertIn("move it to", err)

    def test_the_new_location_wins_silently(self) -> None:
        (self.old / "compute.local.toml").write_text('[compute.stale]\nkind = "ssh"\n')
        (self.res / "compute.local.toml").write_text('[compute.mine]\nkind = "ssh"\n')
        cfg, err = self.load()
        self.assertEqual(sorted(cfg["compute"]), ["box", "mine"])
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
