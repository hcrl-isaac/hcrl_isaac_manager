"""resolve_workspace.py: an optional (data) dependency this user can't read is skipped with a warning, never fatal."""

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import yaml  # noqa: F401
except ImportError:  # the CI test env is stdlib-only; nothing under test reads or writes YAML
    sys.modules["yaml"] = types.ModuleType("yaml")
import resolve_workspace as rw

DEPS = {
    "hhlm_tasks": [
        {"name": "hcrl_isaaclab", "ref": "main"},
        {"name": "motion_datasets", "git": "https://example.org/data", "optional": True, "note": "a copy is on box"},
    ]
}


class OptionalDepsTest(unittest.TestCase):
    def _resolve(self) -> dict:
        manifest = {"always": ["hcrl_isaaclab"], "projects": ["hhlm"], "refs": {}}
        with mock.patch.object(rw, "_read_deps", lambda d: DEPS.get(d.name, [])):
            return rw.resolve(manifest)

    def test_resolve_carries_optional_and_note(self) -> None:
        resolved = self._resolve()
        self.assertTrue(resolved["motion_datasets"]["optional"])
        self.assertEqual(resolved["motion_datasets"]["note"], "a copy is on box")
        self.assertFalse(resolved["hcrl_isaaclab"]["optional"])

    def test_an_unreachable_optional_dep_is_dropped_with_one_warning(self) -> None:
        resolved, warned, out = self._resolve(), set(), io.StringIO()
        with mock.patch.object(rw, "reachable", return_value=False) as probe, contextlib.redirect_stdout(out):
            kept = rw.drop_unreachable(resolved, warned)
            again = rw.drop_unreachable(resolved, warned)
        self.assertNotIn("motion_datasets", kept)
        self.assertEqual(set(kept), {"hcrl_isaaclab", "hhlm_tasks"}, "required repos are never dropped")
        self.assertEqual(kept, again)
        self.assertEqual(probe.call_count, 1, "only optional deps are probed, once")
        self.assertEqual(
            out.getvalue().count("WARNING motion_datasets: no access to https://example.org/data"), 1, out.getvalue()
        )
        self.assertIn("tasks that load it stop at load; a copy is on box", out.getvalue())

    def test_a_reachable_optional_dep_is_kept(self) -> None:
        with mock.patch.object(rw, "reachable", return_value=True):
            self.assertIn("motion_datasets", rw.drop_unreachable(self._resolve(), set()))

    def test_the_probe_never_prompts(self) -> None:
        seen = {}

        def run(cmd: list[str], env: dict[str, str], **_kw: object) -> types.SimpleNamespace:
            seen["cmd"], seen["env"] = cmd, env
            return types.SimpleNamespace(returncode=128)

        with mock.patch.object(rw.subprocess, "run", run):
            self.assertFalse(rw.reachable("https://example.org/data", "main"))
        self.assertEqual(seen["cmd"][:3], ["git", "ls-remote", "--exit-code"])
        self.assertEqual(seen["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertIn("BatchMode=yes", seen["env"]["GIT_SSH_COMMAND"])


if __name__ == "__main__":
    unittest.main()
