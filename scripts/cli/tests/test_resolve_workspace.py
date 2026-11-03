"""resolve_workspace.py: optional deps a user can't read are skipped, own-repo projects resolve from their own URL."""

import contextlib
import io
import sys
import tempfile
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

    def test_a_host_that_never_answers_is_unreachable(self) -> None:
        def run(cmd: list[str], timeout: float, **_kw: object) -> None:
            self.assertEqual(timeout, rw.PROBE_TIMEOUT_S)
            raise rw.subprocess.TimeoutExpired(cmd, timeout)

        with mock.patch.object(rw.subprocess, "run", run):
            self.assertFalse(rw.reachable("https://example.org/data", "main"))


CATALOG = {"org": "hcrl-isaac", "always": ["hcrl_isaaclab", "robot_rl"], "available_projects": [{"name": "hhlm"}]}
OWN = {"name": "foo", "git": "https://github.com/someone/foo_tasks.git", "ref": "dev"}
OWN_DEPS = {"foo_tasks": [{"name": "hcrl_isaaclab", "ref": "main"}, {"name": "hcrl_robots", "ref": "main"}]}


class OwnProjectTest(unittest.TestCase):
    def _manifest(self, user: dict) -> dict:
        """``load_manifest`` over ``CATALOG`` and the per-user selection ``user`` (YAML parsing stubbed)."""
        with tempfile.TemporaryDirectory() as tmp:
            defaults, selection = Path(tmp, "defaults.yaml"), Path(tmp, "workspace.yaml")
            defaults.write_text("defaults")
            selection.write_text("user")
            parsed = {"defaults": CATALOG, "user": user}
            with (
                mock.patch.object(rw, "DEFAULTS", defaults),
                mock.patch.object(rw.yaml, "safe_load", lambda text: parsed[text], create=True),
            ):
                return rw.load_manifest(selection)

    def test_an_own_project_skips_the_catalog_check_and_resolves_from_its_url(self) -> None:
        manifest = self._manifest({"projects": ["hhlm", OWN]})
        self.assertEqual(manifest["projects"], ["hhlm", "foo"])
        with mock.patch.object(rw, "_read_deps", lambda d: OWN_DEPS.get(d.name, [])):
            resolved = rw.resolve(manifest)
        self.assertEqual(resolved["foo_tasks"]["git"], OWN["git"])
        self.assertEqual(resolved["foo_tasks"]["ref"], "dev")
        self.assertEqual(resolved["hhlm_tasks"]["git"], "git@github.com:hcrl-isaac/hhlm_tasks.git")
        self.assertEqual(
            resolved["hcrl_robots"]["git"], "git@github.com:hcrl-isaac/hcrl_robots.git", "its deps resolve"
        )

    def test_a_refs_pin_still_wins_over_an_own_projects_ref(self) -> None:
        manifest = self._manifest({"projects": [OWN], "refs": {"foo_tasks": "feat/x"}})
        with mock.patch.object(rw, "_read_deps", lambda d: []):
            self.assertEqual(rw.resolve(manifest)["foo_tasks"]["ref"], "feat/x")

    def test_a_catalog_name_without_a_url_is_still_checked(self) -> None:
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
            self._manifest({"projects": [{"name": "nope"}]})
        self.assertIn("unknown project 'nope'", str(ctx.exception))

    def test_the_committed_core_leaves_hcrl_sim2real_to_the_projects_that_need_it(self) -> None:
        defaults = (Path(__file__).resolve().parents[3] / "workspace.defaults.yaml").read_text()
        always = next(line for line in defaults.splitlines() if line.startswith("always:"))
        repos = [r.strip() for r in always.split("#")[0].split("[", 1)[1].split("]")[0].split(",")]
        self.assertEqual(repos, ["hcrl_isaaclab", "robot_rl", "hcrl_robots"])
        with mock.patch.object(rw, "_read_deps", lambda d: []):
            self.assertNotIn("hcrl_sim2real", rw.resolve({"always": repos, "projects": [], "refs": {}}))


if __name__ == "__main__":
    unittest.main()
