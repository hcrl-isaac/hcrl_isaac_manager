"""collect_workspace_deps.py: the image's repo list covers the catalogue and the user's own-repo projects."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import yaml  # noqa: F401
except ImportError:  # the CI test env is stdlib-only; YAML parsing is stubbed below
    sys.modules["yaml"] = types.ModuleType("yaml")
import collect_workspace_deps as cd

PARSED = {
    "defaults": {"always": ["hcrl_isaaclab", "robot_rl"], "available_projects": [{"name": "hhlm"}]},
    "user": {"projects": ["hhlm", {"name": "foo", "git": "https://example.org/foo_tasks.git"}]},
}


class CatalogueReposTest(unittest.TestCase):
    def _repos(self, with_selection: bool) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            defaults, selection = Path(tmp, "defaults.yaml"), Path(tmp, "workspace.yaml")
            defaults.write_text("defaults")
            selection.write_text("user")
            with mock.patch.object(cd.yaml, "safe_load", lambda text: PARSED[text], create=True):
                return cd.catalogue_repos(defaults, Path(tmp, "resources"), selection if with_selection else None)

    def test_an_own_repo_project_is_added(self) -> None:
        self.assertEqual(self._repos(True), ["hcrl_isaaclab", "robot_rl", "hhlm_tasks", "foo_tasks"])

    def test_without_a_selection_only_the_catalogue(self) -> None:
        self.assertEqual(self._repos(False), ["hcrl_isaaclab", "robot_rl", "hhlm_tasks"])


if __name__ == "__main__":
    unittest.main()
