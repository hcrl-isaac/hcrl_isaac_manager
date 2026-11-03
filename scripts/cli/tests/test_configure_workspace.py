"""The workspace picker (scripts/configure_workspace.py): Isaac Lab modes, own-repo entries, unreadable defaults."""

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
except ImportError:  # the CI test env is stdlib-only; the mode logic under test never reads or writes YAML
    sys.modules["yaml"] = types.ModuleType("yaml")
import configure_workspace as cw


class ModeTest(unittest.TestCase):
    def test_the_default_is_source_unless_the_catalog_names_none(self) -> None:
        self.assertEqual(cw._default_mode({}), "source")
        self.assertEqual(cw._default_mode({"isaaclab": {"source": False}}), "source")
        self.assertEqual(cw._default_mode({"isaaclab": {"mode": "pip"}}), "source")
        self.assertEqual(cw._default_mode({"isaaclab": {"mode": "none"}}), "none")

    def test_a_retired_saved_mode_reads_as_source(self) -> None:
        for il in ({"mode": "pip"}, {"source": False}, {"source": True}, {"mode": "source"}):
            with self.subTest(il=il):
                self.assertEqual(cw._current_mode({"isaaclab": il}, {}), "source")
        self.assertEqual(cw._current_mode({"isaaclab": {"mode": "none"}}, {}), "none")
        self.assertEqual(cw._current_mode({}, {"isaaclab": {"mode": "none"}}), "none")

    def test_the_picker_offers_source_and_none_only(self) -> None:
        seen = {}

        class Answer:
            def __init__(self, value: object) -> None:
                self.value = value

            def ask(self) -> object:
                return self.value

        def select(_prompt: str, choices: list, default: str) -> Answer:
            seen["modes"], seen["default"] = [c.value for c in choices], default
            return Answer(default)

        fake = types.SimpleNamespace(
            Choice=lambda title, value, checked=False: types.SimpleNamespace(title=title, value=value),
            checkbox=lambda *_a, **_k: Answer([]),
            select=select,
        )
        with mock.patch.dict(sys.modules, {"questionary": fake}):
            picked = cw._prompt({"available_projects": []}, {"isaaclab": {"mode": "pip"}})
        self.assertEqual(seen, {"modes": list(cw.MODES), "default": "source"})
        self.assertEqual(picked["isaaclab"], {"mode": "source"})


class Answer:
    def __init__(self, value: object) -> None:
        self.value = value

    def ask(self) -> object:
        return self.value


OWN = {"name": "foo", "git": "https://github.com/someone/foo_tasks.git", "ref": "main"}
CATALOG = {
    "org": "hcrl-isaac",
    "available_projects": [{"name": "ssti", "default": True}, {"name": "hhlm", "default": True}, {"name": "mbrl"}],
}


class SelectionTest(unittest.TestCase):
    def _pick(self, current: dict, keep: list[str] | None = None) -> tuple[dict, list]:
        """Run the picker on ``current``, answering with the pre-checked projects (or ``keep``)."""
        offered = []

        def checkbox(_prompt: str, choices: list) -> Answer:
            offered.extend(choices)
            return Answer(keep if keep is not None else [c.value for c in choices if c.checked])

        fake = types.SimpleNamespace(
            Choice=lambda title, value, checked=False: types.SimpleNamespace(title=title, value=value, checked=checked),
            checkbox=checkbox,
            select=lambda *_a, default, **_k: Answer(default),
        )
        with mock.patch.dict(sys.modules, {"questionary": fake}):
            return cw._prompt(CATALOG, current), offered

    def test_the_picker_keeps_an_own_repo_entry_and_the_files_other_keys(self) -> None:
        current = {"projects": ["hhlm", OWN], "refs": {"hhlm_tasks": "feat/x"}, "isaaclab": {"mode": "none"}}
        picked, offered = self._pick(current)
        self.assertEqual(picked["projects"], ["hhlm", OWN])
        self.assertEqual(picked["refs"], {"hhlm_tasks": "feat/x"})
        self.assertEqual(picked["isaaclab"], {"mode": "none"})
        foo = next(c for c in offered if c.value == "foo")
        self.assertTrue(foo.checked)
        self.assertIn(OWN["git"], foo.title)

    def test_an_own_repo_entry_replaces_the_catalog_project_of_its_name(self) -> None:
        fork = {**OWN, "name": "hhlm"}
        picked, offered = self._pick({"projects": [fork]})
        self.assertEqual([c.value for c in offered].count("hhlm"), 1)
        self.assertEqual(picked["projects"], [fork])

    def test_unticking_an_own_repo_drops_it(self) -> None:
        picked, _ = self._pick({"projects": ["hhlm", OWN]}, keep=["hhlm"])
        self.assertEqual(picked["projects"], ["hhlm"])

    def test_a_new_user_gets_only_the_default_projects_they_can_read(self) -> None:
        out = io.StringIO()
        with mock.patch.object(cw, "reachable", lambda url, _ref: "hhlm" in url), contextlib.redirect_stdout(out):
            selection = cw._default_selection(CATALOG)
        self.assertEqual(selection["projects"], ["hhlm"])
        self.assertIn("leaving out default project 'ssti': no read access to git@github.com:hcrl-isaac/ssti_tasks.git",
                      out.getvalue())  # fmt: skip

    def test_a_rename_leaves_own_repo_entries_alone(self) -> None:
        with (
            mock.patch.object(cw, "OUT", mock.Mock()),
            mock.patch.object(cw.yaml, "safe_dump", lambda *_a, **_k: "", create=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            migrated = cw._migrate_renamed({"projects": ["umrl", OWN]})
        self.assertEqual(migrated["projects"], ["hhlm", OWN])


if __name__ == "__main__":
    unittest.main()
