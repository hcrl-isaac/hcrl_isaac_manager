"""The workspace picker (scripts/configure_workspace.py): Isaac Lab 3 offers only source and none, source by default."""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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


if __name__ == "__main__":
    unittest.main()
