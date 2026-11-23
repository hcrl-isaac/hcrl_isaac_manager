"""`pls run` refuses an RTX recording pass where Kit films an empty scene, before anything is staged."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev
from inventory import Pool

SLURM = Pool("horizon", "slurm", {})
LARG = Pool("larg-a100", "ssh", {})


class RenderCheckTest(unittest.TestCase):
    def refused(self, script: str, args: list[str], pool: Pool, model: str) -> bool:
        try:
            ev.refuse_unrenderable(script, args, pool, model)
        except SystemExit as exc:
            self.assertIn("render = true", str(exc.code))
            return True
        return False

    def test_a_recorder_on_a_gb200_is_refused(self) -> None:
        for script in ("video_recorder.py", "hcrl_isaaclab:scripts/video_logger.py"):
            with self.subTest(script=script):
                self.assertTrue(self.refused(script, ["--mode", "async"], SLURM, "NVIDIA GB200"))

    def test_cards_that_render_and_other_scripts_pass(self) -> None:
        self.assertFalse(self.refused("video_recorder.py", [], LARG, "NVIDIA A100 80GB PCIe"), "A100s film here")
        self.assertFalse(self.refused("train.py", [], SLURM, "NVIDIA GB200"))

    def test_a_viser_recording_runs_anywhere(self) -> None:
        for args in (["--video_format", "viser"], ["--video_format=viser"]):
            with self.subTest(args=args):
                self.assertFalse(self.refused("video_logger.py", args, SLURM, "NVIDIA GB200"))

    def test_the_pools_render_setting_decides_over_the_model(self) -> None:
        self.assertTrue(self.refused("video_recorder.py", [], Pool("x", "ssh", {"render": False}), "NVIDIA A40"))
        self.assertFalse(self.refused("video_recorder.py", [], Pool("y", "ssh", {"render": True}), "NVIDIA GB200"))


if __name__ == "__main__":
    unittest.main()
