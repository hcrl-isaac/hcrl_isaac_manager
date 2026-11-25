"""`pls run --seeds`: one detached run per (card, seed), launched at once."""

import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seeds

SCRIPT = "/m/resources/hcrl_isaaclab/scripts/train.py"
ARGS = ["--task", "T", "--seed", "7", "--run_name", "gaze=0.05"]


def _argv(*selector: str, extra: tuple[str, ...] = ()) -> list[str]:
    return ["--holder", "Latent", *selector, "--seeds", "42,43", "--detach", *extra, SCRIPT, "--", *ARGS]


class PlanTest(unittest.TestCase):
    def test_one_card_per_seed_each_with_its_seed_and_run_name(self) -> None:
        runs = seeds.plan(_argv("--on", "c102-003:105036:2,c102-003:105036:3"))
        self.assertEqual([(w, s) for w, s, _ in runs], [("c102-003:105036:2", 42), ("c102-003:105036:3", 43)])
        _, _, argv = runs[1]
        head, args = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
        self.assertEqual(head, ["--holder", "Latent", "--detach", "--on", "c102-003:105036:3", SCRIPT])
        self.assertEqual(args, ["--task", "T", "--seed", "43", "--run_name", "gaze=0.05 seed=43"])

    def test_leases_and_any_free_card_split_the_same_way(self) -> None:
        by_lease = seeds.plan(_argv("--lease", "a1,b2"))
        self.assertEqual([a[a.index("--lease") + 1] for _, _, a in by_lease], ["a1", "b2"])
        pooled = seeds.plan(_argv("--any", "--pool", "horizon"))
        self.assertTrue(all(a[:5] == ["--holder", "Latent", "--any", "--pool", "horizon"] for _, _, a in pooled))
        self.assertNotIn("--seeds", pooled[0][2])

    def test_a_run_without_a_name_is_named_by_its_seed(self) -> None:
        self.assertEqual(seeds.seeded(["--task", "T"], 5), ["--task", "T", "--seed", "5", "--run_name", "seed=5"])
        self.assertEqual(seeds.seeded(["--run_name=x", "--seed=1"], 5), ["--seed", "5", "--run_name", "x seed=5"])

    def test_launches_it_cannot_split_are_refused(self) -> None:
        bad = [
            ["--on", "a:0,b:1", "--seeds", "1,2,3", "--detach", SCRIPT, "--", "x"],  # 2 cards for 3 seeds
            ["--on", "a:0,b:1", "--seeds", "1,2", SCRIPT, "--", "x"],  # not detached
            ["--on", "a:0,b:1", "--seeds", "1,1", "--detach", SCRIPT, "--", "x"],  # repeated seed
            ["--on", "a:0", "--seeds", "1", "--detach", "--gpus", "2", SCRIPT, "--", "x"],
            ["--on", "a:0", "--seeds", "one", "--detach", SCRIPT, "--", "x"],
            ["--seeds", "1", "--detach", SCRIPT, "--", "x"],  # no card selection
        ]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                seeds.plan(argv)


class MainTest(unittest.TestCase):
    def test_every_run_launches_at_once_and_streams_with_its_seed(self) -> None:
        launched = []

        class Proc:
            def __init__(self, cmd: list[str], **kw: object) -> None:
                launched.append(cmd)
                seed = cmd[cmd.index("--seed") + 1]
                self.stdout = io.StringIO(f"[res] started detached on x (stage /s/{seed})\n")
                self.rc = 0 if seed == "42" else 3

            def wait(self) -> int:
                return self.rc

        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            rc = seeds.main(_argv("--on", "a:0,b:1"), popen=Proc)
        self.assertEqual((rc, len(launched)), (3, 2), "the worst launch status is the command's")
        out = err.getvalue()
        self.assertIn("[seed=42 a:0] [res] started detached on x (stage /s/42)", out)
        self.assertIn("[seed=43 b:1] [res] started detached on x (stage /s/43)", out)
        self.assertIn("[seeds] seed 43 (b:1): exit 3", out)


if __name__ == "__main__":
    unittest.main()
