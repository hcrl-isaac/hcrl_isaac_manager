"""`pls res claim --adopt --stage <id>` records the stage's full path, so `pls run --stop` reaches the run."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import res
from inventory import Pool
from probe import Card

POOLS = [
    Pool("horizon", "slurm", {"login": "u@login", "scratch": "/scratch/h/artifacts"}),
    Pool("amd-rtx", "slurm", {"login": "u@s3", "scratch": "/work/s3/artifacts"}),
    Pool("larg-a40", "ssh", {"hosts": ["pepi"], "user": "u", "scratch": "/var/local/u/res"}),
]


def _card(pool: str, host: str = "c109-014", job: str = "106404") -> Card:
    return Card(pool, host, 0, "GB200", 6, 189471, 0, job=job, uuid="GPU-a")


class AdoptStageTest(unittest.TestCase):
    def setUp(self) -> None:
        import evaluate as ev

        def target(pool: Pool, host: str, gpu: int, job: str = "", uuid: str = "") -> ev.Target:
            return ev.Target(pool.name, pool.kind, host, gpu, "u@login", "/w", pool.settings["scratch"], job=job,
                             uuid=uuid)  # fmt: skip

        p = mock.patch.object(ev, "make_target", target)
        p.start()
        self.addCleanup(p.stop)

    def test_a_stage_id_resolves_under_the_pools_res_eval_dir(self) -> None:
        self.assertEqual(res.stage_dir("4cb0dd85", _card("horizon (horizon)"), POOLS),
                         "/scratch/h/artifacts/res-eval/4cb0dd85")  # fmt: skip
        self.assertEqual(res.stage_dir("/abs/res-eval/x", _card("horizon (horizon)"), POOLS), "/abs/res-eval/x")

    def test_cluster_and_box_labels_name_their_pools(self) -> None:
        self.assertEqual(res.pool_for_label("stampede3 (amd-rtx, stampede)", POOLS).name, "amd-rtx")
        self.assertEqual(res.pool_for_label("larg-a40/pepi", POOLS).name, "larg-a40")
        self.assertIsNone(res.pool_for_label("nowhere (x)", POOLS))


if __name__ == "__main__":
    unittest.main()
