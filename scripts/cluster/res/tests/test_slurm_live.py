"""A SLURM lease whose detached run idles its card (a --follow scorer between checkpoints) still counts as live."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leases as ls
import res
from inventory import Pool

POOLS = [Pool("horizon", "slurm", {"login": "u@login", "scratch": "/s"})]


def _lease(lid: str, stage: str) -> ls.Lease:
    return ls.Lease(lid, "k" + lid, "c1:0", "horizon (horizon)", "Latent", job="5", stage=stage, stages=[stage])


class SlurmLiveTest(unittest.TestCase):
    def setUp(self) -> None:
        import evaluate as ev

        def target(pool: Pool, host: str, gpu: int, job: str = "", uuid: str = "") -> ev.Target:
            return ev.Target(pool.name, "slurm", host, gpu, pool.settings["login"], "/w", pool.settings["scratch"],
                             job=job, uuid=uuid)  # fmt: skip

        p = mock.patch.object(ev, "make_target", target)
        p.start()
        self.addCleanup(p.stop)

    def test_a_fresh_heartbeat_is_live_and_a_stale_or_ended_one_is_not(self) -> None:
        leases = [_lease("a", "/s/res-eval/fresh"), _lease("b", "/s/res-eval/stale"), _lease("c", "/s/res-eval/done")]
        out = "/s/res-eval/done ended\n/s/res-eval/fresh 3\n/s/res-eval/stale 900\n"
        with mock.patch("follow._probe", lambda st, cmd: out):
            self.assertEqual(res.slurm_live(leases, POOLS), {"a"})

    def test_a_multi_profile_cluster_label_finds_its_pool(self) -> None:
        pools = [Pool("amd-rtx", "slurm", {"login": "u@s3", "scratch": "/w"})]
        x = ls.Lease("s", "ks", "c571-003:6", "stampede3 (amd-rtx, stampede)", "Latent", job="9", stage="/w/r/a",
                     stages=["/w/r/a"])  # fmt: skip
        with mock.patch("follow._probe", lambda st, cmd: "/w/r/a 4\n"):
            self.assertEqual(res.slurm_live([x], pools), {"s"})

    def test_an_unreachable_cluster_marks_nothing_live(self) -> None:
        def down(st: object, cmd: str) -> str:
            raise SystemExit("no master")

        with mock.patch("follow._probe", down):
            self.assertEqual(res.slurm_live([_lease("a", "/s/res-eval/fresh")], POOLS), set())


if __name__ == "__main__":
    unittest.main()
