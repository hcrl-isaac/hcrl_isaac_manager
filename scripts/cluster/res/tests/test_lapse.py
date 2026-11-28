"""The idle-lapse countdown `pls res status` and `pls res leases` show next to an idle lease."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leases as ls

WINDOWS = {"grace_min": 30.0, "idle_min": 20.0}


def _lease(**kw: float) -> ls.Lease:
    return ls.Lease("a1", "k", "pepi:3", "larg", "Latent", **kw)


class LapseTest(unittest.TestCase):
    def test_an_unused_lease_counts_down_its_grace_and_a_used_one_its_idle_window(self) -> None:
        self.assertEqual(ls.lapses_in(_lease(idle_since=1000.0), WINDOWS, now=1600.0), 1200.0)
        self.assertEqual(ls.lapses_in(_lease(idle_since=1000.0, last_active=900.0), WINDOWS, now=1600.0), 600.0)

    def test_a_busy_lease_or_a_cpu_lease_has_no_countdown_and_it_never_goes_negative(self) -> None:
        self.assertIsNone(ls.lapses_in(_lease(last_active=900.0), WINDOWS, now=1600.0))
        self.assertIsNone(ls.lapses_in(_lease(idle_since=1000.0, cpus=4), WINDOWS, now=1600.0))
        self.assertEqual(ls.lapses_in(_lease(idle_since=1000.0), WINDOWS, now=9000.0), 0.0)


if __name__ == "__main__":
    unittest.main()
