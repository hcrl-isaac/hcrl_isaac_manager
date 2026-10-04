"""Unit tests for the GPU probe parser (run: python3 -m unittest discover -s scripts/compute)."""

import unittest

from probe import parse_gpu_query, short_cmd

OUT = """\
0, GPU-a, NVIDIA A100 80GB PCIe, 18584, 81920, 97
1, GPU-b, NVIDIA A100 80GB PCIe, 0, 81920, 0
2, GPU-c, NVIDIA A100 80GB PCIe, 30000, 81920, 0
3, GPU-d, NVIDIA A100 80GB PCIe, 0, 81920, 0
@@APPS
GPU-a, 351842, 18570
GPU-d, 777, 2048
@@PS
 351842 sturman  01:02:03 /isaac-sim/kit/python/bin/python3 scripts/train.py --task T1-Kick-v0 --headless
    777 other    5-01:00:00 python serve.py
"""


class ParseGpuQueryTest(unittest.TestCase):
    def setUp(self):
        self.cards = parse_gpu_query(OUT, "larg", "mckennie")

    def test_states(self):
        self.assertEqual([c.state for c in self.cards], ["busy", "free", "held", "busy"])

    def test_owner_and_command(self):
        proc = self.cards[0].procs[0]
        self.assertEqual((proc.pid, proc.user, proc.elapsed, proc.cmd), (351842, "sturman", "01:02:03", "train.py T1-Kick-v0"))

    def test_memory_takes_process_total_when_card_reads_zero(self):
        self.assertEqual(self.cards[3].mem_used, 2048)

    def test_garbage_is_ignored(self):
        self.assertEqual(parse_gpu_query("No devices were found\n@@APPS\n@@PS\n", "p", "h"), [])

    def test_short_cmd(self):
        self.assertEqual(short_cmd("bash run.sh"), "bash")


if __name__ == "__main__":
    unittest.main()
