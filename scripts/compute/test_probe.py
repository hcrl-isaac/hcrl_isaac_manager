"""Unit tests for the GPU probe parser (run: python3 -m unittest discover -s scripts/compute)."""

import unittest

from probe import Card, ProbeError, Report, _gres_gpus, parse_gpu_query, ray_card, short_cmd
from res import merge_duplicates

CARDS = """\
0, GPU-a, NVIDIA A100 80GB PCIe, 18584, 81920, 97
1, GPU-b, NVIDIA A100 80GB PCIe, 0, 81920, 0
2, GPU-c, NVIDIA A100 80GB PCIe, 30000, 81920, 0
3, GPU-d, NVIDIA A100 80GB PCIe, 0, 81920, 0
4, GPU-e, NVIDIA A100 80GB PCIe, 600, 81920, 100
"""
APPS = """\
GPU-a, 351842, 18570
GPU-d, 777, 2048
"""
PS = """\
 351842 sturman  01:02:03 /isaac-sim/kit/python/bin/python3 scripts/train.py --task T1-Kick-v0 --headless
    777 other    5-01:00:00 python serve.py --task=Serve
"""


def output(
    cards: str = CARDS, apps: str = APPS, ps: str = PS, rc_cards: int = 0, rc_apps: int = 0, end: bool = True
) -> str:
    text = f"@@CARDS\n{cards}@@APPS rc={rc_cards}\n{apps}@@PS rc={rc_apps}\n{ps}"
    return text + ("@@END\n" if end else "")


class ParseGpuQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cards = parse_gpu_query(output(), "larg", "mckennie")

    def test_states(self) -> None:
        self.assertEqual([c.state for c in self.cards], ["busy", "free", "held", "busy", "held"])

    def test_owner_and_command(self) -> None:
        proc = self.cards[0].procs[0]
        self.assertEqual(
            (proc.pid, proc.user, proc.elapsed, proc.cmd), (351842, "sturman", "01:02:03", "train.py T1-Kick-v0")
        )

    def test_memory_takes_process_total_when_card_reads_zero(self) -> None:
        self.assertEqual(self.cards[3].mem_used, 2048)

    def test_comma_in_model_name(self) -> None:
        cards = parse_gpu_query(output(cards="0, GPU-a, Weird, Name, 100, 81920, 0\n", apps=""), "p", "h")
        self.assertEqual((cards[0].model, cards[0].mem_used, cards[0].state), ("Weird, Name", 100, "free"))

    def test_login_banner_before_cards_is_ignored(self) -> None:
        cards = parse_gpu_query("Welcome to LARG\n" + output(), "p", "h")
        self.assertEqual(len(cards), 5)

    def test_unreported_memory_is_unknown_not_free(self) -> None:
        cards = parse_gpu_query(output(cards="0, GPU-a, A40, [N/A], 46068, [N/A]\n", apps=""), "p", "h")
        self.assertEqual(cards[0].state, "unknown")


class ParseFailureTest(unittest.TestCase):
    def assert_fails(self, text: str) -> None:
        with self.assertRaises(ProbeError):
            parse_gpu_query(text, "p", "h")

    def test_truncated_before_apps(self) -> None:
        self.assert_fails(CARDS)

    def test_missing_end_marker(self) -> None:
        self.assert_fails(output(end=False))

    def test_failed_process_query(self) -> None:
        self.assert_fails(output(apps="Unable to determine the device handle: Unknown Error\n", ps="", rc_apps=15))

    def test_failed_card_query(self) -> None:
        self.assert_fails(output(cards="NVIDIA-SMI has failed\n", apps="", ps="", rc_cards=9))

    def test_no_cards(self) -> None:
        self.assert_fails(output(cards="", apps="", ps=""))

    def test_garbage_process_line(self) -> None:
        self.assert_fails(output(apps="something odd\n"))

    def test_process_on_unknown_card(self) -> None:
        self.assert_fails(output(apps="GPU-zzz, 5, 4000\n", ps=""))

    def test_repeated_marker(self) -> None:
        self.assert_fails(output().replace("@@PS rc=0", "@@APPS rc=1\n@@PS rc=0"))

    def test_output_after_end(self) -> None:
        self.assert_fails(output() + "stray line\n")


class RayCardTest(unittest.TestCase):
    GOOD = {
        "index": 0,
        "uuid": "GPU-r",
        "name": "RTX 5090",
        "memoryUsed": 545,
        "memoryTotal": 32607,
        "utilizationGpu": 0,
        "processesPids": [],
    }

    def test_idle_card_is_free(self) -> None:
        self.assertEqual(ray_card(dict(self.GOOD), "ray", "n").state, "free")

    def test_missing_or_none_readings_are_unknown(self) -> None:
        for key in ("memoryUsed", "utilizationGpu"):
            for value in (None, "absent"):
                g = dict(self.GOOD, processesPids=[{"pid": 7, "gpuMemoryUsage": 9000}])
                if value == "absent":
                    del g[key]
                else:
                    g[key] = value
                self.assertEqual(ray_card(g, "ray", "n").state, "unknown", (key, value))

    def test_renamed_keys_fail(self) -> None:
        with self.assertRaises(ProbeError):
            ray_card({"gpu_index": 0, "uuid": "GPU-r", "memory_used": 0}, "ray", "n")


class MergeDuplicatesTest(unittest.TestCase):
    def test_keeps_the_more_cautious_state(self) -> None:
        local = Card("local", "h", 0, "RTX", 400, 32000, 0, "free", uuid="GPU-x")
        ray = Card("ray", "h", 0, "RTX", 9000, 32000, 90, "busy", uuid="GPU-x", note="in use")
        reports = merge_duplicates([Report("local", "local", [local]), Report("ray", "ray", [ray])])
        self.assertEqual([c.state for r in reports for c in r.cards], ["busy"])


class HelpersTest(unittest.TestCase):
    def test_short_cmd(self) -> None:
        self.assertEqual(short_cmd("bash run.sh"), "bash")
        self.assertEqual(short_cmd("python x.py --task=Foo-v0"), "x.py Foo-v0")

    def test_gres_gpus(self) -> None:
        self.assertEqual([_gres_gpus(g) for g in ("gres/gpu:4", "gpu:a40:2", "N/A")], [4, 2, None])


if __name__ == "__main__":
    unittest.main()
