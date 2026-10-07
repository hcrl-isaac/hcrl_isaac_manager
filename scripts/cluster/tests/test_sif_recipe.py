"""The Apptainer recipe (natively built .sif, e.g. arm64) installs what the Dockerfile installs, from the same base."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = (ROOT / "scripts" / "docker" / "Dockerfile").read_text()
RECIPE = (ROOT / "scripts" / "cluster" / "hcrl-isaac.def").read_text()


def pip_installs(text: str) -> list[str]:
    """Every `pip install ...` argument list, whitespace-normalized and with file paths cut to their names, in order."""
    flat = re.sub(r"\\\n", " ", text)
    args = [" ".join(m.split()) for m in re.findall(r"-m pip install ([^&\n]+)", flat)]
    return [re.sub(r"/\S+/([^/\s]+)", r"\1", a) for a in args]


def apt_packages(text: str) -> list[str]:
    flat = re.sub(r"\\\n", " ", text)
    m = re.search(r"apt-get install -y --no-install-recommends ([^&\n]+)", flat)
    return sorted(m.group(1).split()) if m else []


class RecipeTest(unittest.TestCase):
    def test_same_pip_installs_in_the_same_order(self) -> None:
        self.assertEqual(pip_installs(RECIPE), pip_installs(DOCKERFILE))
        self.assertTrue(pip_installs(DOCKERFILE), "the Dockerfile's pip installs were found")

    def test_same_apt_packages(self) -> None:
        self.assertEqual(apt_packages(RECIPE), apt_packages(DOCKERFILE))

    def test_the_base_is_the_dockerfiles_pulled_as_a_local_image(self) -> None:
        self.assertIn("Bootstrap: localimage\nFrom: isaac-sim-base.sif", RECIPE)
        interface = (ROOT / "scripts" / "cluster" / "cluster_interface.sh").read_text()
        self.assertIn("ARG ISAACSIM_BASE_IMAGE=", interface, "setup pulls the Dockerfile's base, not its own copy")
        self.assertIn("ARG ISAACSIM_VERSION=", interface)

    def test_the_entrypoint_and_bind_points_are_there(self) -> None:
        self.assertIn("entrypoint.sh /usr/local/bin/hcrl-entrypoint", RECIPE)
        for path in ("/workspace/ext", "/root/.cache/ov", "kit/cache", "/var/run/nvidia-persistenced/socket"):
            self.assertIn(path, RECIPE)


if __name__ == "__main__":
    unittest.main()
