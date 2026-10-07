"""The Apptainer recipe (a .sif built on an arm64 cluster) does what the Dockerfile does, from the same base."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = (ROOT / "scripts" / "docker" / "Dockerfile").read_text()
RECIPE = (ROOT / "scripts" / "cluster" / "hcrl-isaac.def").read_text()
# variables only the build needs: the recipe sets them in %post, not in the runtime environment
BUILD_ONLY_ENV = {"DEBIAN_FRONTEND"}


def flat(text: str) -> str:
    return re.sub(r"\\\n", " ", text)


def pip_installs(text: str) -> list[str]:
    """Every `pip install ...` argument list, whitespace-normalized and with file paths cut to their names, in order."""
    args = [" ".join(m.split()) for m in re.findall(r"-m pip install ([^&\n]+)", flat(text))]
    return [re.sub(r"/\S+/([^/\s]+)", r"\1", a) for a in args]


def apt_installs(text: str) -> list[list[str]]:
    """The package list of every `apt-get install`, in order."""
    return [sorted(m.split()) for m in re.findall(r"apt-get install -y --no-install-recommends ([^&\n]+)", flat(text))]


def placeholders(text: str) -> set[str]:
    """Paths the image pre-creates (mkdir -p / touch) as bind points, with ${ISAACSIM_ROOT_PATH} resolved."""
    text = flat(text).replace("${ISAACSIM_ROOT_PATH}", "/isaac-sim")
    found = set()
    for cmd in re.findall(r"(?:mkdir -p|touch) ([^&\n]+)", text):
        found |= {p for p in cmd.split() if p.startswith("/")}
    return found


class RecipeTest(unittest.TestCase):
    def test_same_pip_installs_in_the_same_order(self) -> None:
        self.assertTrue(pip_installs(DOCKERFILE), "the Dockerfile's pip installs were found")
        self.assertEqual(pip_installs(RECIPE), pip_installs(DOCKERFILE))

    def test_same_apt_installs(self) -> None:
        self.assertTrue(apt_installs(DOCKERFILE))
        self.assertEqual(apt_installs(RECIPE), apt_installs(DOCKERFILE))

    def test_the_runtime_env_is_the_dockerfiles(self) -> None:
        docker_env = set(re.findall(r"^ENV (\w+)=", DOCKERFILE, re.M)) - BUILD_ONLY_ENV
        environment = RECIPE.split("%environment", 1)[1].split("%post", 1)[0]
        self.assertEqual(set(re.findall(r"export (\w+)=", environment)), docker_env)
        self.assertNotIn("omni.usd", environment, "Kit's USD libraries are Python's alone (setup_python_env.sh)")

    def test_kits_usd_is_one_copy_added_for_python(self) -> None:
        post = RECIPE.split("%post", 1)[1].split("%test", 1)[0]
        self.assertIn('"${#usd[@]}" -ne 1', post, "more than one omni.usd.libs fails the build")
        self.assertIn(">> /isaac-sim/setup_python_env.sh", post)

    def test_same_bind_point_placeholders(self) -> None:
        self.assertEqual(placeholders(RECIPE) - {"/opt/hcrl-build"}, placeholders(DOCKERFILE))

    def test_the_entrypoint_runs(self) -> None:
        self.assertIn('ENTRYPOINT ["/usr/local/bin/hcrl-entrypoint"]', DOCKERFILE)
        self.assertIn("entrypoint.sh /usr/local/bin/hcrl-entrypoint", RECIPE)
        self.assertIn("exec /usr/local/bin/hcrl-entrypoint", RECIPE.split("%runscript", 1)[1])

    def test_the_base_is_the_dockerfiles_pulled_as_a_local_image(self) -> None:
        self.assertIn("Bootstrap: localimage\nFrom: isaac-sim-base.sif", RECIPE)
        interface = (ROOT / "scripts" / "cluster" / "cluster_interface.sh").read_text()
        self.assertIn("ARG ISAACSIM_BASE_IMAGE=", interface, "setup pulls the Dockerfile's base, not its own copy")
        self.assertIn("ARG ISAACSIM_VERSION=", interface)


if __name__ == "__main__":
    unittest.main()
