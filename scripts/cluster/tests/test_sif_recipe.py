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


def uv_steps(text: str) -> list[str]:
    """Every `uv sync` / `uv pip ...` argument list, whitespace-normalized and with file paths cut to their names."""
    args = [" ".join(m.split()) for m in re.findall(r"\buv (sync [^&\n]+|pip [^&\n]+)", flat(text))]
    return [re.sub(r"/\S+/([^/\s]+)", r"\1", a) for a in args]


def apt_installs(text: str) -> list[list[str]]:
    """The package list of every `apt-get install`, in order."""
    return [sorted(m.split()) for m in re.findall(r"apt-get install -y --no-install-recommends ([^&\n;]+)", flat(text))]


def placeholders(text: str) -> set[str]:
    """Paths the image pre-creates (mkdir -p / touch) as bind points."""
    found = set()
    for cmd in re.findall(r"(?:mkdir -p|touch) ([^&\n;]+)", flat(text)):
        found |= {p for p in cmd.split() if p.startswith("/")}
    return found


def docker_args(text: str) -> dict[str, str]:
    """The Dockerfile's Isaac Lab ARG defaults."""
    return {k: v.strip('"') for k, v in re.findall(r"^ARG (ISAACLAB_\w+)=(.+)$", text, re.M)}


class RecipeTest(unittest.TestCase):
    def test_same_uv_steps_in_the_same_order(self) -> None:
        steps = uv_steps(DOCKERFILE)
        self.assertTrue(any(s.startswith("sync ") and "--frozen" in s for s in steps), "the venv comes from the lock")
        self.assertEqual(uv_steps(RECIPE), steps)

    def test_same_apt_installs(self) -> None:
        self.assertTrue(apt_installs(DOCKERFILE))
        self.assertEqual(apt_installs(RECIPE), apt_installs(DOCKERFILE))

    def test_the_runtime_env_is_the_dockerfiles(self) -> None:
        docker_env = set(re.findall(r"^ENV (\w+)=", DOCKERFILE, re.M)) - BUILD_ONLY_ENV
        environment = RECIPE.split("%environment", 1)[1].split("%post", 1)[0]
        self.assertEqual(set(re.findall(r"export (\w+)=", environment)), docker_env)

    def test_same_isaaclab_defaults(self) -> None:
        arguments = RECIPE.split("%arguments", 1)[1].split("%files", 1)[0]
        recipe_args = dict(line.strip().split("=", 1) for line in arguments.strip().splitlines())
        self.assertTrue(docker_args(DOCKERFILE))
        for key, value in docker_args(DOCKERFILE).items():
            self.assertEqual(recipe_args.get(key), value, key)
        uv_image = re.search(r"^FROM ghcr\.io/astral-sh/uv:(\S+) AS uv$", DOCKERFILE, re.M)
        self.assertIsNotNone(uv_image, "the Dockerfile copies a pinned uv")
        self.assertEqual(recipe_args.get("UV_VERSION"), uv_image.group(1))

    def test_same_bind_point_placeholders(self) -> None:
        self.assertEqual(placeholders(RECIPE) - {"/opt/hcrl-build"}, placeholders(DOCKERFILE))

    def test_the_entrypoint_runs(self) -> None:
        self.assertIn('ENTRYPOINT ["/usr/local/bin/hcrl-entrypoint"]', DOCKERFILE)
        self.assertIn("entrypoint.sh /usr/local/bin/hcrl-entrypoint", RECIPE)
        self.assertIn("exec /usr/local/bin/hcrl-entrypoint", RECIPE.split("%runscript", 1)[1])

    def test_the_base_is_the_dockerfiles_pulled_as_a_local_image(self) -> None:
        self.assertIn("Bootstrap: localimage\nFrom: base.sif", RECIPE)
        interface = (ROOT / "scripts" / "cluster" / "cluster_interface.sh").read_text()
        self.assertIn("ARG BASE_IMAGE=", interface, "setup pulls the Dockerfile's base, not its own copy")
        self.assertIn("isaaclab-args", interface, "and passes the docker build's Isaac Lab ref and extras")


if __name__ == "__main__":
    unittest.main()
