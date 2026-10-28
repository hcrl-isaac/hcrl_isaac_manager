"""Unit tests for the pool inventory's config files (run all script tests: pls test-scripts)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inventory as inv

BASE = '[compute.box]\nkind = "local"\n'


class LocalOverridesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.res, self.main = tmp / "wt" / "res", tmp / "main"
        self.res.mkdir(parents=True)
        (self.main / "scripts" / "cluster" / "res").mkdir(parents=True)
        (self.res / "compute.toml").write_text(BASE)
        self.main_local = self.main / "scripts" / "cluster" / "res" / "compute.local.toml"
        self.main_local.write_text('[compute.larg]\nkind = "ssh"\nuser = "me"\n')
        for p in (
            mock.patch.object(inv, "RES_DIR", self.res),
            mock.patch.object(inv, "_main_checkout", lambda: self.main),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_a_worktree_without_one_uses_the_main_checkouts(self) -> None:
        """compute.local.toml is gitignored, so a worktree has none and would otherwise lose its ssh logins."""
        cfg = inv.load_config()
        self.assertEqual(cfg["compute"]["larg"]["user"], "me")

    def test_this_checkouts_own_wins(self) -> None:
        (self.res / "compute.local.toml").write_text('[compute.larg]\nkind = "ssh"\nuser = "mine"\n')
        self.assertEqual(inv.load_config()["compute"]["larg"]["user"], "mine")

    def test_the_tracked_file_ships_only_this_machine(self) -> None:
        """Logins, hosts and addresses are per user: compute.toml holds no pool but local."""
        tracked = inv._read_toml(Path(inv.__file__).resolve().parent / "compute.toml")
        self.assertEqual({n: s["kind"] for n, s in tracked["compute"].items()}, {"local": "local"})


class AddPoolTest(unittest.TestCase):
    """`pls setup --add <name> --kind ssh|ray` writes the user's compute.local.toml."""

    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.res, self.main = tmp / "wt" / "res", tmp / "main"
        self.res.mkdir(parents=True)
        (self.res / "compute.toml").write_text(BASE)
        self.main_local = self.main / "scripts" / "cluster" / "res" / "compute.local.toml"
        for p in (
            mock.patch.object(inv, "RES_DIR", self.res),
            mock.patch.object(inv, "_main_checkout", lambda: self.main),
            mock.patch.object(inv, "slurm_pools", lambda: []),
            mock.patch.dict(inv.os.environ, {"LARG_USER": ""}),
        ):
            p.start()
            self.addCleanup(p.stop)

    def add(self, *argv: str) -> None:
        with mock.patch("sys.stdout"):
            inv.main(["add", *argv])

    def test_a_new_file_goes_in_the_main_checkout_and_reads_back_as_a_pool(self) -> None:
        self.add(
            "lab", "--kind", "ssh", "--hosts", "a, b", "--user", "u", "--domain", "x.org", "--pin", "device", "--render"
        )
        pools = {p.name: p for p in inv.load_pools()}
        self.assertTrue(self.main_local.is_file())
        self.assertEqual(pools["lab"].kind, "ssh")
        self.assertEqual(
            pools["lab"].settings,
            {"kind": "ssh", "hosts": ["a", "b"], "user": "u", "domain": "x.org", "pin": "device", "render": True},
        )
        self.assertEqual(inv.resolve_ssh_target("a", list(pools.values())), "u@a.x.org")
        self.assertEqual(inv.resolve_ssh_target("b.x.org", list(pools.values())), "u@b.x.org")

    def test_a_ray_pool_gives_the_ray_address(self) -> None:
        self.add("ray", "--kind", "ray", "--address", "http://h:8265/")
        self.assertEqual(inv.ray_address(inv.load_pools()), "http://h:8265")

    def test_an_existing_pool_needs_update_and_the_rest_of_the_file_survives(self) -> None:
        self.main_local.parent.mkdir(parents=True)
        self.main_local.write_text(
            '# mine\n[compute.lab]\nkind = "ssh"\nhosts = ["a"]\nuser = "old"\n\n[compute.delta]\nenabled = false\n'
        )
        with self.assertRaises(SystemExit) as ctx:
            self.add("lab", "--kind", "ssh", "--hosts", "c", "--user", "new")
        self.assertIn("--update", str(ctx.exception.code))
        self.add("lab", "--kind", "ssh", "--hosts", "c", "--user", "new", "--update")
        text = self.main_local.read_text()
        self.assertIn("# mine", text)
        cfg = inv._read_toml(self.main_local)["compute"]
        self.assertEqual(cfg["lab"], {"kind": "ssh", "hosts": ["c"], "user": "new"})
        self.assertEqual(cfg["delta"], {"enabled": False})

    def test_kind_settings_are_checked(self) -> None:
        for argv in (
            ["lab", "--kind", "ssh", "--hosts", "a"],  # no user
            ["lab", "--kind", "ssh", "--user", "u"],  # no hosts
            ["r", "--kind", "ray"],  # no address
            ["r", "--kind", "ray", "--address", "http://h", "--user", "u"],
            ["bad name", "--kind", "ray", "--address", "http://h"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                self.add(*argv)
        self.assertFalse(self.main_local.exists())

    def test_ssh_user_and_unlisted_hosts(self) -> None:
        pools = [inv.Pool("a", "ssh", {"hosts": ["h1"], "user": "u"}), inv.Pool("b", "ssh", {"hosts": ["h2"]})]
        self.assertEqual(inv.ssh_user(pools), "u")
        with self.assertRaises(ValueError):
            inv.resolve_ssh_target("h2", pools)  # its pool names no user
        with self.assertRaises(ValueError):
            inv.resolve_ssh_target("elsewhere", pools)
        with mock.patch.dict(inv.os.environ, {"LARG_USER": "env"}):
            self.assertEqual(inv.resolve_ssh_target("elsewhere", pools), "env@elsewhere")
            self.assertEqual(inv.ssh_user(pools), "env")
        pools.append(inv.Pool("c", "ssh", {"hosts": ["h3"], "user": "v"}))
        with self.assertRaises(ValueError):
            inv.ssh_user(pools)
        with self.assertRaises(ValueError):
            inv.ray_address(pools)


if __name__ == "__main__":
    unittest.main()
