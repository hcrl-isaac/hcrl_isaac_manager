"""`pls stage --on <ssh host|pool> data` (push_data.py) against a stub `ssh` that runs locally, so rsync is real."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RES))
import evaluate as ev
import push_data as pd
from inventory import Pool

SSH_STUB = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-J|-i|-p|-l|-F|-E|-c|-m|-L|-R|-D|-W|-b|-e|-S|-w|-Q|-B) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
exec bash -c "$*"
"""
POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 9\n"


class PushDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "bin").mkdir()
        (self.tmp / "bin" / "ssh").write_text(SSH_STUB)
        (self.tmp / "bin" / "ssh").chmod(0o755)
        path = mock.patch.dict(os.environ, {"PATH": f"{self.tmp / 'bin'}:{os.environ['PATH']}"})
        path.start()
        self.addCleanup(path.stop)
        self.ws, self.root = self.tmp / "box_ws", self.tmp / "isaac3"
        self.pool = Pool("larg-a40", "ssh", {"user": "u", "hosts": ["pepi", "salah"], "workspace": str(self.ws)})
        self.t = ev.make_target(self.pool, "pepi", 0)
        self.src = self.tmp / "local" / "motion_datasets"
        (self.src / "lafan1").mkdir(parents=True)
        (self.src / "lafan1" / "new.pt").write_bytes(b"x" * 2048)
        (self.src / "lafan1" / "pointer.pt").write_bytes(POINTER)
        (self.src / "lafan1" / "notes.txt").write_text("not a training file")
        (self.src / "worktrees" / "wt" / "lafan1").mkdir(parents=True)
        (self.src / "worktrees" / "wt" / "lafan1" / "wt.pt").write_bytes(b"w")

    def test_a_host_or_a_pool_names_its_ssh_hosts(self) -> None:
        """A host is itself; a pool (prefix) is every host of it; anything else is refused."""
        self.assertEqual([h for _, h in pd.hosts_of("salah", [self.pool])], ["salah"])
        self.assertEqual([h for _, h in pd.hosts_of("larg", [self.pool])], ["pepi", "salah"])
        with self.assertRaises(SystemExit):
            pd.hosts_of("nowhere", [self.pool])

    def test_the_copy_the_box_reads_is_the_destination(self) -> None:
        """The workspace's copy first, else --root's, else the workspace's (created)."""
        self.assertEqual(
            pd.destination(self.t, "motion_datasets", str(self.root)), f"{self.ws}/resources/motion_datasets"
        )
        (self.root / "resources" / "motion_datasets").mkdir(parents=True)
        self.assertEqual(
            pd.destination(self.t, "motion_datasets", str(self.root)), f"{self.root}/resources/motion_datasets"
        )
        (self.ws / "resources" / "motion_datasets").mkdir(parents=True)
        self.assertEqual(
            pd.destination(self.t, "motion_datasets", str(self.root)), f"{self.ws}/resources/motion_datasets"
        )

    def test_only_new_training_files_are_added_and_nothing_is_deleted(self) -> None:
        """Training files are added; an LFS pointer, other files and worktrees stay behind; the box's own files stay."""
        dest = self.root / "resources" / "motion_datasets"
        (dest / "lafan1").mkdir(parents=True)
        (dest / "lafan1" / "pointer.pt").write_bytes(b"real content")
        (dest / "lafan1" / "old.pt").write_bytes(b"kept")
        self.assertEqual(pd.push(self.t, str(self.src), str(dest)), 0)
        self.assertEqual((dest / "lafan1" / "new.pt").read_bytes(), b"x" * 2048)
        self.assertEqual((dest / "lafan1" / "pointer.pt").read_bytes(), b"real content", "a pointer never overwrites")
        self.assertEqual((dest / "lafan1" / "old.pt").read_bytes(), b"kept", "nothing is deleted")
        self.assertFalse((dest / "lafan1" / "notes.txt").exists(), "only the training-file allowlist ships")
        self.assertFalse((dest / "worktrees").exists(), "worktrees never ship")


if __name__ == "__main__":
    unittest.main()
