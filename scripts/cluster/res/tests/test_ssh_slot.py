"""ssh_slot.sh against a fake ssh: refused sessions are retried, and sessions over one master never exceed its slots."""

import os
import subprocess
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

SHIM = Path(__file__).resolve().parents[2] / "ssh_slot.sh"

# the fake ssh: -G prints a controlpath per host; otherwise it logs its call and the stdin it got, then refuses while
# the refusal counter is above zero, or holds a session for $HOLD s while counting live ones
FAKE = textwrap.dedent("""\
    #!/usr/bin/env bash
    d=$FAKE_DIR
    if [ "$1" = -G ]; then for a; do last=$a; done; echo "controlpath $d/cm-${HOST_TAG:-h}"; exit 0; fi
    echo "$*" >> $d/calls
    if [ -s $d/refuse ] && [ "$(cat $d/refuse)" -gt 0 ]; then
        echo $(( $(cat $d/refuse) - 1 )) > $d/refuse
        echo "mux_client_request_session: session request failed: Session open refused by peer" >&2
        exit 255
    fi
    cat >> $d/stdin
    ( flock 9; n=$(( $(cat $d/live 2>/dev/null || echo 0) + 1 )); echo $n > $d/live
      m=$(cat $d/max 2>/dev/null || echo 0); [ $n -gt $m ] && echo $n > $d/max ) 9>$d/lock
    sleep ${HOLD:-0}
    ( flock 9; echo $(( $(cat $d/live) - 1 )) > $d/live ) 9>$d/lock
    exit ${RC:-0}
""")


class SshSlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        (self.d / "bin").mkdir()
        fake = self.d / "bin" / "ssh"
        fake.write_text(FAKE)
        fake.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{self.d / 'bin'}:{os.environ['PATH']}", "FAKE_DIR": str(self.d),
                    "HOME": str(self.d), "TMPDIR": str(self.d)}  # fmt: skip

    def ssh(self, *args: str, stdin: str = "", **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SHIM), *args], input=stdin, capture_output=True, text=True, env={**self.env, **env}, timeout=60
        )

    def calls(self) -> list[str]:
        return (self.d / "calls").read_text().splitlines()

    def test_a_refused_session_is_retried_and_its_stdin_still_arrives(self) -> None:
        (self.d / "refuse").write_text("2")
        out = self.ssh("-o", "ControlPath=/x/%C", "u@h", "bash", "-s", stdin="echo probe\n")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(len(self.calls()), 3, "two refusals, then the session")
        self.assertEqual((self.d / "stdin").read_text(), "echo probe\n")

    def test_another_failure_is_not_retried(self) -> None:
        out = self.ssh("-o", "ControlPath=/x/%C", "u@h", "false", RC="1")
        self.assertEqual((out.returncode, len(self.calls())), (1, 1))

    def test_calls_without_a_master_control_commands_and_terminals_pass_straight_through(self) -> None:
        for args in (
            ["u@h", "true"],
            ["-o", "ControlPath=/x/%C", "-O", "check", "u@h"],
            ["-tt", "-o", "ControlPath=/x/%C", "u@h"],
        ):
            with self.subTest(args=args):
                (self.d / "refuse").write_text("1")
                self.ssh(*args)
                self.assertTrue(self.calls()[-1].endswith(args[-1]))
        self.assertEqual(len(self.calls()), 3, "a pass-through call is never retried")
        self.assertFalse((self.d / ".ssh" / "cm" / "slots").exists(), "and takes no slot")

    def test_sessions_over_one_master_never_exceed_its_slots(self) -> None:
        runs = [threading.Thread(target=self.ssh, args=("-o", "ControlPath=/x/%C", "u@h", "true"), kwargs={"HOLD": "1"})
                for _ in range(9)]  # fmt: skip
        t0 = time.monotonic()
        for r in runs:
            r.start()
        for r in runs:
            r.join()
        self.assertEqual(len(self.calls()), 9)
        self.assertLessEqual(int((self.d / "max").read_text()), 4)
        self.assertGreater(time.monotonic() - t0, 2, "9 one-second sessions in 4 slots take at least 3 rounds")


if __name__ == "__main__":
    unittest.main()
