"""Unit tests for merge_profile (run: python3 -m unittest discover -s scripts/cluster/tests)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from merge_profile import merge_env, merge_submit

OLD_JOB = """\
module load tacc-apptainer/1.4.1
module load nvidia/26.1
cat <<EOT > job.sh
#SBATCH -p old-queue
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=48:00:00
echo custom step
EOT
"""

NEW_JOB = """\
module load tacc-apptainer
cat <<EOT > job.sh
#SBATCH -p new-queue
#SBATCH -N 1
#SBATCH --mem-per-cpu=0
#SBATCH --time=24:00:00
EOT
"""


class MergeSubmitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text, self.lost = merge_submit(OLD_JOB, NEW_JOB)
        self.lines = self.text.splitlines()

    def test_prompted_flags_take_the_new_value(self) -> None:
        self.assertIn("#SBATCH -p new-queue", self.lines)
        self.assertIn("#SBATCH --time=24:00:00", self.lines)

    def test_flags_the_template_lacks_are_kept(self) -> None:
        self.assertIn("#SBATCH --gpus-per-node=4", self.lines)
        self.assertIn("#SBATCH --exclusive", self.lines)

    def test_mutually_exclusive_memory_flags_keep_the_old_choice(self) -> None:
        self.assertIn("#SBATCH --mem=0", self.lines)
        self.assertNotIn("#SBATCH --mem-per-cpu=0", self.lines)

    def test_module_pins_and_extra_modules_are_kept(self) -> None:
        self.assertEqual(self.lines[:2], ["module load tacc-apptainer/1.4.1", "module load nvidia/26.1"])

    def test_unknown_lines_are_reported(self) -> None:
        self.assertEqual(self.lost, ["echo custom step"])


class MergeEnvTest(unittest.TestCase):
    def test_keeps_hand_set_values_and_extra_keys(self) -> None:
        old = 'CLUSTER_LOGIN=a@x\nCLUSTER_PYTHON_EXECUTABLE="-m torch.distributed.run x"\nCLUSTER_SRUN_EXTRA=--foo\n'
        new = 'CLUSTER_LOGIN=b@y\nCLUSTER_PYTHON_EXECUTABLE="scripts/train.py"\nCLUSTER_APPTAINER_FLAGS="--fakeroot"\n'
        merged = merge_env(old, new).splitlines()
        self.assertEqual(
            merged,
            [
                "CLUSTER_LOGIN=b@y",
                'CLUSTER_PYTHON_EXECUTABLE="-m torch.distributed.run x"',
                'CLUSTER_APPTAINER_FLAGS="--fakeroot"',
                "CLUSTER_SRUN_EXTRA=--foo",
            ],
        )


if __name__ == "__main__":
    unittest.main()
