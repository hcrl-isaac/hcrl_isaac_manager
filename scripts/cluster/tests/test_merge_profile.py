"""Unit tests for merge_profile (run: python3 -m unittest discover -s scripts/cluster/tests)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from merge_profile import flag_value, merge_env, merge_submit

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

ALIAS_JOB = """\
cat <<EOT > job.sh
#SBATCH --partition=gpuA40x4
#SBATCH --ntasks=4
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --gres=gpu:a40:4
EOT
"""

TEMPLATE_JOB = """\
cat <<EOT > job.sh
#SBATCH -p gpuA40x4
#SBATCH -n 4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
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

    def test_dropped_commands_are_reported_but_answered_prompts_are_not(self) -> None:
        self.assertEqual(self.lost, ["echo custom step"])

    def test_shadowed_duplicate_flag_is_reported(self) -> None:
        dup = "cat <<EOT > job.sh\n#SBATCH --gres=gpu:4\n#SBATCH --gres=gpu:a40:4\nEOT\n"
        tmpl = "cat <<EOT > job.sh\n#SBATCH -p q\n#SBATCH --gres=gpu:1\nEOT\n"
        self.assertEqual(merge_submit(dup, tmpl)[1], ["#SBATCH --gres=gpu:4"])


class AliasTest(unittest.TestCase):
    def test_flag_value_reads_any_spelling(self) -> None:
        values = [flag_value(ALIAS_JOB, f) for f in ("-p", "-n", "-c", "-t", "-A")]
        self.assertEqual(values, ["gpuA40x4", "4", "16", "48:00:00", ""])

    def test_aliases_do_not_duplicate_flags(self) -> None:
        text, lost = merge_submit(ALIAS_JOB, TEMPLATE_JOB)
        flags = [ln.split()[1].split("=")[0] for ln in text.splitlines() if ln.startswith("#SBATCH")]
        self.assertEqual(sorted(flags), sorted(["-p", "-n", "--cpus-per-task", "--time", "--gres", "--gres"]))
        self.assertEqual(lost, [])


class MergeEnvTest(unittest.TestCase):
    def test_keeps_old_lines_and_order_and_appends_new_keys(self) -> None:
        old = (
            "BASE=/scratch/u\nCLUSTER_ISAAC_SIM_CACHE_DIR=$BASE/docker-isaac-sim\nexport HELPER=1\nCLUSTER_LOGIN=a@x\n"
        )
        new = (
            "CLUSTER_LOGIN=b@y\nCLUSTER_ISAAC_SIM_CACHE_DIR=/new/docker-isaac-sim\n# flags\nCLUSTER_APPTAINER_FLAGS=x\n"
        )
        merged = merge_env(old, new).splitlines()
        self.assertEqual(
            merged,
            [
                "BASE=/scratch/u",
                "CLUSTER_ISAAC_SIM_CACHE_DIR=$BASE/docker-isaac-sim",
                "export HELPER=1",
                "CLUSTER_LOGIN=b@y",
                "# flags",
                "CLUSTER_APPTAINER_FLAGS=x",
            ],
        )

    def test_derived_keys_follow_a_changed_answer(self) -> None:
        old = "CLUSTER_ISAAC_SIM_CACHE_DIR=/old/docker-isaac-sim\nOMP_NUM_THREADS=16\n"
        new = "CLUSTER_ISAAC_SIM_CACHE_DIR=/new/docker-isaac-sim\nOMP_NUM_THREADS=8\n"
        merged = merge_env(old, new, follow={"CLUSTER_ISAAC_SIM_CACHE_DIR"}).splitlines()
        self.assertEqual(merged, ["CLUSTER_ISAAC_SIM_CACHE_DIR=/new/docker-isaac-sim", "OMP_NUM_THREADS=16"])


if __name__ == "__main__":
    unittest.main()
