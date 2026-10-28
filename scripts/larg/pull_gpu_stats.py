"""GPU use on every host of your ssh pools (scripts/cluster/res/compute.local.toml), one nvidia-smi over ssh each."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster" / "res"))
from inventory import load_pools, ssh_target


def servers() -> list[str]:
    """The ssh target of every host of the ssh pools that name a user."""
    pools = [p for p in load_pools() if p.kind == "ssh"]
    if not pools:
        sys.exit("[gpu_stats] no ssh pools: `pls setup --add <pool> --kind ssh --hosts ... --user ...` adds one")
    for p in pools:
        if not p.settings.get("user"):
            print(f"[gpu_stats] skipping pool {p.name}: no user set in compute.local.toml", file=sys.stderr)
    return [ssh_target(p, h) for p in pools if p.settings.get("user") for h in p.settings.get("hosts", [])]


NVIDIA_CMD = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"


def get_gpu_stats(server: str) -> str:
    try:
        ssh = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", server, NVIDIA_CMD], capture_output=True, text=True, timeout=10
        )

        if ssh.returncode != 0:
            return f"Error: {ssh.stderr.strip()}"

        return ssh.stdout.strip()

    except subprocess.TimeoutExpired:
        return "Error: Connection Timed Out"
    except Exception as e:
        return f"Error: {e!s}"


def main() -> None:
    print(f"{'SERVER':<20} | {'ID':<3} | {'GPU NAME':<20} | {'UTIL':<5} | {'MEM USED':<10}")
    print("-" * 75)

    for server in servers():
        output = get_gpu_stats(server)

        if output.startswith("Error"):
            print(f"{server:<20} | {output}")
            continue

        for line in output.split("\n"):
            if not line:
                continue

            parts = [x.strip() for x in line.split(",")]

            if len(parts) >= 5:
                idx, name, util, mem_used, mem_total = parts

                # Create a visual indicator for high usage
                mem_str = f"{mem_used}/{mem_total} MB"

                print(f"{server:<20} | {idx:<3} | {name:<20} | {util + '%':<5} | {mem_str:<10}")

        print("-" * 75)


if __name__ == "__main__":
    main()
