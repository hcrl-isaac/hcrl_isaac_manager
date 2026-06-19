import subprocess
import sys

# --- Configuration ---
# List your servers here (or load from a file)
SERVERS = [
    "sturman@mckennie.cs.utexas.edu",
    "sturman@hazard.cs.utexas.edu",
    "sturman@debruyne.cs.utexas.edu",
    "sturman@aaronson.cs.utexas.edu",
    "sturman@pepi.cs.utexas.edu",
    "sturman@pulisic.cs.utexas.edu",
    "sturman@salah.cs.utexas.edu",
    "sturman@pogba.cs.utexas.edu",
]

# The nvidia-smi query to run
# We ask for CSV format to make parsing easy
NVIDIA_CMD = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"


def get_gpu_stats(server):
    try:
        # Run SSH command with a timeout
        ssh = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", server, NVIDIA_CMD], capture_output=True, text=True, timeout=10
        )

        if ssh.returncode != 0:
            return f"Error: {ssh.stderr.strip()}"

        return ssh.stdout.strip()

    except subprocess.TimeoutExpired:
        return "Error: Connection Timed Out"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    print(f"{'SERVER':<20} | {'ID':<3} | {'GPU NAME':<20} | {'UTIL':<5} | {'MEM USED':<10}")
    print("-" * 75)

    for server in SERVERS:
        output = get_gpu_stats(server)

        # specific error handling or parsing
        if output.startswith("Error"):
            print(f"{server:<20} | {output}")
            continue

        # Parse the CSV output from nvidia-smi
        for line in output.split("\n"):
            if not line:
                continue

            # parts: [index, name, util, mem_used, mem_total]
            parts = [x.strip() for x in line.split(",")]

            if len(parts) >= 5:
                idx, name, util, mem_used, mem_total = parts

                # Create a visual indicator for high usage
                mem_str = f"{mem_used}/{mem_total} MB"

                print(f"{server:<20} | {idx:<3} | {name:<20} | {util + '%':<5} | {mem_str:<10}")

        print("-" * 75)


if __name__ == "__main__":
    main()
