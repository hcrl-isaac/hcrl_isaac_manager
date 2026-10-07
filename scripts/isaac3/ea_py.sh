#!/bin/bash
# Run the Isaac Lab 3.0 EA python with our feature/isaac-3.0 clones on the path.
R=$(readlink -f "$(dirname "$0")/../../resources")
export PYTHONPATH=$R/hcrl_isaaclab:$R/robot_rl:$R/hhlm_tasks:$R/ssti_tasks:$R/hcrl_sim2real${PYTHONPATH:+:$PYTHONPATH}
exec $R/IsaacLab/.venv/bin/python "$@"
