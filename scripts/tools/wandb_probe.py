"""One W&B poll for watch_run.sh: a run's state, its last iteration and the seconds since it last logged.

usage: wandb_probe.py <entity/project/run_id>
Prints ``state=<state> it=<iteration|none> age=<seconds|-1>``, or nothing when W&B cannot be read (a failed probe).
"""

import sys
import time


def main() -> None:
    import wandb

    run = wandb.Api(timeout=60).run(sys.argv[1])  # a fresh Api per poll: a long-lived one can serve a stale run
    summary = run.summary
    it = summary.get("local_step")  # the iteration (robot_rl's step metric); W&B's _step is not
    logged = summary.get("_timestamp")
    age = int(time.time() - logged) if logged else -1
    print(f"state={run.state} it={int(it) if it is not None else 'none'} age={age}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001  any failure reads as no answer, which watch_run.sh counts toward BLIND
        pass
