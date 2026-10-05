# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Submit aggregate job(s) to the Ray cluster(s) listed in a config file.

The config file holds one ``name: <NAME> address: http://<IP>:<PORT>`` line per cluster. Aggregate jobs are
separated by the ``*`` delimiter and assigned by ``job_index % cluster_count``; ``--aggregate_jobs`` must be the
last argument. An aggregate job is a ``wrap_resources.py`` (sub-jobs separated by ``+``), ``tuner.py`` or
``task_runner.py`` invocation.

Usage:
    python scripts/ray/submit_job.py --config_file <cfg> --job_config <yaml> --aggregate_jobs <job> [* <job>]
"""

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import yaml
from dotenv import dotenv_values
from job_args import split_jobs
from ray import job_submission


def read_cluster_spec(fn: str | None = None) -> list[dict]:
    cluster_spec_path = os.path.expanduser("~/.cluster_config" if fn is None else fn)

    if not os.path.exists(cluster_spec_path):
        raise FileNotFoundError(f"Cluster spec file not found at {cluster_spec_path}")

    clusters = []
    with open(cluster_spec_path) as f:
        for line in f:
            parts = line.strip().split(" ")
            http_address = parts[3]
            cluster_info = {"name": parts[1], "address": http_address}
            print(f"[INFO] Setting {cluster_info['name']}")
            clusters.append(cluster_info)

    return clusters


def submit_job(cluster: dict, job_command: str, runtime_env: dict, metadata: dict, other_data: dict) -> None:
    """
    Submits a job to a single cluster, prints the final result and Ray dashboard URL at the end.
    """
    try:
        address = cluster["address"]
        cluster_name = cluster["name"]
        print(f"[INFO] Submitting job to cluster '{cluster_name}' at {address}")
        client = job_submission.JobSubmissionClient(address)
        print(f"[INFO] Checking contents of the directory: {runtime_env['working_dir']}")
        try:
            dir_contents = os.listdir(runtime_env["working_dir"])
            print(f"[INFO] Directory contents: {dir_contents}")
        except Exception as e:
            print(f"[INFO] Failed to list directory contents: {e!s}")
        entrypoint = f'{runtime_env["py_executable"]} {job_command} --file-mounts "{other_data["file_mounts"]}" --init-commands "{other_data.get("init_commands", "[]")}" --sub-jobs {other_data["python_script"]}'
        print(f"[INFO] Attempting entrypoint {entrypoint} in cluster {cluster}")
        job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env, metadata=metadata)

        print(f"[INFO] Submitted job with ID {job_id}.")
    except Exception as e:
        print(traceback.format_exc())
        raise e


def submit_jobs_to_clusters(
    jobs: list[str], clusters: list[dict], runtime_env: dict, metadata: dict, other_data: dict
) -> None:
    """
    Submit all jobs to their respective clusters, cycling through clusters if there are more jobs than clusters.
    """
    if not clusters:
        raise ValueError("No clusters available for job submission.")

    if len(jobs) < len(clusters):
        print("[INFO] Less jobs than clusters, some clusters will not receive jobs")
    elif len(jobs) == len(clusters):
        print("[INFO] Exactly one job per cluster")
    else:
        print("[INFO] More jobs than clusters, jobs submitted as clusters become available.")

    with ThreadPoolExecutor() as executor:
        for idx, job_command in enumerate(jobs):
            cluster = clusters[idx % len(clusters)]
            executor.submit(submit_job, cluster, job_command, runtime_env, metadata, other_data)


def parse_env_file(fp: str | None) -> dict:
    if fp is None:
        return {}
    return dict(dotenv_values(fp))


def parse_job_config(cfg_file: str) -> tuple[dict, dict, dict]:
    with open(cfg_file) as job_yaml:
        job_config = yaml.safe_load(job_yaml)
    working_dir = job_config["ext_dir"]
    env_dict = parse_env_file(job_config.get("env_file"))
    py_modules = list(job_config.get("file_mounts", {}).keys())
    if len(py_modules) == 0:
        py_modules = None
        file_mounts = "{}"
    else:
        # keyed by the container-side repo name: worktrees of different repos share a local basename
        file_mounts = json.dumps({v.rstrip("/").split("/")[-1]: v for v in job_config["file_mounts"].values()})
    init_commands = json.dumps(job_config.get("init_commands", []))
    runtime_env = {
        "working_dir": working_dir,
        "env_vars": env_dict,
        "py_modules": py_modules,
        "py_executable": job_config["py_executable"],
        "excludes": job_config.get("excludes", []),
    }
    metadata = {"user_id": job_config["user_id"]}
    other_data = {
        "python_script": job_config["python_script"],
        "file_mounts": file_mounts.replace('"', '\\"'),
        "init_commands": init_commands.replace('\\"', '"').replace('"', '\\"'),
    }
    return runtime_env, metadata, other_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple GPU jobs to multiple Ray clusters.")
    parser.add_argument("--config_file", default="~/.cluster_config", help="The cluster config path.")
    parser.add_argument("--job_config", help="Path to the job config file.")
    parser.add_argument(
        "--aggregate_jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="This should be last argument. The aggregate jobs to submit separated by the * delimiter.",
    )
    args = parser.parse_args()
    formatted_jobs = split_jobs(args.aggregate_jobs or [])
    if len(formatted_jobs) > 1:
        print("Warning; Split jobs by cluster with the * delimiter")
    print(f"[INFO] Isaac Ray Wrapper received jobs {formatted_jobs=}")

    clusters = read_cluster_spec(args.config_file)
    runtime_env, metadata, other_data = parse_job_config(args.job_config)
    submit_jobs_to_clusters(formatted_jobs, clusters, runtime_env, metadata, other_data)
