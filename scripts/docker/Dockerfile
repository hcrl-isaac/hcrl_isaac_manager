# Shared Isaac image, reused by BOTH the Ray worker and the HPC Apptainer (.sif). Decoupled from the
# IsaacLab source tree: nvcr isaacsim/Kit base + pip Isaac Lab; the workspace extensions are NOT baked in
# -- they're mounted + PYTHONPATH'd at job start (see entrypoint.sh) so code syncs without a rebuild.
# Build: scripts/docker/docker_interface.sh build  (or `just docker build`).

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=5.1.0
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION} AS base

SHELL ["/bin/bash", "-c"]
USER root
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV ISAACSIM_ROOT_PATH=/isaac-sim
# Kit's bundled interpreter -- everything installs into and runs with this python.
ENV ISAAC_PY=/isaac-sim/python.sh
# Accept the Isaac Sim EULA non-interactively (headless cluster/Ray runs).
ENV OMNI_KIT_ACCEPT_EULA=YES

# System deps: build toolchain, Ray-liveness wget, ssh+rsync for code sync, git(-lfs) for asset pulls.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl openssh-client rsync graphviz libglib2.0-0 ncurses-term && \
    apt -y autoremove && apt clean autoclean && rm -rf /var/lib/apt/lists/*

# Isaac Lab (pip) + RL + Ray + tuning + tooling, into Kit's python. No source COPY, no [isaacsim] extra
# (isaacsim is provided by the base image's Kit). flatdict pinned first to avoid a known resolution quirk.
# Pins here are INLINE: constraints.workspace.txt covers only the generated deps below.
# rsl-rl-lib is explicit on purpose -- isaaclab gates it behind its "rsl-rl" extra AT 3.0.1, but the
# bundled isaaclab_rl.rsl_rl is imported by hcrl_isaaclab.robot_rl and we run 5.4.1 to match robot_rl.
RUN ${ISAAC_PY} -m pip install --upgrade pip && \
    ${ISAAC_PY} -m pip install flatdict==4.0.0 && \
    ${ISAAC_PY} -m pip install "isaaclab==2.3.2.post1" "rsl-rl-lib==5.4.1" && \
    ${ISAAC_PY} -m pip install "ray[default,tune]==2.50.1" optuna bayesian-optimization gitman

# Runtime deps of the mounted workspace packages (PYTHONPATH'd, not pip-installed, so their deps must
# be baked). Generated from each repo's own pyproject/setup.py by collect_workspace_deps.py, so it
# cannot drift; the constraints file holds the pins a loose transitive requirement would silently break.
# No --upgrade: the base image's torch/torchvision are +cu128 builds that PyPI wheels would replace.
COPY requirements.workspace.txt constraints.workspace.txt /tmp/
RUN ${ISAAC_PY} -m pip install -r /tmp/requirements.workspace.txt -c /tmp/constraints.workspace.txt

# git-lfs (robot assets / motion datasets / policies are pulled via LFS during code sync).
COPY install-git-lfs.sh /tmp/install-git-lfs.sh
RUN bash /tmp/install-git-lfs.sh && git lfs version

# Make `ray` resolve to Kit's python (Ray liveness probe + CLI on the cluster).
RUN if [ -f /isaac-sim/kit/python/bin/ray ]; then \
        sed -i "1i #!${ISAAC_PY}" /isaac-sim/kit/python/bin/ray; \
        ln -sf /isaac-sim/kit/python/bin/ray /usr/local/bin/ray; \
    fi

# NVIDIA runtime paths.
ENV PATH="/usr/local/nvidia/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin 2>/dev/null || true

# Apptainer/Singularity placeholders + cache dirs, so the SAME image converts cleanly to a .sif.
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache /root/.cache/ov /root/.cache/pip \
    /root/.cache/nvidia/GLCache /root/.nv/ComputeCache /root/.nvidia-omniverse/logs \
    /root/.local/share/ov/data /root/Documents /tmp /workspace/ext && \
    touch /bin/nvidia-smi /bin/nvidia-debugdump /bin/nvidia-persistenced \
    /bin/nvidia-cuda-mps-control /bin/nvidia-cuda-mps-server /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && touch /var/run/nvidia-persistenced/socket

# Entrypoint PYTHONPATHs the mounted workspace extensions (+ IsaacLab source in source mode) ahead of the
# baked pip isaaclab, then execs the requested command. Mount points are bound at run time.
COPY entrypoint.sh /usr/local/bin/hcrl-entrypoint
RUN chmod +x /usr/local/bin/hcrl-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/hcrl-entrypoint"]
