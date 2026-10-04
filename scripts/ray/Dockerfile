FROM isaac-lab-base

# wget: the Ray liveness checks need it to mark the container ready.
RUN apt-get update && apt-get install wget

RUN apt-get update && apt-get install -y openssh-client

RUN apt-get update && apt-get install -y rsync && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y graphviz

# NVIDIA runtime paths
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin

# Ray, with its CLI run by Kit's python
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install "ray[default, tune]"==2.50.1 && \
sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" \
/isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray

COPY install-git-lfs.sh /workspace/isaaclab/install-git-lfs.sh
RUN bash /workspace/isaaclab/install-git-lfs.sh && git lfs version

# Tuning dependencies
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install optuna bayesian-optimization

RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install gitman