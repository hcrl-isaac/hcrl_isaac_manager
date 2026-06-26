venv_name := "ilab"

rc_file := "$HOME/.bashrc"
bash_utils := "$( pwd )/scripts/utils.sh"

set shell := ["bash", "-c"]

deps:
    if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
    uv sync
    uv tool install gitman && gitman update --skip-changes
    @if [ ! -f "scripts/.env.wandb" ]; then \
        read -p "W&B Username: " wandb_username; \
        read -p "W&B API Key: " wandb_api_key; \
        echo "[INFO] Writing wandb env file..."; \
        WANDB_USERNAME=$wandb_username WANDB_API_KEY=$wandb_api_key envsubst < scripts/tools/.env.wandb.template > scripts/.env.wandb; \
    fi
    @if ! grep -Fq "source {{bash_utils}}" {{rc_file}}; then \
        echo "[INFO] Adding $( basename {{bash_utils}} ) to $( basename {{rc_file}} )"; \
        echo "VENV_NAME={{venv_name}} source {{bash_utils}}" >> {{rc_file}}; \
        echo -e "[INFO] Successfully added $( basename {{bash_utils}} ) to $( basename {{rc_file}} )\n"; \
        echo -e "\t\t1. For development in the main hcrl_isaaclab extension, run:   ilab"; \
        echo -e "\t\t2. For manager use (e.g. cluster scripts), run:                manager"; \
        echo -e "\n"; \
    fi

setup:
    just deps
    uv venv --python 3.11 resources/IsaacLab/{{venv_name}}; \
    cd resources/IsaacLab && source {{venv_name}}/bin/activate; \
    uv pip install --upgrade pip; \
    uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com; \
    uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128; \
    ./isaaclab.sh -u {{venv_name}}; \
    ./isaaclab.sh -i rsl_rl

clean:
    @venv_dir="$( pwd )/resources/IsaacLab/{{venv_name}}"; \
    if [ -d $venv_dir ]; then \
        echo "[INFO] Removing virtual environment at ${venv_dir}."; \
        rm -rf $venv_dir; \
    fi
    @if grep -Fq "source {{bash_utils}}" {{rc_file}}; then \
        echo "[INFO] Removing $( basename {{bash_utils}} ) from $( basename {{rc_file}} )"; \
        sed -i "\|^.*source {{bash_utils}}|d" {{rc_file}}; \
    fi
    @echo "[INFO] Successfully cleaned up environment."

docker:
    if ! command -v docker >/dev/null 2>&1; then \
        curl -fsSL https://get.docker.com -o get-docker.sh; \
        sudo sh get-docker.sh; \
        sudo groupadd docker; \
        sudo usermod -aG docker $USER; \
        newgrp docker; \
    fi
    scripts/container.sh start

cluster name="default":
    just deps
    if ! command -v nvidia-container-toolkit >/dev/null 2>&1; then \
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
            && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
            && sudo apt-get update; \
        sudo apt-get install -y nvidia-container-toolkit; \
        sudo systemctl restart docker; \
        sudo nvidia-ctk runtime configure --runtime=docker; \
        sudo systemctl restart docker; \
    fi;
    if ! command -v apptainer >/dev/null 2>&1; then \
        sudo apt update; \
        sudo apt install -y software-properties-common; \
        sudo add-apt-repository -y ppa:apptainer/ppa; \
        sudo apt update; \
        sudo apt install -y apptainer; \
    fi;
    just docker;
    CLUSTER={{name}} scripts/cluster.sh push;

add-cluster:
    @read -p "Cluster Nickname (leave blank for default): " cluster_name; \
    if [ -z $cluster_name ]; then cluster_name="default"; fi; \
    outdir="scripts/cluster/${cluster_name}_config"; \
    if [ -d $outdir ]; then \
        echo "[ERROR] Cluster config with nickname $cluster_name already exists. Delete it, edit it directly, or pick a different name."; \
        exit 1; \
    fi; \
    read -p "Cluster Login (username@address): " cluster_login; \
    read -p "Home Directory (`echo '$HOME'` from cluster machine): " home; \
    read -p "Scratch Directory (`echo '$SCRATCH'` from cluster machine): " scratch; \
    read -p "Email (for job notifications): " email; \
    read -p "Queue Name: " queue; \
    read -p "GPUs per Node: " num_procs; \
    read -p "CPUs per Task/GPU: " num_cpus; \
    case "$home" in /*) ;; *) home="/$home" ;; esac; \
    case "$scratch" in /*) ;; *) scratch="/$scratch" ;; esac; \
    mkdir $outdir; \
    echo "[INFO] Writing cluster env file..."; \
    HOME=$home SCRATCH=$scratch CLUSTER_LOGIN=$cluster_login NUM_PROCS=$num_procs NUM_CPUS=$num_cpus envsubst < scripts/cluster/tools/.env.cluster.template > $outdir/.env.cluster; \
    echo "[INFO] Writing SLURM job config file..."; \
    EMAIL=$email QUEUE=$queue NUM_PROCS=$num_procs NUM_CPUS=$num_cpus envsubst < scripts/cluster/tools/submit_job_slurm.template.sh > $outdir/submit_job_slurm.sh;

ray:
    just deps
    @read -p "UT EID: " ut_eid; \
    if [ ! -f "scripts/.env.wandb" ]; then \
        echo "[ERROR] wandb configuration file scripts/.env.wandb not found. Exiting."; \
        exit 1; \
    fi; \
    source scripts/.env.wandb; \
    UT_EID=$ut_eid envsubst < scripts/ray/tools/.env.ray.template > scripts/ray/.env.ray; \
    for cfg in job_config bench_job_config job_config_distributed; do \
        UT_EID=$ut_eid MANAGER_DIR="$( pwd )" envsubst < scripts/ray/tools/$cfg.template.yaml > scripts/ray/$cfg.yaml; \
    done; \
    echo "[INFO] Created ray configuration files (.env.ray + job_config/bench_job_config/job_config_distributed .yaml) in scripts/ray."

# Usage:  just upload-artifacts --list   (registry + local presence)
#         just upload-artifacts --all    (every registered resource)
#         just upload-artifacts <key>... (specific resource keys)
# Upload managed large-file resources (assets, motion datasets, policies) to W&B as versioned artifacts.
upload-artifacts *args:
    @if [ ! -f "scripts/.env.wandb" ]; then \
        echo "[ERROR] scripts/.env.wandb not found; run 'just deps' first."; \
        exit 1; \
    fi; \
    set -a; source scripts/.env.wandb; set +a; \
    .venv/bin/python \
        resources/IsaacLab/source/hcrl_isaaclab/scripts/tools/upload_artifacts.py {{args}}

# Resolve workspace.yaml -> flat deduped gitman.yml, then fetch all repos (flat under resources/).
resolve:
    .venv/bin/python scripts/resolve_workspace.py --manifest workspace.yaml --update

# Scaffold a new <name>_tasks extension repo under resources/ (registers under the <name>/ namespace).
new-tasks name:
    .venv/bin/python scripts/new_tasks.py {{name}}
