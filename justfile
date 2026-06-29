venv_name := "ilab"
venv_py := venv_name / "bin" / "python"

rc_file := "$HOME/.bashrc"
bash_utils := "$( pwd )/scripts/utils.sh"

set shell := ["bash", "-c"]

# Single uv-managed venv at the manager root (./ilab). `uv sync` / `uv run` target it via this var;
# `uv pip install` targets it via `--python {{venv_py}}`.
export UV_PROJECT_ENVIRONMENT := venv_name
# Neutralize any venv the user's shell profile auto-activates, so uv never operates on it by accident
# (a stray `VIRTUAL_ENV` + `uv sync --active` will prune that env to this project's deps).
export VIRTUAL_ENV := ""

# Manager-only dependencies (ray/cluster tooling). Creates the single `ilab` venv and installs the
# lightweight base into it; fetches the workspace subrepos via gitman. Called by `setup`, `cluster`, `ray`.
deps:
    if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
    uv venv --python 3.11 {{venv_name}}
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
        echo -e "\t\tRun  ilab  to activate the venv and enter the manager dir.\n"; \
    fi

# Full local install: manager deps + the Isaac Lab / Isaac Sim stack + every workspace package,
# all into the single `ilab` venv. Branches on whether IsaacLab was resolved as source or pip.
setup:
    just deps
    just resolve            # workspace.yaml -> flat deduped gitman.yml + fetch repos as siblings under resources/
    @# Install the Blackwell-correct torch (cu128) FIRST so nothing downstream pulls a cu126 build that
    @# then "satisfies" the torch>=2.7 constraint and sticks. Everything after sees torch already present.
    uv pip install --python {{venv_py}} --torch-backend cu128 torch==2.7.0 torchvision==0.22.0
    @# isaacsim/torch otherwise come transitively from isaaclab (pip mode) or explicitly (source mode).
    if grep -Eq '^[[:space:]]*source:[[:space:]]*true' workspace.yaml; then \
        echo "[setup] IsaacLab source mode -> editable install (+ explicit isaacsim; source isaaclab has no isaacsim extra)"; \
        uv pip install --python {{venv_py}} "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com; \
        for d in resources/IsaacLab/source/isaaclab*/; do [ -d "$d" ] && uv pip install --python {{venv_py}} --torch-backend cu128 -e "$d"; done; \
        uv pip install --python {{venv_py}} --torch-backend cu128 -e resources/hcrl_isaaclab; \
    else \
        echo "[setup] IsaacLab via pip -> hcrl_isaaclab[isaacsim] pulls isaaclab[isaacsim]==2.3.2.post1 (isaacsim 5.1 + torch)"; \
        uv pip install --python {{venv_py}} --torch-backend cu128 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match -e "resources/hcrl_isaaclab[isaacsim]"; \
    fi
    uv pip install --python {{venv_py}} rsl_rl-lib
    @# Editable-install only the Python packages; some *_robots repos are data/asset-only (no setup.py)
    @# and are materialized by the artifact resolver, not pip-installed.
    for d in resources/robot_rl resources/*_tasks resources/*_robots; do \
        if [ -d "$d" ] && { [ -f "$d/setup.py" ] || [ -f "$d/pyproject.toml" ]; }; then \
            uv pip install --python {{venv_py}} --torch-backend cu128 --extra-index-url https://pypi.nvidia.com -e "$d"; \
        elif [ -d "$d" ]; then echo "[setup] skipping non-package data repo: $d"; fi; \
    done
    just link-resources
    just vscode

# Symlink the flat-cloned asset/dataset repos into the hcrl_isaaclab extension's resources/ dir, so
# LOCAL runs reference the checked-out repos directly. The W&B artifact resolver (hcrl_isaaclab/utils/
# artifacts.py) leaves real dirs alone and only fetches when a resource is absent -- i.e. on Ray, where
# these repos aren't checked out. Re-run after `just resolve` adds/updates a resource repo.
link-resources:
    @mkdir -p resources/hcrl_isaaclab/resources
    @for r in hcrl_robots motion_datasets datasets; do \
        if [ -d "resources/$r" ]; then \
            ln -sfn "../../$r" "resources/hcrl_isaaclab/resources/$r"; \
            echo "[link-resources] resources/hcrl_isaaclab/resources/$r -> ../../$r"; \
        fi; \
    done

# Generate .vscode/settings.json so the workspace can be developed from the manager directory.
# Boots a headless Isaac Sim to snapshot the Kit extension paths (works for both source and pip
# Isaac Lab; the official `python -m isaaclab --generate-vscode-settings` can't import omni.kit_app
# under a pip-installed isaacsim).
vscode:
    @echo "[vscode] generating .vscode/settings.json via headless SimulationApp..."
    OMNI_KIT_ACCEPT_EULA=YES {{venv_py}} scripts/tools/setup_vscode.py || echo "[vscode][WARN] settings generation failed"

clean:
    @venv_dir="$( pwd )/{{venv_name}}"; \
    if [ -d $venv_dir ]; then \
        echo "[INFO] Removing virtual environment at ${venv_dir}."; \
        rm -rf $venv_dir; \
    fi
    @if grep -Fq "source {{bash_utils}}" {{rc_file}}; then \
        echo "[INFO] Removing $( basename {{bash_utils}} ) from $( basename {{rc_file}} )"; \
        sed -i "\|^.*source {{bash_utils}}|d" {{rc_file}}; \
    fi
    @echo "[INFO] Successfully cleaned up environment."

# Docker interface (passthrough to scripts/container.sh): build the shared Isaac image -- isaacsim from
# the nvcr base + Isaac Lab from pip, workspace code mounted at job start. Reused by Ray + the HPC .sif.
#   just docker build   (or `just docker`)
docker *args:
    if ! command -v docker >/dev/null 2>&1; then \
        curl -fsSL https://get.docker.com -o get-docker.sh; \
        sudo sh get-docker.sh; \
        sudo groupadd docker; \
        sudo usermod -aG docker $USER; \
        newgrp docker; \
    fi
    scripts/container.sh {{args}}

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

# Ray interface (passthrough to scripts/ray.sh): `just ray setup` writes the configs, `just ray job ...`
# submits, plus list/logs/stop/push. Run `just deps` first so the configs + venv exist.
ray *args:
    scripts/ray.sh {{args}}

# Upload managed large-file resources (assets, motion datasets, policies) to W&B as versioned artifacts.
# Usage:  just upload-artifacts            (auto: every present resource >= 50MB -- the ones too big for the job upload)
#         just upload-artifacts --list     (registry + sizes + large unregistered dirs)
#         just upload-artifacts --all      (every registered resource, any size)
#         just upload-artifacts <key>...   (specific resource keys)
upload-artifacts *args:
    @if [ ! -f "scripts/.env.wandb" ]; then \
        echo "[ERROR] scripts/.env.wandb not found; run 'just deps' first."; \
        exit 1; \
    fi; \
    set -a; source scripts/.env.wandb; set +a; \
    {{venv_py}} \
        resources/hcrl_isaaclab/scripts/tools/upload_artifacts.py {{args}}

# Resolve workspace.yaml -> flat deduped gitman.yml, then fetch all repos (flat under resources/).
resolve:
    {{venv_py}} scripts/resolve_workspace.py --manifest workspace.yaml --update

# Scaffold a new <name>_tasks extension repo under resources/ (registers under the <name>/ namespace).
new-tasks name:
    {{venv_py}} scripts/new_tasks.py {{name}}

# Run any hcrl_isaaclab script from the manager dir (no need to cd into the extension), e.g.:
#   just run train --task <id> --num_envs 4096
#   just run play  --task <id> --checkpoint <path>
#   just run video_logger --mode async --task <id> --wandb_project <entity>/<project>
# Runs the script *file* directly (so AppLauncher runs before the package import) with the ilab venv.
run script *args:
    OMNI_KIT_ACCEPT_EULA=YES {{venv_py}} resources/hcrl_isaaclab/scripts/{{script}}.py {{args}}
