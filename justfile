venv_name := "ilab"
venv_py := venv_name / "bin" / "python"

rc_file := "$HOME/.bashrc"
bash_utils := "$( pwd )/scripts/utils.sh"

set shell := ["bash", "-c"]

# Single uv venv at ./ilab: `uv sync`/`uv run` use this var; `uv pip install` uses `--python {{venv_py}}`.
export UV_PROJECT_ENVIRONMENT := venv_name
# Neutralize any profile-activated venv so a stray VIRTUAL_ENV + `uv sync --active` can't prune it.
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
    @# Install cu128 torch FIRST, else something later pulls a cu126 build that "satisfies" torch>=2.7 and sticks.
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
    just vscode

# Generate .vscode/settings.json by booting a headless Isaac Sim to snapshot the Kit extension paths
# (the official `python -m isaaclab --generate-vscode-settings` can't import omni.kit_app under pip isaacsim).
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

# Docker interface (-> scripts/container.sh): build the shared Isaac image (nvcr isaacsim + pip Isaac Lab,
# code mounted at job start), reused by Ray + the HPC .sif. E.g. `just docker build`.
docker *args:
    if ! command -v docker >/dev/null 2>&1; then \
        curl -fsSL https://get.docker.com -o get-docker.sh; \
        sudo sh get-docker.sh; \
        sudo groupadd docker; \
        sudo usermod -aG docker $USER; \
        newgrp docker; \
    fi
    scripts/container.sh {{args}}

# Cluster interface (-> scripts/cluster.sh). First arg may be a cluster name (if config/<name> exists),
# else CLUSTER env / "default". E.g. `just cluster multi-delta repush` or `CLUSTER=delta just cluster setup`.
cluster *args:
    @set -- {{args}}; \
    if [ -n "${1:-}" ] && [ -d "scripts/cluster/config/${1}" ]; then \
        name="$1"; shift; CLUSTER="$name" scripts/cluster.sh "$@"; \
    else \
        scripts/cluster.sh "$@"; \
    fi

# Create a cluster config (scripts/cluster/config/<name>) -- alias for `scripts/cluster.sh add-cluster`.
add-cluster:
    scripts/cluster.sh add-cluster

# Ray interface (passthrough to scripts/ray.sh): `just ray setup` writes the configs, `just ray job ...`
# submits, plus list/logs/stop/push. Run `just deps` first so the configs + venv exist.
ray *args:
    scripts/ray.sh {{args}}

# Upload managed large-file resources (assets, datasets, policies) to W&B as versioned artifacts.
# Args: none/--auto (present resources >=50MB), --list, --all, or specific <key>...
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
new name:
    {{venv_py}} scripts/new_tasks.py {{name}}

# Run any hcrl_isaaclab script from the manager dir (no need to cd into the extension), e.g.:
#   just run train --task <id> --num_envs 4096
#   just run play  --task <id> --checkpoint <path>
#   just run video_logger --mode async --task <id> --wandb_project <entity>/<project>
# Runs the script *file* directly (so AppLauncher runs before the package import) with the ilab venv.
run script *args:
    OMNI_KIT_ACCEPT_EULA=YES {{venv_py}} resources/hcrl_isaaclab/scripts/{{script}}.py {{args}}
