venv_name := "ilab"
venv_py := venv_name / "bin" / "python"

rc_file := "$HOME/.bashrc"
bash_utils := "$( pwd )/scripts/utils.sh"

set shell := ["bash", "-c"]

# Single uv venv at ./ilab: `uv sync`/`uv run` use this var; `uv pip install` uses `--python {{venv_py}}`.
export UV_PROJECT_ENVIRONMENT := venv_name
# Neutralize any profile-activated venv so a stray VIRTUAL_ENV + `uv sync --active` can't prune it.
export VIRTUAL_ENV := ""

# Manager base env: the `ilab` venv + lightweight tooling (uv, gitman, git-lfs). Called by setup/cluster/ray.
deps:
    if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
    @# --relocatable: console scripts derive the interpreter from their own path (survives a dir rename).
    uv venv --relocatable --python 3.11 {{venv_name}}
    uv sync
    @# git-lfs must materialize LFS assets as real files -- the Ray mount ships pointers as-is otherwise.
    @if ! command -v git-lfs >/dev/null 2>&1; then \
        echo "[deps] git-lfs not found; attempting install..."; \
        if command -v brew >/dev/null 2>&1; then brew install git-lfs; \
        elif command -v apt-get >/dev/null 2>&1; then sudo apt-get update -qq && sudo apt-get install -y git-lfs; \
        elif command -v dnf >/dev/null 2>&1; then sudo dnf install -y git-lfs; \
        elif command -v pacman >/dev/null 2>&1; then sudo pacman -S --noconfirm git-lfs; \
        fi; \
    fi
    @if command -v git-lfs >/dev/null 2>&1; then git lfs install; \
        else echo "[deps][WARN] git-lfs missing -- LFS assets (e.g. crab/MCP policies) will be pointers and fail on the cluster; install git-lfs and re-run."; fi
    @# Install the gitman tool only; fetching subrepos (+ generating gitman.yaml) is `just resolve`'s job.
    uv tool install gitman
    @if [ ! -f "scripts/.env.wandb" ]; then \
        {{venv_py}} scripts/tools/ask.py wandb-env scripts/tools/.env.wandb.template scripts/.env.wandb; \
    fi
    @if ! grep -Fq "source {{bash_utils}}" {{rc_file}}; then \
        echo "[INFO] Adding $( basename {{bash_utils}} ) to $( basename {{rc_file}} )"; \
        echo "VENV_NAME={{venv_name}} source {{bash_utils}}" >> {{rc_file}}; \
        echo -e "[INFO] Successfully added $( basename {{bash_utils}} ) to $( basename {{rc_file}} )\n"; \
        echo -e "\t\tRun  ilab  to activate the venv and enter the manager dir.\n"; \
    fi

# Full local install into the `ilab` venv: manager deps + Isaac Lab / Isaac Sim + every workspace package.
setup:
    just deps
    @# Re-open the project/IsaacLab picker pre-filled (plain `just setup` reconfigures); no TTY keeps it.
    {{venv_py}} scripts/configure_workspace.py --interactive
    just resolve            # merge selection + defaults -> gitman.yaml + fetch repos under resources/
    @# Install torch + IsaacLab/Sim + workspace packages -- one gated block so `mode: none` skips it all.
    @mode=$(grep -E '^[[:space:]]*mode:' workspace.yaml | head -1 | sed -E 's/.*mode:[[:space:]]*//; s/[[:space:]#].*//'); \
    if [ "$mode" = "none" ]; then \
        echo "[setup] IsaacLab mode 'none': repos fetched under resources/; skipping IsaacLab + package installs."; \
        exit 0; \
    fi; \
    {{venv_py}} scripts/tools/ui.py section "PyTorch (CUDA 12.8)"; \
    uv pip install --python {{venv_py}} --torch-backend cu128 torch==2.7.0 torchvision==0.22.0; \
    {{venv_py}} scripts/tools/ui.py section "Isaac Lab / Isaac Sim"; \
    if [ "$mode" = "source" ]; then \
        echo "[setup] IsaacLab source mode -> editable install (+ explicit isaacsim; source isaaclab has no isaacsim extra)"; \
        uv pip install --python {{venv_py}} "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com; \
        for d in resources/IsaacLab/source/isaaclab*/; do [ -d "$d" ] && uv pip install --python {{venv_py}} --torch-backend cu128 -e "$d"; done; \
        uv pip install --python {{venv_py}} --torch-backend cu128 -e resources/hcrl_isaaclab; \
    else \
        echo "[setup] IsaacLab via pip -> hcrl_isaaclab[isaacsim] pulls isaaclab[isaacsim]==2.3.2.post1 (isaacsim 5.1 + torch)"; \
        uv pip install --python {{venv_py}} --torch-backend cu128 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match -e "resources/hcrl_isaaclab[isaacsim]"; \
    fi; \
    uv pip install --python {{venv_py}} rsl_rl-lib; \
    {{venv_py}} scripts/tools/ui.py section "Workspace packages"; \
    for d in resources/robot_rl resources/*_tasks resources/*_robots; do \
        if [ -d "$d" ] && { [ -f "$d/setup.py" ] || [ -f "$d/pyproject.toml" ]; }; then \
            uv pip install --python {{venv_py}} --torch-backend cu128 --extra-index-url https://pypi.nvidia.com -e "$d"; \
        elif [ -d "$d" ]; then echo "[setup] skipping non-package data repo: $d"; fi; \
    done; \
    just vscode

# Generate .vscode/settings.json by booting headless Isaac Sim to snapshot the Kit extension paths.
vscode:
    @OMNI_KIT_ACCEPT_EULA=YES {{venv_py}} scripts/tools/ui.py spin \
        "Booting headless Isaac Sim to snapshot VS Code extension paths (~1 min)" \
        -- {{venv_py}} scripts/tools/setup_vscode.py || echo "[vscode][WARN] settings generation failed"

# Remove the ilab venv, generated workspace config (workspace.yaml/gitman.yaml), and shell alias (prompts first).
clean:
    @if [ -x "{{venv_py}}" ] && ! {{venv_py}} scripts/tools/ask.py confirm "Remove the ilab venv, generated workspace config, and shell alias?"; then \
        echo "[clean] aborted."; exit 0; \
    fi; \
    venv_dir="$( pwd )/{{venv_name}}"; \
    if [ -d "$venv_dir" ]; then \
        echo "[INFO] Removing virtual environment at ${venv_dir}."; \
        rm -rf "$venv_dir"; \
    fi; \
    for f in workspace.yaml gitman.yaml gitman.yml; do \
        if [ -f "$f" ]; then echo "[INFO] Removing generated $f."; rm -f "$f"; fi; \
    done; \
    if grep -Fq "source {{bash_utils}}" {{rc_file}}; then \
        echo "[INFO] Removing $( basename {{bash_utils}} ) from $( basename {{rc_file}} )"; \
        sed -i "\|^.*source {{bash_utils}}|d" {{rc_file}}; \
    fi; \
    echo "[INFO] Successfully cleaned up environment."

# Docker interface (scripts/docker/): build the shared Isaac image, reused by Ray + the HPC .sif.
docker *args:
    if ! command -v docker >/dev/null 2>&1; then \
        curl -fsSL https://get.docker.com -o get-docker.sh; \
        sudo sh get-docker.sh; \
        sudo groupadd docker; \
        sudo usermod -aG docker $USER; \
        newgrp docker; \
    fi
    scripts/docker/docker_interface.sh {{args}}

# Cluster interface (scripts/cluster/): `add`/`setup`/`job`/`develop`/...; bare picks; leading name -> config/<name>.
cluster *args:
    @set -- {{args}}; \
    name=""; \
    if [ -n "${1:-}" ] && [ -d "scripts/cluster/config/${1}" ]; then name="$1"; shift; fi; \
    if [ -z "${1:-}" ]; then \
        set -- "$( {{venv_py}} scripts/tools/ask.py select 'Cluster subcommand:' setup job develop repush build add )"; \
    fi; \
    if [ -z "$name" ] && [ -z "${CLUSTER:-}" ]; then \
        case "${1:-}" in setup|job|develop|repush) \
            cfgs=$(ls -d scripts/cluster/config/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null); \
            if [ "$(printf '%s\n' $cfgs | grep -c .)" -gt 1 ]; then \
                name="$( {{venv_py}} scripts/tools/ask.py select 'Target cluster:' $cfgs )"; \
            fi ;; \
        esac; \
    fi; \
    if [ -n "$name" ]; then CLUSTER="$name" scripts/cluster/cluster_interface.sh "$@"; \
    else scripts/cluster/cluster_interface.sh "$@"; fi

# Ray interface (scripts/ray/): `setup`, `job`, `bench`, `push`, `list`, `logs`, `stop`; bare shows a picker.
ray *args:
    @set -- {{ args }}; \
    if [ -z "${1:-}" ]; then \
        set -- "$( {{venv_py}} scripts/tools/ask.py select 'Ray subcommand:' setup job bench push list logs stop )"; \
    fi; \
    scripts/ray/ray_interface.sh "$@"

# Upload managed large-file resources to W&B as versioned artifacts; args: --list, --all, or specific <key>...
upload-artifacts *args:
    @if [ ! -f "scripts/.env.wandb" ]; then \
        echo "[ERROR] scripts/.env.wandb not found; run 'just deps' first."; \
        exit 1; \
    fi; \
    set -a; source scripts/.env.wandb; set +a; \
    {{venv_py}} \
        resources/hcrl_isaaclab/scripts/tools/upload_artifacts.py {{args}}

# Merge selection + committed defaults -> gitman.yaml, then fetch all repos under resources/ (defaults if none).
resolve:
    {{venv_py}} scripts/configure_workspace.py   # ensure a workspace.yaml exists (no prompt)
    {{venv_py}} scripts/resolve_workspace.py --manifest workspace.yaml --update
    @# materialize LFS in every fetched repo (repos cloned before git-lfs was active hold pointers)
    @if command -v git-lfs >/dev/null 2>&1; then \
        for d in resources/*/; do [ -d "$d/.git" ] && (echo "[resolve] git lfs pull $d"; git -C "$d" lfs pull) || true; done; \
    else echo "[resolve][WARN] git-lfs missing -- LFS files stay as pointers; run 'just deps' to install it."; fi

# Scaffold a new <name>_tasks extension repo under resources/ (registers under the <name>/ namespace).
new name:
    {{venv_py}} scripts/new_tasks.py {{name}}

# Run any hcrl_isaaclab/scripts/<script>.py from the manager dir with the ilab venv, e.g.
#   just run train --task <id> --num_envs 4096   |   just run play --task <id> --checkpoint <path>
run script *args:
    OMNI_KIT_ACCEPT_EULA=YES {{venv_py}} resources/hcrl_isaaclab/scripts/{{script}}.py {{args}}

# Run a workspace repo's CPU smoke tests (task registration + env-cfg build; skips `-m gpu`). Launches
# Isaac Sim, so it needs the ilab venv + a GPU. e.g. `just test ssti_tasks` (default: the core repo).
test repo="hcrl_isaaclab" *args:
    cd "{{justfile_directory()}}/resources/{{repo}}" && OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH= "{{justfile_directory()}}/{{venv_py}}" -m pytest -m "not gpu" {{args}}
