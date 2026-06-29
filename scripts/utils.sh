# requires variables: VENV_NAME (the single uv venv at the manager root, e.g. "ilab")

if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_PATH="$0"
else
    SCRIPT_PATH="$BASH_SOURCE[0]"
fi

MANAGER_DIR="$(realpath $( cd "$( dirname "$SCRIPT_PATH" )" &> /dev/null && pwd )/../)"

if [[ -z "$VENV_NAME" ]]; then
    echo "[ERROR] VENV_NAME must be set!"
    return
fi

VENV_ACTIVATE="${MANAGER_DIR}/${VENV_NAME}/bin/activate"
WANDB_ENV="${MANAGER_DIR}/scripts/.env.wandb"

# Single venv + single entry point: `ilab` activates the venv, sources W&B creds, and drops you in
# the manager dir. From there the hcrl_isaaclab scripts run from anywhere (e.g. `uv run` / `python -m`),
# so there's no separate extension-dir alias.
if [ -f "$VENV_ACTIVATE" ]; then
    alias ilab="{ [ -f '${WANDB_ENV}' ] && source '${WANDB_ENV}'; }; source '${VENV_ACTIVATE}' && cd ${MANAGER_DIR}"
else
    alias ilab="cd ${MANAGER_DIR}"
fi
