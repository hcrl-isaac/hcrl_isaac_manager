# requires variables: PACKAGE_MANAGER, VENV_NAME

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

# change ilab alias depending on if Isaac Lab is installed locally
if [ -d "${MANAGER_DIR}/resources/IsaacLab/${VENV_NAME}" ]; then
    alias ilab="source ${MANAGER_DIR}/scripts/.env.wandb && source ${MANAGER_DIR}/resources/IsaacLab/${VENV_NAME}/bin/activate && cd ${MANAGER_DIR}/resources/IsaacLab/source/hcrl_isaaclab"
else
    alias ilab="cd ${MANAGER_DIR}/resources/IsaacLab/source/hcrl_isaaclab"
fi

alias manager="cd ${MANAGER_DIR} && source ${MANAGER_DIR}/.venv/bin/activate"
