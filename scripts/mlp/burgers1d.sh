# run: scripts/mlp/burgers1d.sh
# tmux attach: tmux attach -t train_burgers1d_mlp_tag

#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=mlp
EXPERIMENT_NAME=burgers1d
TAG=0
SESSION_NAME="train_${EXPERIMENT_NAME}_${MODEL_NAME}_${TAG}"

# ==== Check tmux ====
if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed in this container."
  echo "Install it first, e.g.: apt-get update && apt-get install -y tmux"
  exit 1
fi

# ==== Training command ====
TRAIN_CMD="python -m pinnlab.train \
  --model_name ${MODEL_NAME} \
  --experiment_name ${EXPERIMENT_NAME} \
  --common_config configs/common_config.yaml \
  --model_config configs/model/${MODEL_NAME}.yaml \
  --exp_config configs/experiment/${EXPERIMENT_NAME}.yaml"

# ==== Start (or restart) tmux session ====
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[INFO] tmux session '${SESSION_NAME}' already exists. Killing and restarting it."
  tmux kill-session -t "${SESSION_NAME}"
fi

# Run training in background tmux session, in the current working directory
tmux new-session -d -s "${SESSION_NAME}" "cd $(pwd) && ${TRAIN_CMD}"

echo "[OK] Launched training in tmux session '${SESSION_NAME}'."
echo "To attach later: tmux attach -t ${SESSION_NAME}"
echo "To detach (while attached): Ctrl-b then d"