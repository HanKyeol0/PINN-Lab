# run: scripts/mlp/z_reactiondiffusion1d_fore.sh

#!/usr/bin/env bash
set -e

MODEL_NAME=mlp
EXPERIMENT_NAME=reactiondiffusion1d

python -m pinnlab.train \
  --model_name $MODEL_NAME \
  --experiment_name $EXPERIMENT_NAME \
  --common_config configs/common_config.yaml \
  --model_config configs/model/${MODEL_NAME}.yaml \
  --exp_config configs/experiment/${EXPERIMENT_NAME}.yaml