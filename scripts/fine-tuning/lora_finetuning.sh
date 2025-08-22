set -e

MODEL_NAME=mlp
EXPERIMENT_NAME=laplace_bc_src

python -m pinnlab.train \
  --model_name $MODEL_NAME \
  --experiment_name $EXPERIMENT_NAME \
  --common_config configs/common_config.yaml \
  --model_config configs/model/$MODEL_NAME.yaml \
  --exp_config configs/experiment/$EXPERIMENT_NAME.yaml