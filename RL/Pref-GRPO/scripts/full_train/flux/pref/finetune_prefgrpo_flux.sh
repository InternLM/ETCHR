#!/bin/bash
set -x


# CUDA 环境配置
export PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8

# activate conda env
source /mnt/shared-storage-user/mllm/xinglong/miniconda3/bin/activate Dance
echo "current conda env: $CONDA_DEFAULT_ENV"
cd /mnt/shared-storage-user/mllm/xinglong/Pref-GRPO
export PYTHONPATH=/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO:$PYTHONPATH

export WANDB_MODE=offline
export WANDB_API_KEY=1697adda76308c9a62b666be1b8caeb1a39704a3
export WANDB_PROJECT="Pref-GRPO"
export WANDB_NAME="pref_flux"
export WANDB_DIR="/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/wandb/wandb_1_16_2026/${WANDB_NAME}"
wandb login
echo "now wandb dir:${WANDB_DIR}"
mkdir -p $WANDB_DIR


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common_finetune.sh"

export EXP_NAME=${WANDB_NAME}


API_URL="http://localhost:8080"
OUTPUT_DIR="outputs/${EXP_NAME}"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path /mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/FLUX.1-dev
  --vae_model_path /mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/FLUX.1-dev
  --data_json_path /mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/data/unigenbench_train_prompt_rl_embeddings/videos2caption.json
  --exp_name "${EXP_NAME}"
  --num_sample 100000
  --train_batch_size 1
  --dataloader_num_workers 1
  --learning_rate 1e-5
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 25
  --eta 0.7
  --num_generations 8
  --reward_spec '{"unifiedreward_think": 0.4, "clip": 0.6}'
  --api_url "${API_URL}"
)


export NNODES=${NODE_COUNT:-2}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR}
export NODE_RANK=${NODE_RANK}
export MASTER_PORT=29513

torchrun --nnodes=$NNODES --nproc_per_node=$PROC_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  fastvideo/train_flux.py \
  "${TRAIN_ARGS[@]}"
