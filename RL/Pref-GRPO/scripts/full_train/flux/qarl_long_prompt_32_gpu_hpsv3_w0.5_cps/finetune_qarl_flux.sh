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
export WANDB_NAME="qarl_long_prompt_32_gpu_hpsv3_w0.5_cps"
export WANDB_DIR="/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/wandb/wandb_1_18/${WANDB_NAME}"
wandb login
echo "now wandb dir:${WANDB_DIR}"
mkdir -p $WANDB_DIR


# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source "${SCRIPT_DIR}/../_common_finetune.sh"
source "/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/scripts/_common_finetune.sh"


export EXP_NAME=${WANDB_NAME}

# QARL VLM server 配置
QARL_HOST="10.102.98.163"
QARL_BASE_PORT=10000
QARL_NUM_SERVERS=8
QARL_MODEL_NAME="qwen3"
QARL_NUM_THREADS=96

OUTPUT_DIR="outputs/${EXP_NAME}"

# 训练数据（包含 keep_qa_list）
DATA_JSON_PATH="/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/grocery/qa_construction_3/4_filter_group_acc_std0.075_5035.jsonl"
# Embed 文件目录（prompt_embed, pooled_prompt_embeds, text_ids）
EMBED_DIR="/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/data/expand_long_prompt_rl_embeddings"

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path /mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/FLUX.1-dev
  --vae_model_path /mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/FLUX.1-dev
  --data_json_path "${DATA_JSON_PATH}"
  --embed_dir "${EMBED_DIR}"
  --exp_name "${EXP_NAME}"
  --num_sample 100000
  --train_batch_size 1
  --dataloader_num_workers 1
  --learning_rate 1e-5
  --checkpointing_steps 20
  --num_train_epochs 1
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 25
  --eta 0.7
  --num_generations 16
  --eval_every_steps 20
  --flow_mode "cps"
  # QARL reward 配置
  --reward_spec '{"qarl": 1.0, "hpsv3": 0.5, "clip": 0.0}'
  --qarl_host "${QARL_HOST}"
  --qarl_base_port ${QARL_BASE_PORT}
  --qarl_num_servers ${QARL_NUM_SERVERS}
  --qarl_model_name "${QARL_MODEL_NAME}"
  --qarl_num_threads ${QARL_NUM_THREADS}
  --qarl_all_qa
  --qarl_temperature 0
)


export NNODES=${NODE_COUNT:-2}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR}
export NODE_RANK=${NODE_RANK}
export MASTER_PORT=29513

torchrun --nnodes=$NNODES --nproc_per_node=$PROC_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  fastvideo/train_flux.py \
  "${TRAIN_ARGS[@]}"

