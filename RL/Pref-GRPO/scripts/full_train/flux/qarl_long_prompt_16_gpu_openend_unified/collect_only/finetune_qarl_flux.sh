#!/bin/bash
set -x

# ==================== Collect Only Mode ====================
# 用于收集前 N 个 prompt 的图片及其 QARL 问答详情
# 与主训练脚本保持一致的配置，仅用于观察 QARL 组内问答情况
# ===========================================================

# CUDA 环境配置
export PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8

# activate conda env
source /mnt/shared-storage-user/mllm/xinglong/miniconda3/bin/activate Dance
echo "current conda env: $CONDA_DEFAULT_ENV"
cd /mnt/shared-storage-user/mllm/xinglong/Pref-GRPO
export PYTHONPATH=/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO:$PYTHONPATH

# collect_only 模式不需要 wandb
export WANDB_MODE=disabled

source "/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/scripts/_common_finetune.sh"

export EXP_NAME="qarl_long_prompt_16_gpu_openend_unified"

# QARL VLM server 配置（需要确保服务已启动）
QARL_HOST="10.102.97.48"
QARL_BASE_PORT=10000
QARL_NUM_SERVERS=8
QARL_MODEL_NAME="qwen3"
QARL_NUM_THREADS=32

# UnifiedReward API（需要确保服务已启动）
API_URL="http://10.102.250.23:8081"

# 输出目录：主训练目录的子文件夹
OUTPUT_DIR="outputs/${EXP_NAME}/collect_only"
COLLECT_OUTPUT_DIR="${OUTPUT_DIR}"

# 训练数据（包含 keep_qa_list）- 与主训练脚本一致
DATA_JSON_PATH="/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/grocery/qa_construction_3/5_open_end/open_end_qa_5035.jsonl"
# Embed 文件目录（prompt_embed, pooled_prompt_embeds, text_ids）
EMBED_DIR="/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/data/expand_long_prompt_rl_embeddings"

# 收集的 prompt 数量（每个 GPU 会处理这么多个 prompt）
NUM_PROMPTS_TO_COLLECT=48

TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path /mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/FLUX.1-dev
  --vae_model_path /mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/FLUX.1-dev
  --data_json_path "${DATA_JSON_PATH}"
  --embed_dir "${EMBED_DIR}"
  --exp_name "${EXP_NAME}"
  --num_sample ${NUM_PROMPTS_TO_COLLECT}
  --train_batch_size 1
  --dataloader_num_workers 1
  --learning_rate 1e-5
  --checkpointing_steps 999999
  --num_train_epochs 1
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 25
  --eta 0.7
  --num_generations 16
  --eval_every_steps 999999
  --api_url "${API_URL}"
  # QARL reward 配置 - 与主训练脚本一致（开放式问答 + unified reward）
  --reward_spec '{"qarl": 1.0, "unifiedreward_style": 0.5, "clip": 0.0}'
  --open_end_reward
  --qarl_host "${QARL_HOST}"
  --qarl_base_port ${QARL_BASE_PORT}
  --qarl_num_servers ${QARL_NUM_SERVERS}
  --qarl_model_name "${QARL_MODEL_NAME}"
  --qarl_num_threads ${QARL_NUM_THREADS}
  --qarl_all_qa
  --qarl_temperature 0
  # collect_only 模式专用参数
  --collect_only
  --collect_output_dir "${COLLECT_OUTPUT_DIR}"
)

export NNODES=${NODE_COUNT:-2}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR}
export NODE_RANK=${NODE_RANK}
export MASTER_PORT=29513

torchrun --nnodes=$NNODES --nproc_per_node=$PROC_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  fastvideo/train_flux.py \
  "${TRAIN_ARGS[@]}"
