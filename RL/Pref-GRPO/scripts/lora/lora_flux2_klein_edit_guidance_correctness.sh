#!/bin/bash
set -x

export PATH=/mnt/shared-storage-user/mllm/zhangzhixiong/Shared/Envs/cuda/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mllm/zhangzhixiong/Shared/Envs/cuda/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/shared-storage-user/mllm/zhangzhixiong/Shared/Envs/cuda/cuda-12.8

source /mnt/shared-storage-user/zhangbeichen/.bashrc
source ~/miniconda3/bin/activate dancegrpo
echo "current conda env: $CONDA_DEFAULT_ENV"
cd /mnt/shared-storage-user/mllm/zhangbeichen/Pref-GRPO
export PYTHONPATH=/mnt/shared-storage-user/mllm/zhangbeichen/Pref-GRPO:$PYTHONPATH

export DIFFUSERS_NO_BNB=1
export WANDB_MODE=offline
export WANDB_API_KEY=1697adda76308c9a62b666be1b8caeb1a39704a3
export WANDB_PROJECT="Pref-GRPO-flux2-klein"
export WANDB_NAME="grpo_correctness_guidance"
export WANDB_DIR="wandb/${WANDB_NAME}"
wandb login
echo "now wandb dir:${WANDB_DIR}"
mkdir -p $WANDB_DIR


source "scripts/_common_finetune.sh"


export EXP_NAME=${WANDB_NAME}

GUIDANCE_REWARD_HOST="10.102.198.20"
GUIDANCE_REWARD_BASE_PORT=10000
GUIDANCE_REWARD_NUM_SERVERS=2
GUIDANCE_REWARD_MODEL_NAME="qwen3"
GUIDANCE_REWARD_NUM_THREADS=64

OUTPUT_DIR="outputs/flux2_klein/${EXP_NAME}"
mkdir -p $OUTPUT_DIR
DATA_JSON_PATH="/mnt/shared-storage-user/mllm/zhangbeichen/Pref-GRPO/data/rldata_0402/edit_data.json"
EMBED_DIR="/mnt/shared-storage-user/mllm/zhangbeichen/Pref-GRPO/data/rl_0402"



TRAIN_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --pretrained_model_name_or_path /mnt/shared-storage-user/mllm/zhangbeichen/results_klein/safetensors/models--black-forest-labs--FLUX.2-klein-base-9B-lora/snapshots/17c3b160520b7dd44665dbf0b9ed9dd30c15cd06
  --data_json_path "${DATA_JSON_PATH}"
  --embed_dir "${EMBED_DIR}"
  --exp_name "${EXP_NAME}"
  --num_sample 100000
  --train_batch_size 1
  --train_guidance_scale 1.0
  --dataloader_num_workers 4
  --checkpointing_steps 60
  --num_train_epochs 1
  --learning_rate 1e-5
  --output_dir "${OUTPUT_DIR}"
  --t 1
  --sampling_steps 25
  --eta 0.7
  --num_generations 8
  --gradient_accumulation_steps 4
  --timestep_fraction 0.6
  --eval_every_steps 60
  --eval_num_prompts 64
  --eval_guidance_scale 4.0
  --eval_num_inference_steps 60
  --apply_gdpo
  --use_lora
  --lora_rank 64
  --lora_alpha 128
  --check_qa
  --reward_spec '{"guidance_reward": 0.5, "correctness_reward": 0.5}'
  --guidance_reward_host "${GUIDANCE_REWARD_HOST}"
  --guidance_reward_base_port ${GUIDANCE_REWARD_BASE_PORT}
  --guidance_reward_num_servers ${GUIDANCE_REWARD_NUM_SERVERS}
  --guidance_reward_model_name "${GUIDANCE_REWARD_MODEL_NAME}"
  --guidance_reward_num_threads ${GUIDANCE_REWARD_NUM_THREADS}
  --guidance_reward_temperature 0.7
  --guidance_reward_qa_num 1
  --open_end_reward
)



export NNODES=${NODE_COUNT:-1}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_PORT=8081

torchrun --nnodes=$NNODES --nproc_per_node=$PROC_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  fastvideo/train_flux2_klein_edit.py \
  "${TRAIN_ARGS[@]}"
