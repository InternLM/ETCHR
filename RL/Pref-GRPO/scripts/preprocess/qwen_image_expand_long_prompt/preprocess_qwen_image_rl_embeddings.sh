#!/bin/bash
set -x


# CUDA 环境配置
export PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8


source /mnt/shared-storage-user/mllm/xinglong/miniconda3/bin/activate Qimage

echo "current conda env: $CONDA_DEFAULT_ENV"

cd /mnt/shared-storage-user/mllm/xinglong/Pref-GRPO
export PYTHONPATH=/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO:$PYTHONPATH


export PROC_PER_NODE=${PROC_PER_NODE:-8}
GPU_NUM=${PROC_PER_NODE}
MODEL_PATH="/mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/Qwen-Image"





OUTPUT_DIR="/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/data/unigenbench_train_data_qwenimage"

# pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_qwenimage_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO/scripts/preprocess/qwen_image_expand_long_prompt/4_filter_group_acc_std0.075_5035_clean.jsonl" \
    --prompt_key "generated_prompt"

