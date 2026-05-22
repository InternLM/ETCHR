#!/bin/bash
set -x

export PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8

# activate conda env
cd /root/miniconda3/bin
source activate lmm-r1
conda activate lmm-r1

cd /mnt/shared-storage-user/mllm/xinglong/Pref-GRPO

export PYTHONPATH=/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO:$PYTHONPATH


# 模型配置
MODEL_PATH="/mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/models--CodeGoat24--UnifiedReward-2.0-qwen3vl-8b/snapshots/4175e6b78d62f639282797d05ccea35d3be823cf"


vllm serve $MODEL_PATH \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt.image 32 \
    --port 8081 \
    --enable-prefix-caching \
    --disable-log-requests \
    --mm_processor_cache_gb=500
