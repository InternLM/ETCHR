#!/bin/bash
set -x


# CUDA 环境配置
export PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/shared-storage-user/mllm/xinglong/Envs/cuda-12.8



# source /mnt/shared-storage-user/mllm/xinglong/.bashrc

# activate conda env
source /mnt/shared-storage-user/mllm/xinglong/miniconda3/bin/activate Dance

echo "current conda env: $CONDA_DEFAULT_ENV"

# 强制重新安装 opencv-python-headless 解决共享存储同步问题
# pip install --force-reinstall opencv-python-headless -q

cd /mnt/shared-storage-user/mllm/xinglong/Pref-GRPO
export PYTHONPATH=/mnt/shared-storage-user/mllm/xinglong/Pref-GRPO:$PYTHONPATH


GPU_NUM=8 # 2,4,8
MODEL_PATH="/mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/FLUX.1-dev"

prompt_name="unigenbench_train_prompt"
OUTPUT_DIR="data/${prompt_name}_rl_embeddings"
# 如果没有建立这个目录，需要手动创建
mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "/mnt/shared-storage-user/mllm/xinglong/UniGenBench/data/train_prompt.txt"
