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
MODEL_NAME="qwen3"
MODEL_PATH="/mnt/shared-storage-user/mllm/xinglong/LLaMA-Factory-mllm-main/mllm_ckpts/Qwen3-VL-8B-Instruct"

# 服务器配置
MAX_MODEL_LEN=1500
GPU_MEMORY_UTILIZATION=0.95
HOST="0.0.0.0"
BASE_PORT=10000  # 端口范围: 10000-10007

# GPU 数量
export PROC_PER_NODE=${PROC_PER_NODE:-8}
NUM_GPUS=${PROC_PER_NODE}

# 基本可达性检查
if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "[ERROR] 找不到 ${MODEL_PATH}/config.json（模型目录不可达或路径写错）" >&2
  exit 1
fi

echo "=========================================="
echo "启动 ${NUM_GPUS} 个 vLLM Server 实例"
echo "模型: ${MODEL_NAME}"
echo "模型路径: ${MODEL_PATH}"
echo "端口范围: ${BASE_PORT} - $((BASE_PORT + NUM_GPUS - 1))"
echo "=========================================="

# 启动 8 个独立的 vLLM server，每个使用一张 GPU
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    
    echo "启动 GPU ${GPU_ID} 上的服务，端口: ${PORT}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${MODEL_PATH} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --served-model-name ${MODEL_NAME} \
        --host ${HOST} \
        --port ${PORT} \
        --enable-prefix-caching \
        --tensor-parallel-size 1 \
        --allowed-local-media-path / \
        &
    
    # 等待一下，避免同时启动造成资源竞争
    sleep 5
done

echo "=========================================="
echo "所有服务已在后台启动"
echo "端口列表: ${BASE_PORT} - $((BASE_PORT + NUM_GPUS - 1))"
echo "=========================================="

# 等待所有后台进程
wait
