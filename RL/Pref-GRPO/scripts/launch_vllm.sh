#!/bin/bash
set -x


MODEL_NAME="qwen3"
MODEL_PATH="YOUR_MODEL_PATH"

MAX_MODEL_LEN=10240
GPU_MEMORY_UTILIZATION=0.8
HOST="0.0.0.0"
BASE_PORT=10000 

NUM_GPUS=2

if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "[ERROR] Cannot Find Model Path" >&2
  exit 1
fi

echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Model Path: ${MODEL_PATH}"
echo "Port: ${BASE_PORT}"
echo "=========================================="

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
        
    CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${MODEL_PATH} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --served-model-name ${MODEL_NAME} \
        --host ${HOST} \
        --port ${PORT} \
        --tensor-parallel-size 1 \
        &
    
    sleep 15
done

echo "=========================================="
echo "VLLM Server Start"
echo "=========================================="

wait