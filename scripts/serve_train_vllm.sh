#!/usr/bin/env bash
# TRL vLLM generation server for GRPO server mode. Pin to the inference GPU:
#   CUDA_VISIBLE_DEVICES=1 scripts/serve_train_vllm.sh [model]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:-Qwen/Qwen3.5-4B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

if [ ! -x ".venv-train/bin/trl" ]; then
  echo "Missing .venv-train/bin/trl; run scripts/create_train_venv.sh first." >&2
  exit 2
fi

# Same WSL workarounds as the serving venv's customized activate:
# pip-wheel CUDA 13 visible to native extensions (nvcc, libnvJitLink), and
# the FlashInfer sampling JIT disabled (fails on this CUDA 13/Blackwell stack).
CU13="$ROOT/.venv-train/lib/python3.12/site-packages/nvidia/cu13"
if [ -d "$CU13" ]; then
  export CUDA_HOME="$CU13"
  if [ -d "$CUDA_HOME/bin" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
  if [ -d "$CUDA_HOME/lib" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
  fi
fi
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Starting TRL vLLM server for $MODEL on port $PORT (gpu_util=$GPU_UTIL, max_len=$MAX_LEN)"
exec .venv-train/bin/trl vllm-serve --model "$MODEL" --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN"
