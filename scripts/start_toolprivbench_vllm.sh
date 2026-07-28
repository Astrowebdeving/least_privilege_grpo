#!/usr/bin/env bash
# Serve the base model plus all final LoRA adapters for ToolPrivBench through
# vLLM's OpenAI-compatible tool-calling API. This deliberately uses GPU 0;
# GPU 1 may be occupied by an unrelated Windows/shared-user process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${TOOLPRIV_VLLM_PORT:-8010}"
GPU="${TOOLPRIV_VLLM_GPU:-0}"
GPU_UTIL="${TOOLPRIV_VLLM_GPU_UTIL:-0.85}"
MAX_LEN="${TOOLPRIV_VLLM_MAX_LEN:-8192}"
LOG="${TOOLPRIV_VLLM_LOG:-logs/artifacts/toolprivbench_vllm.log}"
PIDFILE="${TOOLPRIV_VLLM_PIDFILE:-logs/artifacts/toolprivbench_vllm.pid}"
RUN_ROOT="${TOOLPRIV_ADAPTER_ROOT:-logs/artifacts/bigrun_v2_2026-07-01_003238}"
UNIT="${TOOLPRIV_VLLM_UNIT:-lp-toolpriv-vllm}"
START_MODE="${TOOLPRIV_VLLM_START_MODE:-systemd}"

# TOOLPRIV_LORA_SPECS overrides the served adapter set (comma-separated
# name=path list; used by the checkpoint-trajectory ablation). Default is the
# registered final checkpoints, unchanged.
MAX_LORAS="${TOOLPRIV_MAX_LORAS:-3}"
if [ -n "${TOOLPRIV_LORA_SPECS:-}" ]; then
  IFS=',' read -r -a LORAS <<< "$TOOLPRIV_LORA_SPECS"
else
  declare -a LORAS=(
    "seed-0=$RUN_ROOT/seed-0/checkpoint-1500"
    "seed-1=$RUN_ROOT/seed-1/checkpoint-1500"
    "seed-2=$RUN_ROOT/seed-2/checkpoint-1500"
  )
fi

for spec in "${LORAS[@]}"; do
  path="${spec#*=}"
  [ -f "$path/adapter_model.safetensors" ] || {
    echo "missing adapter: $path" >&2
    exit 2
  }
done

if [ -f "$PIDFILE" ]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "${old_pid:-}" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ToolPrivBench vLLM already running pid=$old_pid port=$PORT"
    exit 0
  fi
fi

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

CU13="$ROOT/.venv-train/lib/python3.12/site-packages/nvidia/cu13"
if [ -d "$CU13" ]; then
  export CUDA_HOME="$CU13"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0

ARGS=(
  .venv-train/bin/python -m vllm.entrypoints.openai.api_server
  --model Qwen/Qwen3.5-4B
  --served-model-name base
  --host 127.0.0.1
  --port "$PORT"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN"
  --enable-lora
  --max-loras "$MAX_LORAS"
  --max-cpu-loras "$MAX_LORAS"
  --max-lora-rank 32
  --lora-modules "${LORAS[@]}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
)

if [ "${TOOLPRIV_VLLM_FOREGROUND:-0}" = "1" ]; then
  exec env CUDA_VISIBLE_DEVICES="$GPU" "${ARGS[@]}"
fi

if [ "$START_MODE" != "nohup" ] && command -v systemd-run >/dev/null 2>&1 \
  && systemctl --user is-system-running >/dev/null 2>&1; then
  systemctl --user reset-failed "${UNIT}.service" >/dev/null 2>&1 || true
  log_abs="$(realpath -m "$LOG")"
  systemd-run --user \
    --unit="$UNIT" \
    --collect \
    --property="WorkingDirectory=$ROOT" \
    --property="StandardOutput=append:$log_abs" \
    --property="StandardError=append:$log_abs" \
    --setenv="TOOLPRIV_VLLM_FOREGROUND=1" \
    --setenv="TOOLPRIV_VLLM_PORT=$PORT" \
    --setenv="TOOLPRIV_VLLM_GPU=$GPU" \
    --setenv="TOOLPRIV_VLLM_GPU_UTIL=$GPU_UTIL" \
    --setenv="TOOLPRIV_VLLM_MAX_LEN=$MAX_LEN" \
    --setenv="TOOLPRIV_VLLM_LOG=$LOG" \
    --setenv="TOOLPRIV_VLLM_PIDFILE=$PIDFILE" \
    --setenv="TOOLPRIV_ADAPTER_ROOT=$RUN_ROOT" \
    --setenv="TOOLPRIV_LORA_SPECS=${TOOLPRIV_LORA_SPECS:-}" \
    --setenv="TOOLPRIV_MAX_LORAS=$MAX_LORAS" \
    "$ROOT/scripts/start_toolprivbench_vllm.sh"
  sleep 1
  systemctl --user show "${UNIT}.service" --property=MainPID --value > "$PIDFILE"
  pid="$(cat "$PIDFILE")"
else
  setsid env CUDA_VISIBLE_DEVICES="$GPU" nohup "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PIDFILE"
fi
echo "started ToolPrivBench vLLM pid=$pid gpu=$GPU port=$PORT log=$LOG mode=$START_MODE"