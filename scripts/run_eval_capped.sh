#!/usr/bin/env bash
# Primitive: ONE capped frozen evaluation/diagnostic run on GPU 0 (vLLM must
# already be serving on GPU 1). learning-rate 0, no checkpoint saving, Docker
# sandbox, server-mode vLLM, optional PEFT adapter. Used directly for a single
# adapter, or by the run_eval_suite.sh orchestrator.
#
#   EVAL_LABEL                run label (default base)
#   EVAL_OUT                  output dir
#   EVAL_ADAPTER_PATH         PEFT adapter dir (default: base model, no adapter)
#   EVAL_TASK_IDS             comma task ids (default: data/validation_task_ids.txt)
#   EVAL_REPEAT               repeat the task list N times (default 1)
#   EVAL_NUM_GENERATIONS      rollouts sampled per task/step (default 8)
#   EVAL_SEED                 sampling seed (default 0)
#   EVAL_SYSTEM_PROMPT_VARIANT full | para | min | none (default full)
#   EVAL_CAP                  inner wall-clock cap (default 3h)
#   EVAL_MONITOR_CLIPPED_MAX  completion-clipping abort threshold (default 0.5;
#                             set 1.01 for verbose-policy measurement while
#                             preserving ISR/GPU/container abort guards)
#   EVAL_MODE                 normal | mem-observe | mem-mitigate (default normal)
#                             mem-* add a GPU-memory log + summary; observe turns
#                             off allocator mitigation and monitor aborts,
#                             mitigate forces expandable_segments + empty-cache 1.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"  # cd-to-root + shared helpers
lp_pin_path

DEFAULT_TASK_IDS="$(tr -d '\r\n' < data/validation_task_ids.txt)"
LABEL="${EVAL_LABEL:-base}"
OUT="${EVAL_OUT:-logs/artifacts/eval_${LABEL}_$(date +%F_%H%M%S)}"
CAP="${EVAL_CAP:-10800s}"
TASK_IDS="${EVAL_TASK_IDS:-$DEFAULT_TASK_IDS}"
REPEAT="${EVAL_REPEAT:-1}"
NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-8}"
SEED="${EVAL_SEED:-0}"
SYSTEM_PROMPT_VARIANT="${EVAL_SYSTEM_PROMPT_VARIANT:-full}"
ADAPTER_PATH="${EVAL_ADAPTER_PATH:-}"
MODE="${EVAL_MODE:-normal}"
CUDA_EMPTY_CACHE_STEPS="${EVAL_CUDA_EMPTY_CACHE_STEPS:-1}"
MONITOR_CLIPPED_MAX="${EVAL_MONITOR_CLIPPED_MAX:-0.5}"

case "$MODE" in
  normal) EXTRA_ARGS=() ;;
  mem-observe)
    unset PYTORCH_CUDA_ALLOC_CONF
    CUDA_EMPTY_CACHE_STEPS=0
    EXTRA_ARGS=(--no-monitor-abort) ;;
  mem-mitigate)
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+$PYTORCH_CUDA_ALLOC_CONF,}expandable_segments:True"
    CUDA_EMPTY_CACHE_STEPS=1
    EXTRA_ARGS=() ;;
  *) echo "[eval] EVAL_MODE must be normal|mem-observe|mem-mitigate, got: $MODE" >&2; exit 2 ;;
esac

if ! [[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then echo "[eval] invalid repeat: $REPEAT" >&2; exit 2; fi
if ! [[ "$NUM_GENERATIONS" =~ ^[1-9][0-9]*$ ]]; then echo "[eval] invalid num-generations: $NUM_GENERATIONS" >&2; exit 2; fi
TASK_COUNT="$(lp_count_csv "$TASK_IDS")"
[ "$TASK_COUNT" -eq 0 ] && { echo "[eval] no task ids requested" >&2; exit 2; }
# Episodes per optimizer step = the generation batch = per_device_train_batch_size
# (1) * gradient_accumulation_steps (8) = 8, INDEPENDENT of num-generations: TRL
# samples 8/num_generations distinct prompts per step, each repeated
# num_generations times. So covering TASK_COUNT*REPEAT tasks at NUM_GENERATIONS
# generations each needs TASK_COUNT*REPEAT*NUM_GENERATIONS/8 steps. The old
# formula (TASK_COUNT*REPEAT) is correct ONLY at num_generations=8; at 4 it
# silently ran each task twice (2208 vs 1104 in the frozen-500 ordinary tier).
GEN_BATCH="${EVAL_GEN_BATCH:-8}"
if [ $((GEN_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
  echo "[eval] generation batch $GEN_BATCH not divisible by num-generations $NUM_GENERATIONS" >&2; exit 2
fi
TOTAL_EPISODES=$((TASK_COUNT * REPEAT * NUM_GENERATIONS))
if [ $((TOTAL_EPISODES % GEN_BATCH)) -ne 0 ]; then
  echo "[eval] task_count*repeat*num_generations=$TOTAL_EPISODES not divisible by generation batch $GEN_BATCH; cannot produce exact membership" >&2
  exit 2
fi
MAX_STEPS="${EVAL_MAX_STEPS:-$((TOTAL_EPISODES / GEN_BATCH))}"

mkdir -p "$OUT"
LOG="$OUT/eval.log"

_cleaned=0
cleanup() {
  [ "$_cleaned" = "1" ] && return 0
  _cleaned=1
  {
    echo "[eval] cleanup begin $(date -Is)"
    scripts/stop_train_vllm.sh || true
    scripts/prune_lp_containers.sh || true
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
    if [[ "$MODE" == mem-* ]] && [ -f "$OUT/gpu_memory.jsonl" ]; then
      PYTHONPATH=. .venv/bin/python scripts/summarize_gpu_memory.py "$OUT/gpu_memory.jsonl" \
        --rollouts "$OUT/rollouts" --report "$OUT/gpu_memory_summary.json" || true
    fi
    echo "[eval] cleanup done $(date -Is)"
  } >>"$LOG" 2>&1
}
trap 'cleanup' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ADAPTER_ARGS=()
[ -n "$ADAPTER_PATH" ] && ADAPTER_ARGS=(--adapter-path "$ADAPTER_PATH")

{
  echo "[eval] start $(date -Is) mode=$MODE cap=$CAP label=$LABEL out=$OUT"
  echo "[eval] config: tasks=$TASK_COUNT max-steps=$MAX_STEPS repeat=$REPEAT num-generations=$NUM_GENERATIONS seed=$SEED lr=0 save-no cuda-empty-cache-steps=$CUDA_EMPTY_CACHE_STEPS monitor-clipped-max=$MONITOR_CLIPPED_MAX allocator=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
  echo "[eval] adapter-path=${ADAPTER_PATH:-none}"
} >>"$LOG" 2>&1
START=$(date +%s)

LP_RL_SANDBOX=docker CUDA_VISIBLE_DEVICES=0 \
  timeout --signal=INT --kill-after=180 "$CAP" \
  .venv-train/bin/python train/train_trl_drgrpo.py \
    --seed "$SEED" \
    --system-prompt-variant "$SYSTEM_PROMPT_VARIANT" \
    --learning-rate 0 \
    --max-steps "$MAX_STEPS" \
    --task-ids "$TASK_IDS" \
    --repeat "$REPEAT" \
    --num-generations "$NUM_GENERATIONS" \
    --vllm-mode server \
    --vllm-server-host 127.0.0.1 \
    --vllm-server-port 8000 \
    --save-strategy no \
    --cuda-empty-cache-steps "$CUDA_EMPTY_CACHE_STEPS" \
    --monitor-clipped-max "$MONITOR_CLIPPED_MAX" \
    --output-dir "$OUT" \
    "${ADAPTER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
  >>"$LOG" 2>&1
RC=$?

END=$(date +%s)
{
  echo "[eval] trainer exit rc=$RC elapsed=$((END-START))s at $(date -Is)"
  [ "$RC" = "124" ] || [ "$RC" = "137" ] && echo "[eval] NOTE: hit hard wall-clock cap ($CAP)."
} >>"$LOG" 2>&1
exit "$RC"
