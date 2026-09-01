#!/usr/bin/env bash
# Primitive: ONE capped Dr. GRPO training run on GPU 0 (vLLM must already be
# serving on GPU 1). Used directly for a single seed, or by the run_multiseed.sh
# orchestrator. All knobs are env vars so the same script covers a quick test,
# a full-epoch run, or a resumed continuation.
#
#   TRAIN_OUT                 output dir (required in practice; default below)
#   TRAIN_SEED                trainer/data seed (default 0)
#   TRAIN_MAX_STEPS           optimizer steps (default 200)
#   TRAIN_SAVE_STEPS          checkpoint every N steps (default 50)
#   TRAIN_CAP                 inner wall-clock cap, e.g. 10800s (default 3h30m)
#   TRAIN_RESUME              HF checkpoint dir to resume from (default: none)
#   TRAIN_ADAPTER_PATH        PEFT adapter to load as trainable weights with a
#                             fresh optimizer/global step (default: none).
#                             Mutually exclusive with TRAIN_RESUME.
#   TRAIN_SPLIT               task split (default train)
#   TRAIN_TASK_IDS            explicit task ids (overrides split when set)
#   TRAIN_CUDA_EMPTY_CACHE_STEPS  default 1
#   TRAIN_SHUFFLE_DATASET     1 preserves historical shuffled sampling; 0
#                             preserves the supplied task-id order (default 1)
#   TRAIN_TEMP_THRESHOLD_C    optional pre-launch thermal gate: wait until GPU
#                             temp <= this before starting the trainer (default
#                             0 = off; run_multiseed.sh already gates between
#                             seeds, so leave off there). Bounded by
#                             TRAIN_TEMP_MAX_WAIT_SEC (default 1800) — never
#                             deadlocks; relies on the hardware throttle if the
#                             card stays hot or the sensor is unreadable.
#
# Guarantees: hard wall-clock cap, periodic checkpoints, and cleanup (stop vLLM,
# prune lp-rl containers, print GPU) on ANY exit. The thermal gate (if enabled)
# only runs BEFORE the trainer starts — never mid-step.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"  # cd-to-root + shared helpers
lp_pin_path

OUT="${TRAIN_OUT:-logs/artifacts/train_$(date +%F_%H%M%S)}"
SEED="${TRAIN_SEED:-0}"
MAX_STEPS="${TRAIN_MAX_STEPS:-200}"
SAVE_STEPS="${TRAIN_SAVE_STEPS:-50}"
CAP="${TRAIN_CAP:-12600s}"
RESUME="${TRAIN_RESUME:-}"
ADAPTER_PATH="${TRAIN_ADAPTER_PATH:-}"
SPLIT="${TRAIN_SPLIT:-train}"
TASK_IDS="${TRAIN_TASK_IDS:-}"
SHUFFLE_DATASET="${TRAIN_SHUFFLE_DATASET:-1}"
CUDA_EMPTY_CACHE_STEPS="${TRAIN_CUDA_EMPTY_CACHE_STEPS:-1}"
MONITOR_CLIPPED_MAX="${TRAIN_MONITOR_CLIPPED_MAX:-0.5}"  # 1.01 disables the clip-only abort (verbose tasks legitimately clip); ISR/GPU/container guards unchanged
TEMP_THRESHOLD_C="${TRAIN_TEMP_THRESHOLD_C:-0}"
TEMP_MAX_WAIT_SEC="${TRAIN_TEMP_MAX_WAIT_SEC:-1800}"
TEMP_POLL_SEC="${TRAIN_TEMP_POLL_SEC:-30}"
mkdir -p "$OUT"
LOG="$OUT/train.log"

TASK_ARGS=()
[ -n "$TASK_IDS" ] && TASK_ARGS=(--task-ids "$TASK_IDS") || TASK_ARGS=(--split "$SPLIT")
case "$SHUFFLE_DATASET" in
  1) SHUFFLE_ARGS=(--shuffle-dataset) ;;
  0) SHUFFLE_ARGS=(--no-shuffle-dataset) ;;
  *) echo "[train] TRAIN_SHUFFLE_DATASET must be 0 or 1" >&2; exit 2 ;;
esac


if [ -n "$RESUME" ] && [ -n "$ADAPTER_PATH" ]; then
  echo "[train] TRAIN_RESUME and TRAIN_ADAPTER_PATH are mutually exclusive" >&2
  exit 2
fi
if [ -n "$ADAPTER_PATH" ] && [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
  echo "[train] missing adapter weights: $ADAPTER_PATH/adapter_model.safetensors" >&2
  exit 2
fi

ADAPTER_ARGS=()
[ -n "$ADAPTER_PATH" ] && ADAPTER_ARGS=(--adapter-path "$ADAPTER_PATH")
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume-from-checkpoint "$RESUME")

lp_install_cleanup_trap "$LOG" train gpu

{
  echo "[train] start $(date -Is) cap=$CAP out=$OUT"
  echo "[train] config: ${TASK_ARGS[*]} seed=$SEED max-steps=$MAX_STEPS save-steps=$SAVE_STEPS 4096/1/8/8 tool-iters=20 token_truncate lora shuffle-dataset=$SHUFFLE_DATASET cuda-empty-cache-steps=$CUDA_EMPTY_CACHE_STEPS resume=${RESUME:-none} adapter-path=${ADAPTER_PATH:-none}"
} >>"$LOG" 2>&1
START=$(date +%s)

lp_temp_gate "$TEMP_THRESHOLD_C" "$TEMP_MAX_WAIT_SEC" "$TEMP_POLL_SEC" "$LOG" train

LP_RL_SANDBOX=docker CUDA_VISIBLE_DEVICES=0 \
  timeout --signal=INT --kill-after=180 "$CAP" \
  .venv-train/bin/python train/train_trl_drgrpo.py \
    --seed "$SEED" \
    "${TASK_ARGS[@]}" \
    --max-steps "$MAX_STEPS" \
    "${SHUFFLE_ARGS[@]}" \
    --vllm-mode server \
    --vllm-server-host 127.0.0.1 \
    --vllm-server-port 8000 \
    --save-strategy steps \
    --save-steps "$SAVE_STEPS" \
    --cuda-empty-cache-steps "$CUDA_EMPTY_CACHE_STEPS" \
    --monitor-clipped-max "$MONITOR_CLIPPED_MAX" \
    --output-dir "$OUT" \
    "${ADAPTER_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
  >>"$LOG" 2>&1
RC=$?

END=$(date +%s)
{
  echo "[train] trainer exit rc=$RC elapsed=$((END-START))s at $(date -Is)"
  if [ "$RC" = "124" ] || [ "$RC" = "137" ]; then
    echo "[train] NOTE: hit hard wall-clock cap ($CAP); stopped by timeout (rc=$RC)."
  fi
} >>"$LOG" 2>&1

exit "$RC"
