#!/usr/bin/env bash
# Orchestrator: sequential multi-seed training over the train split, calling the
# run_train_capped.sh primitive once per seed. Restarts vLLM (GPU 1, util ~0.85)
# before each seed and writes rollout/memory summaries afterward.
#
# RESUME IS AUTOMATIC: for each seed it skips one already at MAX_STEPS, resumes
# from the newest full HF checkpoint if one exists below MAX_STEPS, or starts
# fresh otherwise. So this single script covers a brand-new run, a continuation
# of a capped run, and a quick test (set MULTISEED_MAX_STEPS low).
#
#   MULTISEED_OUT_ROOT        run dir (default logs/artifacts/multiseed_<ts>)
#   MULTISEED_SEEDS           default 0,1,2
#   MULTISEED_MAX_STEPS       default = train task count (one full epoch)
#   MULTISEED_SAVE_STEPS      default 50
#   MULTISEED_SEED_CAP        per-seed inner cap (default 18000s)
#   MULTISEED_VLLM_GPU_UTIL   default 0.85
#   MULTISEED_DRY_RUN=1       enumerate seeds/checkpoints, launch nothing
#
# Thermal guard (OPERATIONAL protection only, enforced at safe boundaries —
# between seeds and before each trainer launch, never mid-step; see
# docs/task_authoring_standard.md §10). All default OFF so quick tests and the
# historical 200-step runs behave exactly as before; the week-long campaign sets
# them explicitly.
#   MULTISEED_COOLDOWN_SEC    fixed rest BETWEEN seeds (default 0; campaign ~7200)
#   MULTISEED_TEMP_THRESHOLD_C  soft pre-launch gate: wait until GPU temp <= this
#                               before launching a seed (default 0 = disabled; ~83)
#   MULTISEED_TEMP_MAX_WAIT_SEC max wait for the soft gate before proceeding
#                               anyway — NEVER deadlock a multi-day run (default 1800)
#   MULTISEED_TEMP_POLL_SEC     gate poll interval (default 30)
#
# Launch detached (outer systemd cap covers all seeds AND all cooldowns — size
# RuntimeMaxSec to include them):
#   OUT=logs/artifacts/multiseed_$(date +%F_%H%M%S)
#   systemd-run --user --unit=lp-rl-multiseed --collect \
#     --property=WorkingDirectory=<repo> \
#     --property=RuntimeMaxSec=57600 --property=KillSignal=SIGTERM \
#     --property=TimeoutStopSec=180 \
#     --setenv=MULTISEED_OUT_ROOT="$OUT" \
#     bash <repo>/scripts/run_multiseed.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"  # cd-to-root + shared helpers
lp_pin_path
export VLLM_GPU_UTIL="${MULTISEED_VLLM_GPU_UTIL:-0.85}"

SEEDS="${MULTISEED_SEEDS:-0,1,2}"
OUT_ROOT="${MULTISEED_OUT_ROOT:-logs/artifacts/multiseed_$(date +%F_%H%M%S)}"
SAVE_STEPS="${MULTISEED_SAVE_STEPS:-50}"
SEED_CAP="${MULTISEED_SEED_CAP:-18000s}"
CUDA_EMPTY_CACHE_STEPS="${MULTISEED_CUDA_EMPTY_CACHE_STEPS:-1}"
DRY_RUN="${MULTISEED_DRY_RUN:-0}"
COOLDOWN_SEC="${MULTISEED_COOLDOWN_SEC:-0}"
TEMP_THRESHOLD_C="${MULTISEED_TEMP_THRESHOLD_C:-0}"
TEMP_MAX_WAIT_SEC="${MULTISEED_TEMP_MAX_WAIT_SEC:-1800}"
TEMP_POLL_SEC="${MULTISEED_TEMP_POLL_SEC:-30}"

TRAIN_TASK_COUNT="$(PYTHONPATH=. .venv/bin/python -c \
  "from least_privilege_env.tasks.loader import load_tasks; print(sum(t.split=='train' for t in load_tasks()))")" || {
  echo "[multiseed] failed to count train tasks" >&2; exit 2; }
MAX_STEPS="${MULTISEED_MAX_STEPS:-$TRAIN_TASK_COUNT}"
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "[multiseed] invalid max steps: $MAX_STEPS" >&2; exit 2; }

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/multiseed.log"
echo "$OUT_ROOT" > logs/artifacts/.multiseed_current_out_root

cleanup() {
  [ "$DRY_RUN" = "1" ] && return 0
  scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
  scripts/prune_lp_containers.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

latest_full_ckpt() {  # echo newest checkpoint-N dir (N<MAX) that has full HF state
  local dir="$1" best="" n
  for c in "$dir"/checkpoint-*; do
    [ -d "$c" ] || continue
    n="${c##*/checkpoint-}"; [[ "$n" =~ ^[0-9]+$ ]] || continue
    [ "$n" -ge "$MAX_STEPS" ] && continue
    [ -f "$c/optimizer.pt" ] && [ -f "$c/trainer_state.json" ] || continue
    if [ -z "$best" ] || [ "$n" -gt "${best##*/checkpoint-}" ]; then best="$c"; fi
  done
  echo "$best"
}

THERMAL_LOG="$OUT_ROOT/thermal_events.jsonl"

thermal_event() {  # thermal_event EVENT DETAIL — structured, never breaks the run
  local event="$1" detail="$2" temp
  temp="$(lp_gpu_max_temp || true)"
  printf '{"ts":"%s","event":"%s","gpu_max_temp_c":"%s","detail":"%s"}\n' \
    "$(date -Is)" "$event" "${temp:-unknown}" "$detail" >>"$THERMAL_LOG" 2>/dev/null || true
  echo "[multiseed][thermal] $event temp=${temp:-unknown}C $detail" | tee -a "$LOG"
}

inter_seed_cooldown() {  # fixed rest between seeds; safe boundary (no trainer mid-step)
  [ "$COOLDOWN_SEC" -gt 0 ] 2>/dev/null || return 0
  if [ "$DRY_RUN" = "1" ]; then thermal_event "cooldown_skip_dryrun" "would sleep ${COOLDOWN_SEC}s"; return 0; fi
  thermal_event "cooldown_begin" "sleeping ${COOLDOWN_SEC}s between seeds"
  local waited=0
  while [ "$waited" -lt "$COOLDOWN_SEC" ]; do
    sleep "$TEMP_POLL_SEC"; waited=$((waited + TEMP_POLL_SEC))
  done
  thermal_event "cooldown_end" "rested ${COOLDOWN_SEC}s"
}

pre_seed_temp_gate() {  # soft wait until temp<=threshold; bounded so it can NEVER deadlock
  [ "$TEMP_THRESHOLD_C" -gt 0 ] 2>/dev/null || return 0
  [ "$DRY_RUN" = "1" ] && { thermal_event "temp_gate_skip_dryrun" "threshold=${TEMP_THRESHOLD_C}C"; return 0; }
  local waited=0 temp
  temp="$(lp_gpu_max_temp || true)"
  if [ -z "$temp" ]; then
    thermal_event "temp_unreadable" "nvidia-smi gave no temp; relying on hardware throttle, proceeding"
    return 0
  fi
  if [ "$temp" -le "$TEMP_THRESHOLD_C" ]; then
    thermal_event "temp_ok" "temp=${temp}C <= ${TEMP_THRESHOLD_C}C"
    return 0
  fi
  thermal_event "temp_gate_wait" "temp=${temp}C > ${TEMP_THRESHOLD_C}C; waiting up to ${TEMP_MAX_WAIT_SEC}s"
  while [ "$waited" -lt "$TEMP_MAX_WAIT_SEC" ]; do
    sleep "$TEMP_POLL_SEC"; waited=$((waited + TEMP_POLL_SEC))
    temp="$(lp_gpu_max_temp || true)"
    [ -z "$temp" ] && { thermal_event "temp_unreadable_midwait" "proceeding"; return 0; }
    if [ "$temp" -le "$TEMP_THRESHOLD_C" ]; then
      thermal_event "temp_gate_clear" "temp=${temp}C after ${waited}s"; return 0
    fi
  done
  thermal_event "temp_gate_timeout" "temp=${temp}C still > ${TEMP_THRESHOLD_C}C after ${TEMP_MAX_WAIT_SEC}s; proceeding (hardware throttle backstop)"
}

IFS=',' read -r -a SEED_LIST <<<"$SEEDS"
launched_any=0
{
  echo "[multiseed] start $(date -Is)"
  echo "[multiseed] seeds=$SEEDS train-tasks=$TRAIN_TASK_COUNT max-steps=$MAX_STEPS save-steps=$SAVE_STEPS seed-cap=$SEED_CAP vllm-util=$VLLM_GPU_UTIL dry-run=$DRY_RUN"
  echo "[multiseed] thermal: cooldown=${COOLDOWN_SEC}s temp-threshold=${TEMP_THRESHOLD_C}C max-wait=${TEMP_MAX_WAIT_SEC}s poll=${TEMP_POLL_SEC}s"
} | tee -a "$LOG"

for raw_seed in "${SEED_LIST[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "[multiseed] invalid seed: $raw_seed" | tee -a "$LOG"; exit 2; }
  out="$OUT_ROOT/seed-$seed"

  if [ -f "$out/checkpoint-$MAX_STEPS/adapter_model.safetensors" ]; then
    echo "[multiseed] seed=$seed already at checkpoint-$MAX_STEPS; skipping" | tee -a "$LOG"; continue
  fi
  resume=""
  if [ -d "$out" ]; then
    resume="$(latest_full_ckpt "$out")"
    if [ -z "$resume" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
      echo "[multiseed] seed=$seed dir exists but has no resumable checkpoint; refusing (move it aside)" | tee -a "$LOG"; exit 3
    fi
  fi

  # Safe-boundary thermal guard: rest BETWEEN seeds (not before the first, not
  # around skipped seeds), then verify the card is cool enough before launching.
  # Both no-op unless configured; both can never deadlock the run.
  if [ "$launched_any" = "1" ]; then inter_seed_cooldown; fi
  pre_seed_temp_gate
  launched_any=1

  if [ "$DRY_RUN" = "1" ]; then
    echo "[multiseed] dry-run seed=$seed out=$out resume=${resume:-fresh} -> checkpoint-$MAX_STEPS" | tee -a "$LOG"; continue
  fi

  echo "[multiseed] seed=$seed start $(date -Is) resume=${resume:-fresh}" | tee -a "$LOG"
  if ! lp_vllm_restart 120 "$LOG"; then
    echo "[multiseed] seed=$seed vLLM start/readiness failed" | tee -a "$LOG"; exit 4; fi

  TRAIN_OUT="$out" TRAIN_SEED="$seed" TRAIN_MAX_STEPS="$MAX_STEPS" \
  TRAIN_SAVE_STEPS="$SAVE_STEPS" TRAIN_CAP="$SEED_CAP" TRAIN_RESUME="$resume" \
  TRAIN_CUDA_EMPTY_CACHE_STEPS="$CUDA_EMPTY_CACHE_STEPS" \
    bash scripts/run_train_capped.sh
  rc=$?
  echo "[multiseed] seed=$seed trainer-rc=$rc $(date -Is)" | tee -a "$LOG"
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$rc" -ne 137 ]; then
    echo "[multiseed] seed=$seed crash rc=$rc; stopping" | tee -a "$LOG"; exit "$rc"; fi
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && echo "[multiseed] seed=$seed hit inner cap before $MAX_STEPS (resumable)" | tee -a "$LOG"

  PYTHONPATH=. .venv/bin/python scripts/summarize_rollouts.py "$out/rollouts" \
    --archive data/risk_archive.json --report "$out/rollout_review.json" >>"$LOG" 2>&1 \
    || echo "[multiseed] seed=$seed rollout summary failed" | tee -a "$LOG"
  PYTHONPATH=. .venv/bin/python scripts/summarize_gpu_memory.py "$out/gpu_memory.jsonl" \
    --rollouts "$out/rollouts" --report "$out/gpu_memory_summary.json" >>"$LOG" 2>&1 || true

  if [ -f "$out/checkpoint-$MAX_STEPS/adapter_model.safetensors" ]; then
    echo "[multiseed] seed=$seed complete checkpoint-$MAX_STEPS $(date -Is)" | tee -a "$LOG"
  else
    echo "[multiseed] seed=$seed did NOT reach checkpoint-$MAX_STEPS (capped; rerun to resume) $(date -Is)" | tee -a "$LOG"
  fi
done
echo "[multiseed] all requested seeds processed $(date -Is)" | tee -a "$LOG"
