#!/usr/bin/env bash
# Orchestrator: frozen evaluation of the base model plus one or more PEFT
# adapters on the same task list, calling the run_eval_capped.sh primitive once
# per policy. Restarts vLLM before each policy, guards episode counts and
# adapter loading, then compares each adapter to base and writes report-only
# safety metrics. Replaces the rebaseline / confirmatory / pilot-frozen eval
# launchers.
#
#   EVALSUITE_OUT_ROOT     run dir (default logs/artifacts/eval_suite_<ts>)
#   EVALSUITE_ADAPTERS     comma list of name=path, e.g.
#                          "seed-0=logs/.../seed-0/checkpoint-200,seed-1=..."
#   EVALSUITE_INCLUDE_BASE default 1 (also evaluate the un-adapted base)
#   EVALSUITE_TASK_IDS     default data/validation_task_ids.txt
#   LP_RL_TASK_FILE        active task catalog used by both the evaluator and
#                          report-only safety metrics (default data/spike_tasks.json)
#   EVALSUITE_SEED         sampling seed (default 0)
#   EVALSUITE_CAP          per-policy inner cap (default 10800s)
#   EVALSUITE_GENERATIONS  rollouts per task for the episode guard (default 8)
#   EVALSUITE_VLLM_GPU_UTIL default 0.85
#
# Launch detached: wrap in systemd-run --user like the other orchestrators.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"  # cd-to-root + shared helpers
lp_pin_path
export VLLM_GPU_UTIL="${EVALSUITE_VLLM_GPU_UTIL:-0.85}"

OUT_ROOT="${EVALSUITE_OUT_ROOT:-logs/artifacts/eval_suite_$(date +%F_%H%M%S)}"
ADAPTERS="${EVALSUITE_ADAPTERS:-}"
INCLUDE_BASE="${EVALSUITE_INCLUDE_BASE:-1}"
TASK_IDS="${EVALSUITE_TASK_IDS:-$(tr -d '\r\n' < data/validation_task_ids.txt)}"
TASK_FILE="${LP_RL_TASK_FILE:-data/spike_tasks.json}"
SEED="${EVALSUITE_SEED:-0}"
CAP="${EVALSUITE_CAP:-10800s}"
GENERATIONS="${EVALSUITE_GENERATIONS:-8}"

TASK_COUNT="$(lp_count_csv "$TASK_IDS")"
[ "$TASK_COUNT" -eq 0 ] && { echo "[evalsuite] no task ids" >&2; exit 2; }
[ -f "$TASK_FILE" ] || { echo "[evalsuite] task catalog not found: $TASK_FILE" >&2; exit 2; }
EXPECTED=$((TASK_COUNT * GENERATIONS))

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/eval_suite.log"

lp_install_cleanup_trap "$LOG" evalsuite

assert_episodes() {
  PYTHONPATH=. .venv/bin/python -c \
    "import json,sys; r=json.load(open('$1')); n=r['episode_count']; sys.exit(0 if n==$EXPECTED else (print(f'episode_count {n} != $EXPECTED') or 1))"
}

# Build the policy list: optional base + each name=path adapter.
declare -a NAMES PATHS
if [ "$INCLUDE_BASE" = "1" ]; then NAMES+=("base"); PATHS+=(""); fi
if [ -n "$ADAPTERS" ]; then
  IFS=',' read -r -a APAIRS <<<"$ADAPTERS"
  for pair in "${APAIRS[@]}"; do
    name="${pair%%=*}"; path="${pair#*=}"
    [ -n "$name" ] && [ -n "$path" ] || { echo "[evalsuite] bad adapter spec: $pair" | tee -a "$LOG"; exit 2; }
    [ -f "$path/adapter_model.safetensors" ] || { echo "[evalsuite] missing adapter: $path" | tee -a "$LOG"; exit 2; }
    NAMES+=("$name"); PATHS+=("$path")
  done
fi
[ "${#NAMES[@]}" -eq 0 ] && { echo "[evalsuite] no policies (set EVALSUITE_ADAPTERS or INCLUDE_BASE=1)" | tee -a "$LOG"; exit 2; }

{
  echo "[evalsuite] start $(date -Is) out=$OUT_ROOT"
  echo "[evalsuite] policies=${NAMES[*]} tasks=$TASK_COUNT task-catalog=$TASK_FILE expected-episodes=$EXPECTED seed=$SEED cap=$CAP"
} | tee -a "$LOG"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; apath="${PATHS[$i]}"; pout="$OUT_ROOT/$name"
  echo "[evalsuite] policy=$name start $(date -Is) adapter=${apath:-none}" | tee -a "$LOG"
  lp_vllm_restart 120 "$LOG" || { echo "[evalsuite] $name vLLM start failed" | tee -a "$LOG"; exit 4; }
  EVAL_LABEL="$name" EVAL_OUT="$pout" EVAL_ADAPTER_PATH="$apath" \
  EVAL_TASK_IDS="$TASK_IDS" EVAL_NUM_GENERATIONS="$GENERATIONS" \
  EVAL_SEED="$SEED" EVAL_CAP="$CAP" \
    bash scripts/run_eval_capped.sh
  rc=$?; echo "[evalsuite] policy=$name evaluator-rc=$rc $(date -Is)" | tee -a "$LOG"
  [ "$rc" -ne 0 ] && { echo "[evalsuite] $name eval rc=$rc; stopping" | tee -a "$LOG"; exit "$rc"; }
  # adapter integrity (mirror the rebaseline guards)
  if [ -n "$apath" ]; then
    if grep -Eqi "missing adapter keys|unexpected keys.*adapter" "$pout/eval.log"; then
      echo "[evalsuite] $name adapter-load warning; refusing" | tee -a "$LOG"; exit 6; fi
    grep -Fq "adapter-path=$apath" "$pout/eval.log" || { echo "[evalsuite] $name adapter path not recorded" | tee -a "$LOG"; exit 6; }
  fi
  PYTHONPATH=. .venv/bin/python scripts/summarize_rollouts.py "$pout/rollouts" \
    --archive data/risk_archive.json --report "$pout/rollout_review.json" >>"$LOG" 2>&1 \
    || { echo "[evalsuite] $name summary failed" | tee -a "$LOG"; exit 5; }
  assert_episodes "$pout/rollout_review.json" >>"$LOG" 2>&1 \
    || { echo "[evalsuite] $name episode count != $EXPECTED" | tee -a "$LOG"; exit 5; }
  echo "[evalsuite] policy=$name complete $(date -Is)" | tee -a "$LOG"
done

# Compare each adapter to base (if base present) and write report-only safety metrics.
base_report="$OUT_ROOT/base/rollout_review.json"
safety_args=()
[ -f "$base_report" ] && safety_args+=(--policy base "$base_report")
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; [ "$name" = "base" ] && continue
  rep="$OUT_ROOT/$name/rollout_review.json"
  safety_args+=(--policy "$name" "$rep")
  if [ -f "$base_report" ]; then
    PYTHONPATH=. .venv/bin/python scripts/compare_rollout_reports.py \
      "$base_report" "$rep" --report "$OUT_ROOT/base_vs_${name}.json" >>"$LOG" 2>&1 \
      || echo "[evalsuite] compare base vs $name failed" | tee -a "$LOG"
  fi
done
PYTHONPATH=. .venv/bin/python scripts/safety_metrics_report.py "${safety_args[@]}" \
  --tasks "$TASK_FILE" \
  --report "$OUT_ROOT/safety_metrics.json" >>"$LOG" 2>&1 \
  || echo "[evalsuite] safety metrics failed" | tee -a "$LOG"
echo "[evalsuite] all policies evaluated; comparisons + safety metrics written $(date -Is)" | tee -a "$LOG"
