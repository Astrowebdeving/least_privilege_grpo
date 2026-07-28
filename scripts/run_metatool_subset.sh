#!/usr/bin/env bash
# Base-plus-three-seed capability diagnostic on a deterministic 1,000-example
# subset of the official MetaTool prepared prompts (200 per task/subtask).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${METATOOL_OUT_ROOT:-logs/artifacts/metatool_subset_$(date +%F_%H%M%S)}"
PORT="${METATOOL_VLLM_PORT:-8010}"
SAMPLE="${METATOOL_SAMPLE_PER_TASK:-200}"
CAP="${METATOOL_POLICY_CAP_SEC:-10800}"
LOG="$OUT_ROOT/suite.log"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

export TOOLPRIV_VLLM_PORT="$PORT"
export TOOLPRIV_VLLM_GPU="${METATOOL_VLLM_GPU:-0}"
export TOOLPRIV_VLLM_LOG="$OUT_ROOT/vllm.log"
export TOOLPRIV_VLLM_PIDFILE="$OUT_ROOT/vllm.pid"
export TOOLPRIV_VLLM_UNIT="lp-metatool-vllm"

cleanup() {
  echo "[metatool] cleanup begin $(date -Is)"
  scripts/stop_toolprivbench_vllm.sh || true
  echo "[metatool] cleanup done $(date -Is)"
}
trap cleanup EXIT INT TERM

cat > "$OUT_ROOT/RUN_CONFIG.txt" <<EOF
launched_at=$(date -Is)
metatool_revision=$(git -C external/metatool rev-parse HEAD)
models=base,seed-0,seed-1,seed-2
sample_per_task=$SAMPLE
sample_seed=0
max_new_tokens=128
temperature=0
thinking_disabled=true
policy_cap_sec=$CAP
gpu=$TOOLPRIV_VLLM_GPU
port=$PORT
EOF

scripts/start_toolprivbench_vllm.sh
code=""
for _ in $(seq 1 180); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" = "200" ] || exit 4

declare -a NAMES=(base seed0 seed1 seed2)
declare -a MODELS=(base seed-0 seed-1 seed-2)
pids=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  echo "[metatool] launch arm=$name model=${MODELS[$i]} $(date -Is)"
  timeout --signal=TERM --kill-after=120 "$CAP" \
    .venv/bin/python scripts/evaluate_metatool_subset.py \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model "${MODELS[$i]}" \
      --sample-per-task "$SAMPLE" \
      --output "$OUT_ROOT/$name" \
      >"$OUT_ROOT/$name.log" 2>&1 &
  pids+=("$!")
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[metatool] complete arm=${NAMES[$i]} $(date -Is)"
  else
    arm_rc=$?
    echo "[metatool] FAILED arm=${NAMES[$i]} rc=$arm_rc" >&2
    rc=1
  fi
done
[ "$rc" -eq 0 ] || exit "$rc"

expected=$((5 * SAMPLE))
for name in "${NAMES[@]}"; do
  .venv/bin/python -c \
    "import json,sys; d=json.load(open('$OUT_ROOT/$name/summary.json')); sys.exit(0 if d['total']==$expected else 1)" \
    || { echo "[metatool] result-count mismatch arm=$name expected=$expected" >&2; exit 5; }
done

.venv/bin/python scripts/summarize_metatool_subset.py "$OUT_ROOT"
echo "[metatool] COMPLETE out=$OUT_ROOT $(date -Is)"
