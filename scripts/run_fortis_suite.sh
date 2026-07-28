#!/usr/bin/env bash
# Run full FORTIS Task 1/2 for base and all final v2 seeds through one vLLM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${FORTIS_OUT_ROOT:-logs/artifacts/fortis_$(date +%F_%H%M%S)}"
PORT="${FORTIS_VLLM_PORT:-8010}"
LIMIT="${FORTIS_LIMIT_PER_DOMAIN:-}"
CAP="${FORTIS_POLICY_CAP_SEC:-21600}"
LOG="$OUT_ROOT/suite.log"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

export TOOLPRIV_VLLM_PORT="$PORT"
export TOOLPRIV_VLLM_GPU="${FORTIS_VLLM_GPU:-0}"
export TOOLPRIV_VLLM_LOG="$OUT_ROOT/vllm.log"
export TOOLPRIV_VLLM_PIDFILE="$OUT_ROOT/vllm.pid"
export TOOLPRIV_VLLM_UNIT="lp-fortis-vllm"

cleanup() {
  echo "[fortis-suite] cleanup begin $(date -Is)"
  scripts/stop_toolprivbench_vllm.sh || true
  echo "[fortis-suite] cleanup done $(date -Is)"
}
trap cleanup EXIT INT TERM

cat > "$OUT_ROOT/RUN_CONFIG.txt" <<EOF
launched_at=$(date -Is)
fortis_revision=$(git -C external/fortis rev-parse HEAD)
models=base,seed-0,seed-1,seed-2
tasks=task1,task2
domains=email,ecommerce,filesystem
task1_max_tokens=128
task2_max_tokens=1024
temperature=0
thinking_disabled=true
limit_per_domain=${LIMIT:-none}
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
[ "$code" = "200" ] || {
  echo "[fortis-suite] vLLM readiness failed" >&2
  tail -n 160 "$TOOLPRIV_VLLM_LOG" || true
  exit 4
}

declare -a NAMES=(base_default seed0_default seed1_default seed2_default)
declare -a MODELS=(base seed-0 seed-1 seed-2)
limit_args=()
[ -n "$LIMIT" ] && limit_args=(--limit-per-domain "$LIMIT")

pids=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  echo "[fortis-suite] launch arm=$name model=${MODELS[$i]} $(date -Is)"
  timeout --signal=TERM --kill-after=120 "$CAP" \
    .venv/bin/python scripts/evaluate_fortis_api.py \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model "${MODELS[$i]}" \
      --output "$OUT_ROOT/$name" \
      "${limit_args[@]}" \
      >"$OUT_ROOT/$name.log" 2>&1 &
  pids+=("$!")
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[fortis-suite] complete arm=${NAMES[$i]} $(date -Is)"
  else
    arm_rc=$?
    echo "[fortis-suite] FAILED arm=${NAMES[$i]} rc=$arm_rc" >&2
    rc=1
  fi
done
[ "$rc" -eq 0 ] || exit "$rc"

if [ -n "$LIMIT" ]; then
  expected=$((6 * LIMIT))
else
  expected=2143
fi
for name in "${NAMES[@]}"; do
  .venv/bin/python -c \
    "import json,sys; d=json.load(open('$OUT_ROOT/$name/summary.json')); sys.exit(0 if d['total']==$expected else 1)" \
    || { echo "[fortis-suite] result-count mismatch arm=$name expected=$expected" >&2; exit 5; }
done

.venv/bin/python scripts/summarize_fortis_suite.py "$OUT_ROOT"
echo "[fortis-suite] COMPLETE out=$OUT_ROOT $(date -Is)"
