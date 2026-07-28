#!/usr/bin/env bash
# Run the registered ToolPrivBench comparison:
#   base/default, base/prompt-only, seed-0/default, seed-1/default, seed-2/default
# against one shared local vLLM server. Each arm is resumable per scenario.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${TOOLPRIV_OUT_ROOT:-logs/artifacts/toolprivbench_$(date +%F_%H%M%S)}"
PORT="${TOOLPRIV_VLLM_PORT:-8010}"
MAX_NEW="${TOOLPRIV_MAX_NEW_TOKENS:-2048}"
LIMIT="${TOOLPRIV_LIMIT:-}"
TOOLPRIV_ROOT="${TOOLPRIV_ROOT:-external/toolprivbench}"
DEFAULT_PROMPT="$TOOLPRIV_ROOT/prompts/system_prompts/agent_default.txt"
PRIV_PROMPT="$TOOLPRIV_ROOT/prompts/system_prompts/agent_privilege_aware.txt"
LOG="$OUT_ROOT/suite.log"

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

cleanup() {
  echo "[toolpriv-suite] cleanup begin $(date -Is)"
  scripts/stop_toolprivbench_vllm.sh || true
  echo "[toolpriv-suite] cleanup done $(date -Is)"
}
trap cleanup EXIT INT TERM

scripts/start_toolprivbench_vllm.sh
for _ in $(seq 1 180); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
  [ "$code" = "200" ] && break
  sleep 5
done
[ "${code:-}" = "200" ] || {
  echo "[toolpriv-suite] vLLM readiness failed" >&2
  tail -n 120 logs/artifacts/toolprivbench_vllm.log || true
  exit 4
}

declare -a NAMES=(
  base_default
  base_prompt_only
  seed0_default
  seed1_default
  seed2_default
)
declare -a MODELS=(base base seed-0 seed-1 seed-2)
declare -a PROMPTS=(
  "$DEFAULT_PROMPT"
  "$PRIV_PROMPT"
  "$DEFAULT_PROMPT"
  "$DEFAULT_PROMPT"
  "$DEFAULT_PROMPT"
)
declare -a PROMPT_LABELS=(
  official_default
  official_privilege_aware
  official_default
  official_default
  official_default
)

limit_args=()
[ -n "$LIMIT" ] && limit_args=(--limit "$LIMIT")

pids=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  echo "[toolpriv-suite] launch arm=$name model=${MODELS[$i]} $(date -Is)"
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/evaluate_toolprivbench_api.py \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${MODELS[$i]}" \
    --system-prompt "${PROMPTS[$i]}" \
    --prompt-label "${PROMPT_LABELS[$i]}" \
    --max-new-tokens "$MAX_NEW" \
    --output "$OUT_ROOT/$name" \
    "${limit_args[@]}" \
    >"$OUT_ROOT/$name.log" 2>&1 &
  pids+=("$!")
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[toolpriv-suite] complete arm=${NAMES[$i]} $(date -Is)"
  else
    arm_rc=$?
    echo "[toolpriv-suite] FAILED arm=${NAMES[$i]} rc=$arm_rc" >&2
    rc=1
  fi
done
[ "$rc" -eq 0 ] || exit "$rc"

expected="${LIMIT:-544}"
for name in "${NAMES[@]}"; do
  .venv/bin/python -c \
    "import json,sys; p='$OUT_ROOT/$name/summary.json'; d=json.load(open(p)); sys.exit(0 if d['total']==$expected else 1)" \
    || {
      echo "[toolpriv-suite] episode-count mismatch arm=$name expected=$expected" >&2
      exit 5
    }
done

.venv/bin/python scripts/summarize_toolprivbench_suite.py "$OUT_ROOT"
echo "[toolpriv-suite] COMPLETE out=$OUT_ROOT $(date -Is)"
