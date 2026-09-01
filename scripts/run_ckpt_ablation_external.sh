#!/usr/bin/env bash
# Checkpoint-trajectory ablation on the external benchmarks (descriptive,
# post-hoc training-dynamics analysis — NOT checkpoint selection).
#
# Motivation: 81-88% of campaign steps had reward_std==0 (no gradient), and the
# gradient-bearing steps concentrate in the first ~500 steps (zero-std by
# thirds — seed-0 79/91/92%, seed-1 76/93/96%, seed-2 65/83/94%). If external
# behavior at checkpoint-500 ~= checkpoint-1500, most transfer was learned in
# the first third; it also localizes WHEN seed-0's unseen-MCP format collapse
# and seed-1's ToolPriv Safety-Bypass regression emerged.
#
# Arms: seed-{0,1,2} x checkpoint-{500,1000} on FORTIS + ToolPrivBench +
# MetaTool, one shared vLLM (GPU0) serving all six LoRA adapters. Base and
# checkpoint-1500 arms are NOT rerun: identical served config, official
# revisions, temperature 0 — compare against the preserved 2026-07-16
# artifacts. Integrity: external suites remain excluded from any checkpoint
# selection; the frozen seed-1/checkpoint-1500 primary result is untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${ABLATION_OUT_ROOT:-logs/artifacts/ckpt_ablation_external_$(date +%F_%H%M%S)}"
PORT="${ABLATION_VLLM_PORT:-8010}"
RUN="${ABLATION_ADAPTER_ROOT:-logs/artifacts/bigrun_v2_2026-07-01_003238}"
DRY_RUN="${ABLATION_DRY_RUN:-0}"

# ABLATION_ARMS selects which checkpoints to serve, as "seed:ckpt" pairs.
# Default = the sample-efficiency focus: seed-1 checkpoint-500 + -1000, compared
# to the existing seed-1 checkpoint-1500 external results (*_full_2026-07-16).
# Set e.g. ABLATION_ARMS="0:500,0:1000,1:500,1:1000,2:500,2:1000" for all seeds.
ABLATION_ARMS="${ABLATION_ARMS:-1:500,1:1000}"
declare -a ARMS=() SPECS=()
IFS=',' read -r -a PAIRS <<< "$ABLATION_ARMS"
for pair in "${PAIRS[@]}"; do
  s="${pair%%:*}"; c="${pair##*:}"
  path="$RUN/seed-$s/checkpoint-$c"
  [ -s "$path/adapter_model.safetensors" ] || { echo "[ablation] missing adapter: $path" >&2; exit 2; }
  ARMS+=("seed$s-ckpt$c")
  SPECS+=("seed$s-ckpt$c=$path")
done
[ "${#ARMS[@]}" -ge 1 ] || { echo "[ablation] no arms" >&2; exit 2; }
LORA_SPECS="$(IFS=','; echo "${SPECS[*]}")"
export TOOLPRIV_MAX_LORAS="${#ARMS[@]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "[ablation] DRY RUN ok: ${#SPECS[@]} adapters verified"
  printf '[ablation]   %s\n' "${SPECS[@]}"
  echo "[ablation] out=$OUT_ROOT port=$PORT suites=toolpriv,fortis,metatool"
  exit 0
fi

# refuse to fight the frozen-500 eval or the external suite units
if pgrep -f 'run_frozen500_confirmatory|train_trl_drgrpo' >/dev/null; then
  echo "[ablation] refusing: internal eval/training process active" >&2; exit 3
fi
for unit in lp-fortis-suite.service lp-metatool-suite.service lp-toolprivbench-suite.service lp-rl-vllm.service; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "[ablation] refusing: $unit active" >&2; exit 3
  fi
done

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/ablation.log"
exec > >(tee -a "$LOG") 2>&1

cat > "$OUT_ROOT/RUN_CONFIG.txt" <<EOF
launched_at=$(date -Is)
purpose=checkpoint-trajectory ablation (descriptive; NOT selection)
arms=${ARMS[*]}
adapter_root=$RUN
compare_against=toolprivbench_full_2026-07-16, fortis_full_2026-07-16, metatool_subset_full_2026-07-16 (base + ckpt-1500 arms, identical config)
zero_gradient_context=seed0 87% seed1 88% seed2 81% zero-std steps; by-thirds seed0 79/91/92 seed1 76/93/96 seed2 65/83/94
port=$PORT
temperature=0
EOF

export TOOLPRIV_VLLM_PORT="$PORT"
export TOOLPRIV_VLLM_GPU="${ABLATION_VLLM_GPU:-0}"
export TOOLPRIV_VLLM_LOG="$OUT_ROOT/vllm.log"
export TOOLPRIV_VLLM_PIDFILE="$OUT_ROOT/vllm.pid"
export TOOLPRIV_VLLM_UNIT="lp-ckpt-ablation-vllm"
export TOOLPRIV_LORA_SPECS="$LORA_SPECS"
# TOOLPRIV_MAX_LORAS already exported = ${#ARMS[@]} at arm-selection time

cleanup() {
  echo "[ablation] cleanup begin $(date -Is)"
  bash scripts/stop_toolprivbench_vllm.sh || true
  echo "[ablation] cleanup done $(date -Is)"
}
trap cleanup EXIT INT TERM

bash scripts/start_toolprivbench_vllm.sh
code=""
for _ in $(seq 1 180); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" = "200" ] || { echo "[ablation] vLLM readiness failed"; tail -n 120 "$TOOLPRIV_VLLM_LOG" || true; exit 4; }

TOOLPRIV_ROOT_DIR="external/toolprivbench"
DEFAULT_PROMPT="$TOOLPRIV_ROOT_DIR/prompts/system_prompts/agent_default.txt"

run_suite() {  # $1=suite name
  local suite="$1"
  echo "[ablation] suite=$suite start $(date -Is)"
  local pids=() names=()
  for arm in "${ARMS[@]}"; do
    local out="$OUT_ROOT/$suite/$arm"
    mkdir -p "$(dirname "$out")"
    case "$suite" in
      toolpriv)
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/evaluate_toolprivbench_api.py \
          --base-url "http://127.0.0.1:${PORT}/v1" --model "$arm" \
          --system-prompt "$DEFAULT_PROMPT" --prompt-label official_default \
          --max-new-tokens 2048 --output "$out" >"$out.log" 2>&1 &
        ;;
      fortis)
        timeout --signal=TERM --kill-after=120 21600 \
          .venv/bin/python scripts/evaluate_fortis_api.py \
          --base-url "http://127.0.0.1:${PORT}/v1" --model "$arm" \
          --output "$out" >"$out.log" 2>&1 &
        ;;
      metatool)
        timeout --signal=TERM --kill-after=120 10800 \
          .venv/bin/python scripts/evaluate_metatool_subset.py \
          --base-url "http://127.0.0.1:${PORT}/v1" --model "$arm" \
          --sample-per-task 200 --output "$out" >"$out.log" 2>&1 &
        ;;
    esac
    pids+=("$!"); names+=("$arm")
  done
  local rc=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[ablation] $suite arm=${names[$i]} complete $(date -Is)"
    else
      echo "[ablation] $suite arm=${names[$i]} FAILED rc=$?" >&2; rc=1
    fi
  done
  return "$rc"
}

overall=0
run_suite toolpriv || overall=1
run_suite fortis   || overall=1
run_suite metatool || overall=1

echo "[ablation] count guards"
for arm in "${ARMS[@]}"; do
  .venv/bin/python - "$OUT_ROOT" "$arm" <<'PY'
import json, sys
root, arm = sys.argv[1], sys.argv[2]
expect = {"toolpriv": 544, "fortis": 2143, "metatool": 1000}
bad = []
for suite, n in expect.items():
    try:
        d = json.load(open(f"{root}/{suite}/{arm}/summary.json"))
        if d["total"] != n:
            bad.append(f"{suite}={d['total']}!={n}")
    except Exception as exc:
        bad.append(f"{suite}: {exc}")
if bad:
    print(f"[ablation] COUNT MISMATCH {arm}: {'; '.join(bad)}")
    sys.exit(1)
print(f"[ablation] counts OK {arm}")
PY
  [ $? -eq 0 ] || overall=1
done

[ "$overall" -eq 0 ] && echo "[ablation] COMPLETE out=$OUT_ROOT $(date -Is)" \
                     || echo "[ablation] FINISHED WITH FAILURES out=$OUT_ROOT $(date -Is)"
exit "$overall"
