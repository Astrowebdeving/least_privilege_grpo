#!/usr/bin/env bash
# Reward-arm ablation FROM THE PRODUCTION POLICY, evaluated where there is headroom.
#
# Differs from the first campaign in the two ways that matter:
#   * starts from seed-1/checkpoint-1500 (the mature model) instead of base, so it
#     tests whether the reward terms matter to a real policy rather than to a
#     400-step-old one;
#   * adds ToolPrivBench, which shows ~40% over-privileged selections on these
#     models, versus ~0% on the internal held-out set. A benchmark that cannot
#     register over-privilege cannot discriminate between privilege rewards.
#
# Stage 1: train each arm 400 steps on continuation400_v3 from ckpt-1500.
# Stage 2: internal held-out eval (control scorer, gating on, arm-neutral).
# Stage 3: ToolPrivBench on every resulting checkpoint. REPORT-ONLY -- it is the
#          ablation's outcome measure, never a checkpoint selector; step 400 is
#          fixed for every arm so there is nothing to select.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/lp_common.sh
lp_pin_path

OUT_ROOT="${C4A_OUT_ROOT:-logs/artifacts/cont400_arm_campaign}"
CATALOG="${C4A_CATALOG:-data/continuation400_v3.json}"
HELDOUT="${C4A_HELDOUT:-data/continuation_transient_heldout_v4.json}"
PARENT="${C4A_PARENT:-logs/artifacts/bigrun_v2_2026-07-01_003238/seed-1/checkpoint-1500}"
EXPECTED_CATALOG_SHA="${C4A_CATALOG_SHA:-f28a3653e122d37b98197345097f8124c53e1196d4fa9344eb4ea41db6a52eaa}"
ARMS="${C4A_ARMS:-A0 A4 A2 B1}"
SEEDS="${C4A_SEEDS:-1 2}"
STEPS="${C4A_STEPS:-400}"
GENS="${C4A_GENS:-8}"
DRY_RUN="${C4A_DRY_RUN:-0}"
DO_TPB="${C4A_RUN_TPB:-1}"
TPB_PORT="${C4A_TPB_PORT:-8011}"
TPB_BATCH="${C4A_TPB_BATCH:-4}"

PY=".venv/bin/python"
fail() { echo "[c4a] ERROR: $*" >&2; exit 2; }
say()  { echo "[c4a] $*"; }

[ -s "$PARENT/adapter_model.safetensors" ] || fail "missing parent adapter $PARENT"
[ -f "$CATALOG" ] || fail "missing $CATALOG"
[ -f "$HELDOUT" ] || fail "missing $HELDOUT"
a="$(sha256sum "$CATALOG" | cut -d' ' -f1)"
[ "$a" = "$EXPECTED_CATALOG_SHA" ] || fail "catalog hash drift: $a"

# The shared runner defaults to ablation500's 500/50 shape. This campaign trains
# on continuation400_v3, so state its shape explicitly and verify it before
# handing it down -- a size guard written for another catalog is how the first
# launch died.
N_CAT="${C4A_N_TASKS:-400}"
N_HELD="${C4A_N_HELDOUT:-50}"
count_ids() { "$PY" -c 'import json,sys
print(len({r["task_id"] for r in json.load(open(sys.argv[1]))}))' "$1"; }
[ "$(count_ids "$CATALOG")" -eq "$N_CAT" ] \
  || fail "$CATALOG has $(count_ids "$CATALOG") unique task_ids, expected $N_CAT"
[ "$(count_ids "$HELDOUT")" -eq "$N_HELD" ] \
  || fail "$HELDOUT has $(count_ids "$HELDOUT") unique task_ids, expected $N_HELD"

for arm in $ARMS; do
  "$PY" -c 'import sys
from least_privilege_env.reward import scoring as S
sys.exit(0 if sys.argv[1] in S.REWARD_ARMS else 1)' "$arm" \
    || fail "unknown arm '$arm'"
done

say "arms=[$ARMS] seeds=[$SEEDS] steps=$STEPS from=$PARENT"
say "catalog=$CATALOG heldout=$HELDOUT tpb=$DO_TPB -> $OUT_ROOT"

if [ "$DRY_RUN" = 1 ]; then
  say "DRY RUN: inputs, hashes and arms verified; nothing started"
  exit 0
fi

# ---- stage 0: reuse the existing A0-seed1 run instead of repeating it ----------
# continuation400_v3 IS this design's A0/seed-1 cell: same catalog sha f28a3653,
# same starting adapter 41ce33e4, seed 1, 400 steps, shuffle_dataset=false, full
# production reward. Retraining it would burn ~7 GPU-hours to reproduce a result
# we already have. Link it into the campaign layout so ABL_RESUME=1 skips its
# training and only runs the (cheap) matched held-out eval.
V3=logs/artifacts/continuation400_v3_seed1_run/train/checkpoint-400
if [ -s "$V3/adapter_model.safetensors" ] && echo "$ARMS" | grep -qw A0 \
   && echo "$SEEDS" | grep -qw 1; then
  slot="$OUT_ROOT/A0-seed1/train"
  if [ ! -e "$slot/checkpoint-$STEPS" ]; then
    mkdir -p "$slot"
    ln -s "$(cd "$(dirname "$V3")" && pwd)/$(basename "$V3")" "$slot/checkpoint-$STEPS"
    cat > "$OUT_ROOT/A0-seed1/REUSED_FROM.txt" <<EOF
This cell is NOT a fresh training run.
source=logs/artifacts/continuation400_v3_seed1_run/train/checkpoint-400
adapter_sha256=$(sha256sum "$V3/adapter_model.safetensors" | cut -d' ' -f1)
rationale=identical condition (catalog f28a3653, parent 41ce33e4, seed 1, 400
  steps, shuffle_dataset=false, production reward = arm A0); retraining would
  only reproduce it. The held-out eval below IS run fresh, so the evaluation is
  matched to the other arms.
EOF
    say "reusing continuation400_v3 as the A0-seed1 cell (see REUSED_FROM.txt)"
  fi
fi

# ---- stage 1+2: train + internal eval, reusing the validated campaign runner ----
C1_ARMS="$ARMS" ; C1_SEEDS="$SEEDS"
if [ "${C4A_TPB_PLAN_ONLY:-0}" = 1 ]; then
  say "PLAN ONLY: skipping train+eval stage"
else
ABL_OUT_ROOT="$OUT_ROOT" \
ABL_CATALOG="$CATALOG" ABL_HELDOUT="$HELDOUT" \
ABL_EXPECTED_CATALOG_SHA="$EXPECTED_CATALOG_SHA" \
ABL_EXPECTED_HELDOUT_SHA="$(sha256sum "$HELDOUT" | cut -d' ' -f1)" \
ABL_SKIP_BUILDER_CHECK=1 \
ABL_EXPECTED_TASKS="$N_CAT" ABL_EXPECTED_HELDOUT="$N_HELD" \
ABL_BASE_ADAPTER="$PARENT" \
ABL_ARMS="$C1_ARMS" ABL_SEEDS="$C1_SEEDS" ABL_STEPS="$STEPS" ABL_GENS="$GENS" \
ABL_RESUME=1 \
  bash scripts/run_ablation_campaign.sh || fail "training campaign failed"
fi

# ---- stage 3: ToolPrivBench over every checkpoint, plus the untrained parent ----
if [ "$DO_TPB" != 1 ]; then
  say "TPB disabled; done"; exit 0
fi
EXT="$OUT_ROOT/toolprivbench"
mkdir -p "$EXT"
say "ToolPrivBench stage start $(date -Is)"
scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
scripts/stop_toolprivbench_vllm.sh >/dev/null 2>&1 || true

# Collect name=path for every checkpoint that still needs scoring.
#
# SEED-MAJOR on purpose. Re-scoring one adapter under a different LoRA batch
# composition moves its result by up to 8/544 (1.5 pts) even at temperature 0,
# so which policies share a vLLM batch is a real nuisance variable. Arm-major
# ordering would have put all of one arm's seeds in one batch and split another
# across two, confounding arm with batch. Seed-major with TPB_BATCH = number of
# arms puts exactly one checkpoint from each arm in every batch, so any batch
# effect hits all arms equally.
#
# Already-scored policies are left out of SPECS entirely rather than added and
# skipped later: a spec still gets loaded into vLLM as a LoRA, so including one
# would change the batch composition for everything beside it.
SPECS=()
[ -f "$EXT/parent/summary.json" ] || SPECS+=("parent=$PARENT")
for seed in $SEEDS; do
  for arm in $ARMS; do
    ck="$OUT_ROOT/$arm-seed$seed/train/checkpoint-$STEPS"
    [ -s "$ck/adapter_model.safetensors" ] || continue
    [ -f "$EXT/$arm-s$seed/summary.json" ] && continue
    SPECS+=("$arm-s$seed=$ck")
  done
done
if [ "${#SPECS[@]}" -eq 0 ]; then
  say "TPB: every requested policy already scored; nothing to do"
  printf '{"status":"complete","report_only":true,"used_for_selection":false}\n' \
    > "$EXT/COMPLETION_GUARD.json"
  exit 0
fi
say "TPB will score ${#SPECS[@]} policies (batch=$TPB_BATCH, seed-major)"

if [ "${C4A_TPB_PLAN_ONLY:-0}" = 1 ]; then
  say "PLAN ONLY -- batches that would run:"
  i=0
  while [ "$i" -lt "${#SPECS[@]}" ]; do
    b=("${SPECS[@]:$i:$TPB_BATCH}")
    names=""; for s in "${b[@]}"; do names="$names ${s%%=*}"; done
    say "  batch $((i / TPB_BATCH + 1)):$names"
    i=$((i + TPB_BATCH))
  done
  exit 0
fi

# vLLM cannot hold many LoRAs at once; batch them.
i=0
while [ "$i" -lt "${#SPECS[@]}" ]; do
  batch=("${SPECS[@]:$i:$TPB_BATCH}")
  joined="$(IFS=,; echo "${batch[*]}")"
  say "TPB batch: $joined"
  export TOOLPRIV_VLLM_PORT="$TPB_PORT" TOOLPRIV_VLLM_GPU=0
  export TOOLPRIV_VLLM_LOG="$EXT/vllm.log" TOOLPRIV_VLLM_PIDFILE="$EXT/vllm.pid"
  export TOOLPRIV_VLLM_UNIT=lp-c4a-tpb-vllm
  export TOOLPRIV_LORA_SPECS="$joined" TOOLPRIV_MAX_LORAS="${#batch[@]}"
  bash scripts/start_toolprivbench_vllm.sh || fail "TPB vLLM start failed"
  code=""
  for _ in $(seq 1 180); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$TPB_PORT/health" || true)"
    [ "$code" = 200 ] && break
    sleep 5
  done
  [ "$code" = 200 ] || fail "TPB vLLM readiness failed"
  for spec in "${batch[@]}"; do
    name="${spec%%=*}"
    if [ -f "$EXT/$name/summary.json" ]; then say "  $name resume-skip"; continue; fi
    timeout --signal=TERM --kill-after=120 "${C4A_TPB_CAP:-14400}" \
      "$PY" scripts/evaluate_toolprivbench_api.py \
      --base-url "http://127.0.0.1:$TPB_PORT/v1" --model "$name" \
      --output "$EXT/$name" >"$EXT/$name.log" 2>&1 \
      || say "  WARN: TPB failed for $name (report-only, continuing)"
  done
  scripts/stop_toolprivbench_vllm.sh >/dev/null 2>&1 || true
  i=$((i + TPB_BATCH))
done

printf '{"status":"complete","report_only":true,"used_for_selection":false}\n' \
  > "$EXT/COMPLETION_GUARD.json"
say "COMPLETE $(date -Is) -> $OUT_ROOT"
say "REMINDER: ToolPrivBench is report-only. Step $STEPS is the fixed endpoint for every arm."
