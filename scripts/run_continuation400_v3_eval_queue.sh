#!/usr/bin/env bash
# Post-training evaluation queue for the continuation400-v3 fixed-endpoint run.
#
# Waits (fail-closed) for training to reach its registered step-400 endpoint,
# then evaluates PARENT (seed-1/checkpoint-1500) against CANDIDATE (step 400):
#   1. lockbox50   data/continuation_confirmatory_v2.json   (untouched endpoint lockbox)
#   2. frozen500   scripts/run_frozen500_confirmatory.sh    (retention panel)
#   3. external    MetaTool / FORTIS / ToolPrivBench        (REPORT ONLY)
#
# Externals are report-only by standing policy and are NEVER used to select a
# checkpoint. Structurally there is nothing to select: step 400 is the
# registered fixed endpoint, so this queue only measures parent vs step 400.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"
lp_pin_path

RUN_ROOT="${CONT400_RUN_ROOT:-logs/artifacts/continuation400_v3_seed1_run}"
OUT_ROOT="${CONT400_EVAL_OUT_ROOT:-logs/artifacts/continuation400_v3_evals}"
PARENT="${CONT400_PARENT_ADAPTER:-logs/artifacts/bigrun_v2_2026-07-01_003238/seed-1/checkpoint-1500}"
CANDIDATE="$RUN_ROOT/train/checkpoint-400"
LOCKBOX="data/continuation_confirmatory_v2.json"
V2_RISK_ARCHIVE="data/risk_archive_v2.json"
EXPECTED_LOCKBOX_SHA="cb6177d09250e791b8d2733f917a994d11f4e5b208a91e73e855a27b39ca88d9"

WAIT_POLL="${CONT400_EVAL_POLL:-120}"
WAIT_MAX="${CONT400_EVAL_WAIT_MAX:-345600}"
GENS="${CONT400_EVAL_GENS:-8}"
ESEED="${CONT400_EVAL_SEED:-1}"
SMALL_CAP="${CONT400_SMALL_CAP:-21600s}"
RESUME="${CONT400_EVAL_RESUME:-0}"
DRY_RUN="${CONT400_EVAL_DRY_RUN:-0}"
DO_FROZEN500="${CONT400_RUN_FROZEN500:-1}"
DO_EXTERNAL="${CONT400_RUN_EXTERNALS:-1}"

export LP_RL_SANDBOX=docker
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
export TMPDIR="${TMPDIR:-/dev/shm}"
PY=".venv/bin/python"

fail() { echo "[cont400-evalq] ERROR: $*" >&2; exit 2; }
say()  { echo "[cont400-evalq] $*"; }

[ -x "$PY" ] || fail "missing $PY"
[ -s "$PARENT/adapter_model.safetensors" ] || fail "missing parent adapter: $PARENT"
[ "$(sha256sum "$LOCKBOX" | cut -d' ' -f1)" = "$EXPECTED_LOCKBOX_SHA" ] \
  || fail "confirmatory lockbox changed since launch"
# ------------------------------------------------------------ fail-closed wait
say "waiting for the registered step-400 endpoint (poll ${WAIT_POLL}s)"
waited=0
while [ ! -f "$RUN_ROOT/COMPLETION_GUARD.json" ] || [ ! -s "$CANDIDATE/adapter_model.safetensors" ]; do
  if ! pgrep -f 'run_continuation400_v3(_train_only)?\.sh|train/train_trl_drgrpo.py' >/dev/null; then
    if [ -f "$RUN_ROOT/reward_variance_by_source.json" ]; then
      st="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status","?"))' \
            "$RUN_ROOT/reward_variance_by_source.json" 2>/dev/null || echo '?')"
      [ "$st" = pass ] || fail "variance gate did not pass (status=$st): the run refused to train. Nothing to evaluate; do NOT loosen thresholds."
    fi
    fail "no trainer process and no COMPLETION_GUARD.json: died before step 400. Diagnose $RUN_ROOT/continuation.log first."
  fi
  [ "$waited" -lt "$WAIT_MAX" ] || fail "waited ${waited}s without completion"
  sleep "$WAIT_POLL"; waited=$((waited + WAIT_POLL))
done
say "training complete; step-400 adapter present"
$PY -c 'import json,sys; g=json.load(open(sys.argv[1])); assert g.get("status")=="pass", g; assert g.get("final_checkpoint")=="train/checkpoint-400", g' \
  "$RUN_ROOT/COMPLETION_GUARD.json" || fail "COMPLETION_GUARD.json is not a passing step-400 guard"

if pgrep -f '[r]un_eval_capped.sh|[r]un_frozen500_confirmatory' >/dev/null; then
  fail "a conflicting evaluator is active"
fi

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/queue.log"
exec > >(tee -a "$LOG") 2>&1
lp_install_cleanup_trap "$LOG" cont400-evalq with_gpu

# --------------------------------------------- preflight: risk-metric liveness
# safe_success only means anything if required_risk is nonzero after TaskSpec
# loading. Assert the observable: loader.py never consults a risk archive
# (schema.py:217 takes the inline field), so archive presence proves nothing.
$PY - "$LOCKBOX" <<'PY'
import sys, pathlib
from least_privilege_env.tasks.loader import load_tasks
DIMS = ("write", "exec", "external", "secret", "scope", "persistent")
bad = []
for cat in sys.argv[1:]:
    specs = load_tasks(pathlib.Path(cat))
    nz = sum(1 for s in specs
             if any(float(getattr(s.required_risk, d, 0.0)) > 0 for d in DIMS))
    print(f"  risk-liveness {cat}: {nz}/{len(specs)} nonzero required_risk")
    if nz != len(specs):
        bad.append(cat)
if bad:
    raise SystemExit(f"required_risk not live for {bad}; safe_success would be meaningless")
PY
[ "$?" = 0 ] || fail "risk-liveness preflight failed"

if [ "$DRY_RUN" = 1 ]; then
  say "DRY RUN: guard, catalog hashes, adapters, risk-liveness verified; nothing started"
  exit 0
fi

"$PY" scripts/check_rootless_docker.py || fail "rootless Docker preflight failed"
[ "$(docker info --format '{{.DockerRootDir}}')" = /mnt/nvme-docker/lp-docker ] \
  || fail "Docker root is not the NVMe root"
scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
scripts/stop_toolprivbench_vllm.sh >/dev/null 2>&1 || true
scripts/prune_lp_containers.sh >/dev/null 2>&1 || true

cat >"$OUT_ROOT/RUN_CONFIG.txt" <<EOF
queued_at=$(date -Is)
purpose=post-training confirmatory + retention + external panel for continuation400-v3
parent=$PARENT
candidate=$CANDIDATE
candidate_sha256=$(sha256sum "$CANDIDATE/adapter_model.safetensors" | cut -d' ' -f1)
lockbox=$LOCKBOX sha256=$EXPECTED_LOCKBOX_SHA
stage_order=lockbox50,frozen500,external
endpoint_policy=step400 fixed; no checkpoint selection is performed
external_results_excluded_from_selection=true (structurally: nothing is selected)
EOF

ADAPTERS="parent=$PARENT,cont400=$CANDIDATE"

# ---- stages 1-2: the two 50-task panels ------------------------------------
run_panel() {  # $1=label $2=catalog
  local label="$1" catalog="$2" ids
  if [ "$RESUME" = 1 ] && [ -f "$OUT_ROOT/$label/VALIDATION_GUARD.json" ]; then
    say "stage=$label resume-skip"; return 0
  fi
  ids="$("$PY" -c 'import json,sys; print(",".join(x["task_id"] for x in json.load(open(sys.argv[1]))))' "$catalog")"
  [ "$(lp_count_csv "$ids")" -eq 50 ] || fail "$label id count is not 50"
  say "stage=$label start $(date -Is)"
  LP_RL_TASK_FILE="$catalog" LP_RL_RISK_ARCHIVE="$V2_RISK_ARCHIVE" \
  EVALSUITE_OUT_ROOT="$OUT_ROOT/$label" EVALSUITE_INCLUDE_BASE=0 \
  EVALSUITE_ADAPTERS="$ADAPTERS" EVALSUITE_TASK_IDS="$ids" \
  EVALSUITE_GENERATIONS="$GENS" EVALSUITE_SEED="$ESEED" EVALSUITE_CAP="$SMALL_CAP" \
    bash scripts/run_eval_suite.sh
  # validate_and_select_continuation.py, NOT validate_frozen500_run.py: the
  # latter takes --tier-manifest/--ordinary-rollouts/--safety-rollouts and
  # argparse-fails on these flags, killing the queue under `set -e` AFTER the
  # eval has already succeeded. Used here purely as a completion guard --
  # step 400 is fixed, so nothing is being selected.
  # --policies is comma-parsed (_csv at validate_and_select_continuation.py:211),
  # NOT nargs; passing two bare words would argparse-fail. No --candidates is
  # passed, so this validates only and selects nothing.
  LP_RL_TASK_FILE="$catalog" "$PY" scripts/validate_and_select_continuation.py \
    --eval-root "$OUT_ROOT/$label" --tasks "$catalog" --policies "parent,cont400" \
    --generations "$GENS" --report "$OUT_ROOT/$label/VALIDATION_GUARD.json"
  say "stage=$label complete $(date -Is)"
}
run_panel lockbox50 "$LOCKBOX"

# ---- stage 3: frozen-500 retention panel -----------------------------------
if [ "$DO_FROZEN500" = 1 ]; then
  if [ "$RESUME" = 1 ] && [ -f "$OUT_ROOT/frozen500/COMPLETION_GUARD.json" ]; then
    say "stage=frozen500 resume-skip"
  else
    say "stage=frozen500 start $(date -Is)"
    scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
    FROZEN500_OUT_ROOT="$OUT_ROOT/frozen500" FROZEN500_ADAPTERS="$ADAPTERS" \
    FROZEN500_SEED="$ESEED" \
      bash scripts/run_frozen500_confirmatory.sh
    say "stage=frozen500 complete $(date -Is)"
  fi
fi

# ---- stage 4: external panel (REPORT ONLY) ---------------------------------
if [ "$DO_EXTERNAL" = 1 ]; then
  EXT="$OUT_ROOT/external"
  if [ "$RESUME" = 1 ] && [ -f "$EXT/COMPLETION_GUARD.json" ]; then
    say "stage=external resume-skip"
  else
    say "stage=external start order=metatool,fortis,toolpriv $(date -Is)"
    scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
    mkdir -p "$EXT"
    export TOOLPRIV_VLLM_PORT="${CONT400_EXTERNAL_PORT:-8010}"
    export TOOLPRIV_VLLM_GPU="${CONT400_EXTERNAL_GPU:-0}"
    export TOOLPRIV_VLLM_LOG="$EXT/vllm.log"
    export TOOLPRIV_VLLM_PIDFILE="$EXT/vllm.pid"
    export TOOLPRIV_VLLM_UNIT=lp-cont400-external-vllm
    export TOOLPRIV_LORA_SPECS="$ADAPTERS"
    export TOOLPRIV_MAX_LORAS=2
    bash scripts/start_toolprivbench_vllm.sh
    code=""
    for _ in $(seq 1 180); do
      code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${TOOLPRIV_VLLM_PORT}/health" || true)"
      [ "$code" = 200 ] && break
      sleep 5
    done
    [ "$code" = 200 ] || fail "external vLLM readiness failed"

    for arm in parent cont400; do
      timeout --signal=TERM --kill-after=120 "${CONT400_METATOOL_CAP:-10800}" \
        "$PY" scripts/evaluate_metatool_subset.py \
        --base-url "http://127.0.0.1:${TOOLPRIV_VLLM_PORT}/v1" \
        --model "$arm" --sample-per-task 200 --output "$EXT/metatool-$arm" \
        >"$EXT/metatool-$arm.log" 2>&1 || say "WARN: metatool/$arm nonzero (report-only)"
      timeout --signal=TERM --kill-after=120 "${CONT400_FORTIS_CAP:-14400}" \
        "$PY" scripts/evaluate_fortis_api.py \
        --base-url "http://127.0.0.1:${TOOLPRIV_VLLM_PORT}/v1" \
        --model "$arm" --output "$EXT/fortis-$arm" \
        >"$EXT/fortis-$arm.log" 2>&1 || say "WARN: fortis/$arm nonzero (report-only)"
      # _api, not _local: the local variant takes --model-label/--adapter-path and
      # has no --base-url, so it cannot use the shared LoRA-serving vLLM above.
      timeout --signal=TERM --kill-after=120 "${CONT400_TOOLPRIV_CAP:-14400}" \
        "$PY" scripts/evaluate_toolprivbench_api.py \
        --base-url "http://127.0.0.1:${TOOLPRIV_VLLM_PORT}/v1" \
        --model "$arm" --output "$EXT/toolpriv-$arm" \
        >"$EXT/toolpriv-$arm.log" 2>&1 || say "WARN: toolpriv/$arm nonzero (report-only)"
    done
    scripts/stop_toolprivbench_vllm.sh >/dev/null 2>&1 || true
    printf '{"status":"complete","report_only":true,"used_for_selection":false}\n' \
      > "$EXT/COMPLETION_GUARD.json"
    say "stage=external complete $(date -Is)"
  fi
fi

# ---- postflight: degeneracy check on the safety columns --------------------
"$PY" - "$OUT_ROOT" <<'PY' || true
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
hits = 0
for m in sorted(root.rglob("safety_metrics*.json")):
    try: d = json.loads(m.read_text())
    except Exception: continue
    rows = d if isinstance(d, list) else (d.get("policies") or d.get("rows") or [])
    if not isinstance(rows, list): continue
    for r in rows:
        if not isinstance(r, dict): continue
        s, ss = r.get("success"), r.get("safe_success")
        if s is None or ss is None: continue
        hits += 1
        note = ""
        if s and ss == 0:
            note = "  <-- DEGENERATE: every success flagged over-privileged (cf. unexplained pool218 0/400)"
        elif s and ss == s:
            note = "  <-- DEGENERATE: nothing flagged; verify the risk envelope is live"
        print(f"  {m.parent.name}/{r.get('name','?')}: success={s} safe_success={ss}{note}")
if not hits:
    print("  (no safety_metrics rows found to check)")
PY

say "COMPLETE $(date -Is) -> $OUT_ROOT"
say "REMINDER: externals are report-only. Step 400 is the registered endpoint; no checkpoint selection is authorized."
