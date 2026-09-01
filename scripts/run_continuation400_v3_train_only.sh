#!/usr/bin/env bash
# User-authorized direct optimizer launch for continuation400-v3.
#
# This deliberately does not rerun the 400 x 8 rollout precheck. The complete
# parent-policy precheck already exists for the unchanged 370/400 tasks; the 30
# repaired EAS rows changed prompt disclosure only and retain the exhaustively
# gated verifier/oracle/risk contracts. The interrupted fresh precheck remains
# under OUT_ROOT/precheck for provenance and is never treated as passing.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"
lp_pin_path

OUT_ROOT="${CONTINUATION_OUT_ROOT:-logs/artifacts/continuation400_v3_seed1_run}"
TASK_FILE="data/continuation400_v3.json"
ADAPTER_PATH="${CONTINUATION_ADAPTER_PATH:-logs/artifacts/bigrun_v2_2026-07-01_003238/seed-1/checkpoint-1500}"
EXPECTED_TASK_SHA="f28a3653e122d37b98197345097f8124c53e1196d4fa9344eb4ea41db6a52eaa"
PRIOR_FULL_PRECHECK="logs/artifacts/continuation400_v2_seed1_run/reward_variance_precheck.json"
TRAIN_CAP="${CONTINUATION_TRAIN_CAP:-259200s}"
SEED="${CONTINUATION_SEED:-1}"
MONITOR_CLIPPED_MAX="${CONTINUATION_MONITOR_CLIPPED_MAX:-1.01}"
export VLLM_GPU_UTIL="${CONTINUATION_VLLM_GPU_UTIL:-0.85}"
export LP_RL_TASK_FILE="$TASK_FILE"
export LP_RL_RISK_ARCHIVE="data/risk_archive_v2.json"
export LP_RL_SANDBOX=docker
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
export TMPDIR="${TMPDIR:-/dev/shm}"

PY=.venv/bin/python
LOG="$OUT_ROOT/train_only.log"
fail() { echo "[continuation400-v3-train-only] ERROR: $*" >&2; exit 2; }

[ -x "$PY" ] || fail "missing $PY"
[ -x .venv-train/bin/python ] || fail "missing training interpreter"
[ -s "$TASK_FILE" ] || fail "missing $TASK_FILE"
[ -s "$ADAPTER_PATH/adapter_model.safetensors" ] || fail "missing starting adapter"
[ -s "$PRIOR_FULL_PRECHECK" ] || fail "missing prior complete parent precheck"
[ "$(sha256sum "$TASK_FILE" | cut -d' ' -f1)" = "$EXPECTED_TASK_SHA" ] || fail "task catalog hash drift"
[ -d "$OUT_ROOT/precheck" ] || fail "interrupted precheck provenance is missing"
[ ! -e "$OUT_ROOT/train" ] || fail "refusing to overwrite existing training output"

TASK_IDS="$($PY -c 'import json,sys; print(",".join(x["task_id"] for x in json.load(open(sys.argv[1]))))' "$TASK_FILE")"
[ "$(lp_count_csv "$TASK_IDS")" -eq 400 ] || fail "catalog is not exactly 400 rows"

if pgrep -f 'train/train_trl_drgrpo.py' >/dev/null; then
  fail "an internal trainer/evaluator is already active"
fi

mkdir -p "$OUT_ROOT"
lp_install_cleanup_trap "$LOG" continuation400-v3-train-only with_gpu
scripts/stop_train_vllm.sh >>"$LOG" 2>&1 || true
scripts/prune_lp_containers.sh >>"$LOG" 2>&1 || true
"$PY" scripts/check_rootless_docker.py >>"$LOG" 2>&1 || fail "rootless Docker preflight failed"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')" || fail "docker info failed"
[ "$DOCKER_ROOT" = /mnt/nvme-docker/lp-docker ] || fail "Docker root is $DOCKER_ROOT, expected NVMe"
[ "$TMPDIR" = /dev/shm ] && [ -w "$TMPDIR" ] || fail "RAM scratch is not /dev/shm or not writable"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -ge 2 ] || fail "two GPUs are required"
IFS=',' read -r _ GPU0_FREE _ <<<"${GPU_ROWS[0]}"
GPU0_FREE="${GPU0_FREE//[[:space:]]/}"
[ "$GPU0_FREE" -ge "${CONTINUATION_MIN_GPU0_FREE_MIB:-28000}" ] || fail "GPU0 has only ${GPU0_FREE} MiB free"

"$PY" - "$OUT_ROOT/TRAIN_ONLY_OVERRIDE.json" "$PRIOR_FULL_PRECHECK" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
payload = {
    "status": "user_authorized",
    "decision": "skip_redundant_3200_episode_precheck_and_train_fixed_400_step_endpoint",
    "date": "2026-08-01",
    "prior_complete_parent_precheck": sys.argv[2],
    "fresh_partial_precheck_preserved": True,
    "basis": [
        "370 of 400 tasks are unchanged from the complete parent-policy precheck",
        "30 EAS repairs change prompt disclosure, not verifier/oracle/risk contracts",
        "v3 catalog, oracle separation, adversarial assertions, and disjointness gates passed before launch",
    ],
    "not_a_result": "The interrupted v3 precheck is preserved but is not represented as passing.",
}
out.write_text(json.dumps(payload, indent=2) + "\n")
PY

{
  echo "[continuation400-v3-train-only] direct training authorized $(date -Is)"
  echo "[continuation400-v3-train-only] task_sha=$EXPECTED_TASK_SHA tasks=400 seed=$SEED steps=400 shuffle=false"
  echo "[continuation400-v3-train-only] adapter=$ADAPTER_PATH docker_root=$DOCKER_ROOT tmpdir=$TMPDIR"
} | tee -a "$LOG" "$OUT_ROOT/continuation.log"

lp_vllm_restart 180 "$LOG" || fail "training vLLM start/readiness failed"
TRAIN_OUT="$OUT_ROOT/train" TRAIN_SEED="$SEED" TRAIN_MAX_STEPS=400 \
TRAIN_SAVE_STEPS=50 TRAIN_CAP="$TRAIN_CAP" TRAIN_TASK_IDS="$TASK_IDS" \
TRAIN_ADAPTER_PATH="$ADAPTER_PATH" TRAIN_MONITOR_CLIPPED_MAX="$MONITOR_CLIPPED_MAX" \
TRAIN_TEMP_THRESHOLD_C=83 TRAIN_TEMP_MAX_WAIT_SEC=1800 TRAIN_TEMP_POLL_SEC=30 \
TRAIN_SHUFFLE_DATASET=0 \
  bash scripts/run_train_capped.sh

[ -s "$OUT_ROOT/train/checkpoint-400/adapter_model.safetensors" ] || fail "training exited without checkpoint-400"
"$PY" - "$OUT_ROOT/COMPLETION_GUARD.json" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "pass",
    "starting_checkpoint": "seed-1/checkpoint-1500",
    "continuation_steps": 400,
    "final_checkpoint": "train/checkpoint-400",
    "fresh_optimizer": True,
    "precheck_status": "waived_by_user_2026-08-01",
    "precheck_evidence": "TRAIN_ONLY_OVERRIDE.json",
    "checkpoint_selection": "none; step400 fixed endpoint",
    "confirmatory_lockbox": "data/continuation_confirmatory_v2.json",
    "shuffle_dataset": False,
}, indent=2) + "\n")
PY
echo "[continuation400-v3-train-only] COMPLETE $(date -Is)" | tee -a "$LOG" "$OUT_ROOT/continuation.log"
