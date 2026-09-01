#!/usr/bin/env bash
# Fixed-endpoint post-hoc continuation:
# seed-1 checkpoint-1500 -> fresh 400-step optimizer over 200 repaired
# corrective tasks + 200 variance-aware original-task replay rows.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"
lp_pin_path

OUT_ROOT="${CONTINUATION_OUT_ROOT:-logs/artifacts/continuation400_v3_seed1_$(date +%F_%H%M%S)}"
TASK_FILE="data/continuation400_v3.json"
CORRECTIVE_FILE="data/continuation_transient_v4.json"
HELDOUT_FILE="data/continuation_transient_heldout_v4.json"
CONFIRMATORY_FILE="data/continuation_confirmatory_v2.json"
RISK_ARCHIVE="data/risk_archive_v2.json"
VOCAB_FILE="data/external_benchmark_vocab.json"
READINESS_DIR="logs/artifacts/continuation400_v2_readiness_2026-07-31"
ADAPTER_PATH="${CONTINUATION_ADAPTER_PATH:-logs/artifacts/bigrun_v2_2026-07-01_003238/seed-1/checkpoint-1500}"
EXPECTED_TASK_SHA="f28a3653e122d37b98197345097f8124c53e1196d4fa9344eb4ea41db6a52eaa"
EXPECTED_CORRECTIVE_SHA="b99408112e104ff756b4abfea53c61fdde3d2355f2f82587210846d26137822b"
EXPECTED_HELDOUT_SHA="46f6ee559f32435d6450ec40a84ab038d7b83bc7eeb97cf69cf4546599cd3e5a"
EXPECTED_CONFIRMATORY_SHA="cb6177d09250e791b8d2733f917a994d11f4e5b208a91e73e855a27b39ca88d9"
EXPECTED_PARITY_LOCAL_SHA="d5b8976daf2f687c9627e4708ca88c010ba6ba41aea9f8d1fb3931b72f6941b6"
EXPECTED_PARITY_DOCKER_SHA="28ec81a1d611a5cb79013c363037592e5db27e7555946f1d19530e9dd0d6f7a1"
EXPECTED_PARITY_COMPARE_SHA="9971745edd1f2ecca3eb963a326ca6b180ca12563175d51be086b06d9cfc794c"
PRECHECK_CAP="${CONTINUATION_PRECHECK_CAP:-129600s}"
TRAIN_CAP="${CONTINUATION_TRAIN_CAP:-259200s}"
SEED="${CONTINUATION_SEED:-1}"
DRY_RUN="${CONTINUATION_DRY_RUN:-0}"
REUSE_PRECHECK="${CONTINUATION_REUSE_PRECHECK:-0}"
MONITOR_CLIPPED_MAX="${CONTINUATION_MONITOR_CLIPPED_MAX:-1.01}"
export VLLM_GPU_UTIL="${CONTINUATION_VLLM_GPU_UTIL:-0.85}"
export LP_RL_TASK_FILE="$TASK_FILE"
export LP_RL_RISK_ARCHIVE="$RISK_ARCHIVE"
export LP_RL_SANDBOX="docker"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
export TMPDIR="${TMPDIR:-/dev/shm}"

PY=".venv/bin/python"
fail() { echo "[continuation400-v3] ERROR: $*" >&2; exit 2; }

[ -x "$PY" ] || fail "missing $PY"
[ -x ".venv-train/bin/python" ] || fail "missing .venv-train/bin/python"
[ -s "$ADAPTER_PATH/adapter_model.safetensors" ] || fail "missing starting adapter: $ADAPTER_PATH"
for file in "$TASK_FILE" "$CORRECTIVE_FILE" "$HELDOUT_FILE" "$CONFIRMATORY_FILE" "$RISK_ARCHIVE" "$VOCAB_FILE"; do
  [ -f "$file" ] || fail "missing $file"
done
for file in local_report.json docker_report.json parity_comparison.json; do
  [ -f "$READINESS_DIR/$file" ] || fail "missing focused Docker-parity artifact: $file"
done

check_sha() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum "$file" | cut -d' ' -f1)"
  [ "$actual" = "$expected" ] || fail "$file hash drift: $actual"
}
check_sha "$TASK_FILE" "$EXPECTED_TASK_SHA"
check_sha "$CORRECTIVE_FILE" "$EXPECTED_CORRECTIVE_SHA"
check_sha "$HELDOUT_FILE" "$EXPECTED_HELDOUT_SHA"
check_sha "$CONFIRMATORY_FILE" "$EXPECTED_CONFIRMATORY_SHA"
check_sha "$READINESS_DIR/local_report.json" "$EXPECTED_PARITY_LOCAL_SHA"
check_sha "$READINESS_DIR/docker_report.json" "$EXPECTED_PARITY_DOCKER_SHA"
check_sha "$READINESS_DIR/parity_comparison.json" "$EXPECTED_PARITY_COMPARE_SHA"

TASK_IDS="$($PY -c 'import json,sys; print(",".join(row["task_id"] for row in json.load(open(sys.argv[1]))))' "$TASK_FILE")"
[ "$(lp_count_csv "$TASK_IDS")" -eq 400 ] || fail "mixed catalog does not contain exactly 400 tasks"

echo "[continuation400-v3] static gates start $(date -Is)"
"$PY" scripts/build_continuation_v4.py --check
"$PY" scripts/build_continuation_replay_scores.py --check
"$PY" scripts/build_continuation400_v3.py --check
LP_RL_SANDBOX=local "$PY" scripts/check_continuation400_v3.py
for catalog in "$CORRECTIVE_FILE" "$HELDOUT_FILE" "$CONFIRMATORY_FILE"; do
  LP_RL_SANDBOX=local "$PY" scripts/check_transient_family.py --catalog "$catalog"
  "$PY" scripts/check_external_disjointness.py --catalog "$catalog" --vocab "$VOCAB_FILE" --strict
done
"$PY" scripts/check_external_disjointness.py --catalog "$TASK_FILE" --vocab "$VOCAB_FILE" --strict
"$PY" -c 'import json; p=json.load(open("logs/artifacts/continuation400_v2_readiness_2026-07-31/parity_comparison.json")); assert p.get("task_count")==12 and not p.get("violations"), p'
echo "[continuation400-v3] static gates pass $(date -Is)"

if [ "$DRY_RUN" = "1" ]; then
  echo "[continuation400-v3] DRY RUN: immutable inputs, gates, lockbox separation, adapter, and focused Docker parity verified"
  echo "[continuation400-v3] out=$OUT_ROOT adapter=$ADAPTER_PATH seed=$SEED precheck=400x8 train=400 fixed endpoint"
  echo "[continuation400-v3] checkpoints=50,100,...,400 are diagnostic/safety-stop only; no checkpoint selection"
  echo "[continuation400-v3] caps precheck=$PRECHECK_CAP train=$TRAIN_CAP vllm-util=$VLLM_GPU_UTIL; no services or containers touched"
  exit 0
fi

if pgrep -f 'train/train_trl_drgrpo.py' >/dev/null; then
  fail "an internal trainer/evaluator process is already active"
fi
for unit in lp-fortis-suite.service lp-metatool-suite.service lp-toolprivbench-suite.service lp-ckpt-ablation-vllm.service; do
  if systemctl --user is-active --quiet "$unit"; then
    fail "conflicting project unit is active: $unit"
  fi
done

if [ "$REUSE_PRECHECK" = "1" ]; then
  [ -d "$OUT_ROOT/precheck/rollouts" ] || fail "reuse requested but precheck rollouts are missing"
  [ ! -e "$OUT_ROOT/train" ] || fail "refusing to overwrite an existing train directory"
else
  [ ! -e "$OUT_ROOT" ] || fail "refusing to append to existing output: $OUT_ROOT"
  mkdir -p "$OUT_ROOT"
fi
LOG="$OUT_ROOT/continuation.log"
lp_install_cleanup_trap "$LOG" continuation400-v3 with_gpu

scripts/stop_train_vllm.sh >>"$LOG" 2>&1 || true
scripts/prune_lp_containers.sh >>"$LOG" 2>&1 || true
"$PY" scripts/check_rootless_docker.py >>"$LOG" 2>&1 || fail "rootless Docker preflight failed"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')" || fail "docker info failed"
[ "$DOCKER_ROOT" = "/mnt/nvme-docker/lp-docker" ] || fail "Docker root is $DOCKER_ROOT, expected /mnt/nvme-docker/lp-docker"
[ "$TMPDIR" = "/dev/shm" ] || fail "TMPDIR is $TMPDIR, expected /dev/shm"
[ -d "$TMPDIR" ] && [ -w "$TMPDIR" ] || fail "RAM scratch is not writable: $TMPDIR"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -ge 2 ] || fail "two GPUs are required"
IFS=',' read -r _ GPU0_FREE GPU0_TOTAL <<<"${GPU_ROWS[0]}"
IFS=',' read -r _ GPU1_FREE GPU1_TOTAL <<<"${GPU_ROWS[1]}"
GPU0_FREE="${GPU0_FREE//[[:space:]]/}"; GPU0_TOTAL="${GPU0_TOTAL//[[:space:]]/}"
GPU1_FREE="${GPU1_FREE//[[:space:]]/}"; GPU1_TOTAL="${GPU1_TOTAL//[[:space:]]/}"
[ "$GPU0_FREE" -ge "${CONTINUATION_MIN_GPU0_FREE_MIB:-28000}" ] || fail "GPU0 has only ${GPU0_FREE} MiB free"
VLLM_REQUIRED="$(awk -v total="$GPU1_TOTAL" -v util="$VLLM_GPU_UTIL" 'BEGIN { printf "%.0f", total * util }')"
[ "$GPU1_FREE" -ge "$VLLM_REQUIRED" ] || echo "[continuation400-v3] WARN: GPU1 free=${GPU1_FREE} MiB vs nominal vLLM request=${VLLM_REQUIRED} MiB; readiness is the hard gate" | tee -a "$LOG"

{
  echo "launched_at=$(date -Is)"
  echo "purpose=post-hoc fixed-endpoint corrective continuation; primary checkpoint remains seed-1 checkpoint-1500"
  echo "task_file=$TASK_FILE"
  echo "task_file_sha256=$EXPECTED_TASK_SHA"
  echo "corrective_file=$CORRECTIVE_FILE"
  echo "corrective_file_sha256=$EXPECTED_CORRECTIVE_SHA"
  echo "heldout_development=$HELDOUT_FILE"
  echo "confirmatory_lockbox=$CONFIRMATORY_FILE"
  echo "starting_adapter=$ADAPTER_PATH"
  echo "starting_adapter_sha256=$(sha256sum "$ADAPTER_PATH/adapter_model.safetensors" | cut -d' ' -f1)"
  echo "seed=$SEED"
  echo "composition=200 corrective-v4 + 200 unchanged variance-aware original replay"
  echo "precheck_tasks=400 precheck_generations=8 precheck_episodes=3200"
  echo "variance_gate=overall>=0.20; family variance or parent success>=0.90; corrective variance>=0.10; replay variance>=0.05 and success>=0.90"
  echo "training_steps=400 save_steps=50 fresh_optimizer=true trainer_resume=false shuffle_dataset=false"
  echo "curriculum=exact corrective/replay alternation from catalog order"
  echo "endpoint_policy=step400 fixed; intermediate checkpoints diagnostic/safety-stop only"
  echo "precheck_cap=$PRECHECK_CAP train_cap=$TRAIN_CAP"
  echo "vllm_gpu=1 trainer_gpu=0 vllm_gpu_util=$VLLM_GPU_UTIL"
  echo "monitor_clipped_max=$MONITOR_CLIPPED_MAX"
  echo "docker_host=$DOCKER_HOST docker_root=$DOCKER_ROOT tmpdir=$TMPDIR"
  echo "gpu0_free_mib=$GPU0_FREE gpu0_total_mib=$GPU0_TOTAL"
  echo "gpu1_free_mib=$GPU1_FREE gpu1_total_mib=$GPU1_TOTAL"
  echo "evaluation_policy=development heldout is diagnostic; untouched confirmatory lockbox is opened only for final step400"
} >"$OUT_ROOT/RUN_CONFIG.txt"

PRECHECK_OUT="$OUT_ROOT/precheck"
if [ "$REUSE_PRECHECK" = "1" ]; then
  echo "[continuation400-v3] re-auditing completed precheck; no rollout rerun $(date -Is)" | tee -a "$LOG"
else
  echo "[continuation400-v3] precheck start: parent checkpoint, 400x8 episodes $(date -Is)" | tee -a "$LOG"
  lp_vllm_restart 180 "$LOG" || fail "precheck vLLM start/readiness failed"
  EVAL_LABEL="continuation400-v3-precheck" EVAL_OUT="$PRECHECK_OUT" \
  EVAL_ADAPTER_PATH="$ADAPTER_PATH" EVAL_TASK_IDS="$TASK_IDS" \
  EVAL_NUM_GENERATIONS=8 EVAL_SEED="$SEED" EVAL_CAP="$PRECHECK_CAP" \
  EVAL_MONITOR_CLIPPED_MAX="$MONITOR_CLIPPED_MAX" \
    bash scripts/run_eval_capped.sh
fi

"$PY" scripts/analyze_reward_variance.py \
  --rollouts "$PRECHECK_OUT/rollouts" \
  --task-file "$TASK_FILE" \
  --expected-generations 8 \
  --min-overall-variable-fraction 0.20 \
  --min-family-variable-fraction 0.10 \
  --min-family-variable-groups 2 \
  --min-saturated-family-success-rate 0.90 \
  --report "$OUT_ROOT/reward_variance_precheck.json" | tee -a "$LOG"
"$PY" scripts/check_continuation400_precheck.py \
  --variance-report "$OUT_ROOT/reward_variance_precheck.json" \
  --catalog "$TASK_FILE" \
  --corrective "$CORRECTIVE_FILE" \
  --min-corrective-variable-fraction 0.10 \
  --min-replay-variable-fraction 0.05 \
  --min-replay-success-rate 0.90 \
  --report "$OUT_ROOT/reward_variance_by_source.json" | tee -a "$LOG"
echo "[continuation400-v3] precheck PASS; optimizer authorized $(date -Is)" | tee -a "$LOG"

lp_vllm_restart 180 "$LOG" || fail "training vLLM start/readiness failed"
TRAIN_OUT="$OUT_ROOT/train" TRAIN_SEED="$SEED" TRAIN_MAX_STEPS=400 \
TRAIN_SAVE_STEPS=50 TRAIN_CAP="$TRAIN_CAP" TRAIN_TASK_IDS="$TASK_IDS" \
TRAIN_ADAPTER_PATH="$ADAPTER_PATH" TRAIN_MONITOR_CLIPPED_MAX="$MONITOR_CLIPPED_MAX" \
TRAIN_TEMP_THRESHOLD_C=83 TRAIN_TEMP_MAX_WAIT_SEC=1800 TRAIN_TEMP_POLL_SEC=30 \
TRAIN_SHUFFLE_DATASET=0 \
  bash scripts/run_train_capped.sh

[ -s "$OUT_ROOT/train/checkpoint-400/adapter_model.safetensors" ] || fail "training exited without checkpoint-400 adapter"
"$PY" -c 'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); a=json.load(open(root/"reward_variance_precheck.json")); b=json.load(open(root/"reward_variance_by_source.json")); assert a["status"]=="pass" and b["status"]=="pass"; (root/"COMPLETION_GUARD.json").write_text(json.dumps({"status":"pass","starting_checkpoint":"seed-1/checkpoint-1500","continuation_steps":400,"final_checkpoint":"train/checkpoint-400","fresh_optimizer":True,"precheck_status":"pass","checkpoint_selection":"none; step400 fixed endpoint","confirmatory_lockbox":"data/continuation_confirmatory_v2.json","shuffle_dataset":False},indent=2)+"\n")' "$OUT_ROOT"
echo "[continuation400-v3] COMPLETE checkpoint-400 is the fixed candidate endpoint $(date -Is)" | tee -a "$LOG"
