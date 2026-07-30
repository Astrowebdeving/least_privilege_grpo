#!/usr/bin/env bash
# Publication-blocking frozen v2 confirmatory evaluation.
# Runs base + checkpoint-1500 seeds 0/1/2 over the exact frozen 500 tasks with
# registered cost tiers: 276 ordinary x4 and 224 safety-heavy x8 = 2896
# episodes/policy. Every tier is independently capped and every policy must pass
# exact task/generation membership, unique-episode, adapter, and terminal-stop
# telemetry guards before comparisons are written.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"
lp_pin_path

OUT_ROOT="${FROZEN500_OUT_ROOT:-logs/artifacts/frozen500_confirmatory_$(date +%F_%H%M%S)}"
TASK_FILE="data/expansion_v2_microslice.json"
TASK_IDS_FILE="data/validation_task_ids_v2.txt"
FREEZE="data/validation_frozen_2000_2026-06-30.json"
TIER_MANIFEST="data/validation_generation_tiers_v2.json"
BASE_RISK_ARCHIVE="data/risk_archive.json"
EXTRA_RISK_ARCHIVE="data/risk_archive_v2.json"
ADAPTER_ROOT="${FROZEN500_ADAPTER_ROOT:-logs/artifacts/bigrun_v2_2026-07-01_003238}"
SEED="${FROZEN500_SEED:-0}"
ORDINARY_CAP="${FROZEN500_ORDINARY_CAP:-64800s}"
SAFETY_CAP="${FROZEN500_SAFETY_CAP:-64800s}"
MONITOR_CLIPPED_MAX="${FROZEN500_MONITOR_CLIPPED_MAX:-1.01}"
DRY_RUN="${FROZEN500_DRY_RUN:-0}"
export VLLM_GPU_UTIL="${FROZEN500_VLLM_GPU_UTIL:-0.85}"
export LP_RL_TASK_FILE="$TASK_FILE"
export LP_RL_RISK_ARCHIVE="$EXTRA_RISK_ARCHIVE"
export LP_RL_SANDBOX="docker"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
export TMPDIR="${TMPDIR:-/dev/shm}"

PY=".venv/bin/python"
TRAIN_PY=".venv-train/bin/python"

fail() { echo "[frozen500] ERROR: $*" >&2; exit 2; }
[ -x "$PY" ] || fail "missing $PY"
[ -x "$TRAIN_PY" ] || fail "missing $TRAIN_PY"
[ -f "$TASK_FILE" ] || fail "missing task catalog $TASK_FILE"
[ -f "$TASK_IDS_FILE" ] || fail "missing task ID file $TASK_IDS_FILE"
[ -f "$FREEZE" ] || fail "missing freeze $FREEZE"
[ -f "$TIER_MANIFEST" ] || fail "missing tier manifest $TIER_MANIFEST"
[ -f "$BASE_RISK_ARCHIVE" ] || fail "missing base risk archive $BASE_RISK_ARCHIVE"
[ -f "$EXTRA_RISK_ARCHIVE" ] || fail "missing additive risk archive $EXTRA_RISK_ARCHIVE"
"$PY" scripts/build_validation_tiers_v2.py --check || fail "tier manifest/freeze check failed"

ORDINARY_IDS="$("$PY" scripts/build_validation_tiers_v2.py --print-csv ordinary)" || fail "cannot load ordinary tier"
SAFETY_IDS="$("$PY" scripts/build_validation_tiers_v2.py --print-csv safety_heavy)" || fail "cannot load safety-heavy tier"
[ "$(lp_count_csv "$ORDINARY_IDS")" -eq 276 ] || fail "ordinary tier is not 276 tasks"
[ "$(lp_count_csv "$SAFETY_IDS")" -eq 224 ] || fail "safety-heavy tier is not 224 tasks"

# FROZEN500_POLICIES selects registered arms. FROZEN500_ADAPTERS optionally
# supplies name=path[,name=path...] post-hoc arms; when set, only those custom
# adapters run, without pretending they are registered seed checkpoints.
POLICIES="${FROZEN500_POLICIES:-base seed-0 seed-1 seed-2}"
CUSTOM_ADAPTERS="${FROZEN500_ADAPTERS:-}"
declare -a NAMES=() PATHS=()
if [ -n "$CUSTOM_ADAPTERS" ]; then
  IFS=',' read -r -a CUSTOM_PAIRS <<<"$CUSTOM_ADAPTERS"
  for pair in "${CUSTOM_PAIRS[@]}"; do
    name="${pair%%=*}"; path="${pair#*=}"
    [ -n "$name" ] && [ -n "$path" ] && [ "$name" != "$path" ] \
      || fail "bad FROZEN500_ADAPTERS entry: $pair"
    NAMES+=("$name"); PATHS+=("$path")
  done
else
  for p in $POLICIES; do
    case "$p" in
      base) NAMES+=(base); PATHS+=("") ;;
      seed-0|seed-1|seed-2) NAMES+=("$p"); PATHS+=("$ADAPTER_ROOT/$p/checkpoint-1500") ;;
      *) fail "unknown policy in FROZEN500_POLICIES: $p" ;;
    esac
  done
fi
for i in "${!NAMES[@]}"; do
  [ -z "${PATHS[$i]}" ] && continue
  [ -s "${PATHS[$i]}/adapter_model.safetensors" ] || fail "missing adapter ${PATHS[$i]}"
done

if [ "$DRY_RUN" = "1" ]; then
  echo "[frozen500] DRY RUN: freeze and adapters verified; no services or containers touched"
  echo "[frozen500] out=$OUT_ROOT policies=${NAMES[*]} ordinary=276x4 safety-heavy=224x8 episodes/policy=2896"
  echo "[frozen500] caps ordinary=$ORDINARY_CAP safety=$SAFETY_CAP vllm-util=$VLLM_GPU_UTIL seed=$SEED"
  exit 0
fi

# Refuse resource conflicts before touching project services. Set
# CONFLICTING_UNITS to any systemd --user units that may legitimately own the
# generation GPU on your machine; empty by default.
for unit in ${CONFLICTING_UNITS:-}; do
  if systemctl --user is-active --quiet "$unit"; then
    fail "conflicting unit is active: $unit"
  fi
done

# Normalize only this project's internal eval lifecycle after conflict checks.
scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
scripts/prune_lp_containers.sh >/dev/null 2>&1 || true

# Storage policy, enforced fail-closed because both mistakes are silent and
# expensive: a Docker root on spinning disk collapses episode throughput, and a
# disk-backed TMPDIR makes per-command workspace materialization the dominant
# cost. Override for your hardware; the defaults are ours.
EXPECTED_DOCKER_ROOT="${EXPECTED_DOCKER_ROOT:-/mnt/nvme-docker/lp-docker}"
EXPECTED_TMPDIR="${EXPECTED_TMPDIR:-/dev/shm}"

"$PY" scripts/check_rootless_docker.py || fail "rootless Docker preflight failed"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')" || fail "docker info failed"
[ "$DOCKER_ROOT" = "$EXPECTED_DOCKER_ROOT" ] || fail "Docker root is $DOCKER_ROOT, expected $EXPECTED_DOCKER_ROOT (set EXPECTED_DOCKER_ROOT to override)"
[ "$TMPDIR" = "$EXPECTED_TMPDIR" ] || fail "TMPDIR is $TMPDIR, expected RAM scratch $EXPECTED_TMPDIR (set EXPECTED_TMPDIR to override)"
[ -d "$TMPDIR" ] && [ -w "$TMPDIR" ] || fail "RAM scratch is not writable: $TMPDIR"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -ge 2 ] || fail "two GPUs are required"
IFS=',' read -r _ GPU0_FREE GPU0_TOTAL <<<"${GPU_ROWS[0]}"
IFS=',' read -r _ GPU1_FREE GPU1_TOTAL <<<"${GPU_ROWS[1]}"
GPU0_FREE="${GPU0_FREE//[[:space:]]/}"; GPU0_TOTAL="${GPU0_TOTAL//[[:space:]]/}"
GPU1_FREE="${GPU1_FREE//[[:space:]]/}"; GPU1_TOTAL="${GPU1_TOTAL//[[:space:]]/}"
MIN_GPU0_FREE="${FROZEN500_MIN_GPU0_FREE_MIB:-28000}"
VLLM_REQUIRED="$(awk -v total="$GPU1_TOTAL" -v util="$VLLM_GPU_UTIL" 'BEGIN { printf "%.0f", total * util }')"
[ "$GPU0_FREE" -ge "$MIN_GPU0_FREE" ] || fail "GPU0 has ${GPU0_FREE} MiB free; need >=${MIN_GPU0_FREE} for trainer"
# GPU1 free-memory check is WARN-ONLY (owner override 2026-07-16): the ~10 GB
# nvidia-smi shows "used" on GPU1 at idle is Windows WDDM desktop/graphics
# working set (no WSL compute process), and WDDM pages it to system RAM under
# CUDA pressure — the campaign and diag50 both ran vLLM at util 0.85 on GPU1
# with it present. vLLM startup itself is the real gate; watch readiness.
[ "$GPU1_FREE" -ge "$VLLM_REQUIRED" ] || echo "[frozen500] WARN: GPU1 has ${GPU1_FREE} MiB free; vLLM util=$VLLM_GPU_UTIL requests about ${VLLM_REQUIRED} MiB (WDDM shared-graphics expected to page out)"

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/frozen500.log"
cp "$TIER_MANIFEST" "$OUT_ROOT/validation_generation_tiers_v2.json"
{
  echo "launched_at=$(date -Is)"
  echo "task_file=$TASK_FILE"
  echo "task_file_sha256=$(sha256sum "$TASK_FILE" | cut -d' ' -f1)"
  echo "task_ids_file=$TASK_IDS_FILE"
  echo "task_ids_sha256=$(sha256sum "$TASK_IDS_FILE" | cut -d' ' -f1)"
  echo "freeze=$FREEZE"
  echo "tier_manifest=$TIER_MANIFEST"
  echo "tier_assignment_sha256=$("$PY" -c 'import json; print(json.load(open("data/validation_generation_tiers_v2.json"))["assignment_sha256"])')"
  echo "task_catalog_count=2000"
  echo "validation_tasks=500"
  echo "ordinary_tasks=276 ordinary_generations=4 ordinary_episodes=1104"
  echo "safety_heavy_tasks=224 safety_heavy_generations=8 safety_heavy_episodes=1792"
  echo "episodes_per_policy=2896"
  echo "policies=${NAMES[*]}"
  echo "adapter_root=$ADAPTER_ROOT"
  echo "seed=$SEED"
  echo "ordinary_cap=$ORDINARY_CAP"
  echo "safety_cap=$SAFETY_CAP"
  echo "monitor_clipped_max=$MONITOR_CLIPPED_MAX"
  echo "vllm_gpu=1 trainer_gpu=0 vllm_gpu_util=$VLLM_GPU_UTIL"
  echo "docker_host=$DOCKER_HOST docker_root=$DOCKER_ROOT"
  echo "tmpdir=$TMPDIR"
  echo "gpu0_free_mib=$GPU0_FREE gpu0_total_mib=$GPU0_TOTAL"
  echo "gpu1_free_mib=$GPU1_FREE gpu1_total_mib=$GPU1_TOTAL"
} >"$OUT_ROOT/RUN_CONFIG.txt"
{
  echo "FROZEN500_OUT_ROOT=$OUT_ROOT"
  echo "FROZEN500_ADAPTER_ROOT=$ADAPTER_ROOT"
  echo "FROZEN500_SEED=$SEED"
  echo "FROZEN500_ORDINARY_CAP=$ORDINARY_CAP"
  echo "FROZEN500_SAFETY_CAP=$SAFETY_CAP"
  echo "FROZEN500_MONITOR_CLIPPED_MAX=$MONITOR_CLIPPED_MAX"
  echo "FROZEN500_VLLM_GPU_UTIL=$VLLM_GPU_UTIL"
  echo "FROZEN500_ADAPTERS=$CUSTOM_ADAPTERS"
  echo "DOCKER_HOST=$DOCKER_HOST"
  echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
  echo "TMPDIR=$TMPDIR"
} >"$OUT_ROOT/RUN_ENV.txt"

lp_install_cleanup_trap "$LOG" frozen500 with_gpu
{
  echo "[frozen500] start $(date -Is) out=$OUT_ROOT"
  echo "[frozen500] ${#NAMES[@]} policies; ordinary=276x4; safety-heavy=224x8; expected=2896/policy"
  echo "[frozen500] measured reference: terminal-stop diag50 400 eps/policy in 2926-4502s; per-tier 18h caps exceed projection by >25%"
} | tee -a "$LOG"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; adapter="${PATHS[$i]}"; policy_out="$OUT_ROOT/$name"
  mkdir -p "$policy_out"
  for tier in ordinary safety_heavy; do
    if [ "$tier" = "ordinary" ]; then
      ids="$ORDINARY_IDS"; generations=4; cap="$ORDINARY_CAP"; expected=1104
    else
      ids="$SAFETY_IDS"; generations=8; cap="$SAFETY_CAP"; expected=1792
    fi
    tier_out="$policy_out/$tier"
    if [ -e "$tier_out/eval.log" ] || [ -d "$tier_out/rollouts" ]; then
      fail "refusing to append to existing incomplete tier output: $tier_out"
    fi
    echo "[frozen500] policy=$name tier=$tier start $(date -Is) adapter=${adapter:-none}" | tee -a "$LOG"
    lp_vllm_restart 180 "$LOG" || fail "$name/$tier vLLM start failed"
    EVAL_LABEL="$name-$tier" EVAL_OUT="$tier_out" EVAL_ADAPTER_PATH="$adapter" \
    EVAL_TASK_IDS="$ids" EVAL_NUM_GENERATIONS="$generations" \
    EVAL_SEED="$SEED" EVAL_CAP="$cap" EVAL_MONITOR_CLIPPED_MAX="$MONITOR_CLIPPED_MAX" \
      bash scripts/run_eval_capped.sh
    rc=$?
    echo "[frozen500] policy=$name tier=$tier evaluator-rc=$rc $(date -Is)" | tee -a "$LOG"
    [ "$rc" -eq 0 ] || exit "$rc"
    grep -Fq "num-generations=$generations" "$tier_out/eval.log" || fail "$name/$tier generation count not recorded"
    if [ -n "$adapter" ]; then
      grep -Fq "adapter-path=$adapter" "$tier_out/eval.log" || fail "$name/$tier adapter path not recorded"
      if grep -Eqi "missing adapter keys|unexpected keys.*adapter" "$tier_out/eval.log"; then
        fail "$name/$tier adapter-load warning"
      fi
    fi
    echo "[frozen500] policy=$name tier=$tier complete expected-episodes=$expected $(date -Is)" | tee -a "$LOG"
  done

  "$PY" scripts/summarize_rollouts.py \
    "$policy_out/ordinary/rollouts" "$policy_out/safety_heavy/rollouts" \
    --archive "$BASE_RISK_ARCHIVE" --report "$policy_out/rollout_review.json" >>"$LOG" 2>&1 \
    || fail "$name rollout summary failed"
  "$PY" scripts/validate_frozen500_run.py \
    --tier-manifest "$TIER_MANIFEST" \
    --ordinary-rollouts "$policy_out/ordinary/rollouts" \
    --safety-rollouts "$policy_out/safety_heavy/rollouts" \
    --policy "$name" --report "$policy_out/run_validation.json" >>"$LOG" 2>&1 \
    || fail "$name exact membership/telemetry guard failed"
  echo "[frozen500] policy=$name complete and guarded $(date -Is)" | tee -a "$LOG"
done

base_report="$OUT_ROOT/base/rollout_review.json"
have_base=0
for name in "${NAMES[@]}"; do [ "$name" = base ] && have_base=1; done
safety_args=()
[ "$have_base" = 1 ] && safety_args=(--policy base "$base_report")
for name in "${NAMES[@]}"; do
  [ "$name" = base ] && continue
  report="$OUT_ROOT/$name/rollout_review.json"
  if [ "$have_base" = 1 ]; then
    "$PY" scripts/compare_rollout_reports.py "$base_report" "$report" \
      --report "$OUT_ROOT/base_vs_${name}.json" >>"$LOG" 2>&1 \
      || fail "base vs $name comparison failed"
  fi
  safety_args+=(--policy "$name" "$report")
done
"$PY" scripts/safety_metrics_report.py "${safety_args[@]}" \
  --tasks "$TASK_FILE" --report "$OUT_ROOT/safety_metrics_v2.json" >>"$LOG" 2>&1 \
  || fail "v2 safety metrics failed"

"$PY" - "$OUT_ROOT" "${NAMES[@]}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
names = sys.argv[2:]
reports = [json.load(open(root / name / "run_validation.json")) for name in names]
assert all(report["status"] == "pass" for report in reports)
assert len({report["assignment_sha256"] for report in reports}) == 1
assert all(report["observed_episode_count"] == 2896 for report in reports)
assert all(report["post_terminal_calls"] == 0 for report in reports)
(root / "COMPLETION_GUARD.json").write_text(json.dumps({
    "status": "pass",
    "policies": [report["policy"] for report in reports],
    "episodes_per_policy": 2896,
    "total_episodes": 2896 * len(reports),
    "assignment_sha256": reports[0]["assignment_sha256"],
    "post_terminal_calls": 0,
    "identical_membership": True,
}, indent=2) + "\n")
PY

echo "[frozen500] COMPLETE: ${#NAMES[@]}x2896 episodes, all exact-membership and terminal-stop guards passed $(date -Is)" | tee -a "$LOG"
