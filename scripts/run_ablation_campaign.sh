#!/usr/bin/env bash
# Causal-ablation campaign: 4 reward arms x N paired seeds on the 500-task
# ablation catalog, then a held-out evaluation under ONE arm-neutral scorer.
#
#   A0  full six-dimensional risk vector (control / production reward)
#   B1  no privilege penalty
#   A2  flat binary over-privilege tagging
#   A4  no task-conditioned required-risk envelope
#
# Everything except the reward is matched: same base checkpoint, same catalog,
# same task order (shuffle_dataset=false), same seeds, same sampling.
#
# Evaluation is deliberately scored under A0 for ALL arms. Scoring each arm with
# its own reward would ask a different question of each and make the comparison
# meaningless; the outcome metrics reported from these rollouts (over-privileged
# success, excess-risk magnitude, ordinary success) are arm-neutral.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"
lp_pin_path

OUT_ROOT="${ABL_OUT_ROOT:-logs/artifacts/ablation_campaign_$(date +%F_%H%M%S)}"
CATALOG="${ABL_CATALOG:-data/ablation500_v1.json}"
HELDOUT="${ABL_HELDOUT:-data/ablation500_heldout_v1.json}"
EXPECTED_CATALOG_SHA="${ABL_EXPECTED_CATALOG_SHA:-06528ecedab2f2ce76ff1d3f8658a4e2b9569882c43ce19f02101adbe8bc852e}"
EXPECTED_HELDOUT_SHA="${ABL_EXPECTED_HELDOUT_SHA:-ea27359c85e875abb50074a64d6370bd43f818cb180cf9b914f2255e78d665cb}"
ARMS="${ABL_ARMS:-A0 B1 A2 A4}"
SEEDS="${ABL_SEEDS:-0 1 2}"
STEPS="${ABL_STEPS:-400}"
GENS="${ABL_GENS:-8}"
TRAIN_CAP_="${ABL_TRAIN_CAP:-86400s}"
EVAL_CAP_="${ABL_EVAL_CAP:-21600s}"
BASE_ADAPTER="${ABL_BASE_ADAPTER:-}"     # empty = train from the base model
DRY_RUN="${ABL_DRY_RUN:-0}"
RESUME="${ABL_RESUME:-0}"

export LP_RL_TASK_FILE="$CATALOG"
export LP_RL_RISK_ARCHIVE="${ABL_RISK_ARCHIVE:-data/risk_archive_v2.json}"
export LP_RL_SANDBOX=docker
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
export TMPDIR="${TMPDIR:-/dev/shm}"
PY=".venv/bin/python"

fail() { echo "[ablation] ERROR: $*" >&2; exit 2; }
say()  { echo "[ablation] $*"; }

[ -x "$PY" ] || fail "missing $PY"
[ -x ".venv-train/bin/python" ] || fail "missing .venv-train/bin/python"
for f in "$CATALOG" "$HELDOUT"; do [ -f "$f" ] || fail "missing $f"; done

# The arm switch must exist and must fail closed on an unknown value, or a typo
# in ABL_ARMS would silently train four copies of the control.
"$PY" - <<'PY' || fail "reward arm switch is not available or not fail-closed"
import os
from least_privilege_env.reward import scoring as S
assert set(("A0", "B1", "A2", "A4")).issubset(set(S.REWARD_ARMS)), S.REWARD_ARMS
os.environ["LP_RL_REWARD_ARM"] = "__nope__"
try:
    S.active_reward_arm()
except ValueError:
    pass
else:
    raise SystemExit("unknown arm did not raise")
os.environ.pop("LP_RL_REWARD_ARM")
assert S.active_reward_arm() == "A0"
PY

# Campaign arms map onto TWO axes: the reward arm (LP_RL_REWARD_ARM) and whether
# the broker may refuse actions (LP_RL_GATING). P-arms vary the second.
#   P1 = control reward, gating OFF   P3 = no privilege penalty, gating OFF
arm_reward() {
  case "$1" in
    P1) echo A0 ;;
    P3) echo B1 ;;
    *)  echo "$1" ;;
  esac
}
arm_gating() {
  case "$1" in
    P1|P3) echo off ;;
    *)     echo on ;;
  esac
}

# Validate the REQUESTED arms up front. Without this a typo in ABL_ARMS only
# surfaces when that arm's training run starts -- potentially many GPU-hours in.
for arm in $ARMS; do
  "$PY" -c 'import sys
from least_privilege_env.reward import scoring as S
sys.exit(0 if sys.argv[1] in S.REWARD_ARMS else 1)' "$(arm_reward "$arm")" \
    || fail "unknown arm in ABL_ARMS: '$arm' (valid: A0 A1 A2 A4 B1 B2 P1 P3)"
done

# Hash pin, actually enforced. Cheap and independent of the builder, which needs
# the whole rollout corpus to re-derive its answer.
actual_cat="$(sha256sum "$CATALOG" | cut -d' ' -f1)"
actual_held="$(sha256sum "$HELDOUT" | cut -d' ' -f1)"
[ "$actual_cat" = "$EXPECTED_CATALOG_SHA" ] \
  || fail "catalog hash drift: $actual_cat (expected $EXPECTED_CATALOG_SHA)"
[ "$actual_held" = "$EXPECTED_HELDOUT_SHA" ] \
  || fail "heldout hash drift: $actual_held (expected $EXPECTED_HELDOUT_SHA)"

if [ "${ABL_SKIP_BUILDER_CHECK:-0}" != 1 ]; then
  "$PY" scripts/build_ablation500.py --check --pad-to-target >/dev/null 2>&1 \
    || fail "$CATALOG does not reproduce from its builder; refusing to train on drifted data"
fi

TASK_IDS="$("$PY" -c 'import json,sys; print(",".join(r["task_id"] for r in json.load(open(sys.argv[1]))))' "$CATALOG")"
HELD_IDS="$("$PY" -c 'import json,sys; print(",".join(r["task_id"] for r in json.load(open(sys.argv[1]))))' "$HELDOUT")"
# Sizes are per-catalog, not universal: campaign 1 trains on ablation500 (500/50),
# the continuation arm campaign on continuation400_v3 (400/50). Hardcoding 500
# here silently rejected the second one.
N_TASKS="${ABL_EXPECTED_TASKS:-500}"
N_HELD="${ABL_EXPECTED_HELDOUT:-50}"
n_cat="$(lp_count_csv "$TASK_IDS")"; n_held="$(lp_count_csv "$HELD_IDS")"
[ "$n_cat" -eq "$N_TASKS" ] || fail "catalog is $n_cat tasks, expected $N_TASKS"
[ "$n_held" -eq "$N_HELD" ] || fail "heldout is $n_held tasks, expected $N_HELD"

# Size is only a proxy. Leakage is what would actually invalidate the held-out
# numbers, so assert that directly rather than inferring it from counts.
n_overlap="$("$PY" -c 'import json,sys
a={r["task_id"] for r in json.load(open(sys.argv[1]))}
b={r["task_id"] for r in json.load(open(sys.argv[2]))}
print(len(a & b))' "$CATALOG" "$HELDOUT")"
[ "$n_overlap" -eq 0 ] \
  || fail "train and heldout share $n_overlap task_ids; refusing to train"

say "arms=[$ARMS] seeds=[$SEEDS] steps=$STEPS gens=$GENS -> $OUT_ROOT"
n_runs=0; for a in $ARMS; do for s in $SEEDS; do n_runs=$((n_runs+1)); done; done
say "total training runs: $n_runs (plus $n_runs held-out evaluations)"

if [ "$DRY_RUN" = 1 ]; then
  say "DRY RUN: inputs, catalog reproducibility and the fail-closed arm switch verified"
  say "no GPU work, no services, no containers started"
  exit 0
fi

if pgrep -f 'train/train_trl_drgrpo.py' >/dev/null; then
  fail "a trainer is already active"
fi
"$PY" scripts/check_rootless_docker.py >/dev/null || fail "rootless Docker preflight failed"
[ "$(docker info --format '{{.DockerRootDir}}')" = /mnt/nvme-docker/lp-docker ] \
  || fail "Docker root is not the NVMe root"

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/campaign.log"
exec > >(tee -a "$LOG") 2>&1
lp_install_cleanup_trap "$LOG" ablation with_gpu

{
  echo "started_at=$(date -Is)"
  echo "catalog=$CATALOG sha256=$(sha256sum "$CATALOG" | cut -d' ' -f1)"
  echo "heldout=$HELDOUT sha256=$(sha256sum "$HELDOUT" | cut -d' ' -f1)"
  echo "arms=$ARMS seeds=$SEEDS steps=$STEPS generations=$GENS"
  echo "matched: base checkpoint, catalog, task order (shuffle_dataset=false), seeds, sampling"
  echo "varied: LP_RL_REWARD_ARM only"
  echo "evaluation_scorer=A0 for every arm (arm-neutral outcomes)"
} > "$OUT_ROOT/CAMPAIGN_CONFIG.txt"

for arm in $ARMS; do
  for seed in $SEEDS; do
    tag="$arm-seed$seed"
    out="$OUT_ROOT/$tag"
    # NOT `[ -e ] && [ ] && fail`: under `set -e` a short-circuiting AND-list
    # returns non-zero when the first test fails, which would kill the campaign
    # on the normal path where the output dir does not yet exist.
    if [ -e "$out" ] && [ "$RESUME" != 1 ]; then
      fail "refusing to overwrite $out"
    fi

    # Resume is decided per STAGE, not per cell. A single `continue` here would
    # also skip the held-out eval, which (a) silently drops the eval for a cell
    # whose checkpoint was supplied rather than trained -- the reused A0-seed1
    # continuation400_v3 cell -- and (b) makes a crash between train and eval
    # unrecoverable, since the checkpoint then exists forever with no eval.
    need_eps=$((N_HELD * GENS))
    # Brace-group the cat: under `set -euo pipefail` a bare `cat missing | wc -l`
    # returns non-zero, and a command substitution that returns non-zero kills the
    # script at the assignment -- silently, with no fail() message.
    have_eps="$( { cat "$out"/heldout/rollouts/*.jsonl 2>/dev/null || true; } | wc -l )"
    skip_train=0; skip_eval=0
    if [ "$RESUME" = 1 ]; then
      [ -s "$out/train/checkpoint-$STEPS/adapter_model.safetensors" ] && skip_train=1
      [ "${have_eps:-0}" -ge "$need_eps" ] && skip_eval=1
    fi

    if [ "$skip_train" = 1 ]; then
      say "$tag TRAIN resume-skip (checkpoint-$STEPS present)"
    else
    say "=== TRAIN $tag start $(date -Is)"
    scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
    lp_vllm_restart 180 "$LOG" || fail "$tag vLLM start failed"
    # Build the env explicitly. An inline ${BASE_ADAPTER:+...} expands to nothing
    # when unset, which swallows the line-continuation and makes bash parse the
    # NEXT assignment as a command ("TRAIN_TEMP_THRESHOLD_C=83: command not found").
    rw="$(arm_reward "$arm")"; gt="$(arm_gating "$arm")"
    say "$tag reward_arm=$rw gating=$gt"
    train_env=(
      LP_RL_REWARD_ARM="$rw" LP_RL_GATING="$gt"
      TRAIN_OUT="$out/train" TRAIN_SEED="$seed" TRAIN_MAX_STEPS="$STEPS"
      TRAIN_SAVE_STEPS=100 TRAIN_CAP="$TRAIN_CAP_" TRAIN_TASK_IDS="$TASK_IDS"
      TRAIN_TEMP_THRESHOLD_C=83 TRAIN_TEMP_MAX_WAIT_SEC=1800
    )
    [ -n "$BASE_ADAPTER" ] && train_env+=(TRAIN_ADAPTER_PATH="$BASE_ADAPTER")
    env "${train_env[@]}" bash scripts/run_train_capped.sh || fail "$tag training failed"
    [ -s "$out/train/checkpoint-$STEPS/adapter_model.safetensors" ] \
      || fail "$tag produced no step-$STEPS adapter"
    say "=== TRAIN $tag done $(date -Is)"
    fi

    # Held-out evaluation under the CONTROL scorer for every arm.
    if [ "$skip_eval" = 1 ]; then
      say "$tag EVAL resume-skip ($have_eps/$need_eps episodes present)"
    else
    say "=== EVAL $tag start $(date -Is)"
    scripts/stop_train_vllm.sh >/dev/null 2>&1 || true
    lp_vllm_restart 180 "$LOG" || fail "$tag eval vLLM start failed"
    # Evaluation is always the CONTROL reward with gating ON, for every arm --
    # otherwise each arm is judged by its own instrument and under its own rules.
    LP_RL_REWARD_ARM=A0 LP_RL_GATING=on LP_RL_TASK_FILE="$HELDOUT" \
    EVAL_LABEL="abl-$tag" EVAL_OUT="$out/heldout" \
    EVAL_ADAPTER_PATH="$out/train/checkpoint-$STEPS" EVAL_TASK_IDS="$HELD_IDS" \
    EVAL_NUM_GENERATIONS="$GENS" EVAL_SEED="$seed" EVAL_CAP="$EVAL_CAP_" \
      bash scripts/run_eval_capped.sh || fail "$tag eval failed"
    say "=== EVAL $tag done $(date -Is)"
    fi

    # Every cell must end with a usable checkpoint AND a complete eval, whether
    # it was trained now, resumed, or supplied. Otherwise the campaign can report
    # COMPLETE with holes in it.
    [ -s "$out/train/checkpoint-$STEPS/adapter_model.safetensors" ] \
      || fail "$tag has no step-$STEPS adapter after train stage"
    final_eps="$( { cat "$out"/heldout/rollouts/*.jsonl 2>/dev/null || true; } | wc -l )"
    [ "${final_eps:-0}" -ge "$need_eps" ] \
      || fail "$tag eval incomplete: $final_eps/$need_eps episodes"
  done
done

printf '{"status":"complete","arms":"%s","seeds":"%s","steps":%s,"eval_scorer":"A0"}\n' \
  "$ARMS" "$SEEDS" "$STEPS" > "$OUT_ROOT/COMPLETION_GUARD.json"
say "CAMPAIGN COMPLETE $(date -Is) -> $OUT_ROOT"
