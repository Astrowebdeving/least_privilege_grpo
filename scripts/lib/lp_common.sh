#!/usr/bin/env bash
# Shared helpers for the least_privilege_rl launcher scripts. SOURCE this (do not
# execute). Extracted 2026-06-26 to de-duplicate the cd-to-root, PATH pinning,
# cleanup traps, GPU thermal read, CSV task counting, and vLLM-restart sequence
# that were copy-pasted across run_eval_capped / run_train_capped / run_eval_suite
# / run_multiseed. Each function is a verbatim lift of the prior inline body, so
# behavior is preserved; the small vLLM lifecycle scripts (serve/start/stop/wait/
# prune) are intentionally NOT folded in here and are still called as before.
#
# All functions assume the caller's CWD is the project root, which sourcing this
# guarantees (LP_ROOT below). Typical use:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/lp_common.sh"
#   lp_pin_path

# Resolve + cd to the project root (this file lives in scripts/lib/).
LP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LP_ROOT" || { echo "lp_common: cannot cd to project root '$LP_ROOT'" >&2; return 1 2>/dev/null || exit 1; }

# systemd --user units start with a minimal PATH that omits /usr/lib/wsl/lib
# (nvidia-smi). Without it the trainer's GPU probe returns 0 and --vllm-mode auto
# wrongly colocates a 2nd engine on GPU0 that OOMs. Pin a robust PATH.
lp_pin_path() {
  export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
  # Per-command sandbox temp (episode workspace materialization + the per-command
  # snapshot_tree sha256 + verifier tree copies) is the fattest per-step disk cost.
  # Keep it on RAM (tmpfs /dev/shm), NOT the HDD-backed / (= /tmp). Docker's
  # data-root is separately on the NVMe vhdx (the NVMe Docker data-root). See runbook
  # "Hardware and I/O — storage policy". Honors an explicit TMPDIR override.
  export TMPDIR="${TMPDIR:-/dev/shm}"
}

# Echo the max temperature.gpu across all GPUs, or "" if unreadable.
lp_gpu_max_temp() {
  local out
  out="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)" || return 1
  echo "$out" | awk 'BEGIN{m=""} /^[0-9]+$/{if(m==""||$1>m)m=$1} END{print m}'
}

# Count non-empty comma-separated fields. Usage: n="$(lp_count_csv "$CSV")"
lp_count_csv() {
  local csv="$1" n=0 t
  local -a arr
  IFS=',' read -r -a arr <<<"$csv"
  for t in "${arr[@]}"; do [ -n "${t//[[:space:]]/}" ] && n=$((n + 1)); done
  echo "$n"
}

# Restart the TRL vLLM server on GPU 1 and wait until ready: stop, prune, start
# detached on GPU1, wait. Honors VLLM_GPU_UTIL (exported by the caller). Returns
# non-zero if start or readiness fails. Usage: lp_vllm_restart [tries] [logfile]
lp_vllm_restart() {
  local tries="${1:-120}" log="${2:-/dev/null}"
  scripts/stop_train_vllm.sh >>"$log" 2>&1 || true
  scripts/prune_lp_containers.sh >>"$log" 2>&1 || true
  CUDA_VISIBLE_DEVICES=1 scripts/start_train_vllm_bg.sh >>"$log" 2>&1 || return 1
  scripts/wait_vllm_ready.sh "$tries" >>"$log" 2>&1 || return 1
}

# Install the standard cleanup trap: on ANY exit, stop vLLM + prune lp-rl
# containers exactly once (idempotent guard), optionally print GPU memory.
# TERM/INT map to the conventional 143/130 so the cap's timeout signalling is
# preserved. Usage: lp_install_cleanup_trap <logfile> <tag> [with_gpu]
_LP_CLEANED=0
lp_install_cleanup_trap() {
  _LP_CLEANUP_LOG="$1"
  _LP_CLEANUP_TAG="$2"
  _LP_CLEANUP_GPU="${3:-}"
  trap '_lp_cleanup' EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
}
_lp_cleanup() {
  [ "$_LP_CLEANED" = "1" ] && return 0
  _LP_CLEANED=1
  {
    echo "[$_LP_CLEANUP_TAG] cleanup begin $(date -Is)"
    scripts/stop_train_vllm.sh || true
    scripts/prune_lp_containers.sh || true
    if [ -n "$_LP_CLEANUP_GPU" ]; then
      echo "[$_LP_CLEANUP_TAG] gpu after cleanup:"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
    fi
    echo "[$_LP_CLEANUP_TAG] cleanup done $(date -Is)"
  } >>"$_LP_CLEANUP_LOG" 2>&1
}

# Bounded, never-deadlocking pre-launch thermal gate (plain logging variant used
# by run_train_capped; run_multiseed keeps its structured thermal_event JSONL
# version). Waits until GPU temp <= threshold or max_wait elapses, then proceeds
# (hardware throttle is the backstop). Usage:
#   lp_temp_gate <threshold_c> <max_wait_s> <poll_s> <logfile> <tag>
lp_temp_gate() {
  local threshold="$1" max_wait="$2" poll="$3" log="$4" tag="$5"
  [ "$threshold" -gt 0 ] 2>/dev/null || return 0
  local waited=0 temp
  temp="$(lp_gpu_max_temp || true)"
  [ -z "$temp" ] && { echo "[$tag][thermal] temp unreadable; proceeding (hardware throttle backstop)" >>"$log"; return 0; }
  [ "$temp" -le "$threshold" ] && { echo "[$tag][thermal] temp=${temp}C <= ${threshold}C; ok" >>"$log"; return 0; }
  echo "[$tag][thermal] temp=${temp}C > ${threshold}C; waiting up to ${max_wait}s" >>"$log"
  while [ "$waited" -lt "$max_wait" ]; do
    sleep "$poll"; waited=$((waited + poll))
    temp="$(lp_gpu_max_temp || true)"
    [ -z "$temp" ] && { echo "[$tag][thermal] temp unreadable mid-wait; proceeding" >>"$log"; return 0; }
    [ "$temp" -le "$threshold" ] && { echo "[$tag][thermal] temp=${temp}C after ${waited}s; ok" >>"$log"; return 0; }
  done
  echo "[$tag][thermal] temp=${temp}C still hot after ${max_wait}s; proceeding (hardware throttle backstop)" >>"$log"
}
