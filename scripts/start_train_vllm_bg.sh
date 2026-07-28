#!/usr/bin/env bash
# Start the TRL vLLM generation server detached from the caller.
#
# Default placement is GPU 1 so GPU 0 stays free for the trainer:
#   scripts/start_train_vllm_bg.sh
#   scripts/wait_vllm_ready.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${VLLM_PORT:-8000}"
LOG="${VLLM_LOG:-logs/artifacts/trl_vllm_serve.log}"
PIDFILE="${VLLM_PIDFILE:-logs/artifacts/trl_vllm_serve.pid}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
UNIT="${VLLM_UNIT:-lp-rl-vllm}"
START_MODE="${VLLM_START_MODE:-systemd}"

mkdir -p logs/artifacts

if pgrep -u "$(id -u)" -f ".venv-train/bin/trl vllm-serve .*--port ${PORT}" >/dev/null; then
  echo "vLLM server already appears to be running on port ${PORT}"
  exit 0
fi

: > "$LOG"
if [ "$START_MODE" != "nohup" ] && command -v systemd-run >/dev/null 2>&1 && systemctl --user is-system-running >/dev/null 2>&1; then
  systemctl --user reset-failed "${UNIT}.service" >/dev/null 2>&1 || true
  # systemd --user units start from a clean environment, so forward the optional
  # vLLM memory/context knobs explicitly when the caller sets them (e.g. a
  # smaller gpu-memory-utilization when the inference GPU is partly occupied).
  EXTRA_SETENV=()
  [ -n "${VLLM_GPU_UTIL:-}" ] && EXTRA_SETENV+=(--setenv="VLLM_GPU_UTIL=$VLLM_GPU_UTIL")
  [ -n "${VLLM_MAX_MODEL_LEN:-}" ] && EXTRA_SETENV+=(--setenv="VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN")
  systemd-run --user \
    --unit="$UNIT" \
    --collect \
    --property="WorkingDirectory=$ROOT" \
    --property="StandardOutput=append:$ROOT/$LOG" \
    --property="StandardError=append:$ROOT/$LOG" \
    --setenv="CUDA_VISIBLE_DEVICES=$GPU" \
    "${EXTRA_SETENV[@]}" \
    "$ROOT/scripts/serve_train_vllm.sh" "$@"
  sleep 1
  systemctl --user show "${UNIT}.service" --property=MainPID --value > "$PIDFILE" 2>/dev/null || true
  echo "started vLLM systemd unit=${UNIT}.service gpu=$GPU port=$PORT log=$LOG"
else
  # Fallback for non-systemd environments. WSL should normally take the branch
  # above; plain backgrounding can be fragile when launched through wsl.exe.
  setsid env CUDA_VISIBLE_DEVICES="$GPU" nohup scripts/serve_train_vllm.sh "$@" > "$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PIDFILE"
  echo "started vLLM server pid=$pid gpu=$GPU port=$PORT log=$LOG start-mode=$START_MODE"
fi
