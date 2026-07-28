#!/usr/bin/env bash
# Stop the project TRL vLLM generation server and its engine process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${VLLM_PORT:-8000}"
PIDFILE="${VLLM_PIDFILE:-logs/artifacts/trl_vllm_serve.pid}"
UNIT="${VLLM_UNIT:-lp-rl-vllm}"

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop "${UNIT}.service" >/dev/null 2>&1 || true
  systemctl --user reset-failed "${UNIT}.service" >/dev/null 2>&1 || true
fi

if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE" || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
fi

pgrep -u "$(id -u)" -f ".venv-train/bin/trl vllm-serve .*--port ${PORT}" | xargs -r kill -TERM || true
pgrep -u "$(id -u)" -f "VLLM::EngineCore" | xargs -r kill -TERM || true
sleep 2
pgrep -u "$(id -u)" -f ".venv-train/bin/trl vllm-serve .*--port ${PORT}" | xargs -r kill -KILL || true
pgrep -u "$(id -u)" -f "VLLM::EngineCore" | xargs -r kill -KILL || true
rm -f "$PIDFILE"
echo "stopped vLLM server on port ${PORT}"
