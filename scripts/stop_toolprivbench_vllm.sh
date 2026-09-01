#!/usr/bin/env bash
# Stop only the ToolPrivBench vLLM process/unit recorded by its dedicated
# unit name and pidfile. Do not touch unrelated WSL or Windows GPU processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${TOOLPRIV_VLLM_PORT:-8010}"
PIDFILE="${TOOLPRIV_VLLM_PIDFILE:-logs/artifacts/toolprivbench_vllm.pid}"
UNIT="${TOOLPRIV_VLLM_UNIT:-lp-toolpriv-vllm}"

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop "${UNIT}.service" >/dev/null 2>&1 || true
  systemctl --user reset-failed "${UNIT}.service" >/dev/null 2>&1 || true
fi

if [ ! -f "$PIDFILE" ]; then
  echo "ToolPrivBench vLLM stopped; pidfile absent"
  exit 0
fi

pid="$(cat "$PIDFILE" 2>/dev/null || true)"
if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PIDFILE"
  echo "ToolPrivBench vLLM already stopped"
  exit 0
fi

cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
case "$cmdline" in
  *vllm.entrypoints.openai.api_server*"--port $PORT"*) ;;
  *)
    echo "refusing to stop pid=$pid; command does not match ToolPrivBench port $PORT" >&2
    exit 2
    ;;
esac

kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
fi
rm -f "$PIDFILE"
echo "stopped ToolPrivBench vLLM pid=$pid port=$PORT"