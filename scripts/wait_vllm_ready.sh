#!/usr/bin/env bash
# Poll the TRL vLLM server health endpoint until ready or timeout.
set -uo pipefail

PORT="${VLLM_PORT:-8000}"
TRIES="${1:-84}"
LOG="logs/artifacts/trl_vllm_serve.log"
UNIT="${VLLM_UNIT:-lp-rl-vllm}"

for _ in $(seq 1 "$TRIES"); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health/" 2>/dev/null || true)"
  if [ "$code" = "200" ]; then
    echo "server-ready"
    exit 0
  fi
  if command -v systemctl >/dev/null 2>&1 && systemctl --user status "${UNIT}.service" >/dev/null 2>&1; then
    state="$(systemctl --user show "${UNIT}.service" --property=ActiveState --value 2>/dev/null || true)"
    if [ "$state" = "failed" ] || [ "$state" = "inactive" ]; then
      echo "server-exited; ${UNIT}.service state=$state"
      if [ -f "$LOG" ]; then
        tail -n 80 "$LOG"
      fi
      exit 1
    fi
  fi
  sleep 5
done

echo "server-timeout; last log lines:"
if [ -f "$LOG" ]; then
  tail -n 40 "$LOG"
fi
exit 1
