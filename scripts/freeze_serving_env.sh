#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="logs/artifacts/serving-venv-freeze.txt"
mkdir -p "$(dirname "$OUT")"

{
  echo "# Protected serving venv freeze"
  echo "# Generated: $(date -Is)"
  echo "# This documents the existing .venv baseline and must not mutate it."
  echo
  .venv/bin/python --version
  .venv/bin/python -m pip --version
  echo
  .venv/bin/python -m pip freeze
} > "$OUT"

echo "wrote $OUT"
