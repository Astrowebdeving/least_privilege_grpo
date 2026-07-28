#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -n "${UV:-}" ]; then
  :
elif command -v uv >/dev/null 2>&1; then
  UV="uv"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
  echo "uv not found; install uv or set UV=/path/to/uv" >&2
  exit 2
fi
TRAIN_PY=".venv-train/bin/python"
OUT="logs/artifacts/train-venv-freeze.txt"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing protected serving .venv; refusing to continue." >&2
  exit 2
fi

if [ ! -x "$TRAIN_PY" ]; then
  "$UV" venv .venv-train
else
  echo "Reusing existing .venv-train; remove it first or run with UV_VENV_CLEAR=1 uv venv .venv-train to rebuild from scratch."
fi
"$UV" pip install --python "$TRAIN_PY" -e ".[test]"
echo "Installing the environment_factory-compatible TRL/vLLM stack."
echo "Note: TRL's vllm extra currently resolves to vLLM<=0.18, which requires transformers<5."
echo "environment_factory requires transformers>=5.2, so install vLLM explicitly instead of using trl[vllm]."
"$UV" pip install --python "$TRAIN_PY" "trl==1.5.1" "transformers==5.10.2" "vllm==0.22.1"

mkdir -p "$(dirname "$OUT")"
{
  echo "# Training venv freeze"
  echo "# Generated: $(date -Is)"
  echo
  "$TRAIN_PY" --version
  "$UV" --version
  echo
  "$UV" pip freeze --python "$TRAIN_PY"
} > "$OUT"

PYTHONPATH=. "$TRAIN_PY" scripts/check_train_stack.py
PYTHONPATH=. "$TRAIN_PY" scripts/check_train_import.py
"$UV" pip check --python "$TRAIN_PY"

echo "training venv ready; wrote $OUT"
