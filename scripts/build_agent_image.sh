#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${LP_RL_DOCKER_IMAGE:-lp-rl-worker:latest}"
docker build -f docker/Dockerfile.agent-python -t "$IMAGE" .
echo "built $IMAGE"
