#!/usr/bin/env bash
# Select the complete 2,000-task benchmark and merge its expansion risk archive
# with the 260-task base archive loaded by default.
set -euo pipefail

LP_BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LP_RL_TASK_FILE="$LP_BENCHMARK_ROOT/data/expansion_v2_microslice.json"
export LP_RL_RISK_ARCHIVE="$LP_BENCHMARK_ROOT/data/risk_archive_v2.json"

printf 'LP_RL_TASK_FILE=%s\nLP_RL_RISK_ARCHIVE=%s\n' \
  "$LP_RL_TASK_FILE" "$LP_RL_RISK_ARCHIVE"
