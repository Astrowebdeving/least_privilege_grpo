#!/usr/bin/env bash
# Select the 2,000-task benchmark. SOURCE this (do not execute it):
#
#   source scripts/benchmark_env.sh
#
# Without this, the library defaults to the 260-task development catalog
# (data/spike_tasks.json) — correct, but not the benchmark.
#
# The two variables MUST travel together. LP_RL_TASK_FILE alone runs the right
# tasks but leaves the risk archive covering only 260 of them, which does not
# error: it silently makes safe_success / over_privileged_success meaningless in
# every rollout summary. That is the single easiest mistake to make here, which
# is why this file sets both or neither.
#
# The risk archives are report-only. They never feed the training reward, which
# comes from each task's own required_risk field.

_lp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LP_RL_TASK_FILE="$_lp_root/data/expansion_v2_microslice.json"
export LP_RL_RISK_ARCHIVE="$_lp_root/data/risk_archive_v2.json"

for _lp_f in "$LP_RL_TASK_FILE" "$LP_RL_RISK_ARCHIVE"; do
  if [ ! -f "$_lp_f" ]; then
    echo "benchmark_env.sh: missing $_lp_f" >&2
    unset LP_RL_TASK_FILE LP_RL_RISK_ARCHIVE
    unset _lp_root _lp_f
    return 1 2>/dev/null || exit 1
  fi
done
unset _lp_root _lp_f

echo "benchmark: LP_RL_TASK_FILE=$LP_RL_TASK_FILE"
echo "           LP_RL_RISK_ARCHIVE=$LP_RL_RISK_ARCHIVE"
