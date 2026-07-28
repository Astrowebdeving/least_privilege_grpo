#!/usr/bin/env bash
# Detached launcher for the 2000-task v2 multi-week RL campaign.
#
# Self-determines the output root (FRESH vs RESUME) via
# logs/artifacts/.bigrun_out_root, exports the campaign env (v2 catalog + v2 risk
# archive + rootless-docker + RAM scratch), records tty/docker-root health
# markers, then exec's the multi-seed orchestrator under an outer timeout.
# run_multiseed.sh auto-resumes each seed from its newest full checkpoint, so
# re-launching with the SAME .bigrun_out_root continues a capped/crashed run.
#
# Launch from Windows PowerShell with the winredir pattern (NOT systemd-run --user,
# which dies on WSL teardown). See <host-scratch>\launch_bigrun_winredir.ps1, or:
#   $stdin='<host-scratch>\wsl_stdin_empty.txt'; Set-Content -LiteralPath $stdin -Value ''
#   Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d','Ubuntu-24.04','--exec',
#     'setsid','-w','bash','<repo>/scripts/launch_bigrun.sh') `
#     -RedirectStandardInput $stdin `
#     -RedirectStandardOutput '<host-scratch>\bigrun_v2_windows_stdout.log' `
#     -RedirectStandardError  '<host-scratch>\bigrun_v2_windows_stderr.log' `
#     -WindowStyle Hidden -PassThru
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PENDING=logs/artifacts/.bigrun_out_root
if [ -f "$PENDING" ] && [ -d "$(cat "$PENDING" 2>/dev/null)" ]; then
  OUT="$(cat "$PENDING")"          # RESUME the existing campaign dir
else
  OUT="logs/artifacts/bigrun_v2_$(date +%F_%H%M%S)"
  mkdir -p "$OUT"
  printf '%s' "$OUT" > "$PENDING"  # FRESH campaign dir, recorded for resume
fi

export DOCKER_HOST=unix:///run/user/1000/docker.sock
export XDG_RUNTIME_DIR=/run/user/1000
export TMPDIR=/dev/shm
export LP_RL_TASK_FILE=data/expansion_v2_microslice.json
export LP_RL_RISK_ARCHIVE=data/risk_archive_v2.json
export MULTISEED_OUT_ROOT="$OUT"
export MULTISEED_SEEDS="${MULTISEED_SEEDS:-0,1,2}"
export MULTISEED_SAVE_STEPS="${MULTISEED_SAVE_STEPS:-50}"
export MULTISEED_SEED_CAP="${MULTISEED_SEED_CAP:-604800s}"      # 7d/seed ceiling: verbose v2 tasks run ~315s/step so a 1500-step epoch can take ~5.5d (72h was too short; 2026-07-01)
export MULTISEED_COOLDOWN_SEC="${MULTISEED_COOLDOWN_SEC:-7200}" # 2h fixed rest between seeds
export MULTISEED_TEMP_THRESHOLD_C="${MULTISEED_TEMP_THRESHOLD_C:-83}"
export MULTISEED_VLLM_GPU_UTIL="${MULTISEED_VLLM_GPU_UTIL:-0.85}"
# Disable ONLY the completion-clipping abort: verbose tasks (structured_output/JSON) legitimately
# clip at the token cap and can cluster, tripping the default clipped_max=0.5/patience=3 guard
# (seed-0 aborted at step 317 on a benign clip cluster, ISR 0.97/reward 0.80 = healthy). ISR/GPU/
# container aborts stay at full sensitivity; clipped_ratio is still logged for manual monitoring.
export TRAIN_MONITOR_CLIPPED_MAX="${TRAIN_MONITOR_CLIPPED_MAX:-1.01}"
# MULTISEED_MAX_STEPS left unset -> run_multiseed derives it from the train split (1500 = one epoch).

L="$OUT/launcher_stdout.log"
{
  printf '[bigrun-launcher] start %s out=%s\n' "$(date -Is)" "$OUT"
  printf '[bigrun-launcher] seeds=%s save=%s seed-cap=%s cooldown=%ss temp=%sC util=%s\n' \
    "$MULTISEED_SEEDS" "$MULTISEED_SAVE_STEPS" "$MULTISEED_SEED_CAP" \
    "$MULTISEED_COOLDOWN_SEC" "$MULTISEED_TEMP_THRESHOLD_C" "$MULTISEED_VLLM_GPU_UTIL"
  printf '[bigrun-launcher] catalog=%s archive=%s\n' "$LP_RL_TASK_FILE" "$LP_RL_RISK_ARCHIVE"
  printf '[bigrun-launcher] tty='; (tty || true)
  printf '[bigrun-launcher] docker-root='; (docker info --format '{{.DockerRootDir}}' || true)
} >> "$L" 2>&1

# Outer ceiling: covers 3 seeds * 7d cap + cooldowns; the run exits naturally when each
# seed reaches checkpoint-MAX well before this. (22 days.)
exec timeout --foreground --kill-after=300s 1900800s bash scripts/run_multiseed.sh >> "$L" 2>&1
