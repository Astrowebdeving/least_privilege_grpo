#!/usr/bin/env bash
# Fast, RAM-backed Docker data-root for Docker-heavy batch jobs (oracle smoke,
# backend parity, eval rollouts) — ~20x faster than the F: HDD data-root, which
# thrashes under parallel container churn (K=12 went 0 tasks in 3 min on HDD;
# ~8.5 min for the full 2000-task parity on RAM).
#
# HOW IT WORKS: a real ext4 filesystem on a loopback file in /dev/shm. tmpfs alone
# cannot back Docker's overlayfs driver, but a loopback ext4 IS a genuine ext4
# block device, so overlayfs works — and the bytes live in RAM, so there is zero
# disk contention.
#
# VOLATILE SCRATCH ONLY: the data-root + the rebuilt image are lost on a WSL or
# Windows restart. Nothing of value is here — results, the resume-progress JSONL,
# and model checkpoints all live on the F: HDD under logs/artifacts. After a
# restart, just re-run this script (~1 min) to restore the fast Docker scratch.
#
# RUN AS ROOT (WSL needs no sudo password for this):
#   wsl -d Ubuntu-24.04 -u root -- bash -lc 'bash /home/<linux-user>/projects/least_privilege_rl/scripts/setup_ram_docker_root.sh'
#
# Also export TMPDIR=/dev/shm for any Docker run so per-command workspace temp +
# snapshot/verifier copies hit RAM too (the launchers/wrappers do this).
set -euo pipefail
IMG=/dev/shm/docker_ext4.img
MNT=/mnt/ramdocker
SIZE=16G
U=alex
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ "$(id -u)" = "0" ] || { echo "must run as root (wsl -u root)"; exit 1; }

echo "=== 1/2  (re)create $SIZE ext4 loopback in RAM and mount at $MNT ==="
mountpoint -q "$MNT" && umount "$MNT" || true
rm -f "$IMG"
truncate -s "$SIZE" "$IMG"
mkfs.ext4 -q -F "$IMG"
mkdir -p "$MNT"
mount -o loop "$IMG" "$MNT"
install -d -o "$U" -g "$U" "$MNT/lp-docker"
findmnt -no SOURCE,TARGET,FSTYPE "$MNT"

echo "=== 2/2  point rootless Docker ($U) at $MNT/lp-docker + rebuild image ==="
runuser -l "$U" -c "
  set -e
  export XDG_RUNTIME_DIR=/run/user/\$(id -u)
  mkdir -p ~/.config/docker
  printf '{ \"data-root\": \"$MNT/lp-docker\" }\n' > ~/.config/docker/daemon.json
  systemctl --user restart docker
  for i in \$(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 0.5; done
  echo \"docker: \$(docker info --format '{{.DockerRootDir}} driver={{.Driver}}')\"
  cd '$REPO' && bash scripts/build_agent_image.sh >/dev/null 2>&1 && echo 'image rebuilt'
  docker images --format '  {{.Repository}}:{{.Tag}} {{.Size}}' | grep -i lp-rl
"
echo "=== DONE: fast RAM Docker scratch ready (volatile — re-run after a restart) ==="
