#!/usr/bin/env bash
# System phase of rootless Docker setup. Must run as root, with no sudo:
#   wsl -d Ubuntu-24.04 -u root -- bash scripts/setup_rootless_docker_system.sh alex
# sudo prompts cannot be answered when driven non-interactively from the
# Windows side, which is why this phase is separate from the user phase.
set -euo pipefail

TARGET_USER="${1:?usage: setup_rootless_docker_system.sh <target-user>}"

if [ "$(id -u)" -ne 0 ]; then
  echo "This phase must run as root: wsl -d Ubuntu-24.04 -u root -- bash $0 ${TARGET_USER}" >&2
  exit 2
fi
if ! id -u "${TARGET_USER}" >/dev/null 2>&1; then
  echo "Target user ${TARGET_USER} does not exist." >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl gnupg uidmap dbus-user-session slirp4netns fuse-overlayfs

install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
fi

. /etc/os-release
ARCH="$(dpkg --print-architecture)"
echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras

# Rootless mode owns the daemon path for this project; the rootful daemon
# auto-starts on install and must not linger as an ambient fallback.
systemctl disable --now docker.service docker.socket || true

grep -q "^${TARGET_USER}:" /etc/subuid || echo "${TARGET_USER}:100000:65536" >> /etc/subuid
grep -q "^${TARGET_USER}:" /etc/subgid || echo "${TARGET_USER}:100000:65536" >> /etc/subgid

# Linger keeps the user's systemd manager (and the rootless daemon) alive
# across logouts, which long unattended training runs require.
loginctl enable-linger "${TARGET_USER}"

echo "system phase complete; now run scripts/setup_rootless_docker_user.sh as ${TARGET_USER}"
