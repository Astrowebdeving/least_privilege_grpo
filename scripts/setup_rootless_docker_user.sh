#!/usr/bin/env bash
# User phase of rootless Docker setup. Run as the normal WSL user after
# setup_rootless_docker_system.sh has completed:
#   wsl -d Ubuntu-24.04 -- bash -lc 'cd ~/projects/least_privilege_rl && scripts/setup_rootless_docker_user.sh'
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this phase as the normal WSL user, not root." >&2
  exit 2
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ ! -d "${XDG_RUNTIME_DIR}" ]; then
  echo "No systemd user runtime dir at ${XDG_RUNTIME_DIR}; is systemd enabled and the user session running?" >&2
  exit 2
fi

dockerd-rootless-setuptool.sh install

systemctl --user enable docker.service
systemctl --user restart docker.service

DOCKER_HOST_LINE="export DOCKER_HOST=unix://${XDG_RUNTIME_DIR}/docker.sock"
if ! grep -qF "${DOCKER_HOST_LINE}" "${HOME}/.bashrc" 2>/dev/null; then
  printf '\n# least_privilege_rl: rootless Docker endpoint\n%s\n' "${DOCKER_HOST_LINE}" >> "${HOME}/.bashrc"
  echo "Appended DOCKER_HOST export to ~/.bashrc"
fi

export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"
docker info
echo "user phase complete; DOCKER_HOST=${DOCKER_HOST}"
