# WSL cgroup v2 and Rootless Docker

Last verified: 2026-06-18 on Ubuntu-24.04.

The current machine already exposes cgroup v2 through systemd, and the
rootless Docker daemon successfully enforces the resource limits used by this
project. No `%UserProfile%\.wslconfig` change was required.

Current verified state:

- rootless Docker runs under the user systemd manager with linger enabled;
- Docker uses cgroup v2 with the systemd driver;
- memory, CPU, and PID limits bind in smoke tests;
- `cpuset` and `io` are not delegated, which is acceptable because the current
  sandbox does not request those controllers;
- rootful Docker/containerd services are disabled and stopped.

## Verification

Run from the canonical project:

```bash
cd <repo>
.venv/bin/python scripts/check_rootless_docker.py
docker info --format '{{.CgroupVersion}}'
docker info --format '{{json .SecurityOptions}}'
docker run --rm --memory=64m --pids-limit=32 --cpus=1 ubuntu:24.04 true
```

Do not assume limits work merely because Docker accepts the flags. Trust the
project smoke check and an actual constrained container.

## Only if cgroup v2 verification fails

On another WSL installation, first confirm systemd is enabled and rerun the
project's split setup:

```powershell
wsl -d Ubuntu-24.04 -u root -- bash <repo>/scripts/setup_rootless_docker_system.sh alex
wsl -d Ubuntu-24.04 -- bash -lc 'cd <repo> && scripts/setup_rootless_docker_user.sh'
```

If WSL still exposes a mixed or v1 hierarchy, the fallback is:

```ini
[wsl2]
kernelCommandLine = cgroup_no_v1=all
```

in `%UserProfile%\.wslconfig`, followed by:

```powershell
wsl --shutdown
```

This is a troubleshooting fallback, not a required current-machine step.
`wsl --shutdown` terminates all running trainers, vLLM services, containers,
and other distro processes, so schedule it deliberately.

