from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def run(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def main() -> int:
    failures: list[str] = []
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    expected_host = f"unix:///run/user/{uid}/docker.sock"
    print(f"expected DOCKER_HOST={expected_host}")
    print(f"actual DOCKER_HOST={os.environ.get('DOCKER_HOST', '')}")

    for cmd in ("docker", "dockerd-rootless-setuptool.sh", "newuidmap", "newgidmap"):
        path = shutil.which(cmd)
        print(f"{cmd}: {path or 'missing'}")
        if path is None:
            failures.append(f"missing {cmd}")

    for file_name in ("/etc/subuid", "/etc/subgid"):
        text = Path(file_name).read_text(errors="replace") if Path(file_name).exists() else ""
        if f"{os.environ.get('USER', '')}:" not in text:
            failures.append(f"missing user range in {file_name}")

    proc = run(["docker", "info", "--format", "{{json .}}"])
    if proc.returncode != 0:
        failures.append(f"docker info failed: {proc.stderr.strip() or proc.stdout.strip()}")
    else:
        info = json.loads(proc.stdout)
        security = info.get("SecurityOptions", [])
        print("SecurityOptions:", security)
        if not any("rootless" in str(item).lower() for item in security):
            failures.append("docker daemon is not rootless")
        print("CgroupVersion:", info.get("CgroupVersion"))
        print("Driver:", info.get("Driver"))

    user_status = run(["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"], timeout=10)
    print(user_status.stdout.strip())
    if "Linger=yes" not in user_status.stdout:
        failures.append("loginctl linger is not enabled")

    if failures:
        print("rootless docker check failed:")
        for item in failures:
            print(f"- {item}")
        return 1
    print("rootless docker check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
