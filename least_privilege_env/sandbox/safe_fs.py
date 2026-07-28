from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil


class SafeFSError(ValueError):
    pass


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_inside(root: Path, cwd: Path, user_path: str) -> Path:
    root = root.resolve()
    user = Path(user_path)
    if user.is_absolute():
        parts = user.parts[1:]
        # `/workspace` is the model-visible mount root. Treat it as an alias
        # for the host-side episode root instead of resolving it to
        # `<root>/workspace`.
        if parts and parts[0] == "workspace":
            parts = parts[1:]
        raw_candidate = root.joinpath(*parts)
    else:
        raw_candidate = cwd / user
    current = root
    try:
        rel_parts = raw_candidate.relative_to(root).parts
    except ValueError:
        rel_parts = raw_candidate.parts[1:] if raw_candidate.is_absolute() else raw_candidate.relative_to(root).parts
    current = root
    for part in rel_parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise SafeFSError(f"path escapes workspace: {user_path}")
        current = current / part
        if current.exists() and Path(current).is_symlink():
            raise SafeFSError(f"symlink traversal rejected: {user_path}")
    candidate = current
    if not _is_relative_to(candidate.resolve(strict=False), root):
        raise SafeFSError(f"path escapes workspace: {user_path}")
    return candidate


def lstat_no_follow(path: Path) -> os.stat_result:
    return os.lstat(path)


def write_file_no_follow(path: Path, data: bytes, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise SafeFSError(f"refusing to write through symlink: {path}")
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if append else os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "ab" if append else "wb") as handle:
        handle.write(data)


def read_file_no_follow(path: Path) -> bytes:
    if path.is_symlink():
        raise SafeFSError(f"refusing to read symlink: {path}")
    return path.read_bytes()


def copy_tree_no_follow(src: Path, dst: Path) -> None:
    src = src.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for root, dirs, files in os.walk(src, topdown=True, followlinks=False):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        out_root = dst / rel
        for dirname in list(dirs):
            item = root_path / dirname
            if item.is_symlink():
                dirs.remove(dirname)
                raise SafeFSError(f"symlink in fixture rejected: {item}")
            (out_root / dirname).mkdir(exist_ok=True)
        for filename in files:
            item = root_path / filename
            if item.is_symlink():
                raise SafeFSError(f"symlink in fixture rejected: {item}")
            shutil.copy2(item, out_root / filename)


@dataclass(frozen=True)
class FileState:
    kind: str
    digest: str


def snapshot_tree(root: Path) -> dict[str, FileState]:
    root = root.resolve()
    state: dict[str, FileState] = {}
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for dirname in list(dirs):
            path = current_path / dirname
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                state[rel] = FileState(kind="symlink", digest="")
                dirs.remove(dirname)
        for filename in files:
            path = current_path / filename
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                state[rel] = FileState(kind="symlink", digest="")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            state[rel] = FileState(kind="file", digest=digest)
    return state


def diff_snapshots(before: dict[str, FileState], after: dict[str, FileState]) -> tuple[set[str], set[str], set[str]]:
    before_keys = set(before)
    after_keys = set(after)
    added = after_keys - before_keys
    deleted = before_keys - after_keys
    changed = {key for key in before_keys & after_keys if before[key] != after[key]}
    return added, changed, deleted
