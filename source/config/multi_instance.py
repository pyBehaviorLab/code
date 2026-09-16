"""Multi-instance safety: atomic writes, cross-process file locks, and
optimistic-concurrency project saves with automatic three-way merge.

One module, by intent: when two pyOperant GUIs save the same project at
the same time and something goes wrong, the developer opens THIS file
(not five).

Public surface:
    write_atomic(path, payload), tmp + os.replace
    file_lock(path, timeout), cross-process exclusive lock
    three_way_merge(base, local, remote), auto-reconcile dict trees
    ProjectFileGuard.stamp(path), snapshot at load
    ProjectFileGuard.save_json(obj, on_conflict=...), OCC + auto-merge save

The OCC + auto-merge path:
    1. At load: ``guard = ProjectFileGuard.stamp(experiment_config_path)``.
    2. At save: ``guard.save_json(new_obj, on_conflict=callback)``.
       * If on-disk file is unchanged since load -> write.
       * If on-disk file changed but changes are DISJOINT from local
         edits -> three-way merge silently, write the merged result.
       * If on-disk file changed AND any field conflicts -> call
         ``on_conflict(conflicts)`` which returns "overwrite" / "reload"
         / "cancel".
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# Import the hash helper from the leaf `hashing` module, NOT `experiment`
# (which imports back into this module), breaks the experiment<->multi_instance
# import cycle. `experiment` only re-exports this symbol anyway.
from source.config.hashing import djb2_hex_from_bytes


# ============================================================================
# Atomic write
# ============================================================================


def write_atomic(path: str | Path, payload: bytes | str,
                 encoding: str = "utf-8") -> None:
    """Write a file via tmp + os.replace so readers never see partial content.

    Used everywhere we touch project state. POSIX guarantees os.replace is
    atomic for files on the same filesystem; Windows guarantees the same
    for MoveFileEx-style replace (which os.replace dispatches to).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if isinstance(payload, (bytes, bytearray)):
        with open(tmp, "wb") as fh:
            fh.write(payload)
    else:
        with open(tmp, "w", encoding=encoding, newline="\n") as fh:
            fh.write(payload)
    os.replace(tmp, p)


# ============================================================================
# Cross-process file lock
# ============================================================================


class FileLockTimeout(TimeoutError):
    """Raised when file_lock can't acquire within the requested timeout."""


@contextmanager
def file_lock(lock_path: str | Path, timeout: float = 5.0,
              poll: float = 0.05) -> Iterator[None]:
    """Acquire an exclusive cross-process lock.

    On Windows uses msvcrt.locking; on POSIX uses fcntl.flock. Both
    auto-release on process death, so a crashed GUI never strands the
    lock for the rest.

    Raises FileLockTimeout if the lock can't be acquired within
    ``timeout`` seconds.
    """
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fh = open(p, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise FileLockTimeout(f"timeout acquiring lock {p}")
                    time.sleep(poll)
            try:
                yield
            finally:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise FileLockTimeout(f"timeout acquiring lock {p}")
                    time.sleep(poll)
            try:
                yield
            finally:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        fh.close()


# ============================================================================
# Three-way merge, auto-reconcile two GUIs saving the same project
# ============================================================================


@dataclass
class MergeConflict:
    """A path in the config tree where local and remote both diverged
    from the common-ancestor value in incompatible ways. Only produced
    when auto-merge can't resolve."""
    path: list[str]
    base: Any
    local: Any
    remote: Any


_MISSING = object()  # marker: key absent from a side
_DROP = object()     # marker: drop the key from the merged result entirely


def _strip_paths(obj: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    """Return a deep-ish copy of ``obj`` with each ``paths[i]`` removed.

    Only dicts are descended; non-dict intermediates abort that path
    silently. Top-level dicts are shallow-copied; nested dicts on a
    strip path are copied as needed so the original isn't mutated.
    """
    if not paths or not isinstance(obj, dict):
        return obj
    # Group paths by their first key so we only deep-copy what we touch.
    head_groups: dict[str, list[tuple[str, ...]]] = {}
    direct_removes: set[str] = set()
    for p in paths:
        if not p:
            continue
        if len(p) == 1:
            direct_removes.add(p[0])
        else:
            head_groups.setdefault(p[0], []).append(p[1:])
    out = dict(obj)
    for k in direct_removes:
        out.pop(k, None)
    for head, sub_paths in head_groups.items():
        if head in out and isinstance(out[head], dict):
            out[head] = _strip_paths(out[head], tuple(sub_paths))
    return out


def _copy_paths_in_place(dst: dict, src: dict,
                         paths: tuple[tuple[str, ...], ...]) -> None:
    """For each path, copy ``src[path]`` into ``dst[path]`` (creating
    intermediate dicts as needed). Missing src values are skipped, not
    erased from dst, caller should pre-strip if that's required."""
    for p in paths:
        if not p:
            continue
        src_cur: Any = src
        for k in p:
            if not isinstance(src_cur, dict) or k not in src_cur:
                src_cur = _MISSING
                break
            src_cur = src_cur[k]
        if src_cur is _MISSING:
            continue
        dst_cur = dst
        for k in p[:-1]:
            if k not in dst_cur or not isinstance(dst_cur[k], dict):
                dst_cur[k] = {}
            dst_cur = dst_cur[k]
        dst_cur[p[-1]] = src_cur


def three_way_merge(base: Any, local: Any, remote: Any,
                    path: Optional[list[str]] = None
                    ) -> tuple[Any, list[MergeConflict]]:
    """Merge ``local`` and ``remote`` against common ancestor ``base``.

    Returns (merged, conflicts).
      * conflicts == [] -> safe to write merged.
      * conflicts != [] -> caller must resolve (e.g. user prompt).

    Strategy:
      * Dicts -> recurse key-by-key. Keys present in only one side flow
        through unchanged.
      * Anything else (scalars, lists, tuples, sets) -> treated as atomic
        leaves. List merging by element is intentionally not attempted
        because (a) re-ordering is indistinguishable from edits in JSON,
        and (b) atomic lists give a deterministic, predictable answer:
          - both sides unchanged             -> base
          - only one side changed            -> that side
          - both sides made the SAME change  -> the (matching) new value
          - both sides made DIFFERENT change -> conflict
    """
    path = path or []
    if isinstance(base, dict) and isinstance(local, dict) and isinstance(remote, dict):
        merged: dict = {}
        conflicts: list[MergeConflict] = []
        for key in set(base) | set(local) | set(remote):
            b = base.get(key, _MISSING)
            l_ = local.get(key, _MISSING)
            r = remote.get(key, _MISSING)
            if isinstance(b, dict) and isinstance(l_, dict) and isinstance(r, dict):
                sub_merged, sub_conf = three_way_merge(b, l_, r, path + [key])
                merged[key] = sub_merged
                conflicts.extend(sub_conf)
                continue
            value, conflict = _merge_leaf(b, l_, r, path + [key])
            if value is not _DROP:
                merged[key] = value
            if conflict is not None:
                conflicts.append(conflict)
        return merged, conflicts

    value, conflict = _merge_leaf(base, local, remote, path)
    return (value if value is not _DROP else None,
            [conflict] if conflict else [])


def _merge_leaf(base: Any, local: Any, remote: Any,
                path: list[str]) -> tuple[Any, Optional[MergeConflict]]:
    local_changed = (local is not _MISSING) and (local != base)
    remote_changed = (remote is not _MISSING) and (remote != base)
    local_present = local is not _MISSING
    remote_present = remote is not _MISSING

    if not local_changed and not remote_changed:
        if base is _MISSING:
            return _DROP, None
        return base, None
    if local_changed and not remote_changed:
        return (local if local_present else _DROP), None
    if remote_changed and not local_changed:
        return (remote if remote_present else _DROP), None
    if local == remote:
        return (local if local_present else _DROP), None
    return base, MergeConflict(path=path, base=base, local=local, remote=remote)


# ============================================================================
# ProjectFileGuard, OCC for experiment_config.json
# ============================================================================


class SaveConflictError(RuntimeError):
    """Raised when guard.save_json detects a conflict and on_conflict
    declined to resolve it (returned 'cancel' / 'reload' / None)."""

    def __init__(self, conflicts: list[MergeConflict], decision: str = "cancel"):
        self.conflicts = conflicts
        self.decision = decision
        super().__init__(
            f"Project save aborted ({decision}): "
            f"{len(conflicts)} conflict(s)"
        )


@dataclass
class ProjectFileGuard:
    """Optimistic concurrency control for a single JSON project file.

    Snapshot at load (mtime, size, djb2, raw payload), then on save:
    re-stat the file. If unchanged, write. If changed, attempt
    three-way merge. If merge has conflicts, call on_conflict callback;
    if no callback or callback says no, raise SaveConflictError.

    Auto-refreshes its snapshot after a successful write, the same
    guard can be reused for the next save.
    """
    path: Path
    loaded_mtime: float
    loaded_size: int
    loaded_djb2: str
    loaded_payload: bytes

    @classmethod
    def stamp(cls, path: str | Path) -> "ProjectFileGuard":
        """Snapshot ``path`` for later OCC. The file must exist."""
        p = Path(path)
        return cls.stamp_from_bytes(p, p.read_bytes())

    @classmethod
    def stamp_from_bytes(cls,
                         path: str | Path,
                         payload: bytes) -> "ProjectFileGuard":
        """Snapshot ``path`` using already-read ``payload`` bytes.

        Saves a redundant file read when the caller has just loaded the
        same file."""
        p = Path(path)
        st = p.stat()
        return cls(
            path=p,
            loaded_mtime=st.st_mtime,
            loaded_size=st.st_size,
            loaded_djb2=djb2_hex_from_bytes(payload),
            loaded_payload=payload,
        )

    def is_unchanged_on_disk(self) -> bool:
        """Fast path: mtime + size. Slow path: djb2 only when fast
        path differs (handles SMB/cloud-share mtime jitter)."""
        if not self.path.exists():
            return False
        st = self.path.stat()
        if st.st_mtime == self.loaded_mtime and st.st_size == self.loaded_size:
            return True
        try:
            live = djb2_hex_from_bytes(self.path.read_bytes())
        except OSError:
            return False
        return live == self.loaded_djb2

    def save_json(self,
                  new_obj: dict,
                  on_conflict: Optional[Callable[[list[MergeConflict]], str]] = None,
                  indent: int = 2,
                  volatile_paths: tuple[tuple[str, ...], ...] = (),
                  ) -> tuple[bool, list[MergeConflict]]:
        """Save ``new_obj`` with OCC + auto-merge.

        Returns (written, conflicts):
            (True, [])    - written (fast path or silent merge)
            (True, [...]) - written after on_conflict said "overwrite"
            (False, ...)  - aborted (callback said "reload"/"cancel"
                            or none provided when conflicts exist)

        Auto-merge wins whenever the local edits and the on-disk edits
        touch disjoint fields. Only when both sides changed the SAME
        field with different values does on_conflict get called.

        ``volatile_paths`` lists dotted-key paths whose values are
        EXCLUDED from conflict detection and ALWAYS taken from
        ``new_obj`` in the merged result. Use for auto-managed
        timestamps and hashes that every save updates, e.g.
        ``(('last_modified',), ('meta', 'modified_at'))``.
        """
        if self.is_unchanged_on_disk():
            return self._write_and_refresh(new_obj, indent=indent), []

        try:
            base = json.loads(self.loaded_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            base = {}
        try:
            remote_bytes = self.path.read_bytes()
            remote = json.loads(remote_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            remote = {}

        # Strip volatile keys so they don't generate phantom conflicts.
        base_m = _strip_paths(base, volatile_paths)
        local_m = _strip_paths(new_obj, volatile_paths)
        remote_m = _strip_paths(remote, volatile_paths)

        merged, conflicts = three_way_merge(base_m, local_m, remote_m)
        # Restore volatile fields from new_obj, this save's values win.
        _copy_paths_in_place(merged, new_obj, volatile_paths)

        if not conflicts:
            return self._write_and_refresh(merged, indent=indent), []

        if on_conflict is None:
            return False, conflicts

        decision = on_conflict(conflicts)
        if decision == "overwrite":
            return self._write_and_refresh(new_obj, indent=indent), conflicts
        return False, conflicts

    def _write_and_refresh(self, obj: dict, indent: int) -> bool:
        payload = json.dumps(obj, indent=indent, ensure_ascii=False).encode("utf-8")
        write_atomic(self.path, payload)
        self.loaded_payload = payload
        st = self.path.stat()
        self.loaded_mtime = st.st_mtime
        self.loaded_size = st.st_size
        self.loaded_djb2 = djb2_hex_from_bytes(payload)
        return True


__all__ = [
    "write_atomic",
    "file_lock",
    "FileLockTimeout",
    "three_way_merge",
    "MergeConflict",
    "ProjectFileGuard",
    "SaveConflictError",
]
