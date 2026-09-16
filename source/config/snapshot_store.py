"""Project-scoped source-file + change-log store.

Central module shared by ``source.gui.operant`` and ``source.gui.maze``.
Both main_windows instantiate ONE ``SnapshotStore`` per project; all
hashing, content-storage and audit logging lives here, no other module
touches the ``source/`` tree.

Identity
========

Each .py / .json source file is identified by its 8-char djb2 hex
string. For .py files (task / hw_def / api_class) the hash uses the
SAME 4-byte LE djb2 pyControl uses for its MCU upload hash, so the
value in ``setup_config.boxes[i].init_hw_def.djb2`` matches the
``hardware_def_hash`` line in the MCU TSV header. For JSON files
(action_config / stats_template / DLC manifest) we use byte-wise djb2
of the UTF-8 file content.

    <project>/source/<djb2>.py            task / hw_def / api_class
    <project>/source/<djb2>.json          action_config / stats_template
    <project>/source/configs/<djb2>.json  project_config snapshots
    <project>/source/dlc/<djb2>.json      DLC model manifest

Files are stored RAW, exact bytes. No wrapper, no normalization.

Two-tier capture
================

``capture_source(path, kind=..., box_id=...)``:

    Tier 1, stat + path cache              (~0.05 ms)
       mtime + size match the cached FileRef? → return cached, done.

    Tier 2, djb2 lookup                    (~0.5 ms)
       ``source/<djb2>.<ext>`` already exists on disk? → return ref.

    Otherwise, new content                 (~1 ms)
       Copy raw bytes to ``_pending/box<N>_<kind>.<ext>`` (transient).
       Overwrites any prior pending file for that (box, kind) slot.

Pending → committed
===================

Commit happens when a real recording starts:

    ``commit_box_sources(cfg, box_id, extra_refs={})``

For each ref on the box (init_hw_def / action_config / api_class), and
each ref passed in ``extra_refs`` (e.g. {"task": task_ref}):
    * If ``source/<djb2>.<ext>`` already exists → no-op.
    * Else if ``_pending/box<N>_<kind>.<ext>`` exists → atomic rename.

``sweep_orphan_pending()`` runs at project-load: deletes pending
files whose djb2 already exists in ``source/`` (stale duplicates).

DLC models
==========

DLC/SLEAP model trees are multi-file binary blobs. They get a separate
path:

    <project>/source/dlc/<djb2>.json

The manifest filename hash and the per-file hashes inside it are all
djb2. The manifest is JSON listing each significant model file's djb2;
stored once per unique model; cached at the
``cfg.tracking.dlc/sleap.model_path`` level by the caller so
``commit_box_sources`` reads it with no I/O.

Tracking toggle
===============

``cfg.meta.tracking_enabled`` drives the store's behaviour. When False:

    * ``capture_source()`` returns a stat-only FileRef; no pending write.
    * ``commit_box_sources()`` is a no-op.
    * A ``{"kind":"upload","tracking":"off",…}`` event still goes to the
      change_log so the gap is auditable.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from source.config import history as _history
from source.config.experiment import (
    FileRef,
    djb2_hex_from_file,
    djb2_hex_from_bytes,
    now_ts,
    source_dir,
    source_dlc_dir,
    source_hd_dir,
    source_devices_dir,
    relpath_for_storage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

PENDING_DIR_NAME       = "_pending"
INDEX_NAME             = "index.jsonl"

# Source kinds we know how to capture. Each maps to a file extension and
# (for box-scoped kinds) one slot per box in _pending/.
SOURCE_KINDS = ("task", "hw_def", "device", "action_config", "api_class", "stats_template")

_EXTENSIONS: Dict[str, str] = {
    "task":           ".py",
    "hw_def":         ".py",
    "device":         ".py",
    "api_class":      ".py",
    "action_config":  ".json",
    "stats_template": ".json",
}


# DLC snapshot files, covers TF + PyTorch + ONNX exports.
_DLC_PATTERNS = [
    "*.meta", "*.index", "*.data-*",
    "config.yaml", "pose_cfg.yaml",
    "*.pb", "*.pt", "*.pth", "*.onnx",
    # sleap-nn (PyTorch): checkpoint weights + training config; legacy SLEAP TF.
    "*.ckpt", "training_config.yaml", "training_config.json",
    "best_model.h5", "*.trt",
]


def _extension_for(kind: str) -> str:
    return _EXTENSIONS.get(kind, ".py")


def _walk_model_files(model_path: Path, prefix: str = "") -> List[Dict[str, Any]]:
    """Hash each significant file under a model path (djb2), rel-pathed and
    optionally ``prefix``-namespaced (used to fold a top-down centroid model
    into the same manifest without rel_path collisions)."""
    files: List[Dict[str, Any]] = []
    if model_path.is_dir():
        seen: set = set()
        for pat in _DLC_PATTERNS:
            for f in model_path.rglob(pat):
                if f.is_file() and f not in seen:
                    seen.add(f)
                    try:
                        size = f.stat().st_size
                    except OSError:
                        size = 0
                    try:
                        djb2 = djb2_hex_from_bytes(f.read_bytes())
                    except OSError:
                        djb2 = ""
                    rel = str(f.relative_to(model_path)).replace(os.sep, "/")
                    files.append({"rel_path": prefix + rel,
                                  "djb2": djb2, "size_bytes": size})
    elif model_path.is_file():
        try:
            djb2 = djb2_hex_from_bytes(model_path.read_bytes())
            size = model_path.stat().st_size
        except OSError:
            djb2, size = "", 0
        files.append({"rel_path": prefix + model_path.name,
                      "djb2": djb2, "size_bytes": size})
    return files


def _build_dlc_manifest(model_path: Path,
                        extra_paths: Optional[List[Path]] = None
                        ) -> Tuple[str, Dict[str, Any]]:
    """Walk a DLC/SLEAP model path, hash each significant file with djb2,
    build a deterministic manifest. Returns ``(canonical_json, manifest)``.

    ``extra_paths``: additional model trees folded into the same manifest
    (e.g. a SLEAP top-down **centroid** model, so a two-model pipeline is fully
    recorded). Each is namespaced ``__extra<i>__/`` to avoid rel_path
    collisions. No model binaries are stored, only their djb2 hashes.
    """
    files = _walk_model_files(model_path)
    for i, ep in enumerate(extra_paths or []):
        files += _walk_model_files(Path(ep), prefix=f"__extra{i}__/")
    files.sort(key=lambda x: x["rel_path"])
    manifest = {
        "schema": 1,
        "captured_at": now_ts(),
        # Recorded only (never read back to load the model). Store it
        # storage-relative so a shared project artifact doesn't leak an
        # absolute ``D:/...`` / ``C:/Users/<name>/...`` path.
        "model_path": relpath_for_storage(model_path),
        "files": files,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return canonical, manifest


# ---------------------------------------------------------------------------
# SnapshotStore
# ---------------------------------------------------------------------------


class SnapshotStore:
    """Project-scoped source + change-log store. One instance per project."""

    def __init__(self, project_dir: Path, *,
                 tracking_enabled: bool = True) -> None:
        self.project_dir = Path(project_dir)
        self.tracking_enabled = bool(tracking_enabled)
        # Tier 1 cache: resolved path str → (mtime, size, FileRef).
        self._stat_cache: Dict[str, Tuple[float, int, FileRef]] = {}
        # Tier 2 cache: djb2 → True (set means file exists in source/).
        self._known_djb2: set = set()
        self._sources_loaded = False
        # Re-entrant lock serialising commit_box_sources() / capture_dlc()
        # / capture_source() across parallel-start workers, so two workers
        # committing the same djb2 (boxes sharing an HD) don't race on
        # os.replace(pending → committed). Reentrant so helpers can re-acquire.
        import threading as _t
        self._lock = _t.RLock()

    # ---- properties ------------------------------------------------------

    @property
    def sources_dir(self) -> Path:
        return source_dir(self.project_dir)

    @property
    def pending_dir(self) -> Path:
        return self.project_dir / PENDING_DIR_NAME

    @property
    def dlc_dir(self) -> Path:
        return source_dlc_dir(self.project_dir)

    @property
    def hd_dir(self) -> Path:
        return source_hd_dir(self.project_dir)

    @property
    def devices_dir(self) -> Path:
        return source_devices_dir(self.project_dir)

    def _dir_for_kind(self, kind: str) -> Path:
        """Committed-snapshot directory for a source kind. Hardware
        definitions live in ``source/hd/``, device drivers in
        ``source/devices/``; everything else in ``source/``."""
        if kind == "hw_def":
            return self.hd_dir
        if kind == "device":
            return self.devices_dir
        return self.sources_dir

    @property
    def index_path(self) -> Path:
        return self.sources_dir / INDEX_NAME

    @property
    def change_log_path(self) -> Path:
        return _history.change_log_path(self.project_dir)

    # ---- tracking toggle -------------------------------------------------

    def set_tracking_enabled(self, on: bool) -> None:
        """Toggle source-tracking capture; appends a ``tracking_toggle`` change entry on transition."""
        prev = self.tracking_enabled
        self.tracking_enabled = bool(on)
        if prev != self.tracking_enabled:
            self.append_change({
                "kind": "tracking_toggle",
                "from": prev,
                "to":   self.tracking_enabled,
            })

    # ---- capture --------------------------------------------------------

    def capture_source(self,
                       source_path: str | Path,
                       *,
                       kind: str,
                       setup_id: int,
                       label: str = "") -> Optional[FileRef]:
        """Hash + stage a just-uploaded source file.

        Two-tier capture (stat cache, then on-disk djb2 lookup). New
        content is COPIED to ``_pending/box<N>_<kind>.<ext>``, replacing
        any prior pending file for the same (box, kind) slot.

        Returns the FileRef (djb2 populated when the file exists).
        Returns None when the source path doesn't exist.
        """
        path = Path(source_path)
        # Single stat() that doubles as the existence check, is_file()
        # then stat() would be two syscalls on the same path.
        try:
            stat = path.stat()
        except (FileNotFoundError, OSError) as e:
            logger.warning("capture_source: stat failed (%s): %s", path, e)
            return None
        kind = str(kind)
        if kind not in SOURCE_KINDS:
            logger.warning("capture_source: unknown kind %r (allowed: %s)",
                           kind, SOURCE_KINDS)
        label = label or path.name

        # Tracking OFF, stat-only FileRef + audit line, no pending write.
        if not self.tracking_enabled:
            ref = FileRef.from_path(path)
            self.append_change({
                "kind":     "upload",
                "source":   kind,
                "box_id":   setup_id,
                "label":    label,
                "tracking": "off",
            })
            return ref

        # TIER 1, stat + path cache (stat already resolved above).
        path_key = str(path.resolve())
        cached = self._stat_cache.get(path_key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            ref = cached[2]
            self.append_change({
                "kind":   "upload",
                "source": kind,
                "box_id": setup_id,
                "label":  label,
                "djb2":   ref.djb2,
                "tier":   1,
                "new":    False,
            })
            return ref

        # Compute djb2 of the file content.
        ref = FileRef.from_path(path)
        self._stat_cache[path_key] = (stat.st_mtime, stat.st_size, ref)

        # TIER 2, already committed?
        if ref.djb2 and self._has_committed_source(ref.djb2, kind):
            self.append_change({
                "kind":   "upload",
                "source": kind,
                "box_id": setup_id,
                "label":  label,
                "djb2":   ref.djb2,
                "tier":   2,
                "new":    False,
            })
            return ref

        # New content, copy raw bytes into _pending/box<N>_<kind>.<ext>.
        try:
            self.pending_dir.mkdir(parents=True, exist_ok=True)
            pending_path = self._pending_path(setup_id, kind)
            shutil.copyfile(path, pending_path)
        except OSError as e:
            logger.error("capture_source pending write failed (%s): %s",
                         path, e)
            return ref

        self.append_change({
            "kind":   "upload",
            "source": kind,
            "box_id": setup_id,
            "label":  label,
            "djb2":   ref.djb2,
            "tier":   "new",
            "new":    True,
        })
        return ref

    def commit_box_sources(self,
                           cfg,
                           setup_id: int,
                           *,
                           extra_refs: Optional[Dict[str, FileRef]] = None,
                           device_refs: Optional[List[FileRef]] = None
                           ) -> List[str]:
        """Promote pending source files for one box to ``source/``.

        For each captured kind on this box (init_hw_def, action_config,
        api_class) plus any kinds passed in ``extra_refs`` (typically
        ``{"task": task_ref}`` since task is run-only on the cfg side):

            * No-op if ``source/<djb2>.<ext>`` already exists.
            * Otherwise rename ``_pending/box<N>_<kind>.<ext>`` →
              source/<djb2>.<ext>.
            * Append ``source/index.jsonl`` + change_log "commit" line.

        ``device_refs`` is a LIST (one HD uses many drivers), each committed
        under kind ``"device"`` by copying from the original device file,
        records the exact driver code each run used.

        Returns the list of djb2 hex strings now resident in source/ for
        this box.
        """
        if not self.tracking_enabled:
            return []
        # Serialise the commit so parallel-start workers don't race on the
        # same djb2's pending→committed rename when N boxes share one HD.
        return self._commit_box_sources_locked(cfg, setup_id, extra_refs,
                                               device_refs)

    def _commit_box_sources_locked(self,
                                   cfg,
                                   setup_id: int,
                                   extra_refs,
                                   device_refs=None):
        committed: List[str] = []
        with self._lock:
            # Collect refs to commit: from the box (persistent) and from extras (run-time).
            box = self._find_box(cfg, setup_id)
            all_refs: Dict[str, Optional[FileRef]] = {
                "hw_def":        (box.init_hw_def if box is not None else None),
                "action_config": (box.action_config if box is not None else None),
                "api_class":     (box.api_class if box is not None else None),
            }
            if extra_refs:
                all_refs.update(extra_refs)

            for kind, ref in all_refs.items():
                if ref is None or not ref.djb2:
                    continue
                ext = _extension_for(kind)
                djb2 = ref.djb2
                committed.append(djb2)
                committed_path = self._dir_for_kind(kind) / f"{djb2}{ext}"
                # Used by both the already-committed cleanup and the promote branch.
                pending_path = self._pending_path(setup_id, kind)

                # Already committed, clean any stale pending duplicate.
                if committed_path.exists():
                    self._known_djb2.add(djb2)
                    if pending_path.exists():
                        try:
                            pending_path.unlink()
                        except OSError as e:
                            logger.debug("pending cleanup failed (%s): %s",
                                         pending_path, e)
                    continue

                try:
                    committed_path.parent.mkdir(parents=True, exist_ok=True)
                    if not pending_path.exists():
                        # Fall back to copying from the original path.
                        src = Path(ref.path)
                        if not src.is_file():
                            logger.warning(
                                "commit_box_sources(box=%s,kind=%s): pending "
                                "missing and original gone (%s)", setup_id, kind, src)
                            continue
                        shutil.copyfile(src, committed_path)
                    else:
                        os.replace(pending_path, committed_path)
                except OSError as e:
                    logger.error("commit_box_sources(box=%s,kind=%s) failed: %s",
                                 setup_id, kind, e)
                    continue

                self._record_commit(djb2, kind, ref, box, setup_id)

            # Device driver files, a list (one HD uses many), each content-
            # addressed into source/devices/ by copying from the original file.
            for ref in (device_refs or []):
                if ref is None or not ref.djb2:
                    continue
                djb2 = ref.djb2
                committed.append(djb2)
                committed_path = self.devices_dir / f"{djb2}.py"
                if committed_path.exists():
                    self._known_djb2.add(djb2)
                    continue
                try:
                    committed_path.parent.mkdir(parents=True, exist_ok=True)
                    src = Path(ref.path)
                    if not src.is_file():
                        logger.warning(
                            "commit_box_sources(box=%s,kind=device): original "
                            "gone (%s)", setup_id, src)
                        continue
                    shutil.copyfile(src, committed_path)
                except OSError as e:
                    logger.error("commit_box_sources(box=%s,kind=device) "
                                 "failed: %s", setup_id, e)
                    continue
                self._record_commit(djb2, "device", ref, box, setup_id)
        return committed

    def _record_commit(self, djb2: str, kind: str, ref, box, setup_id) -> None:
        """Index + change-log one committed artifact (shared by the primary
        and device-driver commit loops)."""
        self._known_djb2.add(djb2)
        self._append_index({
            "djb2":        djb2,
            "kind":        kind,
            "name":        ref.name,
            "captured_at": now_ts(),
            # Relative, never persist an absolute D:/ or C:/Users path.
            "from":        relpath_for_storage(ref.path, self.project_dir),
            "bytes":       ref.size_bytes,
            "box_number":  getattr(box, "setup_number", 0) if box else 0,
        })
        self.append_change({
            "kind":   "commit",
            "source": kind,
            "box_id": setup_id,
            "djb2":   djb2,
            "name":   ref.name,
        })

    def capture_dlc(self, model_path: str | Path,
                    centroid_model_path: str | Path | None = None) -> Optional[str]:
        """Walk + hash a DLC/SLEAP model directory, write the manifest as
        ``source/dlc/<djb2>.json``. Returns the manifest's djb2 hex
        (used as the model identity in the committed source refs).

        ``centroid_model_path``: the paired centroid model for a SLEAP
        top-down pipeline; folded into the same manifest so both stages are
        recorded. Idempotent. Returns None when tracking is off or the path
        is missing.
        """
        if not self.tracking_enabled:
            self.append_change({"kind": "dlc_capture", "tracking": "off"})
            return None
        path = Path(model_path)
        if not path.exists():
            logger.warning("capture_dlc: model path missing: %s", path)
            return None
        extra = ([Path(centroid_model_path)]
                 if centroid_model_path and Path(centroid_model_path).exists()
                 else None)
        canonical, manifest = _build_dlc_manifest(path, extra_paths=extra)
        djb2 = djb2_hex_from_bytes(canonical.encode("utf-8"))
        manifest_path = self.dlc_dir / f"{djb2}.json"
        try:
            self.dlc_dir.mkdir(parents=True, exist_ok=True)
            if not manifest_path.exists():
                tmp = manifest_path.with_suffix(".json.tmp")
                # Stamp the manifest itself with its own djb2 for self-verification.
                manifest_with_id = {"djb2": djb2, **manifest}
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(manifest_with_id, fh, indent=2, ensure_ascii=False)
                os.replace(tmp, manifest_path)
                self.append_change({
                    "kind": "dlc_capture",
                    "djb2": djb2,
                    "new":  True,
                })
            else:
                self.append_change({
                    "kind": "dlc_capture",
                    "djb2": djb2,
                    "new":  False,
                })
        except OSError as e:
            logger.error("capture_dlc manifest write failed: %s", e)
            return None
        return djb2

    def sweep_orphan_pending(self) -> int:
        """Project-load housekeeping: remove ``_pending/`` files whose
        djb2 already lives in ``source/``. Returns the count removed.
        """
        if not self.pending_dir.is_dir():
            return 0
        removed = 0
        for pending_path in list(self.pending_dir.glob("box*_*")):
            try:
                # The pending file's content djb2 is what would resolve
                # in source/. We can't easily know its kind from filename
                # alone, so fall back to checking against ANY committed
                # source extension.
                djb2 = djb2_hex_from_file(pending_path)
            except OSError:
                continue
            if self._has_any_committed_source(djb2):
                try:
                    pending_path.unlink()
                    removed += 1
                except OSError as e:
                    logger.debug("sweep_orphan_pending unlink failed (%s): %s",
                                 pending_path, e)
        return removed

    def resolve(self, djb2: str, *, kind: str = "") -> Optional[str]:
        """Return the path to ``source/<djb2>.<ext>`` if present, else None.

        When ``kind`` is supplied, looks up the corresponding extension.
        Otherwise tries the known extensions (.py first, then .json).
        """
        if kind:
            ext = _extension_for(kind)
            p = self._dir_for_kind(kind) / f"{djb2}{ext}"
            return str(p) if p.is_file() else None
        # Kind unknown, probe source/ (.py/.json) then source/hd/ + devices/.
        for ext in (".py", ".json"):
            p = self.sources_dir / f"{djb2}{ext}"
            if p.is_file():
                return str(p)
        for d in (self.hd_dir, self.devices_dir):
            p = d / f"{djb2}.py"
            if p.is_file():
                return str(p)
        return None

    def append_change(self, event: dict) -> None:
        """Append one source-capture event to ``<project>/change_log.jsonl``.

        Delegates to the shared writer in ``source.config.history`` so the
        change log has one owner (config edits + source captures, one
        timeline, one ts format)."""
        try:
            _history.append_event(self.project_dir, event)
        except OSError as e:
            logger.error("change_log append failed: %s", e)

    # ---- internals ------------------------------------------------------

    def _pending_path(self, setup_id: int, kind: str) -> Path:
        return self.pending_dir / f"box{int(setup_id)}_{kind}{_extension_for(kind)}"

    def _has_committed_source(self, djb2: str, kind: str = "") -> bool:
        self._load_sources_index()
        if djb2 in self._known_djb2:
            return True
        if kind:
            return (self._dir_for_kind(kind) / f"{djb2}{_extension_for(kind)}").is_file()
        return self._has_any_committed_source(djb2)

    def _has_any_committed_source(self, djb2: str) -> bool:
        for ext in (".py", ".json"):
            if (self.sources_dir / f"{djb2}{ext}").is_file():
                return True
        return ((self.hd_dir / f"{djb2}.py").is_file()
                or (self.devices_dir / f"{djb2}.py").is_file())

    def _load_sources_index(self) -> None:
        if self._sources_loaded:
            return
        self._sources_loaded = True
        # Scan source/ (tasks / api_class / configs), source/hd/ (HD
        # snapshots) AND source/devices/ (drivers) so committed-djb2
        # lookups hit every tree.
        for d in (self.sources_dir, self.hd_dir, self.devices_dir):
            if not d.is_dir():
                continue
            try:
                for p in d.glob("*"):
                    if p.is_file():
                        self._known_djb2.add(p.stem)
            except OSError as e:
                logger.debug("source scan failed (%s): %s", d, e)

    def _append_index(self, entry: dict) -> None:
        try:
            self.sources_dir.mkdir(parents=True, exist_ok=True)
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=False,
                                    ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("source index append failed: %s", e)

    @staticmethod
    def _find_box(cfg, setup_id: int):
        """Walk cfg.setup_config.boxes for setup_number == setup_id."""
        if cfg is None:
            return None
        setup = getattr(cfg, "setup_config", None)
        if setup is None or not getattr(setup, "boxes", None):
            return None
        for b in setup.boxes:
            if int(getattr(b, "setup_number", 0)) == int(setup_id):
                return b
        return None
