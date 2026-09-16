"""Project save/load + per-day history journal + template flow.

One module owns the entire feature so it can be debugged in one file.

Load flow (single file picker, dispatch by filename):

    Click "Load Config"
       │
       ▼
    QFileDialog.getOpenFileName
      filter:  "Project or Template
                (experiment_config.json template.json);;
                JSON Files (*.json);;All Files (*)"
      default dir: experiments/projects/   (absolute, resolved)
       │
       ▼
    Picked file's name:
        experiment_config.json  ──▶  load as active project
        template.json           ──▶  prompt for new name
                                     ──▶  new_from_template
        anything else           ──▶  warn + cancel

Save flow:
    First save:  QInputDialog name prompt -> save under that name.
                 On success an info popup shows the FULL ABSOLUTE PATH
                 so the user can see exactly where the file landed.
    Subsequent:  silent overwrite of the active project's
                 experiment_config.json + template.json.

Save invariants:
    * Allowed even with all-empty boxes (e.g. box exists but no camera,
      no COM port, no MCU connected). The schema serialises the empty
      strings verbatim. The only hard refusal is **zero boxes** (caller
      gates that, with a "No boxes to save" warning).
    * Always writes BOTH ``experiment_config.json`` and ``template.json``
      atomically (tmp + os.replace). The template is the structural slice
      used by "New From Template" later.
    * On every error the user sees a critical popup with the underlying
      exception text. No silent failures.

History journal:
    * ``history/<YYYY-MM-DD>.json`` rollup, one row per Record click.
    * ``day_start_snapshot`` pinned at the first Record of each day.
    * ``mid_day_changes`` filtered from change_log.jsonl for Apply events.
    * Dry-run gate: empty ``subject_id`` -> no history write.

Public API:
    save_project(mw, cfg)                                  -> Optional[Path]
    load_project(mw, mode)                                 -> Optional[(Config, Path)]
    new_from_template(mw, template_path, name, mode)       -> Optional[(Config, Path)]
    open_run(project_dir_path, cfg, box_id, subject_id, *, pycboard) -> Optional[str]
    close_run(project_dir_path, run_id, *, status, files)            -> None
    scan_unfinished_runs(project_dir_path)                -> List[(str, dict)]
    mark_run_crashed(project_dir_path, run_id)            -> None
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtWidgets

from source.config.experiment import (
    Config,
    EXPERIMENT_CONFIG_FILENAME,
    experiment_config_path,
    projects_root,
    save_experiment,
)
from source.config.multi_instance import MergeConflict, SaveConflictError, write_atomic
from source.config import history as _hist
from source.log import get_logger

logger = get_logger()


class _NullCtx:
    """No-op context manager, used when the host hasn't set up a save lock
    yet (e.g. tests call _save_to_disk without initializing autosave)."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# Per-day run rollup layout, mirroring the data tree (get_session_dirs):
#   <project>/runs/<task>/<date>.json   when a task family is known
#   <project>/runs/<date>.json          loose runs (no task subfolder)
# No "_loose" placeholder folder, when there's no family the file sits
# directly under runs/, exactly like data/<project>/<date>/ omits the
# task level.
RUNS_DIR_NAME = "runs"
TEMPLATE_FILENAME = "template.json"


# ============================================================================
# Filesystem helpers
# ============================================================================


def runs_dir(project_dir_path: str | Path) -> Path:
    """``<project>/runs/``, root of the per-family per-date run rollups."""
    return Path(project_dir_path) / RUNS_DIR_NAME


def _resolve_family(task_family: Optional[str]) -> str:
    """Normalise a task_family arg: stripped name, or ``""`` when none.
    ``""`` means "no task subfolder": the run file sits directly under
    ``runs/`` (no ``_loose`` placeholder)."""
    return (task_family or "").strip()


def _runs_file(project_dir_path: str | Path,
               task_family: Optional[str],
               date_iso: str) -> Path:
    """``<project>/runs/<task>/<date>.json`` when a family is known,
    else ``<project>/runs/<date>.json``."""
    base = runs_dir(project_dir_path)
    fam = _resolve_family(task_family)
    return (base / fam / f"{date_iso}.json") if fam else (base / f"{date_iso}.json")


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _date_iso_from_run_id(run_id: str) -> str:
    """run_id is ``YYYY-MM-DD-HHMMSS_box<N>`` -> ``YYYY-MM-DD``."""
    return run_id[:10] if len(run_id) >= 10 else ""


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Thin JSON wrapper around the canonical
    :func:`source.config.multi_instance.write_atomic`. Kept as a
    typed helper because callers pass dicts, not pre-serialised bytes,
    the single line of json.dumps lives here instead of being
    repeated at every call site."""
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    write_atomic(path, payload, encoding="utf-8")


def _make_project_conflict_dialog(mw):
    """Build the modal callback ``save_experiment`` invokes when the
    project file changed on disk AND the change collides with a local
    edit (same field, different values). Auto-mergeable changes never
    reach this prompt, they merge silently.
    """
    def _callback(conflicts: list[MergeConflict]) -> str:
        # Up to 5 lines of preview; truncate the rest.
        preview = [f"  • {'.'.join(c.path) or '<root>'}: "
                   f"yours={c.local!r}  theirs={c.remote!r}"
                   for c in conflicts[:5]]
        if len(conflicts) > 5:
            preview.append(f"  • ... and {len(conflicts) - 5} more")
        text = (
            "Another instance changed this project on disk since you "
            "loaded it. The following field(s) conflict:\n\n"
            + "\n".join(preview)
            + "\n\nReload discards your local edits and picks up the "
              "on-disk version.\nOverwrite saves yours, discarding "
              "the other instance's edits.\nCancel does nothing."
        )
        box = QtWidgets.QMessageBox(mw)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Project changed on disk")
        box.setText(text)
        reload_btn = box.addButton("Reload from disk",
                                   QtWidgets.QMessageBox.ButtonRole.ResetRole)
        overwrite_btn = box.addButton("Overwrite (use mine)",
                                      QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is overwrite_btn:
            return "overwrite"
        if clicked is reload_btn:
            return "reload"
        return "cancel"
    return _callback


def _default_load_dir() -> str:
    """Absolute starting folder for the file picker (resolves CWD-relatives)."""
    root = projects_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug("projects_root.mkdir failed: %s", e)
    return str(root.resolve())


# ============================================================================
# Template writer
# ============================================================================


# save_template lives in source.config.experiment; re-exported here for
# callers that import it via project_workflow.
from source.config.experiment import save_template  # noqa: F401


# ============================================================================
# In-memory history cache
# ============================================================================

_runs_cache_lock = threading.RLock()
# Cache keyed by (project_dir, task_family_folder, date_iso).
_runs_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

# Thread-local "batch mode", when set, open_run / close_run skip their
# per-call disk flush. The batch driver (UniversalStartDialog) sets this
# around the per-box loop and calls flush_pending_runs() once at the end
# so N starts share a single write.
_BATCH_DEFER = threading.local()


def _batch_defer_active() -> bool:
    return bool(getattr(_BATCH_DEFER, "active", False))


def _cache_key(project_dir_path: str | Path,
               task_family: Optional[str],
               date_iso: str) -> Tuple[str, str, str]:
    return (str(Path(project_dir_path).resolve()),
            _resolve_family(task_family),
            date_iso)


def _fresh_skeleton(project_dir_path: str | Path,
                    task_family: Optional[str],
                    date_iso: str,
                    cfg: Optional[Config] = None) -> Dict[str, Any]:
    """Day file is ``{date, runs[]}``; lineage is derivable from per-run
    config_djb2 snapshots + change_log.jsonl."""
    return {"date": date_iso, "runs": []}


def _load_into_cache(project_dir_path: str | Path,
                     task_family: Optional[str],
                     date_iso: str,
                     cfg: Optional[Config] = None) -> Dict[str, Any]:
    key = _cache_key(project_dir_path, task_family, date_iso)
    with _runs_cache_lock:
        if key in _runs_cache:
            return _runs_cache[key]
        rp = _runs_file(project_dir_path, task_family, date_iso)
        if rp.exists():
            try:
                with open(rp, encoding="utf-8") as fh:
                    data = json.load(fh)
                data.setdefault("runs", [])
                data.setdefault("date", date_iso)
            except (OSError, json.JSONDecodeError) as e:
                logger.error("runs file read failed (%s); starting fresh: %s",
                             rp, e)
                data = _fresh_skeleton(project_dir_path, task_family, date_iso, cfg=cfg)
        else:
            data = _fresh_skeleton(project_dir_path, task_family, date_iso, cfg=cfg)
        _runs_cache[key] = data
        return data


def _flush_cache_entry(project_dir_path: str | Path,
                       task_family: Optional[str],
                       date_iso: str) -> None:
    key = _cache_key(project_dir_path, task_family, date_iso)
    with _runs_cache_lock:
        data = _runs_cache.get(key)
    if data is None:
        return
    _atomic_write_json(_runs_file(project_dir_path, task_family, date_iso), data)


def flush_pending_runs(project_dir_path: str | Path) -> int:
    """Flush every cached runs/<family>/<date>.json for this project.

    Used by the multi-box start coordinator: each per-box ``open_run``
    is called with ``defer_flush=True``, and a single
    ``flush_pending_runs`` call writes all (family, date) entries to
    disk in one pass. Returns the number of files flushed.

    Safe to call repeatedly, entries that have already been flushed by
    a previous immediate-mode call are simply re-written with the same
    content.
    """
    wanted_dir = str(Path(project_dir_path).resolve())
    flushed = 0
    with _runs_cache_lock:
        targets = [
            (fam, date) for (cached_dir, fam, date) in list(_runs_cache.keys())
            if cached_dir == wanted_dir
        ]
    for family, date_iso in targets:
        try:
            _flush_cache_entry(project_dir_path, family, date_iso)
            flushed += 1
        except Exception as e:
            logger.warning(
                "runs: flush_pending_runs(%s, %s, %s) failed: %s",
                project_dir_path, family, date_iso, e,
            )
    return flushed


# ============================================================================
# Per-run row builder
# ============================================================================


def _build_run_row(cfg: Config,
                   setup_id: int,
                   run_id: str,
                   subject_id: str) -> Dict[str, Any]:
    """Build the per-day runs JSON row.

    The row is a 6-field index; anything reconstructable from the MCU TSV
    header or the project_config snapshot is omitted. The snapshot is
    pinned at record-start under
    ``<project>/source/configs/<config_djb2>.json`` by the caller, so
    ``config_djb2`` is the only lineage link the row needs.
    """
    return {
        "id":          run_id,
        "subject_id":  subject_id,
        "box_number":  int(setup_id),
        "config_djb2": cfg.config_djb2 or "",
        "mcu_tsv":     "",       # filled by close_run
        "video_mp4":   None,
    }


# ============================================================================
# History writers (all routed through the cache)
# ============================================================================


def _run_id_now(setup_id: int, datetime_now: Optional[datetime] = None) -> str:
    """``<FILE_STEM_TS_FMT>_box<N>``, run identifier.

    Accepts the master ``datetime_now`` captured at record-click so the
    run_id timestamp matches every other artifact of the same session
    (MCU TSV filename, video filename, _video_data.txt filename). Without
    this argument the function falls back to ``datetime.now()``, useful
    for tests but introduces sub-second drift in production paths.
    """
    from source.datetime_formats import FILE_STEM_TS_FMT
    dt = datetime_now if datetime_now is not None else datetime.now()
    return f"{dt.strftime(FILE_STEM_TS_FMT)}_box{int(setup_id)}"


def _family_for_box(cfg: Config, setup_id: int) -> str:
    """Task-family folder for a box's run rows.

    Task is run-only and not on the cfg, so the family can't be resolved
    here; every row files directly under ``runs/<date>.json``. The run
    scanners cover both that flat layout and ``runs/<family>/``.
    """
    return ""


def open_run(project_dir_path: str | Path,
             cfg: Config,
             setup_id: int,
             subject_id: str,
             *,
             pycboard=None,
             defer_flush: bool = False,
             datetime_now: Optional[datetime] = None,
             date_iso: Optional[str] = None) -> Optional[str]:
    """Append a new run row to today's ``runs/<family>/<date>.json`` and
    return the run_id. Returns None for empty ``subject_id`` (dry run).

    ``datetime_now`` is the master record-click timestamp captured in
    the caller (BoxControlWidget.on_record_clicked). Passing it aligns the
    run_id, the runs JSON ``started_at`` field, the MCU TSV filename
    stem, the video filename stem and the _video_data.txt header all on
    the SAME instant. Default ``None`` -> fresh ``datetime.now()`` for
    tests / non-parallel paths.

    ``defer_flush=True`` skips the per-call disk write; the caller must
    invoke ``flush_pending_runs(project_dir_path)`` once at end of the
    batch so N starts share a single read-modify-write on the runs
    file. Used by UniversalStartDialog to collapse 8–16 per-box
    open_run() disk writes into one.
    """
    if not subject_id:
        logger.info("runs: box %s dry run (no subject_id), skipped",
                    setup_id)
        return None

    if datetime_now is None:
        datetime_now = datetime.now()
    run_id = _run_id_now(setup_id, datetime_now)
    row = _build_run_row(cfg, setup_id, run_id, subject_id)

    # ``date_iso`` lets a recovery tool file the row under the run's REAL date
    # (the recovered record-click day) instead of wall-clock today, so the
    # runs index matches the data folder. Live callers pass None → today.
    if date_iso is None:
        date_iso = _today_iso()
    family = _family_for_box(cfg, setup_id)
    with _runs_cache_lock:
        data = _load_into_cache(project_dir_path, family, date_iso, cfg=cfg)
        data["runs"].append(row)
    if not defer_flush and not _batch_defer_active():
        _flush_cache_entry(project_dir_path, family, date_iso)
    logger.info("runs: opened %s in %s/%s.json (flush=%s)",
                run_id, _resolve_family(family), date_iso,
                "deferred" if defer_flush or _batch_defer_active() else "immediate")
    return run_id


def _locate_run_in_cache(project_dir_path: str | Path,
                         run_id: str
                         ) -> Optional[Tuple[str, str, Dict[str, Any], int]]:
    """Walk ``runs/<family>/<date>.json`` looking for the run_id. Returns
    ``(family, date_iso, data, row_idx)`` on hit; ``None`` otherwise.

    The cache is checked first (covers the common case of a freshly-opened
    run being closed in the same session). If no hit, the on-disk tree is
    scanned, at most one nested level (``runs/*/*.json``).
    """
    target_date = _date_iso_from_run_id(run_id)

    # 1) In-memory cache, fast path.
    with _runs_cache_lock:
        for (pd_key, family_key, date_key), data in _runs_cache.items():
            if pd_key != str(Path(project_dir_path).resolve()):
                continue
            for i, row in enumerate(data.get("runs", [])):
                if row.get("id", row.get("run_id")) == run_id:
                    return family_key, date_key, data, i

    # 2) On-disk scan, target date first. Rows with no task family live
    # directly in runs/<date>.json, scan both levels or those rows are
    # invisible after a restart.
    rd = runs_dir(project_dir_path)
    if not rd.exists():
        return None
    candidates = sorted(rd.glob("*.json"))
    for fam_dir in rd.iterdir():
        if fam_dir.is_dir():
            candidates.extend(sorted(fam_dir.glob("*.json")))
    # Try target_date matches first for cache locality.
    candidates.sort(key=lambda p: 0 if p.stem == target_date else 1)
    for rp in candidates:
        date_iso = rp.stem
        family = "" if rp.parent == rd else rp.parent.name
        try:
            data = _load_into_cache(project_dir_path, family, date_iso)
        except Exception:
            continue
        for i, row in enumerate(data.get("runs", [])):
            if row.get("id", row.get("run_id")) == run_id:
                return family, date_iso, data, i
    return None


def close_run(project_dir_path: str | Path,
              run_id: str,
              *,
              status: str = "completed",
              files: Optional[Dict[str, Any]] = None) -> None:
    """Merge final output paths into the 6-field run row.

    The runs JSON row's ``mcu_tsv`` and ``video_mp4`` fields are filled in
    here from the runtime ``files`` dict produced by
    ``_collect_output_files`` (paths only, no hashes, analyzer hashes on
    demand). ``status`` is logged but not persisted; the MCU TSV end-marker
    is the canonical "did the run complete" signal.
    """
    with _runs_cache_lock:
        hit = _locate_run_in_cache(project_dir_path, run_id)
        if hit is None:
            logger.debug("runs: close_run(%s), row not found", run_id)
            return
        family, date_iso, data, idx = hit
        row = data["runs"][idx]
        if files:
            # Store RELATIVE output paths so the shared runs JSON never leaks
            # an absolute D:/ or C:/Users/<name> path.
            from source.config.experiment import relpath_for_storage
            if files.get("mcu_tsv"):
                row["mcu_tsv"] = relpath_for_storage(files["mcu_tsv"], project_dir_path)
            if files.get("video_mp4"):
                row["video_mp4"] = relpath_for_storage(files["video_mp4"], project_dir_path)
    if not _batch_defer_active():
        _flush_cache_entry(project_dir_path, family, date_iso)
    logger.info("runs: closed %s (status=%s)", run_id, status)


def scan_unfinished_runs(project_dir_path: str | Path
                         ) -> List[Tuple[str, Dict[str, Any]]]:
    """Walk every ``runs/<family>/<date>.json``, return runs without
    output paths (heuristic: no mcu_tsv path means the run didn't reach
    close_run cleanly).

    The row carries no ``ended_at``; the MCU TSV's end-marker is the
    canonical "did it complete" signal. Best-effort for crash recovery only.
    """
    rd = runs_dir(project_dir_path)
    if not rd.exists():
        return []
    # No-family rows live directly in runs/<date>.json; family rows one
    # level down. Both must be scanned or crash recovery never fires.
    run_files = sorted(rd.glob("*.json"))
    for fam_dir in rd.iterdir():
        if fam_dir.is_dir():
            run_files.extend(sorted(fam_dir.glob("*.json")))
    out: List[Tuple[str, Dict[str, Any]]] = []
    for rp in run_files:
        try:
            with open(rp, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        out.extend((row.get("id") or row.get("run_id", ""), row)
                   for row in data.get("runs", [])
                   if not row.get("mcu_tsv") and not row.get("crashed"))
    return out


def mark_run_crashed(project_dir_path: str | Path, run_id: str) -> None:
    """Stamp ``crashed: true`` on the run row.

    Not derivable elsewhere, a crashed run may have no MCU TSV at all,
    and it stops ``scan_unfinished_runs`` re-prompting for the same run
    on every startup."""
    with _runs_cache_lock:
        hit = _locate_run_in_cache(project_dir_path, run_id)
        if hit is None:
            logger.debug("runs: mark_run_crashed(%s), row not found", run_id)
            return
        family, date_iso, data, idx = hit
        data["runs"][idx]["crashed"] = True
    if not _batch_defer_active():
        _flush_cache_entry(project_dir_path, family, date_iso)
    logger.info("runs: marked %s crashed", run_id)


# ============================================================================
# Dialogs
# ============================================================================


def ask_project_folder(parent, title: str = "Save Project",
                       default_name: str = "MyProject") -> Optional[Path]:
    """One save-file dialog, same UX as camera config save.

    The user types a project name in the filename field (or navigates
    elsewhere first). We extract the stem as the project name and
    create ``<dir>/<stem>/experiment_config.json``.

    Returns the resolved project FOLDER Path on accept, ``None`` on cancel.
    """
    default_dir = Path(_default_load_dir())
    suggested = str(default_dir / f"{default_name}.json")
    chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        title,
        suggested,
        "Project (*.json);;All Files (*)",
    )
    if not chosen:
        return None
    p = Path(chosen).resolve()
    project_name = p.stem.strip()
    if not project_name:
        return None
    return (p.parent / project_name).resolve()


# ============================================================================
# Save / Load / Save-As / New-from-Template dispatchers
# ============================================================================


def _append_save_event(project_dir_path: Path, cfg: Config,
                       *, action: str = "save", source: str = "Save Config",
                       extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        kwargs = {"action": action, "source": source,
                  "path": str(project_dir_path / EXPERIMENT_CONFIG_FILENAME),
                  "new": cfg.config_djb2}
        if extra:
            kwargs["reason"] = json.dumps(extra, default=str)
        _hist.append_change(project_dir_path, **kwargs)
    except Exception as e:
        logger.debug("append_change(save) failed: %s", e)


def _is_projects_root_itself(project_dir: Path) -> bool:
    """True when ``project_dir`` is the ``experiments/projects/`` root itself
    (i.e. NOT a named subfolder under it).

    Guards against saving ``experiment_config.json`` straight at the root,
    which would make every subsequent save overwrite the same stray file
    instead of a real per-project subfolder.
    """
    try:
        return project_dir.resolve() == projects_root().resolve()
    except Exception:
        return False


def _save_to_disk(mw, cfg: Config, project_dir: Path) -> Optional[Path]:
    """Atomic: write experiment_config.json + template.json + ensure history/.

    Returns the resolved experiment_config.json path on success, None on
    failure (after showing a critical popup with the full exception).
    The popup is the ONE place save errors surface, never silent.

    Hard refusal: if ``project_dir`` resolves to the ``projects/`` root itself
    (instead of a named subfolder), bail loudly.
    """
    if _is_projects_root_itself(project_dir):
        QtWidgets.QMessageBox.critical(
            mw, "Invalid project path",
            f"Refusing to save at the projects root itself:\n{project_dir}\n\n"
            "A project must live in a NAMED subfolder. Pick a different "
            "name (Save Config) or pick a real project file (Load Config).")
        return None

    # If this project was loaded earlier this session and another GUI has
    # since edited the same fields, route through the OCC guard so the user
    # gets a Reload / Overwrite / Cancel dialog instead of silently
    # clobbering the other instance's work. isinstance (not truthiness):
    # test mocks auto-create Mock attributes for any getattr.
    from source.config.multi_instance import ProjectFileGuard as _PFG
    _g = getattr(mw, "_project_guard", None)
    guard = _g if isinstance(_g, _PFG) else None
    on_conflict = _make_project_conflict_dialog(mw) if guard else None
    # Serialize against the async autosave worker, both touch the same
    # guard + project files. Lock is host-owned (init'd by _init_autosave).
    save_lock = getattr(mw, "_project_save_lock", None)
    try:
        from source.config.experiment import serialize_for_save
        ctx = save_lock if save_lock is not None else _NullCtx()
        with ctx:
            payload = serialize_for_save(cfg)
            cfg_path = save_experiment(
                cfg, project_dir_path=project_dir,
                guard=guard, on_conflict=on_conflict,
                cached_payload=payload,
            )
            save_template(cfg, project_dir, cached_payload=payload)
            _ensure_history_dir(project_dir)
        _ensure_project_data_folder(cfg)
    except SaveConflictError as ce:
        logger.info("save aborted by user, %d conflict(s)", len(ce.conflicts))
        return None
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("save failed: %s\n%s", e, tb)
        QtWidgets.QMessageBox.critical(
            mw, "Save failed",
            f"Could not save project:\n\n{e}\n\nFull traceback in the log.")
        return None
    return Path(cfg_path).resolve()


def _ensure_project_data_folder(cfg: Config) -> None:
    """Pre-create ``<code>/data/<project_name>/`` on save.

    Done eagerly so the data target exists before the user clicks Record
    (no I/O latency at run-start, and Explorer/Finder can find it).
    """
    from source import paths as _app_paths
    name = (cfg.meta.project or cfg.experiment_name or "").strip()
    if not name:
        return
    try:
        (Path(_app_paths.top_dir) / "data" / name).mkdir(
            parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("project data folder mkdir failed: %s", e)


def save_project(mw, cfg: Config) -> Optional[Path]:
    """Save dispatcher.

    Active project loaded → silent overwrite (status bar message).
    No active project     → ONE directory dialog. The folder the user
                            picks/creates IS the project; basename is the
                            project name. No separate name prompt.
    """
    active_pd = getattr(mw, "_active_project_dir", None)

    # Stale "active project" points at the projects root itself.
    # Reset so the user gets the picker.
    if active_pd is not None and _is_projects_root_itself(Path(active_pd)):
        logger.warning(
            "Active project dir points at projects root (%s), ignoring; "
            "treating as no project loaded.", active_pd)
        mw._active_project_dir = None
        active_pd = None

    # ---- silent overwrite ---------------------------------------------
    if active_pd is not None:
        project_dir = Path(active_pd).resolve()
        cfg_path = _save_to_disk(mw, cfg, project_dir)
        if cfg_path is None:
            return None
        _append_save_event(project_dir, cfg)
        mw._active_config = cfg
        mw._active_project_dir = cfg_path.parent
        mw.active_config_path = str(cfg_path)
        _maybe_status(mw, f"Saved: {cfg_path}", 5000)
        logger.info("project save (silent): %s", cfg_path)
        return cfg_path.parent

    # ---- first save: ONE save-file dialog (camera-config style) ------
    default_name = (cfg.meta.project or "MyProject").strip() or "MyProject"
    project_dir = ask_project_folder(mw, title="Save Project",
                            default_name=default_name)
    if project_dir is None:
        return None
    name = project_dir.name

    if experiment_config_path(project_dir).exists():
        reply = QtWidgets.QMessageBox.question(
            mw, "Project exists",
            f"Project '{name}' already exists at:\n{project_dir}\n\nOverwrite?",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return None

    cfg.meta.project = name
    cfg.experiment_name = name
    cfg_path = _save_to_disk(mw, cfg, project_dir)
    if cfg_path is None:
        return None

    mw._active_config = cfg
    mw._active_project_dir = project_dir
    mw.active_config_path = str(cfg_path)
    # First Save: move the draft snapshot workspace (HD/DLC sources +
    # change_log staged while creating) into the named project folder and
    # re-root the store, before the create event so its lines precede it.
    _materialize = getattr(mw, "_materialize_draft_store", None)
    if callable(_materialize):
        try:
            _materialize(project_dir)
        except Exception as e:
            logger.warning("materialize draft store failed: %s", e)
    _append_save_event(project_dir, cfg, action="create", source="New Project")

    QtWidgets.QMessageBox.information(
        mw, "Project saved",
        f"Project '{name}' saved.\n\nLocation:\n{cfg_path}")
    logger.info("project save (new): %s", cfg_path)
    return project_dir


def load_project(mw, mode: str) -> Optional[Tuple[Config, Path]]:
    """One file picker, dispatch by filename.

    User picks ``experiment_config.json`` -> load as project.
    User picks ``template.json``         -> prompt name + new_from_template.
    User picks anything else             -> warn + None.
    """
    start_dir = _default_load_dir()
    chosen, _filt = QtWidgets.QFileDialog.getOpenFileName(
        mw, "Load Project or Template",
        os.path.join(start_dir, EXPERIMENT_CONFIG_FILENAME),
        f"Project or Template "
        f"({EXPERIMENT_CONFIG_FILENAME} {TEMPLATE_FILENAME});;"
        "JSON Files (*.json);;All Files (*)",
    )
    if not chosen:
        return None

    picked = Path(chosen).resolve()
    if picked.name == EXPERIMENT_CONFIG_FILENAME:
        return _load_project_from_path(mw, picked, mode)
    if picked.name == TEMPLATE_FILENAME:
        # New flow: no name prompt, let the folder picker take the
        # name (basename of whatever folder the user creates).
        return new_from_template(mw, picked, mode=mode)

    QtWidgets.QMessageBox.warning(
        mw, "Not a project or template",
        f"Pick one of:\n  - {EXPERIMENT_CONFIG_FILENAME}  (load existing project)\n"
        f"  - {TEMPLATE_FILENAME}  (start new project from template)\n\n"
        f"You picked: {picked.name}")
    return None


def _load_project_from_path(mw, cfg_path: Path,
                            mode: str) -> Optional[Tuple[Config, Path]]:
    """Load an existing project given the absolute path of its config file."""
    project_dir = cfg_path.parent.resolve()

    # Same guard as save: refuse to treat the projects ROOT as a project.
    # Loading a stray config there would corrupt subsequent saves.
    if _is_projects_root_itself(project_dir):
        QtWidgets.QMessageBox.warning(
            mw, "Not a valid project",
            f"That file lives at the projects root, not inside a named "
            f"project subfolder:\n{cfg_path}\n\n"
            "Delete it if it's stray, or pick a project that lives inside "
            "its own subfolder under experiments/projects/.")
        return None

    try:
        from source.config.experiment import load_experiment_with_guard
        cfg, guard = load_experiment_with_guard(project_dir)
    except Exception as e:
        logger.error("load_experiment(%s) failed: %s", project_dir, e)
        QtWidgets.QMessageBox.critical(
            mw, "Load failed", f"Could not load project:\n\n{e}")
        return None

    # Refuse an incompatible-major-schema file BEFORE committing the load.
    # from_dict would drop its boxes/cameras/tracking (blank rig), and load
    # then sets active_config_path so the next autosave would overwrite the
    # original file, silent data loss. Returning None leaves it untouched.
    from source.config.experiment import (
        SCHEMA_VERSION as _SV, schema_major_compatible)
    if not schema_major_compatible(cfg.schema_version):
        QtWidgets.QMessageBox.critical(
            mw, "Incompatible project version",
            f"Project '{project_dir.name}' was saved with schema "
            f"{cfg.schema_version}, which this app (schema {_SV}) can't "
            "read. It was NOT loaded, so your file is left untouched.\n\n"
            "Open it with a matching app version, or re-create the project "
            "here.")
        return None

    # Stash the guard on the main window so subsequent saves go through
    # OCC + auto-merge, preventing multiple GUIs from silently overwriting
    # each other.
    mw._project_guard = guard

    # One-shot folder rename: <project>/backgrounds/ → background_images/.
    old_bg = project_dir / "backgrounds"
    new_bg = project_dir / "background_images"
    if old_bg.is_dir() and not new_bg.exists():
        try:
            old_bg.rename(new_bg)
            logger.info("Migrated %s -> %s", old_bg, new_bg)
        except OSError as e:
            logger.warning("Could not rename %s: %s", old_bg, e)

    if cfg.mode and cfg.mode != mode:
        QtWidgets.QMessageBox.warning(
            mw, "Wrong mode",
            f"Project '{project_dir.name}' is mode={cfg.mode!r}; "
            f"this app is mode={mode!r}.")
        return None

    _ensure_history_dir(project_dir)
    _autoload_cohort_metadata(mw, cfg, project_dir)
    logger.info("project loaded: %s", project_dir)
    return cfg, project_dir


def _autoload_cohort_metadata(mw, cfg: Config, project_dir: Path) -> None:
    """Restore metadata_manager state from ``cfg.meta.metadata_file``.

    Silent: missing file or no metadata_manager → no-op. The cohort sheet
    became a project artefact when ``_project_adopt_metadata_file`` copied
    it on the original load; this just re-points the in-memory cache so
    the assign-subjects picker is ready immediately after open.
    """
    rel = getattr(cfg.meta, "metadata_file", "") or ""
    if not rel:
        return
    full = project_dir / rel
    if not full.exists():
        logger.warning("metadata file %s referenced by cfg is missing", full)
        return
    mm = getattr(mw, "metadata_manager", None)
    if mm is None or not hasattr(mm, "load_metadata"):
        return
    try:
        mm.load_metadata(str(full), mw)
        logger.info("cohort metadata auto-loaded from %s", full)
    except Exception as e:
        logger.warning("cohort metadata auto-load failed (%s): %s", full, e)


def new_from_template(mw,
                      source_template_path: str | Path,
                      new_name: Optional[str] = None,
                      mode: Optional[str] = None) -> Optional[Tuple[Config, Path]]:
    """Instantiate a new project from a template file.

    When ``new_name`` is None (GUI flow), opens the folder picker, the
    chosen folder's basename becomes the project name. When given (test
    callers), uses it under the default projects root .
    """
    from source.config.experiment import load as _cfg_load
    tpath = Path(source_template_path).resolve()
    if not tpath.exists():
        QtWidgets.QMessageBox.warning(
            mw, "Template Missing", f"Template not found:\n{tpath}")
        return None

    try:
        cfg = _cfg_load(tpath)
    except Exception as e:
        logger.error("load_template(%s) failed: %s", tpath, e)
        QtWidgets.QMessageBox.critical(
            mw, "Template Load Failed", f"Could not load template:\n\n{e}")
        return None

    from source.config.experiment import (
        SCHEMA_VERSION as _SV, schema_major_compatible)
    if not schema_major_compatible(cfg.schema_version):
        QtWidgets.QMessageBox.critical(
            mw, "Incompatible template version",
            f"This template was saved with schema {cfg.schema_version}, "
            f"which this app (schema {_SV}) can't read, creating a project "
            "from it would produce a blank rig. Use a matching app version.")
        return None

    if mode and cfg.mode and cfg.mode != mode:
        QtWidgets.QMessageBox.warning(
            mw, "Wrong mode",
            f"Template is mode={cfg.mode!r}; this app is mode={mode!r}.")
        return None

    if new_name:
        project_dir = (_default_load_dir_path() / new_name).resolve()
    else:
        project_dir = ask_project_folder(mw, title="New Project from Template")
        if project_dir is None:
            return None
        new_name = project_dir.name

    if experiment_config_path(project_dir).exists():
        QtWidgets.QMessageBox.warning(
            mw, "Project exists",
            f"Project '{new_name}' already exists at:\n{project_dir}\n\n"
            "Pick a different name/folder.")
        return None

    cfg.meta.project = new_name
    cfg.experiment_name = new_name
    cfg.meta.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg.meta.session_label = ""

    cfg_path = _save_to_disk(mw, cfg, project_dir)
    if cfg_path is None:
        return None

    mw._active_config = cfg
    mw._active_project_dir = project_dir
    mw.active_config_path = str(cfg_path)
    # Materialise any draft snapshot workspace into the new project + re-root.
    _materialize = getattr(mw, "_materialize_draft_store", None)
    if callable(_materialize):
        try:
            _materialize(project_dir)
        except Exception as e:
            logger.warning("materialize draft store (template) failed: %s", e)
    _append_save_event(
        project_dir, cfg, action="new_from_template", source="Template",
        extra={"source_template": str(tpath)})

    QtWidgets.QMessageBox.information(
        mw, "Project created from template",
        f"Project '{new_name}' created.\n\nLocation:\n{cfg_path}")
    logger.info("project new_from_template: %s -> %s", tpath, project_dir)
    return cfg, project_dir


def _default_load_dir_path() -> Path:
    """Same as ``_default_load_dir`` but returns Path, not str."""
    return Path(_default_load_dir())


# ============================================================================
# Misc helpers
# ============================================================================


def _ensure_history_dir(project_dir_path: Path) -> None:
    """Pre-create ``<project>/runs/`` on save/load. Per-family subdirs are
    created lazily by ``_flush_cache_entry`` when the first run row for
    that family lands."""
    try:
        runs_dir(project_dir_path).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("runs_dir mkdir(%s) failed: %s", project_dir_path, e)


def _maybe_status(mw, msg: str, ms: int = 3000) -> None:
    sb = getattr(mw, "statusbar", None)
    if sb is None:
        return
    try:
        sb.showMessage(msg, ms)
    except Exception:
        pass
