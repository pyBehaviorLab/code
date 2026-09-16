"""Per-task variable lifecycle, classification store + runtime push/capture.

Every variable is one of two kinds, decided by a single per-variable flag:

  * RESET (default): snaps back to the task-file module default on every
    Upload (the MCU re-imports ``task_file``). Never remembered.
  * PERSISTENT, the final MCU value is captured at Stop and restored at the
    next Upload. Remembered by variable NAME (no subject dimension).

Two cohesive layers live here:

  CLASSIFICATION (which variables persist)
      Layered, resolved per box:
        1. Project-specific ``flags`` in
           ``<project>/<task>/persistent_variables.json``: used when a
           project is loaded and it has a flag for the variable.
        2. Per-task template ``<task>.variables.json`` next to the task .py,
           the always-editable fallback (also the only store in draft /
           no-project mode).
      ``resolve_specs`` overlays (1) on (2) → ``List[BoxVariableSpec]``.

  RUNTIME PIPELINE
      Consumes that spec list to drive the MCU and the remembered values in
      ``<project>/<task>/persistent_variables.json``. Pushes at Upload,
      captures at Stop.

Where variables are pushed
--------------------------

All pushes happen at UPLOAD time (``setup_state_machine`` success), never at
Record-click time. Record is just ``start_framework()``.

Two sources, applied in order at Upload:

  1. ``apply_pre_run_hw(pycboard, specs, hw_prompt_values)``: pushes
        ``hw_*`` vars from the rig store; missing values populate
        ``hw_missing`` for the prompt dialog.

  2. ``restore_pers_vars(pycboard, project_dir, task, specs)``: pushes the
        last-session value of every PERSISTENT variable (by name). RESET vars
        are left at the task-file default the MCU already holds.

At STOP time:

  3. ``capture_persistent(pycboard, specs) -> {name: value}``: ONE
        ``get_variables()`` round-trip filtered to persistent names.

  4. ``write_pers_vars(project_dir, task, pers_vars)``: merges captured
        values into the project store, keyed by name.

File shapes
-----------

Per-task template, one ``<task>.variables.json`` per task, next to its .py::

    {
        "task_hash": 1234567890,        # djb2 of the task .py at save time
        "task_path": "Reversal/Reversal",
        "saved_at": "2026-07-21 18:42:11",
        "variables": [
            {"name": "n_trials", "persistent": false},
            {"name": "stage",    "persistent": true}
        ]
    }

Absent rows imply RESET (``persistent=False``).

Project store, one ``persistent_variables.json`` per task per project::

    {
      "flags":  {"stage": true, "n_trials": false},   # project-specific class.
      "values": {"stage": 3}                           # last-run value by name
    }

``flags`` overrides the template when a project is loaded; ``values`` holds
the remembered value of each persistent variable.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from source.config.experiment import (
    BoxVariableSpec,
    _djb2_int_from_file as _djb2_int,
)
from source.log import get_logger

logger = get_logger()

SIDECAR_SUFFIX = ".variables.json"
PERSISTENT_FILE = "persistent_variables.json"
HW_PREFIX = "hw_"

_SCALAR = (int, float, str, bool)


# ============================================================================
#  TEMPLATE STORE, <task>.variables.json (per task; List[BoxVariableSpec])
# ============================================================================


def sidecar_path(task_py_path: str | Path) -> Path:
    """Return the sidecar JSON path for a task .py file.

    Accepts either the task's .py path or its stem-style path; the sidecar
    always sits next to the .py with the same stem and the
    ``.variables.json`` suffix.

        tasks/Reversal/Reversal.py   → tasks/Reversal/Reversal.variables.json
        tasks/Reversal/Reversal      → tasks/Reversal/Reversal.variables.json
    """
    p = Path(task_py_path)
    if p.suffix == ".py":
        p = p.with_suffix("")
    return p.with_name(p.name + SIDECAR_SUFFIX)


def task_py_path_from_relname(task_relname: str,
                              tasks_root: str | Path = "tasks") -> Path:
    """Resolve a task relative name (``"Reversal/Reversal"``) to the on-disk
    .py path under ``tasks/``. Matches ``RunTask.mcu_upload_task``:
    sm_dir = tasks/<parent>, sm_name = <stem>.
    """
    rel = Path(str(task_relname).strip())
    if rel.suffix == ".py":
        rel = rel.with_suffix("")
    return Path(tasks_root) / rel.parent / (rel.name + ".py")


def task_hash(task_py_path: str | Path) -> int:
    """djb2 hash of the task .py contents (0 if file missing)."""
    return _djb2_int(task_py_path)


def load_specs(task_py_path: str | Path) -> List[BoxVariableSpec]:
    """Load template specs from the task-scoped sidecar. Returns ``[]`` when
    the file is absent or unreadable, caller treats every variable as RESET
    (the MCU's ``import task_file`` value is used)."""
    p = sidecar_path(task_py_path)
    if not p.is_file():
        return []
    return _load_from_path(p)


def save_specs(task_py_path: str | Path,
         specs: List[BoxVariableSpec],
         task_relname: Optional[str] = None) -> None:
    """Atomically write the task-scoped template sidecar next to the .py."""
    _write_payload(sidecar_path(task_py_path), task_py_path, specs,
                   task_relname=task_relname)


def hash_matches(task_py_path: str | Path) -> Optional[bool]:
    """True if the sidecar's ``task_hash`` matches the live .py hash, False if
    it differs, None if the sidecar is absent."""
    return _hash_matches_at(sidecar_path(task_py_path), task_py_path)


def _hash_matches_at(p: Path,
                     task_py_path: str | Path) -> Optional[bool]:
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    saved = data.get("task_hash") if isinstance(data, dict) else None
    if not isinstance(saved, int):
        return None
    return int(saved) == int(task_hash(task_py_path))


def _load_from_path(p: Path) -> List[BoxVariableSpec]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("variables sidecar read failed (%s): %s", p, e)
        return []
    rows = data.get("variables") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    return [BoxVariableSpec.from_dict(row) for row in rows
            if isinstance(row, dict) and row.get("name")]


def _write_payload(p: Path,
                   task_py_path: str | Path,
                   specs: List[BoxVariableSpec],
                   task_relname: Optional[str] = None) -> Optional[Path]:
    """Atomic write of the template sidecar payload."""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("variables sidecar mkdir failed (%s): %s", p.parent, e)
        return None

    payload = {
        "task_hash": task_hash(task_py_path),
        "task_path": task_relname or Path(task_py_path).with_suffix("").as_posix(),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variables": [asdict(s) for s in specs if s.name],
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, p)
        return p
    except OSError as e:
        logger.error("variables sidecar write failed (%s): %s", p, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


# ============================================================================
#  PROJECT STORE, <project>/<task>/persistent_variables.json
#  (flags = project-specific classification; values = remembered by name)
# ============================================================================


def _project_store_path(project_dir, task_family: str) -> Optional[Path]:
    if not (project_dir and task_family):
        return None
    return Path(project_dir) / task_family / PERSISTENT_FILE


def _read_project_store(project_dir, task_family: str) -> Dict[str, Any]:
    path = _project_store_path(project_dir, task_family)
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("_read_project_store %s: %s", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def _merge_project_store(project_dir,
                         task_family: str,
                         section: str,
                         mapping: Dict[str, Any],
                         *,
                         scalar_only: bool = False) -> Optional[Path]:
    """Merge ``mapping`` into one section (``flags`` / ``values``) of the
    project store under a cross-process lock (boxes run the same task
    concurrently). ``scalar_only`` drops non-scalar entries so the ``values``
    schema stays flat (self-heals on first write). Atomic tmp + replace.
    """
    from source.config.multi_instance import file_lock, FileLockTimeout
    path = _project_store_path(project_dir, task_family)
    if path is None or not mapping:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path.with_suffix(".json.lock"), timeout=5.0):
            data: Dict[str, Any] = {}
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8")) or {}
                except Exception:
                    data = {}
            if not isinstance(data, dict):
                data = {}
            existing = data.get(section)
            if not isinstance(existing, dict):
                existing = {}
            if scalar_only:
                existing = {k: v for k, v in existing.items()
                            if v is None or isinstance(v, _SCALAR)}
            existing.update(mapping)
            data[section] = existing
            data.pop("persistent", None)  # not part of the {flags, values} schema
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True, indent=2, ensure_ascii=False)
            tmp.replace(path)
        return path
    except (OSError, FileLockTimeout) as e:
        logger.error("_merge_project_store %s/%s: %s", path, section, e)
        return None


# ============================================================================
#  CLASSIFICATION, layered flag resolution + project-flag writes
# ============================================================================


def read_project_flags(project_dir, task_family: str) -> Dict[str, bool]:
    """Project-specific ``{name: persistent}`` flags, or ``{}`` when absent."""
    flags = _read_project_store(project_dir, task_family).get("flags")
    if not isinstance(flags, dict):
        return {}
    return {k: bool(v) for k, v in flags.items() if isinstance(k, str)}


def write_project_flags(project_dir,
                        task_family: str,
                        flags: Dict[str, bool]) -> Optional[Path]:
    """Merge project-specific persistent flags into the project store."""
    return _merge_project_store(
        project_dir, task_family, "flags",
        {k: bool(v) for k, v in flags.items() if isinstance(k, str)})


def resolve_specs(task_py_path: str | Path,
                  project_dir=None,
                  task_family: Optional[str] = None) -> List[BoxVariableSpec]:
    """Effective specs for a box: the per-task template overlaid by the
    project's own flags when a project is loaded (project wins per name)."""
    specs = load_specs(task_py_path)
    if not (project_dir and task_family):
        return specs
    flags = read_project_flags(project_dir, task_family)
    if not flags:
        return specs
    by_name: Dict[str, BoxVariableSpec] = {s.name: s for s in specs if s.name}
    for name, persistent in flags.items():
        if name in by_name:
            by_name[name].persistent = persistent
        else:
            by_name[name] = BoxVariableSpec(name=name, persistent=persistent)
    return list(by_name.values())


# ============================================================================
#  RUNTIME PIPELINE, push at Upload, capture at Stop
# ============================================================================


@dataclass
class ApplyResult:
    set_lines: List[Tuple[str, str, str]] = field(default_factory=list)  # (name, value_repr, source)
    hw_missing: Dict[str, Any] = field(default_factory=dict)
    pushed: Dict[str, Any] = field(default_factory=dict)  # {name: value} to batch-write


def _push(pycboard, name: str, value: Any, source: str,
          result: ApplyResult) -> None:
    """Collect a (name, value) for the batched write. Does NOT touch the MCU,
    the caller flushes ``result.pushed`` in one ``set_variables`` round-trip.
    ``pycboard`` is unused but kept so the collectors read uniformly."""
    result.pushed[name] = value
    result.set_lines.append((name, repr(value), source))


# ---- (1) hw_* push ------------------------------------------------------


def apply_pre_run_hw(pycboard,
                     specs: List[BoxVariableSpec],
                     hw_prompt_values: Optional[Dict[str, Any]] = None,
                     ) -> ApplyResult:
    """Push ``hw_*`` variables to the MCU from the rig store; names not in the
    store land in ``result.hw_missing`` for the HardwareVariablesDialog."""
    result = ApplyResult()
    if pycboard is None:
        return result
    sm_info = getattr(pycboard, "sm_info", None)
    mcu_vars = list((sm_info.variables or {}).keys()) if sm_info else []
    hw_prompt_values = hw_prompt_values or {}

    for name in mcu_vars:
        if not name.startswith(HW_PREFIX):
            continue
        if name in hw_prompt_values:
            _push(pycboard, name, hw_prompt_values[name], "hw_prompt", result)
        else:
            result.hw_missing[name] = (sm_info.variables or {}).get(name)

    already = {n for n, _, _ in result.set_lines}
    for name, val in hw_prompt_values.items():
        if name in already or name not in mcu_vars:
            continue
        _push(pycboard, name, val, "hw_prompt", result)
    return result


# ---- (2) persistent restore ---------------------------------------------


def read_pv_dict(project_dir, task_family: str) -> Dict[str, Any]:
    """Return the remembered ``{name: value}`` dict from the project store.

    Returns ``{}`` when no project/task or no saved values. Only scalar
    entries are returned, so a non-scalar ``values`` nesting is ignored
    (treated as nothing remembered)."""
    values = _read_project_store(project_dir, task_family).get("values")
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items()
            if v is None or isinstance(v, _SCALAR)}


def restore_pers_vars(pycboard,
                      project_dir,
                      task_family: str,
                      specs: List[BoxVariableSpec],
                      ) -> ApplyResult:
    """Push each PERSISTENT variable's last-session value back to the MCU."""
    result = ApplyResult()
    if pycboard is None:
        return result
    pv_dict = read_pv_dict(project_dir, task_family)
    if not pv_dict:
        return result
    spec_names = {s.name for s in specs if s and s.persistent and s.name}
    sm_info = getattr(pycboard, "sm_info", None)
    mcu_vars = set((sm_info.variables or {}).keys()) if sm_info else set()

    for name, value in pv_dict.items():
        if name not in spec_names or name not in mcu_vars:
            continue
        _push(pycboard, name, value, "(persistent value)", result)
    return result


# ---- (3) + (4) persistent capture + write -------------------------------


def capture_persistent(pycboard,
                       specs: List[BoxVariableSpec],
                       ) -> Dict[str, Any]:
    """Read final values of every persistent-flagged variable from the MCU in
    ONE ``get_variables()`` round-trip, filtered to the persistent names.

    Returns a name→value dict for ``write_pers_vars``; empty when the board is
    absent, wedged, or nothing is flagged persistent (on error-stop we read
    nothing and the caller keeps the previous values)."""
    if pycboard is None:
        return {}
    names = {s.name for s in specs if s and s.persistent and s.name}
    if not names:
        return {}
    try:
        allvars = pycboard.get_variables()
    except BaseException as e:
        logger.warning("capture_persistent get_variables failed: %s", e)
        return {}
    if not isinstance(allvars, dict):
        return {}
    return {k: v for k, v in allvars.items() if k in names and v is not None}


def write_pers_vars(project_dir,
                    task_family: str,
                    pers_vars: Dict[str, Any],
                    ) -> Optional[Path]:
    """Merge ``pers_vars`` (``{name: value}``) into the project store's
    ``values`` section. Idempotent; returns the file path on success.

    The merge runs under a cross-process ``file_lock`` because boxes run the
    SAME task concurrently, and it drops any non-scalar nesting so the file
    stays on the flat schema."""
    return _merge_project_store(project_dir, task_family, "values",
                                pers_vars, scalar_only=True)
