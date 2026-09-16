"""Per-project change log, the single writer for ``<project>/change_log.jsonl``.

Holds two event families on one timeline: config mutations (Apply / Save /
dialog OK, via :func:`append_change`) and source-capture events (upload /
commit / dlc_capture / tracking_toggle, written by ``SnapshotStore`` through
:func:`append_event`). Every line carries a ``ts`` plus a schema discriminator
(``action`` for config edits, ``kind`` for captures).

(The per-run rollup at ``<project>/runs/<task_family>/<date>.json`` is owned by
``project_workflow.open_run / close_run``.)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

CHANGE_LOG_FILENAME = "change_log.jsonl"


def change_log_path(project_dir_path: str | Path) -> Path:
    """Return the change_log.jsonl path inside a project folder."""
    return Path(project_dir_path) / CHANGE_LOG_FILENAME


def append_event(project_dir_path: str | Path, event: Dict[str, Any]) -> None:
    """Append one pre-built event dict to the project's change_log.jsonl.

    The single low-level writer, shared by :func:`append_change` (config
    edits) and ``SnapshotStore`` (source-capture events). Prepends a ``ts``
    field; the caller supplies the rest of the schema. Single-line append
    in "a" mode is sufficient at human-click / upload cadence.
    """
    record = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **event}
    log = change_log_path(project_dir_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_change(project_dir_path: str | Path,
                  *,
                  source: str,
                  path: str,
                  old: Any = None,
                  new: Any = None,
                  action: str = "apply",
                  actor: str = "user",
                  reason: Optional[str] = None) -> None:
    """Append one config-mutation entry to the project's change_log.jsonl.

    Called from dialog Apply handlers. Delegates the write to
    :func:`append_event` so both event families share one writer.
    """
    append_event(project_dir_path, {
        "actor": actor,
        "action": action,
        "source": source,
        "path": path,
        "old": old,
        "new": new,
        "reason": reason,
    })
