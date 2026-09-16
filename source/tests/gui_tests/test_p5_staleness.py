"""HD staleness + task/HD disk-hash poll + schema guard.

An edited hardware-definition .py (disk != board) is detected so the box is
flagged for re-upload. The task .py is only re-hashed when its mtime changes.
Loading a config with an incompatible schema_version warns.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import source.config.hashing as hashing
from source.config.hashing import djb2_int_from_file as _djb2_file
from source.gui.widgets.run_task import RunTask


class _Box(RunTask):
    """Bare RunTask (no Qt) for the disk-poll helpers."""
    def __init__(self):
        self.pycboard = None
        self.setup_id = 1


# ---- edited HD on disk is detected ---------------------------------------
def test_hd_change_detected(tmp_path):
    box = _Box()
    hd = tmp_path / "hardware_definition.py"
    hd.write_text("v1 = 1\n")
    box.pycboard = SimpleNamespace(
        _loaded_hwd_path=str(hd),
        _loaded_hwd_hash=_djb2_file(str(hd)),  # what's "on the board"
    )
    assert box._hd_changed_on_disk() is False, "fresh upload: disk == board"

    hd.write_text("v1 = 2  # edited\n")          # edit on disk
    os.utime(hd, (os.path.getmtime(hd) + 5, os.path.getmtime(hd) + 5))
    assert box._hd_changed_on_disk() is True, "edited HD must read as stale"


def test_hd_no_board_hash_is_not_stale():
    box = _Box()
    box.pycboard = SimpleNamespace(_loaded_hwd_path="", _loaded_hwd_hash=0)
    assert box._hd_changed_on_disk() is False


# ---- task hash gated on mtime --------------------------------------------
def test_task_hash_only_recomputed_on_mtime_change(tmp_path, monkeypatch):
    calls = []
    # run_task hashes via the canonical source.config.hashing, not the MCU
    # layer's private mirror, host code must never import that one.
    monkeypatch.setattr(hashing, "djb2_int_from_file",
                        lambda p: (calls.append(p), 999)[1])

    box = _Box()
    box.task_uploaded = True
    box.task_file_hash = 999           # matches the stubbed hash -> no change
    box.task_combo = SimpleNamespace(text=lambda: "T")

    # _check_task_consistency resolves "T" -> tasks/T.py; create it so the
    # .exists() check passes.
    import pathlib
    real_task = pathlib.Path("tasks") / "T.py"
    created = False
    if not real_task.exists():
        real_task.write_text("x = 1\n")
        created = True
    try:
        box._check_task_consistency()      # first call -> hashes once
        box._check_task_consistency()      # mtime unchanged -> skipped
        assert len(calls) == 1, "task must NOT be re-hashed when mtime unchanged"
        # Bump mtime -> next check re-hashes.
        os.utime(real_task, (os.path.getmtime(real_task) + 5,
                             os.path.getmtime(real_task) + 5))
        box._check_task_consistency()
        assert len(calls) == 2, "an mtime change must trigger a re-hash"
    finally:
        if created:
            real_task.unlink()


# ---- schema version guard ------------------------------------------------
def test_incompatible_schema_warns(caplog):
    from source.config.experiment import Config, SCHEMA_VERSION
    with caplog.at_level("WARNING"):
        cfg = Config.from_dict({"schema_version": "2.0"})
    assert any("schema_version" in r.getMessage() for r in caplog.records), \
        "an incompatible schema_version must warn"
    assert cfg.schema_version == "2.0"


def test_current_schema_does_not_warn(caplog):
    from source.config.experiment import Config, SCHEMA_VERSION
    with caplog.at_level("WARNING"):
        Config.from_dict({"schema_version": SCHEMA_VERSION})
    assert not any("schema_version" in r.getMessage() for r in caplog.records)
