"""RunTask._capture_device_snapshots: device drivers captured from the MCU.

Fires on both the Load-HD path and every task upload. The file set comes from
``pycboard.device_files_on_pyboard`` (the board's own {file: djb2} map), so it
is correct whether the HD was just loaded or already on the board.
"""
from __future__ import annotations

from types import SimpleNamespace

from source.gui.widgets.run_task import RunTask


class _FakeStore:
    """Records capture_source calls; returns a FileRef-ish per .py file."""

    def __init__(self):
        self.calls = []

    def capture_source(self, path, *, kind, setup_id, label):
        self.calls.append((str(path), kind, setup_id, label))
        return SimpleNamespace(djb2=f"hash_{label}", name=label, path=str(path))


def test_capture_device_snapshots_uses_mcu_file_set():
    pyc = SimpleNamespace(device_files_on_pyboard={
        "poke.py": 0x11111111,
        "audio_board.py": 0x22222222,
        "__init__.py": 0x33333333,   # must be skipped
        "readme.txt": 0x44444444,    # non-.py must be skipped
    })
    stub = SimpleNamespace(pycboard=pyc, setup_id=1, _device_refs=None)
    store = _FakeStore()

    RunTask._capture_device_snapshots(stub, store)

    captured = {label for (_p, kind, _s, label) in store.calls}
    assert captured == {"poke.py", "audio_board.py"}
    assert all(kind == "device" for (_p, kind, _s, _l) in store.calls)
    # Refs stashed for _commit_box_sources_for_run to promote at record-start.
    assert {r.name for r in stub._device_refs} == {"poke.py", "audio_board.py"}


def test_capture_device_snapshots_noops_without_board_or_store():
    store = _FakeStore()
    # No pycboard → nothing captured, no crash.
    stub = SimpleNamespace(pycboard=None, setup_id=1, _device_refs=None)
    RunTask._capture_device_snapshots(stub, store)
    assert store.calls == []
    # No store → no crash.
    pyc = SimpleNamespace(device_files_on_pyboard={"poke.py": 1})
    stub2 = SimpleNamespace(pycboard=pyc, setup_id=1, _device_refs=None)
    RunTask._capture_device_snapshots(stub2, None)
