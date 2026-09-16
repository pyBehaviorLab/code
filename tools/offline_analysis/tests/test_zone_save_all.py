"""Saving zones into every loaded recording, in one action.

"Copy to all" put the drawing onto every recording but left it beside them,
and "Save into file" wrote the files but committed only the recording on
screen. Getting zones into a whole folder therefore meant pressing both, in
that order, and nothing said so. These tests pin the combined action: apply to
all, then write every recording that carries zones.
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONUTF8", "1")

from PySide6 import QtWidgets


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


@pytest.fixture
def bench(app, monkeypatch):
    """A workbench over three recordings, with every side effect recorded."""
    from tools.offline_analysis.analyze import zone_workbench as zw

    seen = {"applied": [], "written": []}
    zones = {"a": [{"name": "left", "type": "circle"}], "b": [], "c": []}

    def apply_to(drawing, targets, frame_size):
        seen["applied"].append(list(targets))
        for key in targets:
            zones[key] = list(drawing)
        return len(targets)

    def write_files(keys):
        seen["written"].append(list(keys))
        return len(keys), []

    w = zw.ZoneWorkbench(
        frame_for=lambda _k: None,
        zones_for=lambda k: zones.get(k, []),
        apply_to=apply_to,
        scale_for=lambda _k: 0.0,
        size_for=lambda _k: (640, 480),
        zoned=lambda: [k for k, v in zones.items() if v],
        write_files=write_files,
    )
    w.set_recordings([("A", "a"), ("B", "b"), ("C", "c")], 0)
    # The confirmation is the operator's decision, not this test's subject.
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: QtWidgets.QMessageBox.StandardButton.Yes)
    try:
        yield w, seen, zones
    finally:
        w.deleteLater()


def test_the_button_sits_beside_save(bench):
    w, _seen, _zones = bench
    assert hasattr(w, "btn_write_all")
    assert w.btn_write_all.text() == "Save into all files"
    # Same row, directly after the single-file save. The buttons sit in a
    # nested row, so the tree is walked rather than assuming the top layout.
    def row_holding(layout, button):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is button:
                return layout
            sub = item.layout()
            if sub is not None:
                found = row_holding(sub, button)
                if found is not None:
                    return found
        return None

    row = row_holding(w.layout(), w.btn_write)
    assert row is not None, "the save button is in no layout"
    order = [row.itemAt(i).widget() for i in range(row.count())]
    assert order.index(w.btn_write_all) == order.index(w.btn_write) + 1


def test_save_into_all_files_applies_to_every_recording(bench):
    w, seen, _zones = bench
    w._write_into_all_files()
    assert seen["applied"], "the drawing was never applied"
    assert sorted(seen["applied"][-1]) == ["a", "b", "c"]


def test_save_into_all_files_then_writes_every_one(bench):
    w, seen, _zones = bench
    w._write_into_all_files()
    assert seen["written"], "no file was written"
    # Applying first is the point: all three now carry zones, so all three
    # are written, not only the one that had them to begin with.
    assert sorted(seen["written"][-1]) == ["a", "b", "c"]


def test_plain_save_still_touches_only_the_current_recording(bench):
    w, seen, _zones = bench
    w._write_into_files()
    assert seen["applied"] in ([], [["a"]]), seen["applied"]
    # Only "a" had zones, so only "a" is written: the old behaviour, intact.
    assert seen["written"][-1] == ["a"]


def test_nothing_is_written_when_no_recording_is_loaded(bench):
    w, seen, _zones = bench
    w.set_recordings([], 0)
    w._write_into_all_files()
    assert not seen["written"]
