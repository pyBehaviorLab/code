"""Central robustness behaviour + the dry-run safety net.

  - Changing a task notifies the main window so master buttons re-gate.
  - Clear Meta re-enables when idle (1 Hz idle tick + authoritative scan).
  - Re-applying a stats config does NOT wipe other boxes' table rows.
  - No-subject runs are captured to data/temp/ and recoverable into the tree.
"""
from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6 import QtWidgets

from source.tests.qt_dispose import WidgetBin

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain_widget_bin():
    """Destroy the widgets this file's helpers built (see qt_dispose)."""
    yield
    _BIN.drain()



@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _operant_with_box(qapp):
    from source.gui.operant import MainWindow
    w = _BIN.add(MainWindow())
    w.add_setup()
    return w, w._setup_widget_for(1)


# ---------------------------------------------------------------------------
# Stats table is data-preserving across re-applies (per-box reset only)
# ---------------------------------------------------------------------------
def test_setuptable_preserves_rows_on_same_columns(qapp):
    from source.stats.canvas import StatisticsTableWidget
    tw = _BIN.add(StatisticsTableWidget())
    cols = [{"key": "trials", "label": "Trials"},
            {"key": "rew", "label": "Reward"}]
    tw.setupTable(cols, 2)
    tw.item(0, 0).setText("42")     # box 1 mid-run
    tw.item(1, 0).setText("7")      # box 2 mid-run

    # Another box's upload re-applies the SAME config, must NOT wipe.
    tw.setupTable(cols, 2)
    assert tw.item(0, 0).text() == "42"
    assert tw.item(1, 0).text() == "7"


def test_setuptable_grows_for_new_box_without_wiping(qapp):
    from source.stats.canvas import StatisticsTableWidget
    tw = _BIN.add(StatisticsTableWidget())
    cols = [{"key": "a", "label": "A"}]
    tw.setupTable(cols, 1)
    tw.item(0, 0).setText("keep")
    tw.setupTable(cols, 2)          # a box was added
    assert tw.rowCount() == 2
    assert tw.item(0, 0).text() == "keep"   # existing box preserved


def test_setupplots_does_not_rebuild_on_same_config(qapp):
    """Re-applying the same plot config (another box's upload) must NOT
    delete the existing per-box plot widgets."""
    from source.stats.canvas import StatisticsPlotWidget
    pw = _BIN.add(StatisticsPlotWidget())
    cfg = {"plots": [{"name": "trials", "type": "circular", "title": "Trials"}]}
    pw.setupPlots(cfg, 2)
    widgets_after_first = dict(pw.plot_widgets)
    assert widgets_after_first, "first setupPlots should build the plots"

    pw.setupPlots(cfg, 2)           # same config again (upload of same task)
    # Same widget objects preserved (not deleted + recreated).
    assert pw.plot_widgets is not None
    assert set(pw.plot_widgets.keys()) == set(widgets_after_first.keys())
    for k in widgets_after_first:
        assert pw.plot_widgets[k] is widgets_after_first[k], \
            "plot widget was rebuilt, live stats would be wiped"


# ---------------------------------------------------------------------------
# Task change notifies the main window so master buttons re-gate
# ---------------------------------------------------------------------------
def test_ontaskchanged_notifies_main_window(qapp):
    w, bw = _operant_with_box(qapp)
    calls = []
    w.refresh_ui_state = lambda *a, **k: calls.append(1)
    bw.on_task_changed()
    assert calls, "on_task_changed must call main_window.refresh_ui_state"


# ---------------------------------------------------------------------------
# Clear Meta gating: disabled while running, re-enabled once idle
# ---------------------------------------------------------------------------
def test_clear_meta_reenables_when_idle(qapp):
    w, bw = _operant_with_box(qapp)
    bw.subject_id_edit.setText("M01")

    bw.framework_running = True
    w.update_metadata_button_states()
    assert not w.clear_metadata_button.isEnabled(), \
        "Clear Meta must be locked while a box runs"

    # The 1 Hz idle-tick hook re-gates once nothing is running.
    bw.framework_running = False
    w._idle_button_refresh()
    assert w.clear_metadata_button.isEnabled(), \
        "Clear Meta must re-enable once idle"


# ---------------------------------------------------------------------------
# Safety net, flat temp files under data/temp/ (overwritten each dry run):
# data/temp/Box<N>.tsv, video_data_Box<N>.txt, video_Box<N>.mp4
# ---------------------------------------------------------------------------
def test_temp_tsv_path_is_flat_per_box_file(qapp):
    w, bw = _operant_with_box(qapp)
    p = bw.temp_tsv_path()
    assert p.name == "Box1.tsv"
    # Lives under data/temp/, a single flat file (no nested per-box folders).
    assert p.parent.name == "temp"
    assert p.parent.parent.name == "data"
    # Sibling dry-run artifacts share the same temp dir + Box<N> stem.
    assert bw.temp_video_data_path().name == "video_data_Box1.txt"
    assert bw.temp_video_path().name == "video_Box1.mp4"
    assert bw.temp_video_data_path().parent == p.parent
    assert bw.temp_video_path().parent == p.parent


def test_sink_writes_tracking_text_without_a_video_recorder():
    """Dry-run safety net: the per-frame _video_data row is written whenever
    a tracking writer is registered, even with recorder=None (no video)."""
    from datetime import datetime

    import numpy as np

    from source.video.framebus.recorder_sink import RecorderSink
    from source.video.framebus.types import BoxFrame

    sink = RecorderSink()

    class _TW:
        def __init__(self):
            self.rows = 0

        def write_frame(self, **kw):
            self.rows += 1

        def close(self):
            pass

    tw = _TW()
    sink.start_recording(1, recorder=None, tracking_writer=tw)
    rec_start = sink._rec_start_host_ns[1]
    frame = BoxFrame(
        image=np.zeros((4, 4, 3), np.uint8), setup_id=1, cam_frame_id=5,
        camera_id=0, capture_host_ns=rec_start + 1_000_000_000,
        capture_wall=datetime(2026, 6, 10),
        is_shared_camera=False,
    )
    sink.process(frame)
    assert tw.rows == 1, "tracking row must be written even with no recorder"
    sink.stop_recording(1)   # closes the writer cleanly
