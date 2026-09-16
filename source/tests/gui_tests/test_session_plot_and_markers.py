"""Session Plot lifecycle + annotation marker size.

Three faults on 2026-09-09, all of which passed every existing test:

* the GUI died building the Session Plot with four boxes running, after
  registering boxes 1-3 and before box 4, with no Python traceback and six
  ``QBasicTimer::start: Timers cannot be started from another thread``
  warnings. A native crash leaves nothing to assert on afterwards, so the
  guard has to be the Qt warning itself, caught while the window is driven;
* a block's minimum height was computed once at construction, while the
  trigger lane is revealed later, so three lanes were squeezed into a floor
  sized for two;
* the marker-size setting was stored, saved and restored, and the renderer
  drew with a hardcoded 4 the whole time.
"""
from __future__ import annotations

import numpy as np
import pytest

from PySide6 import QtCore, QtWidgets

pytest.importorskip("pyqtgraph")
cv2 = pytest.importorskip("cv2")

from source.gui import plotting as P                       # noqa: E402
from source.video.trigger_engine import RuleState, TriggerFrame   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _Box:
    """The minimum a setup widget must offer ``update_boxes``."""

    def __init__(self, n):
        self.setup_number = n
        self.pycboard = None            # no board: no sm_info, no consumers
        self.framework_running = True
        self.recording_data = False


def _destroy(widget, qapp):
    """Free the C++ side. ``processEvents`` alone does not run a queued
    DeferredDelete, so the widget would survive and trip the leak guard."""
    widget.close()
    widget.deleteLater()
    qapp.processEvents()
    qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture
def window(qapp):
    w = P.UniversalPlotWindow()
    try:
        yield w
    finally:
        _destroy(w, qapp)


# ── the crash ──────────────────────────────────────────────────────────────

def test_every_running_box_gets_a_block(window):
    """It died after box 3. Four boxes must produce four blocks."""
    window.update_boxes([_Box(n) for n in (1, 2, 3, 4)])
    assert sorted(window.box_plots) == [1, 2, 3, 4]
    assert sorted(window.box_blocks) == [1, 2, 3, 4]


def test_reopening_with_the_same_boxes_does_not_duplicate(window):
    """The Plot button can be pressed repeatedly; blocks must not stack up."""
    boxes = [_Box(n) for n in (1, 2, 3, 4)]
    window.update_boxes(boxes)
    window.update_boxes(boxes)
    assert sorted(window.box_plots) == [1, 2, 3, 4]


def test_a_stopped_box_loses_its_block(window):
    window.update_boxes([_Box(n) for n in (1, 2, 3, 4)])
    window.update_boxes([_Box(n) for n in (1, 3)])
    assert sorted(window.box_plots) == [1, 3]


def test_driving_the_plot_starts_no_timer_off_the_gui_thread(window, qapp):
    """The guard for the crash itself.

    A cross-thread timer start is undefined behaviour, and the process that
    died had emitted exactly this warning six times. Qt reports it through
    the message handler rather than by raising, so the only way to fail a
    test on it is to listen.
    """
    seen: list[str] = []

    def handler(_mode, _ctx, message):
        seen.append(message)

    previous = QtCore.qInstallMessageHandler(handler)
    try:
        boxes = [_Box(n) for n in (1, 2, 3, 4)]
        window.update_boxes(boxes)
        for n in (1, 2, 3, 4):
            window.push_trigger_frame(n, TriggerFrame(
                setup_id=n, cam_frame_id=1,
                states=[RuleState(id="r", name="reward",
                                  active=True, fired=False)]))
        window.update_all_plots()
        qapp.processEvents()
        window.update_boxes([])
        qapp.processEvents()
    finally:
        QtCore.qInstallMessageHandler(previous)

    offending = [m for m in seen if "another thread" in m.lower()]
    assert not offending, f"Qt objects touched off the GUI thread: {offending}"


# ── block height follows the lanes it holds ────────────────────────────────

def test_the_trigger_lane_raises_the_block_floor(window):
    """Revealed later than the block was built, so the floor must be redone."""
    window.update_boxes([_Box(1)])
    before = window.box_blocks[1].minimumHeight()
    assert before == P.MIN_BOX_PLOT_H

    window.push_trigger_frame(1, TriggerFrame(
        setup_id=1, cam_frame_id=1,
        states=[RuleState(id="r", name="reward", active=True, fired=False)]))

    # isHidden, not isVisible: the window is never shown in a test, and a
    # child of an unshown window is never "visible".
    assert not window.box_triggers[1].isHidden()
    after = window.box_blocks[1].minimumHeight()
    assert after == before + P.TRIGGER_LANE_H, (
        "a third lane appeared and the block kept its two-lane floor")


def test_a_frame_without_rule_states_keeps_the_block_compact(window):
    """Boxes that never trigger must not pay for a lane they do not show."""
    window.update_boxes([_Box(1)])
    window.push_trigger_frame(1, TriggerFrame(setup_id=1, cam_frame_id=1,
                                              states=[]))
    assert window.box_triggers[1].isHidden()
    assert window.box_blocks[1].minimumHeight() == P.MIN_BOX_PLOT_H


def test_the_floor_is_not_applied_twice(window):
    """Later trigger frames must not keep growing the floor."""
    window.update_boxes([_Box(1)])
    frame = TriggerFrame(setup_id=1, cam_frame_id=1,
                         states=[RuleState(id="r", name="reward",
                                           active=True, fired=False)])
    window.push_trigger_frame(1, frame)
    once = window.box_blocks[1].minimumHeight()
    for _ in range(5):
        window.push_trigger_frame(1, frame)
    assert window.box_blocks[1].minimumHeight() == once


class _SM:
    """The parts of ``sm_info`` the three lanes read."""

    def __init__(self, n_states, n_events, analog=None):
        self.states = {f"state_{i}": i for i in range(n_states)}
        self.events = {f"event_{i}": 100 + i for i in range(n_events)}
        self.analog_inputs = analog or {}


def _floor_for(qapp, n_states, n_events, analog=None):
    task = P.TaskPlot()
    try:
        task.set_state_machine(_SM(n_states, n_events, analog))
        return P.UniversalPlotWindow._block_floor(task)
    finally:
        _destroy(task, qapp)


def test_a_task_with_more_states_gets_a_taller_block(qapp):
    """The whole complaint: a fixed height cannot hold arbitrary content."""
    small = _floor_for(qapp, n_states=3, n_events=3)
    large = _floor_for(qapp, n_states=24, n_events=3)
    assert large > small, "block height did not follow the number of states"
    # The states lane alone must fit its own rows, which is the property that
    # stops the tick labels colliding. (The small case is clamped up to the
    # base floor, so the DIFFERENCE understates the growth.)
    assert large >= 24 * P.PLOT_ROW_H


def test_more_events_also_grow_the_block(qapp):
    few = _floor_for(qapp, n_states=4, n_events=2)
    many = _floor_for(qapp, n_states=4, n_events=30)
    assert many > few


def test_growth_is_proportional_not_a_single_step(qapp):
    """Two steps up in state count must both raise the floor, so the
    behaviour is fluid rather than one 'big task' bucket."""
    a = _floor_for(qapp, 10, 4)
    b = _floor_for(qapp, 20, 4)
    c = _floor_for(qapp, 30, 4)
    assert a < b < c


def test_a_tiny_task_is_not_squashed(qapp):
    """Few rows still need axis, title and padding."""
    assert _floor_for(qapp, 1, 1) >= P.MIN_BOX_PLOT_H


def test_an_analog_lane_adds_its_own_height(qapp):
    plain = _floor_for(qapp, 12, 12)
    analog = _floor_for(qapp, 12, 12,
                        analog={1: {"name": "photometry", "fs": 100,
                                    "plot": True}})
    assert analog == plain + P.ANALOG_LANE_H


def test_a_box_with_no_state_machine_keeps_the_base_floor(window):
    """Counts are unknown until the board reports them; the block must look
    empty rather than broken."""
    window.update_boxes([_Box(1)])
    assert window.box_blocks[1].minimumHeight() == P.MIN_BOX_PLOT_H


def test_blocks_still_expand_rather_than_being_pinned(window):
    """The floor must stay a floor: blocks grow into a tall window."""
    window.update_boxes([_Box(1)])
    block = window.box_blocks[1]
    assert block.maximumHeight() > P.MIN_BOX_PLOT_H * 4
    assert (block.sizePolicy().verticalPolicy()
            == QtWidgets.QSizePolicy.Policy.Expanding)


# ── marker size actually reaches the renderer ──────────────────────────────

def _drawn_radius(base_radius):
    """Paint one keypoint through the real draw path; return its radius."""
    from source.gui.base import MainWindowBase

    class _State:
        pose = [(50.0, 50.0, 1.0)]
        body_parts = ["snout"]
        confidence_threshold = 0.5
        skeleton = ()

    class _Host:
        # Borrowed wholesale, as test_overlay_lands_on_the_animal does: the
        # point is to exercise the REAL drawing path, not a reimplementation
        # of it that could agree with a bug.
        _DLC_COLORS = MainWindowBase._DLC_COLORS
        _draw_pose_layer = MainWindowBase._draw_pose_layer
        _draw_keypoints_cython = MainWindowBase._draw_keypoints_cython
        _draw_keypoints_python = MainWindowBase._draw_keypoints_python
        _draw_pose_figure = MainWindowBase._draw_pose_figure
        # staticmethod on purpose: a plain assignment unwraps the descriptor
        # and every call would arrive one argument over.
        _to_tile = staticmethod(MainWindowBase._to_tile)
        MARKER_SIZE_DEFAULT = 4

    frame = np.zeros((100, 100, 3), np.uint8)
    _Host()._draw_pose_layer(frame, _State(), 1.0, 1.0, 1.0,
                             base_radius=base_radius)
    ys, xs = np.nonzero(frame.any(axis=2))
    assert xs.size, "nothing was drawn"
    return max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0


def test_a_bigger_marker_size_draws_a_bigger_marker():
    """The renderer used a literal 4, so this was the whole bug."""
    small = _drawn_radius(3)
    large = _drawn_radius(12)
    assert large > small * 2, (
        f"radius did not follow the setting (3 -> {small}, 12 -> {large})")


def test_marker_size_defaults_when_the_box_has_not_chosen_one():
    assert _drawn_radius(None) == pytest.approx(_drawn_radius(4), abs=1.0)


# ── marker size survives a project round trip ──────────────────────────────

def test_marker_size_round_trips_through_the_project():
    from source.config.experiment import (MARKER_SIZE_KEY, Config, _apply_ui,
                                          _read_ui)

    class _Host:
        pass

    src = _Host()
    src._tracking_dialog_globals = {}
    src._marker_sizes = {1: 9, 4: 15}

    cfg = Config()
    _read_ui(src, cfg)
    # JSON object keys are strings; an int-keyed map would lose its entries.
    assert cfg.ui.dialog_overrides[MARKER_SIZE_KEY] == {"1": 9, "4": 15}

    dst = _Host()
    _apply_ui(cfg, dst)
    assert dst._marker_sizes == {1: 9, 4: 15}


def test_an_untouched_project_carries_no_marker_key():
    """Only what someone actually chose belongs in the file."""
    from source.config.experiment import MARKER_SIZE_KEY, Config, _read_ui

    class _Host:
        pass

    host = _Host()
    host._tracking_dialog_globals = {MARKER_SIZE_KEY: {"1": 9}}
    host._marker_sizes = {}

    cfg = Config()
    _read_ui(host, cfg)
    assert MARKER_SIZE_KEY not in cfg.ui.dialog_overrides


def test_a_corrupt_marker_entry_does_not_break_loading():
    from source.config.experiment import MARKER_SIZE_KEY, Config, _apply_ui

    cfg = Config()
    cfg.ui.dialog_overrides = {MARKER_SIZE_KEY: {"1": "huge", "2": 7,
                                                 "bad": 5, "3": 999}}

    class _Host:
        pass

    host = _Host()
    _apply_ui(cfg, host)
    assert host._marker_sizes[2] == 7
    assert host._marker_sizes[3] == 20      # clamped, not dropped
    assert 1 not in host._marker_sizes      # unparseable value skipped


# ── the control is annotation-only ─────────────────────────────────────────

def test_marker_buttons_stay_live_when_a_box_has_no_zones(qapp):
    """Marker size is how the overlay is drawn, so it must not be gated on
    zones, a model or inference."""
    from source.gui.widgets.zone_adjust_row import ZoneAdjustRow

    row = ZoneAdjustRow(setup_id=1)
    row.set_zones_enabled(False)
    assert not row.zone_step_edit.isEnabled()
    assert row.marker_bigger_btn.isEnabled()
    assert row.marker_smaller_btn.isEnabled()
    _destroy(row, qapp)


def test_marker_buttons_emit_the_new_size(qapp):
    from source.gui.widgets.zone_adjust_row import ZoneAdjustRow

    row = ZoneAdjustRow(setup_id=3)
    got = []
    row.marker_size_changed.connect(lambda sid, size: got.append((sid, size)))
    row.marker_bigger_btn.click()
    assert got == [(3, ZoneAdjustRow.MARKER_DEFAULT + 1)]
    _destroy(row, qapp)


def test_restoring_a_size_does_not_look_like_an_operator_edit(qapp):
    """Loading a project must not mark it dirty."""
    from source.gui.widgets.zone_adjust_row import ZoneAdjustRow

    row = ZoneAdjustRow(setup_id=1)
    got = []
    row.marker_size_changed.connect(lambda *a: got.append(a))
    row.set_marker_size(11)
    assert row.marker_size() == 11
    assert got == []
    _destroy(row, qapp)


def test_the_size_is_clamped_at_both_ends(qapp):
    from source.gui.widgets.zone_adjust_row import ZoneAdjustRow

    row = ZoneAdjustRow(setup_id=1)
    row.set_marker_size(999)
    assert row.marker_size() == ZoneAdjustRow.MARKER_MAX
    assert not row.marker_bigger_btn.isEnabled()
    row.set_marker_size(-4)
    assert row.marker_size() == ZoneAdjustRow.MARKER_MIN
    assert not row.marker_smaller_btn.isEnabled()
    _destroy(row, qapp)
