"""Multi-Start must agree with the Record buttons it claims to speak for.

``any_ready_to_start`` is not computed from box facts. It is read off the
per-box ``record_button.isEnabled()``, because Multi-Start clicks exactly
those buttons and the widget's own logic is the authority on whether one can
be pressed.

That makes the gate order load-bearing, and ``_apply_ui_state`` runs in this
order:

    _apply_mode_buttons        <- sets Multi-Start from state[...]
    widget.apply_global_state  <- the widget re-decides its Record button
    _apply_test_tracking_gates <- disables Record on previewing boxes

so Multi-Start is set from a reading of the Record buttons taken before the
two steps that change them. Start a test preview and Record greys out at
once while Multi-Start stays lit, offering to press a button that is now
disabled; stop the preview and Multi-Start stays grey while Record is live
again. It corrects itself on the next refresh, which is why it reads as the
button lagging rather than as a rule being wrong.
"""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets

from source.gui.base import MainWindowBase


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Box:
    """The parts of a box widget the gates actually touch."""

    def __init__(self):
        self.record_button = QtWidgets.QPushButton()
        self.record_button.setEnabled(True)
        self.connect_button = QtWidgets.QPushButton()
        self.upload_button = QtWidgets.QPushButton()
        self.disconnect_button = QtWidgets.QPushButton()
        self.task_uploaded = True
        self.is_running = False

    def apply_global_state(self, **_kw):
        """The widget re-decides its own Record button on every pass."""
        self.record_button.setEnabled(True)


def _window(previewing: bool):
    w = MainWindowBase.__new__(MainWindowBase)
    box = _Box()
    w._box_widgets = {1: box}
    w._setup_widget_for = lambda bid: box
    w._box_in_test_preview = lambda bid: previewing
    w._box_indicator_surfaces = lambda sid, widget: ()
    w._box_camera_id_text = lambda sid: ""
    w.video_manager = None
    w.tracking_zones = {}
    w.update_metadata_button_states = lambda: None
    w._apply_common_ui_state = lambda state: {"any_idle_connected": True}

    # Stand in for the mode hook: operant and maze both gate Multi-Start on
    # state["any_ready_to_start"] here.
    w.start_button = QtWidgets.QPushButton()
    w._apply_mode_buttons = lambda state, flags: w.start_button.setEnabled(
        bool(state.get("any_ready_to_start", False)))
    return w, box


def _state(ready: bool):
    return {"box_states": {1: {"framework_running": False,
                               "recording_video": False,
                               "recording_data": False}},
            "any_ready_to_start": ready}


def test_multi_start_is_off_while_a_box_is_previewing(qapp):
    """The regression. Record is disabled by the preview gate in this very
    pass, so offering Multi-Start is offering a button that cannot work."""
    w, box = _window(previewing=True)
    # The reading taken before this pass: Record was enabled, so was ready.
    w._apply_ui_state(_state(ready=True))
    assert not box.record_button.isEnabled(), "preview gate did not fire"
    assert not w.start_button.isEnabled(), (
        "Multi-Start offers to press a Record button this pass just disabled")


def test_multi_start_comes_back_when_the_preview_stops(qapp):
    """And the other direction: the stale reading says not-ready while the
    widget has just re-enabled Record."""
    w, box = _window(previewing=False)
    w._apply_ui_state(_state(ready=False))
    assert box.record_button.isEnabled()
    assert w.start_button.isEnabled(), (
        "Multi-Start still grey while every Record button is live")


def test_multi_start_follows_record_in_the_ordinary_case(qapp):
    """Nothing previewing and the reading agrees: unchanged behaviour."""
    w, box = _window(previewing=False)
    w._apply_ui_state(_state(ready=True))
    assert box.record_button.isEnabled()
    assert w.start_button.isEnabled()
