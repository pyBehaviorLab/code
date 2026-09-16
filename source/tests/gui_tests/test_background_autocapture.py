"""Selecting Blob captures the reference it needs.

Background subtraction without a reference image is not a thing, so choosing
Blob IS the request for one. Taking it at that moment removes the step an
operator otherwise has to know about, and the warning they otherwise have to
read and act on before the box works at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from source.gui.widgets.tracking_panel import TrackingSettingsPanel
from source.tests.qt_dispose import WidgetBin

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()


def _panel(box_ids=(1,)):
    p = _BIN.add(TrackingSettingsPanel(box_ids=list(box_ids)))
    # The trigger tab builds state the mode handler reads; the panel is only
    # half-constructed without it.
    tab = p.build_event_trigger_widget()
    tab.setParent(p)
    return p


def _wire(p, *, has_bg, running=False, camera=True):
    """Give the panel just enough surface to make the decision."""
    taken = []
    p.get_frame_callbacks = (
        {1: lambda: np.full((30, 40, 3), 200, np.uint8)} if camera else {})
    p._background_for_box = lambda sid: ("box1.png" if has_bg else None)
    p._box_is_running = lambda sid: running
    p._take_background = lambda sid, silent=False: (taken.append(sid) or True)
    for cb in p.box_checkboxes.values():
        cb.setChecked(True)
    # The panel is built with Blob already selected, so selecting a box can
    # legitimately capture right here. Reset so each test observes only what
    # ITS mode switch caused.
    taken.clear()
    p._bg_autocapture_tried.clear()
    return taken


def test_selecting_blob_captures_a_missing_background(mock_qapplication):
    """Blob is the panel's DEFAULT chip, so the switch that exercises this is
    coming back to it: away to Simple (which needs no reference), then back."""
    p = _panel()
    taken = _wire(p, has_bg=False)
    p.mode_simple.setChecked(True)
    assert taken == [], "Simple needs no reference and must not take one"
    p.mode_blob.setChecked(True)
    assert taken == [1], "Blob was selected with no reference and none was taken"


def test_an_existing_background_is_not_replaced(mock_qapplication):
    """Silently overwriting a reference someone tuned against would be worse
    than the missing-file case this exists to fix."""
    p = _panel()
    taken = _wire(p, has_bg=True)
    p.mode_blob.setChecked(True)   # fires _on_mode_changed via toggled
    assert taken == []


def test_a_running_box_is_not_touched(mock_qapplication):
    p = _panel()
    taken = _wire(p, has_bg=False, running=True)
    p.mode_blob.setChecked(True)   # fires _on_mode_changed via toggled
    assert taken == []


def test_no_camera_means_nothing_to_capture(mock_qapplication):
    p = _panel()
    taken = _wire(p, has_bg=False, camera=False)
    p.mode_blob.setChecked(True)   # fires _on_mode_changed via toggled
    assert taken == []


def test_simple_mode_does_not_capture(mock_qapplication):
    """Simple is background-FREE, capturing one would be pure ceremony."""
    p = _panel()
    taken = _wire(p, has_bg=False)
    p.mode_simple.setChecked(True)

    assert taken == []


def test_pose_modes_do_not_capture(mock_qapplication):
    p = _panel()
    taken = _wire(p, has_bg=False)
    p.mode_dlc.setChecked(True)

    assert taken == []


def test_a_config_load_does_not_trigger_capture(mock_qapplication):
    """Replaying a saved config must not start writing files."""
    p = _panel()
    taken = _wire(p, has_bg=False)
    p._loading = True
    p.mode_blob.setChecked(True)   # fires _on_mode_changed via toggled
    assert taken == []


def test_an_unselected_box_is_skipped(mock_qapplication):
    p = _panel(box_ids=(1, 2))
    taken = []
    p.get_frame_callbacks = {
        1: lambda: np.zeros((30, 40, 3), np.uint8),
        2: lambda: np.zeros((30, 40, 3), np.uint8),
    }
    p._background_for_box = lambda sid: None
    p._box_is_running = lambda sid: False
    p._take_background = lambda sid, silent=False: (taken.append(sid) or True)
    for bid, cb in p.box_checkboxes.items():
        cb.setChecked(bid == 2)
    p.mode_blob.setChecked(True)   # fires _on_mode_changed via toggled
    assert taken == [2]
