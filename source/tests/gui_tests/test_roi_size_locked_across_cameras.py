"""One ROI size across every box, on every camera.

All boxes feed one pose model and the model takes ONE input shape:
``canonical_shape`` picks the widest width and the tallest height across the
boxes and letterboxes every smaller one into it. Unequal ROIs therefore make
every box pay the largest box's inference cost, while the padded boxes have
the animal filling less of the model input.

Within a single camera the first ROI already locked the rest. What was missing
was across cameras: ``ROISegmentationDialog`` is opened once per camera, and
each editor locked to its own first ROI, so a multi-camera rig ended up with a
different region size per camera.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from source.gui.widgets.frame_display import ROIDrawCanvas
from source.video.tracking.letterbox import canonical_shape


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def canvas(qapp):
    c = ROIDrawCanvas()
    c.set_frame(np.zeros((480, 640, 3), np.uint8))
    try:
        yield c
    finally:
        c.deleteLater()
        qapp.processEvents()
        qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


# ── why it matters, stated against the real function ─────────────────────

def test_unequal_rois_cost_every_box_the_largest_shape():
    """The mechanism, so the rule below is not folklore."""
    equal = canonical_shape([(320, 240), (320, 240), (320, 240)])
    mixed = canonical_shape([(320, 240), (640, 480), (300, 200)])
    assert equal == (320, 240)
    assert mixed == (640, 480), (
        "the model input grows to the biggest box, and the others are padded")


# ── one camera: already worked, keep it working ──────────────────────────

def test_the_first_roi_locks_the_size_for_the_rest(canvas):
    canvas.set_rois({1: (10, 10, 200, 150)})
    assert canvas.locked_size() == (200, 150)


def test_clearing_every_roi_unlocks_the_size(canvas):
    canvas.set_rois({1: (10, 10, 200, 150)})
    canvas.set_rois({})
    assert canvas.locked_size() is None


# ── across cameras: the gap ──────────────────────────────────────────────

def test_a_size_from_another_camera_is_adopted(canvas):
    """The second camera's editor must open already fixed to the first's."""
    assert canvas.set_external_locked_size((200, 150)) is True
    assert canvas.locked_size() == (200, 150)


def test_the_adopted_size_outranks_this_camera_s_own_first_roi(canvas):
    """Otherwise each camera drifts back to its own first region, which is
    exactly the behaviour being fixed."""
    canvas.set_external_locked_size((200, 150))
    canvas.set_rois({2: (0, 0, 500, 400)})
    assert canvas.locked_size() == (200, 150)


def test_a_size_too_big_for_this_camera_is_refused(canvas):
    """640x480 here. Silently shrinking would produce the mismatch this
    exists to prevent, so it is refused and the caller reports it."""
    assert canvas.set_external_locked_size((800, 600)) is False
    assert canvas.locked_size() is None


def test_a_refused_size_leaves_the_camera_free_to_set_its_own(canvas):
    canvas.set_external_locked_size((800, 600))
    canvas.set_rois({1: (0, 0, 320, 240)})
    assert canvas.locked_size() == (320, 240)


def test_a_size_that_exactly_fills_the_frame_is_allowed(canvas):
    assert canvas.set_external_locked_size((640, 480)) is True


def test_clearing_the_external_lock_falls_back_to_this_camera(canvas):
    canvas.set_rois({1: (0, 0, 320, 240)})
    canvas.set_external_locked_size((200, 150))
    assert canvas.locked_size() == (200, 150)
    canvas.set_external_locked_size(None)
    assert canvas.locked_size() == (320, 240)


def test_a_nonsense_size_is_rejected(canvas):
    assert canvas.set_external_locked_size((0, 100)) is False
    assert canvas.set_external_locked_size((-5, -5)) is False


# ── the end state the rig needs ──────────────────────────────────────────

def test_three_cameras_end_up_with_one_shape(qapp):
    """What the chained lock buys: one input shape, so no box is padded."""
    sizes = []
    lock = None
    for frame_wh in ((640, 480), (1280, 720), (800, 600)):
        c = ROIDrawCanvas()
        c.set_frame(np.zeros((frame_wh[1], frame_wh[0], 3), np.uint8))
        if lock is not None:
            c.set_external_locked_size(lock)
        if c.locked_size() is None:
            c.set_rois({1: (0, 0, 320, 240)})     # the first camera sets it
        sizes.append(c.locked_size())
        lock = c.locked_size()
        c.deleteLater()
    qapp.processEvents()
    qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    assert len(set(sizes)) == 1, f"cameras disagreed on the region size: {sizes}"
    assert canonical_shape(sizes) == sizes[0], (
        "with one size the model input is that size, and nothing is padded")
