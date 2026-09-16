"""The maze tile is the camera's shape at EVERY size, expansion included.

Three things had to be true and only the third one held:

1. maze declares ``_tile_aspect_mode = "preserve"`` at class level, and the
   base assigned ``self._tile_aspect_mode = "stretch"`` in ``_setup_pipeline``
   - an instance attribute shadows a subclass's class attribute, so every maze
   window was built in stretch mode;
2. the tile has to be GIVEN a cell of the right shape. A size policy with
   ``heightForWidth`` does not do it: Qt largely ignores height-for-width for a
   widget sitting in a horizontal row, and the maze tile sits in one. Measured
   on the real widget with the policy set and answering correctly, a 640x480
   camera still came out at ratios from 0.68 to 2.60 as the window was resized;
3. the tile paints letterboxed, which it already did.

``AspectBox`` supplies the second. It takes the cell the layout gives it and
places the tile inside as the largest rectangle of the source ratio, so the
tile is exactly the camera's shape, grows with the window in proportion, and
its own letterbox has nothing left to letterbox.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from source.gui.widgets.video_tile import AspectBox, VideoTile

WINDOWS = ((900, 500), (1400, 500), (1400, 1000), (1900, 700), (1200, 1400))


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@contextmanager
def owning(app, widget):
    """Dispose this widget before the test returns.

    Inside the test body rather than in a fixture: the suite's leak check runs
    at the start of teardown, before fixture finalisers, so a fixture that
    tidies up afterwards is already too late. A leaked top-level widget keeps
    its whole tree, and any pipeline it owns, alive for the rest of the run.
    """
    from PySide6 import QtCore

    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()
        # ``processEvents`` does not deliver DeferredDelete, so the widget
        # would still be a live top-level when the leak check looks.
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _ratios(widget, tile, src_w, src_h, app):
    tile.update_frame(np.zeros((src_h, src_w, 3), np.uint8))
    out = []
    for w, h in WINDOWS:
        widget.resize(w, h)
        widget.show()
        app.processEvents()
        g = tile.geometry()
        out.append(g.width() / g.height() if g.height() else 0.0)
    widget.hide()
    return out


@pytest.mark.parametrize("src", [(640, 480), (1280, 720), (480, 640)])
def test_the_maze_tile_holds_the_camera_ratio_at_every_size(app, src):
    """Including when the window is far larger than the camera's own
    resolution, which is the case that was reported."""
    from source.gui.widgets.setup_widget import SetupWidget

    with owning(app, SetupWidget(1)) as sw:
        want = src[0] / src[1]
        for got in _ratios(sw, sw.video_label, *src, app):
            assert got == pytest.approx(want, abs=0.01), (
                f"tile ratio {got:.3f} against camera {want:.3f}")


def test_the_operant_tile_still_fills_its_cell(app):
    """A grid of chambers is read by glancing across it, so the height goes to
    the animals rather than to bars. This must not follow maze."""
    from source.gui.widgets.video_stream import VideoStreamHolder

    with owning(app, VideoStreamHolder(setup_number=1)) as vs:
        ratios = _ratios(vs, vs.videoWidget, 640, 480, app)
    assert any(abs(r - 4 / 3) > 0.1 for r in ratios), (
        "the operant tile should take the shape of its cell, not the camera's")


class TestTheBox:
    def test_with_no_frame_it_simply_fills(self, app):
        """Nothing is known about the camera's shape yet, so there is nothing
        to hold to."""
        with owning(app, AspectBox(VideoTile(lock_widget_aspect=True))) as box:
            box.resize(800, 300)
            box.replace_child()
            assert box.child().geometry().size() == box.size()

    def test_it_centres_what_it_places(self, app):
        tile = VideoTile(lock_widget_aspect=True)
        with owning(app, AspectBox(tile)) as box:
            box.resize(800, 300)
            tile.update_frame(np.zeros((480, 640, 3), np.uint8))
            box.replace_child()
            g = tile.geometry()
            assert g.height() == 300               # limited by height here
            assert g.left() == pytest.approx((800 - g.width()) // 2, abs=1)

    def test_a_camera_change_re_places_the_tile(self, app):
        """The ratio is learned from the frames, so a camera swap has to move
        the tile rather than leave it in the previous camera's shape."""
        tile = VideoTile(lock_widget_aspect=True)
        with owning(app, AspectBox(tile)) as box:
            box.resize(1000, 1000)
            tile.update_frame(np.zeros((480, 640, 3), np.uint8))
            first = tile.geometry()
            tile.update_frame(np.zeros((720, 1280, 3), np.uint8))
            second = tile.geometry()
        assert first != second
        assert second.width() / second.height() == pytest.approx(16 / 9,
                                                                 abs=0.01)

    def test_the_ratio_is_only_re_read_when_it_changes(self, app):
        """``replace_child`` re-runs a layout; doing it on every frame would
        re-lay-out the window thirty times a second for a constant."""
        tile = VideoTile(lock_widget_aspect=True)
        with owning(app, AspectBox(tile)) as box:
            box.resize(900, 600)
            tile.update_frame(np.zeros((480, 640, 3), np.uint8))
            placed = tile.geometry()
            for _ in range(5):
                tile.update_frame(np.zeros((480, 640, 3), np.uint8))
            assert tile.geometry() == placed

    def test_the_child_keeps_working_as_a_label(self, app):
        """Every existing caller still holds the TILE, not the box, so the
        QLabel wrappers have to keep behaving."""
        tile = VideoTile(lock_widget_aspect=True)
        with owning(app, AspectBox(tile)):
            tile.setText("Camera not connected")
            assert tile.text() == "Camera not connected"


def test_an_operant_tile_never_locks_its_shape():
    """The flag is the whole difference between the two modes."""
    assert VideoTile()._lock_widget_aspect is False
    assert VideoTile(lock_widget_aspect=True)._lock_widget_aspect is True
