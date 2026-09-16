"""The picture keeps its shape, whatever shape the tile is.

Two aspect settings exist and only one of them is visible to the operator.
``MainWindowBase._tile_aspect_mode`` decides how the frame is prepared;
``VideoTile._aspect_mode`` decides how it is finally PAINTED, and the tile
paints last, into whatever rect its own mode says. A tile left on stretch
squashed the picture no matter what the window had decided, which is why
setting the window-level mode to "preserve" changed nothing on screen.

Measured from the pixels: render the tile, find the drawn image, and check its
proportions against the source.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtGui, QtWidgets

pytest.importorskip("cv2")

from source.gui.widgets.video_tile import VideoTile  # noqa: E402
from source.tests.qt_dispose import WidgetBin  # noqa: E402

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()

SRC_W, SRC_H = 320, 240          # 4:3


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _frame():
    """Solid white, so the drawn region is exactly the non-black pixels."""
    return np.full((SRC_H, SRC_W, 3), 255, np.uint8)


def _painted_size(tile):
    """(w, h) of the drawn image inside the tile, from the rendered pixels."""
    img = tile.grab().toImage().convertToFormat(
        QtGui.QImage.Format.Format_RGB888)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(
        img.height(), img.bytesPerLine() // 3, 3)[:, :img.width()]
    bright = np.all(a > 200, axis=2)
    ys, xs = np.nonzero(bright)
    assert xs.size, "nothing was painted"
    return (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


@pytest.mark.parametrize("tile_w,tile_h", [
    (320, 240),     # same shape
    (640, 480),     # same shape, enlarged
    (900, 300),     # much wider than the camera
    (300, 900),     # much taller
    (1200, 700),    # enlarged and a different shape
    (160, 400),     # small and tall
])
def test_the_picture_keeps_its_shape_in_any_tile(qapp, tile_w, tile_h):
    tile = _BIN.add(VideoTile())
    tile.resize(tile_w, tile_h)
    tile.update_frame(_frame())
    qapp.processEvents()
    w, h = _painted_size(tile)
    assert w / h == pytest.approx(SRC_W / SRC_H, rel=0.02), (
        f"tile {tile_w}x{tile_h}: painted {w}x{h} = {w/h:.3f}, "
        f"source is {SRC_W/SRC_H:.3f}, the picture is distorted")
    # And it fills the tile in at least one axis, so it is not just small.
    assert w == pytest.approx(tile_w, abs=2) or h == pytest.approx(tile_h, abs=2)


def test_the_default_is_letterbox_not_stretch(qapp):
    """The default is what every construction site gets; ``setup_widget``
    builds its tile with no argument at all."""
    tile = _BIN.add(VideoTile())
    assert tile._aspect_mode == VideoTile.ASPECT_LETTERBOX


def test_an_unknown_mode_falls_back_to_letterbox(qapp):
    """Black bars are always honest; a squashed picture is not."""
    tile = _BIN.add(VideoTile(aspect_mode="nonsense"))
    assert tile._aspect_mode == VideoTile.ASPECT_LETTERBOX


def test_stretch_is_still_available_when_asked_for(qapp):
    """Removing the option would be a different bug; it stays reachable, it
    is just not what anything defaults to."""
    tile = _BIN.add(VideoTile(aspect_mode=VideoTile.ASPECT_STRETCH))
    tile.resize(900, 300)
    tile.update_frame(_frame())
    qapp.processEvents()
    w, h = _painted_size(tile)
    assert w / h == pytest.approx(3.0, rel=0.05)
