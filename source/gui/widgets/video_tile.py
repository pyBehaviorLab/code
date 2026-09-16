"""Video tile widget, QLabel-based, with numpy fast path.

QLabel-based to avoid the Windows native-window recreation that
QOpenGLWidget triggers on first creation. Keeps:
  - ``update_frame(numpy_bgr_or_grayscale)`` zero-copy numpy → QImage
    via ``Format_BGR888`` (no cv2.cvtColor).
  - ``setPixmap``/``setText``/``setAlignment`` wrappers so call sites
    use it as a drop-in QLabel.
  - ``aspect_mode`` (stretch / letterbox) honored at paint time via
    QPainter.

Class name ``VideoTile`` is a stable public symbol; the old "GL" suffix no longer
reflects the implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class VideoTile(QtWidgets.QLabel):
    """Per-tile video display with zero-copy numpy fast path.

    Drop-in replacement for ``QtWidgets.QLabel`` plus an ``update_frame``
    method that takes a raw BGR numpy array. Internally builds a
    ``QImage`` with ``Format_BGR888`` (no colour conversion), wraps it
    in a ``QPixmap``, paints scaled by the active aspect_mode.
    """

    PLACEHOLDER_TEXT = "No Video"

    ASPECT_STRETCH = "stretch"
    ASPECT_LETTERBOX = "letterbox"

    #: Letterbox by DEFAULT. The tile paints last, ``paintEvent`` draws the
    #: source pixmap into whatever rect this mode says, so a tile left on
    #: stretch squashes the picture no matter what the window decided, and
    #: ``MainWindowBase._tile_aspect_mode`` never reaches here. Two aspect
    #: settings, and this is the one the operator actually sees.
    #:
    #: Stretching an arena is not cosmetic: a circular field reads as
    #: elliptical, the animal is the wrong shape, and the distortion changes
    #: as the window is resized, indistinguishable from a lens that needs
    #: calibrating.
    def __init__(self,
                 parent: Optional[QtCore.QObject] = None,
                 aspect_mode: str = ASPECT_LETTERBOX,
                 lock_widget_aspect: bool = False) -> None:
        super().__init__(parent)
        #: Ask the LAYOUT for a cell the camera's shape, so the tile grows in
        #: proportion instead of growing into a differently-shaped rectangle
        #: and filling the difference with black. Letterboxing alone keeps the
        #: PICTURE honest, which is not the same thing: an arena tile in a tall
        #: panel is then mostly bars, and the arena stops growing with the
        #: window. Maze asks for this; operant does not, because its grid tiles
        #: must tile.
        self._lock_widget_aspect = bool(lock_widget_aspect)
        #: Source aspect (w / h), learned from the first frame. Until then
        #: there is nothing to lock to and the tile behaves normally.
        self._source_aspect: Optional[float] = None
        # An unrecognised mode falls back to LETTERBOX, not stretch: showing
        # the picture with black bars is always honest, showing it squashed is
        # not, so the safe answer to "I don't know" is the one that cannot
        # mislead.
        self._aspect_mode = aspect_mode if aspect_mode in (
            self.ASPECT_STRETCH, self.ASPECT_LETTERBOX) else self.ASPECT_LETTERBOX
        # Holding the numpy array until the next frame keeps the QImage's
        # backing memory alive; without this Qt may paint freed bytes.
        self._frame_keepalive: Optional[np.ndarray] = None
        # Source pixmap at native cropped resolution; paintEvent scales.
        self._source_pixmap: Optional[QtGui.QPixmap] = None
        # Small floor so a full grid (up to 16 tiles) can shrink with its window.
        self.setMinimumSize(96, 72)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setText(self.PLACEHOLDER_TEXT)

    # ── Configuration ──────────────────────────────────────────────────

    # ── Public update API ─────────────────────────────────────────────

    def update_frame(self, frame: np.ndarray) -> None:
        """Stash numpy frame, build QImage zero-copy, schedule a paint.

        Accepts BGR (HxWx3 uint8), BGRA (HxWx4 uint8), or grayscale
        (HxW uint8). Other shapes are ignored. Thread-safe, Qt's
        ``update()`` posts a paint event that runs on the GUI thread.
        """
        if frame is None:
            return
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        h, w = frame.shape[:2]
        if frame.ndim == 2:
            fmt = QtGui.QImage.Format.Format_Grayscale8
            bpl = w
        elif frame.ndim == 3 and frame.shape[2] == 3:
            fmt = QtGui.QImage.Format.Format_BGR888
            bpl = 3 * w
        elif frame.ndim == 3 and frame.shape[2] == 4:
            fmt = QtGui.QImage.Format.Format_RGBA8888
            bpl = 4 * w
        else:
            return  # unsupported shape

        # Hold numpy ref BEFORE building the QImage so the buffer cannot
        # get GC'd between assignment and the paint.
        self._frame_keepalive = frame
        qimg = QtGui.QImage(frame.data, w, h, bpl, fmt)
        # Convert to pixmap once per frame, paintEvent scales from this.
        # The QPixmap owns its own memory, detached from the numpy buffer.
        self._source_pixmap = QtGui.QPixmap.fromImage(qimg)
        self._note_source_aspect(w, h)
        # Clear placeholder text once we have a frame.
        super().setText("")
        self.update()

    def clear_frame(self) -> None:
        """Drop the current frame and show the placeholder text."""
        self._source_pixmap = None
        self._frame_keepalive = None
        super().setText(self.PLACEHOLDER_TEXT)
        self.update()

    # ── QLabel-API wrappers ─────────

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:  # noqa: N802
        """Route an external pixmap through the same painted-source pipe."""
        if pixmap is None or pixmap.isNull():
            self.clear_frame()
            return
        self._source_pixmap = pixmap
        self._frame_keepalive = None  # QPixmap owns its memory
        self._note_source_aspect(pixmap.width(), pixmap.height())
        super().setText("")
        self.update()

    def setText(self, text: str) -> None:  # noqa: N802
        """Show ``text`` instead of the video.

        A frame is painted from ``_source_pixmap`` in ``paintEvent``, so
        setting text has to drop that frame or the caller's message is never
        seen, the tile keeps showing the last frame it received. That is what
        made a disconnected camera's tile stay frozen on its final image
        instead of going dark.
        """
        self.PLACEHOLDER_TEXT = text or ""
        self._source_pixmap = None
        self._frame_keepalive = None
        super().setText(self.PLACEHOLDER_TEXT)
        self.update()

    def clear(self) -> None:
        """QLabel contract: clear really clears, frame included."""
        self.clear_frame()

    def pixmap(self) -> QtGui.QPixmap:  # noqa: N802
        """Wrapper: return the current source pixmap (or empty)."""
        return self._source_pixmap if self._source_pixmap is not None else QtGui.QPixmap()

    # ── Widget shape ──────────────────────────────────────────────────

    def source_aspect(self) -> Optional[float]:
        """Width over height of the frames being shown, or ``None``."""
        return self._source_aspect

    def _note_source_aspect(self, w: int, h: int) -> None:
        """Learn the camera's shape from the frame, and ask for it.

        Only when it CHANGES: re-laying-out costs real work, and doing it on
        every frame would re-run the layout thirty times a second for a ratio
        that never moves.
        """
        if not self._lock_widget_aspect or w <= 0 or h <= 0:
            return
        aspect = float(w) / float(h)
        if (self._source_aspect is not None
                and abs(aspect - self._source_aspect) < 1e-3):
            return
        self._source_aspect = aspect
        self.updateGeometry()
        # The enclosing box sizes this widget itself; tell it the shape it is
        # placing has changed. The size POLICY is not enough on its own: a
        # widget in a horizontal layout is given the full column height
        # whatever its heightForWidth says, which is why the first attempt at
        # this left the tile as wrongly-shaped as before.
        box = self.parent()
        if isinstance(box, AspectBox):
            box.replace_child()

    def hasHeightForWidth(self) -> bool:                      # noqa: N802
        if self._lock_widget_aspect and self._source_aspect:
            return True
        return super().hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:              # noqa: N802
        """The height that keeps this width the camera's shape.

        With a height-for-width policy on the widget, the LAYOUT hands the
        tile a cell in the camera's proportions, so it expands with the window
        and stays the right shape the whole way, with no bars. The letterbox
        in ``paintEvent`` stays as the guarantee underneath: a layout that
        cannot honour this (a fixed-height row, a splitter dragged past it)
        still gets an undistorted picture rather than a stretched one.
        """
        if self._lock_widget_aspect and self._source_aspect:
            return max(1, int(round(width / self._source_aspect)))
        return super().heightForWidth(width)

    # ── Paint ─────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if self._source_pixmap is None or self._source_pixmap.isNull():
            # Let QLabel handle the placeholder text path.
            super().paintEvent(event)
            return

        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(
                QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            if self._aspect_mode == self.ASPECT_LETTERBOX:
                painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
                src = self._source_pixmap
                src_w, src_h = src.width(), src.height()
                if src_w > 0 and src_h > 0:
                    wr = self.width()
                    hr = self.height()
                    src_ar = src_w / src_h
                    dst_ar = wr / hr if hr > 0 else src_ar
                    if src_ar > dst_ar:
                        new_w = wr
                        new_h = int(wr / src_ar)
                    else:
                        new_h = hr
                        new_w = int(hr * src_ar)
                    x = (wr - new_w) // 2
                    y = (hr - new_h) // 2
                    painter.drawPixmap(QtCore.QRect(x, y, new_w, new_h),
                                       self._source_pixmap)
            else:
                painter.drawPixmap(self.rect(), self._source_pixmap)
        finally:
            painter.end()


class AspectBox(QtWidgets.QWidget):
    """Holds one video tile at the camera's own shape, centred, at any size.

    A size policy cannot do this. Qt honours ``heightForWidth`` when a layout
    is stacking widgets vertically and largely ignores it for a widget sitting
    in a horizontal row, which is where the maze tile lives: measured on the
    real widget, a 640x480 camera came out at ratios from 0.68 to 2.60 as the
    window was resized, with the policy set and honoured on paper.

    So the geometry is set here instead. The box takes whatever cell the layout
    gives it and places the tile inside as the largest rectangle of the source
    ratio, centred. The tile is then EXACTLY the camera's shape, so its own
    letterbox has nothing to letterbox and no bars are drawn: the picture fills
    the tile, and the tile grows with the window in strict proportion.

    Only the maze uses it. An operant grid tile must tile.
    """

    def __init__(self, child: "VideoTile",
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._child = child
        child.setParent(self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        # The child's floor becomes the BOX's floor, and the child's own is
        # released. The layout negotiates with the box; the child is placed,
        # not negotiated with, and a minimum left on it wins against the
        # geometry set here, a portrait camera came out at 0.80 instead of
        # 0.75 because a 320 px minimum width held the tile wider than its
        # own picture.
        self.setMinimumSize(child.minimumSize())
        child.setMinimumSize(1, 1)

    def child(self) -> "VideoTile":
        return self._child

    def replace_child(self) -> None:
        """Re-place the tile: largest rect of the source ratio, centred."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        aspect = self._child.source_aspect()
        if not aspect or aspect <= 0:
            # No frame yet, so no shape to hold to. Fill, and re-place as soon
            # as the first frame says what the camera's shape is.
            self._child.setGeometry(0, 0, w, h)
            return
        fit_w, fit_h = w, int(round(w / aspect))
        if fit_h > h:
            fit_h, fit_w = h, int(round(h * aspect))
        self._child.setGeometry((w - fit_w) // 2, (h - fit_h) // 2,
                                max(1, fit_w), max(1, fit_h))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:   # noqa: N802
        super().resizeEvent(event)
        self.replace_child()
