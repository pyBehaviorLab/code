"""Single display-widget pattern for camera frames.

Today the only consumer is ``ROIDrawCanvas`` for ROISegmentationDialog
(camera frame + click-drag ROI rect).  Other consumers (live video tile,
zone editor) keep their existing display widgets, see
``VideoStreamHolder`` (QLabel + ``pixmap.scaled``) and the QPainter
zone editor, both of which already adapt to window resize.

``FrameDisplay`` owns:

  - the camera frame + cached QPixmap,
  - the widget→image and image→widget coordinate transforms,
  - automatic re-scaling on ``resizeEvent``,
  - centred KeepAspectRatio rendering,

so subclasses only override ``_paint_overlays`` (Qt drawing) and the
mouse handlers.  No subclass touches scaling math.

Resolution-independence is automatic: the base widget reads ``frame.shape``
on every ``set_frame`` call, so a camera change at runtime just reflows.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap

logger = logging.getLogger(__name__)


# ───────────────────────── base widget ─────────────────────────

class FrameDisplay(QtWidgets.QWidget):
    """Show a BGR ndarray frame, scaled-to-fit, with coord transforms.

    Storage convention:
      - ``self._frame_w`` / ``self._frame_h`` = pixel size of the
        most-recently-received frame (image space).
      - ``self._scale``, ``self._off_x``, ``self._off_y`` cached from the
        last ``_recompute_layout`` call; recomputed on every resize and on
        every ``set_frame`` call where the frame size changed.

    Subclasses override ``_paint_overlays(painter)`` to draw on top of the
    image.  The painter is positioned in widget coordinates; subclasses
    convert via ``image_to_widget`` for any drawing in image space.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None        # cached RGB QPixmap of the frame
        self._frame_w: int = 0
        self._frame_h: int = 0
        self._scale: float = 1.0
        self._off_x: int = 0
        self._off_y: int = 0
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #000;")
        # Smoothing: bilinear when scaling down, nearest when upscaling, so
        # native-resolution frames stay sharp while large frames don't pixelate.
        self._smooth_when_downscaling = True

    # ── frame management ──────────────────────────────────────

    def set_frame(self, frame_bgr: Optional[np.ndarray]) -> None:
        """Replace the displayed frame and request a repaint.

        ``frame_bgr`` is what cv2 / VideoManager hands us, H x W x 3 uint8
        BGR.  Pass None to clear the display.
        """
        if frame_bgr is None:
            self._pixmap = None
            self._frame_w = 0
            self._frame_h = 0
            self.update()
            return

        # Convert BGR → RGB QImage → QPixmap.  Qt expects continuous bytes;
        # ``np.ascontiguousarray`` is a no-op when already contiguous.
        if frame_bgr.ndim == 2:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        else:
            # cvtColor, not ``frame[:, :, ::-1]``, the negative-stride slice
            # forces an element-wise copy (≈370 ms at 4K vs ≈26 ms here).
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = rgb.strides[0]
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # ``copy()`` detaches QImage from the numpy buffer so the ndarray can
        # be GC'd without invalidating the pixmap.
        self._pixmap = QPixmap.fromImage(qimg.copy())
        size_changed = (w != self._frame_w) or (h != self._frame_h)
        self._frame_w, self._frame_h = w, h
        if size_changed:
            self._recompute_layout()
        self.update()

    def frame_size(self) -> Tuple[int, int]:
        """Current frame ``(W, H)`` in image pixels.  ``(0, 0)`` if no frame."""
        return self._frame_w, self._frame_h

    # ── layout ────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recompute_layout()
        self.update()

    def _recompute_layout(self) -> None:
        """Update ``_scale`` and ``_off_x/y`` so the frame fits the widget.

        KeepAspectRatio + centred. Cached to avoid recomputing the fit inside
        ``paintEvent`` (which runs on every mouse move).
        """
        if self._frame_w <= 0 or self._frame_h <= 0:
            self._scale, self._off_x, self._off_y = 1.0, 0, 0
            return
        ws = self.size()
        sf = min(ws.width() / self._frame_w,
                 ws.height() / self._frame_h)
        scaled_w = int(self._frame_w * sf)
        scaled_h = int(self._frame_h * sf)
        self._scale = sf
        self._off_x = (ws.width() - scaled_w) // 2
        self._off_y = (ws.height() - scaled_h) // 2

    # ── coord transforms (subclasses use these, not raw numbers) ──

    def widget_to_image(self, pos) -> QPointF:
        """Widget-space ``QPoint``/``QPointF`` → image-space ``QPointF``.

        Clamped to ``[0, frame_w] x [0, frame_h]`` so a click outside the
        image rect never returns negative coords.
        """
        if self._scale <= 0 or self._frame_w <= 0:
            return QPointF(0.0, 0.0)
        x = (pos.x() - self._off_x) / self._scale
        y = (pos.y() - self._off_y) / self._scale
        x = max(0.0, min(float(self._frame_w), x))
        y = max(0.0, min(float(self._frame_h), y))
        return QPointF(x, y)

    def image_to_widget_rect(self, rect: QRectF) -> QRectF:
        return QRectF(rect.x() * self._scale + self._off_x,
                      rect.y() * self._scale + self._off_y,
                      rect.width() * self._scale,
                      rect.height() * self._scale)

    # ── painting ──────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                        self._smooth_when_downscaling)
        p.fillRect(self.rect(), QColor(17, 17, 17))
        if self._pixmap is not None and self._frame_w > 0:
            sw = int(self._frame_w * self._scale)
            sh = int(self._frame_h * self._scale)
            p.drawPixmap(self._off_x, self._off_y, sw, sh, self._pixmap)
        # Hand off to subclass for ROI / zone / tracker overlays.
        try:
            self._paint_overlays(p)
        except Exception as e:
            logger.debug("overlay paint error: %s", e)
        p.end()

    def _paint_overlays(self, painter: QPainter) -> None:
        """Hook for subclasses.  Default: nothing on top of the image."""
        pass



# ─────────────────────── ROI draw canvas ───────────────────────

class ROIDrawCanvas(FrameDisplay):
    """Click-drag rectangular ROI per box, stored in image-pixel coords.

    Used by the rewritten ROISegmentationDialog.  Multiple ROIs can be
    held simultaneously (one per box); the *current* box is highlighted.
    """

    roi_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rois: dict = {}              # box_id -> (x, y, w, h) in image coords
        self._current_box: Optional[int] = None
        self._drawing = False
        self._start: Optional[QPointF] = None
        self._cur_rect: Optional[QRectF] = None
        # DLC/SLEAP need identical crop dimensions across boxes, so the FIRST
        # ROI drawn locks the size; every later box's ROI is that exact W×H and
        # is only positioned (move-only). Cleared when no ROIs remain.
        self._locked_size: Optional[Tuple[int, int]] = None
        #: A size fixed on ANOTHER camera, so a multi-camera rig ends up with
        #: one ROI size across every box rather than one per camera.
        self._external_lock: Optional[Tuple[int, int]] = None

    # ── public API ────────────────────────────────────────────
    def set_current_box(self, setup_id) -> None:
        self._current_box = setup_id
        self.update()

    def get_roi(self, setup_id) -> Optional[Tuple[int, int, int, int]]:
        return self._rois.get(setup_id)

    def get_rois(self) -> dict:
        return dict(self._rois)

    def locked_size(self) -> Optional[Tuple[int, int]]:
        """The fixed (w, h) all further ROIs snap to, or None before the first
        is drawn."""
        return self._locked_size

    def set_rois(self, rois: dict) -> None:
        self._rois = {k: tuple(map(int, v)) for k, v in rois.items()
                      if v and len(v) == 4}
        self._recompute_locked()
        self.update()

    def clear_roi(self, setup_id) -> None:
        if setup_id in self._rois:
            del self._rois[setup_id]
            self._recompute_locked()
            self.roi_changed.emit()
            self.update()

    def _recompute_locked(self) -> None:
        """Lock the size to the first existing ROI's (w, h); unlock when empty.

        An externally supplied lock outranks this, because it comes from a
        camera whose ROIs are already drawn: the boxes on THIS camera must
        match those, not just each other.
        """
        if self._external_lock is not None:
            self._locked_size = self._external_lock
        elif self._rois:
            _x, _y, w, h = next(iter(self._rois.values()))
            self._locked_size = (int(w), int(h))
        else:
            self._locked_size = None

    def set_external_locked_size(self, size) -> bool:
        """Adopt a size fixed on another camera. True if it was taken.

        Every box's ROI feeds one pose model, and the model gets ONE input
        shape: ``canonical_shape`` takes the widest width and tallest height
        across boxes and letterboxes every smaller box into it. So ROIs of
        different sizes mean every box pays the largest box's inference cost
        while the smaller ones have the animal filling less of the input.
        Identical ROIs remove both. That is easy within one camera, where the
        first ROI already locks the rest, and it is what was missing ACROSS
        cameras: each camera's editor locked to its own first ROI.

        Refused, loudly, when it cannot fit this camera's frame: silently
        shrinking would produce exactly the mismatch this exists to prevent.
        """
        if not size:
            self._external_lock = None
            self._recompute_locked()
            return True
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            return False
        if self._frame_w and self._frame_h and (w > self._frame_w
                                                or h > self._frame_h):
            return False
        self._external_lock = (w, h)
        self._recompute_locked()
        self.update()
        return True

    def _fixed_rect_at(self, cx: float, cy: float) -> QRectF:
        """A locked-size rectangle centred on (cx, cy), clamped inside the frame."""
        w, h = self._locked_size
        x = max(0.0, min(cx - w / 2.0, max(0, self._frame_w - w)))
        y = max(0.0, min(cy - h / 2.0, max(0, self._frame_h - h)))
        return QRectF(x, y, float(w), float(h))

    # ── drawing ───────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or self._current_box is None:
            return
        self._start = self.widget_to_image(ev.pos())
        self._drawing = True
        # Locked: place the fixed-size box under the cursor immediately so a
        # plain click (no drag) still positions it.
        if self._locked_size is not None:
            self._cur_rect = self._fixed_rect_at(self._start.x(), self._start.y())
            self.update()

    def mouseMoveEvent(self, ev):
        if not self._drawing or self._start is None:
            return
        end = self.widget_to_image(ev.pos())
        if self._locked_size is not None:
            # Move-only: the box keeps the locked size and follows the cursor.
            self._cur_rect = self._fixed_rect_at(end.x(), end.y())
        else:
            # First box: free-draw to define the size everything else inherits.
            x = min(self._start.x(), end.x())
            y = min(self._start.y(), end.y())
            w = abs(end.x() - self._start.x())
            h = abs(end.y() - self._start.y())
            self._cur_rect = QRectF(x, y, w, h)
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or not self._drawing:
            return
        self._drawing = False
        if self._cur_rect is None or self._current_box is None:
            self._cur_rect = None
            self.update()
            return
        # Reject a stray click only while free-drawing the first box; a locked
        # box is always valid (it carries the established size).
        if self._locked_size is None and (
                self._cur_rect.width() < 5 or self._cur_rect.height() < 5):
            self._cur_rect = None
            self.update()
            return
        rect = (int(self._cur_rect.x()), int(self._cur_rect.y()),
                int(self._cur_rect.width()), int(self._cur_rect.height()))
        self._rois[self._current_box] = rect
        self._cur_rect = None
        # The first committed ROI locks the size for every later box.
        if self._locked_size is None:
            self._locked_size = (rect[2], rect[3])
        self.roi_changed.emit()
        self.update()

    def _paint_overlays(self, painter: QPainter) -> None:
        if self._frame_w <= 0:
            return
        font = QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        # Existing ROIs
        for bid, rect in self._rois.items():
            x, y, w, h = rect
            wr = self.image_to_widget_rect(QRectF(x, y, w, h))
            if bid == self._current_box:
                pen = QPen(QColor(0, 255, 0), 2)
                fill = QColor(0, 255, 0, 50)
            else:
                pen = QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine)
                fill = QColor(255, 165, 0, 30)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            painter.drawRect(wr)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(wr.topLeft() + QPointF(5, 15), f"Box {bid}")
        # In-progress drag
        if self._drawing and self._cur_rect is not None:
            wr = self.image_to_widget_rect(self._cur_rect)
            painter.setPen(QPen(QColor(0, 255, 255), 2))
            painter.setBrush(QBrush(QColor(0, 255, 255, 50)))
            painter.drawRect(wr)

