"""Zone editor, native PySide6 (QWidget + QPainter), shared by operant + maze.

A ``_ZoneCanvas`` paints the live frame as a ``QPixmap`` and zones with
``QPainter``; interaction is native Qt mouse / wheel / key events with a fit
transform for hit-testing. ``ZoneEditorWidget`` is the shell (canvas + control
panel) and keeps the exact public API the tracking dialog depends on.

Storage contract: ``get_zones()`` returns zone dicts with NORMALIZED [0, 1]
box-space points (``coord_space:"normalized"`` + ``shape_dim``), so resolution
changes never move a zone and every saved project stays compatible. Internally
the canvas works in FRAME-PIXEL space (crisp hit-testing + aspect-correct
rotation); conversion happens only at the get/set boundary.

Features: rotation handle, copy/paste/duplicate, zoom/pan, attach-edges, all
drawn with QPainter, verifiable headlessly via ``grab()``.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF

from source.gui.widgets.numeric_line_edit import NumericLineEdit
from source.video.zones.coords import resolve_to_px

logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:
    # pragma: no cover
    np = None


# Zone palette, mirror the live overlay (zone_overlay.ZONE_COLORS_BGR) so a
# zone renders the same colour in the editor and the live view. Fall back to a
# fixed list if the overlay module is unavailable.
def _palette_hex() -> list[str]:
    try:
        from source.gui.widgets.zone_overlay import ZONE_COLORS_BGR
        return [f'#{r:02X}{g:02X}{b:02X}' for (b, g, r) in ZONE_COLORS_BGR]
    except Exception:
        return ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
                '#00BCD4', '#FFEB3B', '#FF5722', '#8BC34A', '#3F51B5']


ZONE_COLORS = _palette_hex()
ZONE_ICONS = {"rectangle": "▢", "circle": "○", "ellipse": "⬭",
              "polygon": "⬡", "line": "─", "scale": "⟷"}
_SEL_COLOR = '#FF9800'
_HANDLE_GRAB_PX = 11.0


class _ZoneCanvas(QtWidgets.QWidget):
    """Interactive drawing surface. Owns the zone list + all interaction.

    Zones are stored in FRAME-PIXEL coords (lists of ``[x, y]``)
    the shell
    converts to/from normalized at the get/set boundary. A fit transform maps
    frame pixels ↔ widget pixels (aspect-preserving letterbox + zoom/pan)
    hit-testing and geometry run in frame space.
    """

    zones_changed = Signal()
    selection_changed = Signal(int)     # selected_idx or -1
    status_changed = Signal(str)

    def __init__(self, get_frame_cb: Callable | None = None, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

        self._frame_cb = get_frame_cb
        self.zones: list[dict] = []
        self.original_frame_size = (640, 480)
        self.frame_w, self.frame_h = 640, 480
        self._pixmap: QtGui.QPixmap | None = None
        self._frame_keepalive = None

        self.selected_idx: int | None = None
        self.pixel_scale: float | None = None
        self.scale_unit = "cm"

        # view (zoom/pan on top of the fit transform)
        self._view_scale = 1.0
        self._view_ox = 0.0
        self._view_oy = 0.0
        self._panning = False
        self._pan_last = None

        # draw mode
        self._mode: str | None = None
        self._pending_name: str | None = None
        self._draw_start = None
        self._draw_cur = None
        self._poly_pts: list[list] = []
        self._rp_sides = 6
        self._rp_center = None
        self._scale_real = 10.0
        self._scale_unit_input = "cm"

        # edit drag
        self._dragging = False
        self._drag_type = None            # move | handle | ellipse_handle | rotate
        self._drag_handle_idx = None
        self._drag_start = None
        self._drag_pre_snapshot = None
        self._rotate_pivot = None
        self._rotate_start_ang = None

        # magnet
        self._magnet_enabled = False
        self._snap_px = 5
        self._magnet_target = None

        # undo/redo + clipboard
        self._undo_stack: deque = deque(maxlen=40)
        self._redo_stack: deque = deque(maxlen=40)
        self._clipboard = None

        self._cursor_widget = None

        self._live_timer = QtCore.QTimer(self)
        self._live_timer.timeout.connect(self._update_live_frame)
        self._live_timer.setInterval(200)  # 5 fps

    # ── frame source ───────────────────────────────────────────

    def set_frame_callback(self, cb):
        self._frame_cb = cb

    def start_live(self):
        if not self._live_timer.isActive():
            self._live_timer.start()

    def stop_live(self):
        self._live_timer.stop()

    def _update_live_frame(self):
        if self._dragging or self._mode is not None:
            return
        if not self._frame_cb:
            return
        try:
            frame = self._frame_cb()
        except Exception:
            frame = None
        if frame is not None:
            self._set_frame(frame)

    def _set_frame(self, frame):
        h, w = frame.shape[:2]
        # Re-anchor existing zones when the frame size changes (default→real
        # camera, or a resolution switch) so pixel coords keep the same
        # normalized position. Without this a zone set at the 640x480 default
        # would jump when the real frame arrives.
        if (self.frame_w and self.frame_h and self.zones
                and (w, h) != (self.frame_w, self.frame_h)):
            fx, fy = w / self.frame_w, h / self.frame_h
            for z in self.zones:
                z["points"] = [[p[0] * fx, p[1] * fy] for p in z.get("points", [])]
                if len(z.get("center", []) or []) == 2:
                    z["center"] = [z["center"][0] * fx, z["center"][1] * fy]
                if len(z.get("semi_axes", []) or []) == 2:
                    z["semi_axes"] = [z["semi_axes"][0] * fx, z["semi_axes"][1] * fy]
        self.original_frame_size = (w, h)
        self.frame_w, self.frame_h = w, h
        if np is not None and not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        self._frame_keepalive = frame
        if frame.ndim == 2:
            qimg = QtGui.QImage(frame.data, w, h, w,
                                QtGui.QImage.Format.Format_Grayscale8)
        else:
            qimg = QtGui.QImage(frame.data, w, h, 3 * w,
                                QtGui.QImage.Format.Format_BGR888)
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self.update()

    def refresh_background(self):
        if self._frame_cb:
            f = self._frame_cb()
            if f is not None:
                self._set_frame(f)
                self.status_changed.emit(
                    f"Refreshed: {self.frame_w}x{self.frame_h}")

    # ── fit transform ──────────────────────────────────────────

    def _fit(self):
        W, H = self.width(), self.height()
        fw, fh = self.frame_w, self.frame_h
        if fw <= 0 or fh <= 0:
            return 1.0, 0.0, 0.0
        base = min(W / fw, H / fh)
        s = base * self._view_scale
        ox = (W - fw * s) / 2 + self._view_ox
        oy = (H - fh * s) / 2 + self._view_oy
        return s, ox, oy

    def _f2w(self, x, y):
        s, ox, oy = self._fit()
        return ox + x * s, oy + y * s

    def _w2f(self, px, py):
        s, ox, oy = self._fit()
        if s == 0:
            return px, py
        return (px - ox) / s, (py - oy) / s

    def _reset_view(self):
        self._view_scale = 1.0
        self._view_ox = 0.0
        self._view_oy = 0.0
        self.update()

    def _grab_r_frame(self):
        s, _, _ = self._fit()
        return _HANDLE_GRAB_PX / max(s, 1e-6)

    # ── painting ───────────────────────────────────────────────

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor('#111214'))
        s, ox, oy = self._fit()
        if self._pixmap is not None:
            p.drawPixmap(QRectF(ox, oy, self.frame_w * s, self.frame_h * s),
                         self._pixmap, QRectF(self._pixmap.rect()))
        else:
            p.setPen(QColor('#ff9800'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera…")

        color_idx = 0
        for i, z in enumerate(self.zones):
            base = ('#E91E63' if z.get("type") == "scale"
                    else ZONE_COLORS[color_idx % len(ZONE_COLORS)])
            if z.get("type") != "scale":
                color_idx += 1
            self._paint_zone(p, z, base, i == self.selected_idx)

        if self._magnet_target:
            wx, wy = self._f2w(*self._magnet_target)
            p.setPen(QPen(QColor('#FFEB3B'), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(wx, wy), 9, 9)

        self._paint_preview(p)
        self._paint_crosshair(p)
        p.end()

    def _wpoly(self, pts):
        poly = QPolygonF()
        for x, y in pts:
            wx, wy = self._f2w(x, y)
            poly.append(QPointF(wx, wy))
        return poly

    def _paint_zone(self, p, z, base_hex, sel):
        col = QColor(_SEL_COLOR if sel else base_hex)
        pts = z.get("points", [])
        zt = z.get("type", "polygon")
        lw = 3 if sel else 2

        if zt == "scale" and len(pts) >= 2:
            pen = QPen(col, lw)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            w1 = self._f2w(*pts[0])
            w2 = self._f2w(*pts[1])
            p.drawLine(QPointF(*w1), QPointF(*w2))
            self._paint_endpoints(p, [w1, w2], col, sel)
            px_len = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            txt = f"{z.get('scale_length','')} {z.get('scale_unit','')}  ({px_len:.0f}px)"
            self._paint_label(p, txt, (w1[0] + w2[0]) / 2, (w1[1] + w2[1]) / 2 - 14, col)
            return

        if zt == "line" and len(pts) >= 2:
            p.setPen(QPen(col, lw))
            w1 = self._f2w(*pts[0])
            w2 = self._f2w(*pts[1])
            p.drawLine(QPointF(*w1), QPointF(*w2))
            self._paint_endpoints(p, [w1, w2], col, sel)
            if sel:
                self._paint_handles(p, [w1, w2])
            return

        if len(pts) < 3:
            return
        fill = QColor(col)
        fill.setAlpha(90 if sel else 64)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(col, lw))
        p.drawPolygon(self._wpoly(pts))

        if not sel:
            return
        if zt in ("circle", "ellipse") and len(z.get("center", []) or []) == 2:
            cx, cy = z["center"]
            rx, ry = z.get("semi_axes", [50, 50])
            hpts = [self._f2w(cx, cy - ry), self._f2w(cx + rx, cy),
                    self._f2w(cx, cy + ry), self._f2w(cx - rx, cy)]
        else:
            hpts = [self._f2w(x, y) for x, y in pts]
        self._paint_handles(p, hpts)
        # Rotation is Ctrl+drag (about the drift-free area centroid) plus [ ]
        # for ±5°, no stalk-and-knob handle; just mark the pivot the zone turns
        # around. The bottom cheat-sheet bar advertises the gesture.
        if zt not in ("line", "scale"):
            cxp, cyp = self._zone_centroid(z)
            pv = self._f2w(cxp, cyp)
            p.setPen(QPen(QColor('white'), 1.0))
            p.setBrush(QBrush(QColor('#00e5ff')))
            p.drawEllipse(QPointF(*pv), 3.0, 3.0)

    def _paint_handles(self, p, wpts):
        p.setBrush(QBrush(QColor('white')))
        p.setPen(QPen(QColor(_SEL_COLOR), 2))
        for wx, wy in wpts:
            p.drawRect(QRectF(wx - 4, wy - 4, 8, 8))

    def _paint_endpoints(self, p, wpts, col, sel):
        p.setBrush(QBrush(QColor('white') if sel else col))
        p.setPen(QPen(col, 2))
        for wx, wy in wpts:
            p.drawEllipse(QPointF(wx, wy), 4, 4)

    def _paint_label(self, p, text, x, y, col):
        p.setFont(QtGui.QFont("Segoe UI", 8))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 8
        h = fm.height() + 4
        r = QRectF(x - w / 2, y - h / 2, w, h)
        p.setBrush(QBrush(col))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 3, 3)
        p.setPen(QColor('white'))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_preview(self, p):
        if self._mode is None:
            return
        if self._mode == 'manpoly' and self._poly_pts:
            self._paint_manpoly_preview(p)
            return
        if self._mode == 'regpoly':
            self._paint_regpoly_preview(p)
            return
        if self._draw_start is None or self._draw_cur is None:
            return
        p.setPen(QPen(QColor('#00e5ff'), 2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        w0 = self._f2w(*self._draw_start)
        w1 = self._f2w(*self._draw_cur)
        if self._mode == 'rect':
            p.drawRect(QRectF(QPointF(*w0), QPointF(*w1)).normalized())
        elif self._mode == 'ellipse':
            p.drawEllipse(QRectF(QPointF(*w0), QPointF(*w1)).normalized())
        elif self._mode in ('line', 'scale'):
            p.setPen(QPen(QColor('#00e5ff' if self._mode == 'line' else '#ff5252'),
                          3, Qt.PenStyle.SolidLine))
            p.drawLine(QPointF(*w0), QPointF(*w1))

    def _paint_manpoly_preview(self, p):
        p.setPen(QPen(QColor('#b06fe0'), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        wpts = [self._f2w(x, y) for x, y in self._poly_pts]
        for i in range(len(wpts) - 1):
            p.drawLine(QPointF(*wpts[i]), QPointF(*wpts[i + 1]))
        if self._draw_cur and wpts:
            p.drawLine(QPointF(*wpts[-1]), QPointF(*self._f2w(*self._draw_cur)))
        for wx, wy in wpts:
            p.setBrush(QBrush(QColor('#b06fe0')))
            p.drawEllipse(QPointF(wx, wy), 3, 3)

    def _paint_regpoly_preview(self, p):
        if not self._rp_center or not self._draw_cur:
            return
        cx, cy = self._rp_center
        mx, my = self._draw_cur
        r = math.hypot(mx - cx, my - cy)
        a = math.atan2(my - cy, mx - cx)
        pts = [(cx + r * math.cos(a + 2 * math.pi * i / self._rp_sides),
                cy + r * math.sin(a + 2 * math.pi * i / self._rp_sides))
               for i in range(self._rp_sides)]
        p.setPen(QPen(QColor('#2196F3'), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(self._wpoly(pts))

    def _paint_crosshair(self, p):
        if self._mode is None or self._cursor_widget is None:
            return
        cx, cy = self._cursor_widget
        p.setPen(QPen(QColor(0, 229, 255, 150), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(0, cy), QPointF(self.width(), cy))
        p.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))

    # ── geometry helpers (frame space) ─────────────────────────

    @staticmethod
    def _point_in_poly(x, y, pts):
        n = len(pts)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _dist_to_seg(px, py, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _zone_centroid(self, z):
        """True area centroid, the rotation-invariant pivot (no drift on
        repeated rotation). Ellipse/circle use their stored centre."""
        if len(z.get("center", []) or []) == 2:
            return float(z["center"][0]), float(z["center"][1])
        pts = z.get("points", [])
        n = len(pts)
        if n == 0:
            return 0.0, 0.0
        if n < 3:
            return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
        A = cx = cy = 0.0
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            A += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        A *= 0.5
        if abs(A) < 1e-9:
            return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
        return cx / (6 * A), cy / (6 * A)

    def _resize_rectangle(self, z, idx, fx, fy):
        """Resize a (possibly rotated) rectangle by dragging corner ``idx``.

        Works in the rectangle's OWN frame: the drag and all corners are
        un-rotated by the rectangle's angle, the axis-aligned corner
        constraint is applied there (opposite corner stays fixed), then
        everything is rotated back. For an un-rotated rectangle the angle is
        0, so this is identical to the plain screen-axis behaviour, but a
        rotated rectangle now resizes along its own edges instead of shearing.
        """
        pts = z["points"]
        # Orientation = direction of the top edge (corner 0 → 1).
        theta = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
        cx, cy = self._zone_centroid(z)
        cn, sn = math.cos(-theta), math.sin(-theta)
        cp, sp = math.cos(theta), math.sin(theta)

        def to_local(x, y):
            dx, dy = x - cx, y - cy
            return [dx * cn - dy * sn, dx * sn + dy * cn]

        def to_world(x, y):
            return [x * cp - y * sp + cx, x * sp + y * cp + cy]

        loc = [to_local(px, py) for px, py in pts]
        loc[idx] = to_local(fx, fy)
        if idx == 0:
            loc[1][1] = loc[0][1]; loc[3][0] = loc[0][0]
        elif idx == 1:
            loc[0][1] = loc[1][1]; loc[2][0] = loc[1][0]
        elif idx == 2:
            loc[1][0] = loc[2][0]; loc[3][1] = loc[2][1]
        elif idx == 3:
            loc[0][0] = loc[3][0]; loc[2][1] = loc[3][1]
        z["points"] = [to_world(x, y) for x, y in loc]

    # ── selection ──────────────────────────────────────────────

    def set_selected(self, idx):
        self.selected_idx = idx if (idx is not None and 0 <= idx < len(self.zones)) else None
        self.update()
        self.selection_changed.emit(self.selected_idx if self.selected_idx is not None else -1)

    # ── mouse ──────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        self.setFocus()
        pos = ev.position()
        self._cursor_widget = (pos.x(), pos.y())
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = (pos.x(), pos.y())
            return
        fx, fy = self._w2f(pos.x(), pos.y())
        if self._mode is not None:
            self._draw_press(ev, fx, fy)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        gr = self._grab_r_frame()
        # Ctrl+drag anywhere on a zone rotates it about its centroid, the
        # primary rotate gesture (the tiny handle knob is a fiddly fallback).
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            ridx = self._zone_index_at(fx, fy)
            if (ridx is not None
                    and self.zones[ridx].get("type") not in ("line", "scale")):
                self.set_selected(ridx)
                z = self.zones[ridx]
                self._drag_pre_snapshot = self._snapshot()
                cx, cy = self._zone_centroid(z)
                self._rotate_pivot = (cx, cy)
                self._rotate_start_ang = math.atan2(fy - cy, fx - cx)
                self._begin_drag('rotate', None, fx, fy)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
        if self.selected_idx is not None and self.selected_idx < len(self.zones):
            z = self.zones[self.selected_idx]
            pts = z.get("points", [])
            zt = z.get("type", "polygon")
            if zt in ("circle", "ellipse") and len(z.get("center", []) or []) == 2:
                cx, cy = z["center"]
                rx, ry = z.get("semi_axes", [50, 50])
                for idx, (hx, hy) in enumerate(
                        [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]):
                    if math.hypot(fx - hx, fy - hy) < gr:
                        self._begin_drag('ellipse_handle', idx, fx, fy)
                        return
            else:
                for idx, (hx, hy) in enumerate(pts):
                    if math.hypot(fx - hx, fy - hy) < gr:
                        self._begin_drag('handle', idx, fx, fy)
                        return
            if zt in ("line", "scale") and len(pts) >= 2:
                if self._dist_to_seg(fx, fy, pts[0], pts[1]) < gr * 1.4:
                    self._begin_drag('move', None, fx, fy)
                    return
            elif self._point_in_poly(fx, fy, pts):
                self._begin_drag('move', None, fx, fy)
                return
        for i, z in enumerate(self.zones):
            pts = z.get("points", [])
            zt = z.get("type", "polygon")
            if zt in ("line", "scale") and len(pts) >= 2:
                if self._dist_to_seg(fx, fy, pts[0], pts[1]) < gr * 1.4:
                    self.set_selected(i)
                    return
            elif len(pts) >= 3 and self._point_in_poly(fx, fy, pts):
                self.set_selected(i)
                return
        self.set_selected(None)

    def _zone_index_at(self, fx, fy):
        """Index of the zone under (fx, fy), the selected zone first (so a
        Ctrl+drag that starts inside the current selection rotates it), then
        top-most hit. None if the cursor is over empty space."""
        gr = self._grab_r_frame()
        order = []
        if self.selected_idx is not None and self.selected_idx < len(self.zones):
            order.append(self.selected_idx)
        order += [i for i in range(len(self.zones)) if i != self.selected_idx]
        for i in order:
            z = self.zones[i]
            pts = z.get("points", [])
            zt = z.get("type", "polygon")
            if zt in ("line", "scale") and len(pts) >= 2:
                if self._dist_to_seg(fx, fy, pts[0], pts[1]) < gr * 1.4:
                    return i
            elif len(pts) >= 3 and self._point_in_poly(fx, fy, pts):
                return i
        return None

    def _begin_drag(self, dtype, hidx, fx, fy):
        self._dragging = True
        self._drag_type = dtype
        self._drag_handle_idx = hidx
        self._drag_start = (fx, fy)
        if self._drag_pre_snapshot is None:
            self._drag_pre_snapshot = self._snapshot()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        self._cursor_widget = (pos.x(), pos.y())
        if self._panning and self._pan_last is not None:
            self._view_ox += pos.x() - self._pan_last[0]
            self._view_oy += pos.y() - self._pan_last[1]
            self._pan_last = (pos.x(), pos.y())
            self.update()
            return
        fx, fy = self._w2f(pos.x(), pos.y())
        if self._mode is not None:
            self._draw_move(ev, fx, fy)
            self.update()
            return
        if not self._dragging:
            return
        self._drag_move(fx, fy)
        self.update()

    def _drag_move(self, fx, fy):
        if self.selected_idx is None or self.selected_idx >= len(self.zones):
            return
        z = self.zones[self.selected_idx]
        pts = z.get("points", [])
        if self._drag_type == 'rotate':
            cx, cy = self._rotate_pivot
            ang = math.atan2(fy - cy, fx - cx)
            delta = math.degrees(ang - (self._rotate_start_ang or ang))
            self._rotate_start_ang = ang
            self._rotate_zone(z, delta, pivot=(cx, cy))
            return
        if self._drag_type == 'ellipse_handle' and len(z.get("center", []) or []) == 2:
            cx, cy = z["center"]
            rx, ry = z.get("semi_axes", [50, 50])
            idx = self._drag_handle_idx
            if idx == 0:
                ry = max(5, cy - fy)
            elif idx == 1:
                rx = max(5, fx - cx)
            elif idx == 2:
                ry = max(5, fy - cy)
            elif idx == 3:
                rx = max(5, cx - fx)
            z["semi_axes"] = [rx, ry]
            z["points"] = [[cx + rx * math.cos(2 * math.pi * i / 32),
                            cy + ry * math.sin(2 * math.pi * i / 32)] for i in range(32)]
        elif self._drag_type == 'handle' and self._drag_handle_idx is not None:
            idx = self._drag_handle_idx
            if 0 <= idx < len(pts):
                sx, sy, tgt = self._magnet_snap(fx, fy, self.selected_idx)
                self._magnet_target = tgt
                if z.get("type") == "rectangle" and len(pts) == 4:
                    # Resize in the rectangle's OWN frame so a rotated rect
                    # keeps its shape (the screen-axis constraint would shear
                    # it). For an un-rotated rect this is identical to before.
                    self._resize_rectangle(z, idx, sx, sy)
                else:
                    pts[idx] = [sx, sy]
        elif self._drag_type == 'move':
            dx = fx - self._drag_start[0]
            dy = fy - self._drag_start[1]
            self._drag_start = (fx, fy)
            for pt in pts:
                pt[0] += dx
                pt[1] += dy
            if len(z.get("center", []) or []) == 2:
                z["center"][0] += dx
                z["center"][1] += dy

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_last = None
            return
        if self._mode is not None:
            self._draw_release(ev)
            return
        if not self._dragging:
            return
        if self._drag_pre_snapshot is not None:
            self._push_undo(self._drag_pre_snapshot)
            self._drag_pre_snapshot = None
        self._dragging = False
        self._drag_type = None
        self._drag_handle_idx = None
        self._drag_start = None
        self._rotate_pivot = None
        self._rotate_start_ang = None
        self._magnet_target = None
        self.unsetCursor()
        self.zones_changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if self._mode == 'manpoly':
            self._finish_manpoly()

    def wheelEvent(self, ev):
        pos = ev.position()
        cx, cy = pos.x(), pos.y()
        fx, fy = self._w2f(cx, cy)
        factor = 1.15 ** (ev.angleDelta().y() / 120.0)
        self._view_scale = max(0.5, min(20.0, self._view_scale * factor))
        W, H = self.width(), self.height()
        base = (min(W / self.frame_w, H / self.frame_h)
                if self.frame_w and self.frame_h else 1)
        s = base * self._view_scale
        self._view_ox = cx - fx * s - (W - self.frame_w * s) / 2
        self._view_oy = cy - fy * s - (H - self.frame_h * s) / 2
        self.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key.Key_Escape:
            self.cancel_mode()
        elif k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        elif k == Qt.Key.Key_BracketLeft:
            self.rotate_selected(-5.0)
        elif k == Qt.Key.Key_BracketRight:
            self.rotate_selected(5.0)
        elif k == Qt.Key.Key_Home:
            self._reset_view()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._mode == 'manpoly':
                self._finish_manpoly()
        else:
            super().keyPressEvent(ev)

    # ── drawing tools ──────────────────────────────────────────

    def begin_mode(self, mode, name=None, sides=6):
        self._mode = mode
        self._pending_name = name
        self._draw_start = None
        self._draw_cur = None
        self._poly_pts = []
        self._rp_center = None
        self._rp_sides = sides
        self.set_selected(None)
        self.update()

    def cancel_mode(self):
        self._mode = None
        self._pending_name = None
        self._draw_start = None
        self._draw_cur = None
        self._poly_pts = []
        self._rp_center = None
        self.status_changed.emit("Cancelled")
        self.update()

    def _draw_press(self, ev, fx, fy):
        btn = ev.button()
        m = self._mode
        if m in ('rect', 'ellipse', 'line', 'scale'):
            if btn == Qt.MouseButton.RightButton:
                self.cancel_mode()
                return
            self._draw_start = (fx, fy)
            self._draw_cur = (fx, fy)
        elif m == 'manpoly':
            if btn == Qt.MouseButton.RightButton:
                self._finish_manpoly()
                return
            self._poly_pts.append([fx, fy])
            self._draw_cur = (fx, fy)
        elif m == 'regpoly':
            if btn == Qt.MouseButton.RightButton:
                self.cancel_mode()
                return
            if self._rp_center is None:
                self._rp_center = (fx, fy)
                self._draw_cur = (fx, fy)
            else:
                self._finish_regpoly(fx, fy)

    def _draw_move(self, ev, fx, fy):
        if self._mode in ('rect', 'ellipse', 'line', 'scale') and self._draw_start:
            x2, y2 = fx, fy
            if (self._mode in ('line', 'scale')
                    and ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                x1, y1 = self._draw_start
                if abs(x2 - x1) >= abs(y2 - y1):
                    y2 = y1
                else:
                    x2 = x1
            self._draw_cur = (x2, y2)
        elif self._mode in ('manpoly', 'regpoly'):
            self._draw_cur = (fx, fy)

    def _draw_release(self, ev):
        m = self._mode
        if m not in ('rect', 'ellipse', 'line', 'scale'):
            return
        if not self._draw_start or not self._draw_cur:
            return
        x0, y0 = self._draw_start
        x1, y1 = self._draw_cur
        if m == 'rect':
            if abs(x1 - x0) < 3 or abs(y1 - y0) < 3:
                self.cancel_mode()
                return
            self.add_zone(self._pending_name,
                          [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "rectangle")
        elif m == 'ellipse':
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            if rx < 2 or ry < 2:
                self.cancel_mode()
                return
            pts = [[cx + rx * math.cos(2 * math.pi * i / 64),
                    cy + ry * math.sin(2 * math.pi * i / 64)] for i in range(64)]
            self.add_zone(self._pending_name, pts,
                          "circle" if abs(rx - ry) < 1 else "ellipse",
                          {"center": [cx, cy], "semi_axes": [rx, ry]})
        elif m in ('line', 'scale'):
            if math.hypot(x1 - x0, y1 - y0) < 8:
                self.cancel_mode()
                return
            if m == 'line':
                self.add_zone(self._pending_name, [[x0, y0], [x1, y1]], "line")
            else:
                self._commit_scale(x0, y0, x1, y1)
        self.cancel_mode()

    def _finish_manpoly(self):
        if len(self._poly_pts) >= 3:
            self.add_zone(self._pending_name, [list(p) for p in self._poly_pts], "polygon")
        self.cancel_mode()

    def _finish_regpoly(self, fx, fy):
        cx, cy = self._rp_center
        r = math.hypot(fx - cx, fy - cy)
        a = math.atan2(fy - cy, fx - cx)
        if r < 8:
            return
        pts = [[cx + r * math.cos(a + 2 * math.pi * i / self._rp_sides),
                cy + r * math.sin(a + 2 * math.pi * i / self._rp_sides)]
               for i in range(self._rp_sides)]
        self.add_zone(self._pending_name, pts, "polygon",
                      {"regular": True, "sides": self._rp_sides,
                       "center": [cx, cy], "rotation": math.degrees(a)})
        self.cancel_mode()

    def _commit_scale(self, x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 5:
            return
        self.pixel_scale = self._scale_real / d
        self.scale_unit = self._scale_unit_input
        self._push_undo()
        self.zones = [z for z in self.zones if z.get("type") != "scale"]
        self.zones.append({
            "name": "Scale", "type": "scale",
            "points": [[x0, y0], [x1, y1]],
            "shape_dim": list(self.original_frame_size),
            "scale_length": self._scale_real, "scale_unit": self.scale_unit,
            "enabled": True, "coord_space": "pixel",
        })
        self.zones_changed.emit()
        self.status_changed.emit(f"Scale: {self.pixel_scale:.4f} {self.scale_unit}/px")

    # ── zone mutation ──────────────────────────────────────────

    def _unique_name(self, base):
        existing = {z.get("name", "") for z in self.zones}
        if base not in existing:
            return base
        i = 2
        while f"{base} ({i})" in existing:
            i += 1
        return f"{base} ({i})"

    def add_zone(self, name, pts, ztype, extra=None):
        self._push_undo()
        z = {"name": name or self._unique_name(ztype), "points": pts,
             "type": ztype, "enabled": True, "validated": True,
             "coord_space": "pixel"}
        if extra:
            z.update(extra)
        self.zones.append(z)
        self.set_selected(len(self.zones) - 1)
        self.zones_changed.emit()
        self.status_changed.emit(f"Added: {z['name']}")

    def delete_selected(self):
        if self.selected_idx is None or self.selected_idx >= len(self.zones):
            return
        self._push_undo()
        self.zones.pop(self.selected_idx)
        self.set_selected(None)
        self.zones_changed.emit()

    def rename_selected(self, new_name):
        if self.selected_idx is None or not new_name.strip():
            return
        self.zones[self.selected_idx]["name"] = new_name.strip()
        self.zones_changed.emit()
        self.update()

    # ── undo / redo ────────────────────────────────────────────

    def _snapshot(self):
        return copy.deepcopy(self.zones)

    def _push_undo(self, snapshot=None):
        self._undo_stack.append(snapshot if snapshot is not None else self._snapshot())
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self.cancel_mode()
        self._redo_stack.append(self._snapshot())
        self.zones = self._undo_stack.pop()
        self.set_selected(None)
        self.zones_changed.emit()

    def redo(self):
        if not self._redo_stack:
            return
        self.cancel_mode()
        self._undo_stack.append(self._snapshot())
        self.zones = self._redo_stack.pop()
        self.set_selected(None)
        self.zones_changed.emit()

    def undo_depth(self):
        return len(self._undo_stack), len(self._redo_stack)

    # ── copy / paste / duplicate ───────────────────────────────

    def copy_selected(self):
        if self.selected_idx is None or self.selected_idx >= len(self.zones):
            self.status_changed.emit("Select a zone to copy")
            return
        self._clipboard = copy.deepcopy(self.zones[self.selected_idx])
        self.status_changed.emit(f"Copied: {self._clipboard.get('name','')}")

    def paste_clipboard(self):
        if not self._clipboard:
            self.status_changed.emit("Clipboard empty")
            return
        self._push_undo()
        z = copy.deepcopy(self._clipboard)
        w, h = self.original_frame_size
        ox, oy = 0.04 * w, 0.04 * h
        z["points"] = [[p[0] + ox, p[1] + oy] for p in z.get("points", [])]
        if len(z.get("center", []) or []) == 2:
            z["center"] = [z["center"][0] + ox, z["center"][1] + oy]
        z["name"] = self._unique_name(z.get("name", "zone"))
        self.zones.append(z)
        self.set_selected(len(self.zones) - 1)
        self.zones_changed.emit()
        self.status_changed.emit(f"Pasted: {z['name']}")

    def duplicate_selected(self):
        self.copy_selected()
        if self._clipboard:
            self.paste_clipboard()

    # ── rotate ─────────────────────────────────────────────────

    def _rotate_zone(self, z, deg, pivot=None):
        if pivot is None:
            pivot = self._zone_centroid(z)
        cx, cy = pivot
        rad = math.radians(deg)
        ca, sa = math.cos(rad), math.sin(rad)

        def rot(px, py):
            dx, dy = px - cx, py - cy
            return [cx + dx * ca - dy * sa, cy + dx * sa + dy * ca]

        z["points"] = [rot(p[0], p[1]) for p in z.get("points", [])]
        if len(z.get("center", []) or []) == 2:
            z["center"] = rot(z["center"][0], z["center"][1])
        if "rotation" in z:
            z["rotation"] = (z.get("rotation", 0.0) + deg) % 360.0

    def rotate_selected(self, deg):
        if self.selected_idx is None or self.selected_idx >= len(self.zones):
            return
        z = self.zones[self.selected_idx]
        if z.get("type") in ("line", "scale"):
            self.status_changed.emit("Rotate applies to area zones")
            return
        self._push_undo()
        self._rotate_zone(z, deg)
        self.zones_changed.emit()
        self.update()
        self.status_changed.emit(f"Rotated {deg:+.0f}°")

    # ── magnet / attach ────────────────────────────────────────

    def set_snap_px(self, v):
        self._snap_px = int(v)

    def set_magnet(self, on):
        self._magnet_enabled = bool(on)
        self.status_changed.emit(
            "Magnet ON, vertices snap to nearby zones" if on else "Magnet off")

    def _magnet_snap(self, x, y, exclude_idx):
        if not self._magnet_enabled:
            return x, y, None
        thr = float(self._snap_px)
        best = None
        best_d2 = thr * thr
        for zi, z in enumerate(self.zones):
            if zi == exclude_idx or z.get("type") == "scale":
                continue
            for vx, vy in z.get("points", []):
                d2 = (vx - x) ** 2 + (vy - y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (vx, vy)
        if best is None:
            return x, y, None
        return best[0], best[1], best

    def attach_edges(self):
        """Cluster near-coincident vertices across different zones and collapse
        each cluster to its centroid, builds seamless adjacent zones."""
        thr = float(self._snap_px) * 2
        verts = [(zi, pi) for zi, z in enumerate(self.zones)
                 if z.get("type") not in ("line", "scale")
                 for pi in range(len(z.get("points", [])))]
        if len(verts) < 2:
            self.status_changed.emit("Need 2+ area zones to attach")
            return
        self._push_undo()
        used = set()
        moved = 0
        for a in range(len(verts)):
            if a in used:
                continue
            za, pa = verts[a]
            xa, ya = self.zones[za]["points"][pa]
            cluster = [a]
            for b in range(a + 1, len(verts)):
                if b in used:
                    continue
                zb, pb = verts[b]
                if zb == za:
                    continue
                xb, yb = self.zones[zb]["points"][pb]
                if math.hypot(xb - xa, yb - ya) <= thr:
                    cluster.append(b)
            if len(cluster) > 1:
                cpts = [self.zones[verts[c][0]]["points"][verts[c][1]] for c in cluster]
                mx = sum(pt[0] for pt in cpts) / len(cpts)
                my = sum(pt[1] for pt in cpts) / len(cpts)
                for c in cluster:
                    zc, pc = verts[c]
                    self.zones[zc]["points"][pc] = [mx, my]
                    used.add(c)
                    moved += 1
        self.zones_changed.emit()
        self.update()
        self.status_changed.emit(f"Attached {moved} vertices")


class ZoneEditorWidget(QtWidgets.QWidget):
    """Zone editor shell, canvas + control panel.

    Public API preserved for the tracking dialog: ``get_zones`` / ``set_zones``
    (NORMALIZED [0, 1] dicts), ``get_zone_names``, ``get_scale_meta``,
    ``set_frame_callback``, ``has_valid_scale``, plus the ``scale_value_spin`` /
    ``scale_unit_combo`` widgets and the ``setup_id`` attribute.
    """

    zones_changed = QtCore.Signal()

    def __init__(self, parent=None, setup_id=None, get_frame_callback=None,
                 initial_zones_path=None):
        super().__init__(parent)
        self.setup_id = setup_id
        self.get_frame_callback = get_frame_callback

        self.canvas = _ZoneCanvas(get_frame_callback, self)
        self.canvas.zones_changed.connect(self._on_canvas_zones_changed)
        self.canvas.selection_changed.connect(self._on_canvas_selection)
        self.canvas.status_changed.connect(self._set_status)

        self._build_ui()
        self._reload_list()
        self._update_undo_ui()

        if initial_zones_path:
            try:
                from source.video.zones.io import load_zone_config
                self.set_zones(load_zone_config(initial_zones_path))
            except Exception as e:
                logger.debug("initial zones load failed (%s): %s", initial_zones_path, e)

    # ── convenience properties ─────────────────────────────────

    @property
    def pixel_scale(self):
        return self.canvas.pixel_scale

    @property
    def scale_unit(self):
        return self.canvas.scale_unit

    @property
    def zones(self):
        # Callers that read .zones expect normalized dicts (the saved schema).
        return self.get_zones()

    @property
    def original_frame_size(self):
        return self.canvas.original_frame_size

    # ── public API ─────────

    def get_zones(self):
        W, H = self.canvas.frame_w or 1, self.canvas.frame_h or 1
        return [self._px_to_norm(z, W, H) for z in self.canvas.zones]

    def set_zones(self, zones):
        W, H = self.canvas.frame_w or 1, self.canvas.frame_h or 1
        px = [self._norm_to_px(z, W, H) for z in (zones or []) if isinstance(z, dict)]
        self.canvas._undo_stack.clear()
        self.canvas._redo_stack.clear()
        self.canvas.zones = px
        self.canvas.set_selected(None)
        self.canvas.pixel_scale = None
        for z in px:
            # re-derive pixel_scale from a scale zone if present
            if z.get("type") == "scale" and z.get("scale_length") and len(z.get("points", [])) >= 2:
                (a, b) = z["points"][0], z["points"][1]
                d = math.hypot(b[0] - a[0], b[1] - a[1])
                if d > 0:
                    self.canvas.pixel_scale = z["scale_length"] / d
                    self.canvas.scale_unit = z.get("scale_unit", "cm")
        self.canvas.update()
        self._reload_list()
        self._update_undo_ui()
        self._sync_scale_inputs_from_zone()

    def get_zone_names(self):
        return [z.get("name", "") for z in self.canvas.zones]

    def set_frame_callback(self, cb):
        self.get_frame_callback = cb
        self.canvas.set_frame_callback(cb)

    def has_valid_scale(self) -> bool:
        z = self._scale_zone_px()
        if not z or len(z.get("points", [])) < 2:
            return False
        try:
            length = float(z.get("scale_length", 0) or 0)
            a, b = z["points"][0], z["points"][1]
            return length > 0 and math.hypot(b[0] - a[0], b[1] - a[1]) > 0
        except (TypeError, ValueError):
            return False

    def get_scale_meta(self) -> dict:
        z = self._scale_zone_px()
        if not z or len(z.get("points", [])) < 2:
            return {}
        a, b = z["points"][0], z["points"][1]
        px_len = math.hypot(b[0] - a[0], b[1] - a[1])
        try:
            value = float(z.get("scale_length", 0) or 0)
        except (TypeError, ValueError):
            value = 0.0
        unit = z.get("scale_unit") or "cm"
        return {
            "value": value, "unit": unit, "pixel_length": float(px_len),
            "px_per_unit": (px_len / value) if value > 0 else 0.0,
            "unit_per_px": (value / px_len) if px_len > 0 else 0.0,
        }

    def start_live(self):
        self.canvas.start_live()

    def stop_live(self):
        self.canvas.stop_live()

    # ── normalized ↔ pixel conversion (the storage boundary) ──

    @staticmethod
    def _px_to_norm(z, W, H):
        # Route through coords.to_norm, the one converter the module
        # contract mandates (its mirror below already uses resolve_to_px).
        from source.video.zones.coords import to_norm
        out = dict(z)
        out["points"] = to_norm(z.get("points", []), W, H)
        if len(z.get("center", []) or []) == 2:
            out["center"] = to_norm([z["center"]], W, H)[0]
        if len(z.get("semi_axes", []) or []) == 2:
            out["semi_axes"] = to_norm([z["semi_axes"]], W, H)[0]
        out.pop("radius", None)  # metadata only; points are authoritative
        out["coord_space"] = "normalized"
        out["shape_dim"] = [int(W), int(H)]
        return out

    @staticmethod
    def _norm_to_px(z, W, H):
        cs = z.get("coord_space")
        out = dict(z)
        out["points"] = resolve_to_px(z.get("points", []), W, H, cs)
        if len(z.get("center", []) or []) == 2:
            out["center"] = resolve_to_px([z["center"]], W, H, cs)[0]
        if len(z.get("semi_axes", []) or []) == 2:
            out["semi_axes"] = resolve_to_px([z["semi_axes"]], W, H, cs)[0]
        out["coord_space"] = "pixel"
        return out

    def _scale_zone_px(self):
        for z in self.canvas.zones:
            if z.get("type") == "scale":
                return z
        return None

    def _sync_scale_inputs_from_zone(self):
        z = self._scale_zone_px()
        if not z:
            return
        self.scale_value_spin.blockSignals(True)
        self.scale_unit_combo.blockSignals(True)
        try:
            self.scale_value_spin.setValue(float(z.get("scale_length", 0) or 0))
            i = self.scale_unit_combo.findText(z.get("scale_unit") or "cm")
            if i >= 0:
                self.scale_unit_combo.setCurrentIndex(i)
        finally:
            self.scale_value_spin.blockSignals(False)
            self.scale_unit_combo.blockSignals(False)

    # ── Qt lifecycle ───────────────────────────────────────────

    def showEvent(self, event):
        self.canvas.start_live()
        super().showEvent(event)

    def hideEvent(self, event):
        self.canvas.stop_live()
        super().hideEvent(event)

    # ── UI ─────────────────────────────────────────────────────

    def _build_hints_bar(self):
        """A rich-text control cheat-sheet pinned under the canvas, the always
        visible legend for draw / select / move / rotate / resize / scale /
        delete. Keys in blue, values grey (mirrors the reference editor)."""
        pairs = [
            ("Draw", "pick a shape, then drag on the video"),
            ("Select", "click a zone (or its list row)"),
            ("Move", "drag the zone body"),
            ("Rotate", "Ctrl + drag  ·  [ ] = ±5°"),
            ("Resize", "drag the square handles"),
            ("Scale", "set value + unit, draw the line"),
            ("Delete", "select + Del  ·  Undo Ctrl+Z"),
        ]
        html = ' <span style="color:#3a3a44">·</span> '.join(
            f'<b style="color:#6ea8fe">{k}</b> '
            f'<span style="color:#c8c8d0">{v}</span>' for k, v in pairs)
        bar = QtWidgets.QLabel(html)
        bar.setTextFormat(Qt.TextFormat.RichText)
        bar.setWordWrap(True)
        bar.setStyleSheet(
            "background:#23232b;border:1px solid #3a3a44;border-radius:4px;"
            "padding:4px 8px;font-size:10px;")
        return bar

    def _build_ui(self):
        # Outer: [canvas | tool panel] row on top, cheat-sheet bar on the bottom.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(5)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        outer.addLayout(layout, stretch=1)
        layout.addWidget(self.canvas, stretch=1)

        right = QtWidgets.QWidget()
        # Wide enough for two add-zone columns and for zone names to read
        # without truncation.
        right.setFixedWidth(196)
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        rl.addWidget(QtWidgets.QLabel("Zones:"))
        self.zone_list = QtWidgets.QListWidget()
        self.zone_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.zone_list.currentRowChanged.connect(self._on_list_row)
        self.zone_list.itemDoubleClicked.connect(self._rename)
        rl.addWidget(self.zone_list, stretch=1)

        # Two columns throughout the panel, three across leaves each
        # button ~60 px at this width and clips "Paste"; one extra row is
        # the cheaper cost.
        edit_grid = QtWidgets.QGridLayout()
        edit_grid.setSpacing(3)
        self._btn_undo = self._mini_btn("↶ Undo", self._undo)
        self._btn_undo.setToolTip("Undo the last zone edit")
        self._btn_redo = self._mini_btn("Redo ↷", self._redo)
        self._btn_redo.setToolTip("Redo the edit that was just undone")
        edit_grid.addWidget(self._btn_undo, 0, 0)
        edit_grid.addWidget(self._btn_redo, 0, 1)
        edit_buttons = [
            ("Copy", self.canvas.copy_selected, "Copy the selected zone"),
            ("Paste", self.canvas.paste_clipboard, "Paste the copied zone"),
            ("Dup", self.canvas.duplicate_selected,
             "Duplicate the selected zone in place"),
        ]
        for i, (lbl, slot, tip) in enumerate(edit_buttons):
            b = self._mini_btn(lbl, slot)
            b.setToolTip(tip)
            # An odd trailing button spans both columns rather than leaving a
            # hole beside it.
            span = 2 if i == len(edit_buttons) - 1 and i % 2 == 0 else 1
            edit_grid.addWidget(b, 1 + i // 2, i % 2, 1, span)
        rl.addLayout(edit_grid)

        rl.addWidget(QtWidgets.QLabel("Add zone:"))
        # Two columns: five stacked buttons cost ~145 px of the panel's
        # height, which is height the zone list needs to show names in full.
        add_grid = QtWidgets.QGridLayout()
        add_grid.setSpacing(3)
        add_buttons = [
            ("Rectangle", "#5b7e8f", "#fff", lambda: self._start('rect'),
             "Drag a rectangular zone"),
            ("Ellipse", "#6d8b74", "#fff", lambda: self._start('ellipse'),
             "Drag an elliptical zone"),
            ("Polygon", "#7a6f9b", "#fff", self._start_regpoly,
             "Drop a regular polygon, you pick the number of sides"),
            ("Manual Poly", "#8b7355", "#fff", lambda: self._start('manpoly'),
             "Click each corner in turn; right-click or double-click to close"),
            ("Line", "#a08458", "#fff", lambda: self._start('line'),
             "Draw a line zone, the target for facing_line triggers"),
        ]
        for i, (label, color, fg, slot, tip) in enumerate(add_buttons):
            b = QtWidgets.QPushButton(label)
            b.setStyleSheet(f"QPushButton{{background:{color};color:{fg};"
                            f"font-size:10px;border:none;border-radius:3px;}}")
            b.setFixedHeight(25)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            span = 2 if i == len(add_buttons) - 1 and i % 2 == 0 else 1
            add_grid.addWidget(b, i // 2, i % 2, 1, span)
        rl.addLayout(add_grid)

        # Scale: real-world length + unit + draw button (no popups).
        srow = QtWidgets.QHBoxLayout()
        srow.setSpacing(3)
        self.scale_value_spin = NumericLineEdit(decimals=2)
        self.scale_value_spin.setRange(0.0, 100000.0)
        self.scale_value_spin.setValue(10.0)
        self.scale_value_spin.setFixedHeight(24)
        self.scale_value_spin.setToolTip("Real-world length of the scale line you draw")
        srow.addWidget(self.scale_value_spin, stretch=1)
        self.scale_unit_combo = QtWidgets.QComboBox()
        self.scale_unit_combo.addItems(["cm", "mm", "m"])
        self.scale_unit_combo.setFixedHeight(24)
        # 56 px clipped the text behind the drop-arrow, the unit was
        # effectively unreadable.
        self.scale_unit_combo.setMinimumWidth(78)
        srow.addWidget(self.scale_unit_combo)
        rl.addLayout(srow)
        # Editing the value/unit AFTER a scale line exists must rewrite the
        # committed scale zone (not only prime the next 'Draw Scale').
        self.scale_value_spin.valueChanged.connect(self._on_scale_input_changed)
        self.scale_unit_combo.currentTextChanged.connect(self._on_scale_input_changed)
        # The two "start a geometry operation" buttons share a row: both put
        # the canvas into a mode rather than acting on the selection, and
        # stacked full-width they cost the zone list another ~60 px.
        draw_row = QtWidgets.QHBoxLayout()
        draw_row.setSpacing(3)
        scale_btn = QtWidgets.QPushButton("Draw Scale")
        scale_btn.setStyleSheet("QPushButton{background:#d4a574;color:#1a1a1a;"
                                "font-size:10px;border:none;border-radius:3px;}")
        scale_btn.setFixedHeight(25)
        scale_btn.setToolTip(
            "Draw a line of the length entered above to calibrate px → real units")
        scale_btn.clicked.connect(self._start_scale)
        draw_row.addWidget(scale_btn)
        self.attach_btn = QtWidgets.QPushButton("Attach edges")
        self.attach_btn.setStyleSheet(
            "QPushButton{background:#E65100;color:white;font-size:10px;"
            "font-weight:bold;border:none;border-radius:3px;}")
        self.attach_btn.setFixedHeight(25)
        self.attach_btn.setToolTip("Snap the selected zones' touching edges together")
        self.attach_btn.clicked.connect(self.canvas.attach_edges)
        draw_row.addWidget(self.attach_btn)
        rl.addLayout(draw_row)

        # Two columns, like the add-zone block above. Stacked one-per-row these
        # four cost ~115 px that the zone list needs to show names in full.
        act_grid = QtWidgets.QGridLayout()
        act_grid.setSpacing(3)
        for i, (label, color, slot, tip) in enumerate([
            ("Delete", "#a85454", self.canvas.delete_selected,
             "Delete the selected zone"),
            ("Cancel", "#6c757d", self.canvas.cancel_mode,
             "Abort the drawing mode in progress"),
            ("Fit view", "#5e8b88", self.canvas._reset_view,
             "Reset zoom and pan to fit the whole frame"),
            ("Refresh", "#5e7d8b", self.canvas.refresh_background,
             "Grab a fresh frame from the camera as the backdrop"),
        ]):
            b = QtWidgets.QPushButton(label)
            b.setStyleSheet(f"QPushButton{{background:{color};color:#fff;"
                            f"font-size:10px;border:none;border-radius:3px;}}")
            b.setFixedHeight(25)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            act_grid.addWidget(b, i // 2, i % 2)
        rl.addLayout(act_grid)

        snap = QtWidgets.QHBoxLayout()
        snap.setSpacing(3)
        sl = QtWidgets.QLabel("Snap:")
        sl.setStyleSheet("font-size:10px;color:#888;")
        snap.addWidget(sl)
        self.snap_spin = QtWidgets.QSpinBox()
        self.snap_spin.setRange(1, 50)
        self.snap_spin.setValue(5)
        self.snap_spin.setSuffix("px")
        self.snap_spin.setFixedHeight(24)
        self.snap_spin.valueChanged.connect(self.canvas.set_snap_px)
        snap.addWidget(self.snap_spin)
        self.magnet_btn = QtWidgets.QPushButton("🧲")
        self.magnet_btn.setCheckable(True)
        self.magnet_btn.setFixedSize(30, 24)
        self.magnet_btn.setToolTip("Magnet: snap dragged vertex to a nearby zone vertex")
        self.magnet_btn.toggled.connect(self.canvas.set_magnet)
        snap.addWidget(self.magnet_btn)
        rl.addLayout(snap)

        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("font-size:9px;color:#4CAF50;")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        layout.addWidget(right)

        # Bottom cheat-sheet: always-visible control legend (key : value pairs).
        outer.addWidget(self._build_hints_bar())

        from PySide6.QtGui import QKeySequence, QShortcut
        self._shortcuts = []
        for seq, slot in [("Ctrl+Z", self._undo), ("Ctrl+Y", self._redo),
                          ("Ctrl+Shift+Z", self._redo),
                          ("Ctrl+C", self.canvas.copy_selected),
                          ("Ctrl+V", self.canvas.paste_clipboard),
                          ("Ctrl+D", self.canvas.duplicate_selected)]:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    def _mini_btn(self, label, slot):
        b = QtWidgets.QPushButton(label)
        b.setFixedHeight(24)
        b.setStyleSheet("font-size:10px;")
        b.clicked.connect(slot)
        return b

    # ── control-panel handlers ─────────────────────────────────

    def _set_status(self, txt):
        self.status.setText(txt)

    def _prompt_name(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Zone Name", "Name:")
        if not ok or not name.strip():
            return None
        name = name.strip()
        if any(z.get("name") == name for z in self.canvas.zones):
            QtWidgets.QMessageBox.warning(self, "Error", f"'{name}' already exists")
            return None
        return name

    def _start(self, mode):
        name = self._prompt_name()
        if name:
            self.canvas.begin_mode(mode, name)
            self._set_status({"rect": "Drag rectangle (Esc=cancel)",
                              "ellipse": "Drag ellipse",
                              "manpoly": "Click vertices; right-click / Enter = done",
                              "line": "Press+drag line (Shift=H/V)"}.get(mode, ""))

    def _start_regpoly(self):
        name = self._prompt_name()
        if not name:
            return
        sides, ok = QtWidgets.QInputDialog.getInt(self, "Polygon", "Sides (3-12):", 6, 3, 12)
        if not ok:
            return
        self.canvas.begin_mode('regpoly', name, sides=sides)
        self._set_status("Click center, then click to set size")

    def _start_scale(self):
        try:
            val = float(self.scale_value_spin.value())
        except (TypeError, ValueError):
            val = 0.0
        if val <= 0:
            self._set_status("Set a positive scale value before drawing.")
            return
        self.canvas._scale_real = val
        self.canvas._scale_unit_input = self.scale_unit_combo.currentText() or "cm"
        self.canvas.begin_mode('scale', "Scale")
        self._set_status("Press+drag the calibration line (Shift = H/V)")

    def _on_scale_input_changed(self, *args):
        """Value/unit spin changed → rewrite the existing scale zone's
        ``scale_length``/``scale_unit`` and re-derive ``pixel_scale`` from the
        drawn line, so editing the number after (or while) creating the zone
        actually takes effect. Also primes the next 'Draw Scale'."""
        try:
            val = float(self.scale_value_spin.value())
        except (TypeError, ValueError):
            return
        unit = self.scale_unit_combo.currentText() or "cm"
        # Prime the draw-time defaults so a fresh scale uses the new value.
        self.canvas._scale_real = val
        self.canvas._scale_unit_input = unit
        z = self._scale_zone_px()
        if z is None or val <= 0:
            return
        z["scale_length"] = val
        z["scale_unit"] = unit
        pts = z.get("points") or []
        if len(pts) >= 2:
            d = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            if d > 0:
                self.canvas.pixel_scale = val / d
                self.canvas.scale_unit = unit
        self.canvas.update()
        self.canvas.zones_changed.emit()
        self._set_status(f"Scale: {val:g} {unit}")

    def _undo(self):
        self.canvas.undo()
        self._update_undo_ui()

    def _redo(self):
        self.canvas.redo()
        self._update_undo_ui()

    def _update_undo_ui(self):
        u, r = self.canvas.undo_depth()
        self._btn_undo.setEnabled(u > 0)
        self._btn_redo.setEnabled(r > 0)
        self._btn_undo.setText(f"↶ Undo ({u})" if u else "↶ Undo")
        self._btn_redo.setText(f"Redo ↷ ({r})" if r else "Redo ↷")

    def _rename(self, item):
        row = self.zone_list.row(item)
        if row < 0 or row >= len(self.canvas.zones):
            return
        old = self.canvas.zones[row].get("name", "")
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Zone", "New name:", QtWidgets.QLineEdit.EchoMode.Normal, old)
        if ok and new.strip():
            self.canvas.set_selected(row)
            self.canvas.rename_selected(new)
            self._reload_list()

    def _on_list_row(self, row):
        self.canvas.set_selected(row if 0 <= row < len(self.canvas.zones) else None)

    def _on_canvas_selection(self, idx):
        if self.zone_list.currentRow() != idx:
            self.zone_list.blockSignals(True)
            self.zone_list.setCurrentRow(idx)
            self.zone_list.blockSignals(False)

    def _on_canvas_zones_changed(self):
        self._reload_list()
        self._update_undo_ui()
        self._sync_scale_inputs_from_zone()
        self.zones_changed.emit()

    def _reload_list(self):
        self.zone_list.blockSignals(True)
        self.zone_list.clear()
        color_idx = 0
        for z in self.canvas.zones:
            zt = z.get("type", "polygon")
            item = QtWidgets.QListWidgetItem(f"{ZONE_ICONS.get(zt, '?')} {z.get('name','')}")
            if zt == "scale":
                item.setForeground(QColor('#E91E63'))
            else:
                item.setForeground(QColor(ZONE_COLORS[color_idx % len(ZONE_COLORS)]))
                color_idx += 1
            self.zone_list.addItem(item)
        sel = self.canvas.selected_idx
        if sel is not None and 0 <= sel < self.zone_list.count():
            self.zone_list.setCurrentRow(sel)
        self.zone_list.blockSignals(False)


__all__ = ["ZoneEditorWidget"]
