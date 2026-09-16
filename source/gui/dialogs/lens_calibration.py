"""Lens-calibration wizard, the popup UI for correcting a camera's distortion.

A guided, hard-to-get-wrong popup. The user holds a printed checkerboard in
front of the camera and moves it around; the wizard captures frames
*automatically* the moment it sees a sharp board in a spot it hasn't covered
yet, shows live coverage, and won't let the user finish until there are enough
diverse views. Solving is one click, and a raw-vs-corrected split confirms the
walls are straight before Save.

Launched from the Camera Connect dialog for one camera. It is handed a
``get_frame`` callback (returns the latest BGR frame from that camera, the
caller subscribes to the live FrameBus so no second capture is opened), the
camera's stable identity key + friendly name, and the machine
:class:`~source.video.cameras.lens.LensCalibrationStore` to save into.

Two pages in a QStackedWidget:
  1. **Capture**: live preview with detected corners drawn on it, a 3x3
     coverage grid, and an auto-capture toggle. Solve unlocks at enough views.
  2. **Review**: the reprojection-error verdict plus a raw-vs-corrected split
     so the user visually confirms before Save.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from source.log import get_logger
from source.video.cameras.lens import (
    BoardSpec,
    Undistorter,
    board_signature,
    find_board,
    sharpness,
    solve,
)

logger = get_logger()

# Calibration works from ~8 views but is more stable and edge-accurate at 15-20.
RECOMMENDED_VIEWS = 18
MIN_VIEWS = 8
COVERAGE_GRID = 3               # 3x3 spatial coverage cells
SHARPNESS_MIN = 30.0           # variance-of-Laplacian floor to reject blur


def _bgr_to_pixmap(img: Optional[np.ndarray],
                   target: Optional[QtCore.QSize] = None) -> QtGui.QPixmap:
    """Frame → pixmap for the live preview. Format_BGR888 wraps the BGR
    buffer directly (no negative-stride flip, no extra QImage copy,
    ``fromImage`` already copies into the pixmap) and the per-tick scale
    uses the fast path; the smooth variant of this pattern is ~14×
    slower (see frame_display.py)."""
    if img is None:
        return QtGui.QPixmap()
    a = np.ascontiguousarray(img)
    if a.ndim == 2:
        h, w = a.shape
        qimg = QtGui.QImage(a.data, w, h, w,
                            QtGui.QImage.Format.Format_Grayscale8)
    else:
        h, w = a.shape[:2]
        qimg = QtGui.QImage(a.data, w, h, w * 3,
                            QtGui.QImage.Format.Format_BGR888)
    pix = QtGui.QPixmap.fromImage(qimg)
    if target is not None:
        pix = pix.scaled(target, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                         QtCore.Qt.TransformationMode.FastTransformation)
    return pix


class LensCalibrationDialog(QtWidgets.QDialog):
    """Interactive checkerboard lens-calibration popup for one camera."""

    def __init__(self, get_frame: Callable[[], Optional[np.ndarray]],
                 identity_key: str, friendly: str, store, parent=None):
        super().__init__(parent)
        self._get_frame = get_frame
        self._identity_key = identity_key or ""
        self._friendly = friendly or "camera"
        self._store = store

        self._board = BoardSpec()
        self._auto = True
        self._captures: List[np.ndarray] = []      # accepted corner sets
        self._signatures: set = set()              # pose fingerprints (dedup)
        self._covered_cells: set = set()           # (cx, cy) for the grid UI
        self._image_size = (0, 0)                  # (w, h) captures were taken at
        self._last_frame: Optional[np.ndarray] = None
        self._profile = None
        self._review_undist: Optional[Undistorter] = None

        self.setWindowTitle(f"Calibrate lens, {self._friendly}")
        self.setMinimumSize(900, 600)
        self.resize(1000, 660)

        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._dispatch_tick)
        self._timer.start(80)   # ~12 fps preview / detection

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        self._stack = QtWidgets.QStackedWidget()
        root.addWidget(self._stack, 1)
        self._stack.addWidget(self._build_capture_page())
        self._stack.addWidget(self._build_review_page())

    def _build_capture_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(page)

        left = QtWidgets.QVBoxLayout()
        self._preview = QtWidgets.QLabel()
        self._preview.setMinimumSize(560, 420)
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet("background:#0b1211; border-radius:6px;")
        left.addWidget(self._preview, 1)
        self._detect_label = QtWidgets.QLabel("Looking for the checkerboard…")
        self._detect_label.setStyleSheet("color:#888; font-style:italic;")
        left.addWidget(self._detect_label)
        lay.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        head = QtWidgets.QLabel("Auto-capturing")
        head.setStyleSheet("font-size:16px; font-weight:bold;")
        right.addWidget(head)
        tip = QtWidgets.QLabel(
            "Hold a printed checkerboard in front of the camera and move it "
            "slowly, corners, centre, tilted toward and away. Frames are kept "
            "automatically when the board is sharp and in a new spot.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#aaa;")
        right.addWidget(tip)

        self._count_label = QtWidgets.QLabel()
        self._count_label.setStyleSheet("font-size:15px;")
        right.addWidget(self._count_label)

        cov_title = QtWidgets.QLabel("FRAME COVERAGE")
        cov_title.setStyleSheet("color:#888; font-size:10px; letter-spacing:1px;")
        right.addWidget(cov_title)
        self._cov_widget = _CoverageGrid(COVERAGE_GRID)
        right.addWidget(self._cov_widget, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        cov_hint = QtWidgets.QLabel(
            "Fill every cell, the corners matter most; that's where the lens "
            "bends the image the hardest.")
        cov_hint.setWordWrap(True)
        cov_hint.setStyleSheet("color:#888; font-size:11px;")
        right.addWidget(cov_hint)

        spec_row = QtWidgets.QHBoxLayout()
        spec_row.addWidget(QtWidgets.QLabel("Board inner corners:"))
        self._cols_spin = QtWidgets.QSpinBox(); self._cols_spin.setRange(3, 20)
        self._cols_spin.setValue(self._board.cols)
        self._rows_spin = QtWidgets.QSpinBox(); self._rows_spin.setRange(3, 20)
        self._rows_spin.setValue(self._board.rows)
        self._sq_spin = QtWidgets.QDoubleSpinBox(); self._sq_spin.setRange(1, 200)
        self._sq_spin.setValue(self._board.square_mm); self._sq_spin.setSuffix(" mm")
        for w in (self._cols_spin, QtWidgets.QLabel("×"), self._rows_spin,
                  QtWidgets.QLabel("sq"), self._sq_spin):
            spec_row.addWidget(w)
        self._cols_spin.valueChanged.connect(self._on_board_changed)
        self._rows_spin.valueChanged.connect(self._on_board_changed)
        self._sq_spin.valueChanged.connect(self._on_board_changed)
        right.addLayout(spec_row)

        right.addStretch(1)

        self._auto_chk = QtWidgets.QCheckBox("Auto-capture")
        self._auto_chk.setChecked(True)
        self._auto_chk.toggled.connect(self._on_auto_toggled)
        right.addWidget(self._auto_chk)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_capture = QtWidgets.QPushButton("Capture now")
        self._btn_capture.clicked.connect(self._capture_now)
        btn_row.addWidget(self._btn_capture)
        self._btn_discard = QtWidgets.QPushButton("Discard last")
        self._btn_discard.clicked.connect(self._discard_last)
        btn_row.addWidget(self._btn_discard)
        right.addLayout(btn_row)

        self._btn_solve = QtWidgets.QPushButton("Solve →")
        self._btn_solve.setStyleSheet(
            "QPushButton { background:#2196F3; color:white; font-weight:bold; "
            "border:none; border-radius:4px; padding:7px; }"
            "QPushButton:hover { background:#1976D2; }"
            "QPushButton:disabled { background:#555; color:#999; }")
        self._btn_solve.clicked.connect(self._solve)
        right.addWidget(self._btn_solve)

        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        right.addWidget(cancel)

        lay.addLayout(right, 2)
        self._refresh_capture_status()
        return page

    def _build_review_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)

        self._review_head = QtWidgets.QLabel()
        self._review_head.setStyleSheet("font-size:16px; font-weight:bold;")
        lay.addWidget(self._review_head)
        self._review_sub = QtWidgets.QLabel()
        self._review_sub.setWordWrap(True)
        self._review_sub.setStyleSheet("color:#aaa;")
        lay.addWidget(self._review_sub)

        split = QtWidgets.QHBoxLayout()
        self._raw_label = QtWidgets.QLabel()
        self._raw_label.setMinimumSize(420, 320)
        self._raw_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._raw_label.setStyleSheet("background:#0b1211; border-radius:6px;")
        self._fix_label = QtWidgets.QLabel()
        self._fix_label.setMinimumSize(420, 320)
        self._fix_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._fix_label.setStyleSheet("background:#0b1211; border-radius:6px;")
        raw_box = QtWidgets.QVBoxLayout()
        raw_box.addWidget(self._raw_label, 1)
        _cap = QtWidgets.QLabel("Raw, distorted")
        _cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        _cap.setStyleSheet("color:#888;")
        raw_box.addWidget(_cap)
        fix_box = QtWidgets.QVBoxLayout()
        fix_box.addWidget(self._fix_label, 1)
        _cap2 = QtWidgets.QLabel("Corrected, straight")
        _cap2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        _cap2.setStyleSheet("color:#4CAF50;")
        fix_box.addWidget(_cap2)
        split.addLayout(raw_box, 1)
        split.addLayout(fix_box, 1)
        lay.addLayout(split, 1)

        btns = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Capture more")
        back.clicked.connect(self._back_to_capture)
        btns.addWidget(back)
        btns.addStretch(1)
        self._btn_save = QtWidgets.QPushButton("Save calibration")
        self._btn_save.setStyleSheet(
            "QPushButton { background:#5b8a3a; color:white; font-weight:bold; "
            "border:none; border-radius:4px; padding:7px 18px; }"
            "QPushButton:hover { background:#4a7530; }")
        self._btn_save.clicked.connect(self._save)
        btns.addWidget(self._btn_save)
        lay.addLayout(btns)
        return page

    # ── capture logic ─────────────────────────────────────────────────────

    def _on_board_changed(self, *_):
        self._board = BoardSpec(cols=self._cols_spin.value(),
                                rows=self._rows_spin.value(),
                                square_mm=self._sq_spin.value())
        if self._captures:   # changing the board invalidates prior detections
            self._captures.clear()
            self._signatures.clear()
            self._covered_cells.clear()
            self._cov_widget.set_covered(self._covered_cells)
            self._refresh_capture_status()

    def _on_auto_toggled(self, on):
        self._auto = bool(on)

    def _tick(self):
        frame = None
        try:
            frame = self._get_frame()
        except Exception as e:
            logger.debug("lens wizard: get_frame failed: %s", e)
        if frame is None:
            return
        self._last_frame = frame
        h, w = frame.shape[:2]
        if self._image_size == (0, 0):
            self._image_size = (w, h)

        corners = find_board(frame, self._board)
        overlay = frame.copy()
        detected = corners is not None
        if detected:
            try:
                import cv2
                cv2.drawChessboardCorners(
                    overlay, (self._board.cols, self._board.rows), corners, True)
            except Exception:
                pass
            self._detect_label.setText(
                f"✓ Board detected, {self._board.cols}×{self._board.rows} corners")
            self._detect_label.setStyleSheet("color:#4CAF50;")
            if self._auto:
                self._maybe_auto_capture(frame, corners)
        else:
            self._detect_label.setText("Looking for the checkerboard…")
            self._detect_label.setStyleSheet("color:#888; font-style:italic;")

        self._preview.setPixmap(_bgr_to_pixmap(overlay, self._preview.size()))

    def _maybe_auto_capture(self, frame, corners):
        sig = board_signature(corners, self._image_size, COVERAGE_GRID)
        if sig in self._signatures:
            return                      # already have this pose, keep it diverse
        if sharpness(frame) < SHARPNESS_MIN:
            return
        self._accept(corners, sig)

    def _capture_now(self):
        if self._last_frame is None:
            return
        corners = find_board(self._last_frame, self._board)
        if corners is None:
            self._detect_label.setText("No board in view, can't capture.")
            self._detect_label.setStyleSheet("color:#FF9800;")
            return
        sig = board_signature(corners, self._image_size, COVERAGE_GRID)
        self._accept(corners, sig)

    def _accept(self, corners, sig):
        self._captures.append(corners)
        self._signatures.add(sig)
        self._covered_cells.add((sig[0], sig[1]))
        self._cov_widget.set_covered(self._covered_cells)
        self._refresh_capture_status()

    def _discard_last(self):
        if not self._captures:
            return
        self._captures.pop()
        self._signatures.clear()
        self._covered_cells.clear()
        for c in self._captures:
            s = board_signature(c, self._image_size, COVERAGE_GRID)
            self._signatures.add(s)
            self._covered_cells.add((s[0], s[1]))
        self._cov_widget.set_covered(self._covered_cells)
        self._refresh_capture_status()

    def _refresh_capture_status(self):
        n = len(self._captures)
        self._count_label.setText(
            f"<b style='font-size:22px;color:#2196F3'>{n}</b> "
            f"of {RECOMMENDED_VIEWS} recommended captured")
        self._btn_solve.setEnabled(n >= MIN_VIEWS)
        self._btn_solve.setText(
            "Solve →" if n >= MIN_VIEWS
            else f"Solve → (need {MIN_VIEWS - n} more)")
        self._btn_discard.setEnabled(n > 0)

    # ── solve & review ─────────────────────────────────────────────────────

    def _solve(self):
        if len(self._captures) < MIN_VIEWS:
            return
        self.setCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            profile = solve(self._captures, self._image_size, self._board)
        except Exception as e:
            self.unsetCursor()
            QtWidgets.QMessageBox.warning(
                self, "Calibration failed",
                f"Could not solve the calibration:\n{e}\n\n"
                "Capture a few more views from different angles and try again.")
            return
        self.unsetCursor()
        self._profile = profile
        self._review_undist = Undistorter(profile, self._image_size, alpha=1.0)
        self._show_review()

    def _show_review(self):
        p = self._profile
        verdict = p.quality
        color = {"excellent": "#4CAF50", "good": "#8BC34A",
                 "poor": "#FF9800"}.get(verdict, "#aaa")
        self._review_head.setText(
            f"Reprojection error: {p.rms_error:.3f} px  ·  "
            f"<span style='color:{color}'>{verdict.upper()}</span>")
        if verdict == "poor":
            self._review_sub.setText(
                "This fit is loose (over ~1 px). It will still help, but for the "
                "best result go back and capture more views, especially with the "
                "board tilted and near the frame corners.")
        else:
            self._review_sub.setText(
                "Confirm the corrected view on the right has straighter walls "
                "than the raw view on the left, then Save. Correction is stored "
                "for this camera and can be toggled off any time.")
        self._btn_save.setEnabled(True)
        self._stack.setCurrentIndex(1)

    def _back_to_capture(self):
        self._stack.setCurrentIndex(0)

    def _tick_review(self):
        if self._last_frame is None or self._review_undist is None:
            return
        raw = self._last_frame
        try:
            fixed = self._review_undist.apply(raw)
        except Exception:
            return
        self._raw_label.setPixmap(_bgr_to_pixmap(raw, self._raw_label.size()))
        self._fix_label.setPixmap(_bgr_to_pixmap(fixed, self._fix_label.size()))

    def _dispatch_tick(self):
        if self._stack.currentIndex() == 0:
            self._tick()
        else:
            try:                        # keep the review split live
                f = self._get_frame()
                if f is not None:
                    self._last_frame = f
            except Exception:
                pass
            self._tick_review()

    def _save(self):
        if self._profile is None:
            return
        if not self._identity_key:
            QtWidgets.QMessageBox.warning(
                self, "Cannot save",
                "This camera has no stable identity, so its calibration can't "
                "be stored. Re-open after the camera is fully connected.")
            return
        try:
            self._store.put(self._identity_key, self._profile,
                            friendly=self._friendly, enabled=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", f"Could not save calibration:\n{e}")
            return
        logger.info("lens calibration saved for %s (rms=%.3f, %d views)",
                    self._identity_key, self._profile.rms_error,
                    self._profile.n_views)
        self.accept()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

class _CoverageGrid(QtWidgets.QWidget):
    """A small grid that lights a cell once the board has been seen there."""

    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self._n = n
        self._covered: set = set()
        self.setFixedSize(n * 34, n * 34)

    def set_covered(self, cells):
        self._covered = set(cells)
        self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        cell = 30
        gap = 4
        for gy in range(self._n):
            for gx in range(self._n):
                x = gx * (cell + gap)
                y = gy * (cell + gap)
                covered = (gx, gy) in self._covered
                p.setBrush(QtGui.QColor("#2196F3" if covered else "#33383f"))
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.drawRoundedRect(x, y, cell, cell, 4, 4)
        p.end()
