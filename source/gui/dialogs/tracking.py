"""Tracking dialogs, single module.

Contains:
  * TrackerCalibrationDialog, blob-tracking calibration sweep.
"""
import cv2
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from source.video.tracking.blob import BlobTracker
from source.video.tracking.smoothing import TrackingEnhancer
from source.log import get_logger
from source.gui.theme import THEME as _TOK
from source.gui.widgets.common import PRESET_NAMES, SUBJECT_PRESETS

_MUTED = _TOK.palette.text_dim     # status / hint labels

# Gap between the preview frames kept for the area estimate. Long enough that
# a mouse ambling at a few cm/s has left its previous outline, short enough
# that four of them span under two seconds of the operator's time.
_AREA_SAMPLE_SPACING_NS = 400_000_000



def _np_to_qpixmap(arr, target=None):
    """Convert a numpy image (BGR colour or 2-D grayscale uint8) to a QPixmap,
    optionally scaled to ``target`` (a ``QSize``) keeping aspect ratio.
    Used by the live calibration preview.
    """
    a = np.ascontiguousarray(arr)
    if a.ndim == 2:
        h, w = a.shape
        qimg = QtGui.QImage(a.data, w, h, w,
                            QtGui.QImage.Format.Format_Grayscale8)
    else:
        h, w = a.shape[:2]
        # Format_BGR888 wraps the BGR buffer directly, no cvtColor and no
        # tobytes() copy per preview tick; ``fromImage`` copies once.
        qimg = QtGui.QImage(a.data, w, h, 3 * w,
                            QtGui.QImage.Format.Format_BGR888)
    pix = QtGui.QPixmap.fromImage(qimg)
    if target is not None:
        pix = pix.scaled(target, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                         QtCore.Qt.TransformationMode.FastTransformation)
    return pix


def _bg_str(entry) -> str:
    """Coerce a background entry (str / Path / dict / None) to a path string."""
    if not entry:
        return ""
    if isinstance(entry, dict):
        entry = entry.get("path") or entry.get("file") or ""
    return str(entry)

logger = get_logger()



# =============================================================================
#  tracker_calibration
# =============================================================================

class TrackerCalibrationDialog(QtWidgets.QDialog):
    """Live calibration dialog for blob tracker parameters.

    Shows a split view with camera feed (left) and binary detection mask (right),
    with sliders for real-time parameter tuning.
    """

    def __init__(self, parent, setup_id, get_frame_callback=None,
                 initial_params=None, background_path=None,
                 box_callbacks=None, box_initial_params=None,
                 box_backgrounds=None):
        """
        Args:
            parent: Parent widget.
            box_id: Initially-selected box id.
            get_frame_callback: Callable that returns the latest camera frame
                (already ROI-cropped) for the initial box. Used when only one
                box is being calibrated; otherwise pass ``box_callbacks``.
            initial_params: Optional dict of initial parameter values for box_id.
            background_path: Optional path to existing background image for box_id.
            box_callbacks: Optional ``{box_id: get_frame_callback}`` for ALL
                connected boxes -- enables the per-box dropdown so the user
                can switch boxes without reopening the dialog.
            box_initial_params: Optional ``{box_id: initial_params_dict}``.
            box_backgrounds: Optional ``{box_id: background_path_str}``.
        """
        super().__init__(parent)
        self.setup_id = setup_id
        # Multi-box state, empty dicts when only one box is in scope.
        self._box_callbacks = dict(box_callbacks) if box_callbacks else {}
        if get_frame_callback is not None:
            self._box_callbacks[setup_id] = get_frame_callback
        self._box_initial_params = dict(box_initial_params) if box_initial_params else {}
        if initial_params is not None:
            self._box_initial_params[setup_id] = dict(initial_params)
        self._box_backgrounds = dict(box_backgrounds) if box_backgrounds else {}
        if background_path is not None:
            self._box_backgrounds[setup_id] = background_path
        # Per-box result snapshot, populated as the user clicks Apply per box.
        self._box_results: dict = {}

        self.get_frame_callback = self._box_callbacks.get(setup_id)
        self._initial_params = self._box_initial_params.get(setup_id, {})
        self._result = None
        self._background_frame = None
        # Preview frames kept for the Auto Threshold area estimate, spaced in
        # time so the animal has cleared its own footprint between them,
        # that displacement is what separates it from the static clutter a
        # background-free mask also lights up.
        self._area_samples: deque = deque(maxlen=4)
        self._area_sample_host_ns = 0
        self._background_path = self._box_backgrounds.get(setup_id)
        self._suppress_preset_update = False

        # Internal tracker for preview (not connected to real tracking)
        self._tracker = BlobTracker(setup_id=setup_id)
        self._enhancer = TrackingEnhancer(setup_id)

        self.setWindowTitle(f"Tracker Calibration, Box {setup_id}")
        self.setMinimumSize(800, 550)
        self.resize(960, 620)

        self._build_ui()
        self._apply_initial_params()

        # Load existing background if provided. ``background_path`` may be
        # a dict (canonical form) or a plain path string.
        bg_for_load = self._resolve_bg_for_load(self._background_path)
        if bg_for_load is not None:
            bg = cv2.imread(str(bg_for_load))
            if bg is not None:
                self._background_frame = bg
                self._tracker.set_background(bg)
                h, w = bg.shape[:2]
                self.status_label.setText(f"Status: Background loaded ({w}x{h})")
                self.status_label.setStyleSheet(
                    "color: #4CAF50; font-style: italic; margin-top: 4px;")

        # Live preview timer (15 FPS)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_preview)
        self._timer.start(66)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Compose the tracker-calibration dialog top-to-bottom:

            top row, box selector + preset + Capture BG + Apply-to-all
            video panels, Camera Feed (left) | Detection Mask (right)
            quality box, live success/jitter/area readout + traffic-light dot
            params group, Dark/CLAHE/Adaptive checks + 5 slider rows + status
            bottom row, Reset / Cancel / Apply

        Each section is built by one private helper.
        """
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.addLayout(self._build_top_row())
        main_layout.addLayout(self._build_video_panels(), stretch=1)
        main_layout.addWidget(self._build_quality_box())
        main_layout.addWidget(self._build_params_group())
        main_layout.addLayout(self._build_bottom_buttons())

    # ----- top row (box selector + preset + Capture BG + Apply-to-all) -----

    def _build_top_row(self):
        row = QtWidgets.QHBoxLayout()
        cal_label = QtWidgets.QLabel("Calibrating: Box")
        cal_label.setStyleSheet("font-weight: bold;")
        row.addWidget(cal_label)

        # Box selector, always visible so the user sees which box is
        # being calibrated, even with a single box.
        self.box_combo = QtWidgets.QComboBox()
        if self._box_callbacks:
            for bid in sorted(self._box_callbacks):
                self.box_combo.addItem(str(bid), userData=bid)
        else:
            self.box_combo.addItem(str(self.setup_id), userData=self.setup_id)
        for i in range(self.box_combo.count()):
            if self.box_combo.itemData(i) == self.setup_id:
                self.box_combo.setCurrentIndex(i)
                break
        self.box_combo.currentIndexChanged.connect(self._on_box_changed)
        row.addWidget(self.box_combo)
        row.addSpacing(20)

        row.addWidget(QtWidgets.QLabel("Subject Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(PRESET_NAMES)
        self.preset_combo.setCurrentText("Dark Mouse / Light BG")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        row.addWidget(self.preset_combo)
        row.addSpacing(20)

        self.capture_bg_btn = QtWidgets.QPushButton("Capture Background")
        self.capture_bg_btn.setToolTip(
            "Grab current camera frame as background reference")
        self.capture_bg_btn.clicked.connect(self._capture_background)
        row.addWidget(self.capture_bg_btn)

        # Auto-threshold (Li's method, mousefinder-style): estimate a good
        # threshold from the current frame instead of hand-tuning the slider.
        self.auto_thresh_btn = QtWidgets.QPushButton("Auto Threshold")
        self.auto_thresh_btn.setToolTip(
            "Estimate the detection threshold from the current frame using\n"
            "Li's method. For 'self_norm' mode this works with no background;\n"
            "for the subtraction modes, capture a background first.")
        self.auto_thresh_btn.clicked.connect(self._auto_threshold)
        row.addWidget(self.auto_thresh_btn)

        # Copies camera-level params (CLAHE, blur, morph kernels, bg_mode)
        # to every other connected box. Subject-specific params (threshold,
        # area, detect_dark) stay per-box since they depend on contrast/size.
        self.apply_to_all_btn = QtWidgets.QPushButton("Apply preprocessing to all boxes")
        self.apply_to_all_btn.setToolTip(
            "Copy CLAHE, blur, morph kernels, and bg_mode to every connected\n"
            "box. Subject-specific params (threshold, area, polarity) stay per-box.")
        self.apply_to_all_btn.clicked.connect(self._apply_preprocessing_to_all)
        self.apply_to_all_btn.setVisible(len(self._box_callbacks) > 1)
        row.addWidget(self.apply_to_all_btn)
        row.addStretch()
        return row

    # ----- video panels (Camera Feed | Detection Mask) -----

    def _build_video_panels(self):
        panels = QtWidgets.QHBoxLayout()
        # (group title, attr name, bg color)
        panel_specs = (
            ("Camera Feed",    "camera_label", "#111"),
            ("Detection Mask", "mask_label",   "#000"),
        )
        for title, attr, bg in panel_specs:
            frame = QtWidgets.QGroupBox(title)
            vbox = QtWidgets.QVBoxLayout(frame)
            vbox.setContentsMargins(2, 2, 2, 2)
            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(320, 240)
            label.setStyleSheet(f"background: {bg};")
            vbox.addWidget(label)
            setattr(self, attr, label)
            panels.addWidget(frame)
        return panels

    # ----- quality box (traffic-light dot + 3 metric labels + frozen note) -----

    def _build_quality_box(self):
        group = QtWidgets.QGroupBox("Tracking Quality (last ~120 frames)")
        row = QtWidgets.QHBoxLayout(group)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(12)

        # Traffic-light dot: green/yellow/red per tracker health.
        self._quality_dot = QtWidgets.QLabel("●")
        self._quality_dot.setStyleSheet(f"color: {_MUTED}; font-size: 18pt;")
        self._quality_dot.setToolTip(
            "Green = healthy / Yellow = marginal / Red = poor")
        row.addWidget(self._quality_dot)

        self._quality_success = QtWidgets.QLabel("success: --")
        self._quality_jitter  = QtWidgets.QLabel("jitter: -- px")
        self._quality_area    = QtWidgets.QLabel("area p10/med/p90: -- / -- / --")
        self._quality_frozen  = QtWidgets.QLabel("")
        self._quality_frozen.setStyleSheet("color: #fbbf24;")
        for w in (self._quality_success, self._quality_jitter,
                  self._quality_area, self._quality_frozen):
            w.setStyleSheet(
                w.styleSheet() + " font-family: Consolas, monospace; font-size: 9pt;")
            row.addWidget(w)
        row.addStretch()
        return group

    # ----- params group (3 checkboxes + 5 sliders + status line) -----

    # (label, attr, default-checked)
    _PARAM_CHECKBOXES = (
        ("Dark Animal",        "dark_animal_cb", True),
        ("CLAHE",              "clahe_cb",       True),
        ("Adaptive Threshold", "adaptive_cb",    False),
    )

    # (label, attr_pair, min, max, default, kwargs)
    _PARAM_SLIDERS = (
        ("Threshold:",  ("threshold_slider",  "threshold_val"),  1,   255,    30,  {}),
        ("CLAHE Clip:", ("clahe_clip_slider", "clahe_clip_val"), 10,  100,    30,  {"scale": 10.0}),  # displayed as x/10
        # A mouse contour is 1500–8000 px²: max_area must sit above the
        # animal (it rejects whole-arena blobs), not below it.
        ("Min Area:",   ("min_area_slider",   "min_area_val"),   10,  10000,  150,   {}),
        ("Max Area:",   ("max_area_slider",   "max_area_val"),   100, 200000, 20000, {}),
        ("Blur Size:",  ("blur_slider",       "blur_val"),       1,   15,     5,   {"step": 2, "odd_only": True}),
    )

    def _build_params_group(self):
        group = QtWidgets.QGroupBox("Parameters")
        vbox = QtWidgets.QVBoxLayout(group)
        vbox.setSpacing(4)

        # Checkboxes row.
        checks_row = QtWidgets.QHBoxLayout()
        for label, attr, default in self._PARAM_CHECKBOXES:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_param_changed)
            setattr(self, attr, cb)
            checks_row.addWidget(cb)
        checks_row.addStretch()
        vbox.addLayout(checks_row)

        # Slider rows, table-driven via _PARAM_SLIDERS.
        for label, (slider_attr, val_attr), lo, hi, default, kw in self._PARAM_SLIDERS:
            slider, val_label = self._add_slider_row(vbox, label, lo, hi, default, **kw)
            setattr(self, slider_attr, slider)
            setattr(self, val_attr, val_label)

        self.status_label = QtWidgets.QLabel(
            "Status: No background, click Capture Background")
        self.status_label.setStyleSheet(
            f"color: {_MUTED}; font-style: italic; margin-top: 4px;")
        vbox.addWidget(self.status_label)
        return group

    # ----- bottom buttons (Reset / Cancel / Apply) -----

    def _build_bottom_buttons(self):
        """Reset/Cancel are secondary slate; Apply is the vivid info-gradient CTA."""
        from source.gui.styles import BUTTON_STYLE as _BS, COLORS as _C
        from source.gui.style_builders import button_style as _btn_st

        row = QtWidgets.QHBoxLayout()
        row.addStretch()

        for label, slot in (("Reset",  self._reset_params),
                            ("Cancel", self.reject)):
            b = QtWidgets.QPushButton(label)
            b.setStyleSheet(_btn_st("secondary", height=30))
            b.setMinimumWidth(80)
            b.clicked.connect(slot)
            row.addWidget(b)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.setStyleSheet(_BS.format(
            color=_C['info'], hover_color=_C['info_hover']))
        apply_btn.setMinimumWidth(100)
        apply_btn.setMinimumHeight(32)
        apply_btn.clicked.connect(self._apply)
        row.addWidget(apply_btn)
        return row

    def _add_slider_row(self, parent_layout, label_text, min_val, max_val, default,
                        scale=None, step=1, odd_only=False):
        """Add a labeled slider row and return (slider, value_label)."""
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label_text)
        lbl.setFixedWidth(80)
        row.addWidget(lbl)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setSingleStep(step)
        slider.valueChanged.connect(self._on_param_changed)
        row.addWidget(slider, stretch=1)

        if scale:
            display = f"{default / scale:.1f}"
        else:
            display = str(default)
        val_label = QtWidgets.QLabel(display)
        val_label.setFixedWidth(50)
        val_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(val_label)

        # Store metadata on slider for display updates
        slider.setProperty("scale", scale)
        slider.setProperty("odd_only", odd_only)
        slider.setProperty("val_label", val_label)

        parent_layout.addLayout(row)
        return slider, val_label

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def _get_slider_display(self, slider):
        """Get display string for a slider's current value."""
        val = slider.value()
        scale = slider.property("scale")
        if scale:
            return f"{val / scale:.1f}"
        return str(val)

    def _get_blur_value(self):
        """Get blur kernel size (must be odd)."""
        v = self.blur_slider.value()
        if v % 2 == 0:
            v += 1
        return v

    #: self_norm ("Simple") settings this dialog has no slider for. The
    #: geometry knobs are typed on the settings panel and the ratio is
    #: produced by Auto Threshold; both ride through ``_initial_params`` so
    #: the preview uses them and Apply hands them back instead of dropping
    #: them.
    _SELF_NORM_PASSTHROUGH = ("self_norm_ratio", "self_norm_sigma",
                              "self_norm_smooth_sigma", "self_norm_minsize")

    def _current_params(self):
        """Read current parameter values from UI controls."""
        params = {
            "detect_dark": self.dark_animal_cb.isChecked(),
            "threshold": self.threshold_slider.value(),
            "use_clahe": self.clahe_cb.isChecked(),
            "clahe_clip_limit": self.clahe_clip_slider.value() / 10.0,
            "min_area": self.min_area_slider.value(),
            "max_area": self.max_area_slider.value(),
            "blur_kernel_size": self._get_blur_value(),
            "use_adaptive_threshold": self.adaptive_cb.isChecked(),
        }
        for key in self._SELF_NORM_PASSTHROUGH:
            if key in self._initial_params:
                params[key] = self._initial_params[key]
        return params

    def _set_params_to_ui(self, params):
        """Set UI controls from parameter dict without triggering preset change."""
        self._suppress_preset_update = True
        try:
            if "detect_dark" in params:
                self.dark_animal_cb.setChecked(params["detect_dark"])
            if "threshold" in params:
                self.threshold_slider.setValue(params["threshold"])
            if "use_clahe" in params:
                self.clahe_cb.setChecked(params["use_clahe"])
            if "clahe_clip_limit" in params:
                self.clahe_clip_slider.setValue(int(params["clahe_clip_limit"] * 10))
            if "min_area" in params:
                self.min_area_slider.setValue(params["min_area"])
            if "max_area" in params:
                self.max_area_slider.setValue(params["max_area"])
            if "blur_kernel_size" in params:
                self.blur_slider.setValue(params["blur_kernel_size"])
            if "use_adaptive_threshold" in params:
                self.adaptive_cb.setChecked(params["use_adaptive_threshold"])
        finally:
            self._suppress_preset_update = False
        self._sync_tracker_params()
        self._update_value_labels()

    def _update_value_labels(self):
        """Update all slider value labels."""
        for slider in [self.threshold_slider, self.clahe_clip_slider,
                       self.min_area_slider, self.max_area_slider, self.blur_slider]:
            val_label = slider.property("val_label")
            if val_label:
                val_label.setText(self._get_slider_display(slider))

    def _sync_tracker_params(self):
        """Push current UI params to the internal tracker."""
        p = self._current_params()
        # Include advanced params from initial config (blur_mode, kernels, bg_mode)
        extra = {}
        if self._initial_params:
            for key in ("blur_mode", "bg_mode",
                        "open_kernel_size", "close_kernel_size",
                        *self._SELF_NORM_PASSTHROUGH):
                if key in self._initial_params:
                    extra[key] = self._initial_params[key]
        # Wipe the rolling-quality buffer so the readout reflects only
        # the new params, not an average of old + new.
        if hasattr(self._tracker, "reset_quality_metrics"):
            self._tracker.reset_quality_metrics()
        self._tracker.update_params(
            threshold=p["threshold"],
            min_area=p["min_area"],
            max_area=p["max_area"],
            detect_dark=p["detect_dark"],
            use_clahe=p["use_clahe"],
            clahe_clip_limit=p["clahe_clip_limit"],
            blur_kernel_size=p["blur_kernel_size"],
            use_adaptive_threshold=p["use_adaptive_threshold"],
            **extra,
        )

    def _apply_initial_params(self):
        """Apply initial parameters passed from the settings panel."""
        if self._initial_params:
            self._set_params_to_ui(self._initial_params)
            # Try to match a preset
            self._match_preset_from_current()

    def _match_preset_from_current(self):
        """Check if current params match any preset and update combo."""
        current = self._current_params()
        for name, preset in SUBJECT_PRESETS.items():
            if preset is None:
                continue  # skip Custom
            match = all(current.get(k) == v for k, v in preset.items())
            if match:
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentText(name)
                self.preset_combo.blockSignals(False)
                return
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_preset_changed(self, name):
        """Handle preset combo selection."""
        preset = SUBJECT_PRESETS.get(name)
        if preset is None:
            return  # "Custom" selected, don't change anything
        self._set_params_to_ui(preset)

    def _on_param_changed(self):
        """Handle any parameter slider/checkbox change."""
        self._update_value_labels()
        self._sync_tracker_params()
        # Auto-switch to "Custom" if user manually changed a slider
        if not self._suppress_preset_update:
            self._match_preset_from_current()

    def _capture_background(self):
        """Capture current camera frame as background."""
        if not self.get_frame_callback:
            self.status_label.setText("Status: No camera callback available")
            return
        try:
            frame = self.get_frame_callback()
            if frame is not None:
                self._background_frame = frame.copy()
                self._tracker.set_background(frame)
                self._enhancer.reset()
                h, w = frame.shape[:2]
                self.status_label.setText(f"Status: Background captured ({w}x{h})")
                self.status_label.setStyleSheet("color: #4CAF50; font-style: italic; margin-top: 4px;")
            else:
                self.status_label.setText("Status: No frame available from camera")
                self.status_label.setStyleSheet("color: #f44336; font-style: italic; margin-top: 4px;")
        except Exception as e:
            logger.error(f"Calibration: error capturing background: {e}")
            self.status_label.setText(f"Status: Error, {e}")
            self.status_label.setStyleSheet("color: #f44336; font-style: italic; margin-top: 4px;")

    def _auto_threshold(self):
        """Estimate the threshold from the current frame (Li's method).

        For the background-subtraction modes this sets the fixed-threshold
        slider; for ``self_norm`` it sets the tracker's ratio threshold (no
        slider, the mode is self-calibrating) and reports the value.
        """
        if not self.get_frame_callback:
            self.status_label.setText("Status: No camera callback available")
            return
        try:
            frame = self.get_frame_callback()
            if frame is None:
                self.status_label.setText("Status: No frame available from camera")
                self.status_label.setStyleSheet(
                    "color: #f44336; font-style: italic; margin-top: 4px;")
                return
            # Sync the tracker's current params so estimate_threshold runs the
            # same preprocessing (bg_mode / blur / CLAHE) the live preview uses.
            self._on_param_changed()
            thr = self._tracker.estimate_threshold(frame)
            if thr is None:
                self.status_label.setText(
                    "Status: Capture a background first (needed for this mode)")
                self.status_label.setStyleSheet(
                    "color: #f44336; font-style: italic; margin-top: 4px;")
                return
            if self._tracker.bg_mode != "self_norm":
                self.threshold_slider.setValue(int(round(thr)))
                msg = f"Auto threshold = {int(round(thr))} (Li)"
            else:
                # The ratio has no slider, so _initial_params is where it
                # lives for the rest of this dialog: _sync_tracker_params
                # re-pushes from there on every later edit, and Apply hands
                # it back via _current_params.
                self._initial_params["self_norm_ratio"] = float(thr)
                msg = f"Auto contrast ratio = {thr:.3f} (Li)"
            # Same preprocessing, and the last couple of seconds of preview
            # alongside the current frame: propose an area window around the
            # animal so the operator isn't left guessing pixel counts after
            # the threshold moved under them. The older frames are what let
            # the estimate pick the animal over the arena's own shadows.
            bracket = self._tracker.estimate_area_bracket(
                [*self._area_samples, frame])
            if bracket is not None:
                lo, hi = bracket
                lo = max(self.min_area_slider.minimum(),
                         min(self.min_area_slider.maximum(), int(lo)))
                hi = max(self.max_area_slider.minimum(),
                         min(self.max_area_slider.maximum(), int(hi)))
                self.min_area_slider.setValue(lo)
                self.max_area_slider.setValue(hi)
                msg += f", area {lo}–{hi} px"
            self.status_label.setText(f"Status: {msg}")
            self.status_label.setStyleSheet(
                "color: #4CAF50; font-style: italic; margin-top: 4px;")
        except Exception as e:
            logger.error(f"Calibration: auto-threshold failed: {e}")
            self.status_label.setText(f"Status: Auto-threshold error, {e}")
            self.status_label.setStyleSheet(
                "color: #f44336; font-style: italic; margin-top: 4px;")

    def _reset_params(self):
        """Reset to initial parameters or Black Mouse preset."""
        if self._initial_params:
            self._set_params_to_ui(self._initial_params)
        else:
            self.preset_combo.setCurrentText("Dark Mouse / Light BG")

    def _apply(self):
        """Accept calibration. In multi-box mode this snapshots the current
        box's params and stays open so the user can move to the next box;
        otherwise it closes the dialog."""
        params = self._current_params()
        if self._background_frame is not None:
            try:
                bg_dir = self._resolve_bg_dir()
                if bg_dir is None:
                    QtWidgets.QMessageBox.warning(
                        self, "No project loaded",
                        "Save the project first, backgrounds live inside "
                        "the project's background_images/ folder.")
                else:
                    bg_dir.mkdir(parents=True, exist_ok=True)
                    target = bg_dir / f"box{self.setup_id}.png"
                    cv2.imwrite(str(target), self._background_frame)
                    params["background_path"] = str(target)
                    mw = self._main_window()
                    if mw is not None and hasattr(mw, "_mark_bg_captured"):
                        mw._mark_bg_captured(self.setup_id)
                    logger.info(f"Box {self.setup_id}: BG saved to {target}")
            except Exception as e:
                logger.warning(f"Could not save calibration background: {e}")
        # Snapshot for the per-box result dict.
        self._box_results[self.setup_id] = params
        self._result = params
        if len(self._box_callbacks) <= 1:
            self.accept()
        else:
            self.status_label.setText(
                f"Status: Box {self.setup_id} calibration saved. "
                f"Switch boxes from the dropdown or close the dialog to finish.")
            self.status_label.setStyleSheet(
                "color: #4CAF50; font-style: italic; margin-top: 4px;")

    # ------------------------------------------------------------------
    # Multi-box helpers
    # ------------------------------------------------------------------

    def _on_box_changed(self, _idx: int):
        """User picked a different box from the dropdown -- swap context."""
        new_id = self.box_combo.currentData()
        if new_id is None or new_id == self.setup_id:
            return
        # Snapshot current box's tuned params before leaving so the user
        # doesn't lose work when switching back.
        self._box_results[self.setup_id] = self._current_params()

        self.setup_id = int(new_id)
        self.get_frame_callback = self._box_callbacks.get(self.setup_id)
        self._initial_params = (self._box_results.get(self.setup_id)
                                or self._box_initial_params.get(self.setup_id, {}))
        self._background_path = self._box_backgrounds.get(self.setup_id)
        self._background_frame = None  # forces recapture / reload
        self._area_samples.clear()     # they show the box we just left
        self._tracker = BlobTracker(setup_id=self.setup_id)
        self._enhancer = TrackingEnhancer(self.setup_id)
        self.setWindowTitle(f"Tracker Calibration, Box {self.setup_id}")
        self._apply_initial_params()
        bg_for_load = self._resolve_bg_for_load(self._background_path)
        if bg_for_load is not None:
            bg = cv2.imread(str(bg_for_load))
            if bg is not None:
                self._background_frame = bg
                self._tracker.set_background(bg)

    def _apply_preprocessing_to_all(self):
        """Copy preprocessing params (CLAHE, blur, morph kernels, bg_mode)
        to every other connected box. Subject-specific params stay per-box."""
        src = self._current_params()
        # The self_norm geometry knobs are arena/optics properties, so they
        # copy; the contrast ratio does not; it is per-box lighting.
        share_keys = ("use_clahe", "clahe_clip_limit",
                      "blur_mode", "blur_kernel_size",
                      "bg_mode",
                      "open_kernel_size", "close_kernel_size",
                      "self_norm_sigma", "self_norm_smooth_sigma",
                      "self_norm_minsize")
        for bid in self._box_callbacks:
            if bid == self.setup_id:
                continue
            existing = self._box_results.get(bid, {}).copy()
            for k in share_keys:
                if k in src:
                    existing[k] = src[k]
            self._box_results[bid] = existing
        self.status_label.setText(
            f"Status: preprocessing copied to {len(self._box_callbacks) - 1} other box(es).")
        self.status_label.setStyleSheet(
            "color: #4CAF50; font-style: italic; margin-top: 4px;")

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def _update_preview(self):
        """Timer callback: grab frame, run detection, update both panels."""
        if not self.get_frame_callback:
            return

        try:
            frame = self.get_frame_callback()
            if frame is None:
                return
            host_ns = time.monotonic_ns()
            if host_ns - self._area_sample_host_ns >= _AREA_SAMPLE_SPACING_NS:
                self._area_sample_host_ns = host_ns
                self._area_samples.append(frame.copy())

            # Target size for display labels
            cam_size = self.camera_label.size()
            target = QtCore.QSize(cam_size.width() - 4, cam_size.height() - 4)

            # Run detection pipeline
            success, position, mask = self._tracker.detect_with_mask(frame)

            # Left panel: camera feed with detection overlay
            display_frame = frame.copy()
            smooth_cx, smooth_cy, enh_speed = None, None, 0.0
            timestamp_ns = int(time.time() * 1e9)

            if success and position:
                x, y, w, h = position
                cx, cy = x + w // 2, y + h // 2
                # Raw detection: green rectangle + green dot
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display_frame, (cx, cy), 4, (0, 255, 0), -1)

                # Enhancer smoothing: cyan dot
                if len(frame.shape) == 3:
                    gray_for_enh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_for_enh = frame
                result = self._enhancer.update(
                    cx, cy, 1.0, gray_for_enh, timestamp_ns, detected=True)
                if result is not None:
                    smooth_cx, smooth_cy, vx, vy = result
                    enh_speed = (vx**2 + vy**2) ** 0.5
                    cv2.circle(display_frame, (int(smooth_cx), int(smooth_cy)),
                               4, (255, 255, 0), -1)  # cyan in BGR
            elif self._enhancer._initialized:
                # No detection, try prediction: orange dot
                if len(frame.shape) == 3:
                    gray_for_enh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_for_enh = frame
                result = self._enhancer.update(
                    0, 0, 0.0, gray_for_enh, timestamp_ns, detected=False)
                if result is not None and not self._enhancer.is_tracking_lost():
                    smooth_cx, smooth_cy, vx, vy = result
                    enh_speed = (vx**2 + vy**2) ** 0.5
                    cv2.circle(display_frame, (int(smooth_cx), int(smooth_cy)),
                               5, (0, 140, 255), -1)  # orange in BGR

            self.camera_label.setPixmap(_np_to_qpixmap(display_frame, target))

            # Live tracking-quality readout
            self._refresh_quality_panel()

            # self_norm has no background by design, the mask panel and status
            # must reflect its live detection, not a "capture a background"
            # placeholder (which would make the mode's only tuning screen dead).
            _needs_bg = (self._tracker.background_gray is None
                         and self._tracker.bg_mode != "self_norm")

            # Right panel: binary mask
            if _needs_bg:
                # No background: show placeholder text
                placeholder = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.putText(placeholder, "No background", (20, placeholder.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 180, 2)
                self.mask_label.setPixmap(_np_to_qpixmap(placeholder, target))
            else:
                self.mask_label.setPixmap(_np_to_qpixmap(mask, target))

            # Status line (setText per tick is needed, the readout
            # changes; the stylesheet only changes with the colour).
            if _needs_bg:
                self.status_label.setText("Status: No background, click Capture Background")
                self._style_preview_status(_MUTED)
            elif success and position:
                x, y, w, h = position
                cx, cy = x + w // 2, y + h // 2
                area = w * h
                smooth_info = ""
                if smooth_cx is not None:
                    smooth_info = (f"  |  Smooth: ({int(smooth_cx)}, {int(smooth_cy)})"
                                   f"  Speed: {enh_speed:.0f} px/s")
                self.status_label.setText(
                    f"Status: Detected, area {area} px  Raw: ({cx}, {cy}){smooth_info}")
                self._style_preview_status("#4CAF50")
            elif smooth_cx is not None:
                self.status_label.setText(
                    f"Status: Predicted, ({int(smooth_cx)}, {int(smooth_cy)})"
                    f"  Speed: {enh_speed:.0f} px/s")
                self._style_preview_status("#FF9800")
            else:
                self.status_label.setText("Status: No detection")
                self._style_preview_status("#FF9800")

        except Exception as e:
            logger.debug(f"Calibration preview error: {e}")

    def _style_preview_status(self, color: str) -> None:
        """Restyle the preview status label only when its colour changes,
        this runs from the 15 FPS preview tick, and an unconditional
        setStyleSheet per tick forces a Qt style recompute."""
        if getattr(self, "_preview_status_color", None) == color:
            return
        self._preview_status_color = color
        self.status_label.setStyleSheet(
            f"color: {color}; font-style: italic; margin-top: 4px;")

    # ------------------------------------------------------------------
    # Background storage helpers, talk to MainWindowBase via the parent
    # chain so per-config backgrounds end up next to the JSON file.
    # ------------------------------------------------------------------

    def _main_window(self):
        """Walk up the parent chain to find the MainWindow (which owns
        active_config_path / _background_images_dir). Returns None when the dialog was
        opened without a parent (unit tests)."""
        w = self.parent()
        while w is not None:
            if hasattr(w, "_background_images_dir") and callable(w._background_images_dir):
                return w
            w = w.parent() if hasattr(w, "parent") else None
        return None

    def _resolve_bg_dir(self) -> Optional[Path]:
        """Return the project's ``background_images/`` directory, or None
        when no project is loaded (capture is then refused)."""
        mw = self._main_window()
        if mw is not None:
            return mw._background_images_dir()
        return None

    def _resolve_bg_for_load(self, entry) -> Optional[Path]:
        """Resolve a BG path to an existing Path, or None if nothing on disk.

        Mostly a passthrough: the path handed in is normally
        ``<project>/background_images/box{N}.png``.
        """
        if not entry:
            return None
        s = _bg_str(entry)
        if s and Path(s).exists():
            return Path(s)
        # Fall back to the project folder lookup if the entry was just a
        # name/relative path.
        bg_dir = self._resolve_bg_dir()
        if bg_dir is not None and s:
            cand = bg_dir / Path(s).name
            if cand.exists():
                return cand
        return None

    def _refresh_quality_panel(self):
        """Pull rolling tracking-quality metrics from the tracker and update
        the readout row. Called on every preview tick (~15 Hz)."""
        if not hasattr(self._tracker, "get_quality_metrics"):
            return
        m = self._tracker.get_quality_metrics()
        n = m.get("frames_in_window", 0)
        if n == 0:
            self._quality_dot.setStyleSheet(f"color: {_MUTED}; font-size: 18pt;")
            self._quality_success.setText("success: --")
            self._quality_jitter.setText("jitter: -- px")
            self._quality_area.setText("area p10/med/p90: -- / -- / --")
            self._quality_frozen.setText("")
            return
        sr = m["success_rate"]
        jit = m["jitter_px"]
        # green: >= 90% success AND jitter < 3 px (typical good operating point)
        # red:   <  70% success OR jitter > 8 px
        # yellow: anything in between
        if sr >= 0.90 and jit < 3.0:
            color = "#22c55e"  # green
        elif sr < 0.70 or jit > 8.0:
            color = "#ef4444"  # red
        else:
            color = "#fbbf24"  # yellow
        self._quality_dot.setStyleSheet(f"color: {color}; font-size: 18pt;")
        self._quality_success.setText(f"success: {sr * 100:5.1f}%  ({n} frames)")
        self._quality_jitter.setText(f"jitter: {jit:5.2f} px")
        self._quality_area.setText(
            f"area p10/med/p90: {m['area_p10']:.0f} / {m['area_median']:.0f} / {m['area_p90']:.0f} px"
        )
        # A Max Area below the animal is the most common mis-calibration and
        # is otherwise invisible: the mask shows the animal, the box sits on a
        # speck, and the area readout stays "--" because nothing passed the
        # filter. Say it outright, and say what to raise Max Area to.
        oversize = m.get("rejected_too_large", 0)
        if m.get("tracked_out_of_range"):
            # Strongest signal there is: the bounds are wrong AND we are
            # currently relying on them being soft to track anything at all.
            # Report the area of the blob actually being tracked, not the
            # largest REJECTED one: a Min Area set above the animal rejects
            # nothing for being too large, so that number would read 0 px and
            # tell the operator nothing.
            area = m.get("out_of_range_area", 0.0)
            self._quality_frozen.setText(
                f"[tracking a blob OUTSIDE the area range, Min/Max Area is "
                f"mis-set. The blob being tracked is {area:.0f} px; set Min "
                f"Area below it and Max Area above it.]")
            self._quality_frozen.setStyleSheet("color: #ef4444;")
        elif oversize:
            biggest = m.get("largest_rejected_area", 0.0)
            self._quality_frozen.setText(
                f"[{oversize} blob(s) rejected: larger than Max Area, "
                f"biggest was {biggest:.0f} px. Raise Max Area above it "
                f"if that is the animal.]")
            self._quality_frozen.setStyleSheet("color: #fbbf24;")
        else:
            self._quality_frozen.setStyleSheet("")
            self._quality_frozen.setText(
                "[adaptive bg frozen, animal stationary]"
                if m["adaptive_frozen"] else "")

    def showEvent(self, event):
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start(66)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def get_calibration(self):
        """Return tuned parameters for the currently-selected box, or None
        if cancelled. Single-box callers continue to use just this."""
        return self._result

    def get_calibrations(self) -> dict:
        """Return ``{box_id: params}`` for every box the user calibrated.
        Empty when the dialog was cancelled."""
        return dict(self._box_results)

