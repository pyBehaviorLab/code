"""
Tracking Settings Tab for PyMaze

Configure tracking mode (Blob Detection or DLC) and related settings.
This tab controls the tracking source that feeds into zone-based event triggers.

Features:
- Tracking mode selection (Blob Detection/DLC)
- Blob detection parameters (threshold, area, dark/light animal)
- DLC model configuration
- Body part selection for DLC
- Confidence threshold
- Start/Stop tracking controls
- Background subtraction settings
"""

import copy
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from source.gui.theme import THEME as _TOK
from source.gui.utility import NoWheelComboBox
from source.gui.widgets.numeric_line_edit import NumericLineEdit

_MUTED = _TOK.palette.text_dim     # status-label muted colour

# Simple/self_norm widget defaults. mousefinder's own values, blob.py's
# SELF_NORM_* constants are the owner; these are a GUI-local copy so the panel
# doesn't pull cv2 into its import, and a test asserts the two agree.
_SN_SIGMA = 0.0        # 0 = derive from the frame (height/20)
_SN_AUTO = -1          # "Auto" in the two width boxes; blob.py SELF_NORM_AUTO
_SN_SMOOTH = _SN_AUTO
_SN_MINSIZE = _SN_AUTO


_TIP_COLS = 64          # characters per line; see _wrap_tip


def _wrap_tip(text: str) -> str:
    """Hard-wrap a paragraph into ``<br>``-joined lines.

    Qt honours neither ``width`` attributes nor CSS widths inside tooltip
    markup, so a long rich-text tooltip renders as a single line running off
    the edge of the screen. The breaks have to be baked in. Blank lines in
    ``text`` separate paragraphs.
    """
    return "<br><br>".join(
        "<br>".join(textwrap.wrap(para.strip(), _TIP_COLS))
        for para in text.split("\n\n") if para.strip())


def _tip(title: str, subtitle: str, body: str,
         values: Optional[list[tuple]] = None, footer: str = "") -> str:
    """Compose a structured, multi-paragraph tooltip.

    A one-line hint is enough for a threshold; it is not enough for a knob
    whose effect stays invisible until tracking quietly degrades an hour into
    a session. These carry what the control does, what each end of its range
    costs, and what else it interacts with.
    """
    head = f"<b>{title}</b>"
    if subtitle:
        head += f' <span style="color:{_MUTED}">, {subtitle}</span>'
    parts = [head, _wrap_tip(body)]
    for name, meaning in (values or []):
        lines = textwrap.wrap(f"{name}, {meaning}", _TIP_COLS)
        # Bold the term, but only on the line it actually starts on.
        lines[0] = lines[0].replace(name, f"<b>{name}</b>", 1)
        parts.append("<br>".join(lines))
    if footer:
        parts.append(_wrap_tip(footer))
    return "<br><br>".join(parts)

# Concrete SLEAP model types the folder detector can report (vs "auto"/", ").
_SLEAP_TYPES = ("single", "centroid", "centered_instance", "bottomup",
                "multi_class_topdown", "multi_class_bottomup")

logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-BOX TRACKING COMPONENTS
# =============================================================================

# Default tracking configs directory
from source import paths as app_paths

TRACKING_CONFIGS_DIR = Path(app_paths.tracking_configs_dir)


class MultiSelectZoneButton(QtWidgets.QPushButton):
    """Button that opens a multi-select zone dialog."""

    zones_changed = QtCore.Signal(list)

    def __init__(self, zone_names: List[str] = None, parent=None):
        super().__init__("Select zones...", parent)
        self.zone_names = zone_names or []
        self.selected_zones: List[str] = []
        self.clicked.connect(self._show_dialog)
        self._update_text()

    def _show_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Zones")
        dialog.setMinimumWidth(200)
        layout = QtWidgets.QVBoxLayout(dialog)

        self.zone_checkboxes = {}
        for zone in self.zone_names:
            cb = QtWidgets.QCheckBox(zone)
            cb.setChecked(zone in self.selected_zones)
            self.zone_checkboxes[zone] = cb
            layout.addWidget(cb)

        if not self.zone_names:
            layout.addWidget(QtWidgets.QLabel("No zones defined"))

        btn_row = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.selected_zones = [z for z, cb in self.zone_checkboxes.items() if cb.isChecked()]
            self._update_text()
            self.zones_changed.emit(self.selected_zones)

    def _update_text(self):
        if not self.selected_zones:
            self.setText("Select zones...")
        elif len(self.selected_zones) == 1:
            self.setText(self.selected_zones[0])
        else:
            self.setText(f"{len(self.selected_zones)} zones")
        self.setToolTip(", ".join(self.selected_zones) if self.selected_zones else "Click to select")

    def set_zone_names(self, names: List[str]):
        self.zone_names = names
        self.selected_zones = [z for z in self.selected_zones if z in names]
        self._update_text()

    def get_zones(self) -> List[str]:
        return self.selected_zones.copy()

    def set_zones(self, zones: List[str]):
        self.selected_zones = [z for z in zones if z in self.zone_names]
        self._update_text()


class TrackingSettingsPanel(QtWidgets.QWidget):
    """Tracking settings with box selection and background capture (like AnyMaze)."""

    # Blob params the tracker calibration dialog returns that this panel has
    # no widget for. They are carried as panel state and emitted by
    # get_settings so they reach the live tracker; anything the dialog gains
    # later must be added here or it will be silently discarded on Apply.
    _CALIBRATION_ONLY_KEYS = (
        "use_clahe", "clahe_clip_limit", "clahe_tile_size",
        "use_adaptive_threshold",
        # The Simple/self_norm ratio is produced by Auto Threshold, not typed,
        # so it has no widget, riding here is what lets a calibrated value
        # reach the tracker AND survive save/reload.
        "self_norm_ratio",
    )

    dlc_init_requested = QtCore.Signal(dict)  # emitted when user clicks Initialize DLC

    def __init__(self, parent=None, box_ids: List[int] = None, zone_names: List[str] = None,
                 connected_cameras: Dict[int, Any] = None, get_frame_callbacks: Dict[int, Callable] = None):
        super().__init__(parent)
        self.box_ids = box_ids or []
        self.zone_names = zone_names or []
        self.connected_cameras = connected_cameras or {}
        self.get_frame_callbacks = get_frame_callbacks or {}
        self.box_checkboxes: Dict[int, QtWidgets.QCheckBox] = {}
        # Per-box "Save annotated video" toggle. Default OFF -- saved video
        # is raw frames. ON -- recorder thread burns zone+pose overlays
        # into the saved file using the controller's last_pose snapshot.
        # Named ``box_annotate_checkboxes`` to stay distinct from
        # ``annotate_checkboxes`` (body-part annotation toggles).
        self.box_annotate_checkboxes: Dict[int, QtWidgets.QCheckBox] = {}
        self.box_bg_buttons: Dict[int, QtWidgets.QPushButton] = {}
        self.box_bg_status: Dict[int, QtWidgets.QLabel] = {}
        self.backgrounds: Dict[int, Any] = {}
        # Boxes this panel has already offered to auto-capture a background
        # for. One attempt per box per panel: a retry loop on every chip
        # toggle would be worse than the missing background it is chasing.
        self._bg_autocapture_tried: set = set()
        self.body_parts_list: List[str] = ["center"]
        #: Connected pairs of part NAMES. Auto-filled from the model when
        #: it declares a skeleton (sleap-nn always does; an exported DLC
        #: folder ships none), editable below so a model without one can
        #: still be drawn correctly.
        self.skeleton_edges: List[tuple] = []
        # Blob params the calibration dialog can tune but this panel has no
        # widget for. Without somewhere to live they were dropped on Apply,
        # so the settings that made the preview track never reached the live
        # tracker. See _CALIBRATION_ONLY_KEYS.
        self._blob_calibration_extra: Dict[str, Any] = {}
        # Last real pose-model path picked/loaded. Backs the ``model_path``
        # widget so a transiently-empty field (e.g. after a config load that
        # carried an empty path) can't zero out a known model on the next
        # save. get_settings() falls back to this.
        self.dlc_model_path: str = ""
        # ``True`` while ``_apply_tracking_settings`` is replaying a saved
        # config, short-circuits the mode-change → load-from-model
        # cascade that would otherwise wipe the body-part picker before
        # the saved selection is restored. Reset to False at the end of
        # the load.
        self._loading: bool = False
        # Held while the Simple chip and the BG-mode combo are being kept in
        # step with each other. They encode one fact (bg_mode), so each
        # drives the other, the flag is what stops that being a cycle.
        self._syncing_mode: bool = False
        self._build_ui()

    def _build_ui(self):
        """Compose the tracking-settings panel top-to-bottom.

        Sections, in order:
            box selection       (which boxes + Retake BG)
            output row          (save-annotated-video + smoothing, side by side)
            tracking-mode group (mode chips + Blob/Blob-advanced/DLC widgets)

        The event/trigger authoring (Push-to-MCU + Coord-mapping +
        Event-Triggers) lives in its own "Event & Trigger" tab, built by
        ``build_event_trigger_widget`` and added by ``UnifiedTrackingDialog``.
        Its widgets are still owned by this panel, so ``get_settings`` / load
        continue to read them by name.

        Each section is one private builder.
        """
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        if self.box_ids:
            layout.addWidget(self._build_box_selection_group())
            # Annotation + smoothing are both output options, pair them on one
            # row so the panel reads compactly now the trigger groups have moved.
            out_row = QtWidgets.QHBoxLayout()
            out_row.setContentsMargins(0, 0, 0, 0)
            out_row.setSpacing(6)
            out_row.addWidget(self._build_annotation_toggles_group(), stretch=1)
            out_row.addWidget(self._build_smoothing_group())
            layout.addLayout(out_row)
        else:
            layout.addWidget(self._build_smoothing_group())
        # Tracking mode is the primary content, give it the stretch.
        layout.addWidget(self._build_tracking_mode_group(), stretch=1)

        # Seed the body-part picker with at least "centroid" so the
        # combo is never empty on first show (it gets repopulated by
        # ``_update_zone_body_part_combo`` on every mode change /
        # body_parts load).
        self._update_zone_body_part_combo([])

    def build_event_trigger_widget(self):
        """The "Event & Trigger" tab body: Push-to-MCU gates + coord-mapping +
        the Event-Triggers table. Built once and handed to
        ``UnifiedTrackingDialog`` as a top-level tab; the widgets stay owned by
        this panel so ``get_settings`` / load keep reading them by name."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        # Coord-mapping + Push-to-MCU sit side-by-side: both gate what reaches
        # the MCU at run time, so the layout reflects the conceptual pairing.
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.addWidget(self._build_coord_mapping_group(), stretch=1)
        top_row.addWidget(self._build_mcu_push_group(), stretch=1)
        v.addLayout(top_row)
        v.addWidget(self._build_event_triggers_group(), stretch=1)
        # The mode/body-parts picker feeding the trigger table is seeded by
        # ``_update_zone_body_part_combo`` during ``_build_ui``.
        return w

    def _build_mcu_push_group(self):
        """Two checkboxes that gate what gets pushed to the MCU at run time.

        Both default ON (matches ``TrackingConfig`` dataclass defaults).
        The state is plumbed into per-box ``TrackingConfig.push_zones_to_mcu``
        / ``push_coords_to_mcu`` via ``MainWindowBase._apply_dialog_config``
        when the dialog is accepted; takes effect on the NEXT framework
        start (no live restart of an already-running session).
        """
        group = QtWidgets.QGroupBox("Push to MCU")
        row = QtWidgets.QHBoxLayout(group)
        row.setContentsMargins(6, 4, 6, 4)

        self.push_zones_to_mcu_cb = QtWidgets.QCheckBox("Zone-change events")
        self.push_zones_to_mcu_cb.setChecked(True)
        self.push_zones_to_mcu_cb.setToolTip(
            "On  : intrinsic 'zone_changed' event + per-zone enter/exit "
            "triggers are sent to the MCU on every transition.\n"
            "Off : MCU sees no zone events, useful when the task script "
            "does not react to zones.\n"
            "The body part whose zone occupancy drives the event comes "
            "from the 'Zone part' picker in the pose params row above."
        )

        self.push_coords_to_mcu_cb = QtWidgets.QCheckBox("c.* coordinates")
        self.push_coords_to_mcu_cb.setChecked(True)
        self.push_coords_to_mcu_cb.setToolTip(
            "On  : c.<coord_name> mappings from the zone config are pushed "
            "to the MCU on every accepted pose / blob frame.\n"
            "Off : MCU sees no coordinate updates from the tracker."
        )

        self.push_frame_event_cb = QtWidgets.QCheckBox("Per-frame pose event")
        self.push_frame_event_cb.setChecked(False)
        self.push_frame_event_cb.setToolTip(
            "On  : a silent intrinsic 'frame_event' is sent on EVERY pose "
            "frame (~pose rate), so a task can poll c.* each frame instead "
            "of only on zone-change edges. Not written to the TSV.\n"
            "Off (default): no per-frame event; the task is driven by "
            "'zone_changed' edges only."
        )

        row.addWidget(self.push_zones_to_mcu_cb)
        row.addWidget(self.push_coords_to_mcu_cb)
        row.addWidget(self.push_frame_event_cb)
        row.addStretch()
        return group

    # ----- box selection (row 1: which boxes are in the run) -----

    def _build_box_selection_group(self):
        box_group = QtWidgets.QGroupBox("Boxes")
        row = QtWidgets.QHBoxLayout(box_group)
        row.setContentsMargins(6, 4, 6, 4)

        self.select_all_cb = QtWidgets.QCheckBox("All")
        self.select_all_cb.stateChanged.connect(self._on_select_all)
        row.addWidget(self.select_all_cb)
        row.addWidget(self._vsep())

        for setup_id in self.box_ids:
            cam_id = self.connected_cameras.get(setup_id)
            has_cam = cam_id is not None
            cb = QtWidgets.QCheckBox(f"Box {setup_id}")
            cb.setEnabled(has_cam)
            if not has_cam:
                cb.setStyleSheet(f"color: {_MUTED};")
            cb.stateChanged.connect(self._on_box_changed)
            self.box_checkboxes[setup_id] = cb
            row.addWidget(cb)

        row.addStretch()

        self.retake_bg_btn = QtWidgets.QPushButton("Retake BG")
        self.retake_bg_btn.setToolTip("Re-capture background for all selected boxes")
        self.retake_bg_btn.setMaximumWidth(80)
        self.retake_bg_btn.clicked.connect(self._retake_backgrounds)
        row.addWidget(self.retake_bg_btn)

        self.bg_status_label = QtWidgets.QLabel("")
        self.bg_status_label.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        row.addWidget(self.bg_status_label)
        return box_group

    def _build_annotation_toggles_group(self):
        """Per-box 'Save annotated video' checkboxes. ON -> recorder burns
        zone+pose overlays into the saved file; OFF -> raw frames."""
        annot_group = QtWidgets.QGroupBox("Save annotated video for")
        annot_group.setToolTip(
            "When ON for a box, the saved video has zones + pose dots\n"
            "burned in (good for showcase). Default OFF saves raw frames.")
        row = QtWidgets.QHBoxLayout(annot_group)
        row.setContentsMargins(6, 4, 6, 4)
        for setup_id in self.box_ids:
            has_cam = self.connected_cameras.get(setup_id) is not None
            cb = QtWidgets.QCheckBox(f"Box {setup_id}")
            cb.setEnabled(has_cam)
            if not has_cam:
                cb.setStyleSheet(f"color: {_MUTED};")
            self.box_annotate_checkboxes[setup_id] = cb
            row.addWidget(cb)
        row.addStretch()
        return annot_group

    # ----- smoothing (single checkbox applied to all modes) -----

    def _build_smoothing_group(self):
        """Smooth-tracking toggle wrapped in a compact group so it can sit
        beside the annotation group instead of on its own sparse line."""
        grp = QtWidgets.QGroupBox("Smoothing")
        row = QtWidgets.QHBoxLayout(grp)
        row.setContentsMargins(6, 4, 6, 4)
        self.smooth_tracking_cb = QtWidgets.QCheckBox("Optical Flow + Kalman")
        self.smooth_tracking_cb.setChecked(True)
        self.smooth_tracking_cb.setToolTip(
            "Enable temporal smoothing of tracking output.\n"
            "Uses optical flow to validate detections and a Kalman filter\n"
            "to smooth jitter and predict position during brief occlusions.\n"
            "Works with both Blob and Pose (DLC/SLEAP) modes."
        )
        row.addWidget(self.smooth_tracking_cb)
        row.addStretch()
        return grp

    # ----- tracking mode group (mode chips + Blob/Blob-advanced/DLC) -----

    def _build_tracking_mode_group(self):
        mode_group = QtWidgets.QGroupBox("Tracking Mode")
        vbox = QtWidgets.QVBoxLayout(mode_group)
        vbox.setContentsMargins(6, 16, 6, 8)
        vbox.setSpacing(8)
        vbox.addLayout(self._build_mode_chips_row())
        vbox.addWidget(self._build_blob_widget())
        vbox.addWidget(self._build_blob_advanced_widget())
        vbox.addWidget(self._build_simple_options_widget())
        vbox.addWidget(self._build_dlc_widget())
        # Trailing stretch so the mode rows cluster at the TOP; without it the
        # group's spare height is distributed between rows and the controls
        # (and their separators) stretch apart / look scrambled.
        vbox.addStretch(1)
        return mode_group

    def _build_mode_chips_row(self):
        """Blob / Simple / DLC / SLEAP picker rendered as a chip group. Each
        chip is a checkable QPushButton in a QButtonGroup, so the setChecked
        / toggled API every downstream call site uses keeps working.

        Simple is Blob with ``bg_mode="self_norm"``, not a separate tracker:
        it is a peer chip because that is what it is to the operator, and it
        stays on the blob config path that already round-trips to disk."""
        from source.gui.style_builders import button_style as _btn_st_tm

        def _make_chip(text: str) -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setMinimumHeight(28)
            # Selected -> info-blue gradient pill; unselected -> ghost.
            # Re-applied on toggle by _restyle_mode_chips.
            b.setStyleSheet(_btn_st_tm("ghost", height=28, padding_h=14))
            return b

        self._mode_chip_btn_style = _btn_st_tm
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.mode_blob = _make_chip("Blob")
        self.mode_blob.setChecked(True)
        # "Simple" is not a separate tracker: it is the blob tracker with
        # bg_mode=self_norm, surfaced as a peer chip because that is what it
        # is to the operator, a tracker that needs no background at all.
        self.mode_simple = _make_chip("Simple")
        self.mode_simple.setToolTip(_tip(
            "Simple", "background-free blob tracking",
            "Detects the animal without any reference image. Each frame is "
            "divided by a blurred copy of itself to cancel uneven lighting, "
            "then thresholded automatically. There is no background to "
            "capture, so the whole class of stale-background failures goes "
            "away: no re-take after moving a camera or changing an ROI, no "
            "running-average leakage, and slow drift in room lighting is "
            "simply divided out.",
            [("Pick Simple when", "the animal is reliably darker (or "
                                  "lighter) than the floor, lighting is "
                                  "uneven or drifts, or capturing an "
                                  "empty-arena reference is impractical."),
             ("Pick Blob when", "animal and floor are close in brightness, "
                                "subtracting a real reference separates them "
                                "where a brightness ratio cannot."),
             ("Pick DLC/SLEAP when", "you need body parts or head direction. "
                                     "Simple, like Blob, gives one centre "
                                     "point and no orientation.")],
            "Simple is the Blob tracker with its background mode set to "
            "self_norm, so it keeps the same Kalman smoothing, spatial "
            "gating and zone logic. Retake BG, the BG dropdown and the fixed "
            "Threshold grey out because none of them apply; Min/Max area "
            "still do."))
        self.mode_dlc    = _make_chip("DLC")
        self.mode_sleap  = _make_chip("SLEAP")
        for b in (self.mode_blob, self.mode_simple, self.mode_dlc,
                  self.mode_sleap):
            row.addWidget(b)
        row.addStretch()

        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for b in (self.mode_blob, self.mode_simple, self.mode_dlc,
                  self.mode_sleap):
            self._mode_group.addButton(b)
            b.toggled.connect(self._restyle_mode_chips)
            # Switching to/from Blob changes which trigger conditions can
            # physically be evaluated, a blob has no head-tail axis.
            b.toggled.connect(
                lambda _on: self.refresh_condition_availability())
        self._restyle_mode_chips()
        # _on_mode_changed only wired on the non-default chips (Blob is the
        # implicit default).
        self.mode_dlc.toggled.connect(self._on_mode_changed)
        self.mode_sleap.toggled.connect(self._on_mode_changed)
        self.mode_simple.toggled.connect(self._on_mode_changed)
        return row

    def _build_blob_widget(self):
        """Single-row blob settings: dark-animal toggle, threshold, area
        bounds, Calibrate button."""
        self.blob_widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.blob_widget)
        row.setContentsMargins(0, 2, 0, 0)
        row.setSpacing(8)

        self.detect_dark_cb = QtWidgets.QCheckBox("Dark animal")
        self.detect_dark_cb.setChecked(True)
        self.detect_dark_cb.setToolTip(
            "Check for dark animal on light bedding (default).\n"
            "Uncheck for light/white animal on dark background."
        )
        row.addWidget(self.detect_dark_cb)
        row.addWidget(self._vsep())

        self.threshold_label = QtWidgets.QLabel("Threshold:")
        row.addWidget(self.threshold_label)
        self.threshold_spin = NumericLineEdit()
        self.threshold_spin.setRange(1, 255)
        self.threshold_spin.setValue(30)
        self.threshold_spin.setToolTip("Background subtraction threshold (lower = more sensitive)")
        self.threshold_spin.setMinimumWidth(64)
        self.threshold_spin.setMaximumWidth(80)
        row.addWidget(self.threshold_spin)
        row.addWidget(self._vsep())

        row.addWidget(QtWidgets.QLabel("Min area:"))
        self.min_area_spin = NumericLineEdit()
        self.min_area_spin.setRange(10, 100000)
        self.min_area_spin.setValue(150)
        self.min_area_spin.setSingleStep(10)
        self.min_area_spin.setToolTip("Minimum blob area in pixels")
        self.min_area_spin.setMinimumWidth(74)
        self.min_area_spin.setMaximumWidth(90)
        row.addWidget(self.min_area_spin)

        row.addWidget(QtWidgets.QLabel("Max area:"))
        self.max_area_spin = NumericLineEdit()
        self.max_area_spin.setRange(100, 500000)
        # Above the animal, not below it, see SUBJECT_PRESETS.
        self.max_area_spin.setValue(20000)
        self.max_area_spin.setSingleStep(50)
        self.max_area_spin.setToolTip("Maximum blob area in pixels")
        self.max_area_spin.setMinimumWidth(78)
        self.max_area_spin.setMaximumWidth(96)
        row.addWidget(self.max_area_spin)
        row.addWidget(QtWidgets.QLabel("px"))
        row.addStretch()

        self.calibrate_btn = QtWidgets.QPushButton("Calibrate...")
        self.calibrate_btn.setToolTip("Open live calibration with camera preview")
        self.calibrate_btn.clicked.connect(self._open_calibration)
        row.addWidget(self.calibrate_btn)
        return self.blob_widget

    def _build_blob_advanced_widget(self):
        """Advanced blob row: blur mode/size + background mode + morph
        open/close kernel sizes."""
        self.blob_adv_widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.blob_adv_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        row.addWidget(QtWidgets.QLabel("Blur:"))
        self.blur_mode_combo = QtWidgets.QComboBox()
        self.blur_mode_combo.addItems(["gaussian", "median"])
        self.blur_mode_combo.setToolTip(
            "Gaussian: general smoothing\n"
            "Median: better for salt-and-pepper noise (IR cameras)")
        self.blur_mode_combo.setMinimumWidth(120)  # wide enough for 'gaussian'
        row.addWidget(self.blur_mode_combo)

        self.blur_size_spin = NumericLineEdit()
        self.blur_size_spin.setRange(1, 31)
        self.blur_size_spin.setSingleStep(2)
        self.blur_size_spin.setValue(5)
        self.blur_size_spin.setToolTip("Blur kernel size (odd number)")
        self.blur_size_spin.setMinimumWidth(64)
        self.blur_size_spin.setMaximumWidth(78)
        row.addWidget(self.blur_size_spin)
        row.addWidget(self._vsep())

        row.addWidget(QtWidgets.QLabel("BG:"))
        self.bg_mode_combo = QtWidgets.QComboBox()
        # Every mode the tracker can run is listed, running_avg included. A
        # list that omits one leaves the operator unable to see or change the
        # mode actually in force.
        self.bg_mode_combo.addItems(
            ["static", "running_avg", "mog2", "self_norm"])
        self.bg_mode_combo.setItemData(
            3, "Same as the Simple chip, no background reference at all",
            QtCore.Qt.ItemDataRole.ToolTipRole)
        self.bg_mode_combo.setToolTip(_tip(
            "Background mode", "how the reference frame is maintained",
            "Blob finds the animal by comparing each frame against a "
            "reference. This chooses what that reference is.",
            [("static", "the frame you captured with Retake BG, left "
                        "unchanged. Most predictable, and the mode the "
                        "capture button implies. Best for short sessions."),
             ("running_avg", "slowly adapts the reference, masked by the "
                             "animal so it never fades into it. Handles a "
                             "room that dims over an hour."),
             ("mog2", "a per-pixel statistical model. Copes with the most "
                      "variation, at the cost of a warm-up period and more "
                      "CPU."),
             ("self_norm", "no reference at all; this is what the Simple "
                           "chip selects. Pick the chip instead; it also "
                           "reveals the knobs this mode adds.")],
            "Changing this to self_norm here switches the mode chip to "
            "Simple, because they are the same setting."))
        self.bg_mode_combo.setMinimumWidth(120)
        self.bg_mode_combo.currentTextChanged.connect(self._on_bg_mode_changed)
        row.addWidget(self.bg_mode_combo)
        row.addWidget(self._vsep())

        row.addWidget(QtWidgets.QLabel("Open:"))
        self.open_kernel_spin = NumericLineEdit()
        self.open_kernel_spin.setRange(1, 15)
        self.open_kernel_spin.setSingleStep(2)
        self.open_kernel_spin.setValue(3)
        self.open_kernel_spin.setToolTip(
            "Morphological open kernel (removes noise spots)\n"
            "Smaller = preserves detail, larger = removes more noise")
        self.open_kernel_spin.setMinimumWidth(64)
        self.open_kernel_spin.setMaximumWidth(78)
        row.addWidget(self.open_kernel_spin)

        row.addWidget(QtWidgets.QLabel("Close:"))
        self.close_kernel_spin = NumericLineEdit()
        self.close_kernel_spin.setRange(1, 21)
        self.close_kernel_spin.setSingleStep(2)
        self.close_kernel_spin.setValue(7)
        self.close_kernel_spin.setToolTip(
            "Morphological close kernel (fills body gaps)\n"
            "Larger = fills bigger gaps (good for tails)")
        self.close_kernel_spin.setMinimumWidth(64)
        self.close_kernel_spin.setMaximumWidth(78)
        row.addWidget(self.close_kernel_spin)
        row.addStretch()
        return self.blob_adv_widget

    def _build_simple_options_widget(self):
        """Simple (self_norm) tuning row. Visible only for the Simple chip.

        Ships with mousefinder's own values, since those are what the method
        was tuned with and what makes its masks clean. Each knob can be zeroed
        for the plain divide-and-threshold.
        """
        self.simple_opts_widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.simple_opts_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        row.addWidget(QtWidgets.QLabel("Lighting blur:"))
        self.sn_sigma_spin = NumericLineEdit(decimals=1)
        self.sn_sigma_spin.setRange(0, 400)
        self.sn_sigma_spin.setValue(_SN_SIGMA)
        self.sn_sigma_spin.setToolTip(_tip(
            "Lighting blur", f"default {_SN_SIGMA:g} (automatic)",
            "Width, in pixels, of the blur that each frame is divided by. "
            "That division is the whole trick behind Simple: a heavily "
            "blurred copy of the frame is a map of how the arena is lit, so "
            "dividing by it flattens lamp hot-spots, vignetting and shadows "
            "while leaving the animal, which is far smaller than the blur, "
            "untouched.",
            [("0 = automatic", "one twentieth of the frame height, "
                               "mousefinder's rule. Correct for almost every "
                               "overhead view."),
             ("Too small", "the animal itself starts appearing in the blur, "
                           "so it partly divides itself out and the "
                           "detection thins or breaks up."),
             ("Too large", "the correction stops following real lighting "
                           "gradients and a bright corner can survive as a "
                           "false blob.")],
            "Only override this if the arena fills an unusual fraction of "
            "the frame, a tightly cropped ROI, or a wide shot where the "
            "arena is small in the corner."))
        self.sn_sigma_spin.setMinimumWidth(64)
        self.sn_sigma_spin.setMaximumWidth(80)
        row.addWidget(self.sn_sigma_spin)
        row.addWidget(self._vsep())

        row.addWidget(QtWidgets.QLabel("Texture blur:"))
        self.sn_smooth_spin = NumericLineEdit(decimals=1, auto_value=_SN_AUTO)
        self.sn_smooth_spin.setRange(0, 100)
        self.sn_smooth_spin.setValue(_SN_SMOOTH)
        self.sn_smooth_spin.setToolTip(_tip(
            "Texture blur", "default Auto (scaled to the frame)",
            "A second blur applied after the lighting correction and before "
            "thresholding. Arena texture, bedding, gravel, wire mesh, floor "
            "grain, is fine detail, so it averages away, while the animal "
            "is large enough to survive. Without it, textured floors "
            "produce a scatter of small false blobs.",
            [("Auto", "about a 58th of the frame height, 18 at 1080p, 12 at "
                      "720p, 5 on a small ROI. This reproduces mousefinder's "
                      "10 on the view it was tuned for."),
             ("0 = off", "plain divide-and-threshold. Right for a smooth, "
                         "featureless floor."),
             ("Too high", "a small animal is blurred toward the floor "
                          "brightness and detection becomes intermittent, "
                          "especially when it is still.")],
            "It is an absolute pixel width, which is why Auto scales it: a "
            "fixed 10 is mild at 1080p but on a 570x290 ROI it eroded a "
            "measured mouse from 594 px to 242 px and cost 8 points of "
            "tracking accuracy over a 2700-frame session."
            "\n\nThe threshold is measured on the UNsmoothed image but "
            "applied to the smoothed one. That asymmetry is deliberate: it "
            "pulls detections in toward the dark core of the animal, which "
            "is why the masks come out clean rather than ragged."))
        self.sn_smooth_spin.setMinimumWidth(64)
        self.sn_smooth_spin.setMaximumWidth(80)
        row.addWidget(self.sn_smooth_spin)
        row.addWidget(self._vsep())

        row.addWidget(QtWidgets.QLabel("Speck size:"))
        self.sn_minsize_spin = NumericLineEdit(auto_value=_SN_AUTO)
        self.sn_minsize_spin.setRange(0, 50)
        self.sn_minsize_spin.setValue(_SN_MINSIZE)
        self.sn_minsize_spin.setToolTip(_tip(
            "Speck size", "default Auto (scaled to the frame)",
            "Erodes the thresholded mask by this many pixels, which deletes "
            "anything thinner than that outright, dust, droppings, cable "
            "shadows, the dark line of a door seam, while a solid animal "
            "merely shrinks.",
            [("Auto", "about a 58th of the frame height, matching "
                      "mousefinder's 10 at the scale it was tuned for."),
             ("0 = off", "the morphology Open kernel on the row above still "
                         "removes isolated specks, just less aggressively."),
             ("Too high", "a small or juvenile animal can be eroded away "
                          "entirely and tracking simply stops.")],
            "This one interacts with Min area: erosion shrinks the contour, "
            "so measured blob area drops as you raise it, by roughly a "
            "fifth at 1080p, but well over half at 640x480, because the "
            "value is in absolute pixels. After changing it, press Auto "
            "Threshold in Calibrate...; it re-derives the area bracket from "
            "the eroded mask. Otherwise a Min area that was fine before can "
            "start rejecting the animal."))
        self.sn_minsize_spin.setMinimumWidth(64)
        self.sn_minsize_spin.setMaximumWidth(80)
        row.addWidget(self.sn_minsize_spin)
        row.addWidget(self._vsep())

        self.sn_ratio_label = QtWidgets.QLabel("Contrast: auto")
        self.sn_ratio_label.setToolTip(_tip(
            "Contrast ratio", "read-only, set by Auto Threshold",
            "How much darker than its surroundings a pixel must be to count "
            "as animal, measured on the illumination-corrected image where "
            "1.0 means 'exactly as bright as the local average'. A typical "
            "dark mouse on a light floor lands around 0.85.",
            [("auto", "no value stored yet, so the tracker estimates one "
                      "from the first frame it sees and then holds it. "
                      "Workable, but that frame might be the one where the "
                      "animal was under the hopper."),
             ("A number", "calibrated deliberately and pinned, the tracker "
                          "will not silently re-derive it.")],
            "Set it with Auto Threshold in Calibrate..., which reads it off "
            "the live image using Li's method and also proposes a Min/Max "
            "area window. It is saved with the project and restored on load, "
            "so this is a once-per-rig job, not a once-per-session one."))
        self.sn_ratio_label.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        row.addWidget(self.sn_ratio_label)
        row.addStretch()
        self.simple_opts_widget.setVisible(False)
        return self.simple_opts_widget

    def _refresh_sn_ratio_label(self):
        """Mirror the stored self_norm ratio into the read-only badge."""
        if not hasattr(self, "sn_ratio_label"):
            return
        try:
            r = float(self._blob_calibration_extra.get("self_norm_ratio") or 0.0)
        except (TypeError, ValueError):
            r = 0.0
        self.sn_ratio_label.setText(
            f"Contrast: {r:.3f}" if r > 0 else "Contrast: auto")

    def _build_dlc_widget(self):
        """DLC/SLEAP pose settings: model path + body-parts list + per-row
        Annotate / Marker / Zone-part / Confidence / Resize / Instances /
        Initialize controls. Hidden by default; shown when DLC chip selected."""
        self.dlc_widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(self.dlc_widget)
        vbox.setContentsMargins(0, 2, 0, 0)
        vbox.setSpacing(4)
        vbox.addLayout(self._build_dlc_model_path_row())
        vbox.addWidget(self._build_sleap_options_row())
        vbox.addWidget(self._build_dlc_options_row())
        vbox.addWidget(self._build_pose_colour_row())
        vbox.addWidget(self._build_pose_input_row())
        self.body_parts_label = QtWidgets.QLabel("(Select model to load body parts)")
        self.body_parts_label.setStyleSheet(f"color: {_MUTED}; font-style: italic;")
        vbox.addWidget(self.body_parts_label)
        # What the model's own config declares, family, identities, native
        # scale, crop, channels. Read-only: the file is authoritative for what
        # the model IS, and anything the operator retypes here is a second
        # answer that can disagree with the network.
        self.model_info_label = QtWidgets.QLabel("")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet(f"color: {_MUTED}; font-size: 8pt;")
        vbox.addWidget(self.model_info_label)
        # What the MACHINE is, as opposed to what the model is. Never
        # stored: it belongs to this PC, and a project opened on the Jetson
        # must not carry the workstation's answer.
        self.machine_caps_label = QtWidgets.QLabel("")
        self.machine_caps_label.setStyleSheet(
            f"color: {_MUTED}; font-size: 8pt;")
        vbox.addWidget(self.machine_caps_label)
        vbox.addLayout(self._build_dlc_annotate_row())
        vbox.addLayout(self._build_skeleton_row())
        vbox.addLayout(self._build_dlc_params_row())
        # DLC must be re-initialised when any of these change (model
        # rebuild required). Wire all four signals to ``_mark_dlc_dirty``
        # via tiny adapter methods so the Init button accurately reflects
        # "settings differ from last init".
        self._dlc_initialized = False
        self.dlc_resize_spin.valueChanged.connect(self._on_resize_changed)
        self.dlc_confidence_spin.valueChanged.connect(self._on_dlc_confidence_changed)
        self.pose_instances_spin.valueChanged.connect(self._on_dlc_instances_changed)
        # Engine, precision, device and colour mode are all in the model cache
        # key, so each genuinely needs a rebuild, unlike confidence, which is
        # a post-inference cutoff and applies live.
        self.dlc_model_type_combo.currentTextChanged.connect(
            lambda _t: self._mark_dlc_dirty("Engine changed, re-initialize!"))
        self.dlc_precision_combo.currentTextChanged.connect(
            lambda _t: self._mark_dlc_dirty("Precision changed, re-initialize!"))
        self.dlc_device_combo.currentTextChanged.connect(
            lambda _t: self._mark_dlc_dirty("Device changed, re-initialize!"))
        self.pose_colour_combo.currentTextChanged.connect(
            lambda _t: self._mark_dlc_dirty("Colour mode changed, re-initialize!"))
        # Input mode and window change what the network is FED, so they
        # rebuild, unlike the steering knobs beside them, which change how
        # the result is READ and apply live.
        self.pose_input_mode_combo.currentTextChanged.connect(
            self._on_pose_input_mode_changed)
        for _w in (self.pose_input_w_spin, self.pose_input_h_spin):
            _w.valueChanged.connect(
                lambda *_a: self._mark_dlc_dirty(
                    "Input window changed, re-initialize!"))
            _w.valueChanged.connect(lambda *_a: self._validate_pose_window())
        # The engine identity is (model, runtime, device, precision): whenever
        # one changes, the cached-engine answer changes with it.
        for _w, _sig in ((self.sleap_runtime_combo, "currentTextChanged"),
                         (self.sleap_device_combo, "currentTextChanged"),
                         (self.sleap_fp16_cb, "toggled")):
            getattr(_w, _sig).connect(lambda *_a: self._refresh_engine_cache_state())
        try:
            self.apply_machine_capabilities()
        except Exception as e:
            logger.debug("capability probe skipped: %s", e)
        self.dlc_widget.setVisible(False)
        return self.dlc_widget

    def _build_sleap_options_row(self):
        """SLEAP-only controls: auto-detected model-type badge, paired centroid
        model (top-down), inference runtime / device / fp16. Hidden unless the
        SLEAP chip is active; the centroid picker shows only for top-down."""
        self.sleap_opts_widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.sleap_opts_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.sleap_type_caption = QtWidgets.QLabel("Type:")
        row.addWidget(self.sleap_type_caption)
        self.sleap_type_label = QtWidgets.QLabel("")
        self.sleap_type_label.setToolTip("Auto-detected from the model folder")
        self.sleap_type_label.setStyleSheet(
            "color:#39c5cf; font-weight:600; font-size:10px;")
        row.addWidget(self.sleap_type_label)

        # Paired centroid model (top-down only).
        self.sleap_centroid_widget = QtWidgets.QWidget()
        crow = QtWidgets.QHBoxLayout(self.sleap_centroid_widget)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.setSpacing(4)
        crow.addWidget(self._vsep())
        crow.addWidget(QtWidgets.QLabel("Centroid:"))
        self.sleap_centroid_path = QtWidgets.QLineEdit()
        self.sleap_centroid_path.setPlaceholderText("top-down centroid model folder")
        self.sleap_centroid_path.setMinimumWidth(160)
        crow.addWidget(self.sleap_centroid_path, stretch=1)
        cbtn = QtWidgets.QPushButton("Browse")
        cbtn.clicked.connect(self._browse_sleap_centroid)
        crow.addWidget(cbtn)
        self.sleap_centroid_widget.setVisible(False)
        row.addWidget(self.sleap_centroid_widget, stretch=1)

        # Separates Type from Runtime, so it goes when Type does; otherwise
        # it floats at the left edge with nothing on its left to separate.
        self.sleap_type_sep = self._vsep()
        row.addWidget(self.sleap_type_sep)
        row.addWidget(QtWidgets.QLabel("Runtime:"))
        self.sleap_runtime_combo = QtWidgets.QComboBox()
        self.sleap_runtime_combo.addItems(["auto", "native", "onnx", "tensorrt"])
        self.sleap_runtime_combo.setToolTip(
            "native = PyTorch · onnx/tensorrt = exported engine (Jetson fast path)")
        row.addWidget(self.sleap_runtime_combo)

        row.addWidget(QtWidgets.QLabel("Device:"))
        self.sleap_device_combo = QtWidgets.QComboBox()
        self.sleap_device_combo.addItems(["auto", "cuda", "cpu"])
        row.addWidget(self.sleap_device_combo)

        self.sleap_fp16_cb = QtWidgets.QCheckBox("fp16")
        self.sleap_fp16_cb.setToolTip("Half precision (CUDA only), ~1.5× faster")
        row.addWidget(self.sleap_fp16_cb)

        self.sleap_export_btn = QtWidgets.QPushButton("Export…")
        self.sleap_export_btn.setToolTip(
            "Pre-build the ONNX / TensorRT engine for this model (cached per-PC "
            "so the first Record isn't slow). Pick the runtime first.")
        self.sleap_export_btn.clicked.connect(self._export_sleap_engine)
        row.addWidget(self.sleap_export_btn)
        self.sleap_export_status = QtWidgets.QLabel("")
        self.sleap_export_status.setStyleSheet("font-size:10px;")
        row.addWidget(self.sleap_export_status)
        row.addStretch()

        self.sleap_opts_widget.setVisible(False)
        return self.sleap_opts_widget

    def _build_dlc_options_row(self):
        """DLC-only engine controls, mirroring the SLEAP row above.

        DeepLabCut-Live selects its runner from ``model_type`` and
        ``precision`` in its constructor, so both are part of the model cache
        key: changing one rebuilds rather than being quietly ignored. Hidden
        unless the DLC chip is active, a rig runs one backend at a time, so
        the row can be backend-specific rather than a lowest common
        denominator.
        """
        self.dlc_opts_widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.dlc_opts_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        row.addWidget(QtWidgets.QLabel("Engine:"))
        self.dlc_model_type_combo = QtWidgets.QComboBox()
        # Repopulated from the chosen model by _refresh_dlc_engine_choices:
        # what a folder can be run as is knowable from the folder, and offering
        # three TensorFlow runners for a PyTorch model on a machine with no
        # TensorFlow is how "pytorch does not initialise" looked from here.
        self.dlc_model_type_combo.addItems(["auto"])
        self.dlc_model_type_combo.setToolTip(
            "Which engine runs this model. Pick a model to see its options.\n"
            "Fixed at model construction, so changing it rebuilds the model.")
        row.addWidget(self.dlc_model_type_combo)

        row.addWidget(QtWidgets.QLabel("Precision:"))
        self.dlc_precision_combo = QtWidgets.QComboBox()
        # INT8 is not offered: DeepLabCut-Live's PyTorch runner takes FP16 or
        # FP32 and nothing else, so the third entry could only ever be a
        # silently ignored setting.
        self.dlc_precision_combo.addItems(["FP32", "FP16"])
        self.dlc_precision_combo.setToolTip(
            "Weight precision, for the pytorch engine.\n"
            "FP16 halves weight memory, but it is NOT automatically faster:\n"
            "measured here on an RTX A6000 at 513x315, FP32 ran 17.9 ms/frame\n"
            "and FP16 23.1 ms. Time it on your rig before choosing it.")
        row.addWidget(self.dlc_precision_combo)

        row.addWidget(QtWidgets.QLabel("Device:"))
        self.dlc_device_combo = QtWidgets.QComboBox()
        self.dlc_device_combo.setEditable(True)      # cuda:1 on a multi-GPU host
        self.dlc_device_combo.addItems(["auto", "cuda", "cpu"])
        self.dlc_device_combo.setToolTip(
            "Where the model runs. Editable, type e.g. cuda:1 to pin a box "
            "to the second GPU.")
        self.dlc_device_combo.setMaximumWidth(90)
        row.addWidget(self.dlc_device_combo)
        row.addStretch()

        self.dlc_opts_widget.setVisible(False)
        return self.dlc_opts_widget

    def _build_pose_colour_row(self):
        """Channel mode, applies to whichever backend is active.

        Defaults to following the model's own config. The explicit values
        exist for a model whose config does not state its expectation, and to
        let a one-channel model be fed a grayscale camera at all, a strict
        three-channel expectation refuses a 2-D frame outright.
        """
        self.pose_colour_widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.pose_colour_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel("Colour:"))
        self.pose_colour_combo = QtWidgets.QComboBox()
        self.pose_colour_combo.addItems(["auto", "rgb", "grayscale"])
        self.pose_colour_combo.setToolTip(
            "What the network is fed.\n"
            "auto, follow the model config (recommended)\n"
            "rgb / grayscale, override it for a model whose config is silent.\n"
            "Changing this rebuilds the model.")
        row.addWidget(self.pose_colour_combo)
        row.addStretch()
        self.pose_colour_widget.setVisible(False)
        return self.pose_colour_widget

    # Result of a background export (message, ok) → GUI-thread status update.
    sleap_export_done = QtCore.Signal(str, bool)

    def _export_sleap_engine(self):
        """Pre-build the exported engine for the selected runtime in a
        background thread (export is slow) and report on the status label."""
        runtime = self.sleap_runtime_combo.currentText()
        if runtime not in ("onnx", "tensorrt"):
            QtWidgets.QMessageBox.information(
                self, "Export", "Pick Runtime = onnx or tensorrt first, then "
                "Export builds that engine.")
            return
        model = self.model_path.text().strip()
        if not model or not os.path.exists(model):
            QtWidgets.QMessageBox.warning(self, "Export", "Select a model first.")
            return
        device = self.sleap_device_combo.currentText()
        centroid = self.sleap_centroid_path.text().strip() or None
        # Precision must match what Init will ask for, or this pre-builds an
        # engine into a different cache directory and the first Record still
        # pays the full build.
        precision = "fp16" if self.sleap_fp16_cb.isChecked() else "fp32"
        batch_cap = self._engine_batch_cap()
        peak_thr = self._engine_peak_threshold(model)
        self.sleap_export_btn.setEnabled(False)
        self.sleap_export_status.setText(f"Exporting {runtime}…")
        self.sleap_export_status.setStyleSheet("font-size:10px;color:#e0a95c;")
        try:
            self.sleap_export_done.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.sleap_export_done.connect(self._on_sleap_export_done)

        import threading

        def _work():
            try:
                from source.video.tracking.sleap_export import ensure_exported
                out = ensure_exported(model, runtime, device=device,
                                      centroid_path=centroid,
                                      precision=precision,
                                      max_batch_size=batch_cap,
                                      peak_threshold=peak_thr)
                self.sleap_export_done.emit(f"{runtime} ready", True)
                logger.info("SLEAP export ready: %s", out)
            except Exception as e:
                self.sleap_export_done.emit(f"export failed: {e}", False)
        threading.Thread(target=_work, daemon=True).start()

    def _on_sleap_export_done(self, msg: str, ok: bool):
        self.sleap_export_btn.setEnabled(True)
        self.sleap_export_status.setText(msg)
        self.sleap_export_status.setStyleSheet(
            f"font-size:10px;color:{'#34d399' if ok else '#fb7185'};")

    def _engine_batch_cap(self) -> int:
        """The batch size the engine is built for, from this rig's box count.

        Must match what the pipeline asks for at Init, or Export pre-builds an
        engine the run will not use, the same trap as building it at the wrong
        precision.
        """
        from source.video.tracking.sleap_export import batch_bucket
        return batch_bucket(len(self.box_ids) or 1)

    def _engine_peak_threshold(self, model: str) -> float:
        """The threshold the engine for ``model`` is built with.

        Must match what Init will ask for, or Export pre-builds an engine the
        run never uses, the same trap as building it at the wrong precision.
        Permissive for a fixed-peak family (the host gates instead), the
        operator's value for bottom-up, where the threshold decides how many
        candidates exist at all.
        """
        from source.video.tracking.model_config import ModelInfo
        from source.video.tracking.pose import SLEAPTracker
        from source.video.tracking.sleap_export import PERMISSIVE_PEAK_THRESHOLD
        try:
            family = ModelInfo.read(model).family
        except Exception:
            family = "unknown"
        if family in SLEAPTracker._FIXED_PEAK_FAMILIES:
            return PERMISSIVE_PEAK_THRESHOLD
        return float(getattr(self, "_sleap_peak_threshold", 0.2))

    def _refresh_engine_cache_state(self):
        """Say whether the selected engine is already built.

        An engine build is minutes, and it happens implicitly on the first
        inference if nobody pre-built it. Without this the operator cannot tell
        a fast Init from one that will appear to hang, and on a Jetson, which
        can never reuse a workstation engine, it always will.
        """
        if not hasattr(self, "sleap_export_status"):
            return
        if not (hasattr(self, "mode_sleap") and self.mode_sleap.isChecked()):
            return
        runtime = self.sleap_runtime_combo.currentText()
        model = self.model_path.text().strip()
        if runtime not in ("onnx", "tensorrt") or not model:
            self.sleap_export_status.setText("")
            return
        try:
            from source.video.tracking.sleap_export import is_exported
            cached = is_exported(
                model, runtime,
                device=self.sleap_device_combo.currentText(),
                centroid_path=self.sleap_centroid_path.text().strip() or None,
                precision="fp16" if self.sleap_fp16_cb.isChecked() else "fp32",
                max_batch_size=self._engine_batch_cap(),
                peak_threshold=self._engine_peak_threshold(model))
        except Exception as e:
            logger.debug("engine cache probe failed: %s", e)
            self.sleap_export_status.setText("")
            return
        if cached:
            self.sleap_export_status.setText(f"{runtime} engine cached")
            self.sleap_export_status.setStyleSheet("font-size:10px;color:#34d399;")
        else:
            self.sleap_export_status.setText("not built, first Init builds it")
            self.sleap_export_status.setStyleSheet("font-size:10px;color:#e0a95c;")

    def _browse_sleap_centroid(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select SLEAP centroid model folder",
            os.path.join(app_paths.models_dir, "sleap")
            if os.path.isdir(os.path.join(app_paths.models_dir, "sleap"))
            else app_paths.models_dir)
        if d:
            self.sleap_centroid_path.setText(d)

    def _apply_model_placeholder(self):
        """Describe the model THIS backend loads, not both.

        The field is shared, so the hint has to follow the selected backend;
        naming a DLC exported-model folder while SLEAP is selected is the same
        confusion as showing a DLC control there.
        """
        if not hasattr(self, "model_path"):
            return
        if getattr(self, "mode_sleap", None) is not None and self.mode_sleap.isChecked():
            self.model_path.setPlaceholderText(
                "SLEAP-nn model folder, or a legacy .zip")
        elif getattr(self, "mode_dlc", None) is not None and self.mode_dlc.isChecked():
            self.model_path.setPlaceholderText("DLC exported-model folder")

    def _refresh_sleap_type_badge(self):
        """Update the auto-detected type badge + show the centroid picker only
        for a top-down model. Called on model-path change in SLEAP mode."""
        if not hasattr(self, "sleap_type_label"):
            return
        mt = ""
        path = self.model_path.text().strip()
        if path and self.mode_sleap.isChecked():
            try:
                from source.video.tracking.pose import detect_sleap_model_type
                mt = detect_sleap_model_type(path)
            except Exception:
                mt = "unknown"
        # Nothing detected yet is nothing to say. The placeholder used to be
        # ", ", which drew a bare comma next to "Type:" and read as a broken
        # field rather than an empty one.
        self.sleap_type_label.setText(mt)
        self.sleap_type_label.setVisible(bool(mt))
        for attr in ("sleap_type_caption", "sleap_type_sep"):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(bool(mt))
        is_topdown = mt in ("centroid", "centered_instance", "topdown",
                            "multi_class_topdown")
        self.sleap_centroid_widget.setVisible(is_topdown)

    def _build_dlc_model_path_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Model:"))
        self.model_path = QtWidgets.QLineEdit()
        # Set per backend by _apply_model_placeholder; naming both at once
        # meant the SLEAP panel told the operator about DLC folders.
        self.model_path.setPlaceholderText("exported-model folder")
        self.model_path.textChanged.connect(self._on_model_path_changed)
        row.addWidget(self.model_path, stretch=1)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_model)
        row.addWidget(browse_btn)
        return row

    def apply_machine_capabilities(self, caps=None) -> None:
        """Offer only what this machine can actually run, and say why not.

        An option that is present but unusable ends as a rig doing something
        slower or different from what the dialog says, and the worst of them
        is quiet: onnxruntime-gpu built against the wrong CUDA loads, reports
        success and runs on the CPU.

        Greyed with a reason rather than hidden: a control that has vanished
        reads as a missing feature, while one that says "needs a CUDA GPU"
        teaches the operator something about their machine.
        """
        from source.video.tracking.capability import capabilities
        caps = caps or capabilities()
        self._caps = caps

        if hasattr(self, "sleap_runtime_combo"):
            model = self.sleap_runtime_combo.model()
            for i in range(self.sleap_runtime_combo.count()):
                name = self.sleap_runtime_combo.itemText(i)
                want = {"tensorrt": "tensorrt", "onnx": "onnx"}.get(name)
                ok = True if want is None else caps.can(want)
                item = model.item(i) if hasattr(model, "item") else None
                if item is not None:
                    item.setEnabled(ok)
                self.sleap_runtime_combo.setItemData(
                    i, caps.why_not(want) if (want and not ok) else "",
                    QtCore.Qt.ToolTipRole)

        if hasattr(self, "sleap_fp16_cb"):
            self.sleap_fp16_cb.setEnabled(caps.can("fp16"))
            if not caps.can("fp16"):
                self.sleap_fp16_cb.setToolTip(caps.why_not("fp16"))

        if hasattr(self, "dlc_precision_combo"):
            self.dlc_precision_combo.setEnabled(caps.can("fp16"))
            if not caps.can("fp16"):
                self.dlc_precision_combo.setToolTip(caps.why_not("fp16"))

        if hasattr(self, "machine_caps_label"):
            self.machine_caps_label.setText(caps.summary())

    def _build_pose_input_row(self):
        """How the frame becomes the model's input, and the window it cuts.

        A model trained on crops is trained at one pixel scale, and the
        letterbox, right for a model trained on whole frames, scales that
        away. So the mode is a property of the model, stated here rather than
        assumed.
        """
        self.pose_input_widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.pose_input_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        row.addWidget(QtWidgets.QLabel("Input:"))
        self.pose_input_mode_combo = NoWheelComboBox()
        self.pose_input_mode_combo.addItems(
            ["auto", "letterbox", "full", "crop-track", "dlc-dynamic"])
        self.pose_input_mode_combo.setToolTip(
            "How the box's frame becomes the model's input.\n"
            "auto - read it off the model config (default, and the only\n"
            "  setting that gets a crop-trained model right)\n"
            "letterbox - scale + pad to one shape; right for a model trained\n"
            "  on whole frames, and forced for a fixed-shape export\n"
            "full - hand the frame over unchanged; right for a fully\n"
            "  convolutional runner such as DLC's PyTorch engine\n"
            "crop-track - cut the training window at NATIVE resolution and\n"
            "  follow the animal. Single-instance models only.\n"
            "dlc-dynamic - the same, done by DeepLabCut-Live itself.")
        self.pose_input_mode_combo.setMaximumWidth(110)
        row.addWidget(self.pose_input_mode_combo)

        # OUR following window and its steering, in one container so the
        # labels disappear with the fields. Hiding the widgets alone left
        # "Window:", "x", "Follow conf:" and "min kp:" captioning nothing.
        self.pose_crop_widget = QtWidgets.QWidget()
        crow = QtWidgets.QHBoxLayout(self.pose_crop_widget)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.setSpacing(6)

        crow.addWidget(QtWidgets.QLabel("Window:"))
        self.pose_input_w_spin = NumericLineEdit()
        self.pose_input_w_spin.setRange(0, 4096)
        self.pose_input_w_spin.setValue(0)
        self.pose_input_w_spin.setMaximumWidth(56)
        self.pose_input_w_spin.setToolTip(
            "Training window width in pixels. 0 = take it from the model\n"
            "config. A model can keep its crop in the DATASET rather than the\n"
            "config, and then only the training script knows - type it here.")
        crow.addWidget(self.pose_input_w_spin)
        crow.addWidget(QtWidgets.QLabel("x"))
        self.pose_input_h_spin = NumericLineEdit()
        self.pose_input_h_spin.setRange(0, 4096)
        self.pose_input_h_spin.setValue(0)
        self.pose_input_h_spin.setMaximumWidth(56)
        self.pose_input_h_spin.setToolTip("Training window height in pixels.")
        crow.addWidget(self.pose_input_h_spin)

        # The window SIZE above describes the model's input and matters to
        # either backend. What follows steers OUR crop-track window, which is
        # written for SLEAP single-instance and is not offered to DLC at all
        # (see _DLC_INPUT_MODES). Kept in its own container so it can be
        # hidden as a unit with its labels: leaving "Follow conf:" and
        # "min kp:" on a DLC panel captions settings that backend never reads.
        self.pose_crop_steer_widget = QtWidgets.QWidget()
        srow = QtWidgets.QHBoxLayout(self.pose_crop_steer_widget)
        srow.setContentsMargins(0, 0, 0, 0)
        srow.setSpacing(6)
        crow.addSpacing(10)
        crow.addWidget(self.pose_crop_steer_widget)
        crow = srow
        crow.addWidget(QtWidgets.QLabel("Follow conf:"))
        # decimals=2: without it this is an integer field and a 0.35 the
        # operator typed is stored as 0, which reads as "trust everything".
        self.pose_crop_conf_spin = NumericLineEdit(decimals=2)
        self.pose_crop_conf_spin.setRange(0.0, 1.0)
        self.pose_crop_conf_spin.setSingleStep(0.05)
        self.pose_crop_conf_spin.setValue(0.20)
        self.pose_crop_conf_spin.setMaximumWidth(56)
        self.pose_crop_conf_spin.setToolTip(
            "Confidence at which a keypoint is worth steering the window by.\n"
            "Applies live - it changes how the result is read, not what the\n"
            "network is fed.")
        crow.addWidget(self.pose_crop_conf_spin)

        crow.addWidget(QtWidgets.QLabel("min kp:"))
        self.pose_crop_good_spin = NumericLineEdit()
        self.pose_crop_good_spin.setRange(1, 50)
        self.pose_crop_good_spin.setValue(3)
        self.pose_crop_good_spin.setMaximumWidth(46)
        self.pose_crop_good_spin.setToolTip(
            "Confident keypoints needed before the box counts as tracked.")
        crow.addWidget(self.pose_crop_good_spin)

        self.pose_crop_reacquire_cb = QtWidgets.QCheckBox("re-acquire")
        self.pose_crop_reacquire_cb.setChecked(True)
        self.pose_crop_reacquire_cb.setToolTip(
            "When the window loses the animal, try one more window per frame\n"
            "until it finds it again. Costs no extra inference.")
        crow.addWidget(self.pose_crop_reacquire_cb)
        row.addWidget(self.pose_crop_widget)
        self.pose_crop_steer_widget.setVisible(False)

        # DeepLabCut-Live's OWN dynamic cropping, which is a different
        # mechanism from the window above and belongs to a different backend.
        # It crops to the bounding box of the keypoints it last found and adds
        # the offsets back itself; ours cuts a fixed window at the training
        # size and follows the mean of the confident points. Neither setting
        # means anything to the other, so they are never shown together.
        self.dlc_dynamic_widget = QtWidgets.QWidget()
        drow = QtWidgets.QHBoxLayout(self.dlc_dynamic_widget)
        drow.setContentsMargins(0, 0, 0, 0)
        drow.setSpacing(6)
        drow.addWidget(QtWidgets.QLabel("Detect at:"))
        self.dlc_dynamic_threshold_spin = NumericLineEdit(decimals=2)
        self.dlc_dynamic_threshold_spin.setRange(0.0, 1.0)
        self.dlc_dynamic_threshold_spin.setSingleStep(0.05)
        self.dlc_dynamic_threshold_spin.setValue(0.5)
        self.dlc_dynamic_threshold_spin.setMaximumWidth(56)
        self.dlc_dynamic_threshold_spin.setToolTip(
            "DeepLabCut-Live's own detection threshold: a body part scoring\n"
            "above this counts as found, and the crop is taken around all of\n"
            "them. Below it on every part, the animal is lost and the next\n"
            "frame is analysed whole.")
        drow.addWidget(self.dlc_dynamic_threshold_spin)
        drow.addWidget(QtWidgets.QLabel("margin:"))
        self.dlc_dynamic_margin_spin = NumericLineEdit()
        self.dlc_dynamic_margin_spin.setRange(0, 500)
        self.dlc_dynamic_margin_spin.setValue(10)
        self.dlc_dynamic_margin_spin.setMaximumWidth(56)
        self.dlc_dynamic_margin_spin.setToolTip(
            "Pixels added around that bounding box. It has to cover how far\n"
            "the animal moves between frames, the next crop is computed from\n"
            "this one, so too small a margin loses the animal at speed.")
        drow.addWidget(self.dlc_dynamic_margin_spin)
        row.addWidget(self.dlc_dynamic_widget)
        self.dlc_dynamic_widget.setVisible(False)

        self.pose_input_note = QtWidgets.QLabel("")
        self.pose_input_note.setStyleSheet(f"color: {_MUTED}; font-size: 8pt;")
        row.addWidget(self.pose_input_note, stretch=1)
        # Set the initial enable state here rather than waiting for the combo
        # to change: the default IS letterbox, so the signal never fires and
        # the window fields would sit editable while doing nothing.
        self._sync_pose_input_row()
        return self.pose_input_widget

    def _on_pose_input_mode_changed(self, _text=None):
        self._mark_dlc_dirty("Input mode changed, re-initialize!")
        self._sync_pose_input_row()
        self._validate_pose_window()

    def _sync_pose_input_row(self):
        """Show only the controls the chosen backend and mode actually use.

        The two cropping mechanisms are not variants of each other and their
        settings are not interchangeable. Ours cuts a fixed window at the
        model's training size and steers it by the mean of the confident
        keypoints, written for SLEAP single-instance, the one family with no
        SDK equivalent. DeepLabCut-Live's crops to the bounding box of whatever
        it last found, expanded by a margin, and restores the offsets itself,
        and exists ONLY in its PyTorch runner.

        Showing both at once invited setting a follow-confidence that a DLC run
        never reads, and a margin a SLEAP run never reads. So each appears only
        where it does something.
        """
        if not hasattr(self, "pose_input_mode_combo"):
            return
        mode = self.pose_input_mode_combo.currentText()
        # ``auto`` counts as following: it is the mode that RESOLVES to
        # crop-track for a crop-trained model, and greying the window and its
        # steering out under the default would hide the controls that matter
        # for exactly the models this defaults right.
        # ``crop-track`` is OUR window and is offered to SLEAP only, so under
        # DLC neither "auto" nor anything else can resolve to it. Deciding
        # ``crops`` from the mode string alone left the follow settings live
        # on a DLC panel that can never use them.
        # Read what the combo is ACTUALLY offering, not what the backend would
        # offer: on a freshly built panel the combo is populated before any
        # mode change has narrowed it, and deciding from the backend alone
        # then disabled a control the operator could still see and select.
        # ``_refresh_input_mode_choices`` reconciles the two on every switch.
        offered = [combo.itemText(i) for i in range(combo.count())] \
            if (combo := self.pose_input_mode_combo) is not None else []
        can_crop_track = ("crop-track" in offered if offered
                          else "crop-track" in self._input_modes_for_current_backend())
        crops = can_crop_track and mode in ("auto", "crop-track")
        dynamic = mode == "dlc-dynamic"
        for w in (self.pose_input_w_spin, self.pose_input_h_spin,
                  self.pose_crop_conf_spin, self.pose_crop_good_spin,
                  self.pose_crop_reacquire_cb):
            w.setEnabled(crops)
        if hasattr(self, "pose_crop_steer_widget"):
            self.pose_crop_steer_widget.setVisible(can_crop_track)
        if hasattr(self, "pose_crop_widget"):
            # The whole row describes the following window, so on a backend
            # that has no such window it is hidden rather than greyed: a
            # disabled control still says the setting exists here.
            self.pose_crop_widget.setVisible(can_crop_track and not dynamic)
        if hasattr(self, "dlc_dynamic_widget"):
            self.dlc_dynamic_widget.setVisible(dynamic)

    #: Input modes each backend can actually use. ``dlc-dynamic`` is
    #: DeepLabCut-Live's own and reaches only its PyTorch runner; ``crop-track``
    #: is ours and is written for SLEAP single-instance. Offering either to the
    #: other backend is offering a setting that does nothing.
    _SLEAP_INPUT_MODES: ClassVar[List[str]] = ["auto", "letterbox", "full",
                                               "crop-track"]
    _DLC_INPUT_MODES: ClassVar[List[str]] = ["auto", "letterbox", "full"]

    def _input_modes_for_current_backend(self):
        """The modes the chosen backend and engine can honour, and no others.

        One list, asked by the combo that offers them and by the visibility
        rule that decides which steering controls mean anything, so the panel
        cannot offer a mode it then hides the settings for, or show settings
        for a mode it never offers.
        """
        if self._current_mode() == "sleap":
            return list(self._SLEAP_INPUT_MODES)
        items = list(self._DLC_INPUT_MODES)
        # Only the PyTorch runner carries a DynamicCropper. The TensorFlow
        # runners never receive the keyword, and an exported ONNX/TensorRT
        # graph is not driven through DeepLabCut-Live at all.
        if self._resolved_dlc_engine() == "pytorch":
            items.append("dlc-dynamic")
        return items

    def _refresh_input_mode_choices(self):
        """Offer the modes this backend and engine can honour, and no others."""
        combo = getattr(self, "pose_input_mode_combo", None)
        if combo is None:
            return
        items = self._input_modes_for_current_backend()
        previous = combo.currentText()
        # A mode the new backend cannot use is DROPPED rather than carried
        # over, unlike the engine list: switching the chip from SLEAP to DLC
        # otherwise left ``crop-track`` selected, which is our window and not
        # DeepLabCut's. A mode restored from a saved project is a different
        # matter and is re-added by _apply_sleap_params, so nothing is lost.
        if [combo.itemText(i) for i in range(combo.count())] == items:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentText(previous if previous in items else "auto")
        combo.blockSignals(False)

    def _validate_pose_window(self):
        """Show the plan the pipeline will actually use, while there is still
        time to change it.

        Asked of ``input_policy`` rather than re-derived here: a second copy of
        the rule is a second answer waiting to disagree with the sink, and the
        thing the operator needs to see is what will HAPPEN, including a
        refusal and its reason.
        """
        if not hasattr(self, "pose_input_note"):
            return
        path = self.model_path.text().strip()
        if not path:
            self.pose_input_note.setText("")
            return
        plan = self._resolved_input_plan()
        if plan is None:
            self.pose_input_note.setText("")
            return
        asked = self._LABEL_TO_INPUT_MODE.get(
            self.pose_input_mode_combo.currentText(), "auto")
        # Green when the request was honoured, amber when the model overruled
        # it. A downgrade the operator cannot see is the failure this whole
        # change exists to remove.
        overruled = asked not in ("auto", plan.mode)
        colour = "#b7791f" if overruled else "#34d399"
        self.pose_input_note.setText(plan.describe())
        self.pose_input_note.setStyleSheet(f"color:{colour};font-size:8pt;")

    def _resolved_input_plan(self):
        """The ``InputPlan`` for the current dialog state, or None.

        The box's frame size is not known in the dialog, so the ROI's own size
        is used when there is one and a typical arena crop otherwise. Only the
        crop-vs-letterbox comparison depends on it, and that comparison is a
        ratio well clear of the difference between those two guesses.
        """
        path = self.model_path.text().strip()
        if not path:
            return None
        try:
            from source.video.tracking import input_policy
            w, h = self._pose_frame_size()
            return input_policy.plan(
                self._current_mode(), path, w, h,
                engine=self._resolved_dlc_engine(),
                asked=self._LABEL_TO_INPUT_MODE.get(
                    self.pose_input_mode_combo.currentText(), "auto"),
                asked_size=(int(self.pose_input_w_spin.value()),
                            int(self.pose_input_h_spin.value())))
        except Exception as e:
            logger.debug("input plan preview failed: %s", e)
            return None

    def _pose_frame_size(self):
        """``(w, h)`` of the frame this box will feed pose, best effort."""
        roi = getattr(self, "roi_rect", None) or getattr(self, "_roi_rect", None)
        if roi and len(roi) == 4 and roi[2] and roi[3]:
            return int(roi[2]), int(roi[3])
        return 640, 480

    def _current_mode(self) -> str:
        if getattr(self, "mode_sleap", None) and self.mode_sleap.isChecked():
            return "sleap"
        return "dlc"

    def _resolved_dlc_engine(self) -> str:
        """Which DLC engine the current selection resolves to, or ``""``."""
        if self._current_mode() != "dlc":
            return ""
        try:
            from source.video.tracking import dlc_engine
            return dlc_engine.resolve(
                self.model_path.text().strip(),
                self.dlc_model_type_combo.currentText()
                if hasattr(self, "dlc_model_type_combo") else "auto")[0]
        except Exception as e:
            logger.debug("DLC engine preview failed: %s", e)
            return ""

    def _refresh_dlc_engine_choices(self):
        """Offer only the engines this model and this machine can actually run.

        The combo used to list ``base``/``pytorch``/``tensorrt``/``lite``
        whatever was selected, three TensorFlow runners on a machine with no
        TensorFlow, for a PyTorch model. Choosing one produced
        ``ModuleNotFoundError: tensorflow``, which names the wrong thing
        entirely. What a folder offers is knowable from the folder.
        """
        combo = getattr(self, "dlc_model_type_combo", None)
        if combo is None or self._current_mode() != "dlc":
            return
        path = self.model_path.text().strip()
        try:
            from source.video.tracking import dlc_engine
            runnable = dlc_engine.runnable(path) if path else []
            offered = dlc_engine.available(path) if path else []
        except Exception as e:
            logger.debug("DLC engine list failed: %s", e)
            return
        engine_labels = {"export": "onnx", "pytorch": "pytorch",
                         "tensorflow": "base"}
        items = ["auto"] + [engine_labels[engine] for engine in offered]
        previous = combo.currentText()
        # A selection already made survives even when this model does not offer
        # it. Dropping it would silently reset the engine to ``auto`` whenever
        # the combo is rebuilt, a setting that saves and does not restore,
        # which is worse than one that was never offered. ``resolve`` reports
        # the downgrade at Init if it really cannot run.
        if previous and previous not in items:
            items.append(previous)
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentText(previous if previous in items else "auto")
        combo.blockSignals(False)
        missing = [e for e in offered if e not in runnable]
        combo.setToolTip(
            "Which engine runs this model.\n"
            "auto - the fastest one this folder and this machine can run.\n"
            "onnx - the exported graph; fastest, input size fixed at export.\n"
            "pytorch - the folder's own .pt snapshot; any frame size, and\n"
            "  measured the most accurate of the three here.\n"
            "base - DeepLabCut-Live's TensorFlow runner.\n"
            + (f"\nPresent but not runnable here: {', '.join(missing)}."
               if missing else ""))

    #: Stored value <-> the label the combo shows. The config spelling is the
    #: stored one; the dash is presentation only.
    _INPUT_MODE_TO_LABEL: ClassVar[Dict[str, str]] = {
        "auto": "auto", "letterbox": "letterbox", "full": "full",
        "crop_track": "crop-track", "dlc_dynamic": "dlc-dynamic"}
    _LABEL_TO_INPUT_MODE: ClassVar[Dict[str, str]] = {
        v: k for k, v in _INPUT_MODE_TO_LABEL.items()}

    def _pose_input_settings(self) -> dict:
        """The input contract as it will be stored."""
        if not hasattr(self, "pose_input_mode_combo"):
            return {}
        label = self.pose_input_mode_combo.currentText()
        return {
            "pose_input_mode": self._LABEL_TO_INPUT_MODE.get(label, "auto"),
            "pose_input_w": int(self.pose_input_w_spin.value()),
            "pose_input_h": int(self.pose_input_h_spin.value()),
            "pose_crop_conf_min": float(self.pose_crop_conf_spin.value()),
            "pose_crop_good_min": int(self.pose_crop_good_spin.value()),
            "pose_crop_reacquire": bool(self.pose_crop_reacquire_cb.isChecked()),
            # DeepLabCut-Live's own cropping, kept separate from the three
            # above because it is a different mechanism in a different backend.
            "dlc_dynamic_threshold": float(
                self.dlc_dynamic_threshold_spin.value()),
            "dlc_dynamic_margin": int(self.dlc_dynamic_margin_spin.value()),
        }

    def _build_dlc_annotate_row(self):
        """Annotate-body-parts container + marker-size spinbox."""
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Annotate:"))
        self.annotate_parts_widget = QtWidgets.QWidget()
        self.annotate_parts_layout = QtWidgets.QHBoxLayout(self.annotate_parts_widget)
        self.annotate_parts_layout.setContentsMargins(0, 0, 0, 0)
        self.annotate_parts_layout.setSpacing(6)
        self.annotate_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        row.addWidget(self.annotate_parts_widget, stretch=1)
        row.addWidget(QtWidgets.QLabel("Marker:"))
        self.marker_size_spin = NumericLineEdit()
        self.marker_size_spin.setRange(1, 20)
        self.marker_size_spin.setValue(4)
        self.marker_size_spin.setToolTip("Radius of body part annotation circles (px)")
        self.marker_size_spin.setMaximumWidth(60)
        row.addWidget(self.marker_size_spin)
        return row

    # ── Skeleton ───────────────────────────────────────────

    def _build_skeleton_row(self):
        """The edges drawn between keypoints, auto-filled from the model.

        sleap-nn records the skeleton in ``training_config.yaml`` and the
        reader picks it up, so this arrives already correct for a SLEAP model.
        DeepLabCut keeps its skeleton in the PROJECT ``config.yaml``, which an
        exported model folder does not ship, so for those the table starts
        empty and this is where the operator states the connections.

        Nothing is invented: with no edges the overlay draws points only,
        rather than joining parts in list order and implying anatomy the model
        never declared.
        """
        row = QtWidgets.QVBoxLayout()
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Skeleton:"))
        self.skeleton_summary = QtWidgets.QLabel("")
        self.skeleton_summary.setStyleSheet(f"color: {_MUTED}; font-size: 8pt;")
        header.addWidget(self.skeleton_summary, stretch=1)
        self.skeleton_add_btn = QtWidgets.QPushButton("+ Edge")
        self.skeleton_add_btn.setToolTip(
            "Connect two body parts. The overlay draws a line between them, "
            "and nothing else changes, the skeleton is for reading the "
            "video, not for inference.")
        self.skeleton_add_btn.setMaximumWidth(80)
        self.skeleton_add_btn.clicked.connect(self._add_skeleton_edge)
        header.addWidget(self.skeleton_add_btn)
        self.skeleton_reset_btn = QtWidgets.QPushButton("Reset")
        self.skeleton_reset_btn.setToolTip(
            "Go back to the skeleton the model declares. Empty for a model "
            "that declares none.")
        self.skeleton_reset_btn.setMaximumWidth(70)
        self.skeleton_reset_btn.clicked.connect(self._reset_skeleton)
        header.addWidget(self.skeleton_reset_btn)
        row.addLayout(header)

        self.skeleton_table = QtWidgets.QTableWidget(0, 3)
        self.skeleton_table.setHorizontalHeaderLabels(["From", "To", ""])
        self.skeleton_table.verticalHeader().setVisible(False)
        self.skeleton_table.setMaximumHeight(130)
        head = self.skeleton_table.horizontalHeader()
        head.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.skeleton_table.setColumnWidth(2, 30)
        row.addWidget(self.skeleton_table)
        return row

    def _update_skeleton_table(self):
        """Redraw the table from ``skeleton_edges``."""
        table = getattr(self, "skeleton_table", None)
        if table is None:
            return
        parts = [str(p) for p in (self.body_parts_list or [])]
        table.setRowCount(0)
        for index, (src, dst) in enumerate(self.skeleton_edges):
            table.insertRow(index)
            table.setCellWidget(index, 0, self._skeleton_combo(parts, src, index, 0))
            table.setCellWidget(index, 1, self._skeleton_combo(parts, dst, index, 1))
            drop = QtWidgets.QPushButton("X")
            drop.setToolTip("Remove this edge")
            drop.clicked.connect(lambda _=False, i=index: self._drop_skeleton_edge(i))
            table.setCellWidget(index, 2, drop)
        self._update_skeleton_summary()

    def _skeleton_combo(self, parts, current, row_index, column):
        combo = NoWheelComboBox()
        combo.addItems(parts)
        if current in parts:
            combo.setCurrentText(current)
        combo.currentTextChanged.connect(
            lambda text, i=row_index, c=column: self._set_skeleton_end(i, c, text))
        return combo

    def _set_skeleton_end(self, row_index, column, text):
        if 0 <= row_index < len(self.skeleton_edges):
            edge = list(self.skeleton_edges[row_index])
            edge[column] = str(text)
            self.skeleton_edges[row_index] = tuple(edge)
            self._update_skeleton_summary()

    def _add_skeleton_edge(self):
        parts = [str(p) for p in (self.body_parts_list or [])]
        if len(parts) < 2:
            return
        self.skeleton_edges.append((parts[0], parts[1]))
        self._update_skeleton_table()

    def _drop_skeleton_edge(self, row_index):
        if 0 <= row_index < len(self.skeleton_edges):
            del self.skeleton_edges[row_index]
            self._update_skeleton_table()

    def _reset_skeleton(self):
        """Back to what the model itself declares."""
        from source.video.tracking.model_config import ModelInfo
        info = ModelInfo.read(self.model_path.text().strip())
        parts = set(self.body_parts_list or [])
        self.skeleton_edges = [e for e in info.skeleton
                               if e[0] in parts and e[1] in parts]
        self._update_skeleton_table()

    def _update_skeleton_summary(self):
        label = getattr(self, "skeleton_summary", None)
        if label is None:
            return
        edges = self.skeleton_edges
        if not edges:
            label.setText("no edges, the overlay draws points only")
        else:
            label.setText(f"{len(edges)} edge(s)")


    def _build_dlc_params_row(self):
        """Zone part / Confidence / Resize / Instances / Initialize on one row."""
        row = QtWidgets.QHBoxLayout()

        row.addWidget(QtWidgets.QLabel("Zone part:"))
        self.zone_body_part_combo = QtWidgets.QComboBox()
        self.zone_body_part_combo.setToolTip(
            "Body part used for zone enter/exit detection and zone highlighting")
        self.zone_body_part_combo.setMaximumWidth(120)
        row.addWidget(self.zone_body_part_combo)

        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Confidence:"))
        self.dlc_confidence_spin = NumericLineEdit(decimals=2)
        self.dlc_confidence_spin.setRange(0.0, 1.0)
        self.dlc_confidence_spin.setSingleStep(0.05)
        self.dlc_confidence_spin.setValue(0.5)
        self.dlc_confidence_spin.setToolTip(
            "Minimum confidence to draw keypoint and use for zone detection (0-1)")
        self.dlc_confidence_spin.setMaximumWidth(70)
        row.addWidget(self.dlc_confidence_spin)

        row.addSpacing(12)
        # Kept as an attribute so _refresh_scale_control can retitle it: the
        # same number is an operator setting under DLC and a model-owned fact
        # under SLEAP, and the label has to say which.
        self.dlc_resize_label = QtWidgets.QLabel("Resize:")
        row.addWidget(self.dlc_resize_label)
        self.dlc_resize_spin = NumericLineEdit(decimals=2)
        self.dlc_resize_spin.setRange(0.1, 1.0)
        self.dlc_resize_spin.setSingleStep(0.1)
        self.dlc_resize_spin.setValue(0.8)
        self.dlc_resize_spin.setToolTip(
            "Resize factor for DLC inference (smaller = faster, lower accuracy)")
        self.dlc_resize_spin.setMaximumWidth(70)
        row.addWidget(self.dlc_resize_spin)

        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Instances:"))
        self.pose_instances_spin = NumericLineEdit()
        self.pose_instances_spin.setRange(1, 8)
        self.pose_instances_spin.setValue(1)
        self.pose_instances_spin.setToolTip(
            "Number of parallel model copies to load on the GPU.\n"
            "1 = single shared session (default). 2+ runs that many copies\n"
            "in parallel, useful when you have many boxes (~1 per 4 boxes).\n"
            "Each copy uses extra GPU memory.")
        self.pose_instances_spin.setMaximumWidth(55)
        row.addWidget(self.pose_instances_spin)

        row.addSpacing(12)
        self.dlc_init_btn = QtWidgets.QPushButton("Initialize")
        from source.gui.styles import BUTTON_STYLE as _BS_i
        from source.gui.styles import COLORS as _C_i
        self.dlc_init_btn.setStyleSheet(_BS_i.format(
            color=_C_i['warning'], hover_color=_C_i['warning_hover']))
        self.dlc_init_btn.setToolTip("Load model and run first inference to verify setup")
        self.dlc_init_btn.clicked.connect(self._init_dlc)
        row.addWidget(self.dlc_init_btn)
        self.dlc_init_status = QtWidgets.QLabel("")
        self.dlc_init_status.setStyleSheet("font-size: 10px;")
        row.addWidget(self.dlc_init_status, stretch=1)
        return row

    # ----- event triggers (table + Add Trigger button) -----

    def _build_event_triggers_group(self):
        group = QtWidgets.QGroupBox("Event Triggers")
        vbox = QtWidgets.QVBoxLayout(group)
        vbox.setContentsMargins(6, 18, 6, 4)

        self.triggers_table = QtWidgets.QTableWidget()
        self.triggers_table.setColumnCount(11)
        self.triggers_table.setHorizontalHeaderLabels(
            ["Condition", "Body Part", "Zones", "Event", "Threshold",
             "Part B", "Hold", "Col", "Vid", "Plot", ""])
        header = self.triggers_table.horizontalHeader()
        self.triggers_table.setWordWrap(False)
        # Width by what the FIELD needs, not by how long its header is.
        #
        # Zones and Event carry free text / multi-select and are the two the
        # operator actually reads, so they take the slack. Threshold and Hold
        # hold at most 5 digits and are sized by hand: under ResizeToContents
        # Qt uses a line-edit's generous size hint, and between them they
        # squeeze Zones and Event to ~40 px, enough to truncate the headers
        # to "ONE"/"/EN" and the button text to "ct zon".
        _FIX = QtWidgets.QHeaderView.ResizeMode.Fixed
        _INT = QtWidgets.QHeaderView.ResizeMode.Interactive
        _STR = QtWidgets.QHeaderView.ResizeMode.Stretch
        #        col, mode, width  (width ignored for Stretch)
        _LAYOUT = (
            (0,  _INT, 132),   # Condition, longest item "head_angle_gt"
            (1,  _INT, 108),   # Body Part
            (2,  _STR, 170),   # Zones, multi-select, needs room
            (3,  _STR, 150),   # Event, free-text MCU event name
            (4,  _FIX,  84),   # Threshold, a number
            (5,  _INT, 108),   # Part B
            (6,  _FIX,  86),   # Hold, a number + " ms"
            (7,  _FIX,  46),   # Col, 26 px swatch
            (8,  _FIX,  44),   # Vid, checkbox
            (9,  _FIX,  48),   # Plot, checkbox
            (10, _FIX,  40),   # delete, 30 px button
        )
        for col, mode, width in _LAYOUT:
            header.setSectionResizeMode(col, mode)
            if mode is not _STR:
                # Stretch ignores an explicit width, those two columns take
                # whatever the fixed ones leave, which is the point.
                self.triggers_table.setColumnWidth(col, width)
        header.setMinimumSectionSize(40)

        _HEADER_HELP = {
            0: "What has to be true for this trigger to fire.",
            1: "Which tracked point the condition is measured on.",
            2: "Which zone(s) the condition applies to. Zone rules only.",
            3: "The MCU event name to fire. Must appear VERBATIM in the "
               "task's events list, or the board silently ignores it.",
            4: "Comparison value. Only used by speed / rotation / distance / "
               "angle rules, greyed out for zone rules.",
            5: "The SECOND tracked point, for rules that need two:\n"
               "  distance_gt / distance_lt, measures Body Part ↔ Part B\n"
               "  head_angle_gt / head_angle_lt, Part B is the NECK joint,\n"
               "     and the angle is head-vs-body measured at it\n"
               "Greyed out for every other condition, because they need only "
               "one point.",
            6: "Minimum time the condition must stay true before firing.\n"
               "0 = fire immediately. ~300-500 ms rejects blips on\n"
               "freezing / rearing rules.",
            7: "Colour of this trigger's chip on the video and its lane in "
               "the Session Plot.",
            8: "Draw a chip on the live video when this trigger is active.",
            9: "Give this trigger a lane in the Session Plot.",
        }
        for col, tip in _HEADER_HELP.items():
            item = self.triggers_table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)
        vbox.addWidget(self.triggers_table, stretch=1)

        # Legend. States the two things that are NOT visible from the table:
        # every rule is edge-fired, and the posture/orientation rules need a
        # two-keypoint axis that blob tracking cannot provide.
        legend = QtWidgets.QLabel(
            "Every rule fires ONCE when it becomes true, not continuously "
            ", so <b>in_zone</b> fires on entry and <b>not_in_zone</b> on exit. "
            "Zone rules use Body Part + Zones; speed / rotation / distance / "
            "angle rules use Threshold (+ Part B). Hold = min ms before firing. "
            "Rules needing a head-tail axis are unavailable under blob tracking.")
        legend.setWordWrap(True)
        legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        legend.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        vbox.addWidget(legend)

        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ Add Trigger")
        add_btn.clicked.connect(self._add_trigger)
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)
        return group

    # ----- coord mapping (c.<name> -> body_part rows) -----

    def _build_coord_mapping_group(self):
        group = QtWidgets.QGroupBox("Coord Mapping")
        vbox = QtWidgets.QVBoxLayout(group)
        vbox.setContentsMargins(4, 14, 4, 4)
        vbox.setSpacing(2)

        self._coord_rows_layout = QtWidgets.QVBoxLayout()
        self._coord_rows_layout.setSpacing(1)
        self._coord_mapping_rows = []
        vbox.addLayout(self._coord_rows_layout)
        # Defaults via the single-source-of-truth setter, a later
        # _set_coord_mapping(...) on settings-apply cleanly replaces them
        # instead of stacking on top.
        self._set_coord_mapping({})

        add_coord_btn = QtWidgets.QPushButton("+")
        add_coord_btn.setFixedSize(22, 18)
        add_coord_btn.setToolTip("Add coordinate mapping")
        add_coord_btn.clicked.connect(lambda: self._add_coord_row("", ""))
        vbox.addWidget(add_coord_btn)
        return group

    def _add_coord_row(self, name="", body_part=""):
        """Add a compact coordinate mapping row: c.[name] → [body_part] [x]

        The ``c.`` prefix and name input render as one flush composite
        widget (focus-blue prefix pill on the left, name input on the
        right, no gap between them) so the row reads as a single
        ``c.<name>`` token rather than a loose label plus a textbox.
        """
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(2)
        row.setContentsMargins(0, 0, 0, 0)

        # Flush "c." prefix pill, rounded only on its left edge so it
        # joins seamlessly with the QLineEdit's left edge.
        lbl = QtWidgets.QLabel("c.")
        lbl.setFixedHeight(20)
        lbl.setFixedWidth(20)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "QLabel {"
            " background: rgba(56,189,248,0.12);"
            " border: 1px solid rgba(56,189,248,0.35);"
            " border-right: none;"
            " border-top-left-radius: 5px;"
            " border-bottom-left-radius: 5px;"
            " color: #38bdf8;"
            " font-size: 10px; font-weight: 700;"
            "}"
        )
        row.addWidget(lbl)

        edit = QtWidgets.QLineEdit(name)
        edit.setPlaceholderText("name")
        edit.setFixedHeight(20)
        edit.setFixedWidth(108)
        # No left rounding, the pill sits flush on that side.
        edit.setStyleSheet(
            "QLineEdit {"
            " font-size: 10px;"
            " border: 1px solid rgba(255,255,255,0.18);"
            " border-top-left-radius: 0; border-bottom-left-radius: 0;"
            " border-top-right-radius: 5px; border-bottom-right-radius: 5px;"
            " padding: 0 6px;"
            "}"
        )
        row.addWidget(edit)
        row.addSpacing(6)

        combo = QtWidgets.QComboBox()
        parts, default = self._coord_part_options()
        combo.addItems(parts)
        if body_part and body_part in parts:
            combo.setCurrentText(body_part)
        elif default in parts:
            combo.setCurrentText(default)
        combo.setFixedHeight(20)
        # Wide enough that the "centroid" glyph is not clipped by the arrow.
        combo.setFixedWidth(118)
        combo.setStyleSheet("font-size:10px;")
        row.addWidget(combo)

        rm = QtWidgets.QPushButton("x")
        rm.setFixedSize(16, 16)
        rm.setStyleSheet("color:red;border:none;font-size:9px;")
        row.addWidget(rm)
        row.addStretch()

        w = QtWidgets.QWidget()
        w.setLayout(row)
        w.setFixedHeight(22)

        entry = (edit, combo, w)
        self._coord_mapping_rows.append(entry)
        self._coord_rows_layout.addWidget(w)
        # Identity-bind the remove handler to the (edit, combo, w) tuple
        # rather than a captured index, so deletions (which leave None slots
        # and re-use indices) can't remove the wrong row.
        rm.clicked.connect(lambda checked=False, e=entry: self._remove_coord_row_by_entry(e))

    def _remove_coord_row_by_entry(self, entry):
        """Remove a coord-mapping row by identity, not by captured index."""
        try:
            idx = self._coord_mapping_rows.index(entry)
        except ValueError:
            return  # Already removed
        edit, combo, container = entry
        self._coord_rows_layout.removeWidget(container)
        container.setParent(None)
        container.deleteLater()
        self._coord_mapping_rows.pop(idx)

    def _set_coord_mapping(self, mapping: dict) -> None:
        """Replace the entire coord-mapping list in one shot.

        Wipes every existing row (properly removing each widget from
        the layout AND deleting it, not just clearing the Python list),
        then installs one row per ``mapping`` entry.  Falls back to the
        ``loc_center`` / ``speed`` defaults when ``mapping`` is empty so
        a fresh panel always carries the basics.

        Single source of truth for the rows, both ``_build_ui`` (init)
        and the settings-apply path call this, which guarantees no
        accumulation across re-inits / re-opens / Load Config clicks.
        """
        for entry in list(self._coord_mapping_rows):
            if entry is None:
                continue
            _edit, _combo, container = entry
            self._coord_rows_layout.removeWidget(container)
            container.setParent(None)
            container.deleteLater()
        self._coord_mapping_rows = []
        if mapping:
            for name, part in mapping.items():
                self._add_coord_row(name, part)
        else:
            # Default rows follow the current mode: pose → the picked
            # keypoint (e.g. "center"), blob → "centroid". _add_coord_row
            # applies the right default when the body_part is left blank.
            self._add_coord_row("loc_center", "")
            self._add_coord_row("speed", "")

    def _coord_part_options(self):
        """Body-part (items, default) for the Coord Mapping combos.

        Mirrors the Zone-part combo (``_update_zone_body_part_combo``) so the
        two never disagree: in pose mode with a model the synthetic
        ``"centroid"`` is dropped and the real keypoints are offered, defaulting
        to the picked zone body part (or a center-like keypoint); in blob mode
        / pose-without-model it collapses to ``"centroid"``.
        """
        is_pose = self.mode_dlc.isChecked() or self.mode_sleap.isChecked()
        real_parts = [
            str(bp).strip() for bp in (self.body_parts_list or [])
            if str(bp).strip() and str(bp).strip() != "centroid"]
        if is_pose and real_parts:
            zone_part = ""
            if getattr(self, "zone_body_part_combo", None) is not None:
                zone_part = self.zone_body_part_combo.currentText().strip()
            default = (zone_part if zone_part in real_parts else "") \
                or self._find_center_body_part(real_parts) or real_parts[0]
            return real_parts, default
        return (["centroid"] + real_parts), "centroid"

    def _update_coord_combos(self):
        """Update body part dropdowns in coord mapping when DLC parts change."""
        # The rows live on a tab built separately from this panel, so they can
        # legitimately not exist yet when a model's parts land, which now
        # happens for SLEAP too, where loading used to fail before reaching
        # here. Nothing to update is not an error.
        if not hasattr(self, "_coord_mapping_rows"):
            return
        parts, default = self._coord_part_options()
        for row in self._coord_mapping_rows:
            if row is None:
                continue
            edit, combo, container = row
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(parts)
            if current in parts:
                combo.setCurrentText(current)
            elif default in parts:
                combo.setCurrentText(default)
            combo.blockSignals(False)

    def get_coord_mapping(self) -> dict:
        """Return {coord_name: body_part} dict from user-defined rows."""
        mapping = {}
        for row in self._coord_mapping_rows:
            if row is None:
                continue
            edit, combo, container = row
            name = edit.text().strip()
            part = combo.currentText()
            if name and part:
                mapping[name] = part
        return mapping

    @staticmethod
    def _vsep():
        """Create a thin vertical separator line. Fixed height so it never
        stretches to fill a tall row (which spread the blob controls apart)."""
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        sep.setFixedWidth(2)
        sep.setFixedHeight(22)
        return sep

    def _on_select_all(self, state):
        checked = state == QtCore.Qt.CheckState.Checked.value
        for cb in self.box_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _on_box_changed(self):
        all_checked = all(cb.isChecked() for cb in self.box_checkboxes.values())
        none_checked = not any(cb.isChecked() for cb in self.box_checkboxes.values())
        self.select_all_cb.blockSignals(True)
        if all_checked:
            self.select_all_cb.setCheckState(QtCore.Qt.CheckState.Checked)
        elif none_checked:
            self.select_all_cb.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            self.select_all_cb.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        self.select_all_cb.blockSignals(False)

        # Auto-capture backgrounds for newly checked boxes that don't have one
        for setup_id, cb in self.box_checkboxes.items():
            if cb.isChecked() and not self._background_for_box(setup_id):
                self._take_background(setup_id, silent=True)

    def _retake_backgrounds(self):
        """Re-capture backgrounds for all selected boxes."""
        selected = [bid for bid, cb in self.box_checkboxes.items() if cb.isChecked()]
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No Boxes", "No boxes selected.")
            return
        captured = 0
        for setup_id in selected:
            if self._take_background(setup_id, silent=True):
                captured += 1
        self._update_bg_status()
        if captured:
            logger.info(f"Retook backgrounds for {captured} box(es)")

    def _background_for_box(self, setup_id):
        """This box's background image path, or None.

        Checks what was captured in this dialog session, then falls back to
        the durable copy on disk. ``self.backgrounds`` starts empty on every
        open, so without the fallback a background captured minutes ago
        looked missing: the panel asked for a re-take and the calibration
        dialog opened with no reference at all.

        ``<project>/background_images/box{N}.png`` is the canonical location,
        the same one capture writes to and ``_setup_blob_tracker_for_box``
        reads at run time.
        """
        got = self.backgrounds.get(setup_id)
        if got:
            return got
        getter = getattr(self.window(), "_background_path_for_box", None)
        if not callable(getter):
            return None
        try:
            path = getter(setup_id)
        except Exception as e:
            logger.debug("Box %s: background lookup failed: %s", setup_id, e)
            return None
        if path is None:
            return None
        # Adopt it, so the status line and any later read agree.
        self.backgrounds[setup_id] = str(path)
        return str(path)

    def _known_backgrounds(self) -> dict:
        """Every box's background, disk fallback included."""
        out = {}
        for bid in self.box_ids:
            p = self._background_for_box(bid)
            if p:
                out[bid] = p
        return out

    def _open_calibration(self):
        """Open live tracker calibration dialog."""
        try:
            from source.gui.dialogs.tracking import TrackerCalibrationDialog
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, "Not Available",
                "Tracker calibration dialog is not available in this build.")
            return

        # Find first selected box with a camera
        selected = [bid for bid, cb in self.box_checkboxes.items() if cb.isChecked()]
        if not selected:
            # Fall back to first box with a camera callback
            selected = [bid for bid in self.box_ids if bid in self.get_frame_callbacks]
        if not selected:
            QtWidgets.QMessageBox.warning(
                self, "No Camera", "No box with a camera available for calibration.")
            return

        setup_id = selected[0]
        callback = self.get_frame_callbacks.get(setup_id)
        if not callback:
            QtWidgets.QMessageBox.warning(
                self, "No Camera", f"No camera callback for Box {setup_id}.")
            return

        # Gather current spinbox values as initial params
        initial_params = {
            "detect_dark": self.detect_dark_cb.isChecked(),
            "threshold": self.threshold_spin.value(),
            "min_area": self.min_area_spin.value(),
            "max_area": self.max_area_spin.value(),
            "blur_mode": self.blur_mode_combo.currentText(),
            "blur_kernel_size": self.blur_size_spin.value(),
            "bg_mode": self.bg_mode_combo.currentText(),
            "open_kernel_size": self.open_kernel_spin.value(),
            "close_kernel_size": self.close_kernel_spin.value(),
            "self_norm_sigma": float(self.sn_sigma_spin.value()),
            "self_norm_smooth_sigma": float(self.sn_smooth_spin.value()),
            "self_norm_minsize": int(self.sn_minsize_spin.value()),
        }
        initial_params.update(self._blob_calibration_extra)

        # Pass existing background if already captured
        bg_path = self._background_for_box(setup_id)

        # Multi-box: hand the dialog every connected box's frame callback +
        # background so the user can switch boxes from the dropdown.
        dlg = TrackerCalibrationDialog(
            parent=self, setup_id=setup_id,
            get_frame_callback=callback, initial_params=initial_params,
            background_path=bg_path,
            box_callbacks=dict(self.get_frame_callbacks),
            box_backgrounds=self._known_backgrounds(),
        )

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Multi-box: pull every box's tuned params back, not just the
            # currently-selected one.
            multi = dlg.get_calibrations()
            result = dlg.get_calibration() if not multi else multi.get(setup_id)
            if result is None and multi:
                # User calibrated other boxes but not this one -- treat as ok.
                pass
            elif result is None:
                return
            # CLAHE and adaptive-threshold are tunable in the calibration
            # dialog but have no widget in this panel, without holding them
            # as panel state, Apply would throw away the two settings that
            # made the preview track, and the live tracker would run on the
            # old values. Emitted by get_settings, same as dlc_model_path.
            for key in self._CALIBRATION_ONLY_KEYS:
                if key in result:
                    self._blob_calibration_extra[key] = result[key]
            self._refresh_sn_ratio_label()

            # Write calibrated values back to spinboxes
            if "detect_dark" in result:
                self.detect_dark_cb.setChecked(result["detect_dark"])
            if "threshold" in result:
                self.threshold_spin.setValue(result["threshold"])
            if "min_area" in result:
                self.min_area_spin.setValue(result["min_area"])
            if "max_area" in result:
                self.max_area_spin.setValue(result["max_area"])
            if "blur_mode" in result:
                idx = self.blur_mode_combo.findText(result["blur_mode"])
                if idx >= 0:
                    self.blur_mode_combo.setCurrentIndex(idx)
            if "blur_kernel_size" in result:
                self.blur_size_spin.setValue(result["blur_kernel_size"])
            if "open_kernel_size" in result:
                self.open_kernel_spin.setValue(result["open_kernel_size"])
            if "close_kernel_size" in result:
                self.close_kernel_spin.setValue(result["close_kernel_size"])

            # Dialog writes the BG file straight into the project's
            # background_images/ folder; pick up each box's path string
            # so the panel's status reflects what was captured.
            updated: list[int] = []
            sources: dict = dict(multi or {})
            if result and setup_id not in sources:
                sources[setup_id] = result
            for bid, params in sources.items():
                bg_path = params.get("background_path")
                if bg_path:
                    self.backgrounds[bid] = bg_path
                    updated.append(bid)
            if updated:
                self._update_bg_status()
                logger.info(f"Calibrated BGs applied for boxes: {sorted(updated)}")

    def _restyle_mode_chips(self):
        """Repaint Blob / Simple / DLC / SLEAP chip buttons so the selected one
        wears the info-blue pill fill and the others stay flat slate.
        Called on every toggle; cheap (string assignment)."""
        if not hasattr(self, "_mode_chip_btn_style"):
            return
        for btn in (self.mode_blob, self.mode_simple,
                    self.mode_dlc, self.mode_sleap):
            variant = "info" if btn.isChecked() else "ghost"
            btn.setStyleSheet(self._mode_chip_btn_style(
                variant, height=28, padding_h=14))

    def _on_mode_changed(self):
        is_pose = self.mode_dlc.isChecked() or self.mode_sleap.isChecked()
        is_simple = self.mode_simple.isChecked()
        self.blob_widget.setVisible(not is_pose)
        self.blob_adv_widget.setVisible(not is_pose)
        self.dlc_widget.setVisible(is_pose)
        # SLEAP-only options row shows only for the SLEAP chip.
        if hasattr(self, "sleap_opts_widget"):
            self.sleap_opts_widget.setVisible(self.mode_sleap.isChecked())
            self._refresh_sleap_type_badge()
        # DLC engine options mirror it; colour mode applies to either pose
        # backend, so it follows is_pose rather than one chip.
        if hasattr(self, "dlc_opts_widget"):
            self.dlc_opts_widget.setVisible(self.mode_dlc.isChecked())
        self._apply_model_placeholder()
        if hasattr(self, "pose_colour_widget"):
            self.pose_colour_widget.setVisible(is_pose)
        # The input-mode list is backend-specific, see
        # _refresh_input_mode_choices, so it is rebuilt on every chip change,
        # not only on a model change.
        if hasattr(self, "pose_input_mode_combo"):
            self._refresh_input_mode_choices()
            self._sync_pose_input_row()
        if hasattr(self, "sleap_export_status"):
            self._refresh_engine_cache_state()
        # The scale control means something different per backend, so it is
        # retitled on every switch, before the _loading short-circuit below,
        # because a loaded config flips the mode too.
        self._refresh_scale_control()
        self._apply_simple_mode_ui(is_simple, is_pose)
        # During a saved-config load the caller drives body_parts +
        # picker selection manually after the mode flip; short-circuit
        # the auto-load that would otherwise wipe the picker.
        if self._loading:
            return
        if is_pose:
            self._load_body_parts_from_model()
        else:
            # Blob mode, tracker emits only the synthetic centroid;
            # collapse the picker to centroid-only so it can't suggest
            # stale keypoint names from a previously-selected DLC model.
            self.body_parts_list = ["centroid"]
            self._update_zone_body_part_combo([])
        self._update_trigger_body_parts()
        self._update_coord_combos()
        # Blob checked AND nothing else, both halves are needed. Switching
        # chips fires this handler twice, once for the newly checked button
        # and once for the newly unchecked one, and Qt checks the new one
        # first. So during a Blob→Simple switch there is an instant where
        # Simple is checked and Blob has not been cleared yet: asking either
        # question alone captures a background on the way past.
        if self.mode_blob.isChecked() and not is_pose and not is_simple:
            self._autocapture_missing_backgrounds()

    def _autocapture_missing_backgrounds(self):
        """Capture a reference for any selected box that has none.

        Selecting Blob IS the request for background subtraction, and
        subtraction without a reference is not a thing, so the reference is
        taken now rather than left as a step the operator has to know about
        and a warning they have to read. Boxes that already have one are left
        alone: silently replacing a background someone tuned against would be
        worse than the missing-file case this fixes.

        Only runs where it can do no harm, a live camera, and no session in
        progress on that box.
        """
        if self._loading:
            return
        # Switching chips toggles two buttons (one off, one on) and each fires
        # this handler, so without a guard one click could capture twice,
        # and a failed write would retry on every subsequent toggle.
        tried = self._bg_autocapture_tried
        for setup_id, cb in self.box_checkboxes.items():
            if not cb.isChecked():
                continue
            if setup_id in tried:
                continue
            if self._background_for_box(setup_id):
                continue
            if not self.get_frame_callbacks.get(setup_id):
                continue                      # no camera; nothing to capture
            if self._box_is_running(setup_id):
                continue                      # never mid-session
            tried.add(setup_id)
            logger.info("Box %s: no background and Blob selected, "
                        "capturing one now", setup_id)
            self._take_background(setup_id, silent=True)

    def _box_is_running(self, setup_id) -> bool:
        """Is a session live on this box?

        Unknown counts as RUNNING. The two ways of being wrong are not
        symmetric: skipping a capture costs the operator one button press,
        while capturing mid-session swaps the reference out from under a live
        track. So anything that cannot answer, no main window, a raising
        lookup, is treated as "busy, leave it alone".
        """
        try:
            bw = self.window()._setup_widget_for(setup_id)
        except Exception:
            return True
        return bool(getattr(bw, "framework_running", False)) if bw else False

    def _apply_simple_mode_ui(self, is_simple: bool, is_pose: bool):
        """Show the Simple row and retire the controls Simple makes moot.

        Simple needs no captured reference and thresholds a brightness ratio
        rather than a difference level, so **Retake BG**, the background-mode
        dropdown and the fixed Threshold are disabled rather than left
        inviting edits that would do nothing. Min/Max area still apply.
        """
        if hasattr(self, "simple_opts_widget"):
            self.simple_opts_widget.setVisible(is_simple)
            self._refresh_sn_ratio_label()
        self.bg_mode_combo.setEnabled(not is_simple)
        self.threshold_spin.setEnabled(not is_simple)
        self.threshold_label.setEnabled(not is_simple)
        # Retake BG lives in the box-selection group, and that whole group is
        # only built when there ARE boxes (`if self.box_ids` in the layout).
        # This method is reached from the mode chips, which exist either way,
        # so on a panel with no boxes a direct reference raises AttributeError
        #, inside the Qt event loop, where it surfaces as a traceback on
        # stderr and a mode switch that half-applied, not as an error anyone
        # sees.
        retake = getattr(self, "retake_bg_btn", None)
        if retake is not None:
            retake.setEnabled(not is_simple)
            retake.setToolTip(
                _tip("Retake BG", "not used by Simple",
                     "Simple needs no background reference; that is the "
                     "point of it. Nothing here would change what the "
                     "tracker does.",
                     [], "Switch to the Blob chip if you want background "
                         "subtraction and a captured reference.")
                if is_simple else
                "Re-capture background for all selected boxes")
        self.threshold_spin.setToolTip(
            _tip("Threshold", "not used by Simple",
                 "Simple has no difference image to threshold. It compares "
                 "each pixel against its own local brightness instead, using "
                 "the Contrast ratio on the row below.",
                 [], "Set that ratio with Auto Threshold in Calibrate... . "
                     "Min area and Max area do still apply.")
            if is_simple else
            "Background subtraction threshold (lower = more sensitive)")
        # During a config replay _apply_blob_params is the authority on
        # bg_mode; writing it here would race the value being restored.
        if self._loading or self._syncing_mode or is_pose:
            return
        self._syncing_mode = True
        try:
            current = self.bg_mode_combo.currentText()
            if is_simple and current != "self_norm":
                self.bg_mode_combo.setCurrentText("self_norm")
            elif not is_simple and current == "self_norm":
                self.bg_mode_combo.setCurrentText("static")
        finally:
            self._syncing_mode = False

    def _on_bg_mode_changed(self, text: str):
        """BG dropdown → chip. Picking self_norm by hand is picking Simple."""
        if self._loading or self._syncing_mode:
            return
        if self.mode_dlc.isChecked() or self.mode_sleap.isChecked():
            return
        want_simple = (text == "self_norm")
        if want_simple == self.mode_simple.isChecked():
            return
        self._syncing_mode = True
        try:
            (self.mode_simple if want_simple else self.mode_blob).setChecked(True)
        finally:
            self._syncing_mode = False

    def _browse_model(self):
        if self.mode_sleap.isChecked():
            default_dir = os.path.join(app_paths.models_dir, "sleap")
            if not os.path.isdir(default_dir):
                default_dir = app_paths.models_dir
            # SLEAP-nn models are a trained-model FOLDER; legacy SLEAP is a
            # .zip/.pt FILE. Offer a folder first (the sleap-nn default); if the
            # user cancels, fall back to a file picker for the .zip case.
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select SLEAP model folder (Cancel for a .zip file)",
                default_dir)
            if not path:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Select SLEAP model file", default_dir,
                    "SLEAP Model (*.zip *.pt *.pb);;All Files (*)")
        else:
            default_dir = os.path.join(app_paths.models_dir, "dlc")
            if not os.path.isdir(default_dir):
                default_dir = app_paths.models_dir
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select DLC Model Folder", default_dir)
        if path:
            self.model_path.setText(path)

    def _on_model_path_changed(self, path: str):
        # Remember the last real path so a later empty widget (config load
        # with an empty path, programmatic clear) can't lose it on save.
        if path and path.strip():
            self.dlc_model_path = path.strip()
        if self.mode_dlc.isChecked() or self.mode_sleap.isChecked():
            # Model facts first: reading the config cannot fail in a way that
            # should cost the operator the description of the model they just
            # picked, and the heavier body-part/combo refresh below touches far
            # more of the dialog.
            self._refresh_model_info()
            self._refresh_scale_control()
            self._refresh_engine_cache_state()
            self._load_body_parts_from_model()
            self._refresh_sleap_type_badge()
            self._seed_pose_window_from_model()
            self._refresh_dlc_engine_choices()
            self._refresh_input_mode_choices()
            self._sync_pose_input_row()
            self._validate_pose_window()
            # Model swapped → re-init required.
            self._mark_dlc_dirty("Model changed, re-initialize!")

    def _seed_pose_window_from_model(self):
        """Put the model's own training window into the Window fields.

        They defaulted to 0, and 0 reads as "unset", so an operator who chose
        crop-track was told the window would "fall back to letterbox" while the
        size was sitting in the config all along. The model is the authority on
        what it was trained at; the fields exist to override it, not to
        rediscover it.

        Only ever fills EMPTY fields. A number the operator typed is theirs.
        """
        if not hasattr(self, "pose_input_w_spin"):
            return
        if int(self.pose_input_w_spin.value()) or int(
                self.pose_input_h_spin.value()):
            return
        path = self.model_path.text().strip()
        if not path:
            return
        try:
            from source.video.tracking.model_config import ModelInfo
            info = ModelInfo.read(path)
        except Exception as e:
            logger.debug("pose window seed: model config unreadable (%s)", e)
            return
        if info.input_w and info.input_h:
            w, h = int(info.input_w), int(info.input_h)
        elif info.crop_size:
            w = h = int(info.crop_size)
        else:
            return
        self.pose_input_w_spin.setValue(w)
        self.pose_input_h_spin.setValue(h)
        logger.info("Pose window seeded from %s: %dx%d", path, w, h)

    def _refresh_scale_control(self):
        """Make the scale control say what it actually does on this backend.

        The number means different things per backend, and on one of them it
        meant nothing at all:

          * DLC, forwarded to ``DLCLive(resize=…)``; a real operator lever.
          * SLEAP, the input scale comes from the model's own
            ``data_config.preprocessing.scale``, and for an exported ONNX or
            TensorRT engine the input size is fixed at export. There is nothing
            for a host-side factor to change, so the SPIN BOX IS REMOVED and
            the model's own scale is stated as text.

        It used to be shown greyed instead. A disabled control still says the
        setting exists for this backend, and every option on this panel has to
        be one the selected model actually honours; the number is still
        reported, as information rather than as a lever that does nothing.
        """
        if not hasattr(self, "dlc_resize_spin"):
            return
        sleap = hasattr(self, "mode_sleap") and self.mode_sleap.isChecked()
        if not sleap:
            self.dlc_resize_label.setText("Resize:")
            self.dlc_resize_spin.setEnabled(True)
            self.dlc_resize_spin.setVisible(True)
            self.dlc_resize_spin.setToolTip(
                "Resize factor for DLC inference (smaller = faster, lower "
                "accuracy).\nApplied by DLCLive to the frame before inference.")
            self.dlc_resize_label.setToolTip("")
            return

        native = None
        try:
            from source.video.tracking.model_config import ModelInfo
            native = ModelInfo.read(self.model_path.text().strip()).native_scale
        except Exception as e:
            logger.debug("native scale read failed: %s", e)
        # Still written to the spin so the saved config carries the scale the
        # model will actually run at; it is simply not on screen as a control.
        if native is not None:
            self.dlc_resize_spin.setValue(float(native))
        self.dlc_resize_spin.setVisible(False)
        self.dlc_resize_spin.setEnabled(False)
        self.dlc_resize_label.setText(
            f"Model scale: {float(native):.2f}" if native is not None
            else "Model scale: from the model")
        self.dlc_resize_label.setToolTip(
            "SLEAP takes its input scale from the model's own config "
            "(data_config.preprocessing.scale), and an exported ONNX/TensorRT "
            "engine fixes the input size at export time.\n"
            "Re-export the model to change it, a host-side resize would be "
            "discarded.")

    def _refresh_model_info(self):
        """Show what the selected model's config declares about itself."""
        if not hasattr(self, "model_info_label"):
            return
        path = self.model_path.text().strip()
        if not path:
            self.model_info_label.setText("")
            return
        try:
            from source.video.tracking.model_config import ModelInfo
            info = ModelInfo.read(path)
        except Exception as e:
            logger.debug("model info read failed for %s: %s", path, e)
            self.model_info_label.setText("")
            return
        if not info.ok:
            self.model_info_label.setText(
                "No model config found beside the weights, family, native "
                "scale and channels are unknown.")
            self.model_info_label.setStyleSheet(
                f"color: {_MUTED}; font-size: 8pt; font-style: italic;")
            return
        text = info.summary()
        if info.identities:
            text += "  ·  " + ", ".join(info.identities)
        style = f"color: {_MUTED}; font-size: 8pt;"
        if info.warnings:
            # A warning here means the model can do something the current
            # settings will not ask it for, worth colour, not a dialog.
            text += "\n⚠ " + "  ".join(info.warnings)
            style = "color: #b7791f; font-size: 8pt;"
        self.model_info_label.setText(text)
        self.model_info_label.setStyleSheet(style)

    def _load_body_parts_from_model(self):
        from source.video.tracking.model_config import ModelInfo
        model_path = self.model_path.text().strip()
        if not model_path:
            self.body_parts_label.setText("(Select model to load body parts)")
            self.body_parts_label.setStyleSheet(f"color: {_MUTED}; font-style: italic;")
            self.body_parts_list = ["center"]
            self.skeleton_edges = []
            # No model yet → picker shows centroid only.
            self._update_zone_body_part_combo([])
            self._update_skeleton_table()
            return

        # ``ModelInfo`` is the reader that knows BOTH toolkits: DeepLabCut's
        # ``pose_cfg.yaml`` and sleap-nn's ``training_config.yaml``. Asking it
        # first is what makes a SLEAP model load its parts at all, a SLEAP
        # directory contains none of the DLC file names scanned below, so the
        # scan alone silently produced the ``["center"]`` fallback and every
        # downstream picker, trigger and overlay inherited a single wrong part.
        info = ModelInfo.read(model_path)
        skeleton = list(info.skeleton)
        body_parts = list(info.body_parts)

        # Build list of yaml files to try reading
        path_obj = Path(model_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ('.yml', '.yaml'):
            # User selected a .yml file directly, read it
            pose_yaml_paths = [model_path]
        elif path_obj.is_dir():
            # Path is a directory, search for config files inside
            pose_yaml_paths = [
                os.path.join(model_path, "pose_cfg.yaml"),
                os.path.join(model_path, "pose.yaml"),
                os.path.join(model_path, "config.yaml"),
            ]
        elif body_parts:
            pose_yaml_paths = []
        else:
            self.body_parts_label.setText("(Invalid path)")
            self.body_parts_label.setStyleSheet("color: #f44336;")
            self.body_parts_list = ["center"]
            self.skeleton_edges = []
            self._update_zone_body_part_combo([])
            return

        for yaml_path in (pose_yaml_paths if not body_parts else []):
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, 'r') as f:
                        config = yaml.safe_load(f)

                    if not isinstance(config, dict):
                        continue

                    if 'bodyparts' in config:
                        body_parts = config['bodyparts']
                    elif 'all_joints_names' in config:
                        body_parts = config['all_joints_names']
                    elif 'multianimalbodyparts' in config:
                        body_parts = config['multianimalbodyparts']

                    if body_parts:
                        break
                except Exception as e:
                    logger.warning(f"Failed to read {yaml_path}: {e}")

        if body_parts:
            self.body_parts_list = body_parts
            self.skeleton_edges = [e for e in skeleton
                                   if e[0] in body_parts and e[1] in body_parts]
            label = ", ".join(body_parts)
            if self.skeleton_edges:
                label += f"   ({len(self.skeleton_edges)} skeleton edges)"
            self.body_parts_label.setText(label)
            self.body_parts_label.setStyleSheet("color: #4CAF50;")
            self._update_trigger_body_parts()
            self._update_annotate_checkboxes(body_parts)
            self._update_zone_body_part_combo(body_parts)
            self._update_coord_combos()
            self._update_skeleton_table()
        else:
            self.body_parts_label.setText("(No body parts found in config)")
            self.body_parts_label.setStyleSheet("color: #f44336;")
            self.body_parts_list = ["center"]
            self.skeleton_edges = []
            self._update_zone_body_part_combo([])
            self._update_skeleton_table()

    def _update_annotate_checkboxes(self, body_parts):
        """Rebuild the annotation checkboxes from the loaded body parts."""
        # Clear existing checkboxes
        for cb in self.annotate_checkboxes.values():
            cb.setParent(None)
        self.annotate_checkboxes.clear()

        for part in body_parts:
            cb = QtWidgets.QCheckBox(part)
            cb.setChecked(True)  # All annotated by default
            cb.setStyleSheet("color: #e0e0e0;")
            self.annotate_checkboxes[part] = cb
            self.annotate_parts_layout.addWidget(cb)

    @staticmethod
    def _find_center_body_part(parts):
        """Return the keypoint named ``center``/``centre``, or None.

        Case-insensitive but an exact word match, ``Center`` and ``centre``
        match, ``BodyCenter`` / ``mouse centre`` do not.
        """
        for bp in parts:
            if str(bp).strip().lower() in ("center", "centre"):
                return str(bp).strip()
        return None

    def _update_zone_body_part_combo(self, body_parts):
        """Repopulate the single body-part picker (Zone part dropdown).

        In **pose** mode the picker lists the model's real keypoints only,
        the synthetic ``"centroid"`` is omitted (it confuses against the
        named keypoints), and a keypoint named like ``center``/``centre`` is
        auto-selected. In **blob** mode the tracker emits only the synthetic
        centre, so the picker collapses to ``"centroid"``.

        This combo's selection drives BOTH:
          * PoseSink's centroid (which keypoint defines location +
            zone resolution per frame).
          * MCUPusher's ``zone_change_body_part`` (which keypoint's
            zone occupancy fires the intrinsic ``zone_changed`` event).
        Single source of truth, no separate "zone change body part"
        picker.
        """
        is_pose = self.mode_dlc.isChecked() or self.mode_sleap.isChecked()
        real_parts = [
            str(bp).strip() for bp in (body_parts or [])
            if str(bp).strip() and str(bp).strip() != "centroid"]

        prev = self.zone_body_part_combo.currentText().strip()
        if is_pose and real_parts:
            # Pose mode with a loaded model, keypoints only, no centroid.
            items = real_parts
            default = self._find_center_body_part(real_parts) or real_parts[0]
        else:
            # Blob mode, or pose with no model yet, synthetic centroid.
            items = ["centroid"] + real_parts
            default = "centroid"

        # Deduplicate while preserving order.
        seen = set(); ordered = []
        for it in items:
            if it not in seen:
                seen.add(it); ordered.append(it)
        self.zone_body_part_combo.blockSignals(True)
        self.zone_body_part_combo.clear()
        self.zone_body_part_combo.addItems(ordered)
        if prev in ordered:
            self.zone_body_part_combo.setCurrentText(prev)
        elif default in ordered:
            self.zone_body_part_combo.setCurrentText(default)
        else:
            self.zone_body_part_combo.setCurrentIndex(0)
        self.zone_body_part_combo.blockSignals(False)

    def _update_trigger_body_parts(self):
        # The Event-Triggers table is built lazily with the "Event & Trigger"
        # tab; skip until it exists so an early mode-change can't crash.
        if not hasattr(self, "triggers_table"):
            return
        # Cols 1 (Body Part) and 5 (Part B) are both body-part pickers.
        for row in range(self.triggers_table.rowCount()):
            for col in (1, 5):
                combo = self.triggers_table.cellWidget(row, col)
                if combo:
                    current = combo.currentText()
                    combo.clear()
                    combo.addItems(self.body_parts_list)
                    idx = combo.findText(current)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

    # Condition options for trigger dropdown
    # Conditions authorable with the (body_part + threshold + zones) columns.
    # ``head_angle`` (min/max) and ``distance`` (two parts) need extra inputs,
    # they exist in the evaluator and get added here once the table grows those
    # cells.
    # Every entry here is handled by the MCU push policy's _evaluate_condition
    # (the single evaluator), keep them in lock-step so the dropdown never
    # offers a condition that silently never fires.
    TRIGGER_CONDITIONS = [
        "in_zone", "enter_zone", "exit_zone", "not_in_zone",
        "speed_gt", "speed_lt", "freezing",
        "rotation_gt", "rotation_lt", "facing_line",
        "elongation_gt", "rearing",
        "distance_gt", "distance_lt",       # Body Part ↔ Part B distance
        "head_angle_gt", "head_angle_lt",   # head-vs-body angle at Part B (neck)
    ]

    # Conditions that need a HEAD-TAIL AXIS, two distinct keypoints. The
    # evaluator derives heading from ``FeatureTracker.axis = (tail, head)``;
    # with one point the heading is None and the rule can never fire, silently.
    # Blob tracking yields a single centroid, so under blob these are offered
    # but dead. Disabled with an explanatory tooltip instead.
    AXIS_ONLY_CONDITIONS = frozenset({
        "rotation_gt", "rotation_lt", "facing_line",
        "elongation_gt", "rearing",
        "head_angle_gt", "head_angle_lt",
    })

    CONDITION_HELP = {
        "in_zone": "Fires the frame the body part ENTERS any selected zone.\n"
                   "Despite the name this is an edge, not a level, every rule\n"
                   "fires once on becoming true. Identical to enter_zone.",
        "enter_zone": "Fires the frame the body part enters any selected zone.",
        "exit_zone": "Fires the frame the body part LEAVES every selected zone.",
        "not_in_zone": "Fires the frame the body part is no longer in any\n"
                       "selected zone. Identical to exit_zone.",
        "speed_gt": "Speed rises above Threshold (units follow the scale\n"
                    "calibration; px/s when uncalibrated).",
        "speed_lt": "Speed drops below Threshold.",
        "freezing": "Speed drops below Threshold. Pair with Hold ~500 ms so\n"
                    "a momentary pause is not scored as freezing.",
        "rotation_gt": "Turning rate (deg/s) rises above Threshold.\n"
                       "NEEDS a head-tail axis, two keypoints.",
        "rotation_lt": "Turning rate (deg/s) drops below Threshold.\n"
                       "NEEDS a head-tail axis, two keypoints.",
        "facing_line": "The animal's heading points within Threshold degrees\n"
                       "of a line zone. NEEDS a head-tail axis.",
        "elongation_gt": "Nose-tail distance / typical length rises above\n"
                         "Threshold (stretched). NEEDS a head-tail axis.",
        "rearing": "Foreshortened AND slow, a rearing proxy.\n"
                   "NEEDS a head-tail axis.",
        "distance_gt": "Distance between Body Part and Part B rises above\n"
                       "Threshold. Needs BOTH pickers set.",
        "distance_lt": "Distance between Body Part and Part B drops below\n"
                       "Threshold. Needs BOTH pickers set.",
        "head_angle_gt": "Head-vs-body angle at the neck rises above\n"
                         "Threshold. Part B is the NECK keypoint.\n"
                         "NEEDS tail, neck and head keypoints.",
        "head_angle_lt": "Head-vs-body angle at the neck drops below\n"
                         "Threshold. Part B is the NECK keypoint.\n"
                         "NEEDS tail, neck and head keypoints.",
    }

    def _tracker_has_axis(self) -> bool:
        """True when the active tracker yields ≥2 named keypoints.

        Pose mode (DLC / SLEAP) does. Blob reports a single centroid, so every
        orientation and posture rule is dead under it, see
        ``AXIS_ONLY_CONDITIONS``. Mode comes from the same radio pair
        ``get_settings`` reads, so the gating can never disagree with what is
        actually saved.
        """
        for attr in ("mode_sleap", "mode_dlc"):
            btn = getattr(self, attr, None)
            if btn is not None and btn.isChecked():
                return True
        if getattr(self, "mode_sleap", None) is not None:
            return False          # radios exist and neither pose one is on
        # Radios not built yet: fall back to the configured keypoints.
        return len(getattr(self, "body_parts_list", []) or []) >= 2

    _BLOCKED_NOTE = ("Unavailable with blob tracking: it reports a single "
                     "centroid, and this rule needs two keypoints to form a "
                     "head-tail axis. Switch to pose tracking to use it.")

    def _condition_tip(self, name: str, has_axis: bool) -> tuple:
        """``(tooltip, blocked)`` for one condition under the active tracker."""
        tip = self.CONDITION_HELP.get(name, "")
        if has_axis or name not in self.AXIS_ONLY_CONDITIONS:
            return tip, False
        return f"{tip}\n\n{self._BLOCKED_NOTE}".strip(), True

    def _apply_axis_availability(self, combo) -> None:
        """Grey out the axis-only conditions in one Condition dropdown."""
        has_axis = self._tracker_has_axis()
        model = combo.model()
        for i in range(combo.count()):
            tip, blocked = self._condition_tip(combo.itemText(i), has_axis)
            item = model.item(i) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(not blocked)
                item.setToolTip(tip)
        # A saved rule can still SELECT a blocked condition, loading a config
        # must not silently rewrite the user's rules. Say so on the closed
        # combo, or a dead rule looks configured.
        combo.setToolTip(self._condition_tip(combo.currentText(), has_axis)[0])

    def refresh_condition_availability(self) -> None:
        """Re-apply the blob/pose gating to every Condition dropdown."""
        if not hasattr(self, "triggers_table"):
            return
        for row in range(self.triggers_table.rowCount()):
            combo = self.triggers_table.cellWidget(row, 0)
            if combo is not None:
                self._apply_axis_availability(combo)

    def _add_trigger(self):
        row = self.triggers_table.rowCount()
        self.triggers_table.insertRow(row)

        cond = QtWidgets.QComboBox()
        cond.addItems(self.TRIGGER_CONDITIONS)
        # Identity-bind (not row index), a captured index goes stale after
        # a row deletion and would gate the wrong row's widgets.
        cond.currentTextChanged.connect(
            lambda text, c=cond: self._on_trigger_condition_changed(c, text))
        # Per-item help + grey out anything the active tracker cannot evaluate.
        self._apply_axis_availability(cond)
        cond.currentTextChanged.connect(
            lambda _text, c=cond: c.setToolTip(
                self._condition_tip(c.currentText(), self._tracker_has_axis())[0]))
        self.triggers_table.setCellWidget(row, 0, cond)

        body_part = QtWidgets.QComboBox()
        body_part.addItems(self.body_parts_list)
        body_part.setToolTip(
            "The tracked point this rule is measured on.\n"
            "Blob tracking offers only 'centroid'; pose tracking offers every\n"
            "keypoint in the model.")
        self.triggers_table.setCellWidget(row, 1, body_part)

        zone_btn = MultiSelectZoneButton(self.zone_names)
        zone_btn.setToolTip(
            "Zone(s) this rule applies to, tick as many as you like; the\n"
            "rule treats them as ONE region (in any of them counts as in).\n"
            "Draw zones on the Zones tab first. Ignored by non-zone rules.")
        self.triggers_table.setCellWidget(row, 2, zone_btn)

        event = QtWidgets.QLineEdit()
        event.setPlaceholderText(f"event_{row+1}")
        event.setToolTip(
            "MCU event fired when this rule becomes true.\n\n"
            "The name must appear VERBATIM in your task's `events` list, or\n"
            "the board accepts the message and does nothing, the most common\n"
            "silent failure in the whole trigger chain.\n"
            "Leave blank to score the rule in the plot without telling the MCU.")
        self.triggers_table.setCellWidget(row, 3, event)

        threshold = NumericLineEdit()
        threshold.setRange(0, 9999)
        threshold.setValue(90)
        threshold.setEnabled(False)  # Only enabled for speed/rotation/facing
        threshold.setToolTip(
            "Comparison value for speed / rotation / distance / angle rules.\n"
            "Units follow the zone scale calibration where one exists,\n"
            "otherwise pixels. Greyed out for zone rules, which have nothing\n"
            "to compare.")
        self.triggers_table.setCellWidget(row, 4, threshold)

        # Part B, second body part for distance (Body Part ↔ Part B) or the
        # neck joint for head-angle. Disabled for every other condition.
        part_b = QtWidgets.QComboBox()
        part_b.addItems(self.body_parts_list)
        part_b.setEnabled(False)
        part_b.setToolTip(
            "The SECOND tracked point. Only two kinds of rule use it:\n\n"
            "  distance_gt / distance_lt\n"
            "      measures the distance Body Part ↔ Part B\n"
            "      (e.g. nose ↔ tail_base to score stretching)\n\n"
            "  head_angle_gt / head_angle_lt\n"
            "      Part B is the NECK joint; the angle is head-vs-body\n"
            "      measured at it\n\n"
            "Greyed out for every other condition, which needs only one point.")
        self.triggers_table.setCellWidget(row, 5, part_b)

        # Hold, min duration (ms) the raw condition must hold before the
        # trigger fires (debounce). 0 = fire immediately. Applies to every
        # condition; the evaluator's duration gate reads it.
        hold = NumericLineEdit()
        hold.setRange(0, 60000)
        hold.setSingleStep(50)
        hold.setValue(0)
        hold.setSuffix(" ms")
        hold.setToolTip(
            "Minimum time the condition must stay true before firing.\n"
            "0 = instant. Use ~300–500 ms for freezing / rearing to reject blips.")
        hold.setMaximumWidth(84)
        self.triggers_table.setCellWidget(row, 6, hold)

        # Presentation: chip/plot colour + show-on-video + include-in-plot.
        colour_btn = QtWidgets.QPushButton()
        colour_btn.setFixedSize(26, 18)
        colour_btn._hex = "#39c5cf"
        colour_btn.setStyleSheet(f"background:{colour_btn._hex};border:1px solid #2b3a4d;border-radius:3px;")
        colour_btn.setToolTip("Chip / plot colour")

        def _pick(btn):
            from PySide6 import QtGui
            c = QtWidgets.QColorDialog.getColor(QtGui.QColor(btn._hex), self)
            if c.isValid():
                btn._hex = c.name()
                btn.setStyleSheet(f"background:{btn._hex};border:1px solid #2b3a4d;border-radius:3px;")

        # `clicked` emits the checked bool. Binding the button as a DEFAULT
        # argument does not work: the emitted False is passed positionally and
        # overwrites it, so the slot ran `False._hex`. Swallow the bool in a
        # leading parameter and pass the button explicitly.
        colour_btn.clicked.connect(
            lambda _checked=False, b=colour_btn: _pick(b))
        self.triggers_table.setCellWidget(row, 7, colour_btn)

        video_cb = QtWidgets.QCheckBox(); video_cb.setChecked(True)
        video_cb.setToolTip("Draw a chip on the video for this trigger")
        self.triggers_table.setCellWidget(row, 8, video_cb)
        plot_cb = QtWidgets.QCheckBox(); plot_cb.setChecked(True)
        plot_cb.setToolTip("Include this trigger in the Session-Plot lane")
        self.triggers_table.setCellWidget(row, 9, plot_cb)

        del_btn = QtWidgets.QPushButton("X")
        del_btn.setFixedWidth(30)
        from source.gui.styles import BUTTON_STYLE as _BS_x
        from source.gui.styles import COLORS as _C_x
        del_btn.setStyleSheet(_BS_x.format(
            color=_C_x['danger'], hover_color=_C_x['danger_hover']))
        def make_delete(r):
            return lambda: self._delete_trigger_row(r)
        del_btn.clicked.connect(make_delete(row))
        self.triggers_table.setCellWidget(row, 10, del_btn)

    def _on_trigger_condition_changed(self, cond_combo, condition_text):
        """Update zone/body_part/threshold enablement based on selected condition."""
        row = -1
        for r in range(self.triggers_table.rowCount()):
            if self.triggers_table.cellWidget(r, 0) is cond_combo:
                row = r
                break
        if row < 0:
            return
        zone_btn = self.triggers_table.cellWidget(row, 2)
        body_part = self.triggers_table.cellWidget(row, 1)
        threshold = self.triggers_table.cellWidget(row, 4)
        part_b = self.triggers_table.cellWidget(row, 5)
        c = condition_text
        is_speed = c in ("speed_gt", "speed_lt", "freezing")
        is_rotation = c in ("rotation_gt", "rotation_lt")
        is_facing = c == "facing_line"
        is_head_angle = c in ("head_angle_gt", "head_angle_lt")
        is_angle = is_facing or is_head_angle
        is_ratio = c in ("elongation_gt", "rearing")   # body-length ratio
        is_distance = c in ("distance_gt", "distance_lt")
        is_zone = c in ("in_zone", "enter_zone", "exit_zone", "not_in_zone")
        needs_threshold = is_speed or is_rotation or is_angle or is_ratio or is_distance
        # Only zone conditions use the zone picker.
        if zone_btn:
            zone_btn.setEnabled(is_zone)
        # Speed / ratio conditions don't need a single body part (centroid/axis).
        if body_part:
            body_part.setEnabled(not is_speed and not is_ratio)
        # Part B is the 2nd body part (distance) or the neck joint (head-angle).
        if part_b:
            part_b.setEnabled(is_distance or is_head_angle)
        if threshold:
            threshold.setEnabled(needs_threshold)
            if is_speed:
                threshold.setRange(0, 9999); threshold.setValue(100)
                threshold.setSuffix(" px/s")     # relabelled to cm/s when a scale is set
            elif is_rotation:
                threshold.setRange(0, 3600); threshold.setValue(90)
                threshold.setSuffix(" °/s")
            elif is_angle:
                threshold.setRange(1, 180); threshold.setValue(60)
                threshold.setSuffix("°")
            elif is_ratio:
                threshold.setRange(0, 3); threshold.setValue(0.6 if c == "rearing" else 1.1)
                threshold.setSuffix(" ×L")
            elif is_distance:
                threshold.setRange(0, 9999); threshold.setValue(40)
                threshold.setSuffix(" px")       # → mm when a scale is set
            else:
                threshold.setSuffix("")

    def _delete_trigger_row(self, row):
        for r in range(self.triggers_table.rowCount()):
            btn = self.triggers_table.cellWidget(r, 10)
            if btn and btn == self.sender():
                self.triggers_table.removeRow(r)
                return
        if row < self.triggers_table.rowCount():
            self.triggers_table.removeRow(row)

    def _take_background(self, setup_id: int, silent: bool = False) -> bool:
        """Capture and save background for a box.

        Args:
            box_id: Box identifier
            silent: If True, skip error dialogs (used for auto-capture)

        Returns:
            True if background was captured successfully
        """
        try:
            callback = self.get_frame_callbacks.get(setup_id)
            if not callback:
                if not silent:
                    QtWidgets.QMessageBox.warning(self, "No Camera", f"No camera for Box {setup_id}")
                return False

            # Both live on MainWindowBase, so ask rather than probe, a parent
            # that has neither is not a main window and there is nowhere to
            # put a background anyway.
            mw = self.window()
            try:
                bg_dir = mw._background_images_dir()
                capture = mw._capture_background_for_box
            except AttributeError:
                bg_dir = capture = None
            if bg_dir is None or capture is None:
                if not silent:
                    QtWidgets.QMessageBox.warning(
                        self, "No project loaded",
                        "Save the project first, backgrounds live inside "
                        "the project's background_images/ folder.")
                return False

            # One capture path for the whole app (median burst, write, stamp);
            # this passes the dialog's own camera callback into it.
            if not capture(setup_id, get_frame=callback):
                if not silent:
                    QtWidgets.QMessageBox.warning(
                        self, "No Frame", f"No frame from Box {setup_id}")
                return False
            self.backgrounds[setup_id] = str(bg_dir / f"box{setup_id}.png")
            self._update_bg_status()
            return True

        except Exception as e:
            logger.error(f"Error capturing background for box {setup_id}: {e}")
            if not silent:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to capture background: {e}")
            return False

    def _update_bg_status(self):
        """Update the compact background status label."""
        if not hasattr(self, 'bg_status_label'):
            return
        n = len(self.backgrounds)
        if n == 0:
            self.bg_status_label.setText("")
        else:
            ids = sorted(self.backgrounds.keys())
            self.bg_status_label.setText(f"BG: {', '.join(str(i) for i in ids)}")
            self.bg_status_label.setStyleSheet("color: #4CAF50; font-size: 10px;")

    def _on_resize_changed(self, value):
        self._mark_dlc_dirty("Resize changed, re-initialize!")

    def _on_dlc_confidence_changed(self, value):
        # No dirty mark: confidence is a post-inference cutoff applied in
        # PoseSink, absent from the model cache key, and pushed live by
        # apply_tracking_config. Demanding a re-initialise for it would cost a
        # drain, a close and a fresh model load, minutes on a TensorRT engine
        #, to move a threshold the network never sees.
        return

    def _on_dlc_instances_changed(self, value):
        self._mark_dlc_dirty("Instances changed, re-initialize!")

    def _mark_dlc_dirty(self, reason: str) -> None:
        """Settings changed since last init → re-enable the Init button.

        Called by every setting that requires a model rebuild
        (model_path, resize, confidence, n_instances). Idempotent: only
        flips state when actually initialised before.
        """
        if not self._dlc_initialized:
            return
        self._dlc_initialized = False
        self.dlc_init_status.setText(reason)
        self.dlc_init_status.setStyleSheet("color: #fbbf24; font-size: 10px;")
        self.dlc_init_btn.setEnabled(True)
        from source.gui.styles import BUTTON_STYLE as _BS_r
        from source.gui.styles import COLORS as _C_r
        self.dlc_init_btn.setStyleSheet(_BS_r.format(
            color=_C_r['warning'], hover_color=_C_r['warning_hover']))

    def _mark_dlc_clean(self, status: str = "Ready") -> None:
        """The configured signature matches the live PoseSink, disable
        the Init button and paint it success-green. Called on dialog
        reopen when the host confirms the pose subsystem is already
        initialised for the saved config."""
        self._dlc_initialized = True
        self.dlc_init_status.setText(status)
        self.dlc_init_status.setStyleSheet("color: #34d399; font-size: 10px;")
        self.dlc_init_btn.setEnabled(False)
        from source.gui.styles import BUTTON_STYLE as _BS_s
        from source.gui.styles import COLORS as _C_s
        self.dlc_init_btn.setStyleSheet(_BS_s.format(
            color=_C_s['success'], hover_color=_C_s['success_hover']))

    def _mark_dlc_failed(self, message: str = "Failed") -> None:
        """Init attempt failed, leave button enabled (red) so the user
        can retry. Mirrors clean/dirty so all three init-state writes
        live in one place."""
        self._dlc_initialized = False
        self.dlc_init_btn.setEnabled(True)
        self.dlc_init_status.setText(message)
        self.dlc_init_status.setStyleSheet("color: #fb7185; font-size: 10px;")

    def _init_dlc(self):
        """Emit signal to initialize pose tracker (DLC or SLEAP)."""
        model_path = self.model_path.text().strip()
        if not model_path:
            QtWidgets.QMessageBox.warning(self, "No Model", "Select a model path first.")
            return
        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, "Not Found", f"Path does not exist:\n{model_path}")
            return

        enabled_boxes = [bid for bid, cb in self.box_checkboxes.items() if cb.isChecked()]
        if not enabled_boxes:
            QtWidgets.QMessageBox.warning(self, "No Boxes", "Select at least one box.")
            return

        tracker_type = "sleap" if self.mode_sleap.isChecked() else "dlc"
        self.dlc_init_status.setText(f"Initializing {tracker_type.upper()}...")
        self.dlc_init_status.setStyleSheet("color: #FF9800; font-size: 10px;")
        self.dlc_init_btn.setEnabled(False)

        cfg = {
            "tracker_type": tracker_type,
            "model_path": model_path,
            "dlc_resize": self.dlc_resize_spin.value(),
            "dlc_confidence": self.dlc_confidence_spin.value(),
            "body_parts": self.body_parts_list,
            "skeleton": [list(e) for e in self.skeleton_edges],
            "enabled_boxes": enabled_boxes,
            "pose_instances": self.pose_instances_spin.value(),
        }
        if tracker_type == "sleap":
            cfg.update({
                "sleap_model_type":    "auto",
                "sleap_centroid_path": self.sleap_centroid_path.text().strip() or None,
                "sleap_runtime":       self.sleap_runtime_combo.currentText(),
                "sleap_device":        self.sleap_device_combo.currentText(),
                "sleap_fp16":          self.sleap_fp16_cb.isChecked(),
            })
        elif tracker_type == "dlc":
            cfg.update({
                "dlc_model_type": self.dlc_model_type_combo.currentText(),
                "dlc_precision":  self.dlc_precision_combo.currentText(),
                "dlc_device":     self.dlc_device_combo.currentText().strip() or "auto",
            })
        cfg["pose_colour_mode"] = self.pose_colour_combo.currentText()
        cfg.update(self._pose_input_settings())
        self.dlc_init_requested.emit(cfg)

    def set_dlc_init_result(self, success: bool, message: str = ""):
        """Called by main_window after DLC init attempt. Routes to one
        of three init-state writers depending on outcome.
        ``_mark_dlc_dirty`` re-enables the button the moment any of the
        tracked fields (model_path/resize/conf/instances) changes."""
        if success:
            self._mark_dlc_clean(message or "Ready")
        else:
            self._mark_dlc_failed(message or "Failed")

    def get_settings(self) -> dict:
        triggers = []
        # triggers_table is built with the "Event & Trigger" tab; tolerate its
        # absence so a settings read never depends on tab-build order.
        rows = self.triggers_table.rowCount() if hasattr(self, "triggers_table") else 0
        for row in range(rows):
            cond = self.triggers_table.cellWidget(row, 0)
            body_part = self.triggers_table.cellWidget(row, 1)
            zone_btn = self.triggers_table.cellWidget(row, 2)
            event = self.triggers_table.cellWidget(row, 3)
            threshold_w = self.triggers_table.cellWidget(row, 4)

            if cond and body_part and zone_btn and event:
                zones = zone_btn.get_zones() if hasattr(zone_btn, 'get_zones') else []
                trig = {
                    "condition": cond.currentText(),
                    "body_part": body_part.currentText(),
                    "zones": zones,
                    "event_name": event.text() or f"event_{row+1}",
                }
                if threshold_w and threshold_w.isEnabled():
                    trig["threshold"] = threshold_w.value()
                # Part B, second body part (distance) / neck joint (head-angle).
                part_b = self.triggers_table.cellWidget(row, 5)
                if part_b is not None and part_b.isEnabled():
                    cond_text = cond.currentText()
                    if cond_text in ("distance_gt", "distance_lt"):
                        trig["part_a"] = body_part.currentText()
                        trig["part_b"] = part_b.currentText()
                    elif cond_text in ("head_angle_gt", "head_angle_lt"):
                        trig["neck"] = part_b.currentText()
                # Hold (debounce, ms), only serialise when non-zero.
                hold_w = self.triggers_table.cellWidget(row, 6)
                if hold_w is not None and hold_w.value():
                    trig["duration_ms"] = int(hold_w.value())
                colour_btn = self.triggers_table.cellWidget(row, 7)
                video_cb = self.triggers_table.cellWidget(row, 8)
                plot_cb = self.triggers_table.cellWidget(row, 9)
                if colour_btn is not None:
                    trig["color"] = getattr(colour_btn, "_hex", "#39c5cf")
                trig["show_on_video"] = bool(video_cb.isChecked()) if video_cb else True
                trig["plot"] = bool(plot_cb.isChecked()) if plot_cb else True
                triggers.append(trig)

        # DLC annotation and zone body part settings
        annotate_parts = [
            name for name, cb in self.annotate_checkboxes.items() if cb.isChecked()
        ]
        zone_body_part = self.zone_body_part_combo.currentText() or ""

        # In pose mode the model path is the live widget text, falling back to
        # the last real path we stashed (``dlc_model_path``) so a transiently
        # empty field never zeroes a known model. Blob mode has no model path.
        _pose_mode = self.mode_dlc.isChecked() or self.mode_sleap.isChecked()
        _model_path = (
            (self.model_path.text().strip() or self.dlc_model_path or "")
            if _pose_mode else "")

        result = {
            "mode": "sleap" if self.mode_sleap.isChecked() else ("dlc" if self.mode_dlc.isChecked() else "normal"),
            "tracker_type": "sleap" if self.mode_sleap.isChecked() else ("dlc" if self.mode_dlc.isChecked() else "blob"),
            "model_path": _model_path,
            "body_parts": self.body_parts_list,
            "skeleton": [list(e) for e in self.skeleton_edges],
            "annotate_parts": annotate_parts,
            "zone_body_part": zone_body_part,
            "dlc_confidence": self.dlc_confidence_spin.value(),
            "dlc_resize": self.dlc_resize_spin.value(),
            "pose_instances": self.pose_instances_spin.value(),
            "marker_size": self.marker_size_spin.value(),
            # SLEAP-only options (harmless for DLC/blob, ignored downstream).
            # Type stays "auto" (backend re-detects at init); the badge is a
            # display hint, so only persist a concrete detected type.
            "sleap_model_type": (self.sleap_type_label.text()
                                 if (hasattr(self, "sleap_type_label")
                                     and self.sleap_type_label.text() in _SLEAP_TYPES)
                                 else "auto"),
            "sleap_centroid_path": (self.sleap_centroid_path.text().strip()
                                    if hasattr(self, "sleap_centroid_path") else ""),
            "sleap_runtime": (self.sleap_runtime_combo.currentText()
                              if hasattr(self, "sleap_runtime_combo") else "auto"),
            "sleap_device": (self.sleap_device_combo.currentText()
                             if hasattr(self, "sleap_device_combo") else "auto"),
            "sleap_fp16": (self.sleap_fp16_cb.isChecked()
                           if hasattr(self, "sleap_fp16_cb") else False),
            "dlc_model_type": (self.dlc_model_type_combo.currentText()
                               if hasattr(self, "dlc_model_type_combo") else "auto"),
            "dlc_precision": (self.dlc_precision_combo.currentText()
                              if hasattr(self, "dlc_precision_combo") else "FP32"),
            "dlc_device": (self.dlc_device_combo.currentText().strip() or "auto"
                           if hasattr(self, "dlc_device_combo") else "auto"),
            "pose_colour_mode": (self.pose_colour_combo.currentText()
                                 if hasattr(self, "pose_colour_combo") else "auto"),
            **self._pose_input_settings(),
            "triggers": triggers,
            "enabled_boxes": [bid for bid, cb in self.box_checkboxes.items() if cb.isChecked()],
            # Every box the dialog rendered, ticked or not. Lets
            # ``_apply_dialog_config`` save rig-level fields (mode,
            # model_path, resize, conf, body_parts) to ALL boxes' TCs even
            # when no per-box checkbox is ticked.
            "all_dialog_boxes": list(self.box_ids),
            "coord_mapping": self.get_coord_mapping(),
            "smooth_tracking": self.smooth_tracking_cb.isChecked(),
            # MCU push gates (per-box, same value applied to every selected
            # box). Read back into ``TrackingConfig.push_*_to_mcu`` by
            # ``MainWindowBase._apply_dialog_config``.
            "push_zones_to_mcu":      self.push_zones_to_mcu_cb.isChecked(),
            "push_coords_to_mcu":     self.push_coords_to_mcu_cb.isChecked(),
            "push_frame_event":       self.push_frame_event_cb.isChecked(),
            # Single source of truth: the Zone-part combo above. Its
            # value drives BOTH PoseSink.centroid_body_part AND
            # MCUPusher.zone_change_body_part.
            "zone_change_body_part": (zone_body_part or "centroid"),
            # Per-box "Save annotated video" flags. Read at session start
            # by main_window and applied via controller.set_annotate_saved.
            "annotate_saved": {
                str(bid): cb.isChecked()
                for bid, cb in self.box_annotate_checkboxes.items()
            },
        }

        if not self.mode_dlc.isChecked() and not self.mode_sleap.isChecked():
            result.update({
                "detect_dark": self.detect_dark_cb.isChecked(),
                "threshold": self.threshold_spin.value(),
                "min_area": self.min_area_spin.value(),
                "max_area": self.max_area_spin.value(),
                "blur_mode": self.blur_mode_combo.currentText(),
                "blur_kernel_size": self.blur_size_spin.value(),
                "bg_mode": self.bg_mode_combo.currentText(),
                "open_kernel_size": self.open_kernel_spin.value(),
                "close_kernel_size": self.close_kernel_spin.value(),
                # Simple/self_norm knobs. Emitted whatever the chip says:
                # they are inert unless bg_mode is self_norm, and always
                # writing them means switching to Blob and back cannot
                # silently drop a tuned value.
                "self_norm_sigma": float(self.sn_sigma_spin.value()),
                "self_norm_smooth_sigma": float(self.sn_smooth_spin.value()),
                "self_norm_minsize": int(self.sn_minsize_spin.value()),
            })
            # Calibration-only params (no widget here) ride along, or the
            # settings that made the preview track never reach the tracker.
            result.update(self._blob_calibration_extra)

        return result

    def set_zone_names(self, names: List[str]):
        self.zone_names = names
        for row in range(self.triggers_table.rowCount()):
            zone_btn = self.triggers_table.cellWidget(row, 2)
            if zone_btn and hasattr(zone_btn, 'set_zone_names'):
                zone_btn.set_zone_names(names)


class UnifiedTrackingDialog(QtWidgets.QDialog):
    """Two-tab tracking dialog with zone editor and settings."""

    dlc_init_requested = QtCore.Signal(dict)
    # Emitted on every in-dialog zone/scale edit so the host repaints the
    # live camera overlay with the new geometry. PREVIEW ONLY, nothing is
    # persisted; the host rolls it back on a non-apply close.
    # Args: (setup_id, zones list).
    zones_live_changed = QtCore.Signal(int, object)
    # Emitted by "Save Zones to Project": a zones-only partial commit the
    # host persists immediately and excludes from a later Cancel/Discard
    # rollback. Arg: {setup_id: zones list} for every box.
    zones_save_requested = QtCore.Signal(dict)

    def __init__(
        self,
        parent=None,
        setup_id: Optional[int] = None,
        box_ids: List[int] = None,
        get_frame_callback: Optional[Callable] = None,
        get_frame_callbacks: Dict[int, Callable] = None,
        connected_cameras: Dict[int, Any] = None,
        initial_zones_path: Optional[str] = None,
        initial_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(parent)
        self.box_ids = list(box_ids) if box_ids else []
        if setup_id and setup_id not in self.box_ids:
            self.box_ids.insert(0, setup_id)

        self.current_box_id = self.box_ids[0] if self.box_ids else None

        self.get_frame_callbacks = dict(get_frame_callbacks) if get_frame_callbacks else {}
        if get_frame_callback and self.current_box_id:
            self.get_frame_callbacks[self.current_box_id] = get_frame_callback

        self.connected_cameras = connected_cameras or {}

        self.zones_per_box: Dict[int, List[Dict]] = {bid: [] for bid in self.box_ids}
        # Per-box pixel-to-physical-units scale calibration. Required when
        # zones are present (gated by _validate_scale_for_zones). Each
        # entry is the dict returned by ZoneEditor.get_scale_meta():
        #   {value, unit, pixel_length, px_per_unit, unit_per_px}
        self.scale_per_box: Dict[int, Dict] = {bid: {} for bid in self.box_ids}

        self.setWindowTitle("Tracking Configuration")
        # Size dialog to fit 5x image + UI chrome within screen
        # UI chrome: zone list(140) + margins(30) horizontal
        #            toolbar(28) + buttons(30) + tab_bar(30) + margins(30) vertical
        CHROME_W = 140 + 30
        CHROME_H = 28 + 30 + 30 + 30
        img_w, img_h = 640 * 5, 480 * 5  # 5x default
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            max_canvas_w = avail.width() - CHROME_W - 40
            max_canvas_h = avail.height() - CHROME_H - 40
            # Open a touch smaller than the full fit so the dialog doesn't
            # dominate the screen (still centred, still resizable).
            DIALOG_FRAC = 0.85
            scale = min(max_canvas_w / img_w, max_canvas_h / img_h, 1.0) * DIALOG_FRAC
            canvas_w = int(img_w * scale)
            canvas_h = int(img_h * scale)
            w = canvas_w + CHROME_W
            h = canvas_h + CHROME_H
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
        else:
            self.resize(900, 650)
        self.setMinimumSize(400, 300)

        self._zones_dirty = False
        # Whether the host should fan this dialog's config out to the boxes on
        # close. Only set by an explicit "Apply & Close", Cancel / X / Escape
        # leave it False so a user who just looked at, or aborted, the dialog
        # does NOT overwrite every box's config (the host rolls its live
        # preview back).
        self._apply_on_close = False
        # Set before an explicit apply so closeEvent's confirm prompt is
        # skipped (Apply already committed; don't ask again).
        self._suppress_close_prompt = False

        self._build_ui(initial_zones_path)

        if initial_config:
            try:
                self._apply_config(initial_config)
            except Exception as e:
                logger.warning(f"Failed to apply config: {e}")

        # Reset dirty after initial load, then snapshot the open-time config.
        # The window-X close prompt fires only when the LIVE config differs
        # from this snapshot, so it catches mode/model/blob/trigger edits, not
        # just zone edits (the pose-config-lost bug came from prompting on
        # zones alone). Snapshot is a stable JSON string for a cheap compare.
        self._zones_dirty = False
        self._initial_snapshot = self._config_snapshot()

    def _build_ui(self, zones_path):
        from source.gui.widgets.zone_editor import ZoneEditorWidget

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.tabs = QtWidgets.QTabWidget()

        # Tab 1: Zone Editor
        zone_tab = QtWidgets.QWidget()
        zone_layout = QtWidgets.QVBoxLayout(zone_tab)
        zone_layout.setContentsMargins(5, 5, 5, 5)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Box:"))
        self.box_combo = QtWidgets.QComboBox()
        self.box_combo.setMinimumWidth(100)
        if self.box_ids:
            for bid in self.box_ids:
                self.box_combo.addItem(f"Box {bid}", bid)
        else:
            self.box_combo.addItem("No boxes", None)
            self.box_combo.setEnabled(False)
        self.box_combo.currentIndexChanged.connect(self._on_box_changed)
        selector_row.addWidget(self.box_combo)

        selector_row.addSpacing(20)

        # Export / Import are snippet operations: they read/write a standalone
        # JSON fragment the operator names. They do NOT persist to the project
        #, Apply & Close does. Import stages the fragment into the dialog, so
        # it commits with everything else.
        save_box_btn = QtWidgets.QPushButton("Export Box Zones")
        save_box_btn.setToolTip("Write this box's zones to a JSON file for "
                                "reuse. Does NOT save to the project, use "
                                "\"Save Zones to Project\" for that.")
        save_box_btn.clicked.connect(self._save_box_zones)
        selector_row.addWidget(save_box_btn)

        save_project_btn = QtWidgets.QPushButton("Save Zones to Project")
        save_project_btn.setToolTip(
            "Commit every box's zones to the loaded project NOW. Saved zones "
            "are kept even if you later close this dialog without Apply, "
            "only tracker settings stay pending.")
        save_project_btn.clicked.connect(self._save_zones_to_project)
        selector_row.addWidget(save_project_btn)

        load_box_btn = QtWidgets.QPushButton("Import Box Zones")
        load_box_btn.setToolTip("Load zones from a JSON file into this box.")
        load_box_btn.clicked.connect(self._load_box_zones)
        selector_row.addWidget(load_box_btn)

        # Beside Save/Load Box Zones, because that is where the operator
        # is working on zones. Propagation to other boxes is explicit and
        # on demand, a silent auto-copy makes closing the editor look
        # like it reconfigured boxes the operator never touched.
        self.copy_zones_btn = QtWidgets.QPushButton("Copy zones to all boxes")
        self.copy_zones_btn.clicked.connect(self._copy_zones_to_all_boxes)
        # Disabled rather than hidden with a single box: a control that
        # vanishes is indistinguishable from one that was removed, which is
        # exactly how this one came to be reported missing.
        multi = len(self.box_ids) > 1
        self.copy_zones_btn.setEnabled(multi)
        self.copy_zones_btn.setToolTip(
            "Replace every other box's zones with this box's zones "
            "(and its scale calibration)."
            if multi else
            "Only one box is configured; there is nothing to copy to.")
        selector_row.addWidget(self.copy_zones_btn)

        selector_row.addStretch()
        zone_layout.addLayout(selector_row)

        callback = self.get_frame_callbacks.get(self.current_box_id) if self.current_box_id else None
        self.zone_editor = ZoneEditorWidget(
            parent=self,
            setup_id=self.current_box_id,
            get_frame_callback=callback
        )
        self.zone_editor.zones_changed.connect(self._on_zones_changed)
        zone_layout.addWidget(self.zone_editor, stretch=1)

        self.tabs.addTab(zone_tab, "Zone Editor")

        # Tab 2: Tracking Settings
        self.settings_panel = TrackingSettingsPanel(
            parent=self,
            box_ids=self.box_ids,
            zone_names=self.zone_editor.get_zone_names(),
            connected_cameras=self.connected_cameras,
            get_frame_callbacks=self.get_frame_callbacks
        )
        self.settings_panel.dlc_init_requested.connect(self.dlc_init_requested.emit)
        self.tabs.addTab(self.settings_panel, "Tracking Settings")

        # Tab 3: Event & Trigger, coord-mapping + push-to-MCU gates + the
        # Event-Triggers table (authored here, evaluated by the MCU push policy).
        self.tabs.addTab(
            self.settings_panel.build_event_trigger_widget(), "Event && Trigger")

        layout.addWidget(self.tabs)

        # Bottom buttons
        btn_row = QtWidgets.QHBoxLayout()

        load_btn = QtWidgets.QPushButton("Import Config")
        load_btn.setStyleSheet("padding: 6px 12px;")
        load_btn.setToolTip("Load a tracking-config snippet into the dialog "
                            "(stages it; Apply & Close commits it).")
        load_btn.clicked.connect(self._load_config)
        btn_row.addWidget(load_btn)

        save_btn = QtWidgets.QPushButton("Export Config")
        save_btn.setStyleSheet("background: #2196F3; color: white; padding: 6px 12px; font-weight: bold;")
        save_btn.setToolTip("Write the current config to a reusable JSON "
                            "snippet (does not apply to boxes).")
        save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(save_btn)

        btn_row.addStretch()

        # Explicit apply / cancel. Only "Apply & Close" pushes this config to
        # the boxes; Cancel (and the window X / Escape) discard, so closing
        # the dialog never silently reconfigures every box.
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet("padding: 6px 12px;")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        apply_btn = QtWidgets.QPushButton("Apply && Close")
        apply_btn.setStyleSheet(
            "background: #4CAF50; color: white; padding: 6px 12px; font-weight: bold;")
        apply_btn.clicked.connect(self._apply_and_close)
        btn_row.addWidget(apply_btn)

        layout.addLayout(btn_row)

    def _apply_and_close(self):
        """The single commit action: gate, then fan this config out to the
        boxes and close.

        The scale + model gates run HERE, not only on file export, a config
        applied to the boxes with zones but no calibration, or pose mode with
        no model, is exactly as broken as one saved to a file. If a gate
        fails the dialog stays open so the operator can fix it.
        """
        if not self._pre_commit_gate_ok():
            return
        self._apply_on_close = True
        self._suppress_close_prompt = True
        self.accept()

    def _config_snapshot(self) -> str:
        """Stable JSON string of the full config, for dirty comparison.

        Floats are rounded so a live frame arriving after the open-time
        snapshot, which rescales pixel zones then re-normalizes, cannot flip
        the low-order digits and raise a spurious "unsaved changes" prompt with
        no user edit. 4 decimals of a normalized [0,1] coordinate is sub-pixel.
        """
        import json

        def _round(o):
            if isinstance(o, float):
                return round(o, 4)
            if isinstance(o, dict):
                return {k: _round(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_round(v) for v in o]
            return o

        try:
            return json.dumps(_round(self.get_full_config()),
                              sort_keys=True, default=str)
        except Exception:
            return ""

    def _is_dirty(self) -> bool:
        """True when the live config differs from the open-time snapshot."""
        return self._config_snapshot() != getattr(self, "_initial_snapshot", "")

    def _pre_commit_gate_ok(self) -> bool:
        """Block a commit that would persist a broken config.

        Two gates, both mandatory on the apply path:
          * every box with zones must have a scale calibration (downstream
            analysis converts pixels → physical units);
          * pose mode (DLC/SLEAP) must have a model path.
        Returns True when both pass; shows the offending message and returns
        False otherwise.
        """
        missing = self._missing_scale_boxes()
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Scale Required",
                "These boxes have zones but no scale calibration:\n  "
                + ", ".join(f"Box {b}" for b in missing) + "\n\n"
                "Set the scale value + unit and click Scale to draw the "
                "calibration line for each box, then apply.")
            return False
        settings = self.settings_panel.get_settings()
        if settings.get("mode") in ("dlc", "sleap") \
                and not str(settings.get("model_path") or "").strip():
            QtWidgets.QMessageBox.warning(
                self, "Model Required",
                "Pose mode (DLC/SLEAP) is selected but no model path is set.\n\n"
                "Pick a model with Browse (or switch to Blob) before applying.")
            return False
        return True

    def should_apply(self) -> bool:
        """True when the host should fan this dialog's config out to the boxes
        (explicit Apply & Close)."""
        return self._apply_on_close

    def _on_box_changed(self, index):
        if index < 0:
            return

        new_box_id = self.box_combo.itemData(index)
        if new_box_id is None or new_box_id == self.current_box_id:
            return

        if self.current_box_id:
            self.zones_per_box[self.current_box_id] = self.zone_editor.get_zones()
            # Snapshot the outgoing box's scale calibration too -- the
            # editor's inline value+unit widgets are about to be reused
            # for the new box.
            self.scale_per_box[self.current_box_id] = self.zone_editor.get_scale_meta()

        self.current_box_id = new_box_id
        callback = self.get_frame_callbacks.get(new_box_id)
        self.zone_editor.setup_id = new_box_id
        self.zone_editor.set_frame_callback(callback)
        self.zone_editor.set_zones(self.zones_per_box.get(new_box_id, []))

    def _on_zones_changed(self):
        self._zones_dirty = True
        if self.current_box_id:
            current_zones = self.zone_editor.get_zones()
            self.zones_per_box[self.current_box_id] = current_zones
            # NOTE: zones are NOT auto-copied to other boxes here. Silent
            # per-edit transfer made drawing on Box 1 populate Box 2 (so
            # closing the editor looked like it "configured the next box").
            # Copying to all boxes is opt-in via the explicit "Transfer these
            # zones to all other boxes?" prompt in _load_box_zones.
            # Live PREVIEW: push this box's zones to the host so the camera
            # overlay repaints now. Nothing persists, the host rolls this
            # back on a non-apply close; "Save Zones to Project" or Apply &
            # Close are the commit paths.
            self.zones_live_changed.emit(self.current_box_id, current_zones)
        self.settings_panel.set_zone_names(self.zone_editor.get_zone_names())

    def _copy_zones_to_all_boxes(self):
        """Replace every other box's zones with the current box's.

        Overwrite, not merge: the operator asked for these boxes to match,
        and a merge would leave same-named zones sitting at their old
        geometry, which looks like the copy silently failed. Zones are
        normalised [0, 1], so they transfer across boxes with different ROI
        sizes unchanged.

        The scale calibration travels with them, zone validation gates on a
        scale being present, so copying zones alone would leave every target
        box failing validation.
        """
        if not self.current_box_id:
            return
        source_zones = self.zone_editor.get_zones()
        if not source_zones:
            QtWidgets.QMessageBox.information(
                self, "Copy zones",
                "Draw at least one zone on this box first.")
            return
        targets = [b for b in self.box_ids if b != self.current_box_id]
        if not targets:
            return
        if QtWidgets.QMessageBox.question(
                self, "Copy zones",
                f"Replace the zones on {len(targets)} other box(es) with "
                f"Box {self.current_box_id}'s {len(source_zones)} zone(s)?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        source_scale = self.zone_editor.get_scale_meta()
        for setup_id in targets:
            zones = copy.deepcopy(source_zones)
            self.zones_per_box[setup_id] = zones
            if source_scale:
                self.scale_per_box[setup_id] = copy.deepcopy(source_scale)
            # Same signal the editor fires on an edit, so each target's
            # overlay and the debounced autosave update now rather than
            # only on Apply & Close.
            self.zones_live_changed.emit(setup_id, zones)
        self._zones_dirty = True
        logger.info("Copied %d zone(s) from Box %s to %d box(es)",
                    len(source_zones), self.current_box_id, len(targets))

    def _auto_transfer_zones(self, source_zones: List[Dict]):
        if not source_zones:
            return

        for setup_id in self.box_ids:
            if setup_id == self.current_box_id:
                continue

            existing_zones = self.zones_per_box.get(setup_id, [])
            existing_names = {z.get("name") for z in existing_zones}

            import copy
            new_zones = [copy.deepcopy(z) for z in source_zones if z.get("name") not in existing_names]

            if new_zones:
                existing_zones.extend(new_zones)
                self.zones_per_box[setup_id] = existing_zones
                logger.info(f"Auto-transferred {len(new_zones)} zone(s) to Box {setup_id}")

    def _load_config(self):
        TRACKING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Tracking Config", str(TRACKING_CONFIGS_DIR), "JSON (*.json)"
        )

        if not path:
            return

        try:
            import json
            with open(path) as f:
                data = json.load(f)
            self._apply_config(data)
            QtWidgets.QMessageBox.information(self, "Loaded", f"Config loaded from:\n{path}")
            logger.info(f"Loaded tracking config from {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def _save_config(self):
        """Export the current tracking config to a standalone JSON snippet.

        This is Export, not the commit path: it writes a reusable fragment to
        a file the operator names. It does NOT apply to the boxes or the
        project; that is what Apply & Close does. Import (Load All Config)
        stages a snippet back into the dialog, where it commits with everything
        else.
        """
        import json
        from datetime import datetime

        TRACKING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")  # FILE_STEM_TS_FMT
        default_name = f"tracking_config_{timestamp}.json"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Tracking Config",
            str(TRACKING_CONFIGS_DIR / default_name), "JSON (*.json)"
        )

        if not path:
            return

        # Same calibration + model gate as the commit path, an exported
        # snippet with zones but no scale, or pose mode with no model, is a
        # trap for whoever imports it later.
        if not self._pre_commit_gate_ok():
            return
        tracking_settings = self.settings_panel.get_settings()

        try:
            if self.current_box_id:
                self.zones_per_box[self.current_box_id] = self.zone_editor.get_zones()
                self.scale_per_box[self.current_box_id] = self.zone_editor.get_scale_meta()

            config = {
                "zones": {str(k): v for k, v in self.zones_per_box.items()},
                "scale": {str(k): v for k, v in self.scale_per_box.items() if v},
                "tracking": tracking_settings,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(path, "w") as f:
                json.dump(config, f, indent=2)

            QtWidgets.QMessageBox.information(self, "Exported", f"Config exported to:\n{path}")
            logger.info(f"Exported tracking config to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export config: {e}")

    def _save_box_zones(self):
        """Export the current box's zones to a standalone JSON snippet.

        Explicit Export: the operator names the file. Zones persist to the
        project through Apply & Close; this only writes a reusable fragment.
        """
        import json
        from datetime import datetime

        if not self.current_box_id:
            return

        TRACKING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        default_name = f"box{self.current_box_id}_zones.json"
        default_path = str(TRACKING_CONFIGS_DIR / default_name)

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Box Zones", default_path, "JSON (*.json)"
        )
        if not path:
            return

        try:
            zones = self.zone_editor.get_zones()
            from source.datetime_formats import format_header_ts
            payload = {
                "version": "1.0",
                "box_id": self.current_box_id,
                "created": format_header_ts(datetime.now()),
                "zones": zones,
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)

            QtWidgets.QMessageBox.information(self, "Exported", f"Zones exported to:\n{path}")
            logger.info(f"Exported box {self.current_box_id} zones to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export: {e}")

    def closeEvent(self, event):
        """Route a window-X close through the same apply / discard choice as
        the buttons.

        The old prompt offered "Save", which exported a JSON *file*, while the
        host then discarded the in-memory config, silently losing the DLC
        model, mode, and scale calibration. Here the choice is honest:

          * Apply  → commit to the boxes + project (via ``_apply_and_close``),
          * Discard → close without applying; the host rolls its live preview
                      back to the open-time snapshot,
          * Cancel → stay in the dialog.

        The prompt fires only when the config actually changed since open
        (``_is_dirty`` compares the whole config, so a pose-model edit with no
        zone change still prompts; that was the silent-loss case).
        """
        # Explicit Apply already committed; don't ask again.
        if self._suppress_close_prompt:
            super().closeEvent(event)
            return
        if self._is_dirty():
            msg = ("You have unsaved tracking changes.\n\n"
                   "Apply them to the boxes (and save to the project), "
                   "or discard them?")
            if self._zones_dirty:
                # Say explicitly that zones are part of the discard, the
                # generic wording read as "tracker settings only", and
                # operators discarded freshly-drawn zones without knowing.
                msg = ("You have unsaved tracking changes, INCLUDING zone "
                       "edits.\n\nDiscard also discards the zones you drew "
                       "or moved. To keep the zones without applying "
                       "tracker settings, Cancel and press \"Save Zones to "
                       "Project\" first.\n\n"
                       "Apply everything, or discard everything?")
            reply = QtWidgets.QMessageBox.question(
                self, "Apply tracking configuration?",
                msg,
                QtWidgets.QMessageBox.StandardButton.Apply
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Apply,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Apply:
                # Runs the pre-commit gate; if it fails, stay open.
                if not self._pre_commit_gate_ok():
                    event.ignore()
                    return
                self._apply_on_close = True
                self._suppress_close_prompt = True
                event.accept()
                self.accept()
                return
            if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            # Discard: fall through to close without applying.
        super().closeEvent(event)

    def _save_zones_to_project(self):
        """Zones-only partial commit: persist every box's zones to the
        project immediately, independent of Apply & Close.

        Zones are standalone geometry the operator often finishes before
        (or without ever) picking a tracker, losing them to a later
        Discard was the main way drawn zones silently vanished. Tracker
        settings stay transactional; only the zones escape the rollback.
        """
        if self.current_box_id:
            self.zones_per_box[self.current_box_id] = self.zone_editor.get_zones()
        self.zones_save_requested.emit(
            {int(k): list(v or []) for k, v in self.zones_per_box.items()})
        self._zones_dirty = False

    def _load_box_zones(self):
        if not self.current_box_id:
            return

        TRACKING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load Zones for Box {self.current_box_id}",
            str(TRACKING_CONFIGS_DIR), "JSON (*.JSON *.json)"
        )

        if path:
            try:
                from source.video.zones.io import load_zone_config
                zones = load_zone_config(path)

                self.zone_editor.set_zones(zones)
                self.zones_per_box[self.current_box_id] = zones

                reply = QtWidgets.QMessageBox.question(
                    self, "Transfer Zones", "Transfer these zones to all other boxes?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
                )
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    self._auto_transfer_zones(zones)

                logger.info(f"Loaded zones for box {self.current_box_id} from {path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _apply_config(self, data: dict):
        import copy

        zones = data.get("zones", {})
        loaded_zones = None
        if isinstance(zones, dict):
            for k, v in zones.items():
                try:
                    setup_id = int(k)
                    self.zones_per_box[setup_id] = v
                    if setup_id == self.current_box_id:
                        self.zone_editor.set_zones(v)
                    if v and not loaded_zones:
                        loaded_zones = v
                except (ValueError, TypeError):
                    pass

        if loaded_zones:
            for setup_id in self.box_ids:
                if not self.zones_per_box.get(setup_id):
                    self.zones_per_box[setup_id] = copy.deepcopy(loaded_zones)

        # Restore per-box scale calibration. Same shape as get_full_config
        # writes: {"value": ..., "unit": ..., "pixel_length": ...}.
        scale_map = data.get("scale", {})
        if isinstance(scale_map, dict):
            for k, v in scale_map.items():
                try:
                    setup_id = int(k)
                except (ValueError, TypeError):
                    continue
                if isinstance(v, dict):
                    self.scale_per_box[setup_id] = v
            # Push the current box's calibration into the inline widgets
            # so editing reflects the loaded value immediately.
            cur = self.scale_per_box.get(self.current_box_id) or {}
            try:
                if cur.get("value") and self.zone_editor.scale_value_spin is not None:
                    self.zone_editor.scale_value_spin.blockSignals(True)
                    self.zone_editor.scale_value_spin.setValue(float(cur["value"]))
                    self.zone_editor.scale_value_spin.blockSignals(False)
                if cur.get("unit") and self.zone_editor.scale_unit_combo is not None:
                    i = self.zone_editor.scale_unit_combo.findText(cur["unit"])
                    if i >= 0:
                        self.zone_editor.scale_unit_combo.blockSignals(True)
                        self.zone_editor.scale_unit_combo.setCurrentIndex(i)
                        self.zone_editor.scale_unit_combo.blockSignals(False)
            except Exception:
                pass

        self.settings_panel.set_zone_names(self.zone_editor.get_zone_names())

        tracking = data.get("tracking", {})
        if tracking:
            self._apply_tracking_settings(tracking)

    def _apply_tracking_settings(self, settings: dict):
        """Replay a saved tracking config into the settings panel.

        Order matters: body parts are restored before the annotate checkboxes
        and zone combo that are built from them. The whole replay runs with
        ``_loading`` held, because setting the mode radio otherwise cascades
        into ``_load_body_parts_from_model``, which would refresh the picker
        with an empty list when the model file is not on disk, wiping the
        very selection being restored.
        """
        sp = self.settings_panel
        sp._loading = True
        failed: List[str] = []

        def step(what, fn, *args):
            """Run one restore group, and let the others run if it fails.

            The groups are independent, so a bad trigger list is no reason to
            leave the pose settings at their defaults, which is what a single
            try/except around the whole sequence did. ``logger.exception``
            rather than the message alone: without the traceback there is
            nothing to act on.
            """
            try:
                return fn(*args)
            except Exception:
                failed.append(what)
                logger.exception("Restoring %s from the saved config failed",
                                 what)
                return None

        try:
            step("general toggles", self._apply_toggles, sp, settings)
            mode = step("mode and model", self._apply_mode_and_model, sp, settings)
            step("blob settings", self._apply_blob_params, sp, settings)
            step("pose settings", self._apply_pose_params, sp, settings)
            step("SLEAP settings", self._apply_sleap_params, sp, settings)
            step("body parts", self._apply_body_parts, sp, settings)
            step("pose init state", self._apply_pose_init_state, sp, settings)
            step("zone body part", self._apply_zone_body_part, sp, settings)
            step("marker size", sp.marker_size_spin.setValue,
                 settings.get("marker_size", 4))
            step("box selection", self._apply_box_selection, sp, settings)
            n_triggers = step("triggers", self._apply_triggers, sp, settings)
            step("background images", self._load_background_images, sp)
            step("annotated body parts", self._apply_annotate_saved, sp, settings)
            # Single source of truth: the setter wipes existing rows and
            # rebuilds, so repeated replays (reopen, Load Config, Reinit)
            # cannot stack duplicate c.* rows on the _build_ui defaults.
            step("coordinate mapping", sp._set_coord_mapping,
                 settings.get("coord_mapping", {}) or {})
            logger.info("Applied tracking settings: mode=%s, %s triggers",
                        mode, n_triggers if n_triggers is not None else "?")
        finally:
            sp._loading = False
        if failed:
            self._warn_partial_restore(failed)
        return not failed

    def _warn_partial_restore(self, failed: List[str]) -> None:
        """Tell the operator that part of their saved config did not load.

        This has to be said out loud, not logged. A dialog showing defaults
        where the file held real values looks exactly like a dialog showing
        the file, and pressing Apply then writes those defaults back over the
        config, which is how a restore bug becomes data loss.
        """
        what = "\n".join(f"  \u2022 {name}" for name in failed)
        text = (
            "These settings could not be restored and are showing defaults:\n\n"
            f"{what}\n\n"
            "Do NOT press Apply unless you mean to overwrite them \u2014 Apply "
            "saves what the dialog currently shows.\n\n"
            "The log holds the full error for each.")
        #: Readable without a screen, so a headless or scripted run can
        #: still see that the restore was partial.
        self._restore_warning_text = text
        # Logged first and unconditionally: the popup is the operator's
        # copy, not the record.
        logger.warning("Tracking config only partly restored: %s",
                       ", ".join(failed))
        try:
            # ``show``, never ``exec`` or the static
            # ``QMessageBox.warning``: those are modal and block the
            # caller until someone clicks, and this runs during a config
            # replay, including in automated runs where nobody ever
            # will.
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setWindowTitle("Tracking config only partly loaded")
            box.setText(text)
            box.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            # Held so it is not garbage-collected on return.
            self._restore_warning_box = box
            box.show()
        except Exception as e:
            # Telling the operator must never be able to break the
            # restore it is reporting on.
            logger.debug("could not show the partial-restore warning: %s", e)

    # ── _apply_tracking_settings steps ────────────────────────────────
    # Each restores one group. An absent key leaves its widget untouched, so
    # a partial config never resets unrelated settings to defaults.

    @staticmethod
    def _apply_toggles(sp, settings):
        """Smooth-tracking plus the three MCU push gates."""
        for key, widget in (("smooth_tracking", sp.smooth_tracking_cb),
                            ("push_zones_to_mcu", sp.push_zones_to_mcu_cb),
                            ("push_coords_to_mcu", sp.push_coords_to_mcu_cb),
                            ("push_frame_event", sp.push_frame_event_cb)):
            if key in settings:
                widget.setChecked(bool(settings[key]))

    @staticmethod
    def _apply_mode_and_model(sp, settings) -> str:
        """Tracking mode radio + model path. Returns the mode, for the log."""
        mode = settings.get("mode", "normal")
        {"sleap": sp.mode_sleap, "dlc": sp.mode_dlc}.get(
            mode, sp.mode_blob).setChecked(True)
        # ``dlc_model_path`` is the legacy key.
        model_path = settings.get("model_path",
                                  settings.get("dlc_model_path", ""))
        if model_path:
            sp.model_path.setText(model_path)
        return mode

    @staticmethod
    def _set_combo_text(combo, value):
        """Select ``value`` if the combo offers it; otherwise leave it be,
        a stale saved option must not be forced into a fixed list.

        An EDITABLE combo is the exception: its list is a set of suggestions,
        not the permitted values. ``cuda:1`` is a legitimate device that the
        suggestions do not include, and refusing it would silently reset a
        multi-GPU box to ``auto`` every time its project loaded.
        """
        text = str(value)
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.isEditable() and text:
            combo.setCurrentText(text)

    def _apply_blob_params(self, sp, settings):
        for key, spin in (("threshold", sp.threshold_spin),
                          ("min_area", sp.min_area_spin),
                          ("max_area", sp.max_area_spin),
                          ("blur_kernel_size", sp.blur_size_spin),
                          ("open_kernel_size", sp.open_kernel_spin),
                          ("close_kernel_size", sp.close_kernel_spin),
                          ("self_norm_sigma", sp.sn_sigma_spin),
                          ("self_norm_smooth_sigma", sp.sn_smooth_spin),
                          ("self_norm_minsize", sp.sn_minsize_spin)):
            if key in settings:
                spin.setValue(settings[key])
        if "detect_dark" in settings:
            sp.detect_dark_cb.setChecked(settings["detect_dark"])
        for key, combo in (("blur_mode", sp.blur_mode_combo),
                           ("bg_mode", sp.bg_mode_combo)):
            if key in settings:
                self._set_combo_text(combo, settings[key])
        # Calibration-only params have no widget to restore into; they live on
        # the panel and are re-emitted by get_settings, so a saved config
        # keeps whatever the calibration dialog tuned.
        for key in sp._CALIBRATION_ONLY_KEYS:
            if key in settings:
                sp._blob_calibration_extra[key] = settings[key]
        # The Simple chip and bg_mode are one fact stored once. bg_mode is
        # the stored half, so the chip is reconciled to it after the combo
        # has been restored, never the other way round.
        if not (sp.mode_dlc.isChecked() or sp.mode_sleap.isChecked()):
            want_simple = sp.bg_mode_combo.currentText() == "self_norm"
            if want_simple != sp.mode_simple.isChecked():
                (sp.mode_simple if want_simple else sp.mode_blob).setChecked(True)
            sp._apply_simple_mode_ui(want_simple, False)
        sp._refresh_sn_ratio_label()

    @staticmethod
    def _apply_pose_params(sp, settings):
        if "dlc_confidence" in settings:
            sp.dlc_confidence_spin.setValue(settings["dlc_confidence"])
        if "dlc_resize" in settings:
            sp.dlc_resize_spin.setValue(settings["dlc_resize"])
        if "pose_instances" in settings:
            try:
                sp.pose_instances_spin.setValue(int(settings["pose_instances"]))
            except Exception:
                pass

    def _apply_sleap_params(self, sp, settings):
        """SLEAP-only options, each guarded, the widgets only exist when the
        SLEAP backend is present."""
        if "sleap_centroid_path" in settings and hasattr(sp, "sleap_centroid_path"):
            sp.sleap_centroid_path.setText(settings.get("sleap_centroid_path") or "")
        # The engine list is rebuilt from whichever model is selected, and the
        # model path is restored elsewhere, so the saved engine can arrive
        # before the list that would contain it. Offering it here is what makes
        # the round trip total regardless of which lands first.
        if "dlc_model_type" in settings and hasattr(sp, "dlc_model_type_combo"):
            engine = str(settings["dlc_model_type"] or "auto")
            if engine and sp.dlc_model_type_combo.findText(engine) < 0:
                sp.dlc_model_type_combo.addItem(engine)
        for key, combo_attr in (("sleap_runtime", "sleap_runtime_combo"),
                                ("sleap_device", "sleap_device_combo"),
                                ("dlc_model_type", "dlc_model_type_combo"),
                                ("dlc_precision", "dlc_precision_combo"),
                                ("dlc_device", "dlc_device_combo"),
                                ("pose_colour_mode", "pose_colour_combo")):
            if key in settings and hasattr(sp, combo_attr):
                self._set_combo_text(getattr(sp, combo_attr), settings[key])
        if "pose_input_mode" in settings and hasattr(sp, "pose_input_mode_combo"):
            # The stored value drives; the combo is reconciled TO it. Never the
            # other way, and never both, two directions is a loop, not a
            # wrong value.
            label = sp._INPUT_MODE_TO_LABEL.get(
                str(settings.get("pose_input_mode") or "auto"), "auto")
            # The list is backend-specific and the backend chip may not have
            # been restored yet, so the stored mode is offered before it is
            # selected, otherwise setCurrentText on a fixed combo that does
            # not hold it is a silent no-op and the project loses the setting.
            if sp.pose_input_mode_combo.findText(label) < 0:
                sp.pose_input_mode_combo.addItem(label)
            sp.pose_input_mode_combo.setCurrentText(label)
        for _key, _widget, _cast in (
                ("pose_input_w", "pose_input_w_spin", int),
                ("pose_input_h", "pose_input_h_spin", int),
                ("pose_crop_conf_min", "pose_crop_conf_spin", float),
                ("pose_crop_good_min", "pose_crop_good_spin", int),
                ("dlc_dynamic_threshold", "dlc_dynamic_threshold_spin", float),
                ("dlc_dynamic_margin", "dlc_dynamic_margin_spin", int)):
            if _key in settings and hasattr(sp, _widget):
                try:
                    getattr(sp, _widget).setValue(_cast(settings[_key]))
                except (TypeError, ValueError):
                    pass
        if "pose_crop_reacquire" in settings and hasattr(sp, "pose_crop_reacquire_cb"):
            sp.pose_crop_reacquire_cb.setChecked(bool(settings["pose_crop_reacquire"]))
        if hasattr(sp, "_sync_pose_input_row"):
            sp._sync_pose_input_row()
            sp._validate_pose_window()
        if "sleap_peak_threshold" in settings:
            # No widget for this one: it only matters for bottom-up models,
            # where it is baked into the engine. Held so Export builds the same
            # engine Init will ask for.
            try:
                sp._sleap_peak_threshold = float(settings["sleap_peak_threshold"])
            except (TypeError, ValueError):
                pass
        if "sleap_fp16" in settings and hasattr(sp, "sleap_fp16_cb"):
            sp.sleap_fp16_cb.setChecked(bool(settings["sleap_fp16"]))

    @staticmethod
    def _apply_body_parts(sp, settings):
        """Restore the keypoint names, then everything built from them.

        Runs before the zone combo and annotate checkboxes because both are
        populated from this list.
        """
        body_parts = settings.get("body_parts", [])
        if body_parts:
            sp.body_parts_list = body_parts
            sp.body_parts_label.setText(f"Body parts: {', '.join(body_parts)}")
            sp.body_parts_label.setStyleSheet("color: #4CAF50;")
            sp._update_annotate_checkboxes(body_parts)
            sp._update_zone_body_part_combo(body_parts)
            sp._update_coord_combos()
        # After the parts: an edge naming a part this model does not have
        # cannot be drawn, so it is dropped rather than kept as a silent no-op.
        known = set(sp.body_parts_list or [])
        sp.skeleton_edges = [
            (str(a), str(b)) for a, b in
            (e for e in settings.get("skeleton", []) if len(e) == 2)
            if str(a) in known and str(b) in known]
        sp._update_skeleton_table()
        annotate_parts = settings.get("annotate_parts", [])
        if annotate_parts:
            for name, cb in sp.annotate_checkboxes.items():
                cb.setChecked(name in annotate_parts)

    @staticmethod
    def _apply_pose_init_state(sp, settings):
        """The host tells us whether the live PoseSink already holds this
        exact signature, so Init comes up disabled on reopen."""
        if bool(settings.get("dlc_initialized", False)):
            sp._mark_dlc_clean("Ready (already loaded)")
        else:
            sp._dlc_initialized = False
            sp.dlc_init_btn.setEnabled(True)

    @staticmethod
    def _apply_zone_body_part(sp, settings):
        """``zone_change_body_part`` wins; ``zone_body_part`` is the fallback.

        In pose mode the synthetic "centroid" is no longer offered; it reads
        as a rival to the named keypoints, so a saved one is dropped and the
        picker's auto-selected centre keypoint stands instead.
        """
        zone_bp = (settings.get("zone_change_body_part")
                   or settings.get("zone_body_part") or "")
        is_pose = sp.mode_dlc.isChecked() or sp.mode_sleap.isChecked()
        if is_pose and str(zone_bp).strip().lower() == "centroid":
            zone_bp = ""
        if not zone_bp:
            return
        combo = sp.zone_body_part_combo
        if combo.findText(zone_bp) == -1:
            combo.addItem(zone_bp)
        combo.setCurrentText(zone_bp)

    @staticmethod
    def _apply_box_selection(sp, settings):
        enabled = settings.get("enabled_boxes", []) or []
        for setup_id, cb in sp.box_checkboxes.items():
            cb.setChecked(setup_id in enabled)

    def _apply_triggers(self, sp, settings) -> int:
        """Rebuild the trigger table from scratch. Returns the row count.

        Cleared first, so reopening the dialog cannot append a second copy of
        every rule.
        """
        while sp.triggers_table.rowCount() > 0:
            sp.triggers_table.removeRow(0)
        triggers = settings.get("triggers", []) or []
        for trigger in triggers:
            sp._add_trigger()
            self._fill_trigger_row(sp.triggers_table,
                                   sp.triggers_table.rowCount() - 1, trigger)
        return len(triggers)

    def _fill_trigger_row(self, table, row, trigger):
        """One rule → one table row. Columns are fixed by ``_add_trigger``."""
        for col, key, default in ((0, "condition", "in_zone"),
                                  (1, "body_part", "center")):
            widget = table.cellWidget(row, col)
            if widget is not None:
                self._set_combo_text(widget, trigger.get(key, default))

        zone_btn = table.cellWidget(row, 2)
        if zone_btn is not None and hasattr(zone_btn, "set_zones"):
            zone_btn.set_zones(trigger.get("zones", []))

        event = table.cellWidget(row, 3)
        if event is not None:
            event.setText(trigger.get("event_name", ""))

        threshold_w = table.cellWidget(row, 4)
        if threshold_w is not None and "threshold" in trigger:
            threshold_w.setValue(trigger["threshold"])

        # Part B, 2nd body part (distance) / neck (head-angle). The
        # condition set above has already enabled the combo.
        part_b = table.cellWidget(row, 5)
        if part_b is not None:
            pb = trigger.get("part_b") or trigger.get("neck")
            if pb:
                self._set_combo_text(part_b, pb)

        hold_w = table.cellWidget(row, 6)
        if hold_w is not None and "duration_ms" in trigger:
            hold_w.setValue(int(trigger["duration_ms"]))

        colour_btn = table.cellWidget(row, 7)
        if colour_btn is not None and trigger.get("color"):
            colour_btn._hex = trigger["color"]
            colour_btn.setStyleSheet(
                f"background:{colour_btn._hex};border:1px solid #2b3a4d;"
                "border-radius:3px;")

        for col, key in ((8, "show_on_video"), (9, "plot")):
            cb = table.cellWidget(row, col)
            if cb is not None:
                cb.setChecked(bool(trigger.get(key, True)))

    def _load_background_images(self, sp):
        """Backgrounds live on disk at ``<project>/background_images/boxN.png``.
        Scan the folder so the panel's status reflects what is actually there,
        rather than what a stale config claimed."""
        mw = self.window()
        bg_dir = (mw._background_images_dir()
                  if (mw is not None and hasattr(mw, "_background_images_dir"))
                  else None)
        if bg_dir is not None:
            for bg_file in bg_dir.glob("box*.png"):
                try:
                    setup_id = int(bg_file.stem[3:])
                except (ValueError, TypeError):
                    continue
                sp.backgrounds[setup_id] = str(bg_file)
        sp._update_bg_status()

    @staticmethod
    def _apply_annotate_saved(sp, settings):
        """Per-box "save annotated video" flags. Keys arrive as strings from
        JSON, so each is coerced back to an int box id."""
        for box_id_str, flag in (settings.get("annotate_saved", {}) or {}).items():
            try:
                setup_id = int(box_id_str)
            except (ValueError, TypeError):
                continue
            cb = sp.box_annotate_checkboxes.get(setup_id)
            if cb is not None:
                cb.setChecked(bool(flag))

    # Public API
    def get_zones(self) -> List[Dict]:
        return self.zone_editor.get_zones()

    def get_all_zones(self) -> Dict[int, List[Dict]]:
        if self.current_box_id:
            self.zones_per_box[self.current_box_id] = self.zone_editor.get_zones()
        return self.zones_per_box.copy()

    def get_settings(self) -> dict:
        return self.settings_panel.get_settings()

    def get_full_config(self) -> dict:
        if self.current_box_id:
            self.zones_per_box[self.current_box_id] = self.zone_editor.get_zones()
            self.scale_per_box[self.current_box_id] = self.zone_editor.get_scale_meta()
        return {
            "zones": {str(k): v for k, v in self.zones_per_box.items()},
            "scale": {str(k): v for k, v in self.scale_per_box.items() if v},
            "tracking": self.settings_panel.get_settings(),
            "coord_mapping": self.settings_panel.get_coord_mapping(),
        }

    def _missing_scale_boxes(self) -> list[int]:
        """Return box_ids that have zones drawn but no valid scale set.
        Used by the save-zones gate so the user can't ship un-calibrated
        zone data downstream to analysis."""
        # Sync the live editor first.
        if self.current_box_id:
            self.zones_per_box[self.current_box_id] = self.zone_editor.get_zones()
            self.scale_per_box[self.current_box_id] = self.zone_editor.get_scale_meta()
        missing = []
        for bid, zones in self.zones_per_box.items():
            non_scale_zones = [z for z in (zones or [])
                               if isinstance(z, dict) and z.get("type") != "scale"]
            if not non_scale_zones:
                continue
            meta = self.scale_per_box.get(bid) or {}
            try:
                value = float(meta.get("value", 0) or 0)
                pxlen = float(meta.get("pixel_length", 0) or 0)
            except (TypeError, ValueError):
                value = pxlen = 0.0
            if value <= 0 or pxlen <= 0:
                missing.append(bid)
        return missing

