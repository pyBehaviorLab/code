"""The zone-adjust toolbar, ONE widget for both per-box surfaces.

Nine buttons (home / four shifts / two rotates / two zooms) plus the
dual-purpose step edit (N pixels for shift, N degrees for rotate, ±N
percent for scale). One row, shared: maze embeds it in the SetupWidget's
bottom bar, operant below each video tile.

Signals carry the host's ``setup_id`` so the shared
``MainWindowBase._wire_zone_buttons`` handlers know which box to act
on. Buttons start disabled; the main window enables them (via the
host's ``set_zones_enabled``) when zones are loaded.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from source.gui.theme import THEME as _T


class ZoneAdjustRow(QtWidgets.QWidget):
    """Step edit + the nine zone-adjust buttons, emitting per-box signals."""

    zone_left = QtCore.Signal(int)
    zone_right = QtCore.Signal(int)
    zone_up = QtCore.Signal(int)
    zone_down = QtCore.Signal(int)
    zone_rotate_cw = QtCore.Signal(int)
    zone_rotate_ccw = QtCore.Signal(int)
    zone_zoom_in = QtCore.Signal(int)
    zone_zoom_out = QtCore.Signal(int)
    zone_home = QtCore.Signal(int)  # Reset to pre-adjustment baseline
    #: (setup_id, radius_px). Annotation only, so it is NOT a zone signal and
    #: its buttons stay live when a box has no zones and no model.
    marker_size_changed = QtCore.Signal(int, int)

    SIGNAL_NAMES = ("zone_left", "zone_right", "zone_up", "zone_down",
                    "zone_rotate_cw", "zone_rotate_ccw",
                    "zone_zoom_in", "zone_zoom_out", "zone_home",
                    "marker_size_changed")

    MARKER_MIN = 1
    MARKER_MAX = 20
    MARKER_DEFAULT = 4

    # symbol, tooltip, signal, semantic colour.
    # home -> success (mint, "return to known good state")
    # shifts -> secondary (slate) | rotate -> primary (purple)
    # zoom -> info (blue)
    _BUTTON_TABLE = (
        ("⌂", "Reset to baseline", "zone_home",       "success"),
        ("◀", "Shift left",        "zone_left",       "secondary"),
        ("▶", "Shift right",       "zone_right",      "secondary"),
        ("▲", "Shift up",          "zone_up",         "secondary"),
        ("▼", "Shift down",        "zone_down",       "secondary"),
        ("↻", "Rotate CW",         "zone_rotate_cw",  "primary"),
        ("↺", "Rotate CCW",        "zone_rotate_ccw", "primary"),
        ("+", "Zoom in",           "zone_zoom_in",    "info"),
        ("−", "Zoom out",          "zone_zoom_out",   "info"),
    )

    def __init__(self, setup_id: int, parent=None, *,
                 button_w: int = 22, button_h: int = 22,
                 edit_w: int = 44, font_pt: int = 8,
                 spacing: int = 2):
        super().__init__(parent)
        # All zone-adjust buttons use the LIGHT variant, glassy
        # translucent tint + thin coloured border (calmer than the vivid
        # gradient in this 9-button compact toolbar). light_button_style
        # delegates to button_style for "secondary" / "ghost", so the
        # shift quad keeps its slate look.
        from source.gui.style_builders import light_button_style as _btn_light

        row = QtWidgets.QHBoxLayout(self)
        row.setSpacing(spacing)
        row.setContentsMargins(0, 0, 0, 0)

        # Step-size text edit. Dual-purpose: N pixels for shift, N degrees
        # for rotate, ±N percent for scale. Validator allows positive
        # floats (no cap, 0.5 for fine rotation, 12 for coarse).
        self.zone_step_edit = QtWidgets.QLineEdit("1")
        _v = QtGui.QDoubleValidator(0.001, 100000.0, 3, self.zone_step_edit)
        _v.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        self.zone_step_edit.setValidator(_v)
        self.zone_step_edit.setFixedSize(edit_w, button_h)
        self.zone_step_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.zone_step_edit.setToolTip(
            "Zone-adjust step:\n"
            "  shift  = N pixels\n"
            "  rotate = N degrees\n"
            "  scale  = ±N percent"
        )
        self.zone_step_edit.setStyleSheet(
            "QLineEdit {"
            " background-color: rgba(255,255,255,0.04);"
            f" color: {_T.palette.text};"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            f" border-radius: {_T.radius.sm}px; font-size: {font_pt}pt;"
            " padding: 0 2px;"
            "}"
            f" QLineEdit:disabled {{ color: {_T.palette.text_dim};"
            " background-color: rgba(255,255,255,0.02); }"
        )
        row.addWidget(self.zone_step_edit)

        self._buttons = []
        for symbol, tip, signal_name, color_key in self._BUTTON_TABLE:
            b = QtWidgets.QPushButton(symbol)
            b.setFixedSize(button_w, button_h)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(_btn_light(color_key, height=button_h, padding_h=0))
            b.setToolTip(tip)
            b.setEnabled(False)  # main window enables when zones load
            sig = getattr(self, signal_name)
            b.clicked.connect(
                lambda _checked=False, s=sig, _sid=setup_id: s.emit(_sid))
            self._buttons.append(b)
            row.addWidget(b)

        # ---- annotation marker size -------------------------------------
        # Deliberately outside ``self._buttons``: marker size is how big the
        # keypoint dots are drawn, nothing to do with zones, a model or
        # inference, so it must stay usable on a box that has none of them.
        # It was previously only reachable from the tracking dialog.
        self._setup_id = int(setup_id)
        self._marker_size = self.MARKER_DEFAULT

        sep = QtWidgets.QLabel("│")
        sep.setStyleSheet(f"color: {_T.palette.surface_border_strong};"
                          f" font-size: {font_pt}pt;")
        row.addWidget(sep)

        self.marker_smaller_btn = QtWidgets.QPushButton("·")
        self.marker_bigger_btn = QtWidgets.QPushButton("●")
        self.marker_size_label = QtWidgets.QLabel(str(self._marker_size))
        self.marker_size_label.setFixedSize(edit_w - 16, button_h)
        self.marker_size_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter)
        self.marker_size_label.setStyleSheet(
            f"QLabel {{ color: {_T.palette.text}; font-size: {font_pt}pt; }}")

        for btn, delta, tip in (
                (self.marker_smaller_btn, -1, "Smaller annotation markers"),
                (self.marker_bigger_btn, +1, "Larger annotation markers")):
            btn.setFixedSize(button_w, button_h)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(_btn_light("ghost", height=button_h,
                                         padding_h=0))
            btn.setToolTip(f"{tip} (radius in pixels).\n"
                           "Annotation only: independent of the model.")
            btn.clicked.connect(
                lambda _checked=False, d=delta: self._bump_marker(d))

        row.addWidget(self.marker_smaller_btn)
        row.addWidget(self.marker_size_label)
        row.addWidget(self.marker_bigger_btn)

    # ---- marker size ----------------------------------------------------

    def _bump_marker(self, delta: int) -> None:
        self.set_marker_size(self._marker_size + int(delta), emit=True)

    def set_marker_size(self, size, *, emit: bool = False) -> None:
        """Set the shown radius, clamped. ``emit`` False when restoring from
        a project, so loading a config does not look like an operator edit
        and does not mark the project dirty."""
        try:
            size = round(float(size))
        except (TypeError, ValueError):
            size = self.MARKER_DEFAULT
        size = max(self.MARKER_MIN, min(self.MARKER_MAX, size))
        changed = size != self._marker_size
        self._marker_size = size
        self.marker_size_label.setText(str(size))
        self.marker_smaller_btn.setEnabled(size > self.MARKER_MIN)
        self.marker_bigger_btn.setEnabled(size < self.MARKER_MAX)
        if emit and changed:
            self.marker_size_changed.emit(self._setup_id, size)

    def marker_size(self) -> int:
        return self._marker_size

    def forward_signals_to(self, host) -> None:
        """Chain every row signal into the same-named Signal on ``host``,
        the per-box widgets keep their own signal surface so the shared
        ``_wire_zone_buttons`` sees one protocol."""
        for name in self.SIGNAL_NAMES:
            getattr(self, name).connect(getattr(host, name))

    def set_zones_enabled(self, enabled: bool) -> None:
        """Toggle the nine buttons + step edit. Driven by the main window
        based on whether this box has tracking zones configured."""
        for b in self._buttons:
            b.setEnabled(bool(enabled))
        self.zone_step_edit.setEnabled(bool(enabled))

    def zone_step_value(self) -> float:
        """The parsed zone-adjust step (positive float). Default 1.0 when
        the edit is empty or holds a non-numeric value."""
        txt = (self.zone_step_edit.text() or "").strip().replace(",", ".")
        try:
            v = float(txt)
            return v if v > 0 else 1.0
        except (TypeError, ValueError):
            return 1.0
