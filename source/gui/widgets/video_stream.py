"""Per-box video tile for the operant Video Stream tab."""

from PySide6 import QtCore, QtGui, QtWidgets

from source.log import get_logger
from source.gui.widgets.video_tile import VideoTile

logger = get_logger()


class VideoStreamHolder(QtWidgets.QFrame):
    """Per-box video tile for the operant Video Stream tab.

    Same zone-adjust button row as maze SetupWidget, main_window connects
    the signals to MainWindowBase._shift_zones / _rotate_zones / _scale_zones.
    Buttons are disabled until tracking_zones[box_id] is populated; call
    set_zones_enabled(True/False) when zones are loaded or cleared.
    """

    # Per-box zone-adjust signals (each carries box_id for the slot)
    zone_left = QtCore.Signal(int)
    zone_right = QtCore.Signal(int)
    zone_up = QtCore.Signal(int)
    zone_down = QtCore.Signal(int)
    zone_rotate_cw = QtCore.Signal(int)
    zone_rotate_ccw = QtCore.Signal(int)
    zone_zoom_in = QtCore.Signal(int)
    zone_zoom_out = QtCore.Signal(int)
    zone_home = QtCore.Signal(int)  # Reset to pre-adjustment baseline
    marker_size_changed = QtCore.Signal(int, int)  # box_id, radius px

    def __init__(self, setup_number, parent=None):
        super().__init__(parent)
        self.setup_number = setup_number
        # Tracks whether the running-clock colour is already applied, so the
        # per-second clock update doesn't re-run setStyleSheet every tick.
        self._timer_running_style = False
        # Grow with the surrounding grid cell so the live video tile
        # tracks the user resizing the window (Expanding, not the QFrame
        # default Preferred which would freeze it at the sizeHint).
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._build_ui()
        logger.debug(f"Initialized VideoStreamHolder for box {setup_number}")

    def set_error(self, on, msg=""):
        """Red tile border + ⚠ header + tooltip when this box errors.
        Driven centrally by box_alerts.apply_box_alerts; auto-clears."""
        on = bool(on)
        from source.gui.box_alerts import ERROR_COLOR, error_text, error_tooltip
        from source.gui.theme import THEME as _T
        lbl = getattr(self, "box_label", None)
        if lbl is not None:
            lbl.setText(error_text("Box", self.setup_number, on))
            lbl.setStyleSheet(
                "color: {}; font-weight: bold; font-size: 9pt;".format(
                    ERROR_COLOR if on else "#e0e0e0"))
            lbl.setToolTip(error_tooltip(
                "Box {}".format(self.setup_number), on, msg))
        vw = getattr(self, "videoWidget", None)
        if vw is not None:
            border = ERROR_COLOR if on else _T.palette.surface_border_strong
            vw.setStyleSheet(
                "QWidget#VideoFeed {"
                f" border: {2 if on else 1}px solid {border};"
                f" background-color: #000;"
                f" color: {_T.palette.text_muted};"
                f" border-radius: {_T.radius.md}px;"
                " font-size: 9pt;"
                "}")

    def set_camera_pending(self, on):
        """Configured-but-not-connected indicator: show a red 'Camera not
        connected' prompt in place of the muted 'No Signal'. Harmless while
        live video is showing (the placeholder is only painted when there's
        no frame); cleared once the camera streams or the box has no camera."""
        on = bool(on)
        from source.gui.theme import THEME as _T
        vw = getattr(self, "videoWidget", None)
        if vw is None:
            return
        color = "#ff6b6b" if on else _T.palette.text_muted
        vw.setStyleSheet(
            "QWidget#VideoFeed {"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            " background-color: #000;"
            f" color: {color};"
            f" border-radius: {_T.radius.md}px;"
            " font-size: 9pt;"
            "}")
        vw.setText("Camera not connected" if on else "No Signal")

    def _build_ui(self):
        """Setup the video stream UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(1)
        layout.setContentsMargins(5, 5, 5, 5)

        self.header = QtWidgets.QWidget()
        self.header.setFixedHeight(25)
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(5, 0, 5, 0)
        header_layout.setSpacing(5)

        self.box_label = QtWidgets.QLabel(f"Box {self.setup_number}")
        self.box_label.setStyleSheet("""
            color: #e0e0e0;
            font-weight: bold;
            font-size: 9pt;
        """)

        from source.gui.theme import THEME as _T_vs
        # Subject ID (shown only when assigned), cyan, next to the box
        # label. Fed by operant._refresh_box_id_labels.
        self.subject_label = QtWidgets.QLabel("")
        self.subject_label.setStyleSheet(
            "color: #8be9fd; font-weight: 600; font-size: 9pt;")
        self.subject_label.setVisible(False)

        # Per-box elapsed session clock (HH:MM:SS), fed by the central tick,
        # same value as the box card + Live Status + stats table.
        self.timer_label = QtWidgets.QLabel("00:00:00")
        self.timer_label.setFont(QtGui.QFont("Cascadia Mono", 9, QtGui.QFont.Weight.Bold))
        self.timer_label.setStyleSheet("color: #9aa0b4;")  # muted until running

        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setStyleSheet(
            f"color: {_T_vs.palette.text_muted}; font-size: 8pt;")

        self.timestamp_label = QtWidgets.QLabel("")
        self.timestamp_label.setStyleSheet(
            f"color: {_T_vs.palette.text_muted}; font-size: 8pt;")

        header_layout.addWidget(self.box_label)
        header_layout.addWidget(self.subject_label)
        header_layout.addWidget(self.timer_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        header_layout.addWidget(self.timestamp_label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        self.header.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed
        )

        # Tile fills the cell. This holder is built by operant mode only (the
        # Video Stream grid), and a grid of sixteen chambers is read by
        # glancing across it, so the height goes to the animals rather than to
        # letterbox bars. Maze arenas preserve aspect instead, because their
        # picture is read against drawn zones. The wrappers (setPixmap /
        # setText / setAlignment) keep existing call sites working unchanged.
        from source.gui.theme import THEME as _T
        self.videoWidget = VideoTile(aspect_mode=VideoTile.ASPECT_STRETCH)
        self.videoWidget.setObjectName("VideoFeed")
        self.videoWidget.setStyleSheet(
            "QWidget#VideoFeed {"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            f" background-color: #000;"
            f" color: {_T.palette.text_muted};"
            f" border-radius: {_T.radius.md}px;"
            " font-size: 9pt;"
            "}"
        )
        # Minimum comes from VideoTile (96×72), don't re-pin it larger
        # here or the detached grid window can't shrink.
        self.videoWidget.setText("No Signal")
        self.videoWidget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        layout.addWidget(self.header)
        layout.addWidget(self.videoWidget, stretch=1)

        # Zone-adjust row, the shared toolbar, placed directly below
        # this box's video tile (operant sizing: wider buttons, 9 pt).
        from source.gui.widgets.zone_adjust_row import ZoneAdjustRow
        self.zone_adjust_row = ZoneAdjustRow(
            self.setup_number, button_w=26, button_h=22,
            edit_w=46, font_pt=9)
        self.zone_adjust_row.forward_signals_to(self)
        zone_row = QtWidgets.QHBoxLayout()
        zone_row.setContentsMargins(2, 2, 2, 2)
        zone_row.addWidget(self.zone_adjust_row)
        zone_row.addStretch()
        layout.addLayout(zone_row)

        self.setStyleSheet("""
            QFrame {
                border: 1px solid #333;
                border-radius: 4px;
                background-color: black;
            }
        """)

    def set_zones_enabled(self, enabled: bool) -> None:
        """Enable/disable the zone-adjust row based on whether zones exist."""
        self.zone_adjust_row.set_zones_enabled(enabled)

    def zone_step_value(self) -> float:
        return self.zone_adjust_row.zone_step_value()

    def updateStatus(self, text):
        """Update status text."""
        self.status_label.setText(text)

    def set_subject(self, subject):
        """Show the box's Subject ID in the tile header (hidden when none)."""
        s = (subject or "").strip()
        try:
            self.subject_label.setText(s)
            self.subject_label.setVisible(bool(s))
        except RuntimeError:
            pass

    def set_timer(self, text):
        """Set the elapsed-session clock (fig-green while ticking). Called by
        the central process tick for running boxes; freezes at the final
        value when the box stops."""
        try:
            if self.timer_label.text() != text:
                self.timer_label.setText(text)
                # Apply the running colour once, not on every second flip,
                # setStyleSheet forces a Qt style recompute, costly summed
                # over every box every second on the shared 10 ms tick.
                if not self._timer_running_style:
                    # Match the Live Status widget timer (theme palette.focus).
                    self.timer_label.setStyleSheet("color: #a855f7;")
                    self._timer_running_style = True
        except RuntimeError:
            pass

    # No resizeEvent override, VideoTile repaints at the new size
    # automatically, with letterbox aspect fit handled in paintEvent.


