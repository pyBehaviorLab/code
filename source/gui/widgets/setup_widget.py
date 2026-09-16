"""Per-setup (maze arena) tab widget."""

from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from datetime import datetime

from source.gui.styles import (
    COLORS, BUTTON_STYLE, TIMER_LABEL_STYLE,
)
from source.gui.utility import NestedMenu, TaskInfo
from source.gui.widgets.common import make_button, make_lineedit
from source.gui.widgets.run_task import RunTask, RunMode
from source.log import get_logger
from source.gui.widgets.video_tile import AspectBox, VideoTile

logger = get_logger()


class SetupWidget(QtWidgets.QWidget, RunTask):
    """Per-setup tab widget.

    Row 1 (tall): Maze N | Subject ID | Task (expands) | Upload | Start | Stop | Timer
    Row 2: Video tile (left) | Live Status panel (right: State / Cam / COM / Board FPS / log)
    Row 3: [Experiment Control: Pause Next Prev Doors Controls] | [Adjust Zone] | Detach

    Enable/disable contract:
    - Not connected: everything disabled except Controls
    - Connected: subject_id enabled
    - Task uploaded: Record enabled
    - Running: Stop/Pause/Next/Prev enabled, Start disabled
    - Stopped: back to uploaded state
    """

    WIDGET_HEIGHT = 25
    # The top action row is taller than the 25px card rows, the operator
    # drives Upload / Start / Stop from here, so the buttons get more height.
    INFO_BAR_HEIGHT = 34

    # Signals
    start_clicked = QtCore.Signal(int)
    stop_clicked = QtCore.Signal(int)
    pause_clicked = QtCore.Signal(int)
    next_clicked = QtCore.Signal(int)
    prev_clicked = QtCore.Signal(int)
    metadata_clicked = QtCore.Signal(int)
    doors_clicked = QtCore.Signal(int)
    controls_clicked = QtCore.Signal(int)
    detach_clicked = QtCore.Signal(int)
    tab_title_changed = QtCore.Signal(int, str)  # box_id, new_title
    # Zone adjustment signals
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
    # Emitted by RunTask.mcu_start (every successful framework start) and
    # RunTask.mcu_stop (every exit path: user click / auto-stop on b'\x04'
    # / PyboardError). Main_window listens to the stopped signal for
    # tracking + recording teardown so one slot covers all paths.
    framework_started_signal = QtCore.Signal(int)
    framework_stopped_signal = QtCore.Signal(int)

    def __init__(self, setup_id, parent=None, main_window=None):
        super().__init__(parent)
        # Shared per-box state (pycboard, framework_running, task_uploaded,
        # global_*_locked, _subject_metadata, box_number, box_id).
        self.init_run_task(setup_id, main_window=main_window)
        # Tab-icon state. ``framework_running`` and ``run_mode`` are owned
        # by RunTask; ``is_recording`` is a read-only view of
        # ``run_mode == RECORD`` (see property below). Only ``is_paused`` is
        # maze-local, flipped by set_paused / _after_status_change.
        self.is_paused = False
        self._hw_def_path = None
        self._controls_dialog = None
        # Firmware time (ms) the current state was entered; drives the
        # per-state duration read-out in the Live Status panel.
        self._state_entry_fw_ms = None

        # ROI attributes (set by camera config dialog).  ``roi_normalized``
        # is the canonical, resolution-independent storage.  ``roi_segment``
        # is a transient pixel hint; do not persist it.
        self.roi_segment = None          # (x, y, w, h) in pixels (transient)
        self.roi_normalized = None       # (x, y, w, h) in 0-1 range (canonical)
        self.save_video_enabled = True

        # Per-tab accent color (cycled, for visual distinction)
        _TAB_COLORS = [
            '#4CAF50', '#2196F3', '#FF9800', '#E91E63',
            '#9C27B0', '#00BCD4', '#FF5722', '#8BC34A',
        ]
        self.tab_color = _TAB_COLORS[(setup_id - 1) % len(_TAB_COLORS)]

        self._build_ui()
        self.setupConnections()
        self._update_button_states()
        # Populate task dropdown immediately (operant does this in main_window
        # when adding a box; maze creates SetupWidgets directly, so populate
        # here so the dropdown isn't empty before the user connects MCU).
        self._refresh_task_menu()

    # ==================================================================
    # UI CONSTRUCTION
    # ------------------------------------------------------------------
    # Each row is built by one private ``_build_*`` method so layout
    # problems can be debugged by reading one named builder at a time.
    # ==================================================================

    def _build_ui(self):
        """Compose the three rows: info bar / middle / bottom bar."""
        self._icon_dir = Path(__file__).parent / 'icons'
        H = self.WIDGET_HEIGHT

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        layout.addLayout(self._build_info_bar(H))
        layout.addLayout(self._build_middle(H), stretch=1)
        layout.addLayout(self._build_bottom_bar(H))

    # ----- row 1: info bar -----

    def set_error(self, on, msg=""):
        """Red 'Maze N' label + tooltip on error. Driven centrally by
        box_alerts.apply_box_alerts; auto-clears on recovery."""
        from source.gui.box_alerts import (
            ERROR_COLOR, error_tooltip, identity_button_style)
        on = bool(on)
        lbl = getattr(self, "id_button", None)
        if lbl is not None:
            lbl.setStyleSheet(identity_button_style(
                ERROR_COLOR if on else "#e0e0e0"))
            lbl.setToolTip(error_tooltip(
                "Maze {}".format(self.setup_id), on, msg,
                idle="Click to edit subject metadata for this setup"))

    def set_camera_pending(self, on):
        """Configured-but-not-connected indicator: show a red 'Camera not
        connected' prompt in place of the muted 'No Video'. Harmless while
        live video is showing (placeholder only paints with no frame);
        cleared once the camera streams or the setup has no camera."""
        on = bool(on)
        from source.gui.theme import THEME as _T
        vw = getattr(self, "video_label", None)
        if vw is None:
            return
        color = "#ff6b6b" if on else _T.palette.text_muted
        vw.setStyleSheet(
            "QLabel#VideoFeed {"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            " background-color: #000;"
            f" color: {color};"
            f" border-radius: {_T.radius.md}px;"
            " font-size: 11pt;"
            "}")
        vw.setText("Camera not connected" if on else "No Video")

    def _build_info_bar(self, H):
        # Top action row (taller than the card's 25px rows): holds Start + Stop.
        # State / Cam / COM / FPS live in the Live Status panel.
        TH = self.INFO_BAR_HEIGHT
        info_bar = QtWidgets.QHBoxLayout()
        info_bar.setContentsMargins(1, 1, 1, 1)
        info_bar.setSpacing(4)

        # Per-arena label is a flat button; same cyan-hover idiom as the
        # operant box_label. Reads "Maze N"; the tab title still uses
        # "Setup N" so detached-window titles stay put.
        self.id_button = QtWidgets.QPushButton(f"Maze {self.setup_id}")
        self.id_button.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.Bold))
        self.id_button.setFixedSize(60, TH)
        self.id_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.id_button.setToolTip("Click to edit subject metadata for this setup")
        self.id_button.setFlat(True)
        self.id_button.setStyleSheet(
            "QPushButton{background:transparent;border:none;color:#e0e0e0;"
            "padding:0px;font-weight:bold;}"
            "QPushButton:hover{color:#61dafb;text-decoration:underline;}"
        )
        self.id_button.clicked.connect(
            lambda: self.metadata_clicked.emit(self.setup_id))
        info_bar.addWidget(self.id_button)

        self.subject_id_edit = make_lineedit(
            placeholder="Enter ID", width=110, height=TH)
        info_bar.addWidget(self.subject_id_edit)

        # Task picker grows with the window: a generous minimum width plus
        # an Expanding policy so it (and only it) absorbs the row's slack.
        self.task_combo = NestedMenu("--- Select Task ---", ".py")
        self.task_combo.setMinimumWidth(220)
        self.task_combo.setFixedHeight(TH)
        self.task_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed)
        self.task_combo.set_callback(self._on_task_selected)
        info_bar.addWidget(self.task_combo, stretch=1)

        self.upload_button = make_button(
            "Upload", color="info", icon=self._icon_dir / 'upload.svg',
            width=69, height=TH, enabled=False)
        self.upload_button.clicked.connect(self.on_upload_clicked)
        info_bar.addWidget(self.upload_button)

        # Start + Stop sit in the top row beside Upload (handlers wired in
        # setupConnections).
        self.record_button = make_button(
            "Start", color="warning", icon=self._icon_dir / 'play.svg',
            width=69, height=TH, enabled=False)
        info_bar.addWidget(self.record_button)
        self.stop_button = make_button(
            "Stop", color="danger", icon=self._icon_dir / 'stop.svg',
            width=69, height=TH, enabled=False)
        info_bar.addWidget(self.stop_button)

        self.timer_label = QtWidgets.QLabel("00:00:00")
        self.timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setFont(QtGui.QFont("Cascadia Mono", 9, QtGui.QFont.Weight.Bold))
        self.timer_label.setStyleSheet(TIMER_LABEL_STYLE)
        self.timer_label.setFixedSize(120, TH)
        info_bar.addWidget(self.timer_label)
        # RunTask owns the 1Hz tick via self._timer_label.
        self._timer_label = self.timer_label

        # TaskInfo consumes the live MCU stream (_data_consumers); its State
        # field is shown in the Live Status panel (_build_live_status_panel).
        self.task_info = TaskInfo()
        return info_bar

    # ----- row 2: video tile + live status -----

    def _build_middle(self, H):
        from source.gui.theme import THEME as _T
        middle = QtWidgets.QHBoxLayout()
        middle.setSpacing(4)

        # GPU-blitted tile. wrappers (setPixmap/setText/setAlignment)
        # keep existing callers working; the numpy fast path is
        # update_frame_from_array(ndarray).
        # Maze: the tile keeps the camera's shape at every size. It expands
        # with the window, always in proportion, so an arena is never drawn
        # elliptical and never shrinks into a band of black bars inside a
        # differently-shaped cell.
        self.video_label = VideoTile(lock_widget_aspect=True)
        self.video_label.setObjectName("VideoFeed")
        self.video_label.setStyleSheet(
            "QLabel#VideoFeed {"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            " background-color: #000;"
            f" color: {_T.palette.text_muted};"
            f" border-radius: {_T.radius.md}px;"
            " font-size: 11pt;"
            "}")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("No Video")
        self.video_label.setMinimumSize(320, 240)
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding)
        policy.setHeightForWidth(True)
        self.video_label.setSizePolicy(policy)
        # The box, not the tile, goes in the layout: Qt largely ignores
        # heightForWidth for a widget sitting in a horizontal row, so the
        # policy above is necessary and not sufficient. The box takes the cell
        # and places the tile inside it at the camera's ratio. ``video_label``
        # stays the tile, so every existing caller is unaffected.
        self._video_box = AspectBox(self.video_label)
        middle.addWidget(self._video_box, stretch=3)

        middle.addWidget(self._build_live_status_panel(), stretch=1)
        return middle

    def _build_live_status_panel(self):
        from source.gui.theme import THEME as _T
        panel = QtWidgets.QFrame()
        panel.setStyleSheet(
            "QFrame {"
            f" background-color: {_T.palette.surface};"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            f" border-radius: {_T.radius.md}px;"
            "}")
        # Wide enough to hold the State row + the Cam / COM / FPS read-outs.
        panel.setMinimumWidth(259)
        panel.setMaximumWidth(403)
        self.live_status_panel = panel

        ls = QtWidgets.QVBoxLayout(panel)
        ls.setContentsMargins(8, 8, 8, 8)
        ls.setSpacing(4)

        H = self.WIDGET_HEIGHT

        def _k(text):
            lab = QtWidgets.QLabel(text)
            lab.setFixedHeight(H)
            lab.setStyleSheet(
                f"font-size: 9pt; color: {_T.palette.text_muted};"
                " border: none; background: transparent;")
            return lab

        # ── State line (top): "<state name>          <dur>" ──
        # The state name (TaskInfo-driven ``state_text``) takes the whole
        # left side; "no states" when idle. ``dur`` on the right = how long
        # the CURRENT state has been running, in MCU firmware time; reset on
        # every state change. Stamped by _on_state_changed
        # (TaskInfo.state_change_cb) and ticked once a second by
        # tick_state_duration (driven from base._on_process_tick).
        #
        # process_data/update_state re-apply ``self._STATE_NORMAL_STYLE`` on
        # every state change, so override that instance attribute (not just
        # the live stylesheet) or the name shrinks back to the 10pt operant
        # font on the next update. TaskInfo's 140px fixed width is cleared so
        # the name fills the wide panel. Colours unchanged: name white, dur
        # amber.
        panel_state_style = (
            "QLineEdit { font-size: 13pt; font-weight: 700;"
            " color: #8be9fd; background: transparent; border: none;"
            " font-family: 'Cascadia Mono', 'Consolas', monospace;"
            " padding-left: 2px; }")
        panel_state_warn = (
            "QLineEdit { font-size: 13pt; font-weight: 700;"
            " color: orange; background: transparent; border: none;"
            " font-family: 'Cascadia Mono', 'Consolas', monospace;"
            " padding-left: 2px; }")
        self.task_info._STATE_NORMAL_STYLE = panel_state_style
        self.task_info._STATE_WARN_STYLE = panel_state_warn
        self.task_info.state_text.setFixedHeight(H)
        self.task_info.state_text.setMinimumWidth(0)
        self.task_info.state_text.setMaximumWidth(16777215)
        self.task_info.state_text.setStyleSheet(panel_state_style)
        self.task_info.state_text.setText("no states")

        self.state_dur_label = QtWidgets.QLabel("00:00")
        self.state_dur_label.setFixedHeight(H)
        self.state_dur_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.state_dur_label.setStyleSheet(
            f"font-size: 11pt; font-weight: 700; color: {_T.palette.warning};"
            " font-family: 'Cascadia Mono', 'Consolas', monospace;"
            " border: none; background: transparent;")
        self.state_dur_label.setToolTip("Time the current state has been running")

        state_hdr = QtWidgets.QHBoxLayout()
        state_hdr.setContentsMargins(0, 0, 0, 0)
        state_hdr.setSpacing(6)
        state_hdr.addWidget(self.task_info.state_text, stretch=1)
        state_hdr.addWidget(self.state_dur_label)
        ls.addLayout(state_hdr)
        # Stamp state-entry time on every state change (firmware time).
        self.task_info.state_change_cb = self._on_state_changed

        # Cam / COM / FPS share ONE compact row, the two id fields expand.
        self.camera_id_edit = make_lineedit(
            placeholder="Cam", height=H, readonly=True, expanding=True)
        self.com_id_edit = make_lineedit(
            placeholder="COM", height=H, readonly=True, expanding=True)
        self.fps_label = QtWidgets.QLabel("--/--")
        self.fps_label.setFixedHeight(H)
        self.fps_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setStyleSheet(
            f"font-size: 9pt; color: {_T.palette.success};"
            " font-family: 'Cascadia Mono', 'Consolas', monospace;"
            " font-weight: 700;")
        self.fps_label.setToolTip("display fps / acquiring fps")
        meta_row = QtWidgets.QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(6)
        meta_row.addWidget(_k("Cam"))
        meta_row.addWidget(self.camera_id_edit, stretch=1)
        meta_row.addWidget(_k("COM"))
        meta_row.addWidget(self.com_id_edit, stretch=1)
        meta_row.addWidget(_k("FPS"))
        meta_row.addWidget(self.fps_label)
        ls.addLayout(meta_row)

        # Single-line run-state indicator (Ready/Running/Stopped/Error),
        # distinct from the cumulative log below. RunTask writes here via
        # set_status; nothing else.
        self.state_status = QtWidgets.QLineEdit()
        self.state_status.setReadOnly(True)
        self.state_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status_widget = self.state_status
        self.set_status("Disconnected", "neutral")
        ls.addWidget(self.state_status)

        # "Live status" header, sits directly above the log it labels.
        header = QtWidgets.QLabel("Live status")
        header.setStyleSheet(
            "font-size: 11pt; font-weight: 700;"
            f" color: {_T.palette.text}; border: none;"
            " background-color: transparent;"
            " text-transform: uppercase; letter-spacing: 0.5px;")
        ls.addWidget(header)

        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet(
            "QTextEdit {"
            f" background-color: {_T.palette.bg};"
            f" border: 1px solid {_T.palette.surface_border};"
            f" border-radius: {_T.radius.sm}px; padding: 6px;"
            " font-family: 'Cascadia Mono', 'Consolas', monospace;"
            f" font-size: {_T.font.mono_pt}pt; color: {_T.palette.text};"
            "}")
        ls.addWidget(self.status_text, stretch=1)
        return panel

    # ----- row 3: bottom bar (experiment ctrl | zone adjust | detach) -----

    @staticmethod
    def _group_styles():
        from source.gui.theme import THEME as _T
        group_qss = (
            "QFrame {"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            f" border-radius: {_T.radius.md}px;"
            f" background-color: {_T.palette.surface};"
            "}"
        )
        title_qss = (
            f"color: {_T.palette.text_muted}; font-size: 8pt; font-weight: 700;"
            " background: transparent; padding: 0px; margin: 0px;"
            " text-transform: uppercase; letter-spacing: 0.5px;"
        )
        return group_qss, title_qss

    def _build_bottom_bar(self, H):
        from source.gui.style_builders import button_style as _btn_st
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        bar.addWidget(self._build_experiment_control(H))
        bar.addWidget(self._build_zone_adjust(H))
        bar.addStretch()

        self.detach_btn = QtWidgets.QPushButton("Detach")
        self.detach_btn.setFixedSize(63, H)
        self.detach_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # "secondary" (slate) instead of "ghost", ghost's border is
        # transparent, so the button read as borderless. Secondary has a
        # visible 1px border so Detach looks like a proper button.
        self.detach_btn.setStyleSheet(_btn_st("secondary", height=H, padding_h=6))
        bar.addWidget(self.detach_btn)
        return bar

    def _build_experiment_control(self, H):
        group_qss, title_qss = self._group_styles()
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(group_qss)
        vbox = QtWidgets.QVBoxLayout(frame)
        vbox.setContentsMargins(3, 2, 3, 3)
        vbox.setSpacing(1)

        title = QtWidgets.QLabel("Experiment Control")
        title.setStyleSheet(title_qss)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        # This group holds the stage/door controls (Start + Stop are in the top
        # info bar). Pause toggles to "Continue" at runtime in
        # _extra_button_state, so it starts as warning here.
        self.pause_btn       = make_button("Pause",   color="warning", height=H)
        self.next_btn        = make_button("Next >>", color="primary", height=H)
        self.prev_btn        = make_button("<< Prev", color="info",    height=H)
        self.doors_btn       = make_button(
            "Doors", color="primary", height=H,
            tooltip="Manual door control (MCU connected, framework stopped)",
        )
        self.controls_button = make_button(
            "Controls", color="ghost",
            icon=self._icon_dir / 'settings.svg', height=H)

        for b in (self.pause_btn, self.next_btn, self.prev_btn,
                  self.doors_btn, self.controls_button):
            row.addWidget(b)

        vbox.addLayout(row)
        return frame

    def _build_zone_adjust(self, H):
        group_qss, title_qss = self._group_styles()
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(group_qss)
        vbox = QtWidgets.QVBoxLayout(frame)
        vbox.setContentsMargins(3, 2, 3, 3)
        vbox.setSpacing(1)

        title = QtWidgets.QLabel("Adjust Zone")
        title.setStyleSheet(title_qss)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)

        from source.gui.widgets.zone_adjust_row import ZoneAdjustRow
        self.zone_adjust_row = ZoneAdjustRow(
            self.setup_id, button_w=22, button_h=H, edit_w=44, font_pt=8)
        self.zone_adjust_row.forward_signals_to(self)
        vbox.addWidget(self.zone_adjust_row)
        return frame

    def setupConnections(self):
        """Setup signal connections."""
        sid = self.setup_id
        self.subject_id_edit.textChanged.connect(self.on_subject_id_changed)
        # Paint the empty-subject (no-ID) styling right away.
        self.on_subject_id_changed()
        # Record/Stop drive the SHARED RunTask orchestration directly (same
        # path as operant).
        self.record_button.clicked.connect(self.on_record_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.pause_btn.clicked.connect(lambda: self.pause_clicked.emit(sid))
        self.next_btn.clicked.connect(lambda: self.next_clicked.emit(sid))
        self.prev_btn.clicked.connect(lambda: self.prev_clicked.emit(sid))
        self.doors_btn.clicked.connect(lambda: self.doors_clicked.emit(sid))
        self.controls_button.clicked.connect(lambda: self.controls_clicked.emit(sid))
        self.detach_btn.clicked.connect(lambda: self.detach_clicked.emit(sid))

    # on_subject_id_changed inherited from RunTask; tab-title emit hooks here.
    def _record_button_extra_qss(self) -> str:
        # Bake the tall-row height into the Record/Start restyle, the
        # plain BUTTON_STYLE (min-height 20) would shrink Start to ~22px
        # next to the 34px Upload/Stop.
        TH = self.INFO_BAR_HEIGHT
        return f"QPushButton {{ min-height: {TH}px; max-height: {TH}px; }}"

    def _after_subject_id_change(self, subject_id):
        self._emit_tab_title()

    # _update_button_states + _is_box_connected + _is_box_running live
    # in RunTask. Maze adds the pause/next/prev/doors enables via the
    # hook below.

    def _extra_button_state(self, *, connected, task_ready, running, setup_enabled):
        # Maze's stage-specific buttons.
        self.pause_btn.setEnabled(running)
        # Restyle only when the paused state actually flips; this runs on
        # the 1 Hz refresh, and a per-tick setStyleSheet is a Qt style
        # recompute for zero visual change.
        paused = bool(getattr(self, "is_paused", False))
        if getattr(self, "_pause_btn_paused_style", None) is not paused:
            self._pause_btn_paused_style = paused
            if paused:
                self.pause_btn.setText("Continue")
                self.pause_btn.setStyleSheet(BUTTON_STYLE.format(
                    color="#4ade80", hover_color="#22c55e"))
            else:
                self.pause_btn.setText("Pause")
                self.pause_btn.setStyleSheet(BUTTON_STYLE.format(
                    color=COLORS['warning'], hover_color=COLORS['warning_hover']))
        self.next_btn.setEnabled(running)
        self.prev_btn.setEnabled(running)
        self.doors_btn.setEnabled(connected and not running)

    def set_zones_enabled(self, enabled):
        """Toggle the Adjust-Zone row. Driven by main_window based on
        whether this arena has tracking zones configured."""
        self.zone_adjust_row.set_zones_enabled(enabled)

    def zone_step_value(self) -> float:
        return self.zone_adjust_row.zone_step_value()

    # apply_global_state inherited from RunTask.

    # ── Status methods ──

    def print_to_log(self, message):
        """Print to live status panel."""
        self.append_status(message)

    def clear_log(self) -> None:
        """Clear the maze box's live-status log (called at run start). This is
        what the shared reset drives, maze's log is ``status_text``, not
        operant's LiveStatusWidget, so it needs its own clear."""
        try:
            self.status_text.clear()
        except (AttributeError, RuntimeError):
            pass

    # _com_text + com_port inherited from RunTask (both modes use com_id_edit).

    # ── Public API ──

    def update_frame(self, pixmap):
        if pixmap and not pixmap.isNull():
            # Stretch-fill: matches updateBoxDisplay's tile-fill resize so
            # the maze setup tile fills the cell on window resize (no
            # black side-bands when source aspect != cell aspect).
            scaled = pixmap.scaled(
                self.video_label.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation)
            self.video_label.setPixmap(scaled)

    # ── Box-widget protocol, methods MainWindowBase drives uniformly.
    # update_frame, append_status: already defined above. get_roi and
    # camera_id_text inherited from RunTask. ──

    def clear_video(self):
        """Blank the tile after a camera disconnect. ``clear_frame`` drops the
        painted source, clearing the label alone left the last frame up."""
        try:
            clear_frame = getattr(self.video_label, "clear_frame", None)
            if callable(clear_frame):
                clear_frame()
            self.video_label.setText("No Video")
        except Exception:
            pass

    def video_size(self):
        try:
            return self.video_label.size()
        except Exception:
            return None

    def video_visible(self):
        """Whether this box's tile is on screen, see
        ``MainWindowBase._is_box_tile_visible``. None = unknown, paint anyway."""
        try:
            return self.video_label.isVisible()
        except Exception:
            return None

    def set_fps_text(self, text, tooltip=""):
        if not hasattr(self, "fps_label"):
            return
        try:
            self.fps_label.setText(text)
            if tooltip:
                self.fps_label.setToolTip(tooltip)
        except Exception:
            pass

    def notify_camera_starting(self, camera_id):
        try:
            self.append_status(f"Camera {camera_id} starting...")
        except Exception:
            pass

    def notify_camera_disconnected(self):
        try:
            self.append_status("Camera disconnected")
        except Exception:
            pass

    def update_frame_from_array(self, frame):
        # Fast path: VideoTile accepts numpy directly, it builds one QImage
        # (BGR888, no cv2.cvtColor) and paints it scaled to the tile in
        # paintEvent.
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        self.video_label.update_frame(frame)

    # State/event/print updates flow straight into self.task_info via
    # pycboard's data_consumers (TaskInfo.process_data); fps_label is
    # written by main_window directly. RunTask's mcu_start / mcu_stop /
    # set_status own the run state; _after_status_change below mirrors the
    # running flag for the tab-title icon picker.

    # External dialogs (door.py, mcu.py) read these as flags. Expose them
    # as properties over RunTask's canonical state. is_connected is
    # inherited from RunTask.
    @property
    def is_running(self) -> bool:
        return bool(self.framework_running)

    @property
    def is_recording(self) -> bool:
        """Read-only view of the canonical run mode: True iff this is a
        real saved session (subject present at Start). Drives the red
        tab-title icon."""
        return self.run_mode == RunMode.RECORD

    def _after_status_change(self, text: str, kind: str) -> None:
        """RunTask hook, called from set_status after every status
        change.  Maze uses it to drive the tab-title state icon."""
        running = kind in ("running", "recording")
        if not running:
            self.is_paused = False
            # Run ended, reset the state line to idle and freeze the
            # per-state duration read-out.
            self._state_entry_fw_ms = None
            if hasattr(self, "state_dur_label"):
                self.state_dur_label.setText("00:00")
            ti = getattr(self, "task_info", None)
            if ti is not None and getattr(ti, "state_text", None) is not None:
                ti.state_text.setText("no states")
        self._emit_tab_title()

    # ── Per-state duration ──
    def _on_state_changed(self, name, fw_time) -> None:
        """TaskInfo.state_change_cb, stamp the firmware time the current
        state was entered, so the Live Status panel can show how long it
        has been running. ``fw_time`` is the STATE message's firmware ms;
        the typed/IPC path passes None, so fall back to the board clock."""
        if fw_time is None:
            try:
                fw_time = self.pycboard.get_timestamp()
            except Exception:
                fw_time = self._state_entry_fw_ms
        self._state_entry_fw_ms = fw_time
        if hasattr(self, "state_dur_label"):
            self.state_dur_label.setText("00:00")

    def tick_state_duration(self, fw_ms) -> None:
        """Called once a second from base._on_process_tick with the box's
        current firmware time. Sets the dur label to MM:SS since state entry."""
        if not hasattr(self, "state_dur_label"):
            return
        entry = self._state_entry_fw_ms
        if entry is None or fw_ms is None:
            self.state_dur_label.setText("00:00")
            return
        total_s = max(0, int(fw_ms) - int(entry)) // 1000
        self.state_dur_label.setText("%02d:%02d" % (total_s // 60, total_s % 60))

    def _data_consumers(self) -> list:
        """RunTask hook, what to feed pycboard's ``data_consumers``.
        Maze's ``TaskInfo`` widget (state/event/print line) consumes the
        live MCU stream so the per-state line updates as the task runs."""
        return [self.task_info]

    def set_paused(self, paused):
        # GUI elapsed-time clock keeps ticking through pause, it tracks FW
        # wall-clock, not stage progression. Pause only flips the flag for
        # button enables and tab-title display.
        self.is_paused = paused
        self._update_button_states()
        self._emit_tab_title()

    def get_tab_title(self):
        """Build tab title: Setup{N} · SubjectID {state_icon}"""
        title = f"Setup{self.setup_id}"
        subject = self.subject_id_edit.text().strip()
        if subject:
            title += f" · {subject}"
        # State icon (rightmost wins). framework_running is owned by RunTask.
        if self.is_recording:
            title += " \U0001f534"    # red circle = recording
        elif self.is_paused:
            title += " ⏸"        # pause icon
        elif self.framework_running:
            title += " ▶"        # play icon
        return title

    def _emit_tab_title(self):
        """Notify parent to update this tab's title."""
        self.tab_title_changed.emit(self.setup_id, self.get_tab_title())

    def append_status(self, text, end="\n"):
        """Append to live status panel with timestamp and color.

        Can be passed as Pycboard's print_func, pycboard's progress
        messages use ``self.print(..., end="")`` for inline updates (e.g.
        "Transferring framework...."). Accepts string or anything with
        __str__.
        """
        if not isinstance(text, str):
            text = str(text)
        # Inline-progress writes (end="") shouldn't add a new timestamp/line.
        if end == "" and self.status_text.toPlainText():
            cursor = self.status_text.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            self.status_text.setTextCursor(cursor)
            self.status_text.insertPlainText(text)
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        text_lower = text.lower()
        if "error" in text_lower:
            color = "#ff6b6b"
        elif "warning" in text_lower or text.startswith("! "):
            color = "orange"
        else:
            color = "#f8f8f2"
        self.status_text.setTextColor(QtGui.QColor(color))
        self.status_text.append(f"[{timestamp}] {text}")
        cursor = self.status_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.status_text.setTextCursor(cursor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current = self.video_label.pixmap()
        if current and not current.isNull():
            # Stretch-fill on resize so the maze setup tile fills the
            # cell.  Matches operant VideoStreamHolder.resizeEvent.
            scaled = current.scaled(
                self.video_label.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation)
            self.video_label.setPixmap(scaled)

    # =========================================================================
    # MCU OWNERSHIP, direct (mirrors pyOperant_final BoxControlWidget shape).
    # Pycboard lives on this widget; widgets call its methods directly.
    # =========================================================================

    # connect_mcu / disconnect_mcu are shared on RunTask. Maze's only UI
    # extras (task-menu refresh on connect, clearing the COM field on
    # disconnect) live in the hooks below.
    def _after_mcu_connected(self) -> None:
        self._refresh_task_menu()

    def _after_mcu_disconnected(self) -> None:
        self.com_id_edit.setText("")
        self.upload_button.setText("Upload")

    # _on_task_selected + _refresh_task_menu inherited from RunTask (shared).

    # _on_upload_clicked is RunTask.on_upload_clicked. _upload_log /
    # _upload_task_text are inherited too (defaults route through _log_func /
    # the task combo). Only _after_upload_clicked needs a maze override.

    def _after_upload_clicked(self, raw_text, rel, sm_name):
        # Maze extra: sync the task picker to the resolved relative path so
        # the next click sees the same canonical text.
        if hasattr(self, "task_combo") and hasattr(self.task_combo, "setText"):
            try:
                self.task_combo.setText(rel)
            except Exception:
                pass
        # Shared stats-config load (no folder work at upload time, the dated
        # session dirs are created lazily at record start).
        self._load_stats_config_for_task(raw_text)

    # upload_task inherited from RunTask (sets task_combo + clicks upload_button).
    # start_framework / on_record_clicked / on_stop_clicked are inherited from
    # RunTask (the shared record/stop path). The maze-specific hooks below
    # supply the RECORD/DRY guts.

    # ------------------------------------------------------------------
    # Maze hooks for the shared RunTask record/stop path
    # ------------------------------------------------------------------
    def _start_recording(self, subject_id, datetime_now, metadata):
        """RECORD branch: open the MCU TSV + early tracking writer + FW anchor
        + history row, then the one MP4 recorder, the SAME central
        ``open_session_*`` calls operant uses (see BoxControlWidget). The only
        maze-specific extras are the sidebar ``session`` field, the shared
        session timer, the per-box recording context, and the current-state
        video label, all supplied by maze MainWindow helpers.
        ``open_session_mcu_tsv`` lets an open failure propagate so the shared
        orchestrator aborts before the framework starts."""
        mw = self.main_window
        setup_id = self.setup_number
        # Maze extras on top of the shared sidebar/cohort metadata.
        md = dict(metadata)
        md.setdefault("box_id", setup_id)
        md.setdefault("subject_id", subject_id)
        session = mw.info_fields.get("session") if hasattr(mw, "info_fields") else None
        if session is not None and hasattr(session, "text"):
            val = session.text().strip()
            if val:
                md["session"] = val
        session_dir, pyboard_dir, video_dir = self._ensure_session_dirs()
        # Session timer + drop-count reset fire inside the shared
        # open_session_video_recorder (first box to start owns them).
        # Per-box recording context, marks the box recording (the stop path
        # reads it to close the MCU data file) and stashes the session dir.
        mw._recording_ctx[setup_id] = {
            "subject_id": subject_id,
            "video_dir": str(video_dir),
            "session_dir": str(session_dir),
            "metadata": dict(md),
        }
        # Open _video_data.txt BEFORE start_framework so the first MCU event
        # lands in an existing writer. Skipped during Test Tracking.
        if not getattr(mw, "_test_tracking_dry_run", False):
            try:
                mw.open_session_tracking_writer(
                    setup_id, video_dir, subject_id, datetime_now)
            except Exception as e:
                logger.warning(
                    "Setup%s: early tracking-writer open failed: %s", setup_id, e)
        # MCU-TSV open + source commit + FW anchor + history row via the shared
        # base.open_session_mcu_tsv. video_info stays None: maze pose lives in
        # _video_data.txt, not the TSV header.
        mw.open_session_mcu_tsv(
            setup_id, pyboard_dir, subject_id, datetime_now, md)
        # The one MP4 recorder, same central builder as operant. The current
        # MCU state is a label in the video metadata; the tracking writer was
        # opened above (sw.tracking_writer).
        state_name = mw._current_state_name(setup_id) or "session"
        rec_meta = dict(md)
        rec_meta["state"] = state_name
        annotate_map = (mw._tracking_dialog_globals or {}).get("annotate_saved") or {}
        annotate_cb = (mw._build_annotate_callback(setup_id)
                       if (annotate_map.get(str(setup_id)) or annotate_map.get(setup_id))
                       else None)
        mw.open_session_video_recorder(
            setup_id, video_dir, subject_id,
            datetime_now, tracking_writer=self.tracking_writer,
            annotate_callback=annotate_cb, metadata=rec_meta)

    # _begin_temp_safety_run is shared on RunTask; maze supplies the
    # camera-streaming check + the _recording_ctx stamp via these two hooks.
    # _is_camera_streaming inherited from RunTask (shared VideoManager check).
    def _after_temp_mcu_open(self, ok: bool) -> None:
        if ok:
            self.main_window._recording_ctx[self.setup_number] = {
                "temp_safety": True}

    def _post_record_start(self, subject_id):
        """Bookkeeping after a successful framework start: enable live
        inference, mark the box as recording, refresh UI."""
        mw = self.main_window
        setup_id = self.setup_number
        # Live inference always runs (pose annotation on the video tile),
        # even in dry-run mode; file creation is gated separately.
        mw.start_tracking_for_box(setup_id)
        try:
            mw.start_box_recording(setup_id)
        except Exception as e:
            logger.debug("Setup%s: start_box_recording bookkeeping failed: %s",
                         setup_id, e)
        mw.refresh_ui_state()

    def _cleanup_after_start_failure(self):
        """Close anything opened before a failed framework start, then run
        the normal teardown so no half-open writer/bridge leaks."""
        try:
            self.main_window._on_record_stopped(self.setup_number, {})
        except Exception as e:
            logger.debug("Setup%s: cleanup_after_start_failure: %s",
                         self.setup_number, e)
        self._update_button_states()

    def _after_start(self, record: bool) -> None:
        """Shared mcu_start convergence: register this box as a stats data
        consumer, the SAME hook operant uses, instead of a per-mode
        framework_started_signal slot."""
        try:
            self.main_window._register_stats_consumer(self.setup_number, self)
        except Exception as e:
            logger.debug("Setup%s: _after_start stats register failed: %s",
                         self.setup_number, e)

    def _after_stop(self, reason: str) -> None:
        """Shared mcu_stop convergence: unregister stats, then run the maze
        recording/tracking teardown (matches operant closing in _after_stop)."""
        try:
            self.main_window._unregister_stats_consumer(self.setup_number)
        except Exception as e:
            logger.debug("Setup%s: _after_stop stats unregister failed: %s",
                         self.setup_number, e)
        try:
            self.main_window._on_record_stopped(
                self.setup_number, {"reason": reason})
        except Exception as e:
            logger.debug("Setup%s: _after_stop cleanup failed: %s",
                         self.setup_number, e)

    # _uninstall_mcu_row_mirror inherited from RunTask.

    def pause(self):
        self.mcu_pause()
        self.set_paused(True)

    def resume(self):
        self.mcu_resume()
        self.set_paused(False)

    def next_stage(self):
        self.mcu_next_stage()

    def prev_stage(self):
        self.mcu_prev_stage()

    # The run clock is driven by MainWindowBase.process_timer reading
    # pycboard.get_timestamp(); RunTask owns the label.
