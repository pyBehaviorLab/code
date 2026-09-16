"""Operant per-box control card (BoxControlWidget)."""

from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from datetime import datetime

from source.gui.styles import (
    COLORS, BUTTON_STYLE, COMBOBOX_STYLE, TIMER_LABEL_STYLE,
)
from source.gui.theme import THEME
from source.gui.utility import NestedMenu
from source.gui.widgets.common import make_button, make_lineedit
from source.gui.widgets.run_task import RunTask
from source.log import get_logger
from source.communication.errors import (
    ERR_FRAMEWORK_NOT_LOADED, ERR_TASK_NOT_SELECTED,
    ERR_TASK_FILE_NOT_FOUND, ERR_TASK_UPLOAD_FAILED,
    ERR_TASK_SETUP_FAILED, HINT_UPLOAD_FRAMEWORK,
)

logger = get_logger()


class BoxControlWidget(QtWidgets.QFrame, RunTask):

    # Emitted when the user clicks the box-number header, main_window opens
    # the per-subject metadata dialog for this box. Same UX as maze.
    metadata_clicked = QtCore.Signal(int)

    # Lifecycle signals so operant can keep its multi-box bookkeeping
    # in sync regardless of how the framework started/stopped (button
    # click vs. auto-stop on duration / MCU end-of-run / serial error).
    framework_started_signal = QtCore.Signal(int)  # box_number
    framework_stopped_signal = QtCore.Signal(int)  # box_number

    def __init__(self, setup_number, main_window=None):
        super().__init__()
        # Per-box state owned by RunTask: pycboard, framework_running,
        # task_uploaded, run_mode, _status_widget, _timer_label.
        self.init_run_task(setup_number, main_window=main_window)
        # Operant writer-open flags (NOT the run mode, that's RunTask's
        # ``run_mode``). ``recording_data`` = the MCU TSV is open (real OR
        # temp); ``recording_video`` = the VideoRecorder started OK. Readers
        # in operant.py + base.py use them for close bookkeeping.
        self.action_config_path = None
        self.video_recorder = None
        # Real video path of the active run, stashed at encoder start so the
        # runs JSON can resolve it even after the recorder is torn down.
        self._run_video_path = None
        self.save_video_enabled = True
        self.recording_data = False
        self.recording_video = False
        self._build_ui()
        self.setupConnections()
        # Wire canonical status surface + timer label for RunTask.
        self._status_widget = self.status_edit
        self._timer_label   = self.timer_label
        # Operant's status starts greyed-out and disabled; on connect it
        # flips to enabled + "Connected".
        self.status_edit.setEnabled(False)
        self._update_button_states()
        logger.debug(f"Initialized BoxControlWidget for box {setup_number}")

    def _set_status(self, text, error=False):
        """Routes through RunTask.set_status so there is ONE writer of the
        status_edit widget. Many operant call sites use this wrapper."""
        self.set_status(text, "error" if error else "neutral")

    def _set_error(self, text):
        """Truncates and routes through set_status."""
        short = (text or "Error")[:32]
        self.set_status(short, "error")

    # Unified box-widget API (matches SetupWidget), used by Universal dialogs.
    # is_connected, com_port, connect_mcu(port), disconnect_mcu() are the same
    # surface both BoxControlWidget and SetupWidget expose so dialogs treat
    # box==setup uniformly via main_window.get_all_setup_widgets().
    # is_connected property inherited from RunTask.

    # com_port property inherited from RunTask; hook supplies operant text.
    # The canonical COM value is ``com_id_edit`` (the visible read-only
    # field), written by connect_mcu / project load with the OS-appropriate
    # label. ``serial_combo`` is the hidden per-box picker the master
    # Connect dialog never sets, so it can't be the source here.
    # _com_text inherited from RunTask (both modes use com_id_edit).

    # upload_task inherited from RunTask (sets task_combo text + clicks upload_button).

    # Card row height: 1+1 border + 24 widget + 1+1 margins = 28.
    CARD_H = 28
    ROW_H  = 24  # inner-widget height; QSS min-heights are < ROW_H so this wins.

    def _build_ui(self):
        """Compose the single-row box card. Layout reads top-to-bottom as
        ``[id_group | task_group | recording_group | control_group]``;
        every section is one private builder that returns a QHBoxLayout so
        each is trivially debuggable in isolation."""
        self.setStyleSheet(
            "BoxControlWidget {"
            f" background-color: {THEME.palette.surface};"
            f" border: 1px solid {THEME.palette.surface_border};"
            f" border-radius: {THEME.radius.md}px;"
            "}"
        )
        self.setContentsMargins(0, 0, 0, 0)
        self.setFixedHeight(self.CARD_H)

        self._icon_dir = Path(__file__).parent / 'icons'

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)
        layout.addLayout(self._build_id_group(),        0)
        layout.addLayout(self._build_task_group(),      1)  # task_combo grows
        layout.addLayout(self._build_recording_group(), 0)
        layout.addLayout(self._build_control_group(),   2)  # status grows ~2× task

        self._build_legacy_hidden_widgets()

    def set_error(self, on, msg=""):
        """Render this box's error state: red card border + red box label +
        tooltip with the message. Called centrally by
        ``box_alerts.apply_box_alerts``; idempotent and auto-clears."""
        from source.gui.box_alerts import (
            ERROR_COLOR, error_tooltip, identity_button_style)
        on = bool(on)
        border = ERROR_COLOR if on else THEME.palette.surface_border
        self.setStyleSheet(
            "BoxControlWidget {"
            f" background-color: {THEME.palette.surface};"
            f" border: {2 if on else 1}px solid {border};"
            f" border-radius: {THEME.radius.md}px;"
            "}"
        )
        lbl = getattr(self, "box_label", None)
        if lbl is not None:
            lbl.setStyleSheet(identity_button_style(
                ERROR_COLOR if on else THEME.palette.text))
            lbl.setToolTip(error_tooltip(
                "Box {}".format(self.setup_number), on, msg,
                idle="Click to edit subject metadata for this box.",
                bare="Box {} error".format(self.setup_number)))

    # ----- subgroups (each returns a QHBoxLayout) -----

    def _build_id_group(self):
        g = QtWidgets.QHBoxLayout()
        g.setSpacing(3)

        # Box-number label is a flat button, clicking opens the metadata
        # dialog. Cyan link hover, matches maze.SetupWidget.id_button.
        self.box_label = QtWidgets.QPushButton(f"Box {self.setup_number}")
        self.box_label.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.Bold))
        self.box_label.setFixedSize(45, self.ROW_H)
        self.box_label.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.box_label.setToolTip("Click to edit subject metadata for this box.")
        self.box_label.setFlat(True)
        self.box_label.setStyleSheet(
            "QPushButton{background:transparent;border:none;"
            f"color:{THEME.palette.text};padding:0px;font-weight:bold;}}"
            "QPushButton:hover{color:#61dafb;text-decoration:underline;}"
        )
        self.box_label.clicked.connect(
            lambda: self.metadata_clicked.emit(self.setup_number)
        )
        g.addWidget(self.box_label)

        self.subject_id_edit = make_lineedit(
            placeholder="Enter ID", width=85, height=self.ROW_H)
        g.addWidget(self.subject_id_edit)
        return g

    def _build_task_group(self):
        g = QtWidgets.QHBoxLayout()
        g.setSpacing(3)

        self.task_combo = NestedMenu("--- Select Task ---", ".py")
        self.task_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.task_combo.setMinimumWidth(160)
        self.task_combo.setFixedHeight(self.ROW_H)
        self.task_combo.setMaximumHeight(self.ROW_H)
        self.task_combo.set_callback(self._on_task_selected)
        g.addWidget(self.task_combo)

        self.upload_button = make_button(
            "Upload", color="info", icon=self._icon_dir / 'upload.svg',
            width=78, height=self.ROW_H, enabled=False)
        g.addWidget(self.upload_button)
        return g

    def _build_recording_group(self):
        g = QtWidgets.QHBoxLayout()
        g.setSpacing(3)

        # record_button flips green->bright-green when subject_id is set,
        # see RunTask.on_subject_id_changed.
        self.record_button = make_button(
            "Start", color="start_light", icon=self._icon_dir / 'play.svg',
            width=72, height=self.ROW_H, enabled=False)
        g.addWidget(self.record_button)

        self.stop_button = make_button(
            "Stop", color="danger", icon=self._icon_dir / 'stop.svg',
            width=64, height=self.ROW_H, enabled=False)
        g.addWidget(self.stop_button)

        self.timer_label = QtWidgets.QLabel("00:00:00")
        self.timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setFont(QtGui.QFont("Cascadia Mono", 9, QtGui.QFont.Weight.Bold))
        self.timer_label.setStyleSheet(TIMER_LABEL_STYLE)
        self.timer_label.setFixedSize(110, self.ROW_H)
        g.addWidget(self.timer_label)
        return g

    def _build_control_group(self):
        g = QtWidgets.QHBoxLayout()
        g.setSpacing(3)

        self.controls_button = make_button(
            "Controls", color="ghost", icon=self._icon_dir / 'settings.svg',
            width=92, height=self.ROW_H)
        g.addWidget(self.controls_button)


        self.status_edit = make_lineedit(
            placeholder="Status", value="Not connected", readonly=True,
            expanding=True, width=200, height=self.ROW_H)
        self.status_edit.setEnabled(False)
        g.addWidget(self.status_edit)

        self.camera_id_edit = make_lineedit(
            placeholder="Cam", width=60, height=self.ROW_H, readonly=True)
        g.addWidget(self.camera_id_edit)

        self.com_id_edit = make_lineedit(
            placeholder="COM", width=65, height=self.ROW_H, readonly=True)
        g.addWidget(self.com_id_edit)
        return g

    def _build_legacy_hidden_widgets(self):
        """Hidden widgets the universal connect/configure dialogs reach
        into. Kept constructed so those call sites don't AttributeError;
        the visible UI lives on the master controls."""
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.serial_combo = QtWidgets.QComboBox()
        self.serial_combo.setSizePolicy(size_policy)
        self.serial_combo.setFixedSize(80, self.ROW_H)
        self.serial_combo.setStyleSheet(COMBOBOX_STYLE)
        self.serial_combo.setMaximumHeight(self.ROW_H)
        self.serial_combo.setVisible(False)
        # serial_combo is the hidden per-box COM picker only. com_id_edit
        # is the canonical display, fed directly by connect_mcu / project
        # load (not mirrored from this combo), so the periodic
        # populate_com_ports refresh can't blank it out.

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.setVisible(False)
        self.configure_button = QtWidgets.QPushButton("Configure")
        self.configure_button.setVisible(False)

    # _update_button_states + _check_task_consistency + _check_task_hash
    # live in RunTask. Both modes share them so maze also detects
    # on-disk task edits and reverts the button to "Upload".

    # apply_global_state inherited from RunTask.
    # _ensure_session_dirs inherited from RunTask (shared by both modes).

    def _get_box_roi(self):
        """Return ROI tuple (x, y, w, h) for this box in FULL-camera pixels,
        or None.

        Pixel-space accessor (no frame reference), used only by operant's
        ``_tracking_roi`` for the tracking-writer header. The NORMALIZED-first
        resolution, canonical ``roi_normalized`` scaled to the live frame,
        is done by the caller, which has the frame; here we return only the
        pixel session cache / config pixel geometry so the widget stays
        decoupled from the main window.

        Order: ``roi_segment`` (pixel session cache) → config
        ``geometry.pixel``.
        """
        try:
            roi = getattr(self, "roi_segment", None)
            if roi:
                if isinstance(roi, (tuple, list)) and len(roi) == 4 and roi[2] > 0 and roi[3] > 0:
                    return tuple(int(v) for v in roi)
                if isinstance(roi, dict) and all(
                        k in roi for k in ('x', 'y', 'width', 'height')):
                    return (int(roi['x']), int(roi['y']),
                            int(roi['width']), int(roi['height']))

            main_window = self.window()
            config = getattr(main_window, "video_segment_config", None)
            if config and "boxes" in config:
                for box in config["boxes"]:
                    setup_id = box.get("box_id") or box.get("box_number")
                    if setup_id == self.setup_number:
                        pixel = box.get("geometry", {}).get("pixel", {})
                        if all(k in pixel for k in ("x", "y", "width", "height")):
                            return (int(pixel["x"]), int(pixel["y"]),
                                    int(pixel["width"]), int(pixel["height"]))
                        break
        except Exception as e:
            logger.error(f"Failed to read ROI for box {self.setup_number}: {e}")
        return None

    def setupConnections(self):
        """Setup signal connections"""
        try:
            self.subject_id_edit.textChanged.connect(self.on_subject_id_changed)
            # Paint the empty-subject (no-ID) styling right away.
            self.on_subject_id_changed()
            # NestedMenu uses callback (set_callback) instead of currentTextChanged
            # Keep connections for universal dialog compatibility
            self.configure_button.clicked.connect(self.onConfigureClicked)
            self.connect_button.clicked.connect(self.onConnectClicked)
            self.controls_button.clicked.connect(self.onControlsClicked)
            self.upload_button.clicked.connect(self.on_upload_clicked)
            # record_button / stop_button are connected ONCE, to this widget's
            # on_record_clicked / on_stop_clicked (the shared RunTask path used by
            # both modes). Operant-side bookkeeping (recording_setups, live
            # status, Stop enable) lives in the shared _post_record_start hook
            #; there is no second connection and no double framework start.
            self.record_button.clicked.connect(self.on_record_clicked)
            self.stop_button.clicked.connect(self.on_stop_clicked)
            logger.debug(f"Set up connections for box {self.setup_number}")
        except Exception as e:
            logger.error(f"Failed to set up connections for box {self.setup_number}: {str(e)}")

    # _on_task_selected + _refresh_task_menu inherited from RunTask (shared).

    # on_task_changed inherited from RunTask -- shared with maze.

    def onConfigureClicked(self):
        """Handle configure button click"""
        try:
            if self.main_window is None:
                logger.error(f"No main window reference for box {self.setup_number}")
                return

            self.main_window.show_config_dialog(self.setup_number)
            logger.debug(f"Showing config dialog for box {self.setup_number}")
        except Exception as e:
            logger.error(f"Error showing config dialog for box {self.setup_number}: {str(e)}")

    # connect_mcu / disconnect_mcu are shared on RunTask. Operant's UI
    # extras (Connect<->Disconnect button restyle, status_edit enable, the
    # "load framework" hint, recording_video reset) live in the hooks below.
    def _after_mcu_connected(self) -> None:
        self.connect_button.setText("Disconnect")
        self.connect_button.setStyleSheet(BUTTON_STYLE.format(
            color=COLORS['danger'], hover_color=COLORS['danger_hover']))
        self.status_edit.setEnabled(True)
        if self.pycboard is not None and not self.pycboard.status.get("framework"):
            self.print_to_log("Load pyControl framework using 'Config' button.")

    def _after_mcu_disconnected(self) -> None:
        self.connect_button.setText("Connect")
        self.connect_button.setStyleSheet(BUTTON_STYLE.format(
            color=COLORS['success'], hover_color=COLORS['success_hover']))
        self.recording_video = False
        self.status_edit.setEnabled(False)

    def onConnectClicked(self):
        """Connect/Disconnect button click, thin wrapper over
        connect_mcu / disconnect_mcu."""
        try:
            if self.is_connected:
                self.disconnect_mcu()
            else:
                # The bound USB serial first: that is the binding that survives
                # a replug and travels between machines. ``serial_combo`` is the
                # hidden per-box picker and only ever names a local port, so
                # using it first ignored the board this box is bound to.
                self.connect_mcu((getattr(self, "_mcu_serial", "") or "").strip()
                                 or self.serial_combo.currentText())
        except Exception as e:
            self._set_error("Connection error")
            logger.error(f"Error toggling connection box {self.setup_number}: {e}")

    def onControlsClicked(self):
        """Handle controls button click - open tabbed controls dialog.

        Modeless (``show``, not ``exec``) so the operator can still drive the
        main GUI, watch plots, Stop a run, while tuning variables. Keep a
        reference (or the dialog is GC'd the moment this returns) and reuse an
        already-open one instead of stacking duplicates."""
        try:
            existing = getattr(self, "_controls_dialog", None)
            if existing is not None:
                existing.raise_()
                existing.activateWindow()
                return
            from source.gui.dialogs import ControlsDialog
            dialog = ControlsDialog(setup_widget=self, parent=self)
            self._controls_dialog = dialog
            dialog.finished.connect(
                lambda *_: setattr(self, "_controls_dialog", None))
            dialog.show()
            logger.debug(f"Opened controls dialog for box {self.setup_number}")
        except Exception as e:
            logger.error(f"Error opening controls dialog for box {self.setup_number}: {str(e)}")

    # on_upload_clicked inherited from RunTask (unified for both modes).
    # Hooks below adapt the orchestration to operant's status surface +
    # post-upload statistics-config load.

    # _upload_log, _upload_brief_status, _upload_task_text inherited from
    # RunTask, the defaults already route through _log_func / set_status /
    # the task combo for both modes.

    # _after_upload_clicked inherited from RunTask (loads the task's stats
    # config, operant has no extra post-upload work).

    def _after_mcu_upload_success(self, sm_name, task_path):
        """Operant: no folder work at upload time.

        The dated ``data/<project>/<task>/<date>/`` folders are created
        lazily at *record* start (``on_record_clicked`` → ``_ensure_session_dirs``
        → ``get_session_dirs``), so an upload-without-record leaves no empty
        dated folders. mkdir latency at record time is negligible."""
        return None

    # on_subject_id_changed inherited from RunTask; no auto-lookup on
    # typing, subject↔metadata association comes from SetupID matching
    # in the cohort dialog, not from string compare on the typed value.
    def _after_subject_id_change(self, subject_id):
        return None

    def print_to_log(self, text, end="\n"):
        """Print text to the live status widget (used by Pycboard's data_logger).

        This is the print_func passed to Pycboard. The data_logger uses compact format:
        D timestamp state/event_name  (for states and events)
        P timestamp message           (for print messages)

        Do NOT call processEvents() here; this runs during serial
        communication and processEvents() causes re-entrancy that corrupts
        the state machine setup on the pyboard.
        """
        try:
            if self.main_window:
                widget = (getattr(self.main_window, "_live_status_by_box", None)
                          or {}).get(self.setup_number)
                if widget is not None:
                    text_lower = text.lower()
                    # Color-code log output
                    if text.startswith("! ") or "warning" in text_lower:
                        widget.statusEdit.setTextColor(QtGui.QColor("orange"))
                    elif "error" in text_lower:
                        widget.statusEdit.setTextColor(QtGui.QColor("#ff6b6b"))
                    else:
                        widget.statusEdit.setTextColor(QtGui.QColor("#f8f8f2"))

                    cursor = widget.statusEdit.textCursor()
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                    widget.statusEdit.setTextCursor(cursor)
                    widget.statusEdit.insertPlainText(text + end)
                    widget.statusEdit.ensureCursorVisible()

            # status_edit is written ONLY by RunTask.set_status, never from
            # this streaming log path, so a state name like "error_check"
            # never clobbers the sticky "Recording: <task>" / "Error: <msg>"
            # label.

        except Exception as e:
            logger.error(f"Error printing to log for box {self.setup_number}: {str(e)}")

    def _validate_pre_run(self):
        """Validate that this box is ready to start a run.

        Returns:
            (ok: bool, errors: list[str])
        """
        errors = []
        if self.pycboard is None:
            errors.append("Board not connected.")
        else:
            status = getattr(self.pycboard, "status", {})
            if not status.get("framework", True):
                errors.append(ERR_FRAMEWORK_NOT_LOADED + " " + HINT_UPLOAD_FRAMEWORK)

        task_text = self.task_combo.text()
        if not task_text or task_text in ("--- Select Task ---", "No tasks found"):
            errors.append(ERR_TASK_NOT_SELECTED)
        else:
            # Check task file exists on disk
            task_rel = Path(task_text)
            if task_rel.suffix == ".py":
                task_rel = task_rel.with_suffix("")
            sm_dir = Path("tasks") / task_rel.parent
            task_path = sm_dir / (task_rel.name + ".py")
            if not task_path.exists():
                errors.append(ERR_TASK_FILE_NOT_FOUND.format(path=str(task_path)))

        if not self.task_uploaded:
            errors.append(ERR_TASK_UPLOAD_FAILED + " Upload the task first.")

        if self.pycboard and (not hasattr(self.pycboard, 'sm_info') or self.pycboard.sm_info is None):
            errors.append(ERR_TASK_SETUP_FAILED + " State machine not initialized.")

        ok = len(errors) == 0
        if not ok:
            for err in errors:
                self.print_to_log(f"! {err}")
        return ok, errors

    # ------------------------------------------------------------------
    # Operant hooks for the shared record/stop path (RunTask owns
    # on_record_clicked / on_stop_clicked / start_framework / _record_preflight
    # / _reset_live_status_for_box / _collect_run_metadata).
    # ------------------------------------------------------------------
    def _start_recording(self, subject_id, datetime_now, metadata):
        """RECORD branch: open the MCU TSV + history row + FW anchor (via
        ``base.open_session_mcu_tsv``), and, when a camera streams and
        ``save_video_enabled``: the video recorder + tracking writer.
        ``open_session_mcu_tsv`` lets an open failure propagate so the
        shared orchestrator aborts before the framework starts."""
        camera_id = self.camera_id_edit.text().strip()
        session_dir, pyboard_dir, video_dir = self._ensure_session_dirs()
        will_record_video, cam_id_int, video_info = (
            self._resolve_recording_intent(
                subject_id, camera_id, datetime_now)
        )
        self.recording_data = True
        # Defer the video header (video_info=None) when a video will
        # record, its real filename/extension is written post-start by
        # base.open_session_video_recorder. Otherwise pass the explicit
        # "no video" dict so the TSV states it definitively.
        self.main_window.open_session_mcu_tsv(
            self.setup_number, pyboard_dir, subject_id, datetime_now, metadata,
            video_info=None if will_record_video else video_info)
        if will_record_video:
            tw = self.main_window.open_session_tracking_writer(
                self.setup_number, video_dir, subject_id, datetime_now)
            self.main_window.open_session_video_recorder(
                self.setup_number, video_dir, subject_id, datetime_now,
                tracking_writer=tw, metadata=metadata,
                annotate_callback=self._build_annotate_callback_if_enabled())
        elif camera_id and not self.save_video_enabled:
            logger.info(
                f"Box {self.setup_number}: Video save disabled for this session")
        elif camera_id:
            logger.info(
                f"Box {self.setup_number}: Camera {camera_id} not connected, "
                f"skipping video recording")
            self.recording_video = False

    def _post_record_start(self, subject_id):
        """Operant bookkeeping after a successful framework start.

        Single-path: this runs on the shared ``on_record_clicked`` flow (the
        record button is connected ONCE, to ``on_record_clicked``). It also
        registers in ``recording_setups``, enables Stop and posts the
        LiveStatus line, so none of that needs a second handler, and a second
        connection would start the recording twice.
        """
        import time as _time
        mw = self.main_window
        if mw is None:
            return
        setup_id = self.setup_number
        # The video-record path already added this box via
        # open_session_video_recorder; add here too so a pyControl-only
        # (no-video) session is registered as well. Set-add is idempotent.
        try:
            mw.recording_setups.add(setup_id)
        except AttributeError:
            pass
        if hasattr(mw, "recording_start_times"):
            mw.recording_start_times[setup_id] = _time.time()
        try:
            self.stop_button.setEnabled(True)
        except (AttributeError, RuntimeError):
            pass
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if subject_id:
            has_camera = bool(self.camera_id_edit.text().strip())
            msg = (f"Recording started at {ts}" if has_camera
                   else f"Recording (pyControl) started at {ts}")
        else:
            msg = f"Framework started at {ts}"
        if hasattr(mw, "update_live_status_for_box"):
            try:
                mw.update_live_status_for_box(setup_id, msg, ts)
            except (AttributeError, RuntimeError):
                pass
        try:
            mw.start_box_recording(setup_id)
        except Exception as e:
            logger.debug(
                "Box %s: start_box_recording bookkeeping failed: %s",
                setup_id, e)

    def _on_stop_confirmed(self):
        """Immediate stop UX (operant), fired the instant a stop is
        confirmed, before the synchronous mcu_stop: disable the Stop button
        and post a "Stopping…" LiveStatus line so the click registers. Only
        for a real recording."""
        mw = self.main_window
        if mw is None or self.setup_number not in getattr(mw, "recording_setups", set()):
            return
        try:
            self.stop_button.setEnabled(False)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(mw, "update_live_status_for_box"):
                mw.update_live_status_for_box(
                    self.setup_number, f"Stopping at {ts}", ts)
        except Exception as e:
            logger.error("Box %s: stop UX failed: %s", self.setup_number, e)

    def _post_stop_button_restore(self):
        """Reset the per-widget lock flags so every setup-gated button
        (subject_id, task_combo, upload, record, controls) re-enables."""
        try:
            if not self.framework_running:
                self.apply_global_state(
                    lock_setup=False,
                    lock_controls=False,
                )
            else:
                self._update_button_states()
        except Exception as e:
            logger.warning(
                "Box %s: post-stop button restore: %s",
                self.setup_number, e)

    def _resolve_recording_intent(self, subject_id, camera_id, datetime_now):
        """Decide whether video will actually record this run.

        Returns ``(will_record, cam_id_int, video_info)``. ``video_info``
        is the dict the MCU TSV header embeds so the .tsv knows
        whether a matching video file exists alongside it.
        """
        video_info = {'recorded': False, 'video_name': None, 'video_ts': None}
        will_record_video = False
        cam_id_int = None
        if not subject_id or not (camera_id and self.save_video_enabled):
            return will_record_video, cam_id_int, video_info
        try:
            main_window = self.window()
            if hasattr(main_window, 'video_manager'):
                vm = main_window.video_manager
                cam_id_int = (
                    int(camera_id) if isinstance(camera_id, str)
                    and camera_id.isdigit() else camera_id
                )
                if cam_id_int in vm.cameras:
                    cam_thread = vm.cameras[cam_id_int]
                    if cam_thread.connected:
                        will_record_video = True
                    else:
                        logger.info(
                            f"Box {self.setup_number}: Camera {camera_id} "
                            f"in dict but not yet connected")
        except Exception as e:
            logger.debug(f"Could not check camera status: {e}")
        # The real video filename + extension (.mp4 vs .avi) isn't known
        # until the encoder opens, so we don't guess it here; the TSV video
        # header is written post-start by base.open_session_video_recorder.
        # The returned dict stays the "no video" default, used only when
        # will_record_video is False.
        return will_record_video, cam_id_int, video_info

    # _begin_temp_safety_run is shared on RunTask; _is_camera_streaming is
    # inherited (shared VideoManager.is_camera_streaming). Operant supplies
    # only the writer-open flags via _after_temp_mcu_open.
    def _after_temp_mcu_open(self, ok: bool) -> None:
        self.recording_data = bool(ok)
        self.recording_video = False

    # _collect_run_metadata, _commit_box_sources_for_run and
    # _uninstall_mcu_row_mirror are inherited from RunTask.
    # The MCU-TSV open (+ source commit + FW anchor + history row) is
    # base.open_session_mcu_tsv.

    # The video-recorder build is base.open_session_video_recorder; this
    # Recording geometry (segment size / ROI / fps) is resolved centrally by
    # MainWindowBase._video_recorder_geometry; the operant box supplies only
    # its ROI via _get_box_roi (the base method's operant branch).

    def _build_annotate_callback_if_enabled(self):
        """Honor the per-box ``annotate_saved`` override if set in
        ``_tracking_dialog_globals``."""
        try:
            ov = getattr(self.main_window, "_tracking_dialog_globals", {}) or {}
            ann = (ov.get("annotate_saved") or {})
            if ann.get(str(self.setup_number)) or ann.get(self.setup_number):
                build = getattr(
                    self.main_window, "_build_annotate_callback", None)
                if callable(build):
                    return build(self.setup_number)
        except Exception:
            pass
        return None

    def _cleanup_after_start_failure(self):
        """Close data file + recorder if they were opened; refresh buttons."""
        if self.recording_data:
            self.pycboard.data_logger.close_files()
            self.recording_data = False
        if self.recording_video:
            self.video_recorder.stop_recording()
            self.video_recorder.close()
            self.recording_video = False
        self._update_button_states()

    # on_stop_clicked + startFramework live on RunTask. The stop-confirm gates
    # on run_mode == RECORD; start_framework derives ``record`` from subject
    # presence (run_mode is authoritative).

    def _after_start(self, record: bool) -> None:
        """RunTask hook, operant extras after a successful framework start:
        task_plot.run_start, _register_stats_consumer, and a main_window UI
        refresh. Status / run_timer / button states / framework_started_signal
        are already done by RunTask.mcu_start."""
        if self.main_window and hasattr(self.main_window, "refresh_ui_state"):
            try:
                self.main_window.refresh_ui_state()
            except (AttributeError, RuntimeError):
                pass
        if hasattr(self, 'task_plot') and self.task_plot:
            try:
                from ..plotting import _fw_clock_for
                self.task_plot.run_start(recording=False,
                                         clock=_fw_clock_for(self))
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Box {self.setup_number}: task_plot.run_start error: {e}")
        if self.main_window is not None:
            self.main_window._register_stats_consumer(self.setup_number, self)
        logger.info(f"Box {self.setup_number}: Framework started")

    def stop_framework(self):
        """Stop button → single-stop convergence in RunTask.mcu_stop."""
        if not self.pycboard:
            return
        self.print_to_log("Stopping framework...")
        self.mcu_stop("user")

    def _after_stop(self, reason: str) -> None:
        """RunTask hook, operant cleanup AFTER timers + MCU stop, BEFORE
        state-display clear + status set.  Closes data file, stops video
        recorder, notifies task_plot / statistics, refreshes main_window."""
        if hasattr(self, 'task_plot') and self.task_plot:
            try:
                self.task_plot.run_stop()
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Box {self.setup_number}: task_plot.run_stop error: {e}")
        if self.main_window is not None:
            self.main_window._unregister_stats_consumer(self.setup_number)

        if self.recording_video:
            self.recording_video = False
            if self.main_window and hasattr(self.main_window, 'recording_setups'):
                self.main_window.recording_setups.discard(self.setup_number)
            if self.video_recorder:
                rec = self.video_recorder
                self.video_recorder = None
                # Detach from the RecorderSink FIRST so no more frames flow in,
                # THEN reap the ffmpeg child off the GUI thread. The inline
                # stop_recording()+close() blocks the GUI thread for up to ~2s
                # (thread.join) + the ffmpeg drain per box; delivered serially
                # in the stop coordinator's finalize phase, that serialized
                # multi-stop across all boxes.
                if self.main_window and hasattr(self.main_window, "pipeline"):
                    try:
                        self.main_window.pipeline.stop_recording(self.setup_number)
                    except (AttributeError, RuntimeError):
                        pass
                if self.main_window and hasattr(self.main_window, "_async_stop_recorder"):
                    self.main_window._async_stop_recorder(rec, self.setup_number)
                else:
                    try:
                        rec.stop_recording()
                        rec.close()
                    except (AttributeError, RuntimeError) as e:
                        logger.warning(f"Box {self.setup_number}: Error stopping video: {e}")

        if self.recording_data:
            self.recording_data = False
            if self.pycboard and hasattr(self.pycboard, 'data_logger') and self.pycboard.data_logger:
                try:
                    self.pycboard.data_logger.close_files()
                    logger.info(f"Box {self.setup_number}: Closed data files")
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Box {self.setup_number}: Error closing data files: {e}")

        if self.main_window and hasattr(self.main_window, "refresh_ui_state"):
            try:
                self.main_window.refresh_ui_state()
            except (AttributeError, RuntimeError):
                pass

    # ==================================================================
    # BOX-WIDGET PROTOCOL, surfaces MainWindowBase calls through.
    # Operant's video tile + live-status log live in separate widgets
    # (VideoStreamHolder, LiveStatusWidget) elsewhere in the main_window
    # grid; the forwarders below route protocol calls to those external
    # surfaces so base.py drives operant and maze through one interface.
    # ``get_roi`` and ``camera_id_text`` are inherited from RunTask
    # (identical bodies on both per-box widgets).
    # ==================================================================

    def _stream_holder(self):
        """Return this box's VideoStreamHolder if present, else None."""
        mw = self.main_window
        if mw is None or not hasattr(mw, "_get_cached_stream_widget"):
            return None
        try:
            return mw._get_cached_stream_widget(self.setup_number)
        except Exception:
            return None

    def append_status(self, text, end="\n"):
        """Route status messages (pipeline alarms, zone feedback) to the
        box's LiveStatusWidget log."""
        self.print_to_log(text, end=end)

    def clear_log(self) -> None:
        """Clear this box's LiveStatusWidget log (called at run start)."""
        mw = self.main_window
        widget = (getattr(mw, "_live_status_by_box", None) or {}).get(
            self.setup_number)
        if widget is not None:
            try:
                widget.clearStatus()
            except Exception:
                pass

    def update_frame(self, pixmap):
        holder = self._stream_holder()
        if holder is None:
            return
        try:
            holder.videoWidget.setPixmap(pixmap)
        except Exception as e:
            if "deleted" not in str(e):
                logger.debug("setPixmap for box %s: %s", self.setup_number, e)

    def clear_video(self):
        """Blank the tile after a camera disconnect.

        ``clear_frame`` drops the painted source; setting text alone used to
        leave the last frame on screen, so a disconnected box looked live.
        """
        holder = self._stream_holder()
        if holder is None:
            return
        try:
            holder.updateStatus("Idle")
            widget = holder.videoWidget
            clear_frame = getattr(widget, "clear_frame", None)
            if callable(clear_frame):
                clear_frame()
            widget.setText("No Signal")
        except (RuntimeError, AttributeError):
            pass

    def video_size(self):
        holder = self._stream_holder()
        if holder is None:
            return None
        try:
            return holder.videoWidget.size()
        except Exception:
            return None

    def video_visible(self):
        """Whether this box's tile is on screen, see
        ``MainWindowBase._is_box_tile_visible``. None = unknown, paint anyway."""
        holder = self._stream_holder()
        if holder is None:
            return None
        try:
            return holder.videoWidget.isVisible()
        except Exception:
            return None

    def set_fps_text(self, text, tooltip=""):
        """Operant shows the fps/resolution readout in the VideoStreamHolder
        header. ``text`` already carries the 'fps' unit + resolution."""
        holder = self._stream_holder()
        if holder is None:
            return
        try:
            holder.updateStatus(text)
            if tooltip and hasattr(holder, "setToolTip"):
                holder.setToolTip(tooltip)
        except (AttributeError, RuntimeError):
            pass

    def notify_camera_starting(self, camera_id):
        """Camera connect succeeded; tile not yet streaming."""
        try:
            self.controls_button.setEnabled(True)
            self._set_status("Camera starting...")
        except (AttributeError, RuntimeError):
            pass

    def notify_camera_disconnected(self):
        """Camera went away; lock down the per-box action buttons."""
        try:
            self.record_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.controls_button.setEnabled(False)
            self._set_status("Camera disconnected")
        except Exception:
            pass

    # The run clock is driven by MainWindowBase.process_timer reading
    # pycboard.get_timestamp(); RunTask owns the label.
