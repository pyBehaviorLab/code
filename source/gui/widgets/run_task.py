"""Per-box framework run pipeline, single source of truth.

Both modes (maze ``SetupWidget``, operant ``BoxControlWidget``) inherit
from this class and override ONLY the open/close-recording hooks.

Lifecycle calls (the only entry points the rest of the app uses):

    rt.connect(port)           open serial, status="Ready"
    rt.upload(task)            push task to MCU, status="Uploaded '<task>'"
                               (re-upload is a reset → "Reset '<task>'")
    rt.start(record: bool)     start framework, optionally open recording
    rt.mcu_stop(reason)        stop framework, close recording, status="Stopped"/"Error"
    rt.disconnect()            close serial, status="Disconnected"

Three callers converge on ``mcu_stop``:
    user click on Stop button   reason="user"
    plot_tick saw auto-stop     reason="auto"
    plot_tick caught a board error  reason="error:<msg>"

Stop ALWAYS, in this order:
    1. Flip framework_running off + notify main window so process_timer
       can stop ticking this box.
    2. Send stop to MCU only if user-initiated (auto/error already stopped)
    3. Drain final messages (so logs/data files get the tail)
    4. Subclass _after_stop hook, close data file + close video recorder
    5. Clear stale state_text / event_text / print_text
    6. Status "Stopped" (neutral) or "Error: <msg>" (red, sticky)
    7. Refresh button states
    8. Emit framework_stopped_signal

State owned by this class:
    self.pycboard, self.framework_running, self.task_uploaded,
    self._status_widget, self._timer_label, self.run_mode.

The 1 Hz elapsed clock + 10 ms MCU drain are driven by
MainWindowBase.process_timer (see source/gui/base.py). The clock reads
the MCU framework time (``pycboard.get_timestamp()``) directly, so it
always matches the TSV; ``mcu_stop`` freezes the label at the final
``pycboard.timestamp``.

Subclasses MUST set in _build_ui:
    self._status_widget : QLineEdit  (the canonical 1-line status surface)
    self._timer_label   : QLabel     ("00:00:00")

Subclasses MAY override (small hooks):
    _after_stop(reason)     close data file + video recorder
    _data_consumers()       list of objs with process_data() for Pycboard
    _after_status_change(text, kind)   tab-title icon update
    _after_subject_id_change(subject_id)
    _after_mcu_upload_success(sm_name, task_path)
    _after_upload_clicked(text, rel, sm_name)
    _extra_button_state(connected, task_ready, running, setup_enabled)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from serial import SerialException

from source.communication.pycboard import Pycboard
from source.communication.pyboard import PyboardError
from source.config import task_variables as _tv
from source.gui.box_alerts import ERROR_COLOR as _ERROR_COLOR

logger = logging.getLogger(__name__)


def _noop_log(_msg: str) -> None:
    pass


def _short_cause(exc, limit: int = 48) -> str:
    """One-line, length-bounded cause string for a status label.

    Collapses whitespace and truncates so every "X failed: {cause}" status
    stays readable in the narrow per-box status field, with a single
    truncation policy instead of the ad-hoc ``str(e)[:32/40/80/120]`` that
    was scattered across the start/stop/upload handlers."""
    s = " ".join(str(exc).split()) or exc.__class__.__name__
    return s if len(s) <= limit else s[: limit - 1] + "…"


# Status surface colour palette, used by RunTask.set_status and by
# subclasses that colour their own state-line widgets the same way.
# Convention: running = magenta (framework live, no data file open),
# recording = green (data is being captured to disk), error = red.
STATUS_COLOURS = {
    "neutral":   "#aaaaaa",
    "ready":     "#9fdcff",
    "running":   "#d670d6",
    "recording": "#7ee787",
    # Taken from box_alerts rather than repeated: a box's status text and the
    # red it turns every other surface are one fact, so they share one
    # constant instead of two literals kept in step by hand.
    "error":     _ERROR_COLOR,
}


class RunMode(Enum):
    """Per-box run mode, the ONE authoritative run-state value.

    Exactly one fact decides it: is a subject ID present at Start?

        IDLE, framework not running.
        DRY, running with no subject; data goes to ``data/temp/``
                  (overwritten each dry run), no lineage/history/capture.
        RECORD, running with a subject; a real saved session.

    Owned by ``RunTask``: set at ``mcu_start`` from subject presence, reset
    to ``IDLE`` at stop.  ``framework_running`` is the separate "the MCU
    loop is live" boolean.  Whether a *writer* is open is tracked by the
    subclass writer-open flags (operant ``recording_data`` /
    ``recording_video``): those say a file/encoder exists;
    ``run_mode`` says what the run *is*.
    """
    IDLE   = "idle"
    DRY    = "dry"
    RECORD = "record"


class RunTask:
    """Per-box framework run pipeline.

    Use as a mixin alongside ``QtWidgets.QWidget`` / ``QtWidgets.QFrame``:

        class MyBoxWidget(QtWidgets.QFrame, RunTask):
            framework_started_signal = QtCore.Signal(int)
            framework_stopped_signal = QtCore.Signal(int)

    Signal declarations live on the subclass because PyQt's signal
    metaclass requires them to be class-level on a QObject subclass.
    Declare them on the widget; RunTask emits via ``getattr`` so missing
    signals are silently skipped.
    """

    # Live task-plot redraw cadence: every N active ticks (10 ms each), so
    # 10 ⇒ ~10 Hz. Data capture is unthrottled; only the pyqtgraph redraw is.
    PLOT_REDRAW_EVERY_N_TICKS = 10

    # Class-level default so ``self._log_func(...)`` is callable before
    # init_run_task runs; init replaces it with the real per-box logger.
    # staticmethod so the plain function isn't bound as a method.
    _log_func = staticmethod(_noop_log)

    # ==================================================================
    # INIT
    # ==================================================================

    def init_run_task(self, setup_id: int, main_window: Any = None) -> None:
        """Allocate every per-box attribute. Single init point, no
        scattered state in subclass __init__.

        Idempotent.  Both ``box_id`` and ``box_number`` are set to the
        same value so older call sites in either mode keep working.
        """
        self.setup_id     : int = int(setup_id)
        self.setup_number : int = int(setup_id)
        self.main_window      = main_window

        # MCU state
        self.pycboard          : Optional[Pycboard] = None
        self.framework_running : bool = False
        self.task_uploaded     : bool = False      
        self._uploaded_task_rel : Optional[str] = None
        self.task_file_hash    : Optional[int]  = None
        self.last_task_hash    : Optional[int]  = None

        # Cached sm_info from the last successful upload.
        self._mcu_variables   : dict = {}
        self._mcu_events      : list = []
        self._mcu_coordinates : dict = {}

        # Run mode, the single authoritative run-state value (IDLE/DRY/
        # RECORD). Set at mcu_start from subject presence, reset at stop.
        self.run_mode : RunMode = RunMode.IDLE

        # The HH:MM:SS run clock reads the MCU framework time via
        # ``pycboard.get_timestamp()`` in MainWindowBase._on_process_tick,
        # so the display always matches the TSV timestamps.

        # Subclass MUST assign these in _build_ui before mcu_start runs.
        self._status_widget : Optional[QtWidgets.QLineEdit] = None
        self._timer_label   : Optional[QtWidgets.QLabel]    = None

        # Locks (driven by main_window.refresh_ui_state).
        self.global_setup_locked    : bool = False
        self.global_controls_locked : bool = False

        # Subject metadata cache (populated by main_window's metadata dialog).
        self._subject_metadata : dict = {}

        # User API instance (set by initialise_API when the loaded task
        # declares ``v.api_class``; None otherwise, most tasks).
        self.user_api : Optional[Any] = None

        # Per-box log callable resolved ONCE and stored as a plain attribute
        # (see _resolve_log_func for why this isn't a @property).
        self._log_func = self._resolve_log_func()

    # ==================================================================
    # BOX-WIDGET PROTOCOL, shared surface both BoxControlWidget and
    # SetupWidget expose so MainWindowBase can drive them through one
    # interface. Mode-specific forwarders (append_status, update_frame,
    # clear_video, video_size, set_fps_text, notify_camera_*) live on
    # the widget subclass; the two methods below have identical bodies
    # because both widgets store ROI / camera-id as direct attributes.
    # ==================================================================

    def get_roi(self, frame=None):
        """Return ROI (x, y, w, h) for this box, or None.

        Prefers ``roi_normalized`` (resolution-independent) and scales
        against ``frame``; falls back to ``roi_segment`` (pixel-space
        transient). Shared body, both per-box widgets carry the same
        attribute names.
        """
        roi_norm = getattr(self, "roi_normalized", None)
        if roi_norm and len(roi_norm) == 4 and frame is not None:
            h, w = frame.shape[:2]
            return (
                int(roi_norm[0] * w),
                int(roi_norm[1] * h),
                int(roi_norm[2] * w),
                int(roi_norm[3] * h),
            )
        roi_seg = getattr(self, "roi_segment", None)
        if roi_seg and len(roi_seg) == 4 and roi_seg[2] > 0 and roi_seg[3] > 0:
            return tuple(roi_seg)
        return None

    def camera_id_text(self) -> str:
        """Return stripped camera-id text. Shared body, both per-box
        widgets carry ``camera_id_edit`` as a QLineEdit."""
        edit = getattr(self, "camera_id_edit", None)
        if edit is None:
            return ""
        try:
            return edit.text().strip()
        except Exception:
            return ""

    # ==================================================================
    # STATUS WRITER, the ONLY writer of self._status_widget
    # ==================================================================

    def set_status(self, text: str, kind: str = "neutral") -> None:
        """Update the canonical status QLineEdit.

        ``kind`` selects the colour from ``STATUS_COLOURS``.  Sticky:
        the caller chooses the colour, no auto-downgrade.  Subclasses
        wanting secondary indicators (cumulative log, MCU-state chip)
        use SEPARATE widgets and SEPARATE methods, never write to
        ``self._status_widget`` directly.
        """
        # Unchanged status → nothing to do. This runs from 1 Hz refresh
        # paths; an unconditional setText + setStyleSheet per tick per box
        # forces Qt style recomputes for zero visual change.
        if (getattr(self, "_status_kind", None) == kind
                and getattr(self, "_status_message", None) == text):
            return
        # Remember the kind + text so the per-box error alert (box_alerts.py,
        # via compute_ui_state -> apply_box_alerts) derives from the same
        # sticky status, and the tooltip can show the message.
        self._status_kind = kind
        self._status_message = text
        w = getattr(self, "_status_widget", None)
        if w is None:
            return
        colour = STATUS_COLOURS.get(kind, STATUS_COLOURS["neutral"])
        try:
            w.setText(text)
            from source.gui.theme import THEME as _T
            w.setStyleSheet(
                "QLineEdit {"
                f" background-color: {_T.palette.surface};"
                f" border: 1px solid {_T.palette.surface_border_strong};"
                f" border-radius: {_T.radius.sm}px;"
                " padding: 4px;"
                " font-family: 'Cascadia Mono', 'Consolas', monospace;"
                f" font-size: 10pt; font-weight: 700; color: {colour};"
                "}"
            )
        except Exception as e:
            logger.debug("set_status failed for box %s: %s", self.setup_id, e)
        try:
            self._after_status_change(text, kind)
        except Exception as e:
            logger.debug("after_status_change for box %s: %s", self.setup_id, e)

    def _after_status_change(self, text: str, kind: str) -> None:
        """Subclass hook, e.g. update tab-title icon.  Default no-op."""

    # ==================================================================
    # STICKY PER-BOX ALARM, survives start/stop/connect (unlike status).
    # Set by the recorder/pipeline on a HARD failure (e.g. no GPU encoder
    # session). compute_ui_state ORs ``_box_alarm`` into the per-box
    # ``error`` flag, so the red tile persists until a DELIBERATE user
    # action clears it: reset_task, re-upload, or a successful record.
    # NOT cleared by a plain start, stop, disconnect or reconnect.
    # ==================================================================
    def set_box_alarm(self, msg: str) -> None:
        self._box_alarm = str(msg or "")

    def clear_box_alarm(self) -> None:
        self._box_alarm = ""

    # ==================================================================
    # STATE-DISPLAY CLEAR, wipe MCU state/event/print labels on stop
    # so the next run never opens with the previous run's last line.
    # ==================================================================

    def _clear_state_displays(self) -> None:
        """Clear auto-driven MCU state / event / print line widgets on
        every stop and every error, so stale state names from a previous
        run never linger.
        """
        ti = getattr(self, "task_info", None)
        if ti is None:
            return
        for attr in ("state_text", "event_text", "print_text"):
            w = getattr(ti, attr, None)
            if w is None:
                continue
            try:
                w.setText("")
            except Exception:
                pass

    def _notify_timer_mode(self) -> None:
        """Tell the main window to re-evaluate which timer should fire
        (refresh_timer vs process_timer). Called after framework_running
        flips. Safe no-op if the main window doesn't expose the hook
        (older callers, tests)."""
        mw = self.main_window
        if mw is None:
            return
        sync = getattr(mw, "_sync_timer_mode", None)
        if callable(sync):
            try:
                sync()
            except Exception as e:
                logger.debug("Box %s: _sync_timer_mode error: %s",
                             self.setup_id, e)

    def _set_all_timers(self, txt: str) -> None:
        """Fan a timer value to EVERY surface (box card + Live Status + stats
        table + camera tile) through the main window's single
        ``_apply_box_timer`` path, the SAME path the live tick uses, so the
        start-reset and stop-freeze match the running clock exactly and no
        surface drifts a tick off the others. Falls back to the box-card label
        alone if the main window is unavailable (tests / teardown)."""
        mw = getattr(self, "main_window", None)
        if mw is not None and hasattr(mw, "_apply_box_timer"):
            try:
                mw._apply_box_timer(self, txt)
                return
            except Exception:
                pass
        tl = getattr(self, "_timer_label", None)
        if tl is not None:
            try:
                tl.setText(txt)
            except Exception:
                pass

    # ==================================================================
    # ACTIVE TICK, called once per ``process_timer`` cycle (10 ms) from
    # MainWindowBase._on_process_tick when this box is running.
    # The HH:MM:SS clock advance lives in the central tick too; this
    # widget just exposes ``pycboard`` (for ``get_timestamp()``) +
    # ``_timer_label`` for it to read (frozen at the final fw time on
    # mcu_stop).
    # ==================================================================

    def tick_active(self) -> None:
        """Active-tick body. Three exit paths from one method:
            running normally, return
            MCU sent b'\\x04', self.mcu_stop("auto")
            board / serial error, self.mcu_stop(f"error:{msg[:80]}")
        Transient parse errors are logged + skipped (next tick may succeed).
        """
        # Snapshot pycboard once, disconnect on another thread can null
        # it between the truthy check and the call site.
        pyc = self.pycboard
        if not pyc:
            return
        try:
            was_running = pyc.framework_running
            pyc.process_data()
            auto_stopped = was_running and not pyc.framework_running
        except (Exception, PyboardError) as e:
            # PyboardError subclasses BaseException (not Exception) so
            # it would escape a bare ``except Exception``. Catch both
            # so mcu_stop("error:…") always runs.
            msg = str(e)[:80]
            # Classify by TYPE, not a name-string / "serial" substring. A
            # dead/disconnected port surfaces as SerialException (subclass of
            # OSError) or a raw OSError whose message may not contain "serial".
            # Treat any PyboardError / OSError as a fatal board error → stop.
            if isinstance(e, (PyboardError, OSError)) or "serial" in msg.lower():
                self._log_func(f"Board error: {msg}")
                logger.error("Box %s: tick_active board error: %s",
                             self.setup_id, msg)
                self.mcu_stop(f"error:{msg}")
                return
            logger.warning("Box %s: process_data error: %s", self.setup_id, msg)
            return
        if auto_stopped:
            self.mcu_stop("auto")
            return
        tp = getattr(self, "task_plot", None)
        if tp is not None:
            # Redraw the live task plot at ~10 Hz, not the full 100 Hz tick
            # rate. Data is still captured every tick (pyc.process_data above
            # feeds task_plot.process_data); only the pyqtgraph redraw is
            # throttled.
            self._plot_redraw_tick = getattr(self, "_plot_redraw_tick", 0) + 1
            if self._plot_redraw_tick >= self.PLOT_REDRAW_EVERY_N_TICKS:
                self._plot_redraw_tick = 0
                try:
                    tp.update()
                except Exception:
                    pass
        if self.user_api is not None:
            try:
                self.user_api.plot_update()
            except Exception as e:
                logger.debug("Box %s: user_api.plot_update error: %s",
                             self.setup_id, e)

    # ==================================================================
    # LIFECYCLE: CONNECT / DISCONNECT
    # ==================================================================

    def _data_consumers(self) -> list:
        """Subclass override: list of objects with ``process_data(new_data)``
        for Pycboard to push messages to (e.g. ``TaskInfo``, ``TaskPlot``).
        Default empty list."""
        return []

    def _register_with_main_window(self) -> None:
        """Mirror this box's pycboard into the main_window MCU registry +
        the video pipeline so MCUPusher can attach pose-trigger callbacks."""
        mw = self.main_window
        if mw is None:
            return
        try:
            if hasattr(mw, "mcu") and self.pycboard is not None:
                mw.mcu.register(self.setup_id, self.pycboard)
        except Exception as e:
            logger.debug("MCU register for box %s: %s", self.setup_id, e)
        try:
            if hasattr(mw, "pipeline"):
                mw.pipeline.update_box_pycboard(self.setup_id, self.pycboard)
        except Exception as e:
            logger.debug("update_box_pycboard for box %s: %s", self.setup_id, e)
        # Re-attach rebuilt the push policy from the TrackingConfig (zones,
        # triggers, gates); layer the dialog coord-mapping/trigger tables
        # back on top, they live in GUI state the pipeline can't reach.
        try:
            mw._apply_push_policy(self.setup_id)
        except Exception as e:
            logger.debug("_apply_push_policy for box %s: %s", self.setup_id, e)

    def _unregister_from_main_window(self) -> None:
        mw = self.main_window
        if mw is None:
            return
        try:
            if hasattr(mw, "mcu"):
                mw.mcu.unregister(self.setup_id)
        except Exception as e:
            logger.debug("MCU unregister for box %s: %s", self.setup_id, e)
        try:
            if hasattr(mw, "pipeline"):
                mw.pipeline.update_box_pycboard(self.setup_id, None)
        except Exception as e:
            logger.debug("update_box_pycboard(None) for box %s: %s", self.setup_id, e)

    def mcu_connect(self, port: str,
                    log: Optional[Callable[[str], None]] = None,
                    data_consumers: Optional[Iterable] = None) -> bool:
        """Open a serial connection. Returns True on success.

        Idempotent: returns True immediately if already connected.
        Lower-level, UI code should call ``connect_mcu(mcu_id)``.
        """
        if log is None:
            log = self._log_func
        if self.pycboard is not None:
            return True
        consumers = list(data_consumers) if data_consumers else self._data_consumers()
        try:
            self.pycboard = Pycboard(port, print_func=log, data_consumers=consumers)
            self.task_uploaded = False
            self.framework_running = False
            # Surface silent drain-write failures (port down mid-session) as a
            # per-box banner. Bound to setup_id so it needs no board context.
            # Marshal through QTimer.singleShot so the Qt-widget update always
            # runs on the GUI thread, the drain runs there today, but this
            # stays correct if it ever moves to a worker (degrades to a missed
            # banner rather than a cross-thread crash).
            mw = getattr(self, "main_window", None)
            if mw is not None and hasattr(mw, "_on_box_health"):
                sid = self.setup_id

                def _health_hook(reason, _mw=mw, _sid=sid):
                    QtCore.QTimer.singleShot(
                        0, lambda: _mw._on_box_health(_sid, reason))

                self.pycboard.health_hook = _health_hook
            self._register_with_main_window()
            return True
        except (PyboardError, SerialException) as e:
            log(f"Connect failed (board error): {e}")
            self.pycboard = None
            self._register_with_main_window()  # detach (None pycboard)
            return False
        except Exception as e:
            log(f"Connect failed: {e}")
            self.pycboard = None
            self._register_with_main_window()  # detach (None pycboard)
            return False

    def mcu_disconnect(self,
                       log: Optional[Callable[[str], None]] = None) -> None:
        """Stop framework if running, close pycboard, detach from video.
        Idempotent. Prefer ``self.disconnect()``."""
        if log is None:
            log = self._log_func
        if self.framework_running:
            try:
                self.mcu_stop("user")
            except Exception:
                pass
        else:
            self._notify_timer_mode()
        if self.pycboard is None:
            return
        # Detach from the pusher/pipeline BEFORE closing the serial port,
        # in the reverse order a sink worker could pass the
        # framework_running gate and enqueue onto a dead board's queue.
        self._unregister_from_main_window()
        try:
            self.pycboard.close()
        except Exception as e:
            log(f"Disconnect error: {e}")
        self.pycboard = None
        self.task_uploaded = False
        self.framework_running = False
        # Drop the api instance; it holds a reference to the now-dead
        # pycboard. Next upload re-initialises if the task opts in.
        self.user_api = None

    @property
    def is_connected(self) -> bool:
        return self.pycboard is not None

    # ------------------------------------------------------------------
    # MCU connect / disconnect, shared UI-facing wrappers around
    # mcu_connect / mcu_disconnect. Same body both modes; the per-mode UI
    # extras (button restyle, task-menu refresh, com-field clear) live in
    # the _after_mcu_connected / _after_mcu_disconnected hooks.
    # ------------------------------------------------------------------
    def connect_mcu(self, mcu_id) -> None:
        """Resolve ``mcu_id`` (USB serial number, stable across replug) to a
        device path and open the serial connection. Falls back to treating
        ``mcu_id`` as the device path for projects that only stored a
        ``com_port``; on success the live serial is captured so the next
        autosave makes the binding port-shuffle-immune. A failed connect
        surfaces an error status.
        """
        from source.communication.mcu_ports import (
            device_for_serial, serial_for_device, label_for,
        )
        placeholders = ("--- Select MCU ---", "--- Select COM ---",
                        "", "No USB serial devices")
        if mcu_id in placeholders:
            QtWidgets.QMessageBox.warning(self, "Warning",
                                          "Please select an MCU first")
            return
        log = self._log_func
        device = device_for_serial(mcu_id)
        captured_serial = mcu_id if device else ""
        if device is None:
            # mcu_id wasn't a resolvable serial. If it's a device path (legacy
            # com_port projects: "COMn" / "/dev/tty…") open it directly; else
            # it's a serial we couldn't find → the board isn't plugged in, so
            # don't try to open the serial string as a port (that gives the
            # cryptic "could not open port '<serial>'").
            id_str = (mcu_id or "").strip()
            if id_str.upper().startswith("COM") or id_str.startswith("/dev/"):
                device = id_str
                captured_serial = serial_for_device(device) or ""
            else:
                self.set_status("MCU not found", "error")
                log(f"MCU {mcu_id} not found, replug or check the connection.")
                self._update_button_states()
                return

        self.set_status("Connecting...", "neutral")
        log(f"Connecting to MCU {mcu_id} (at {device})...")
        if not self.mcu_connect(device, log=log):
            self.set_status("Connection failed", "error")
            logger.error("Box %s connection failed", self.setup_id)
            self._update_button_states()
            return

        if captured_serial:
            self._mcu_serial = captured_serial
        if hasattr(self, "com_id_edit"):
            self.com_id_edit.setText(label_for(captured_serial or mcu_id))
        self.task_file_hash = None
        self._after_mcu_connected()
        self._update_button_states()
        mw = self.main_window
        if mw is not None:
            if hasattr(mw, "refresh_ui_state"):
                mw.refresh_ui_state()
            if hasattr(mw, "_project_changed"):
                mw._project_changed(reason="box_wiring_changed")
        self.set_status("Connected", "ready")

    def disconnect_mcu(self) -> None:
        """Close the MCU serial connection. Stops any live recorder first,
        via the mode-agnostic ``_stop_recording_for_box``, so a disconnect
        mid-recording can't orphan an ffmpeg child.
        """
        self.mcu_disconnect(log=self._log_func)
        mw = self.main_window
        if mw is not None and hasattr(mw, "_stop_recording_for_box"):
            try:
                mw._stop_recording_for_box(self.setup_id)
            except Exception as e:
                logger.debug("Box %s: disconnect recorder stop failed: %s",
                             self.setup_id, e)
        self.task_file_hash = None
        self._after_mcu_disconnected()
        self._update_button_states()
        if mw is not None and hasattr(mw, "refresh_ui_state"):
            mw.refresh_ui_state()
        self.set_status("Not connected", "neutral")

    def _after_mcu_connected(self) -> None:
        """Per-mode UI after a successful connect. Default: nothing."""

    def _after_mcu_disconnected(self) -> None:
        """Per-mode UI after a disconnect. Default: nothing."""

    def _uninstall_mcu_row_mirror(self) -> None:
        """Detach the McuRowMirror consumer from ``pyc.data_consumers``.
        Called by ``base._stop_recording_for_box`` on session end so the
        mirror doesn't accumulate every run. Idempotent.
        """
        try:
            mirror = getattr(self, "_mcu_row_mirror", None)
            pyc = self.pycboard
            if (mirror is not None and pyc is not None
                    and getattr(pyc, "data_consumers", None)):
                try:
                    pyc.data_consumers.remove(mirror)
                except ValueError:
                    pass
        except Exception:
            pass
        self._mcu_row_mirror = None

    # ==================================================================
    # LIFECYCLE: UPLOAD
    # ==================================================================

    def mcu_upload_task(self, task_name: str,
                        hw_def_path: Optional[str] = None,
                        log: Optional[Callable[[str], None]] = None,
                        *,
                        run_ui_hooks: bool = True,
                        initialise_api: bool = True) -> bool:
        """Upload (or reset) task on MCU.  ``task_name`` may be
        ``"task"`` or ``"subfolder/task"`` (with or without ``.py``).
        Resets vs full upload: when ``self.task_uploaded`` is already
        True, ``setup_state_machine`` is called with ``uploaded=True``,
        which re-initialises the state machine on the MCU without
        re-flashing the firmware.

        Threading note:
            When driving multi-box uploads in parallel, call with
            ``log=_noop_log`` and ``run_ui_hooks=False`` so the worker
            thread does not touch Qt widgets (status lines, task plots,
            etc). UI hooks can then be run on the main thread.
        """
        if log is None:
            log = self._log_func
        if not self.pycboard:
            return False
        task_rel = Path(str(task_name).strip())
        if task_rel.suffix == ".py":
            task_rel = task_rel.with_suffix("")
        sm_name = task_rel.name
        sm_dir = (Path("tasks") / task_rel.parent).as_posix()
        # Canonical relname: full POSIX path under tasks/, no .py.
        #   "5CSRTT/stage1"       (nested → family = "5CSRTT")
        #   "calibration_check"   (loose → family = "")
        canonical_rel = task_rel.as_posix()
        try:
            if hw_def_path:
                try:
                    self.pycboard.load_hardware_definition(hw_def_path)
                except (PyboardError, SerialException) as e:
                    log(f"Load hw def failed (board error): {e}")
                    return False
                except Exception as e:
                    log(f"Load hw def failed: {e}")
                    return False
            self.pycboard.setup_state_machine(
                sm_name, sm_dir=sm_dir, uploaded=self.task_uploaded,
            )
            self.task_uploaded = True
            self._uploaded_task_rel = canonical_rel
            # Re-upload / reset-via-upload is a deliberate user action → clear
            # any sticky recorder alarm (set string only; the red repaint
            # happens on the GUI thread's next refresh_ui_state).
            self.clear_box_alarm()
            sm = getattr(self.pycboard, "sm_info", None)
            if sm:
                self._mcu_variables = dict(sm.variables or {})
                self._mcu_events = list(sm.events or [])
                self._mcu_coordinates = dict(sm.coordinates or {})
            task_path = Path(sm_dir) / (sm_name + ".py")
            try:
                from source.config.hashing import djb2_int_from_file as _djb2_file
                self.task_file_hash = _djb2_file(str(task_path))
                self.last_task_hash = getattr(sm, "task_hash", None) if sm else None
            except Exception:
                self.task_file_hash = None
                self.last_task_hash = None
            if hw_def_path:
                self._hw_def_path = hw_def_path
                try:
                    from source.config.hd_parser import parse_hardware_definition
                    self._hw_config = parse_hardware_definition(hw_def_path)
                except Exception as parse_err:
                    self._hw_config = None
                    log(f"HD parsed-cache update failed: {parse_err}")
            log(f"Task uploaded: {sm_name}")
            # Snapshot capture: djb2-hash the task .py (and HD, if provided),
            # stage the raw bytes to <project>/_pending/, update the cfg's
            # BoxConfig caches so the next Record click can commit + build the
            # snapshot from memory.
            self._capture_uploaded_sources(task_path,
                                           hw_def_path=hw_def_path)

            # Push hw_* and restore last-session PERSISTENT values. The MCU's
            # ``import task_file`` already ran inside setup_state_machine, so
            # RESET (non-persistent) vars are at their task-file defaults; we
            # only touch hw_ and persistent. See
            # ``source/config/task_variables.py``.
            self._apply_setup_task_pushes(log=log)

            if run_ui_hooks:
                try:
                    self._after_mcu_upload_success(sm_name, task_path)
                except Exception as e:
                    logger.debug("after_mcu_upload_success for box %s: %s",
                                 self.setup_id, e)
            # Bring up the per-task user API (only if the task opts in
            # via ``v.api_class``). Must run AFTER setup_state_machine
            # populates pycboard.sm_info so the Api can read variables.
            if initialise_api:
                try:
                    self.initialise_API(log=log)
                except Exception as e:
                    logger.debug("initialise_API for box %s: %s",
                                 self.setup_id, e)
            return True
        except (PyboardError, SerialException) as e:
            err = str(e)[:200]
            log(f"Upload failed (board error): {err}")
            return False
        except Exception as e:
            log(f"Upload failed: {e}")
            return False

    def _drop_user_api(self) -> None:
        """Remove the current task's api instance from data_consumers (no-op
        when none is set)."""
        if self.user_api is not None and self.pycboard is not None:
            try:
                self.pycboard.data_consumers.remove(self.user_api)
            except (ValueError, AttributeError):
                pass
        self.user_api = None

    def initialise_API(self, log: Optional[Callable[[str], None]] = None) -> None:
        """Import and instantiate the task's user API class if declared.

        Reads ``self.pycboard.sm_info.variables['api_class']``. If absent,
        the call is a no-op, most tasks don't use this. On success the
        api instance is inserted at the FRONT of ``pycboard.data_consumers``
        so it sees data before display consumers (TaskInfo, TaskPlot,
        StatisticsDataConsumer)."""
        if log is None:
            log = self._log_func
        self._drop_user_api()

        if self.pycboard is None:
            return
        sm_info = getattr(self.pycboard, "sm_info", None)
        if sm_info is None:
            return
        variables = getattr(sm_info, "variables", None) or {}
        api_name = variables.get("api_class")
        if not api_name:
            return  # task did not opt in
        module_name = f"api_classes.{api_name}"
        try:
            import importlib
            user_module = importlib.import_module(module_name)
            importlib.reload(user_module)
        except ModuleNotFoundError:
            log(f"Could not find user API module: {module_name}")
            return
        except Exception as e:
            log(f"Failed to import {module_name}: {e}")
            return
        api_class = getattr(user_module, api_name, None)
        if api_class is None:
            log(f"Could not find user API class '{api_name}' in {module_name}")
            return
        try:
            api_instance = api_class()
            api_instance.interface(self.pycboard, log)
            self.pycboard.data_consumers.insert(0, api_instance)
            self.user_api = api_instance
            log(f"Initialised API: {api_name}")
        except Exception as e:
            log(f"Unable to initialise API '{api_name}': {e}")
            logger.error("initialise_API box %s: %s", self.setup_id, e)

    def _after_mcu_upload_success(self, sm_name: str, task_path: Path) -> None:
        """Subclass hook: extras after a successful upload (operant uses
        this to pre-create session dirs + load stats config). Default no-op."""

    def _capture_uploaded_sources(self, task_path: Path,
                                  *, hw_def_path: Optional[str] = None) -> None:
        """Stage the just-uploaded task (and optionally HD) via the
        main_window's SnapshotStore, then update the cfg BoxConfig caches
        so the next Record click can commit + build a snapshot with no I/O.

        Silent no-op when no SnapshotStore is present (e.g. no project
        loaded), recording still works, just without trace-back.
        """
        mw = self.main_window
        if mw is None:
            return
        store = getattr(mw, "_snapshot_store", None)
        cfg   = getattr(mw, "_active_config", None)
        if store is None or cfg is None:
            return
        # The task FileRef is run-only, stored on the widget, never on cfg.
        # The snapshot store captures the .py and commit_box_sources promotes
        # pending → source/<djb2>.py at record-start.
        try:
            ref = store.capture_source(task_path,
                                       kind="task",
                                       setup_id=self.setup_id,
                                       label=task_path.name)
            if ref is not None:
                # Stash on the widget so the recorder / runs writer can
                # read it without touching cfg.
                self._task_ref = ref
        except Exception as e:
            logger.warning("snapshot task capture (box %s): %s", self.setup_id, e)
        if hw_def_path:
            try:
                hd_ref = store.capture_source(hw_def_path,
                                              kind="hw_def",
                                              setup_id=self.setup_id,
                                              label=Path(hw_def_path).name)
                if hd_ref is not None:
                    # Update the project default (setup_config.boxes[i].init_hw_def)
                    # so the next session's Setup dialog reflects the latest pick.
                    box_cfg = next(
                        (b for b in cfg.setup_config.boxes
                         if b.setup_number == self.setup_id),
                        None,
                    )
                    if box_cfg is not None:
                        box_cfg.init_hw_def = hd_ref
            except Exception as e:
                logger.warning("snapshot hw_def capture (box %s): %s",
                               self.setup_id, e)
        # Capture the device drivers the MCU holds (same file set as the .tsv
        # ``devices`` header). Ungated from hw_def_path: a task upload passes
        # hw_def_path=None (HD already on the board), but the MCU still holds
        # the drivers. Idempotent via djb2 dedup.
        self._capture_device_snapshots(store)

    # ==================================================================
    # UPLOAD BUTTON HANDLER, shared
    # ==================================================================

    _TASK_PLACEHOLDERS = ("", "--- Select Task ---", "No tasks found")

    def on_upload_clicked(self) -> None:
        """Per-box Upload/Reset button handler.  Single source of truth
        for both modes; mode-specific extras live in three small hooks
        (``_upload_log``, ``_upload_brief_status``, ``_after_upload_clicked``)."""
        log = self._upload_log
        short = self._upload_brief_status
        # Capture the verb up front (it must be right in every status line,
        # including the exception handlers): once a task is on the board the
        # button reads "Reset" and re-imports the task, otherwise "Upload".
        is_reset = bool(self.task_uploaded)
        verb = "Reset" if is_reset else "Upload"
        sm_name = "task"
        try:
            if not self.pycboard:
                log("Not connected - cannot upload")
                short("Not connected", error=True)
                return
            if not self.pycboard.status.get("framework"):
                log("Framework not loaded - use Config first")
                short("Load framework", error=True)
                return
            text = self._upload_task_text()
            if not text or text in self._TASK_PLACEHOLDERS:
                log("No task selected")
                short("Select task", error=True)
                return
            rel = text[:-3] if text.endswith(".py") else text
            sm_name = Path(rel).name
            if is_reset:
                log(f"Resetting task '{sm_name}'...")
                short(f"Resetting '{sm_name}'…")
            else:
                log(f"Uploading task '{sm_name}'...")
                short(f"Uploading '{sm_name}'…")
            # Pass the per-widget ``_hw_def_path`` so the snapshot store
            # captures the HD into ``<project>/source/<djb2>.py`` and
            # ``box_cfg.init_hw_def`` updates to the resolved FileRef, the
            # mechanism that records "this run used this HD" by hash.
            hd_path = getattr(self, "_hw_def_path", None)
            ok = self.mcu_upload_task(rel, log=log, hw_def_path=hd_path)
            if not ok:
                short(f"{verb} failed", error=True)
                return
            if is_reset:
                log(f"Task '{sm_name}' reset, variables restored to defaults")
                self._post_upload_success_ui(
                    text, rel,
                    status=f"Reset '{sm_name}', variables restored")
            else:
                log(f"Task '{sm_name}' uploaded successfully")
                self._post_upload_success_ui(text, rel)
        except (PyboardError, SerialException) as e:
            log(f"{verb} of '{sm_name}' failed (board error): {str(e)[:200]}")
            short(f"{verb} failed: {_short_cause(e)}", error=True)
            logger.error("%s board error for box %s: %s", verb, self.setup_id, e)
        except Exception as e:
            if "setup_state_machine" in str(e):
                log(f"{verb} failed on '{sm_name}': state-machine setup error")
                short(f"{verb} failed: state-machine setup", error=True)
            else:
                log(f"{verb} error on '{sm_name}': {str(e)[:120]}")
                short(f"{verb} failed: {_short_cause(e)}", error=True)
            logger.error("%s for box %s: %s", verb, self.setup_id, e)

    def _post_upload_success_ui(self, raw_text: str, rel: str, *,
                                status: Optional[str] = None) -> None:
        """Main-thread tail after a successful task upload, the ONE place
        the button flip / status colour / button-state refresh / subclass
        hook run, shared by the per-box Upload button and the parallel
        Upload-All dialog so the two paths can't drift. The board-side
        work (setup_state_machine, snapshots, variable pushes) has
        already happened."""
        sm_name = Path(rel).name
        btn = getattr(self, "upload_button", None)
        if btn is not None:
            btn.setText("Reset")
        # 'ready' (light blue) distinguishes "task loaded, idle" from
        # neutral grey / running magenta / recording green / error red.
        self.set_status(status or f"Uploaded '{sm_name}'", "ready")
        self._update_button_states()
        # Subclass hook, may load stats config etc. May genuinely
        # fail (file I/O); keep narrow exception handling.
        try:
            self._after_upload_clicked(raw_text, rel, sm_name)
        except Exception as e:
            logger.debug("after_upload_clicked for box %s: %s", self.setup_id, e)

    def _upload_log(self, msg: str) -> None:
        """Subclass hook, route a verbose log message.  Default uses _log_func."""
        self._log_func(msg)

    def _upload_brief_status(self, text: str, error: bool = False) -> None:
        """Subclass hook, set a brief status (default uses set_status)."""
        self.set_status(text, "error" if error else "neutral")

    def _upload_task_text(self) -> str:
        combo = getattr(self, "task_combo", None)
        if combo is None:
            return ""
        if hasattr(combo, "text"):
            return combo.text()
        if hasattr(combo, "currentText"):
            return combo.currentText()
        return ""

    def _after_upload_clicked(self, raw_text: str, rel: str, sm_name: str) -> None:
        """Post-upload tail. Default loads the task's stats config into the
        live canvas (shared by both modes); maze also syncs its task picker."""
        self._load_stats_config_for_task(raw_text)

    def _load_stats_config_for_task(self, raw_text: str) -> None:
        """Load ``tasks/<x>/config.json`` into the live stats canvas for this
        box so the statistics window populates the moment the framework
        starts. No-op when the stats canvas hasn't been created yet."""
        mw = getattr(self, "main_window", None) or self.window()
        stab = getattr(mw, "statisticsTab", None)
        if stab is None:
            return
        try:
            stab.loadConfigFromTask(self.setup_number, raw_text)
        except Exception as e:
            logger.warning("Box %s: stats loadConfigFromTask error: %s",
                           self.setup_number, e)

    def upload_task(self, task_name: str) -> None:
        """Programmatic upload, used by the Universal Upload dialog.

        ``NestedMenu.setText()`` doesn't fire the picker callback, so the
        "task changed -> reset task_uploaded" path is skipped. The explicit
        ``on_task_changed()`` here forces it, else ``setup_state_machine`` would
        get ``uploaded=True`` and re-import the previous task_file.py.
        """
        combo = getattr(self, "task_combo", None)
        if combo is not None and hasattr(combo, "setText"):
            combo.setText(task_name)
        self.on_task_changed()
        if hasattr(self, "upload_button") and hasattr(self.upload_button, "click"):
            self.upload_button.click()
            return
        self.on_upload_clicked()

    def on_task_changed(self) -> None:
        """Reset upload state on every task selection.

        Idempotent and unconditional: every call resets ``task_uploaded``,
        flips the upload button back to "Upload", and clears the cached api
        instance. Picking a task ALWAYS puts the widget into the "needs
        upload" state.
        """
        logger.debug("Box %s: task changed, resetting upload state", self.setup_id)
        self.task_uploaded = False
        self.task_file_hash = None
        # Drop the previous task's api instance; the next upload will
        # rebuild it (or leave it None if the new task doesn't opt in).
        self._drop_user_api()
        if hasattr(self, "upload_button"):
            self.upload_button.setText("Upload")
        self._update_button_states()
        # Master buttons (operant multi Upload/Start, maze master row) live
        # on the main window and only re-gate via refresh_ui_state; notify it
        # so they don't lag a task change. Per-box state is already done above.
        mw = self.main_window
        if mw is not None and hasattr(mw, "refresh_ui_state"):
            try:
                mw.refresh_ui_state()
            except Exception as e:
                logger.debug("Box %s: refresh_ui_state on task change: %s",
                             self.setup_id, e)

    def _on_task_selected(self, task_text=None) -> None:
        """``task_combo`` (NestedMenu) callback when a task is picked → reset
        the upload state via ``on_task_changed`` (which refreshes buttons).
        Shared by both modes."""
        self.on_task_changed()

    def _refresh_task_menu(self) -> None:
        """Rebuild the ``task_combo`` NestedMenu from the ``tasks/`` directory.
        Shared: both modes expose ``task_combo``."""
        try:
            from source import paths as app_paths
            self.task_combo.update_menu(app_paths.tasks_dir)
        except Exception as e:
            logger.debug("Box %s: refresh task menu failed: %s",
                         getattr(self, "setup_id", "?"), e)

    # ==================================================================
    # LIFECYCLE: START, single entry for both Record and Start
    # ==================================================================

    def _run_mode_for_start(self) -> RunMode:
        """The run mode this start represents, driven solely by subject
        presence (the one fact in the canonical workflow). Subject set →
        RECORD (real saved session); empty → DRY (temp-only safety run).
        Both GUIs expose ``subject_id_edit``; absent → DRY."""
        edit = getattr(self, "subject_id_edit", None)
        has_subject = bool(edit is not None and edit.text().strip())
        return RunMode.RECORD if has_subject else RunMode.DRY

    # ==================================================================
    # RECORD / STOP, ONE shared orchestration path (both GUIs)
    # ------------------------------------------------------------------
    # ``on_record_clicked`` / ``on_stop_clicked`` / ``start_framework`` and the
    # mode-agnostic helpers below live here so operant (``BoxControlWidget``)
    # and maze (``SetupWidget``) drive the SAME control flow. The only
    # per-mode bits are the same-named hooks: ``_start_recording``
    # (build the one MCU TSV + one video recorder + one tracking writer the
    # mode's way), ``_begin_temp_safety_run`` (the no-subject DRY path),
    # ``_post_record_start``, ``_cleanup_after_start_failure``,
    # ``_post_stop_button_restore``, ``_after_stop``. Everything else is
    # shared: one run-state, one stop-confirm, one error wrapper, one DRY
    # vs RECORD decision (subject presence).
    # ==================================================================

    def on_record_clicked(self) -> None:
        """Record/Start click, the single orchestrator for both GUIs.

        Subject ID present → RECORD (real saved session); empty → DRY RUN
        (temp-only safety net, nothing tracked). Synchronous on the GUI
        thread by design (Qt timer ops must stay on-thread).
        """
        try:
            if not self._record_preflight():
                return
            subject_id = self.subject_id_edit.text().strip()
            self._reset_live_status_for_box()
            # MASTER TIMESTAMP, one datetime for every file of this run.
            datetime_now = datetime.now()
            if subject_id:
                metadata = self._collect_run_metadata(subject_id)
                # Opens the MCU TSV (+ commits sources / history / FW anchor)
                # and the one video recorder + tracking writer the mode's way.
                # Raises on a data-file open failure so the framework never
                # starts on an un-recordable run.
                self._start_recording(subject_id, datetime_now, metadata)
            else:
                self._begin_temp_safety_run(datetime_now)
            # mcu_start already styled the status + refreshed buttons; on
            # failure, roll back whatever opened.
            if not self.start_framework():
                self._cleanup_after_start_failure()
                return
            self._post_record_start(subject_id)
        except (PyboardError, SerialException) as e:
            self.set_status(f"Start failed: {_short_cause(e)}", "error")
            logger.error("Box %s: Start board error: %s", self.setup_number, e)
        except Exception as e:
            self.set_status(f"Start failed: {_short_cause(e)}", "error")
            logger.error("Box %s: Failed to start: %s", self.setup_number, e)

    def on_stop_clicked(self) -> None:
        """Stop click, single convergence on ``mcu_stop`` for both GUIs.

        A REAL recording (``run_mode == RECORD``) prompts for confirmation
        so a tracked session isn't ended by accident; a DRY run (nothing
        saved) stops immediately.
        """
        try:
            if not self.pycboard:
                return
            if not self.framework_running:
                self.set_status("Not running", "neutral")
                return
            if self.run_mode == RunMode.RECORD:
                reply = QtWidgets.QMessageBox.question(
                    self, "Stop recording",
                    "Box {} is recording. Stop the session now?".format(
                        self.setup_number),
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            # Stop is confirmed (or was a dry run), fire the mode hook for any
            # immediate UX (operant disables the button + posts a LiveStatus
            # line).
            self._on_stop_confirmed()
            self.set_status("Stopping...", "neutral")
            # Single-thread stop convergence: mcu_stop stops timers, sends
            # the MCU stop, drains data, runs _after_stop (close files/video),
            # clears state, sets "Stopped", refreshes buttons.
            self.mcu_stop("user")
            self._post_stop_button_restore()
        except (PyboardError, SerialException) as e:
            self.set_status(f"Stop failed: {_short_cause(e)}", "error")
            logger.error("Box %s: Stop board error: %s", self.setup_number, e)
        except Exception as e:
            self.set_status(f"Stop failed: {_short_cause(e)}", "error")
            logger.error("Box %s: Failed to stop: %s", self.setup_number, e)

    def start_framework(self) -> bool:
        """Run the framework start. Returns True on success.

        ``record`` is derived from subject presence, the same signal
        ``mcu_start`` turns into the authoritative ``run_mode``. It only
        flows to the ``_after_start`` hook; ``run_mode`` owns the real
        state, so the value is mode-consistent here.
        """
        self.print_to_log("Starting framework...")
        ok = self.mcu_start(
            record=bool(self.subject_id_edit.text().strip()))
        if ok:
            self.print_to_log("Framework started successfully")
        return ok

    def _record_preflight(self) -> bool:
        """Validate inputs + check not already running. False on fail.

        ``_validate_pre_run`` is a hook: operant validates task/subject;
        maze's default accepts."""
        ok, errors = self._validate_pre_run()
        if not ok:
            self.set_status(
                (errors[0][:32] if errors else "Validation failed"), "error")
            return False
        if self.framework_running:
            self.print_to_log("Framework already running")
            self.set_status("Already running", "error")
            return False
        return True

    def _reset_live_status_for_box(self) -> None:
        """Clear this box's live-status log for a fresh run. Mode-agnostic:
        delegates to the ``clear_log()`` box-widget hook so each mode clears
        its OWN log surface (operant's LiveStatusWidget, maze's status_text),
        no reach-in to an operant-only MainWindow list."""
        try:
            self.clear_log()
        except Exception as e:
            logger.debug("clear_log for box %s failed: %s",
                         getattr(self, "setup_number", "?"), e)

    def clear_log(self) -> None:
        """Clear this box's live-status log. Overridden per mode:
        BoxControlWidget clears its LiveStatusWidget, SetupWidget clears its
        status_text. Default no-op so any RunTask host is safe."""

    def _collect_run_metadata(self, subject_id) -> dict:
        """Merge sidebar fields (experimenter/project) + the cohort row
        (genotype/sex/DOB…, keyed by SetupID == box_number) into one dict
        for the TSV header. Shared by both GUIs; the per-mode
        ``_start_recording`` may add its own extra keys."""
        metadata: dict = {}
        try:
            # Prefer the stored MainWindow ref, a detached maze SetupWidget's
            # window() is its own dialog, not the MainWindow that owns the
            # sidebar fields + metadata_manager.
            mw = self.main_window or self.window()
            if hasattr(mw, "info_fields"):
                experimenter = mw.info_fields.get("experimenter")
                project = mw.info_fields.get("project")
                if experimenter:
                    metadata["experimenter"] = experimenter.text().strip()
                if project:
                    metadata["project"] = project.text().strip()
            if hasattr(mw, "metadata_manager"):
                row = mw.metadata_manager.row_for_setup(self.setup_number)
                if row:
                    for k, v in row.items():
                        if k in ("Subject", "SetupID"):
                            continue
                        if v != "":
                            metadata[k] = v
                    logger.info("Box %s: cohort row attached (%d fields)",
                                self.setup_number, len(metadata))
        except Exception as e:
            logger.warning("Box %s: Could not load metadata: %s",
                           self.setup_number, e)
        return metadata

    # --- Per-mode hooks for the shared record/stop path (default bodies) ---

    def _validate_pre_run(self):
        """Hook: return ``(ok, errors)``. Default accepts. Operant overrides
        with task/subject checks."""
        return True, []

    def _start_recording(self, subject_id, datetime_now, metadata):
        """Hook: open the MCU TSV + the one video recorder + tracking
        writer for a RECORD run (subject present). Must raise on a
        data-file open failure so the framework never starts on an
        un-recordable run. Subclasses implement."""
        raise NotImplementedError

    def _begin_temp_safety_run(self, datetime_now) -> None:
        """No-subject DRY Start: write the MCU TSV (+ tracking text + a fixed
        overwritten-stem video when a camera streams) to ``data/temp/`` so a
        forgotten-subject session isn't lost. Shared by both modes; the
        per-mode bits are the ``_is_camera_streaming`` /
        ``_after_temp_mcu_open`` hooks below.
        """
        mw = self.main_window
        setup_id = self.setup_number
        ok = self.open_temp_mcu_data_logger(datetime_now)
        self._after_temp_mcu_open(ok)
        if ok:
            logger.info(
                "Box %s: no subject, MCU saved to temp/Box%s.tsv "
                "(overwritten each run)", setup_id, setup_id)
        if not self._is_camera_streaming():
            return
        try:
            vd = str(self.temp_video_data_path())
            tw = mw.open_session_tracking_writer(
                setup_id, file_path=vd,
                video_filename=self.temp_video_path().name)
            # Dry run records video too, same central builder, temp dir
            # + a fixed overwritten stem.
            mw.open_session_video_recorder(
                setup_id, str(self.temp_video_path().parent), "",
                datetime_now, file_stem=f"video_Box{setup_id}",
                tracking_writer=tw)
        except Exception as e:
            logger.warning("Box %s: temp tracking/video writer failed: %s",
                           setup_id, e)

    def _is_camera_streaming(self) -> bool:
        """True iff this box's camera is streaming, resolved via the shared
        VideoManager (``box_camera_map`` → connected + has-frames), so the DRY
        temp run opens tracking + video only when frames actually flow. ONE
        path for both modes."""
        try:
            vm = getattr(self.main_window, "video_manager", None)
            return bool(vm and vm.is_camera_streaming(self.setup_number))
        except Exception:
            return False

    def _after_temp_mcu_open(self, ok: bool) -> None:
        """Hook: per-mode bookkeeping after the temp MCU TSV open attempt
        (operant sets its writer-open flags; maze stamps _recording_ctx)."""

    def _post_record_start(self, subject_id) -> None:
        """Hook: bookkeeping after a successful framework start
        (session bookkeeping, tracking enable, UI refresh).
        Subclasses implement."""

    def _cleanup_after_start_failure(self) -> None:
        """Hook: close any data file / recorder opened before a failed
        framework start, then refresh buttons. Subclasses implement."""

    def _on_stop_confirmed(self) -> None:
        """Hook: immediate UX the instant a stop is confirmed, BEFORE the
        synchronous mcu_stop. Operant disables the Stop button + posts a
        LiveStatus "Stopping…" line; maze no-op."""

    def _post_stop_button_restore(self) -> None:
        """Hook: post-stop button re-enable. Operant resets its per-widget
        setup-lock flags; maze relies on mcu_stop's refresh (no-op)."""

    def _after_start(self, record: bool) -> None:
        """Subclass hook: extras after a successful framework start.
        Called AFTER status was set + button states refreshed + start
        signal emitted, so the subclass can attach things that depend
        on the framework being live (e.g. statistics tab, task_plot)."""

    def _after_stop(self, reason: str) -> None:
        """Subclass hook: close data file + stop video recorder.
        ALWAYS called on stop, even on error.  Default no-op."""

    def reset_task(self) -> bool:
        """Wipe every ``v.*`` on the MCU back to its task-module default.

        Mechanism: ``pycboard.setup_state_machine(uploaded=True)`` does a
        soft-reboot of the MCU (which clears ``utility.v`` entirely) and
        then re-imports ``task_file.py``, that re-runs every module-level
        ``v.foo = X`` line. Costs ~1-2 s per box because of the soft-reboot.

        Idempotent: returns True on success, False if no task is uploaded
        or the MCU call raises.
        """
        if not self.pycboard or not self.task_uploaded:
            return False
        task_rel = (self._uploaded_task_rel or "").strip()
        if not task_rel:
            return False
        sm_name = Path(task_rel).name
        sm_dir = (Path("tasks") / Path(task_rel).parent).as_posix()
        try:
            self._log_func(f"Resetting task '{sm_name}' (wipe v.*)…")
            self.pycboard.setup_state_machine(
                sm_name, sm_dir=sm_dir, uploaded=True,
            )
            self._log_func(f"Task '{sm_name}' reset, v.* back to module defaults")
            # Re-apply hw_ + persistent pushes so a Reset doesn't silently drop
            # them, pairs the wipe with the same pushes the Upload path does.
            self._apply_setup_task_pushes(log=self._log_func)
            self.clear_box_alarm()   # deliberate reset clears the sticky alarm
            return True
        except (PyboardError, SerialException) as e:
            self._log_func(f"Reset failed (board error): {str(e)[:120]}")
            logger.error("Box %s: reset_task board error: %s",
                         self.setup_id, e)
            return False
        except Exception as e:
            self._log_func(f"Reset failed: {str(e)[:120]}")
            logger.error("Box %s: reset_task error: %s", self.setup_id, e)
            return False

    def _box_variable_specs(self):
        """Resolve this box's effective per-variable specs: the per-task
        template overlaid by the loaded project's own persistent flags.

        Returns the spec list, or None when no task is uploaded. Per-variable
        ``persistent`` flag drives the restore loop in
        ``_apply_setup_task_pushes`` (at Upload) and ``_capture_persistent_at_stop``
        (at Stop).
        """
        task_rel = (self._uploaded_task_rel or "").strip()
        if not task_rel:
            return None
        mw = self.main_window
        pd = getattr(mw, "_active_project_dir", None) if mw else None
        task = self._persistent_task_folder()
        try:
            task_py = _tv.task_py_path_from_relname(task_rel)
            return _tv.resolve_specs(task_py, pd, task)
        except Exception as e:
            logger.debug("Box %s: spec resolve failed: %s", self.setup_id, e)
            return []

    def _persistent_task_folder(self) -> str:
        """Folder name used under ``<project>/`` for this task's
        ``persistent_variables.json``: ``task_family`` when set, else the
        task name itself.

        Returns "" when no task is uploaded; the persistent helpers
        treat that as "no-op".
        """
        fam = self._task_family()
        if fam:
            return fam
        return self._resolve_task_name() or ""

    def _apply_setup_task_pushes(self,
                                 log: Optional[Callable[[str], None]] = None,
                                 hw_prompt_values: Optional[dict] = None) -> None:
        """Setup-task tail, run at Upload, never at Start. Applies
        irrespective of dry-run vs Record (Upload precedes that choice).

        Composes ONE push dict from two sources, then writes it in a SINGLE
        ``set_variables`` round-trip:

          1. ``apply_pre_run_hw``: hw_* from rig store
          2. ``restore_pers_vars``: last-session value of each PERSISTENT
             variable, by name (project + task known). RESET vars keep the
             task-file default the MCU already holds.

        Persistent wins on key conflict with hw_. No-op when the MCU isn't
        ready.
        """
        if log is None:
            log = self._log_func
        if self.pycboard is None:
            return
        specs = self._box_variable_specs() or []
        mw    = self.main_window
        pd    = getattr(mw, "_active_project_dir", None) if mw else None
        task  = self._persistent_task_folder()

        push: dict = {}
        variables_set_pre_run = []

        # 1. hw_*
        try:
            hw_result = _tv.apply_pre_run_hw(
                self.pycboard, specs, hw_prompt_values=hw_prompt_values)
            push.update(hw_result.pushed)
            variables_set_pre_run.extend(hw_result.set_lines)
        except Exception as e:
            logger.warning("Box %s: apply_pre_run_hw failed: %s",
                           self.setup_id, e)

        # 2. Persistent restore, by name, per project + task
        if pd is not None and task:
            try:
                pv_result = _tv.restore_pers_vars(
                    self.pycboard, pd, task, specs)
                push.update(pv_result.pushed)
                variables_set_pre_run.extend(pv_result.set_lines)
            except Exception as e:
                logger.warning("Box %s: restore_pers_vars failed: %s",
                               self.setup_id, e)

        # ONE serial round-trip for the whole set.
        if push:
            try:
                results = self.pycboard.set_variables(push)
                # Surface MCU rejections instead of silently logging "Setting
                # variables" for values the board never accepted.
                rejected = [k for k, ok in (results or {}).items() if not ok]
                missing = [k for k in push if k not in (results or {})]
                if rejected or missing:
                    bad = rejected + missing
                    logger.warning("Box %s: %d variable(s) NOT applied: %s",
                                   self.setup_id, len(bad), ", ".join(bad))
                    log("WARNING: variables not applied: " + ", ".join(bad))
            except BaseException as e:
                logger.error("Box %s: set_variables batch failed: %s",
                             self.setup_id, e)
                log("ERROR: variables NOT applied (board rejected the batch): "
                    + str(e))

        # "Setting variables." log block.
        if variables_set_pre_run:
            log("Setting variables.")
            name_w = max(len(n) for n, _, _ in variables_set_pre_run)
            val_w  = max(len(v) for _, v, _ in variables_set_pre_run)
            for name, val_repr, source in variables_set_pre_run:
                log(f"  {name.ljust(name_w)}  {val_repr.ljust(val_w)}  {source}")

    def _capture_persistent_at_stop(self) -> None:
        """Read persistent variables off the MCU and merge them (by name) into
        ``<project>/<task_family>/persistent_variables.json`` under
        ``values``. Per-box scope keeps the rule "one box's stop never blocks
        another's".

        Only a real (recorded) session persists, a subject_id in the field is
        the record-vs-dry-run switch, so a throwaway dry run sees the
        remembered values (restored at Upload) but never overwrites them.
        No-op when no project, no task uploaded, dry run, or no
        persistent-flagged variables, quietly logged.
        """
        if self.pycboard is None:
            return
        mw   = self.main_window
        pd   = getattr(mw, "_active_project_dir", None) if mw else None
        task = self._persistent_task_folder()
        edit = getattr(self, "subject_id_edit", None)
        subject = ""
        if edit is not None:
            try:
                subject = (edit.text() or "").strip()
            except Exception:
                subject = ""
        if not (pd and task and subject):
            return
        specs = self._box_variable_specs() or []
        if not any(s.persistent for s in specs if s and s.name):
            return
        # ONE get_variables() round-trip, filtered to persistent names. On an
        # error-stop the board may be wedged → capture_persistent returns {}
        # and we keep the previous saved values rather than overwrite them.
        pers_vars = _tv.capture_persistent(self.pycboard, specs)
        if not pers_vars:
            return
        path = _tv.write_pers_vars(pd, task, pers_vars)
        if path is not None:
            self._log_func(
                f"Persistent vars saved ({len(pers_vars)}): "
                f"{', '.join(sorted(pers_vars))}")

    def _auto_enable_tracking(self) -> bool:
        """Pre-Record tracking precheck.

        Decision tree (matches the user spec):

            cfg = pipe.get_tracking_config(box_id)
            cam = box_id in video_manager.box_camera_map

            # silent exits, proceed without tracking
            if not cfg.online_tracking_enabled:        return True
            if not (cfg.has_dlc() or cfg.has_blob()):  return True
            if cfg.has_dlc()  and not cam:             return True   # no camera, no warning
            if cfg.has_blob() and not cam:             return True

            # DLC / SLEAP path
            if cfg.has_dlc() and cam:
                needs_init = not pose_ready_for(pipe, mw.pose_settings_for_box(box_id))
                if needs_init:
                    choice = mw.prompt_pose_init_required(box_id)
                    if choice == "init":     ... try to init + enable
                    if choice == "disable":  set cfg.online_tracking_enabled=False
                                              (dlc_model_path STAYS)
                    if choice == "cancel":   return False  -> abort Record

            elif cfg.has_blob() and cam:
                pipe.enable_blob_tracking(box_id)

        Returns ``True`` when the Record click should proceed (tracking
        is on, off-by-design, or operator chose Disable), ``False`` when
        the operator clicked Cancel.

        Also calls ``pipe.apply_tracking_config(box_id)`` once at the
        top so mid-session dialog edits (push_zones_to_mcu,
        pose_n_instances, etc.) take effect on THIS Record click.
        """
        mw = self.main_window
        if mw is None:
            return True
        pipe = getattr(mw, "pipeline", None)
        if pipe is None:
            return True
        try:
            tc = pipe.get_tracking_config(self.setup_id)
        except Exception:
            tc = None
        if tc is None:
            return True

        # Push the latest config through to the policy + pose n_instances
        # so a mid-session dialog edit takes effect on THIS Record.
        try:
            pipe.apply_tracking_config(self.setup_id)
        except Exception as e:
            logger.warning("Box %s: apply_tracking_config at Record: %s",
                           self.setup_id, e)

        # Silent exits, operator opted out or nothing to do.
        if not getattr(tc, "online_tracking_enabled", True):
            return True
        if not (tc.has_dlc() or tc.has_blob()):
            return True

        # Camera attached for this box?
        cam_connected = False
        try:
            vm = getattr(mw, "video_manager", None)
            cam_map = getattr(vm, "box_camera_map", {}) if vm is not None else {}
            cam_connected = bool(cam_map.get(self.setup_id))
        except Exception:
            cam_connected = False
        if not cam_connected:
            # NO warning, operator can't track without a camera and
            # explicitly does not want a popup in this state.
            return True

        # ── Pose path (DLC / SLEAP) ─────────────────────────────────
        if tc.has_dlc():
            # One readiness question, asked through the same function the
            # dialog and the load path use, so all three agree. The normal
            # answer here is "ready": the model was prepared when the camera
            # came up, and Record has nothing to do but enable inference.
            settings = {}
            try:
                from source.gui.pose_subsystem import pose_ready_for
                settings = mw.pose_settings_for_box(self.setup_id)
                needs_init = not pose_ready_for(pipe, settings)
            except Exception as e:
                logger.warning("Box %s: pose readiness check failed: %s",
                               self.setup_id, e)
                needs_init = not pipe.has_pose_model()

            if needs_init:
                # Reached only when the operator deferred an init, the model
                # file moved, or a load failed. Modal: Init / Disable / Cancel.
                prompt = getattr(mw, "prompt_pose_init_required", None)
                choice = prompt(self.setup_id) if callable(prompt) else "cancel"
                if choice == "init":
                    enable = getattr(mw, "_enable_pose_for_box", None)
                    if callable(enable):
                        try:
                            # The box's own settings. This used to pass an
                            # empty dict, so the model path had to be
                            # recovered from a session cache that only operant
                            # keeps, and the maze equivalent always failed.
                            ok = enable(self.setup_id, settings)
                        except Exception as e:
                            logger.error("Box %s: _enable_pose_for_box raised: %s",
                                         self.setup_id, e)
                            ok = False
                        if not ok:
                            self._log_func(
                                f"Box {self.setup_id}: pose init failed; "
                                "recording will proceed without tracking.")
                    return True   # continue Record either way
                if choice == "disable":
                    try:
                        pipe.update_tracking_config(
                            self.setup_id, online_tracking_enabled=False)
                        self._log_func(
                            f"Box {self.setup_id}: online tracking disabled "
                            "(model_path preserved in config).")
                    except Exception as e:
                        logger.warning("Box %s: disable tracking update: %s",
                                       self.setup_id, e)
                    return True
                # Cancel, abort the Record click.
                self._log_func(
                    f"Box {self.setup_id}: Record aborted by operator "
                    "(pose init prompt cancelled).")
                return False

            # Model is current, just flip the enable bit if needed.
            try:
                if not pipe.pose.is_enabled(self.setup_id):
                    zm = None
                    tm = getattr(mw, "tracker_manager", None)
                    if tm is not None and hasattr(tm, "get_zone_manager"):
                        try:
                            zm = tm.get_zone_manager(self.setup_id)
                        except Exception:
                            zm = None
                    pipe.enable_pose(self.setup_id, zone_lookup=zm)
                    self._log_func(
                        f"Box {self.setup_id}: pose enabled "
                        "(model already loaded, settings unchanged)")
            except Exception as e:
                logger.warning("Box %s: pose enable failed: %s",
                               self.setup_id, e)
            return True

        # ── Blob path ───────────────────────────────────────────────
        if tc.has_blob():
            try:
                pipe.enable_blob_tracking(self.setup_id)
                self._log_func(
                    f"Box {self.setup_id}: blob tracker auto-enabled")
            except Exception as e:
                logger.warning("Box %s: auto-enable blob failed: %s",
                               self.setup_id, e)
        return True

    def mcu_start(self, record: bool = False) -> bool:
        """Start the framework, optionally with recording open.

        Order:
            1. Pre-checks (pycboard present, not already running)
            2. Auto-enable pose / blob from per-box TrackingConfig
            3. pycboard.start_framework()
            4. Mark framework_running + set run_mode (RECORD/DRY from subject)
            5. Start run_timer + plot_timer
            6. Status = "Recording:" (RECORD, green) / "Dry run:" (DRY,
               magenta) / "Running:" (defensive), keyed off run_mode
            7. Refresh button states
            8. Emit framework_started_signal
        """
        if not self.pycboard or self.framework_running:
            return False
        # Auto-enable pose / blob from the per-box TrackingConfig. The
        # config selection IS the signal, no separate "Start Tracking"
        # click required. May fire a 3-button modal when DLC is
        # configured + camera connected + model not loaded (or
        # settings changed). Returns False ONLY if the operator clicks
        # Cancel on that modal; "Disable" + "Init" both proceed.
        proceed = self._auto_enable_tracking()
        if proceed is False:
            return False
        # Variable pushes (hw_* + persistent restore) happen at Upload time
        # inside ``mcu_upload_task`` → ``_apply_setup_task_pushes``. Start is
        # just fw.run().
        try:
            self.pycboard.start_framework(data_output=True)
        except (Exception, PyboardError) as e:
            self._log_func(f"Start failed: {e}")
            self.set_status(f"Error: {str(e)[:40]}", "error")
            return False

        self.framework_running = True
        # Run mode is the authoritative run-state value, keyed off subject
        # presence (NOT the ``record`` param, which means "any data file
        # incl. temp" in operant but "real recording" in maze). Writer-open
        # tracking stays on the subclass flags / ``_has_*`` hooks.
        self.run_mode = self._run_mode_for_start()
        self._set_all_timers("00:00:00")
        self._notify_timer_mode()

        # Status reads run_mode, not the ``record`` param: a DRY run (no
        # subject; data → data/temp/, overwritten) must NOT show the green
        # "recording" colour even though operant opens a temp data file
        # (record=True). Only a real saved session (RECORD) is green.
        label = self._uploaded_task_rel or ""
        if self.run_mode == RunMode.RECORD:
            self.set_status(f"Recording: {label}", "recording")
        elif self.run_mode == RunMode.DRY:
            self.set_status(f"Dry run: {label}", "running")
        else:
            self.set_status(f"Running: {label}", "running")
        try:
            self._update_button_states()
        except Exception:
            pass
        self._emit_signal("framework_started_signal")
        if self.user_api is not None:
            try:
                self.user_api.run_start()
            except Exception as e:
                logger.error("Box %s: user_api.run_start error: %s",
                             self.setup_id, e)
        try:
            self._after_start(record)
        except Exception as e:
            logger.error("Box %s: _after_start error: %s", self.setup_id, e)
        self._log_func("Framework started")
        return True

    # ==================================================================
    # LIFECYCLE: STOP, the ONE convergence point
    # ==================================================================

    def mcu_stop(self, reason: str = "user") -> None:
        """The single stop method.  Three callers converge here:

            user        Stop button click / disconnect / closeEvent
            auto        plot_tick saw framework_running drop
            error:<msg> plot_tick caught PyboardError / SerialException

        Order is locked.  DO NOT REORDER.  Each step runs unconditionally
        because a failure in one step must not skip the next.
        """
        # 1. Flip framework_running off + ask the main window to swap
        #    process_timer → refresh_timer if this was the last active
        #    box and no camera is streaming. The HH:MM:SS label is frozen
        #    at the final fw timestamp in step 3a (user can read run length
        #    post-stop).
        self.framework_running = False
        self._notify_timer_mode()

        # 2. MCU-side stop, only when user-initiated.  auto / error
        #    paths come from the plot_tick which already saw the MCU
        #    transition; no reason to talk to it again (and on error
        #    the port is in a bad state anyway).
        if reason == "user" and self.pycboard:
            try:
                self.pycboard.stop_framework()
            except (Exception, PyboardError) as e:
                logger.warning("Box %s: stop_framework error: %s", self.setup_id, e)

        # 3. Drain final messages so logs / data files get the tail.
        if self.pycboard:
            try:
                time.sleep(0.05)
                self.pycboard.process_data()
            except (Exception, PyboardError):
                pass

        # 3b. On an ERROR stop, flush the serial input buffer so any residual
        #     traceback / partial bytes left by the framework error can't be
        #     read as the response to the NEXT exec_raw (gc_collect / fw.run /
        #     enter_raw_repl), which would leave the board appearing "stuck".
        if reason.startswith("error") and self.pycboard is not None:
            try:
                self.pycboard.serial.reset_input_buffer()
            except Exception as e:
                logger.debug("Box %s: post-error buffer flush: %s",
                             self.setup_id, e)

        # 3a. Freeze the HH:MM:SS clock at the EXACT final fw timestamp on
        #     EVERY surface (box card + Live Status + stats + tile), not just
        #     the box card. After the drain ``pycboard.timestamp`` holds the
        #     last fw_ms (session-end), so the frozen value equals the last
        #     TSV row to the millisecond, no host clock, no interpolation
        #     overshoot. The tick won't run again for a stopped box, so
        #     without this the tile/stats would freeze one interpolated tick
        #     off the box card. Reason-independent (user / auto / error).
        if self.pycboard is not None:
            try:
                from source.datetime_formats import format_run_clock
                self._set_all_timers(format_run_clock(self.pycboard.timestamp))
            except Exception:
                pass

        # 3b. Read persistent variables from the MCU and write them to
        #     ``<project>/<task>/persistent_variables.json`` under
        #     ``values[subject_id]``.
        try:
            self._capture_persistent_at_stop()
        except Exception as e:
            logger.error("Box %s: persistent capture error: %s",
                         self.setup_id, e)

        # 4-8. Shared GUI-thread finalize tail.
        self._finalize_stop(reason)

    def _finalize_stop(self, reason: str) -> None:
        """The GUI-thread tail of a stop, shared by ``mcu_stop`` (single-box /
        auto / error) and ``finalize_record_stop`` (parallel multi-box) so the
        two paths can't drift: user_api.run_stop → subclass ``_after_stop``
        → run_mode reset → clear displays → status → buttons → stopped signal.
        MUST run on the GUI thread (``_after_stop`` touches Qt)."""
        # 4. User API stop hook (before subclass cleanup so the api can still
        #    print_message / write data_logger).
        if self.user_api is not None:
            try:
                self.user_api.run_stop()
            except Exception as e:
                logger.error("Box %s: user_api.run_stop error: %s",
                             self.setup_id, e)

        # 5. Subclass post-hook (close data file, close video).
        try:
            self._after_stop(reason)
        except Exception as e:
            logger.error("Box %s: _after_stop(%s) error: %s",
                         self.setup_id, reason, e)
        finally:
            self.run_mode = RunMode.IDLE

        # 5b. Clear stale MCU state / event / print labels.
        self._clear_state_displays()

        # 6. Status surface, sticky red on error, distinguishable text for
        #    user-stop vs framework-initiated auto-stop.
        if reason.startswith("error"):
            msg = reason[6:][:40]
            self.set_status(f"Error: {msg}", "error")
            self._log_func(f"Framework stopped, {reason}")
        elif reason == "auto":
            self.set_status("Auto-stopped", "neutral")
            self._log_func("Framework stopped, auto")
        else:
            self.set_status("Stopped", "neutral")
            self._log_func("Framework stopped")

        # 7. Refresh button states (Stop disabled, Start enabled, etc.).
        try:
            self._update_button_states()
        except Exception:
            pass

        # 8. Emit framework_stopped_signal.
        self._emit_signal("framework_stopped_signal")

    # Helper: emit a per-box signal if the subclass declared it.
    def _emit_signal(self, name: str) -> None:
        sig = getattr(self, name, None)
        if sig is None:
            return
        try:
            sig.emit(self.setup_id)
        except Exception as e:
            logger.debug("emit %s for box %s: %s", name, self.setup_id, e)

    # ==================================================================
    # NAVIGATION (NAV_TYP byte protocol, pause/resume/next/prev)
    # ==================================================================

    def mcu_pause(self) -> None:
        if self.pycboard and self.pycboard.framework_running:
            try:
                self.pycboard.trigger_nav("pause")
            except Exception:
                pass

    def mcu_resume(self) -> None:
        if self.pycboard and self.pycboard.framework_running:
            try:
                self.pycboard.trigger_nav("resume")
            except Exception:
                pass

    def mcu_next_stage(self) -> None:
        if self.pycboard and self.pycboard.framework_running:
            try:
                self.pycboard.trigger_nav("next_stage")
            except Exception:
                pass

    def mcu_prev_stage(self) -> None:
        if self.pycboard and self.pycboard.framework_running:
            try:
                self.pycboard.trigger_nav("prev_stage")
            except Exception:
                pass


    def _capture_device_snapshots(self, store) -> None:
        """Capture the device-driver .py files THIS box's MCU holds into the
        snapshot store (``kind="device"`` → ``source/devices/<djb2>.py``),
        mirroring the HD capture.

        The file SET comes from the MCU, ``pycboard.device_files_on_pyboard``,
        the ``{file: djb2}`` map the board already computed in ``reset()``, so
        it's correct whether the HD was just loaded or was already on the board.
        The CONTENT is read from the host devices folder (same files that were
        uploaded). Refs are stashed on ``self._device_refs`` for
        ``_commit_box_sources_for_run`` to promote at record-start.
        """
        pyc = getattr(self, "pycboard", None)
        if store is None or pyc is None:
            return
        self._device_refs = []
        try:
            from source.config.settings import user_folder
            devices_dir = Path(user_folder("devices"))
            for fname in pyc.device_files_on_pyboard:
                if not fname.endswith(".py") or fname == "__init__.py":
                    continue
                dref = store.capture_source(devices_dir / fname,
                                            kind="device",
                                            setup_id=self.setup_id,
                                            label=fname)
                if dref is not None and dref.djb2:
                    self._device_refs.append(dref)
        except Exception as e:
            logger.warning("snapshot device capture (box %s): %s",
                           self.setup_id, e)

    # ==================================================================
    # BUTTON STATES
    # ==================================================================

    def _is_box_connected(self) -> bool:
        return self.pycboard is not None

    def _is_box_running(self) -> bool:
        return bool(self.framework_running)

    def _is_task_uploaded(self) -> bool:
        return bool(self.task_uploaded)

    def _check_task_consistency(self) -> None:
        """Verify the cached uploaded task AND its hardware definition are
        still byte-for-byte the files on disk. If either changed, force a
        re-upload (button → 'Upload').

        The task djb2 is gated on the file's mtime so the (expensive) hash
        only runs when the file actually changed on disk, not on every UI
        refresh. An edited HD that's still stale on the board is caught
        the same way."""
        try:
            if not self.task_uploaded or self.task_file_hash is None:
                return
            task_path_text = self._upload_task_text()
            if not task_path_text or task_path_text in self._TASK_PLACEHOLDERS:
                return
            task_rel = Path(task_path_text)
            if task_rel.suffix == ".py":
                task_rel = task_rel.with_suffix("")
            sm_name = task_rel.name
            sm_dir = Path("tasks") / task_rel.parent
            task_path = sm_dir / (sm_name + ".py")
            if not task_path.exists():
                return

            # an edited hardware definition (on disk != what's on the MCU)
            # also requires a re-upload, otherwise the box runs the stale HD
            # still loaded on the board, logged under the old hash.
            hd_changed = self._hd_changed_on_disk()

            # only re-hash the task when its mtime changed since the last
            # check, a cheap stat gates the djb2.
            try:
                mtime = task_path.stat().st_mtime
            except OSError:
                return
            task_changed = False
            if mtime != getattr(self, "_task_disk_mtime", None):
                self._task_disk_mtime = mtime
                from source.config.hashing import djb2_int_from_file as _djb2_file
                task_changed = _djb2_file(str(task_path)) != self.task_file_hash

            if not task_changed and not hd_changed:
                return

            self.task_uploaded = False
            self.task_file_hash = None
            self.last_task_hash = None
            btn = getattr(self, "upload_button", None)
            if btn is not None:
                try:
                    btn.setText("Upload")
                except Exception:
                    pass
            what = ("Hardware def" if (hd_changed and not task_changed)
                    else f"Task '{sm_name}'")
            try:
                self._log_func(f"{what} modified on disk - re-upload required")
            except Exception:
                pass
            self.set_status(
                ("HD modified - re-upload" if (hd_changed and not task_changed)
                 else "Task modified - re-upload"), "neutral")
        except Exception as e:
            logger.debug("Box %s: task hash check error: %s", self.setup_id, e)

    def _hd_changed_on_disk(self) -> bool:
        """True iff the on-disk hardware-definition .py differs from the HD
        currently loaded on the board. Compares the disk djb2 against the
        board's authoritative uploaded hash (``pycboard._loaded_hwd_hash``);
        cached on ``(mtime, board_hash)`` so the hash only re-runs when the
        file changed OR a fresh HD was uploaded (which clears the staleness)."""
        pyc = self.pycboard
        if pyc is None:
            return False
        path = getattr(pyc, "_loaded_hwd_path", "") or ""
        cached = getattr(pyc, "_loaded_hwd_hash", 0) or 0
        if not path or not cached:
            return False
        try:
            p = Path(path)
            if not p.exists():
                return False
            mtime = p.stat().st_mtime
        except OSError:
            return False
        key = (mtime, cached)
        if key == getattr(self, "_hd_check_key", None):
            return getattr(self, "_hd_stale", False)
        self._hd_check_key = key
        try:
            from source.config.hashing import djb2_int_from_file as _djb2_file
            self._hd_stale = (_djb2_file(path) != cached)
        except Exception:
            self._hd_stale = False
        return self._hd_stale

    def _extra_button_state(self, *, connected: bool, task_ready: bool,
                             running: bool, setup_enabled: bool) -> None:
        """Subclass hook for mode-specific buttons (maze: pause/next/prev/doors)."""

    def _refresh_upload_button_text(self) -> None:
        """Keep ``upload_button`` label in sync with ``task_uploaded``.

        Event-driven flips (``on_task_changed``, post-upload-success,
        ``_check_task_consistency``) cover user actions, but disconnect
        paths flip ``task_uploaded`` directly without touching the
        button. Calling this from ``_update_button_states`` ensures the
        text follows the flag on every UI refresh.
        """
        if not hasattr(self, "upload_button"):
            return
        desired = "Reset" if self.task_uploaded else "Upload"
        if self.upload_button.text() != desired:
            self.upload_button.setText(desired)

    def _update_button_states(self) -> None:
        """Refresh enable/disable on the standard button set both modes share.
        Subclass extras live in ``_extra_button_state``."""
        try:
            self._check_task_consistency()
        except Exception:
            pass
        connected = self._is_box_connected()
        running = self._is_box_running()
        uploaded = self._is_task_uploaded()
        task_ready = connected and uploaded
        setup_enabled = (not running) and (not self.global_setup_locked)
        for attr, enabled in (
            ("subject_id_edit",  setup_enabled and connected),
            ("task_combo",       connected and not running and not self.global_setup_locked),
            ("upload_button",    connected and not running and not self.global_setup_locked),
            ("record_button",    task_ready and not running and not self.global_setup_locked),
            ("stop_button",      running),
            ("controls_button",  connected and not self.global_controls_locked),
        ):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.setEnabled(bool(enabled))
                except Exception:
                    pass
        # Sync the upload button label, disconnect paths flip task_uploaded
        # without touching the text, so a periodic refresh keeps them aligned.
        self._refresh_upload_button_text()
        try:
            self._extra_button_state(
                connected=connected, task_ready=task_ready,
                running=running, setup_enabled=setup_enabled,
            )
        except Exception:
            pass

    def apply_global_state(self, lock_setup: bool = False,
                           lock_controls: bool = False) -> None:
        """Apply MainWindow-level locks and refresh the button states."""
        self.global_setup_locked = bool(lock_setup)
        self.global_controls_locked = bool(lock_controls)
        try:
            self._update_button_states()
        except Exception:
            pass
        try:
            self._after_apply_global_state()
        except Exception:
            pass

    def _after_apply_global_state(self) -> None:
        """Subclass hook, called after locks applied."""

    # ==================================================================
    # SUBJECT ID
    # ==================================================================

    def on_subject_id_changed(self) -> None:
        """Toggle Record/Start label based on subject ID presence."""
        try:
            from source.gui.styles import BUTTON_STYLE, COLORS
        except Exception as e:
            logger.debug("Style import failed in on_subject_id_changed: %s", e)
            return
        try:
            edit = getattr(self, "subject_id_edit", None)
            btn = getattr(self, "record_button", None)
            if edit is None or btn is None:
                return
            subject_id = edit.text().strip()
            has_subject = bool(subject_id)
            # Restyle only on the empty↔non-empty FLIP, two setStyleSheet
            # calls plus a palette write per keystroke per box each force a
            # full Qt style recompute the typing latency pays for.
            if getattr(self, "_has_subject_style", None) is not has_subject:
                self._has_subject_style = has_subject
                extra = self._record_button_extra_qss()
                pal = edit.palette()
                if has_subject:
                    btn.setText("Record")
                    btn.setStyleSheet(BUTTON_STYLE.format(
                        color=COLORS['record_light'],
                        hover_color=COLORS['record_light_hover'],
                    ) + extra)
                    # A real ID is entered → bold fig-green text (good to go).
                    edit.setStyleSheet(
                        "QLineEdit { background: rgba(15,23,42,0.95);"
                        " border: 1px solid #6b8347; border-radius: 4px;"
                        " padding: 2px 10px; min-height: 18px;"
                        " color: #9cbf6f; font-weight: bold; }"
                        "QLineEdit:focus { border: 1px solid #9cbf6f; }")
                    pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText,
                                 QtGui.QColor("#94a3b8"))
                else:
                    btn.setText("Start")
                    btn.setStyleSheet(BUTTON_STYLE.format(
                        color=COLORS['start_light'],
                        hover_color=COLORS['start_light_hover'],
                    ) + extra)
                    # No ID yet: light-red border + light-red "Enter ID"
                    # prompt so the missing ID is an obvious warning before
                    # a DRY run starts.
                    edit.setStyleSheet(
                        "QLineEdit { background: rgba(15,23,42,0.95);"
                        " border: 1px solid #ffb3b3; border-radius: 4px;"
                        " padding: 2px 10px; min-height: 18px;"
                        " color: #ffb3b3; font-weight: bold; }"
                        "QLineEdit:focus { border: 1px solid #ffb3b3; }")
                    pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText,
                                 QtGui.QColor("#ffb3b3"))
                edit.setPalette(pal)
            try:
                self._after_subject_id_change(subject_id)
            except Exception as e:
                logger.debug("after_subject_id_change box %s: %s", self.setup_id, e)
            # Notify main window so toolbar gating that depends on
            # per-box Subject-IDs (Clear Meta enable / disable) refreshes
            # in real time as the user types.
            mw = getattr(self, "main_window", None)
            if mw is not None:
                upd = getattr(mw, "update_metadata_button_states", None)
                if callable(upd):
                    try:
                        upd()
                    except Exception as e:
                        logger.debug("update_metadata_button_states: %s", e)
                # Mirror the Subject ID into the Live Status + camera-tile
                # headers (operant-only hook; maze has neither, so absent).
                idr = getattr(mw, "_refresh_box_id_labels", None)
                if callable(idr):
                    try:
                        idr(self.setup_id)
                    except Exception as e:
                        logger.debug("_refresh_box_id_labels: %s", e)
        except Exception as e:
            logger.error("on_subject_id_changed box %s: %s", self.setup_id, e)

    def _record_button_extra_qss(self) -> str:
        """Mode hook: QSS appended to the Record/Start restyle. Maze bakes
        its tall-row height in here (appending it separately after every
        restyle grew the stylesheet each keystroke)."""
        return ""

    def _after_subject_id_change(self, subject_id: str) -> None:
        """Subclass hook, extra action after subject ID toggle."""

    # ==================================================================
    # ZONE LOAD (per-setup zone-config loader)
    # ==================================================================

    def on_load_zones_clicked(self) -> None:
        if self.main_window is None:
            return
        loader = getattr(self.main_window, "load_zones_for_box", None)
        if loader is None:
            return
        try:
            loader(self.setup_id)
        except Exception as e:
            logger.error("on_load_zones_clicked box %s: %s", self.setup_id, e)

    # ==================================================================
    # SESSION DIRS
    # ==================================================================

    def _data_dir_root_path(self) -> Path:
        """The global ``<pyBehaviorLab>/data/`` folder.

        Data is always rooted here regardless of where the project file
        lives. The project's name becomes a subfolder; the project file
        itself just stores config (portable across machines).
        """
        from source import paths as _app_paths
        return Path(_app_paths.top_dir) / "data"

    def _project_data_dir_path(self) -> Optional[Path]:
        """Where the active project's data lives.

        Honours ``cfg.meta.data_dir`` when set (Browse… picker on the
        sidebar), that path becomes the project's data folder verbatim.
        Falls back to the canonical default
        (``source.config.experiment._default_data_dir_for``) otherwise.
        Returns ``None`` when neither override nor project name is
        available.
        """
        mw = getattr(self, "main_window", None)
        cfg = getattr(mw, "_active_config", None) if mw is not None else None
        if cfg is not None and cfg.meta is not None:
            custom = (cfg.meta.data_dir or "").strip()
            if custom:
                return Path(custom)
        project = self._project_name()
        if not project:
            return None
        from source.config.experiment import _default_data_dir_for
        return Path(_default_data_dir_for(project))

    def _project_name(self) -> str:
        """Active project name (empty when no project loaded)."""
        mw = self.main_window
        try:
            cfg = getattr(mw, "_active_config", None)
            if cfg is not None:
                name = (cfg.meta.project or cfg.experiment_name or "").strip()
                return name
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Task identity, single source of truth
    # ------------------------------------------------------------------
    #
    # Two callers want different things from "which task did the user
    # pick?", keep them strictly separated:
    #
    #   * File lookups (sidecar, source capture, reset_task sm_dir):
    #     need the FULL relative path → ``_uploaded_task_rel``
    #     (or the combo text when nothing is uploaded yet).
    #
    #   * Folder names (data tree, persistent_variables.json):
    #     need the FAMILY = parent folder name under tasks/, with
    #     loose scripts (`tasks/foo.py`) explicitly producing "" so
    #     they DON'T create per-task subfolders → ``_task_family()``.

    def _current_task_rel(self) -> str:
        """Canonical relname for the task active on this box.

        Prefers ``_uploaded_task_rel`` (set after a successful upload);
        falls back to whatever the user has picked in the task combo
        (which is the same ``5CSRTT/stage1.py`` shape). Returns "" when
        nothing is selected. Always strips ``.py`` and placeholder rows.
        """
        rel = (self._uploaded_task_rel or "").strip()
        if not rel:
            rel = self._upload_task_text().strip()
        if not rel or rel.startswith("---"):
            return ""
        # Normalize: strip trailing .py, collapse backslashes.
        p = Path(rel)
        if p.suffix == ".py":
            p = p.with_suffix("")
        return p.as_posix()

    def _task_family(self) -> str:
        """Family folder name under ``tasks/`` for the active task.

        Returns the IMMEDIATE parent directory name (e.g. ``"5CSRTT"``
        for ``tasks/5CSRTT/stage1.py``). Returns "" for loose tasks
        sitting directly in ``tasks/``; those are common scripts
        (calibration, debug) and don't get their own data subfolder.
        """
        rel = self._current_task_rel()
        if not rel:
            return ""
        parent = Path(rel).parent
        # "." for loose scripts (tasks/foo.py). Anything else is a real
        # family folder name.
        return parent.name if str(parent) != "." else ""

    def _resolve_task_name(self) -> str:
        """Folder name for the data tree. Alias of ``_task_family()``
        consumed by ``get_session_dirs`` / ``ensure_task_data_dir``."""
        return self._task_family()

    def get_session_dirs(self) -> tuple:
        """Return ``(session_root, mcu_dir, video_dir)``, creating dirs.

        Layout (the first segment after ``data/`` is whatever the user has
        chosen at this point in the session):
          * Project + task -> ``<code>/data/<project>/<task>/<YYYY-MM-DD>/{mcu,video}/``
          * Project only   -> ``<code>/data/<project>/<YYYY-MM-DD>/{mcu,video}/``
          * Task only      -> ``<code>/data/<task>/<YYYY-MM-DD>/{mcu,video}/``
          * Neither        -> ``<code>/data/<YYYY-MM-DD>/{mcu,video}/``

        Data always lives under the global ``<code>/data/``. The
        ``<project>`` / ``<task>`` segments mirror the experimenter's
        progressive setup so the tree fans out cleanly when multiple tasks
        share one project. Leaf folders are ``mcu`` (.tsv files) + ``video``.
        """
        from source.datetime_formats import format_date_dir
        date_str = format_date_dir()
        task = self._resolve_task_name()
        proj_data = self._project_data_dir_path()
        if proj_data is not None:
            session_root = proj_data / task / date_str if task else proj_data / date_str
        else:
            # No project + no override, fall back to data_root/[task]/date.
            data_root = self._data_dir_root_path()
            session_root = data_root / task / date_str if task else data_root / date_str
        mcu_dir = session_root / "mcu"
        video_dir = session_root / "video"
        # Fast path: skip mkdir if both directories already exist
        # (the common case for every record click after the first one
        # of a session). On slow storage like Jetson eMMC the metadata
        # ops for ``mkdir(parents=True, exist_ok=True)`` cost 10-50 ms
        # per call; the ``Path.exists()`` short-circuit is ~0.5 ms.
        try:
            if not mcu_dir.exists():
                mcu_dir.mkdir(parents=True, exist_ok=True)
            if not video_dir.exists():
                video_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Box %s: session dir mkdir failed: %s", self.setup_id, e)
        return session_root, mcu_dir, video_dir

    def _ensure_session_dirs(self):
        """Reuse cached upload-time dirs when the date + project root are
        unchanged; otherwise delegate to ``get_session_dirs``. Saves an mkdir
        round-trip per record click. Shared by BOTH modes (operant box widget
        + maze setup widget): the record path resolves the session tuple the
        same way regardless of mode."""
        if (getattr(self, "_session_dirs_created", False)
                and hasattr(self, "_session_root")):
            from source.datetime_formats import format_date_dir
            current_date = format_date_dir()
            # Cache is valid only if the date AND the project root still match.
            # A draft-then-save between upload and record changes the project
            # data dir; without this check the session lands in the wrong tree.
            proj = self._project_data_dir_path()
            proj_ok = proj is None or str(proj) in str(self._session_root)
            if current_date in str(self._session_root) and proj_ok:
                return self._session_root, self._mcu_dir, self._video_dir
            self._session_dirs_created = False
        return self.get_session_dirs()

    # ==================================================================
    # DRY-RUN SAFETY NET, temp/box<N>.tsv
    # A Start with NO subject_id writes the MCU TSV to a single flat file
    # ``<code>/temp/box<N>.tsv`` that EVERY dry run overwrites (no
    # accumulation, no folders), so a forgotten-subject session isn't lost.
    # If a run mattered the user copies that file elsewhere before the next
    # run. No popups, no recovery flow.
    # ==================================================================

    def _temp_dir_path(self) -> Path:
        """``<data>/temp/``, holds the overwritten dry-run artifacts."""
        return self._data_dir_root_path() / "temp"

    def temp_tsv_path(self) -> Path:
        """``data/temp/Box<N>.tsv``, overwritten dry-run MCU file."""
        return self._temp_dir_path() / f"Box{self.setup_id}.tsv"

    def temp_video_data_path(self) -> Path:
        """``data/temp/video_data_Box<N>.txt``, overwritten dry-run tracking log."""
        return self._temp_dir_path() / f"video_data_Box{self.setup_id}.txt"

    def temp_video_path(self) -> Path:
        """``data/temp/video_Box<N>.mp4``, overwritten dry-run video."""
        return self._temp_dir_path() / f"video_Box{self.setup_id}.mp4"

    def open_temp_mcu_data_logger(self, datetime_now) -> bool:
        """Open the MCU TSV at ``data/temp/Box<N>.tsv`` (overwrites). Returns
        True on success. The MCU behavioural data so a no-subject run isn't
        lost; the tracking text + video are opened separately by the caller."""
        if self.pycboard is None:
            return False
        try:
            p = self.temp_tsv_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.pycboard.data_logger.open_data_file(
                data_dir=str(p.parent), subject_ID="",
                datetime_now=datetime_now, box_ID=self.setup_id, metadata={},
                file_name=p.name)
            # Same mirror the real-session open installs, without it the
            # dry-run _video_data.txt had permanently empty state/events
            # columns. Writer is late-bound, so ordering vs the tracking
            # writer open doesn't matter.
            mw = self.main_window
            if mw is not None:
                try:
                    mw._attach_mcu_row_mirror(self.setup_id)
                except Exception as e:
                    logger.debug("Box %s: temp mcu row mirror: %s",
                                 self.setup_id, e)
            return True
        except Exception as e:
            logger.warning("Box %s: temp MCU open failed: %s", self.setup_id, e)
            return False

    def _commit_box_sources_for_run(self) -> None:
        """Promote this box's staged source files (task .py, HD .py) from
        ``_pending/`` to ``source/<djb2>.<ext>`` now that the run's MCU TSV
        is open. Shared by operant + maze so BOTH modes commit their source
        lineage. Reads ``main_window``, ``self._task_ref`` (stashed by
        ``_capture_uploaded_sources`` at upload) and ``self.box_number``,
        all set on every RunTask widget."""
        mw = self.main_window
        if mw is None:
            return
        store = getattr(mw, "_snapshot_store", None)
        cfg = getattr(mw, "_active_config", None)
        if store is None or cfg is None:
            return
        extras = {}
        task_ref = getattr(self, "_task_ref", None)
        if task_ref is not None:
            extras["task"] = task_ref
        device_refs = getattr(self, "_device_refs", None)
        try:
            store.commit_box_sources(
                cfg, int(self.setup_number), extra_refs=extras or None,
                device_refs=device_refs or None)
        except Exception as e:
            logger.warning(
                "Box %s: commit_box_sources failed: %s", self.setup_number, e)

    # ==================================================================
    # PARALLEL MULTI-BOX RECORD STOP hooks
    # ------------------------------------------------------------------
    # The ParallelStopCoordinator drives N boxes' record-stop work
    # truly in parallel by calling three hooks on each widget:
    #
    #   prepare_record_stop()    GUI thread, fast. Flip framework_running,
    #                            re-sync tick mode. Return True if the
    #                            worker should run for this box.
    #   run_record_stop_worker() Worker thread. MCU stop, drain, capture
    #                            persistent vars. Returns True/False.
    #   finalize_record_stop()   GUI thread, fast. Run _after_stop,
    #                            clear state displays, refresh buttons.
    #
    # The START side uses a serial for-loop (UniversalStartDialog),
    # per-box start is ~50 ms with the var-push at Upload time, so
    # parallelism would gain nothing.
    # ==================================================================

    def prepare_record_stop(self) -> bool:
        """Hook, paired with ParallelStopCoordinator. Default: flip
        framework_running on the GUI thread + ask the main window to
        re-sync its tick mode, leaving file-close / serial-stop /
        video-finalise to the worker.

        Returns True if a worker should run for this widget.
        """
        if not self.framework_running:
            return False
        self.framework_running = False
        self._notify_timer_mode()
        return True

    def run_record_stop_worker(self) -> bool:
        """Worker thread, serial-only side of stop: MCU stop, drain,
        capture persistent vars. ``pycboard.print`` MUST be swapped to
        a thread-safe buffer by the coordinator before this is called.

        ``_after_stop`` is NOT called here, it touches Qt widgets
        (task_plot, statisticsTab, video_recorder, refresh_ui_state)
        and must run on the GUI thread. ``finalize_record_stop``
        handles it.
        """
        try:
            if self.pycboard:
                try:
                    self.pycboard.stop_framework()
                except (Exception, PyboardError) as e:
                    logger.warning(
                        "Box %s: stop_framework error (worker): %s",
                        self.setup_id, e)
                try:
                    time.sleep(0.05)
                    self.pycboard.process_data()
                except (Exception, PyboardError):
                    pass
            try:
                self._capture_persistent_at_stop()
            except (Exception, PyboardError) as e:
                logger.error(
                    "Box %s: persistent capture (worker): %s",
                    self.setup_id, e)
            return True
        except (Exception, PyboardError) as e:
            logger.error("Box %s: run_record_stop_worker raised: %s",
                         self.setup_id, e)
            return False

    def finalize_record_stop(self, *, ok: bool, error: str = "") -> None:
        """GUI thread, Qt-side cleanup after the parallel worker finishes.
        Delegates to the SAME ``_finalize_stop`` tail ``mcu_stop`` uses,
        so the two stop paths can't drift. Parallel stop is always
        user-initiated; an ``error`` maps to the error status path."""
        self._finalize_stop(f"error:{error}" if error else "user")

    # ==================================================================
    # COM PORT (subclass provides _com_text)
    # ==================================================================

    _COM_PLACEHOLDERS = ("", "--- Select COM ---", "No USB serial devices")

    @property
    def com_port(self) -> str:
        try:
            text = self._com_text()
        except NotImplementedError:
            return ""
        text = (text or "").strip()
        return "" if text in self._COM_PLACEHOLDERS else text

    def _com_text(self) -> str:
        """Raw text from the per-box COM field. Both modes use ``com_id_edit``
        (the visible read-only field written by connect / project load)."""
        if hasattr(self, "com_id_edit"):
            return self.com_id_edit.text()
        return ""

    # ==================================================================
    # LOG FUNCTION (per-box log surface)
    # ==================================================================

    def _resolve_log_func(self) -> Callable[[str], None]:
        """Pick the per-box log callable (``append_status`` for maze,
        ``print_to_log`` for operant); no-op fallback. Called once from
        ``init_run_task`` and the result stored as the plain ``_log_func``
        attribute, NOT a ``@property``. PySide6 on some Python builds does
        not reliably honour the descriptor protocol for a ``property`` on a
        mixin sitting behind the Qt base in the MRO; a stored attribute
        always resolves."""
        fn = getattr(self, "append_status", None)
        if callable(fn):
            return fn
        fn = getattr(self, "print_to_log", None)
        if callable(fn):
            return fn
        return _noop_log


