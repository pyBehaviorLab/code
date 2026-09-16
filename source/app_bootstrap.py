"""Shared application bootstrap for pyOperant and pyMaze.

Both entry points need:
  1. Process-wide environment flags (Qt logging, OpenCV priority).
  2. Cython hot-path build check.
  3. App-name logger initialisation.
  4. QApplication with Fusion style + dark palette + global QSS.
  5. Uncaught-exception hook that surfaces errors via the GUI.
  6. (Maze only) SIGINT/SIGTERM handlers for clean Ctrl+C shutdown.

This module centralises all of the above so the per-mode launchers stay
tiny.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
import traceback
from typing import Callable

# ── before Qt, deliberately ──────────────────────────────────────────────
#
# onnxruntime's native module fails to initialise on Windows if PySide6 has
# already loaded, "DLL load failed while importing onnxruntime_pybind11_state:
# A dynamic link library (DLL) initialization routine failed", because Qt
# brings in a conflicting copy of a runtime it needs. Imported first it works,
# and keeps working once Qt is up.
#
# This module is what both entry points import first, which makes it the only
# place early enough to matter. Without it a SLEAP model exported to ONNX or
# TensorRT runs perfectly from a script and is refused by the GUI, on the same
# machine, for the same folder, and nothing about the message points at Qt.
try:                                    # pragma: no cover - platform specific
    import onnxruntime  # noqa: F401
except Exception:
    pass                                # not installed, or no exported model


def configure_runtime_environment() -> None:
    """Set process-wide env vars BEFORE any Qt/OpenCV import.

    Cross-platform entries are unconditional; Windows-only entries are
    gated by ``sys.platform`` so they don't poison Linux/Jetson runs.
    On Linux Qt auto-detects ``xcb`` / ``wayland`` from the session.
    """
    os.environ['QT_LOGGING_RULES'] = 'qt.qpa.fonts=false;*.debug=false'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

    if sys.platform.startswith('win'):
        # Windows platform plugin with FreeType font engine.
        os.environ['QT_QPA_PLATFORM'] = 'windows:fontengine=freetype'
        # MSMF is the default OpenCV camera backend on Windows but is
        # slow to enumerate; DSHOW is faster. Setting priority=0
        # disables MSMF without affecting other backends.
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        _raise_windows_timer_resolution()
    elif sys.platform.startswith('linux'):
        # Linux/Jetson camera enumeration is noisy out of the box:
        #   * OpenCV's Orbbec depth-camera probe logs ERROR-level
        #     "Camera index out of range" for every /dev/video* that
        #     isn't an Orbbec sensor. We don't ship one, disable.
        #   * OpenCV-bundled FFmpeg's v4l2 backend logs ioctl probe
        #     failures ("Inappropriate ioctl for device") for Jetson CSI /
        #     Argus / NVMM device nodes that don't accept the standard
        #     query. ``OPENCV_FFMPEG_LOGLEVEL`` controls this, set to
        #     -8 (AV_LOG_QUIET) to silence all FFmpeg chatter, since the
        #     probe failures are harmless and we never want them in the
        #     console. Real FFmpeg errors that matter (codec init fail,
        #     out-of-memory) still surface via OpenCV's own error path.
        # ``setdefault`` lets a user override either via the shell.
        os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_OBSENSOR', '0')
        os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
        # Do NOT set QT_QPA_PLATFORM, Qt picks xcb (X11) or wayland
        # automatically from the session. Override via the shell for
        # headless (QT_QPA_PLATFORM=offscreen) or forced-X11 runs.


def _raise_windows_timer_resolution() -> None:
    """Tell Windows we need 1 ms timer granularity.

    Without this, Windows' default ~15 ms scheduler tick makes
    ``time.sleep(0.005)`` in the pipeline tick oversleep, throttling it
    to ~67 Hz. ``timeBeginPeriod(1)`` raises the process-wide tick rate to
    1 ms; the cost is negligible extra scheduler work for the process
    lifetime. No-op on non-Windows / on failure.

    Released automatically when the process exits.

    This buys **scheduling** granularity and nothing else. It does not make
    ``time.monotonic()`` finer, measured on this rig, ``GetTickCount64()``
    stays at 15.625 ms with the period raised, because Windows 11 no longer
    lets a process move it. Timestamps come from ``source.host_clock``
    instead, which is on QueryPerformanceCounter.
    """
    try:
        import sys
        if sys.platform != "win32":
            return
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def ensure_cython_extensions_built() -> None:
    """Compile stale Cython hot-path extensions if needed."""
    from source.cython.build import ensure_built
    ensure_built(verbose=True)


def prewarm_camera_enumeration() -> None:
    """Walk the camera bus once, on a worker, while the window is opening.

    Enumeration opens every index and probes it, ~10 s on this rig, and the
    result is what the Camera-ID pickers list and what an identity resolves
    against. Done lazily it lands on whoever asks first, which is the operator
    opening a dropdown; done here it is finished long before, and the GUI
    thread never walks the bus at all (``identity._may_scan`` forbids it).

    Idempotent and non-fatal: a failure just means the first dropdown pays the
    cost instead.
    """
    import threading
    import time as _time

    from source.log import get_logger as _get_logger

    def _target():
        log = _get_logger("camera_enum_warmup")
        t0 = _time.monotonic()
        try:
            from source.video.cameras.factory import CameraFactory
            cams = CameraFactory.list_available_cameras(["opencv"])
            log.info("camera enumeration warm: %d camera(s) in %.1f s",
                     len(cams), _time.monotonic() - t0)
        except Exception as e:
            log.debug("camera enumeration warmup failed: %s", e)

    threading.Thread(target=_target, name="camera-enum-warmup",
                     daemon=True).start()


def prewarm_video_encoders() -> None:
    """Force codec DLL load on a background thread BEFORE any box records.

    See ``source.video.recording.encoder_pool``, first ``cv2.VideoWriter`` /
    NVENC writer in a process costs 100-300 ms while Windows loads codec
    DLLs, and ``EncoderCapabilities.detect_all()`` adds several seconds
    of ffmpeg / nvidia-smi subprocess probes. Running both off the GUI
    thread means:

      * GUI window appears instantly (no perceived boot delay).
      * By the time the user has connected MCUs, picked tasks and
        clicked Record (typically 10+ s of UI setup), warmup is done.
      * If the user clicks Record before warmup finishes, the worker
        thread for that box's encoder init just blocks until warmup
        completes, still off the GUI thread.

    Idempotent and non-fatal: a failed warmup does not crash the app,
    recording still works, it just pays the DLL cost at record-click.
    """
    import threading
    import time as _time

    from source.log import get_logger as _get_logger

    def _target():
        log = _get_logger("encoder_warmup")
        t0 = _time.monotonic()
        try:
            from source.video.recording.encoder_pool import warm_video_encoders
            warm_video_encoders()
            log.info("video encoder pre-warm OK in %.0f ms",
                     (_time.monotonic() - t0) * 1000.0)
        except Exception as e:
            log.warning(
                "video encoder pre-warm FAILED after %.0f ms (%s), first "
                "record click will pay 100-300 ms of codec DLL load",
                (_time.monotonic() - t0) * 1000.0, e,
            )

    threading.Thread(
        target=_target, name="encoder-warmup", daemon=True,
    ).start()


def initialise_app_logger(app_name: str) -> None:
    """Pin the app name + set the active log level (PYBEHAVIORLAB_LOGLEVEL
    env var, else INFO). No console handler is installed; the per-GUI log
    file (opened later by set_log_directory) follows the same level."""
    from source.log import configure_logging, set_app_name
    set_app_name(app_name)
    configure_logging()
    # Now that logging exists, say what clock this machine gives us. Every
    # frame timestamp, every latency figure and the host↔MCU anchor come off
    # it, so a platform whose clock cannot resolve a frame interval has to
    # announce itself rather than quietly quantise a session's timing.
    from source import host_clock
    host_clock.check()


#: Qt warnings that mean a Qt object was touched from the wrong thread.
#: Qt reports these on stderr and carries on, but the behaviour after one is
#: undefined and the process can die later with no Python traceback at all,
#: which is what a Session Plot crash looks like: the log stops mid-way
#: through building a box's block, having emitted several of these.
_CROSS_THREAD_MARKERS = (
    "cannot be started from another thread",
    "cannot be stopped from another thread",
    "Cannot create children for a parent that is in a different thread",
    "Cannot send events to objects owned by a different thread",
    "is not the object's thread",
)


def install_qt_message_handler() -> None:
    """Route Qt's own messages into the log, with a stack for thread errors.

    Qt names neither the object nor the call site, so on its own the warning
    says only that SOMETHING is wrong SOMEWHERE. The handler runs on the
    thread that provoked it, so the Python stack captured here is the
    offending call, which turns an unactionable warning into a file and line.

    Cheap by construction: every message is logged, and only the handful
    matching a thread marker pays for a formatted stack.
    """
    import threading

    from PySide6 import QtCore

    logger = logging.getLogger("qt")
    levels = {
        QtCore.QtMsgType.QtDebugMsg: logging.DEBUG,
        QtCore.QtMsgType.QtInfoMsg: logging.INFO,
        QtCore.QtMsgType.QtWarningMsg: logging.WARNING,
        QtCore.QtMsgType.QtCriticalMsg: logging.ERROR,
        QtCore.QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _handler(mode, context, message):
        level = levels.get(mode, logging.WARNING)
        if any(m in message for m in _CROSS_THREAD_MARKERS):
            # ``[:-1]`` drops this handler's own frame.
            stack = "".join(traceback.format_stack()[:-1])
            logger.error(
                "Qt THREAD VIOLATION on thread %r: %s\n"
                "This is undefined behaviour and can crash the process later "
                "with no traceback. The call that provoked it:\n%s",
                threading.current_thread().name, message, stack)
            return
        logger.log(level, "%s", message)

    QtCore.qInstallMessageHandler(_handler)


def build_qt_application(application_name: str):
    """Create the QApplication with Fusion style + dark palette + QSS.

    Returns the QApplication instance ready to host the main window.
    """
    from PySide6 import QtGui, QtWidgets
    from PySide6.QtWidgets import QStyleFactory

    # Before the QApplication, so a message emitted during construction is
    # caught too.
    install_qt_message_handler()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(application_name)

    default_font = QtGui.QFont("Segoe UI", 9)
    default_font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    app.setFont(default_font)

    # Force Fusion. Windows native style ignores QSS for QComboBox /
    # QSpinBox / QHeaderView::corner, so any non-Fusion choice fights our
    # dark theme.
    app.setStyle(QStyleFactory.create("Fusion"))

    from source.gui.style_builders import apply_global_palette
    from source.gui.styles import GLOBAL_STYLE
    apply_global_palette(app)
    app.setStyleSheet(GLOBAL_STYLE)

    return app


def install_uncaught_exception_hook(
    *,
    get_window: Callable[[], object | None],
    on_shutdown: Callable[[], None] | None = None,
) -> None:
    """Install a sys.excepthook that logs uncaught exceptions and tries
    to surface them via the main window.

    Args
    ----
    get_window:
        Callable that returns the current MainWindow (or None if it has
        not been constructed yet). Used inside the hook so the hook can
        be installed before the window exists.
    on_shutdown:
        Optional callback fired when an exception arrives, typically
        stops all running framework boxes so the MCU does not keep
        running after the host crashes.
    """
    from source.log import get_logger
    logger = get_logger("boot")

    def _hook(exc_type, exc_value, exc_tb):
        try:
            try:
                from serial import SerialException

                from source.communication.pyboard import PyboardError
            except Exception:
                SerialException = ()  # type: ignore
                PyboardError = ()      # type: ignore

            if isinstance(SerialException, type) and issubclass(exc_type, SerialException):
                logger.error("Serial connection lost: %s", exc_value)
            elif isinstance(PyboardError, type) and issubclass(exc_type, PyboardError):
                logger.error("Unable to execute command on pyboard: %s", exc_value)
            else:
                logger.error(
                    "Uncaught exception of type: %s", exc_type.__name__,
                    exc_info=(exc_type, exc_value, exc_tb),
                )

            if on_shutdown is not None:
                with contextlib.suppress(Exception):
                    on_shutdown()

            window = None
            try:
                window = get_window()
            except Exception:
                window = None
            if window is not None and hasattr(window, 'showError'):
                with contextlib.suppress(Exception):
                    window.showError(
                        f"Internal error: {exc_value}\n\nSee error log for details.")
        except Exception:
            traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def install_signal_handlers(app, *, banner: str) -> None:
    """SIGINT / SIGTERM → app.quit().

    Only the maze entry point uses this, operant is single-process and
    only ever quits via the GUI close button.
    """

    def _handler(sig, frame):
        print(f"\n{banner} Received signal {sig}, shutting down...")
        app.quit()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def attach_error_log_to_logger(window) -> None:
    """Connect the main window's error log widget to the file/UI logger.

    Falls back to file-only logging if the widget is missing.
    """
    from source.log import initialize_logger
    log_widget = getattr(window, 'errorLogBrowser', None)
    if log_widget is not None:
        initialize_logger(log_widget=log_widget, log_to_file=True)
    else:
        initialize_logger(log_to_file=True)


def show_window_with_dark_titlebar(window) -> None:
    """window.show() + Win32 caption-strip → dark mode."""
    from source.gui.style_builders import force_dark_titlebar
    window.show()
    force_dark_titlebar(window)
