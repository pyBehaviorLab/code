#!/usr/bin/env python3
"""pyBehaviorLab, operant-mode launcher.

Bootstrap helpers (Qt setup, encoder pre-warm, exception hook, …) live
in ``source/app_bootstrap.py``, shared with ``pyMaze.py``.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import freeze_support

from source.app_bootstrap import (
    attach_error_log_to_logger,
    build_qt_application,
    configure_runtime_environment,
    ensure_cython_extensions_built,
    initialise_app_logger,
    install_uncaught_exception_hook,
    prewarm_camera_enumeration,
    prewarm_video_encoders,
    show_window_with_dark_titlebar,
)


def main() -> None:
    from source import GUI_VERSION, FW_VERSION
    print("=" * 70)
    print(f"pyBehaviorLab, OPERANT MODE   (GUI v{GUI_VERSION} · firmware v{FW_VERSION})")
    print("=" * 70)

    configure_runtime_environment()
    ensure_cython_extensions_built()
    initialise_app_logger("pyBLOperant")

    # Lazy import, only after env + logger pinned.
    from source.gui.operant import MainWindow

    app = build_qt_application("pyBehaviorLab - Operant")
    # Pre-warm codec DLLs, saves 100-300 ms on first-box record click.
    prewarm_video_encoders()
    prewarm_camera_enumeration()

    window_ref: list = [None]

    def _stop_all_running_boxes() -> None:
        window = window_ref[0]
        if window is None or not hasattr(window, 'get_all_setup_widgets'):
            return
        for box_widget in window.get_all_setup_widgets():
            if getattr(box_widget, 'framework_running', False):
                try:
                    box_widget.on_stop_clicked()
                # BaseException so PyboardError can't escape the shutdown
                # hook; user interrupts still propagate.
                except BaseException as e:
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise

    install_uncaught_exception_hook(
        get_window=lambda: window_ref[0],
        on_shutdown=_stop_all_running_boxes,
    )

    window = MainWindow()
    window_ref[0] = window
    attach_error_log_to_logger(window)

    print(f"[pyOperant] GUI ready (PID: {os.getpid()})")
    show_window_with_dark_titlebar(window)

    exit_code = app.exec()
    print("[pyOperant] Shutdown complete")
    sys.exit(exit_code)


if __name__ == '__main__':
    # freeze_support() is needed for multiprocessing under PyInstaller
    # bundles on Windows; harmless on regular Python.
    freeze_support()
    main()
