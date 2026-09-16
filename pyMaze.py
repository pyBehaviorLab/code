"""pyBehaviorLab, maze-mode launcher.
Bootstrap helpers (Qt setup, encoder pre-warm, signal handlers, …)
live in ``source/app_bootstrap.py``, shared with ``pyOperant.py``.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import freeze_support

from source.app_bootstrap import (
    build_qt_application,
    configure_runtime_environment,
    ensure_cython_extensions_built,
    initialise_app_logger,
    install_signal_handlers,
    install_uncaught_exception_hook,
    prewarm_camera_enumeration,
    prewarm_video_encoders,
    show_window_with_dark_titlebar,
)


def main() -> None:
    from source import GUI_VERSION, FW_VERSION
    print("=" * 70)
    print(f"pyBehaviorLab, MAZE MODE   (GUI v{GUI_VERSION} · firmware v{FW_VERSION})")
    print("=" * 70)

    configure_runtime_environment()
    ensure_cython_extensions_built()
    initialise_app_logger("pyBLmaze")

    from source.gui.maze import MainWindow

    app = build_qt_application("pyBehaviorLab - Maze")
    # Pre-warm codec DLLs, saves 100-300 ms on first-arena record click.
    prewarm_video_encoders()
    prewarm_camera_enumeration()

    window_ref: list = [None]
    install_uncaught_exception_hook(get_window=lambda: window_ref[0])
    install_signal_handlers(app, banner="[pyMaze]")

    window = MainWindow()
    window_ref[0] = window

    print(f"[pyMaze] GUI ready (PID: {os.getpid()})")
    show_window_with_dark_titlebar(window)

    exit_code = app.exec()
    print("[pyMaze] Shutdown complete")
    sys.exit(exit_code)


if __name__ == '__main__':
    # freeze_support() is needed for multiprocessing under PyInstaller
    # bundles on Windows; harmless on regular Python.
    freeze_support()
    main()
