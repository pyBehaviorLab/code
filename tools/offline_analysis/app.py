"""Entry point for the standalone offline analyzer.

Launch as:
    python -m tools.offline_analysis.app

The rig GUI (operant + maze) spawns the analyzer via this same form using
``subprocess.Popen``: see ``source/gui/operant.py`` and ``source/gui/maze.py``.
"""

from __future__ import annotations

import logging
import sys

# ── before Qt, deliberately ──────────────────────────────────────────────
#
# onnxruntime's native module fails to initialise on Windows if PySide6 has
# already loaded, "DLL load failed while importing onnxruntime_pybind11_state:
# A dynamic link library (DLL) initialization routine failed", because Qt
# brings in a conflicting copy of a runtime it needs. Imported first it works,
# and keeps working once Qt is up.
#
# The symptom without this is a SLEAP model that runs perfectly from a script
# and is reported "not importable here" by the app, on the same machine, for
# the same folder. Nothing about the model is wrong.
try:                                    # pragma: no cover - platform specific
    import onnxruntime  # noqa: F401
except Exception:
    pass                                # not installed, or not needed here

from PySide6 import QtWidgets

from .main_window import OfflineAnalyzerMainWindow
from .theming import setup_dark_theme, apply_window_chrome


def _install_basic_logging() -> None:
    """Stdlib logging only, the analyser deliberately does not pull
    ``source.observability`` (that lives behind the rig perimeter)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(argv=None) -> int:
    _install_basic_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting pyBehaviorLab offline analyzer")

    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("pyBehaviorLab Offline Analyzer")
    setup_dark_theme(app)

    # Last-resort exception handler so an unhandled crash inside a view
    # surfaces a dialog instead of silently killing the process.
    def _excepthook(etype, value, tb):
        import traceback
        msg = "".join(traceback.format_exception(etype, value, tb))
        logger.error("Unhandled exception:\n%s", msg)
        try:
            QtWidgets.QMessageBox.critical(
                None, "Unhandled error",
                f"{etype.__name__}: {value}\n\n(See console for full traceback.)"
            )
        except Exception:
            pass
    sys.excepthook = _excepthook

    win = OfflineAnalyzerMainWindow()
    win.show()
    apply_window_chrome(win)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
