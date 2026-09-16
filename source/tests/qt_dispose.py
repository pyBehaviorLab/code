"""Actually destroy a Qt widget in a test.

The obvious incantation does not work, and the repo used it for a long time:

    w.close(); w.deleteLater(); QApplication.processEvents()   # leaks

``deleteLater`` posts a ``DeferredDelete`` event, and Qt deliberately holds
those back: they are only delivered when the event loop unwinds to the level
at or below the one the delete was requested from. A test has no running
event loop, so ``processEvents`` never delivers them and the widget survives,
which is how the suite came to hold thousands of live widgets and stopped
finishing (see ``destroy_widgets_created_by_this_test`` in conftest).

``sendPostedEvents(None, DeferredDelete)`` is the call that actually delivers
them, and it is what this helper adds.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def dispose(*widgets, close: bool = False) -> None:
    """Destroy ``widgets`` and drain the deferred-delete queue.

    Does NOT close by default. ``closeEvent`` is application code, and running
    it from a test is a trap: ``UnifiedTrackingDialog`` opens a modal "save
    zones?" prompt in its own, which with no event loop either corrupts the
    heap or blocks the run forever. Destroying is enough, the widget's
    destructor still stops its timers and children.

    Pass ``close=True`` only for a widget whose ``closeEvent`` you have read
    and actually want (e.g. the ROI dialog releasing its frame source).
    """
    for w in widgets:
        if w is None:
            continue
        try:
            if close:
                w.close()
            w.deleteLater()
        except RuntimeError:
            continue          # C++ side already gone
    app = QtWidgets.QApplication.instance()
    if app is not None:
        # The one call that actually delivers DeferredDelete outside a loop.
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


class WidgetBin(list):
    """Collects widgets a module-level helper builds, for disposal per test.

    Test files here build widgets in plain helper functions (``_editor(app)``,
    ``_operant_with_box(qapp)``), which cannot request a fixture and so have
    nowhere to register cleanup. Rather than a try/finally around every test,
    the helper drops the widget in a bin and one autouse fixture empties it::

        _BIN = WidgetBin()

        @pytest.fixture(autouse=True)
        def _drain():
            yield
            _BIN.drain()

        def _editor(app):
            return _BIN.add(ZoneEditorWidget(...))
    """

    def add(self, widget):
        """Register ``widget`` and return it, so helpers can ``return bin.add(w)``."""
        self.append(widget)
        return widget

    def drain(self, *, close: bool = False) -> None:
        dispose(*self, close=close)
        self.clear()


__all__ = ["WidgetBin", "dispose"]
