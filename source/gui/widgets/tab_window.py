"""Top-level window that hosts a tab detached from DetachableTabWidget."""

from PySide6 import QtCore, QtWidgets

from source.gui.window_behavior import register_independent_window
from source.log import get_logger

logger = get_logger()


class DetachableTabWindow(QtWidgets.QMainWindow):
    """Window for detached tabs (used by both maze and operant)."""

    def __init__(self, widget, title, tab_widget):
        # No Qt parent: a parented widget with the Window flag becomes an
        # *owned* top-level, which Windows pins permanently above its owner,
        # the operator could never put a detached tab behind the main window.
        # See source/gui/window_behavior.py. DetachableTabWidget.detachedTabs
        # holds the reference that keeps this alive.
        super().__init__(None)
        register_independent_window(self)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(800, 600)
        self._tab_widget = tab_widget
        self._tab_name = title
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        widget.show()
        self.setCentralWidget(container)
        self._detached_widget = widget

    def closeEvent(self, event):
        try:
            w = self._detached_widget
            if w is not None and self._tab_widget is not None:
                self.centralWidget().layout().removeWidget(w)
                w.setParent(None)
                self._tab_widget.attach_tab(self._tab_name, w)
                self._detached_widget = None
        except Exception as e:
            logger.error(f"Reattach failed: {e}")
        event.accept()


