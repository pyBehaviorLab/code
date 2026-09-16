"""Qt-aware logging handler that writes log records into a QTextBrowser widget.

Kept separate from ``source.log`` so that module stays Qt-free.
"""

import logging
from PySide6 import QtCore


class QTextBrowserHandler(logging.Handler):
    """Custom logging handler that writes to a QTextBrowser widget.

    Cross-thread safety: emit() routes the append() call onto the Qt event
    loop via QueuedConnection, so log calls from background threads don't
    touch the widget directly.
    """

    def __init__(self, text_browser):
        super().__init__()
        self.text_browser = text_browser

    def emit(self, record):
        try:
            msg = self.format(record)
            QtCore.QMetaObject.invokeMethod(
                self.text_browser,
                "append",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, msg),
            )
        except Exception:
            # Widget may be destroyed during shutdown, drop silently.
            pass


__all__ = ["QTextBrowserHandler"]
