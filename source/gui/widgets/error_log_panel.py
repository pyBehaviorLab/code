"""Error-log viewer panel, read-only QTextEdit + Debug/Clear/Export bar.

Used by both modes: operant embeds two instances (tab + sidebar), maze
embeds one (sidebar). The host window aliases the panel's ``browser``
and ``debug_button`` to its own ``errorLogBrowser`` / ``debugToggleButton``
attributes so the shared helpers in ``MainWindowUtilsMixin``
(``_appendErrorLog``, ``_toggleDebugMode``, ``_exportErrorLog``) keep
working without changes.

Styling is fully token-driven, the ``style="light"`` argument is
accepted but ignored (the app is dark-only).
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from source.gui.theme import THEME as _T
from source.gui.styles import BUTTON_STYLE as _BS, COLORS as _C


def _text_browser_qss() -> str:
    """Dark mono-text panel for the error log itself."""
    p = _T.palette
    return (
        "QTextEdit {"
        f" background-color: {p.bg};"
        f" border: 1px solid {p.surface_border_strong};"
        f" border-radius: {_T.radius.md}px; padding: 10px;"
        " font-family: 'Cascadia Mono', 'Consolas', monospace;"
        f" font-size: {_T.font.mono_pt}pt;"
        f" color: {p.text};"
        "}"
    )


class ErrorLogPanel(QtWidgets.QWidget):
    """Read-only error log + button bar."""

    def __init__(self,
                 parent: QtWidgets.QWidget | None = None,
                 *,
                 style: str = "dark",
                 on_debug_toggled=None,
                 on_export=None) -> None:
        super().__init__(parent)
        self._on_debug_toggled = on_debug_toggled
        self._on_export = on_export

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.browser = QtWidgets.QTextEdit()
        self.browser.setReadOnly(True)
        self.browser.setStyleSheet(_text_browser_qss())
        outer.addWidget(self.browser, stretch=1)
        # Route Python logging into the browser, the ONE hookup site for
        # both modes, so log records show in-app AND the Debug toggle
        # (which lowers the GUI handler to DEBUG) has a handler to act on.
        from source.log import initialize_logger
        initialize_logger(log_widget=self.browser)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(5)

        # Debug: checkable info toggle. Clear: danger. Export: primary.
        self.debug_button = self._mk_button(
            "Debug: OFF", color_key="info", checkable=True)
        self.debug_button.clicked.connect(self._debug_clicked)
        bar.addWidget(self.debug_button)

        clear = self._mk_button("Clear", color_key="danger")
        clear.clicked.connect(self.browser.clear)
        bar.addWidget(clear)

        export = self._mk_button("Export", color_key="primary")
        export.clicked.connect(self._export_clicked)
        bar.addWidget(export)
        bar.addStretch(1)
        outer.addLayout(bar)

    @staticmethod
    def _mk_button(text, *, color_key, checkable=False):
        btn = QtWidgets.QPushButton(text)
        btn.setMinimumHeight(28)
        btn.setStyleSheet(_BS.format(
            color=_C[color_key], hover_color=_C[color_key + '_hover']))
        if checkable:
            btn.setCheckable(True)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        return btn

    def _debug_clicked(self) -> None:
        if callable(self._on_debug_toggled):
            self._on_debug_toggled()

    def _export_clicked(self) -> None:
        if callable(self._on_export):
            self._on_export()
