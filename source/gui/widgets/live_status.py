"""Per-box live status card shown in the Live Status tab.

One QTextEdit transcribes the board's D/P/event stream; the header row
shows ``Box N`` on the left and a right-aligned HH:MM:SS run timer
mirrored from the box's MCU framework clock (``pycboard.get_timestamp()``,
the same value as the TSV). Scrollbars are hidden, the user can scroll
with the mouse wheel.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from source.log import get_logger

logger = get_logger()


class LiveStatusWidget(QtWidgets.QFrame):
    """Widget for displaying live status information for one box."""

    def __init__(self, setup_number, setup_widget=None, parent=None):
        super().__init__(parent)
        self.setup_number = setup_number
        # Optional back-reference to the BoxControlWidget, used by
        # MainWindowBase._on_process_tick to write the elapsed clock
        # into our ``timerLabel`` (no per-widget timer needed).
        self._box_widget = setup_widget
        self._build_ui()
        logger.debug(f"Initialized LiveStatusWidget for box {setup_number}")

    def set_error(self, on, msg=""):
        """Red frame + ⚠ header + tooltip when this box is in error.
        Driven centrally by box_alerts.apply_box_alerts; auto-clears."""
        on = bool(on)
        from source.gui.box_alerts import ERROR_COLOR, error_text, error_tooltip
        from source.gui.theme import THEME as _T
        border = ERROR_COLOR if on else _T.palette.surface_border_strong
        self.setStyleSheet(
            "LiveStatusWidget {"
            f" background-color: {_T.palette.surface};"
            f" border: {2 if on else 1}px solid {border};"
            f" border-radius: {_T.radius.md}px;"
            "}")
        lbl = getattr(self, "statusLabel", None)
        if lbl is not None:
            lbl.setText(error_text("Box", self.setup_number, on))
            color = ERROR_COLOR if on else _T.palette.text
            lbl.setStyleSheet(
                "font-size: 11pt; font-weight: 700;"
                f" color: {color}; border: none;"
                " background-color: transparent;"
                " text-transform: uppercase; letter-spacing: 0.5px;")
            lbl.setToolTip(error_tooltip(
                "Box {}".format(self.setup_number), on, msg))

    def _build_ui(self):
        """Single scrolling log of D/P/event lines with a header that
        shows the box number on the left and a synced run timer on
        the right. Native scrollbars are hidden, mouse wheel still
        scrolls."""
        from source.gui.theme import THEME as _T
        self.setStyleSheet(
            "LiveStatusWidget {"
            f" background-color: {_T.palette.surface};"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            f" border-radius: {_T.radius.md}px;"
            "}")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header row: status label (left) + run timer (right).
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        # Header reads just "Box {N}", the tab is already labelled
        # "Live Status", so repeating it on every per-box card was
        # redundant.
        self.statusLabel = QtWidgets.QLabel(f"Box {self.setup_number}")
        self.statusLabel.setStyleSheet(
            "font-size: 11pt; font-weight: 700;"
            f" color: {_T.palette.text}; border: none;"
            " background-color: transparent;"
            " text-transform: uppercase; letter-spacing: 0.5px;")
        header.addWidget(self.statusLabel)
        # Subject ID (shown only when assigned), cyan, between the box
        # label and the run timer. Fed by operant._refresh_box_id_labels.
        self.subjectLabel = QtWidgets.QLabel("")
        self.subjectLabel.setStyleSheet(
            "font-size: 10pt; font-weight: 600;"
            " color: #8be9fd; border: none; background-color: transparent;")
        self.subjectLabel.setVisible(False)
        header.addWidget(self.subjectLabel)
        header.addStretch(1)
        self.timerLabel = QtWidgets.QLabel("00:00:00")
        self.timerLabel.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.timerLabel.setStyleSheet(
            "font-family: 'Cascadia Mono', 'Consolas', monospace;"
            " font-size: 11pt; font-weight: 700;"
            f" color: {_T.palette.focus};"
            " border: none; background-color: transparent;")
        header.addWidget(self.timerLabel)
        layout.addLayout(header)

        self.statusEdit = QtWidgets.QTextEdit()
        self.statusEdit.setReadOnly(True)
        self.statusEdit.setPlaceholderText(f"Status for Box {self.setup_number}")
        self.statusEdit.setMinimumHeight(40)
        # Hide both scrollbar tracks. Mouse-wheel scrolling still works
        # via QAbstractScrollArea so the user can navigate the log.
        self.statusEdit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.statusEdit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.statusEdit.setStyleSheet(
            "QTextEdit {"
            f" background-color: {_T.palette.bg};"
            f" border: 1px solid {_T.palette.surface_border};"
            f" border-radius: {_T.radius.sm}px; padding: 6px;"
            " font-family: 'Cascadia Mono', 'Consolas', monospace;"
            f" font-size: {_T.font.mono_pt}pt; color: {_T.palette.text};"
            "}")
        layout.addWidget(self.statusEdit, stretch=1)

    def set_subject(self, subject):
        """Show the box's Subject ID in the header (hidden when none)."""
        s = (subject or "").strip()
        try:
            self.subjectLabel.setText(s)
            self.subjectLabel.setVisible(bool(s))
        except RuntimeError:
            pass

    def appendStatus(self, text):
        """Append text to current status."""
        current = self.statusEdit.toPlainText()
        if current:
            current += "\n"
        self.statusEdit.setText(current + text)
        logger.debug(f"Appended status for box {self.setup_number}: {text}")

    def clearStatus(self):
        """Clear the status text."""
        self.statusEdit.clear()
        logger.debug(f"Cleared status for box {self.setup_number}")
