"""Log sidebar, matches the main GUI's pattern exactly.

Main GUI's pattern (see source/gui/widgets/common.py:222 ``CollapsibleSidebar``
and source/gui/base.py:1745 ``_make_rotated_toggle``):

  * A narrow ``RotatedButton`` (22×150 px) sits floating on the central
    widget at a fixed (x=0, y=top-left), NOT a permanent left strip.
  * Clicking the button triggers an animated slide-out ``CollapsibleSidebar``
    that overlays the canvas from the left edge (300 ms cubic ease-out).
  * The button HIDES while the sidebar is expanded; the sidebar
    collapses when the user clicks outside it (event-filter on parent)
    or clicks its close button. ``on_collapse_callback`` then shows the
    button again.

Both classes are copied from the rig's widgets and now read the rig's design
tokens directly rather than a local mirror of them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from tools.offline_analysis.vendor.theme import THEME as TOKENS


# ---------------------------------------------------------------------------
# RotatedButton, vertical text, fixed-position, button hides on expand.
# Same paint logic as the rig's rotated button, minus the observability
# logger import.
# ---------------------------------------------------------------------------

class RotatedButton(QtWidgets.QPushButton):
    """Button with vertical (270° rotated) text for sidebar toggles."""

    def __init__(self, text: str,
                 color: str = "#bd93f9",
                 hover_color: str = "#ff79c6",
                 pressed_color: str = "#8be9fd",
                 parent=None):
        super().__init__(text, parent)
        self._color = color
        self._hover_color = hover_color
        self._pressed_color = pressed_color

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(self.rect()), 5, 5)
        if self.isDown():
            painter.fillPath(path, QtGui.QColor(self._pressed_color))
        elif self.underMouse():
            painter.fillPath(path, QtGui.QColor(self._hover_color))
        else:
            painter.fillPath(path, QtGui.QColor(self._color))
        painter.translate(int(self.width() / 2), int(self.height() / 2))
        painter.rotate(-90)
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(self.font())
        text_rect = QtCore.QRect(
            int(-self.height() / 2), int(-self.width() / 2),
            int(self.height()), int(self.width())
        )
        painter.drawText(
            text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


# ---------------------------------------------------------------------------
# CollapsibleSidebar, overlay slide-out, exact behaviour as main GUI's.
# Vendored from source/gui/widgets/common.py:222.
# ---------------------------------------------------------------------------

class CollapsibleSidebar(QtWidgets.QFrame):
    """Animated overlay sidebar that slides in from the left.

    * Collapses on outside-click (event-filter) or close-button.
    * Animates width 0 ↔ expanded_width with 300 ms cubic ease-out.
    * ``on_collapse_callback`` fires when the sidebar closes, main_window
      uses it to re-show the floating RotatedButton.
    """

    def __init__(self, parent=None, header_text: str = "SIDEBAR",
                 expanded_width: int = 360):
        super().__init__(parent)
        self.is_collapsed = True
        self.collapsed_width = 0
        self.expanded_width = expanded_width
        self.parent_widget = parent
        self.on_collapse_callback: Optional[Callable[[], None]] = None
        self.header_text = header_text
        self._setup_ui()
        if parent is not None:
            parent.installEventFilter(self)
        # The parent filter only ever sees clicks that land on the parent's
        # OWN background. Every click on something you can actually click,
        # the table, a button, a tab, is delivered to that child, so the
        # sidebar stayed open no matter where you clicked. Watching the
        # application catches the press wherever it lands.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # ── Outside-click + parent-resize event filter ──────────────────

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if not self.is_collapsed and self._outside(event):
                self.collapse()
                # NOT consumed: the click was meant for whatever it landed on,
                # and swallowing it made the first click after opening the log
                # do nothing but close it.
                return False
        if (event.type() == QtCore.QEvent.Type.Resize
                and obj is self.parent_widget
                and not self.is_collapsed):
            self.setFixedHeight(self.parent_widget.height())
        return super().eventFilter(obj, event)

    def _outside(self, event) -> bool:
        """Whether this press landed anywhere but on the sidebar.

        Compared in SCREEN coordinates: the press is delivered to whichever
        widget was clicked, so its own `pos()` means nothing here, and the
        sidebar's geometry is in its parent's frame.
        """
        try:
            point = event.globalPosition().toPoint()
        except AttributeError:                        # pragma: no cover
            point = event.globalPos()
        top_left = self.mapToGlobal(self.rect().topLeft())
        # Against the width the sidebar is heading for, not the one it happens
        # to have mid-slide. The open animation takes 300 ms from zero, and
        # measuring the live width means a click during the slide lands
        # "outside" a rectangle that is still empty, shutting the log the user
        # just opened.
        width = self.expanded_width if not self.is_collapsed else self.width()
        size = QtCore.QSize(width, self.height())
        return not QtCore.QRect(top_left, size).contains(point)

    def mousePressEvent(self, event):
        """Clicks INSIDE the sidebar but not on an interactive widget
        also collapse, matches the main GUI behaviour."""
        if not self.is_collapsed and self.width() < self.expanded_width:
            # Still sliding in. Nothing occupies the point the cursor is over
            # yet, so childAt() calls every position background and this would
            # shut the log a moment after it was asked to open.
            super().mousePressEvent(event)
            return
        widget_at_pos = self.childAt(event.position().toPoint())
        interactive = (
            QtWidgets.QPushButton, QtWidgets.QLineEdit,
            QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit,
            QtWidgets.QComboBox, QtWidgets.QSpinBox,
            QtWidgets.QCheckBox, QtWidgets.QRadioButton,
            QtWidgets.QScrollBar,
        )
        if widget_at_pos is None or not isinstance(widget_at_pos, interactive):
            parent = widget_at_pos.parent() if widget_at_pos else None
            if parent and isinstance(parent, interactive):
                super().mousePressEvent(event)
            else:
                self.collapse()
        else:
            super().mousePressEvent(event)

    # ── Layout ──────────────────────────────────────────────────────

    def _setup_ui(self):
        p = TOKENS.palette
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setFixedWidth(self.collapsed_width)
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"QFrame {{ background: {p.surface}; "
            f"border-right: 1px solid {p.surface_border_strong}; }}"
        )

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        header = QtWidgets.QWidget()
        header.setFixedHeight(46)
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 8, 0)
        title = QtWidgets.QLabel(self.header_text)
        title.setStyleSheet(
            f"font-weight: 700; font-size: 11px; color: {p.text}; "
            "text-transform: uppercase; letter-spacing: 0.5px;")
        hl.addWidget(title)
        hl.addStretch(1)
        close_btn = QtWidgets.QPushButton("◀")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.collapse)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: qlineargradient("
            f"x1:0,y1:0,x2:1,y2:1, stop:0 #ec4899, stop:1 #a855f7); "
            f"color: white; border: none; border-radius: 4px; "
            f"font-weight: 700; }}"
            f"QPushButton:hover {{ background: qlineargradient("
            f"x1:0,y1:0,x2:1,y2:1, stop:0 #f472b6, stop:1 #c084fc); }}"
        )
        hl.addWidget(close_btn)
        self.main_layout.addWidget(header)

        # Scrollable content area.
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setAutoFillBackground(True)
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(8)
        self.scroll.setWidget(self.content)
        self.main_layout.addWidget(self.scroll)

        # Width-animation pair (min + max).
        self.anim_min = QtCore.QPropertyAnimation(self, b"minimumWidth")
        self.anim_min.setDuration(300)
        self.anim_min.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.anim_max = QtCore.QPropertyAnimation(self, b"maximumWidth")
        self.anim_max.setDuration(300)
        self.anim_max.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

    # ── Animate expand / collapse ────────────────────────────────────

    def expand(self):
        if not self.is_collapsed:
            return
        if self.parent_widget is not None:
            self.setFixedHeight(self.parent_widget.height())
            self.move(0, 0)
            self.raise_()
        self.is_collapsed = False
        self.anim_min.setStartValue(self.collapsed_width)
        self.anim_min.setEndValue(self.expanded_width)
        self.anim_max.setStartValue(self.collapsed_width)
        self.anim_max.setEndValue(self.expanded_width)
        self.anim_min.start()
        self.anim_max.start()

    def collapse(self):
        if self.is_collapsed:
            return
        self.is_collapsed = True
        self.anim_min.setStartValue(self.expanded_width)
        self.anim_min.setEndValue(self.collapsed_width)
        self.anim_max.setStartValue(self.expanded_width)
        self.anim_max.setEndValue(self.collapsed_width)
        self.anim_min.start()
        self.anim_max.start()
        if self.on_collapse_callback:
            try:
                self.on_collapse_callback()
            except Exception:
                pass

    def toggle(self):
        (self.expand if self.is_collapsed else self.collapse)()


# ---------------------------------------------------------------------------
# LogPanel, content for the sidebar. Stdlib logger bridge + change_log tail.
# ---------------------------------------------------------------------------

class LogPanel(QtWidgets.QWidget):
    """The content widget hosted inside the CollapsibleSidebar.

    * Subscribes a logging.Handler so every ``tools.offline_analysis.*``
      log line appears live (any thread → marshalled to the GUI thread).
    * On project load (``set_project_dir``), tails the project's
      ``change_log.jsonl`` for context.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_dir: Optional[Path] = None
        self._build_ui()
        self._install_logger_bridge()

    def _build_ui(self):
        p = TOKENS.palette
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.refresh_btn = QtWidgets.QPushButton("↻ Refresh")
        self.clear_btn   = QtWidgets.QPushButton("Clear view")
        for b in (self.refresh_btn, self.clear_btn):
            b.setStyleSheet(
                f"QPushButton {{ background: {p.surface_elev}; "
                f"color: {p.text}; border: 1px solid {p.surface_border_strong}; "
                f"border-radius: 4px; padding: 4px 10px; }}"
                f"QPushButton:hover {{ background: {p.surface_elev_2}; }}"
            )
        self.refresh_btn.clicked.connect(self.refresh)
        self.clear_btn.clicked.connect(lambda: self.view.clear())
        row.addWidget(self.refresh_btn)
        row.addWidget(self.clear_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.view = QtWidgets.QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(5000)
        self.view.setStyleSheet(
            f"QPlainTextEdit {{ background: {p.surface}; color: {p.text}; "
            f"border: 1px solid {p.surface_border}; "
            f"font-family: 'Cascadia Mono', 'Consolas', monospace; "
            f"font-size: 9pt; padding: 4px; }}"
        )
        v.addWidget(self.view, 1)

    def _install_logger_bridge(self):
        class _Handler(logging.Handler):
            def __init__(self, sink):
                super().__init__()
                self._sink = sink
            def emit(self, record):
                try:
                    self._sink(self.format(record))
                except Exception:
                    pass
        h = _Handler(self._append_async)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"))
        # Both trees, not just this tool's. A retrack runs the RIG's detector,
        # and when that detector refuses to load it says why on its own logger
        #, "SLEAP could not load the export at ...: tensorrt is required".
        # Listening only to `tools.offline_analysis` meant the operator saw
        # our one-line warning and never the sentence naming the cause.
        for name in ("tools.offline_analysis", "source.video"):
            root = logging.getLogger(name)
            root.addHandler(h)
            root.setLevel(logging.INFO)

    def _append_async(self, msg: str):
        QtCore.QMetaObject.invokeMethod(
            self.view, "appendPlainText",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, msg))

    def set_project_dir(self, project_dir: Optional[Path]):
        self._project_dir = Path(project_dir) if project_dir else None
        self.refresh()

    def refresh(self):
        if self._project_dir is None:
            self.view.appendPlainText(
                "── (no project loaded; rig change_log unavailable) ──")
            return
        log_path = self._project_dir / "change_log.jsonl"
        if not log_path.is_file():
            self.view.appendPlainText(f"── (no change_log.jsonl at {log_path}) ──")
            return
        try:
            with log_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-200:]
        except Exception as e:
            self.view.appendPlainText(f"[change_log read failed] {e}")
            return
        self.view.appendPlainText(
            f"── change_log.jsonl tail ({len(lines)} lines) ──")
        for line in lines:
            try:
                d = json.loads(line)
                self.view.appendPlainText(
                    f"  {d.get('ts','')}  {d.get('actor','')}  "
                    f"{d.get('action','')}  {d.get('source','')}")
            except Exception:
                self.view.appendPlainText(f"  {line.rstrip()}")


__all__ = ["RotatedButton", "CollapsibleSidebar", "LogPanel"]
