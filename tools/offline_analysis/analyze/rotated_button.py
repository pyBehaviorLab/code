"""A vertical toggle button that paints its label rotated 270°.

Used for the collapsible-sidebar tabs (Experiment Info / Error Log /
Documentation). Was defined three times as an identical function-local class
inside MainWindow, differing only in its three theme colors, now one
parameterized widget.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class RotatedButton(QtWidgets.QPushButton):
    """QPushButton whose text is drawn rotated 270°, with a rounded themed
    background (``down`` when pressed, ``hover`` on mouse-over, else ``base``)."""

    def __init__(self, text: str = "", down: str = "#8be9fd",
                 hover: str = "#ff79c6", base: str = "#bd93f9", parent=None,
                 ink: str = "", ink_active: str = "", edge: str = ""):
        super().__init__(text, parent)
        self._c_down = down
        self._c_hover = hover
        self._c_base = base
        #: Label colour when quiet and when this tab's page is open. Empty
        #: keeps the original white-on-everything, for the shell's sidebars.
        self._c_ink = ink
        self._c_ink_active = ink_active
        #: A hairline, so a quiet tab still reads as a tab rather than as a
        #: gap in the panel edge.
        self._c_edge = edge

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(self.rect()), 5, 5)

        # CHECKED, not just pressed: `down` is the state of a mouse button,
        # and the tab has to go on saying which page is open after the click
        # has been released.
        active = self.isChecked() or self.isDown()
        if active:
            painter.fillPath(path, QtGui.QColor(self._c_down))
        elif self.underMouse():
            painter.fillPath(path, QtGui.QColor(self._c_hover))
        else:
            painter.fillPath(path, QtGui.QColor(self._c_base))
        if self._c_edge and not active:
            painter.setPen(QtGui.QPen(QtGui.QColor(self._c_edge), 1))
            painter.drawPath(path)

        painter.translate(int(self.width() / 2), int(self.height() / 2))
        painter.rotate(-90)  # 270 degrees
        ink = (self._c_ink_active or "white") if active else (self._c_ink
                                                              or "white")
        painter.setPen(QtGui.QColor(ink))
        painter.setFont(self.font())

        text_rect = QtCore.QRect(
            int(-self.height() / 2),
            int(-self.width() / 2),
            int(self.height()),
            int(self.width()),
        )
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()
