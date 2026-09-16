"""A layout that wraps its items onto as many rows as it needs.

A `QHBoxLayout` of toggles has one width and refuses to be narrower. That is
fine until the panel it lives in is narrower than that, then the row is not
squeezed, it is CLIPPED, and on a display running at 125% or 150% scaling the
last toggles simply are not on screen. Nothing says so; they are just gone.

This reflows instead: same items, more rows, whatever width it is given.
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class FlowLayout(QLayout):
    """Left-to-right, wrapping onto a new row when the current one is full."""

    def __init__(self, parent=None, margin: int = 0, spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self.setContentsMargins(QMargins(margin, margin, margin, margin))
        self.setSpacing(spacing)

    # ── QLayout plumbing ─────────────────────────────────────────

    def addItem(self, item):
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    # ── the wrapping itself ──────────────────────────────────────

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        """The widest SINGLE item, not the sum of them.

        This is what lets the panel be narrow: the layout promises only that
        it can fit one toggle per row, so it never forces its parent wider
        than its biggest child.
        """
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect: QRect, *, apply: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = eff.x(), eff.y(), 0
        space = self.spacing()

        for item in self._items:
            w = item.sizeHint()
            if not item.widget() or not item.widget().isVisibleTo(
                    self.parentWidget()):
                # A hidden toggle must not reserve a slot, or the row
                # wraps around empty gaps.
                if item.widget() is not None and item.widget().isHidden():
                    continue
            nxt = x + w.width()
            if nxt > eff.right() and row_h > 0:
                x = eff.x()
                y += row_h + space
                nxt = x + w.width()
                row_h = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), w))
            x = nxt + space
            row_h = max(row_h, w.height())
        return y + row_h - rect.y() + m.bottom()
