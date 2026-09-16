"""Nested session picker, modal dialog.

The flat 200-row table in the previous selector panel does not scale to a
project with hundreds of runs. This dialog presents sessions as a
**checkable QTreeWidget**, grouped by the user's choice of axis (Date,
Subject, Task, Box, or any column from the cohort metadata file). The user
expands a group, ticks individual rows or the group header (to tick all
children at once), filters by a free-text search, and clicks OK to commit.

Design:
  * Modal, user is gathering scope, not multitasking. Returns selection
    on accept; nothing happens on cancel.
  * Group-by combo at the top, Date / Subject / Task / Box, plus any
    cohort metadata columns when a cohort Excel is loaded.
  * Tri-state checks, checking a parent ticks all its children, unchecking
    unticks them.
  * Search box filters by substring across visible columns.
  * Bulk actions: All visible / None / Invert.
  * Status strip at the bottom: 'selected N / total · K subjects · D dates'.
  * Compact size by default; the tree itself is the only big widget.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from .session_catalog import (
    INTEGRITY_MISSING, INTEGRITY_MODIFIED, INTEGRITY_OK, INTEGRITY_UNLISTED,
    Session,
)
from .theming import style_button, style_label

logger = logging.getLogger(__name__)


# Column layout in the tree
COL_LABEL, COL_DATE, COL_SUBJECT, COL_TASK, COL_BOX, \
    COL_TIME, COL_VID, COL_MCU, COL_STATUS = range(9)

_DEFAULT_GROUPINGS: List[Tuple[str, str]] = [
    ("date",    "Date  →  Subject"),
    ("subject", "Subject  →  Date"),
    ("task",    "Task  →  Subject  →  Date"),
    ("box",     "Box  →  Subject  →  Date"),
    ("flat",    "(flat list, no grouping)"),
]


class SessionPickerDialog(QDialog):
    """Modal nested-tree picker.

    Usage:
        dlg = SessionPickerDialog(parent, sessions, cohort_df=ctx.cohort_df,
                                  preselected=current_selection)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            sessions = dlg.selected_sessions()
    """

    def __init__(self,
                 parent: Optional[QWidget],
                 sessions: List[Session],
                 *,
                 cohort_df: Optional[pd.DataFrame] = None,
                 preselected: Optional[List[Session]] = None,
                 title: str = "Select sessions"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._sessions = list(sessions)
        self._cohort_df = cohort_df
        self._preselected_ids = self._ids_of(preselected or [])
        self._suppress_recursion = False
        self._build_ui()
        self._populate()
        self.resize(820, 560)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # Top strip: group-by + search
        top = QHBoxLayout()
        top.addWidget(QLabel("Group by:"))
        self.group_combo = QComboBox()
        for key, label in _DEFAULT_GROUPINGS:
            self.group_combo.addItem(label, key)
        # Cohort columns (genotype/group/etc.) as extra grouping axes.
        if self._cohort_df is not None:
            for col in self._cohort_df.columns:
                col_l = str(col).strip().lower()
                if col_l in ("subject_id", "subject"):
                    continue   # already an axis
                self.group_combo.addItem(f"Cohort: {col}", f"cohort:{col}")
        self.group_combo.currentIndexChanged.connect(self._populate)
        top.addWidget(self.group_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("subject, task, date, status…")
        self.search_edit.textChanged.connect(self._apply_search)
        top.addWidget(self.search_edit, 1)
        root.addLayout(top)

        # Bulk action strip
        actions = QHBoxLayout()
        actions.setSpacing(4)
        for txt, slot in (("All visible", self._check_all_visible),
                          ("None", self._uncheck_all),
                          ("Invert", self._invert_all),
                          ("Expand all",   lambda: self.tree.expandAll()),
                          ("Collapse all", lambda: self.tree.collapseAll())):
            b = QPushButton(txt)
            style_button(b, "secondary")
            b.clicked.connect(slot)
            actions.addWidget(b)
        actions.addStretch(1)
        self.summary_lbl = QLabel("0 / 0 selected")
        style_label(self.summary_lbl, "muted")
        actions.addWidget(self.summary_lbl)
        root.addLayout(actions)

        # Tree
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(9)
        self.tree.setHeaderLabels(
            ["Group / Session", "Date", "Subject", "Task", "Box",
             "Time", "Vid", "MCU", "Status"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemChanged.connect(self._on_item_changed)
        h = self.tree.header()
        h.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        h.setStretchLastSection(False)
        h.setSectionResizeMode(COL_LABEL,
                               QtWidgets.QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.tree, 1)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        style_button(btns.button(QDialogButtonBox.StandardButton.Ok), "primary")
        style_button(btns.button(QDialogButtonBox.StandardButton.Cancel),
                     "secondary")
        root.addWidget(btns)

    # ------------------------------------------------------------------ Population

    def _populate(self):
        """Rebuild the tree according to the current Group-by selection."""
        self._suppress_recursion = True
        self.tree.clear()
        key = self.group_combo.currentData() or "date"
        groups = self._group_sessions(key)
        for group_label, items in groups:
            parent = QTreeWidgetItem([str(group_label)] + [""] * 8)
            parent.setFlags(parent.flags()
                            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                            | QtCore.Qt.ItemFlag.ItemIsAutoTristate)
            parent.setCheckState(COL_LABEL, QtCore.Qt.CheckState.Unchecked)
            parent.setFirstColumnSpanned(False)
            self.tree.addTopLevelItem(parent)
            for s in items:
                child = self._make_session_item(s)
                parent.addChild(child)
                if id(s) in self._preselected_ids:
                    child.setCheckState(COL_LABEL, QtCore.Qt.CheckState.Checked)
            parent.setExpanded(len(items) <= 12)
        self.tree.resizeColumnToContents(COL_LABEL)
        self._suppress_recursion = False
        self._update_summary()

    def _make_session_item(self, s: Session) -> QTreeWidgetItem:
        time_str = ""
        if s.started_at and "T" in s.started_at:
            time_str = s.started_at.split("T")[-1][:5]
        item = QTreeWidgetItem([
            s.label() or "<session>",
            s.date,
            s.subject,
            s.task,
            str(s.box) if s.box else "",
            time_str,
            "✓" if s.video_path else "–",
            "✓" if s.mcu_path else "–",
            s.integrity,
        ])
        item.setFlags(item.flags()
                      | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(COL_LABEL, QtCore.Qt.CheckState.Unchecked)
        item.setData(COL_LABEL, QtCore.Qt.ItemDataRole.UserRole, s)
        # Tint status cell
        color = {
            INTEGRITY_OK:        "#86efac",
            INTEGRITY_MODIFIED:  "#fcd34d",
            INTEGRITY_MISSING:   "#fb7185",
            INTEGRITY_UNLISTED:  "#cbd5e1",
        }.get(s.integrity, "#cbd5e1")
        item.setForeground(COL_STATUS, QtGui.QBrush(QtGui.QColor(color)))
        return item

    def _group_sessions(self, key: str) -> List[Tuple[str, List[Session]]]:
        if key == "flat":
            return [("All sessions", sorted(self._sessions, key=_sort_key))]

        def key_func(s: Session) -> Tuple[str, str]:
            if key == "date":
                k1 = s.date or "(no date)"
                k2 = "primary"
            elif key == "subject":
                k1 = s.subject or "(no subject)"
                k2 = "primary"
            elif key == "task":
                k1 = s.task or "(no task)"
                k2 = "primary"
            elif key == "box":
                k1 = f"Box {s.box}" if s.box else "(no box)"
                k2 = "primary"
            elif key.startswith("cohort:"):
                col = key.split(":", 1)[1]
                k1 = str(s.cohort_meta.get(col, "(no value)"))
                k2 = "primary"
            else:
                k1 = "All"
                k2 = "primary"
            return (k1, k2)

        bucket: Dict[str, List[Session]] = {}
        for s in self._sessions:
            k, _ = key_func(s)
            bucket.setdefault(k, []).append(s)

        # Sort buckets by key, sessions within bucket by date/started
        out: List[Tuple[str, List[Session]]] = []
        for k in sorted(bucket.keys(), reverse=(key == "date")):
            items = sorted(bucket[k], key=_sort_key)
            label = f"{k}   ({len(items)})"
            out.append((label, items))
        return out

    # ------------------------------------------------------------------ Slots

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._suppress_recursion or column != COL_LABEL:
            return
        self._update_summary()

    def _check_all_visible(self):
        self._set_all_checked(QtCore.Qt.CheckState.Checked, visible_only=True)

    def _uncheck_all(self):
        self._set_all_checked(QtCore.Qt.CheckState.Unchecked, visible_only=False)

    def _invert_all(self):
        self._suppress_recursion = True
        for _ in range(self.tree.topLevelItemCount()):
            pass
        for it in self._iter_session_items():
            new = (QtCore.Qt.CheckState.Unchecked
                   if it.checkState(COL_LABEL) == QtCore.Qt.CheckState.Checked
                   else QtCore.Qt.CheckState.Checked)
            it.setCheckState(COL_LABEL, new)
        self._suppress_recursion = False
        self._update_summary()

    def _set_all_checked(self, state: QtCore.Qt.CheckState, *,
                         visible_only: bool):
        self._suppress_recursion = True
        for it in self._iter_session_items():
            if visible_only and it.isHidden():
                continue
            it.setCheckState(COL_LABEL, state)
        self._suppress_recursion = False
        self._update_summary()

    def _apply_search(self, query: str):
        q = query.strip().lower()
        for top_idx in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(top_idx)
            any_visible = False
            for ch_idx in range(parent.childCount()):
                ch = parent.child(ch_idx)
                hay = " ".join(ch.text(c) for c in range(self.tree.columnCount())).lower()
                visible = (q in hay) if q else True
                ch.setHidden(not visible)
                any_visible = any_visible or visible
            parent.setHidden(not any_visible)
            if any_visible and q:
                parent.setExpanded(True)
        self._update_summary()

    # ------------------------------------------------------------------ Accessors

    def selected_sessions(self) -> List[Session]:
        out: List[Session] = []
        for it in self._iter_session_items():
            if it.checkState(COL_LABEL) == QtCore.Qt.CheckState.Checked:
                s = it.data(COL_LABEL, QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(s, Session):
                    out.append(s)
        return out

    # ------------------------------------------------------------------ Helpers

    def _iter_session_items(self):
        for ti in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(ti)
            for ci in range(parent.childCount()):
                yield parent.child(ci)

    def _update_summary(self):
        total = sum(1 for _ in self._iter_session_items())
        sel = sum(1 for it in self._iter_session_items()
                  if it.checkState(COL_LABEL) == QtCore.Qt.CheckState.Checked)
        subjects = {it.text(COL_SUBJECT) for it in self._iter_session_items()
                    if it.checkState(COL_LABEL) == QtCore.Qt.CheckState.Checked
                    and it.text(COL_SUBJECT)}
        dates = {it.text(COL_DATE) for it in self._iter_session_items()
                 if it.checkState(COL_LABEL) == QtCore.Qt.CheckState.Checked
                 and it.text(COL_DATE)}
        self.summary_lbl.setText(
            f"selected {sel} / {total}  ·  "
            f"{len(subjects)} subject(s)  ·  {len(dates)} date(s)")

    @staticmethod
    def _ids_of(sessions: List[Session]) -> set:
        return {id(s) for s in sessions}


def _sort_key(s: Session):
    return (s.date or "", s.started_at or "", s.subject or "", s.box or 0)
