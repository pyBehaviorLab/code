"""Cohort dialogs.

  * AssignSubjectsDialog, pick rows from the cohort DataFrame; the
    caller writes each row's ``Subject`` into the box whose
    ``box_number`` equals the row's ``SetupID``.

  * MetadataEditorDialog, table editor over the same DataFrame, saves
    via MetadataManager (used for inline tweaks after load).
"""

from __future__ import annotations

from typing import List

import pandas as pd
from PySide6 import QtCore, QtWidgets

from source.log import get_logger

logger = get_logger()


# =============================================================================
#  AssignSubjectsDialog
# =============================================================================


class AssignSubjectsDialog(QtWidgets.QDialog):
    """Pick one-or-more cohort rows; assignment is by ``SetupID``.

    Required columns in the DataFrame: ``Subject`` and ``SetupID`` (both
    enforced by ``MetadataManager._validate_and_coerce`` at load time).
    The dialog shows them first; other columns ride along for context.

    Usage:
        dlg = AssignSubjectsDialog(parent, df, metadata_filename="...")
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            for idx in dlg.selected_rows:
                row = df.iloc[idx]
                # row["SetupID"] → box_number,  row["Subject"] → subject id
    """

    REQUIRED_COLS = ("Subject", "SetupID")

    def __init__(self,
                 parent,
                 metadata_df: pd.DataFrame,
                 metadata_filename: str = "",
                 manager=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Subjects to Setups")
        self.resize(820, 460)

        for col in self.REQUIRED_COLS:
            if col not in metadata_df.columns:
                raise ValueError(
                    f"Cohort DataFrame is missing required column {col!r}. "
                    "The cohort file's first two columns must be Subject "
                    "and SetupID (case-sensitive, exact).")

        self._df = metadata_df
        # When set, the "Load New Meta" button can swap the cohort file in
        # place (load through the host, then re-read the manager's DataFrame).
        self._manager = manager
        self._selected_rows: List[int] = []

        layout = QtWidgets.QVBoxLayout(self)

        self._header = QtWidgets.QLabel(
            f"Cohort: {metadata_filename or '<unnamed>'}  |  "
            f"{len(metadata_df)} subjects available")
        layout.addWidget(self._header)

        hint = QtWidgets.QLabel(
            "Select one or more rows. Each picked row writes its "
            "<b>Subject</b> into the box whose number equals its "
            "<b>SetupID</b>.  Rows whose SetupID has no matching box "
            "are skipped with a warning.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(hint)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setSelectionBehavior(
            QtWidgets.QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(
            QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        # Hide the row-index column; select by clicking the data cells.
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        # Double-click any row to accept immediately with that row selected.
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self._populate_table()
        layout.addWidget(self.table)

        self._status = QtWidgets.QLabel("No rows selected")
        layout.addWidget(self._status)

        btns = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setDefault(True)
        self.btn_apply.setEnabled(False)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(self.btn_apply)
        # Between Apply and Cancel: load a different cohort file without leaving
        # the picker. Only shown when a manager+host can perform the load.
        if self._manager is not None and callable(
                getattr(parent, "load_cohort_metadata", None)):
            self.btn_load_new = QtWidgets.QPushButton("Load New Meta")
            self.btn_load_new.setToolTip(
                "Load a different cohort Excel/CSV and refresh this list")
            self.btn_load_new.clicked.connect(self._on_load_new)
            btns.addWidget(self.btn_load_new)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

    def _on_load_new(self) -> None:
        """Swap the cohort file in place: load through the host picker, then
        re-read the manager's DataFrame and rebuild the table."""
        host = self.parent()
        loader = getattr(host, "load_cohort_metadata", None)
        if not callable(loader):
            return
        try:
            ok = loader(offer_assign=False)   # opens the file picker + loads
        except Exception as e:
            logger.warning("Load New Meta failed: %s", e)
            return
        mgr = self._manager
        if not ok or mgr is None or getattr(mgr, "metadata_df", None) is None:
            return
        self._df = mgr.metadata_df
        self._selected_rows = []
        self.table.clearSelection()
        self._populate_table()
        self._header.setText(
            f"Cohort: {getattr(mgr, 'metadata_filename', '') or '<unnamed>'}  |  "
            f"{len(self._df)} subjects available")
        self._status.setText("No rows selected")
        self.btn_apply.setEnabled(False)

    def _populate_table(self) -> None:
        df = self._df
        # Subject + SetupID first, then everything else in source order.
        rest = [c for c in df.columns if c not in self.REQUIRED_COLS]
        ordered_cols = list(self.REQUIRED_COLS) + rest

        self.table.setColumnCount(len(ordered_cols))
        self.table.setHorizontalHeaderLabels(ordered_cols)
        self.table.setRowCount(len(df))

        flags = (QtCore.Qt.ItemFlag.ItemIsSelectable
                 | QtCore.Qt.ItemFlag.ItemIsEnabled)
        for r in range(len(df)):
            row = df.iloc[r]
            for c, col_name in enumerate(ordered_cols):
                val = row.get(col_name)
                text = "" if pd.isna(val) else str(val)
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(flags)
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()

    def _selected_row_indices(self) -> List[int]:
        sel = self.table.selectionModel().selectedRows()
        return sorted({idx.row() for idx in sel})

    def _on_selection_changed(self) -> None:
        rows = self._selected_row_indices()
        n = len(rows)
        if n == 0:
            self._status.setText("No rows selected")
        else:
            setups = [str(self._df.iloc[i]["SetupID"]) for i in rows]
            self._status.setText(
                f"{n} row(s) selected → SetupIDs: {', '.join(setups)}")
        self.btn_apply.setEnabled(n > 0)

    def _on_apply(self) -> None:
        rows = self._selected_row_indices()
        if not rows:
            return
        self._selected_rows = rows
        self.accept()

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Double-click any cell → assign that row immediately. If the
        user had multiple rows selected, the whole selection is applied
        (matches the Apply button); otherwise just the double-clicked
        row is applied.
        """
        rows = self._selected_row_indices()
        if row not in rows:
            rows = [row]
        self._selected_rows = sorted(set(rows))
        self.accept()

    @property
    def selected_rows(self) -> List[int]:
        return list(self._selected_rows)


# =============================================================================
#  SubjectCardDialog, compact view when the box already has a subject
# =============================================================================


class SubjectCardDialog(QtWidgets.QDialog):
    """Per-box subject card.

    Shown when Box(N)'s Subj** button is clicked and the box already
    has a subject populated. Lists that subject's cohort row as Field /
    Value rows; offers a "Pick different" button that flips the dialog
    result to ``PickDifferent`` so the caller can open the cohort
    picker.
    """

    PickDifferent = QtWidgets.QDialog.DialogCode.Accepted + 1  # type: ignore[assignment]

    def __init__(self, parent, *,
                 setup_number: int,
                 subject: str,
                 row: dict):
        super().__init__(parent)
        self.setWindowTitle(f"Subject, Box {setup_number}")
        self.setMinimumWidth(360)
        self._wants_pick_different = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        title = QtWidgets.QLabel(
            f"<b>Box {setup_number}</b> &nbsp;·&nbsp; Subject: "
            f"<span style='color:#22c55e'>{subject or '(none)'}</span>")
        title.setStyleSheet("font-size: 11pt;")
        root.addWidget(title)

        tbl = QtWidgets.QTableWidget(0, 2)
        tbl.setHorizontalHeaderLabels(["Field", "Value"])
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        # Show every non-empty field from the row except the two
        # identity columns (Subject is in the title; SetupID == box).
        entries = []
        for k, v in (row or {}).items():
            if k in ("Subject", "SetupID"):
                continue
            text = "" if v is None else str(v)
            if text == "" or text.lower() == "nan":
                continue
            entries.append((str(k), text))
        tbl.setRowCount(len(entries))
        for r, (k, v) in enumerate(entries):
            tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
            tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(v))
        tbl.resizeColumnsToContents()
        root.addWidget(tbl, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_pick = QtWidgets.QPushButton("Pick different…")
        self.btn_pick.clicked.connect(self._on_pick_different)
        btn_row.addWidget(self.btn_pick)
        btn_row.addStretch(1)
        # Load a different cohort file, then jump straight to the picker with
        # it (same as Pick-different, but on the new cohort).
        if callable(getattr(parent, "load_cohort_metadata", None)):
            self.btn_load_new = QtWidgets.QPushButton("Load New Meta")
            self.btn_load_new.setToolTip(
                "Load a different cohort Excel/CSV and pick from it")
            self.btn_load_new.clicked.connect(self._on_load_new)
            btn_row.addWidget(self.btn_load_new)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setDefault(True)
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    def _on_pick_different(self) -> None:
        self._wants_pick_different = True
        self.accept()

    def _on_load_new(self) -> None:
        host = self.parent()
        loader = getattr(host, "load_cohort_metadata", None)
        if not callable(loader):
            return
        try:
            ok = loader(offer_assign=False)   # opens the file picker + loads
        except Exception as e:
            logger.warning("Load New Meta (card) failed: %s", e)
            return
        if ok:
            self._wants_pick_different = True   # caller opens the picker next
            self.accept()

    @property
    def wants_pick_different(self) -> bool:
        return self._wants_pick_different


# =============================================================================
#  MetadataEditorDialog, table editor saved back to the cohort file
# =============================================================================


class MetadataEditorDialog(QtWidgets.QDialog):
    """Edit the loaded cohort in place. Subject and SetupID stay
    validated on save (manager enforces the two required columns).
    """

    def __init__(self, metadata_manager, parent=None):
        super().__init__(parent)
        self.metadata_manager = metadata_manager
        self.setWindowTitle("Edit Cohort")
        self.resize(820, 600)
        self._build_ui()
        self._load_data()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Edit cohort entries. The first two columns (Subject, "
            "SetupID) are required and must stay populated.")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QtWidgets.QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        # Hide the row-index column.
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("Add Row")
        self.add_row_btn.clicked.connect(self._add_row)
        row.addWidget(self.add_row_btn)
        self.remove_row_btn = QtWidgets.QPushButton("Remove Row")
        self.remove_row_btn.clicked.connect(self._remove_row)
        row.addWidget(self.remove_row_btn)
        row.addStretch()
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        row.addWidget(self.save_btn)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        row.addWidget(self.close_btn)
        layout.addLayout(row)

    def _load_data(self):
        df = self.metadata_manager.metadata_df
        if df is None:
            QtWidgets.QMessageBox.warning(
                self, "No Cohort",
                "No cohort loaded. Use 'Load Metadata' first.")
            self.reject()
            return
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        for r, (_, row) in enumerate(df.iterrows()):
            for c, val in enumerate(row):
                text = "" if pd.isna(val) else str(val)
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _add_row(self):
        n = self.table.rowCount()
        self.table.insertRow(n)
        for c in range(self.table.columnCount()):
            self.table.setItem(n, c, QtWidgets.QTableWidgetItem(""))

    def _remove_row(self):
        r = self.table.currentRow()
        if r < 0:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Select a row first.")
            return
        if QtWidgets.QMessageBox.question(
                self, "Confirm", f"Delete row {r + 1}?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
        ) == QtWidgets.QMessageBox.StandardButton.Yes:
            self.table.removeRow(r)

    def _save(self):
        try:
            cols = [self.table.horizontalHeaderItem(i).text()
                    for i in range(self.table.columnCount())]
            rows = [
                [(self.table.item(r, c).text() if self.table.item(r, c) else "")
                 for c in range(self.table.columnCount())]
                for r in range(self.table.rowCount())
            ]
            new_df = pd.DataFrame(rows, columns=cols)
            # Re-run the manager's validator before persisting.
            new_df = self.metadata_manager._validate_and_coerce(new_df)
            self.metadata_manager.metadata_df = new_df
            fp = self.metadata_manager.metadata_filename
            if fp:
                new_df.to_excel(fp, index=False)
            QtWidgets.QMessageBox.information(
                self, "Saved", "Cohort updated.")
            self.accept()
        except Exception as e:
            logger.error("Edit cohort save failed: %s", e)
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save:\n{e}")
