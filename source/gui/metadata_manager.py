"""Cohort metadata manager.

Single source of truth for the loaded cohort. The Excel/CSV file MUST
have two columns:

  * ``Subject``: subject identifier (string). Written to the box's
                   ``subject_id_edit`` when the user assigns this row.
  * ``SetupID``: integer. The box number this subject belongs to.
                   Must match ``BoxConfig.box_number``.

Both columns are compulsory. No aliases, files with ``Subject_ID`` /
``MouseID`` / ``Box_ID`` are rejected at load time. Rename the columns
in the spreadsheet before loading.

Public surface:

    mgr = MetadataManager()
    mgr.load_metadata(filename, parent)        - file picker + parse
    mgr.ensure_loaded(cfg, parent)             - silent reload from cfg
    mgr.assign_subjects_from_metadata(host)    - open dialog + assign
    mgr.row_for_setup(setup_id) -> dict | None - lookup by SetupID
    mgr.metadata_df                            - the loaded DataFrame
    mgr.metadata_filename                      - path the user picked

The host parameter exposes ``iter_box_subject_widgets() ->
Iterable[(box_number: int, subject_id_edit_widget)]`` so the same
manager works for operant + maze without a mode switch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from PySide6 import QtWidgets

from source.log import get_logger

logger = get_logger()


REQUIRED_COLUMNS = ("Subject", "SetupID")
# Note: ROI / video_segment_config persistence lives in the
# source.config.experiment bridge; this module is metadata-only.


class MetadataManager:
    """Owns the cohort DataFrame for the current session."""

    def __init__(self) -> None:
        self.metadata_df: Optional[pd.DataFrame] = None
        self.metadata_filename: str = ""

    # ---- loading ---------------------------------------------------------

    def load_metadata(self, filename: Optional[str] = None,
                      parent=None) -> bool:
        """Parse a cohort file into ``self.metadata_df``.

        Validates both required columns exist; coerces ``SetupID`` to
        int, ``Subject`` to str. Returns True on success.
        """
        try:
            if not filename:
                filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    parent, "Select Cohort File",
                    "", "Excel/CSV (*.xlsx *.xls *.csv)")
            if not filename:
                return False

            df = self._parse_file(filename, parent)
            if df is None:
                return False

            df = self._validate_and_coerce(df)
            self.metadata_df = df
            self.metadata_filename = filename
            logger.info(
                "Cohort loaded: %d rows from %s | SetupIDs=%s",
                len(df), filename,
                sorted(df["SetupID"].unique().tolist()),
            )
            return True

        except Exception as e:
            msg = f"Failed to load cohort: {e}"
            logger.error(msg)
            if parent is not None:
                QtWidgets.QMessageBox.critical(parent, "Metadata", msg)
            return False

    def ensure_loaded(self, cfg, parent=None,
                      project_dir: Optional[Path] = None) -> bool:
        """If ``metadata_df`` is empty but ``cfg.meta.metadata_file`` is
        set on disk, silently reload it. Returns True iff metadata_df is
        populated after the call. ``project_dir`` is used to resolve a
        relative ``metadata_file`` under ``<project_dir>/metadata/``.
        """
        if self.metadata_df is not None and len(self.metadata_df) > 0:
            return True
        path_str = (getattr(getattr(cfg, "meta", None), "metadata_file", "")
                    if cfg is not None else "")
        if not path_str:
            return False
        candidate = Path(path_str)
        if not candidate.is_absolute() and project_dir is not None:
            candidate = Path(project_dir) / "metadata" / candidate
        if not candidate.is_file():
            logger.debug("ensure_loaded: %s not found on disk", candidate)
            return False
        return self.load_metadata(str(candidate), parent)

    def prompt_load_if_missing(self, host) -> bool:
        """Ensure metadata is loaded; if not, offer to load it right here.

        Returns True iff metadata is populated afterwards. Pops a dialog with
        a **Load Metadata…** button so the user can pick a cohort file in
        place. Loading goes through ``host.load_cohort_metadata(
        offer_assign=False)`` (load + copy into the project), with the assign
        follow-up suppressed because the caller continues its own flow."""
        cfg = getattr(host, "_active_config", None)
        proj_dir = getattr(host, "_active_project_dir", None)
        if self.ensure_loaded(cfg, parent=host, project_dir=proj_dir):
            return True

        box = QtWidgets.QMessageBox(host)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Metadata")
        box.setText("No cohort metadata is loaded.")
        box.setInformativeText(
            "Load a cohort file now, or do it later from "
            "Experiment Info → Load Metadata.")
        load_btn = box.addButton(
            "Load Metadata…", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(load_btn)
        box.exec()
        if box.clickedButton() is not load_btn:
            return False

        loader = getattr(host, "load_cohort_metadata", None)
        if callable(loader):
            try:
                loader(offer_assign=False)
            except Exception as e:
                logger.warning("in-context metadata load failed: %s", e)
        return self.ensure_loaded(cfg, parent=host, project_dir=proj_dir)

    # ---- lookup ----------------------------------------------------------

    def row_for_setup(self, setup_id: int) -> Optional[Dict[str, Any]]:
        """Return the first row whose ``SetupID`` matches, as a dict,
        or None when no row matches / no metadata loaded.

        ``Subject`` and ``SetupID`` are included in the returned dict
        (callers strip them if they want only "other" metadata).
        """
        if self.metadata_df is None or self.metadata_df.empty:
            return None
        try:
            sid = int(setup_id)
        except (TypeError, ValueError):
            return None
        hits = self.metadata_df[self.metadata_df["SetupID"] == sid]
        if hits.empty:
            return None
        row = hits.iloc[0]
        return {str(k): ("" if pd.isna(v) else str(v))
                for k, v in row.items()}

    # ---- per-box card (card-first when subject already populated) -------

    def show_subject_card_or_assign(self, host, setup_number: int) -> None:
        """Per-box Subj** click.

        * If Box(N) has no subject text → open the cohort picker directly.
        * If Box(N) has subject text AND the cohort row exists → show a
          compact ``SubjectCardDialog`` with that subject's fields and a
          "Pick different" button. On "Pick different", open the picker.
        * If the typed subject does NOT match any cohort row → open the
          picker (the user is likely re-picking after a typo).
        """
        if not self.prompt_load_if_missing(host):
            return

        # Find this box's subject text without forcing the host to expose
        # a per-box getter, iter_box_subject_widgets covers it.
        current_text = ""
        try:
            for bn, w in host.iter_box_subject_widgets():
                if int(bn) == int(setup_number):
                    current_text = (w.text() or "").strip()
                    break
        except Exception as e:
            logger.debug("subject text lookup failed: %s", e)

        row = self.row_for_setup(setup_number)
        # No subject yet, OR subject text doesn't match the cohort row's
        # Subject → skip the card and go straight to the picker.
        cohort_subject = (str(row.get("Subject") or "") if row else "").strip()
        if not current_text or not row or cohort_subject != current_text:
            self.assign_subjects_from_metadata(host)
            return

        from source.gui.dialogs.subjects import SubjectCardDialog
        card = SubjectCardDialog(
            host, setup_number=int(setup_number),
            subject=current_text, row=row,
        )
        card.exec()
        if card.wants_pick_different:
            self.assign_subjects_from_metadata(host)

    # ---- assignment dialog (the one entry point) ------------------------

    def assign_subjects_from_metadata(self, host) -> Dict[int, str]:
        """Open ``AssignSubjectsDialog`` and write the picked Subjects
        into the host's box widgets keyed by ``SetupID``.

        ``host`` must expose:
            iter_box_subject_widgets() -> Iterable[(box_number, widget_with_setText)]

        Returns a dict ``{setup_id: subject}`` for the rows that landed.
        """
        if not self.prompt_load_if_missing(host):
            return {}

        from source.gui.dialogs.subjects import AssignSubjectsDialog
        dlg = AssignSubjectsDialog(
            host, self.metadata_df,
            metadata_filename=self.metadata_filename,
            manager=self,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return {}

        widgets_by_setup: Dict[int, Any] = {}
        try:
            for setup_number, w in host.iter_box_subject_widgets():
                widgets_by_setup[int(setup_number)] = w
        except Exception as e:
            logger.error("iter_box_subject_widgets failed: %s", e)
            return {}

        assigned: Dict[int, str] = {}
        skipped: list[int] = []
        for row_idx in dlg.selected_rows:
            row = self.metadata_df.iloc[row_idx]
            try:
                sid = int(row["SetupID"])
            except (TypeError, ValueError):
                continue
            subject = str(row["Subject"]).strip()
            widget = widgets_by_setup.get(sid)
            if widget is None:
                skipped.append(sid)
                logger.warning(
                    "Cohort row SetupID=%s has no matching box widget", sid)
                continue
            widget.setText(subject)
            assigned[sid] = subject

        if assigned:
            lines = "\n".join(f"  SetupID {sid}: {sub}"
                              for sid, sub in sorted(assigned.items()))
            extra = ""
            if skipped:
                extra = (f"\n\nSkipped {len(skipped)} row(s) whose SetupID "
                         f"has no box configured: {sorted(set(skipped))}")
            QtWidgets.QMessageBox.information(
                host, "Subjects Assigned",
                f"Assigned {len(assigned)} subject(s):\n\n{lines}{extra}")
        return assigned

    # ---- internal --------------------------------------------------------

    @staticmethod
    def _parse_file(filename: str, parent) -> Optional[pd.DataFrame]:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(filename)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        excel_file = pd.ExcelFile(path)
        sheets = excel_file.sheet_names
        if len(sheets) > 1:
            if parent is None:
                logger.info("Multi-sheet workbook (%s) loaded with no parent; "
                            "using first sheet %r.", path, sheets[0])
                return pd.read_excel(path, sheet_name=sheets[0])
            sheet, ok = QtWidgets.QInputDialog.getItem(
                parent, "Select Sheet",
                f"{len(sheets)} sheets found. Pick one:",
                sheets, 0, False)
            if not ok:
                return None
            return pd.read_excel(path, sheet_name=sheet)
        return pd.read_excel(path)

    @staticmethod
    def _validate_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Cohort file has no rows.")
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Required column(s) missing: {missing}. "
                f"The cohort file must have a column named 'Subject' and "
                f"a column named 'SetupID' (exact, case-sensitive).")
        df["Subject"] = df["Subject"].astype(str).str.strip()
        try:
            df["SetupID"] = df["SetupID"].astype(int)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'SetupID' column must be integer-valued. Got non-int "
                f"entries: {e}") from e
        return df


