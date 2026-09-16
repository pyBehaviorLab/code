"""Ask which sheet column holds the subject IDs.

Lifted from pyBehaveTrack's metadata_manager, which is 37 kB of sheet
handling the analyser does not need; this is the one dialog its Analyze
tab calls.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6 import QtWidgets


#: Column headers that usually mean "subject". Only ever used to PRE-SELECT an
#: entry in the dialog, never to decide. The caller stopped auto-detecting on
#: purpose, because matching hardcoded names misfired on sheets with odd
#: headers; a wrong preselection costs one click, a wrong auto-detection costs
#: a silently mis-joined cohort.
_SUBJECT_ID_ALIASES = (
    "subject", "subjectid", "subjectno", "subjectnumber",
    "mouse", "mouseid", "mouseno", "mousenumber",
    "animal", "animalid", "animalno",
    "rat", "ratid", "id",
)


def _normalise(name: str) -> str:
    """Lower-case, with spaces, underscores, hyphens and dots removed.

    So ``Subject ID``, ``subject_id`` and ``Subject-Id`` all reach the same
    key, which is the only reason a fixed alias list is usable at all.
    """
    out = str(name).strip().lower()
    for ch in (" ", "_", "-", ".", "#"):
        out = out.replace(ch, "")
    return out


def _guess_column(cols, aliases):
    """The first column matching an alias, or ``""``.

    Exact normalised matches first, then a containment pass, so a sheet with
    both ``Subject`` and ``SubjectWeight`` preselects the former.
    """
    normalised = [(c, _normalise(c)) for c in cols]
    for alias in aliases:
        for original, key in normalised:
            if key == alias:
                return original
    for alias in aliases:
        for original, key in normalised:
            if alias in key:
                return original
    return ""


def pick_subject_column(df: pd.DataFrame,
                         parent=None,
                         title: str = "Select Subject-ID column",
                         hint: str = (
                             "Which column in this sheet holds the subject IDs "
                             "that will be matched to recordings?")) -> Optional[str]:
    """Modal dialog: ask the user which column is the Subject-ID.

    Pre-selects a sensible alias if one is present. Returns the chosen
    column name, or ``None`` if cancelled.
    """
    cols = [str(c) for c in df.columns]
    if not cols:
        return None
    preselect = _guess_column(cols, _SUBJECT_ID_ALIASES)
    default_idx = cols.index(preselect) if preselect in cols else 0
    col, ok = QtWidgets.QInputDialog.getItem(
        parent, title, hint, cols, default_idx, False)
    if not ok or not col:
        return None
    return col

