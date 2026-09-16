"""
Analyze Tab v1, Post-hoc tracking data analysis with interactive zone viewer.

Offline analysis of _video_data.txt files produced by pyBehaveTrack's recording
pipeline.  Computes per-session and per-time-bin measures (distance, speed,
freezing, zone occupancy, latency) with per-setup calibration extracted from
embedded scale zones.

Main components
---------------
AnalyzeTab          Main PySide6 widget (tab in the application).
PipelineWorker      QThread that executes each recording's plan.
AnalysisWorker      Legacy per-file worker; still the options→engine bridge.
SessionFile         Dataclass holding parsed header metadata for one .txt file.

There are no "modes". Every recording either has a clock, poses, a space
(zones + scale) and up-to-date measures, or it is missing some of them, and
the work is whatever is missing. ``source.analysis.session_bundle.Bundle.plan``
computes that list, the Plan column shows it, and
``source.analysis.pipeline.run_all`` executes exactly it, so the tab cannot
announce work it will not do.

Data format
-----------
``source.core.video_data_schema`` is the one definition of the session file and
the one reader for it. Every dialect, the current columns, the legacy
``actual_ts`` header, the legacy ``B``/``M``/``V`` lines, one-JSON-per-line,
is adapted there, so nothing in this file parses the format by hand.

Calibration
-----------
Each setup can have a different physical scale.  The scale zone embedded in the
Zones line carries ``scale_length`` and ``scale_unit`` which define the
pixels-per-cm ratio for that specific arena.  The analysis extracts this per-file
so that multi-arena experiments with different camera distances are handled
correctly.  A global fallback value is used when no scale zone is present.

Inspired by the ANY-maze retracking workflow.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QFrame, QMenu,
)

import logging

from tools.offline_analysis.analyze import analysis_theme as _THEME

logger = logging.getLogger(__name__)


# ── Frame data column layout ───────────────────────────────────────
#
# There is no column layout here any more. `source.core.video_data_schema`
# is the one definition and the one reader; this file had its own map, the
# analysis engine had another, and the offline writer invented a third, which
# is how a retrack came to emit a header no reader understood.

# NOTE: px/cm is NOT re-derived here. It is computed once at record time by
# experiment_tab._extract_px_per_cm_from_zones() (which denormalizes the scale
# zone against the live ROI frame), written to the units line of
# _video_data.txt, and read back by offline_analysis. If a fallback for
# uncalibrated sessions is ever needed, share the experiment_tab logic rather
# than reimplementing it: a second implementation has to denormalize against
# the live ROI frame too, and one that skips that is silently wrong.


# ── Presentation ────────────────────────────────────────────────
#
# Presentation lives in `source.gui.widgets.analysis_theme`: one stylesheet on
# the window, objectNames on the few widgets that are a different KIND of
# thing. Inline style constants here would drift from it, so there are none.

#: A recording this workbench will open as a bare video, with no data file
#: beside it yet.
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv")


class Section(QFrame):
    """A titled block of options that folds away.

    Four stacked groups need more height than the window has, and which of
    them you care about depends entirely on what you are doing: someone
    re-running a cohort with settled settings wants CORRECTIONS out of the
    way, someone calibrating a new rig wants nothing but ZONES & SCALE.

    Deliberately not a checkable QGroupBox, which is the obvious Qt idiom:
    that DISABLES its children when unchecked, and re-enabling them on expand
    would undo the per-control enabling that says a setting has been
    overruled by another.
    """

    def __init__(self, title: str, *, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("sectionBox")
        self._title = title
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.head = QPushButton()
        self.head.setObjectName("sectionHead")
        self.head.setCheckable(True)
        self.head.setChecked(expanded)
        self.head.setCursor(Qt.CursorShape.PointingHandCursor)
        self.head.setToolTip("Click to fold this section away.")
        self.head.clicked.connect(self._toggle)
        v.addWidget(self.head)

        self._body = QWidget()
        v.addWidget(self._body)
        self._body.setVisible(expanded)
        self._sync_head()

    def body(self) -> QWidget:
        """The widget to lay the section's controls out in."""
        return self._body

    def title(self) -> str:
        """The section's name, as QGroupBox exposed it."""
        return self._title

    def set_title(self, title: str):
        self._title = title
        self._sync_head()

    def set_expanded(self, on: bool):
        self.head.setChecked(bool(on))
        self._toggle(bool(on))

    @property
    def expanded(self) -> bool:
        return self.head.isChecked()

    def _toggle(self, on: bool):
        self._body.setVisible(bool(on))
        self._sync_head()

    def _sync_head(self):
        self.head.setText(("▾  " if self.head.isChecked() else "▸  ")
                          + self._title)


class StickyMenu(QMenu):
    """A checklist menu that stays open while you tick things.

    Qt closes a menu on any action click, which for a multi-select list means
    one trip through the menu per item, pick, menu shuts, reopen, pick again.
    Checkable actions toggle in place instead; everything else behaves.
    """

    def mouseReleaseEvent(self, e):
        act = self.activeAction()
        if act is not None and act.isEnabled() and act.isCheckable():
            act.trigger()                 # toggles and emits; menu stays up
            e.accept()
            return
        super().mouseReleaseEvent(e)


#: The " · 7.9 px/cm" tail of a Space cell, shown on the right instead.
_PX_PER_CM_CELL = re.compile(r"\s*·\s*[\d.]+\s*px/cm")

#: What the user wants done to the selected recordings. Unlike the old "mode"
#: selector, which was stored in a key nobody read while Run hard-coded
#: DeepLabCut, each of these sets state on the bundles, so the Plan column
#: and the run both follow from it.
INTENT_ANALYSE = "analyse"     # measure the poses that are already there
INTENT_CORRECT = "correct"     # write the corrections out, then measure those
INTENT_RETRACK = "retrack"     # run a detector over the video again


# ── Data structures ─────────────────────────────────────────────

@dataclass
class SessionFile:
    """Metadata for one loaded _video_data.txt file.

    Populated by ``parse_txt_header()`` which reads only the header lines
    (info lines, Zones line, B/M lines) without loading frame data.
    The ``header`` dict contains all parsed key-value pairs and is the
    source for body_parts, subject_id, resolution, etc.
    """
    txt_path: str
    video_path: str = ""
    subject: str = ""
    date: str = ""
    duration_s: float = 0.0
    n_frames: int = 0
    zones: List[str] = field(default_factory=list)
    header: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.txt_path)


# ── Video finder ────────────────────────────────────────────────
#
# Lives in `source.analysis.session_bundle` so the headless engine and the CLI
# can find a recording's video without importing Qt. Re-exported here because
# this module's callers have always used it by this name.
from tools.offline_analysis.engine.session_bundle import find_video_for_txt  # noqa: E402,F401


# ── Quick header parser ─────────────────────────────────────────

def parse_txt_header(txt_path: str) -> SessionFile:
    """Header metadata for one recording, without loading the frame data.

    Everything comes from :mod:`source.core.video_data_schema`, the one
    reader, so a legacy ``B``/``M``/``V`` file, a legacy ``actual_ts`` header
    and a current file all populate the same fields. This function used to
    contain its own scanner, including a positional fallback that landed on
    the right timestamp column by luck rather than by agreement.

    Duration is measured from the first and last timestamps; the tail of a
    large file is read directly rather than streaming all of it.
    """
    from tools.offline_analysis import video_data_schema as vds

    sf = SessionFile(txt_path=txt_path)
    try:
        hdr = vds.VideoDataHeader.parse(txt_path)
    except Exception as e:                                # unreadable file
        logger.warning("Error parsing %s: %s", txt_path, e)
        return sf

    header: Dict[str, Any] = dict(hdr.info)
    if hdr.body_parts:
        header["body_parts"] = hdr.body_parts
    sf.header = header
    sf.subject = hdr.subject_id or header.get("mouseID", "")
    sf.date = header.get("expt_date", "") or hdr.start_time[:10]
    sf.zones = [z.get("name", "") for z in vds.zones_as_list(hdr.zones)
                if z.get("name")]

    # One entry per data row, so the count comes with the times. This used
    # to build a row dict per line, the same 250,000 dicts the clock was
    # building right after it, for the same two numbers.
    ts = vds.timestamps(txt_path, hdr)
    sf.n_frames = len(ts)
    good = [t for t in ts if not math.isnan(t)]
    if good:
        sf.duration_s = (good[-1] - good[0]) / 1000.0
    sf.video_path = find_video_for_txt(txt_path)
    return sf


# ── Frame loader / position helpers ─────────────────────────────

def _load_session_frames(txt_path: str) -> Tuple[Dict[str, Any], Any, pd.DataFrame, list]:
    """Every per-frame row as a DataFrame, via the one reader.

    Returns ``(header_dict, zones, frames_df, body_parts)``. The frame columns
    are the canonical ones, ``frame_number``, ``frame_ts_ms``, ``speed``,
    ``location``, ``pose``, ``stage``: for every dialect, so a caller never
    has to know how old the file is. ``timestamp_ms`` is kept as an alias for
    existing callers.
    """
    from tools.offline_analysis import video_data_schema as vds

    header: Dict[str, Any] = {}
    zones: Any = None
    body_parts: list = []
    rows: List[dict] = []
    try:
        hdr = vds.VideoDataHeader.parse(txt_path)
        header = dict(hdr.info)
        zones = hdr.zones
        body_parts = list(hdr.body_parts)
        if body_parts:
            header["body_parts"] = body_parts
        for r in vds.iter_rows(txt_path, hdr):
            ts = vds.parse_float(r.get("frame_ts_ms"), 0.0)
            rows.append({
                "frame": vds.parse_int(r.get("frame_number"), r["_index"]),
                "frame_ts_ms": ts,
                "timestamp_ms": ts,
                "speed": vds.parse_float(r.get("speed"), 0.0),
                "location": ("" if vds.is_na(r.get("location"))
                             else str(r.get("location"))),
                "stage": ("" if vds.is_na(r.get("stage"))
                          else str(r.get("stage"))),
                "pose": vds.parse_pose(r.get("pose_array")),
            })
    except Exception as e:
        logger.warning(f"_load_session_frames({txt_path}): {e}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["frame", "frame_ts_ms", "timestamp_ms", "speed", "location",
                 "stage", "pose"])

    # Body parts from the first non-empty pose when the header did not say.
    if not body_parts and not df.empty:
        for p in df["pose"]:
            if isinstance(p, dict) and p:
                body_parts = list(p.keys())
                break

    return header, zones, df, body_parts


#: Rows to sniff for a pose when the header does not name the body parts.
#: A recording that has poses at all has them within the first few frames; a
#: recording that does not would otherwise be read end to end to learn that.
_POSE_SNIFF_ROWS = 400


def body_parts_of(txt_path: str) -> List[str]:
    """The keypoint names in a recording, WITHOUT reading it.

    The tab asks this on every refresh, several times over, so it must not
    parse the file. Answering it through ``_load_session_frames``, which
    builds a DataFrame from every row, costs 1.4 s per recording, which at
    ten calls per refresh is fifteen seconds for one file. Almost always the
    header already names the body parts, and when it does not, the first
    handful of rows settle it.
    """
    from tools.offline_analysis import video_data_schema as vds

    try:
        hdr = vds.VideoDataHeader.parse(txt_path)
    except Exception as e:
        logger.debug("body_parts_of(%s): %s", txt_path, e)
        return []
    if hdr.body_parts:
        return list(hdr.body_parts)
    col = vds.pose_column(hdr) or "pose_array"
    try:
        for i, r in enumerate(vds.iter_rows(txt_path, hdr)):
            if i >= _POSE_SNIFF_ROWS:
                break
            pose = vds.parse_pose(r.get(col))
            if isinstance(pose, dict) and pose:
                return list(pose.keys())
    except Exception as e:
        logger.debug("body_parts_of(%s): %s", txt_path, e)
    return []


def _plot_filename(fig, i: int) -> str:
    """A saved plot's name, from the recording it shows.

    The figure already titles itself with the subject and stage; the files it
    was saved as were `plot_1.png` … `plot_12.png`, so a folder of them said
    nothing about which animal was which and re-saving a second run silently
    overwrote the first.
    """
    get = getattr(fig, "get_suptitle", None)              # matplotlib ≥ 3.8
    try:
        title = get() if get is not None else ""
    except Exception:                                     # pragma: no cover
        title = ""
    safe = re.sub(r"[^\w.-]+", "_", (title or "").strip()).strip("_")
    return safe or f"plot_{i + 1}"




# ── Analysis worker ─────────────────────────────────────────────

class AnalysisWorker(QThread):
    """Background thread that runs analysis on loaded session files.

    Iterates over each SessionFile and calls ``_analyze_one()``, which drives
    the shared ``source.analysis.offline_analysis`` engine (the SAME engine the
    CLI uses, so GUI output == CLI output). Per-file it parses, builds the
    track (confidence gate → teleport clamp → min-move de-jitter → distance/
    velocity), computes the selected metrics, merges time-bin columns, and
    builds a live matplotlib figure when plots are selected.

    Option mapping (tab dicts → engine params) lives in one place:
    :meth:`_engine_params`. Per-file calibration (px_per_cm) comes from the
    scale zone in each file's header, falling back to the user's global value.

    Signals
    -------
    finished : object
        Dict with keys: sessions (list of row dicts), summary_df, figures
        (matplotlib Figures), excluded (na files), px_per_cm, errors.
    progress : str
        Status messages for the UI progress indicator.
    """
    finished = Signal(object)
    progress = Signal(str)

    def __init__(self, files: List[SessionFile], metrics: Dict[str, bool],
                 params: dict, metadata_df: Optional[pd.DataFrame] = None,
                 metadata_id_col: Optional[str] = None):
        super().__init__()
        self.files = files
        self.metrics = metrics
        self.params = params
        self.metadata_df = metadata_df
        self.metadata_id_col = metadata_id_col
        self._figures: list = []      # live matplotlib figures (built per file)
        self._ep: Optional[dict] = None  # cached engine params

    def _lookup_metadata(self, subject_id: str) -> Dict[str, Any]:
        """Return metadata for ``subject_id``, using the column the user
        picked on metadata load. No alias-guessing, the user chose.
        """
        if self.metadata_df is None or not subject_id:
            return {}
        id_col = getattr(self, "metadata_id_col", None)
        if not id_col or id_col not in self.metadata_df.columns:
            return {}
        try:
            sid = str(subject_id)
            match = self.metadata_df[
                self.metadata_df[id_col].astype(str) == sid]
            if match.empty:
                return {}
            row = match.iloc[0].to_dict()
            return {k: v for k, v in row.items()
                    if k != id_col and pd.notna(v)}
        except Exception as e:
            logger.warning(f"Metadata lookup failed for {subject_id}: {e}")
            return {}

    # ── engine bridge ────────────────────────────────────────────
    def _engine_params(self) -> dict:
        """Translate the tab's (metrics, params) option dicts into a single
        offline_analysis params dict. This is the ONE place the GUI options map
        onto the analysis engine, the same engine the CLI uses, so GUI == CLI.
        """
        if getattr(self, "_ep", None):
            return self._ep
        from tools.offline_analysis.engine.offline_analysis import DEFAULTS
        m, p = self.metrics, self.params
        ep = dict(DEFAULTS)
        ep["metrics"] = {
            "locomotion":   bool(m.get("distance") or m.get("speed")),
            "immobility":   bool(m.get("immobility")),
            "freezing":     bool(m.get("freezing")),
            # A transition is a zone measure, so it follows the Zones
            # toggle like head occupancy and interaction do. Otherwise
            # unticking every measure still produced Transitions_total.
            "transitions":  bool(m.get("zone_transitions")
                                 and m.get("zone_time")),
            "zone_time":    bool(m.get("zone_time")),
            "zone_entries": bool(m.get("zone_entries")),
            "zone_latency": bool(m.get("latency")),
            "zone_distance": bool(m.get("zone_distance", m.get("zone_time"))),
            "zone_advanced": bool(p.get("zone_advanced", False)),
            # Head occupancy and interaction scoring ARE zone measures, so they
            # follow the Zones toggle rather than defaulting on with no
            # control of their own.
            "head_zones":   bool(m.get("zone_time")),
            "interaction":  bool(m.get("zone_time")),
            "ymaze":        False,          # objective only, no derived Y-maze indices
        }
        ep["figures"] = {
            "track":             bool(m.get("trajectory")),
            "heatmap":           bool(m.get("heatmap")),
            "zone_timeline":     bool(m.get("zone_bar")),
            "distance_velocity": bool(m.get("speed_profile")),
            "group":             False,
        }
        ep["px_per_cm_override"] = float(p.get("pixels_per_cm", 0) or 0)
        ep["body_part"] = p.get("body_part_primary") or ""
        ep["body_parts_multi"] = list(p.get("body_parts_multi") or [])
        ep["min_dwell_s"] = float(p.get("min_dwell_time", 0.2) or 0.0)
        # Immobility and Freezing are now distinct: immobility uses the higher
        # speed cut, freezing the stricter one; both share the min-bout duration.
        ep["immobility_cm_s"] = float(p.get("freeze_threshold", 2.0))
        ep["freeze_cm_s"] = float(p.get("freeze_strict_cm_s", 0.5))
        ep["min_freeze_s"] = float(p.get("freeze_min_ms", 1000)) / 1000.0
        # All three arrive already resolved from `_bin_params`; the switch has
        # zeroed them when it is off. A caller that still speaks the old pair
        # (`time_bin_enabled` + `time_bin_seconds`), a profile saved before
        # the modes existed, or any other caller of this function, is
        # answered rather than silently given no bins at all.
        ep["bin_size_s"] = float(p.get("bin_size_s", 0) or 0)
        ep["bin_count"] = int(p.get("bin_count", 0) or 0)
        ep["bin_edges_s"] = list(p.get("bin_edges_s") or [])
        if not (ep["bin_size_s"] or ep["bin_count"] or ep["bin_edges_s"]):
            if p.get("time_bin_enabled"):
                ep["bin_size_s"] = float(p.get("time_bin_seconds", 60) or 60)
        ep["pcut"] = float(p.get("pcut", 0.55))
        # DEFAULTS, not a literal True: this disagreed with the analysis's
        # own default, so the answer depended on which door you came in by.
        ep["swap_correct"] = bool(p.get("swap_correct",
                                        DEFAULTS["swap_correct"]))
        ep["swap_cost_threshold"] = float(p.get("swap_cost_threshold", 0.3))
        # Which two keypoints can be confused. Left empty the resolver takes
        # the first two in the file's own order, right by luck for
        # Head/Center/Tailbase, silently wrong for any other ordering.
        ep["head_part"] = p.get("head_part", "") or ""
        ep["swap_body_part"] = p.get("swap_body_part", "") or ""
        ep["rolling_median"] = bool(p.get("rolling_median", False))
        # DEFAULTS, not a literal False: the widget defaults ON and the
        # engine defaults ON, so a profile that omitted the key turned
        # smoothing off behind both of them.
        ep["smooth"] = bool(p.get("smooth", DEFAULTS["smooth"]))
        ep["euro_min_cutoff"] = float(p.get("euro_min_cutoff", 1.0))
        ep["euro_beta"] = float(p.get("euro_beta", 0.007))
        ep["min_move_cm"] = float(p.get("min_move_cm", 0.0))
        ep["max_speed_cm_s"] = float(p.get("max_speed_cm_s", 380.0))
        ep["columns"] = p.get("columns") or {"select": [], "rename": {}}
        ep["zones_include"] = list(p.get("zones_include") or [])   # [] = all zones
        # Found by sweeping DEFAULTS against this function: each of these is
        # READ by the analysis and was never written here, so it could only
        # ever hold its default however the user set the UI.
        for key in ("novel_arm", "familiar_arm", "start_arm"):
            ep[key] = str(p.get(key) or "")
        ep["split_stages"] = bool(p.get("split_stages", True))
        ep["trim_start_s"] = float(p.get("trim_start_s", 0.0) or 0.0)
        ep["trim_end_s"] = float(p.get("trim_end_s", 0.0) or 0.0)
        ep["heatmap_bins"] = int(p.get("heatmap_bins", 40) or 40)
        # Sub-zones reported as one ({} = none). `_engine_params` is the
        # ONE place GUI options become analysis params, so a key that
        # stops here never reaches the run.
        ep["regions"] = {str(k): list(v)
                         for k, v in (p.get("regions") or {}).items() if v}
        ep["plots"] = False          # figures are built as live objects, not PNGs
        # How a state machine's states become rows, and where the run's
        # take-away copies go. Both were settings that existed and stopped
        # here, `output_dir` sat in the options dict, read by nothing.
        ep["state_rows"] = str(p.get("state_rows", "auto") or "auto")
        ep["output_dir"] = str(p.get("output_dir", "") or "")
        # `retrack_write_video` was declared in the options dict and read by
        # nothing, a checkbox that did exactly as much as no checkbox. It is
        # the same switch, now connected.
        ep["write_video"] = bool(p.get("write_video", False))
        self._ep = ep
        return ep

    def run(self):
        results = {"sessions": [], "figures": [], "errors": [],
                   "excluded": [], "px_per_cm": {}}
        self._figures = []
        for i, sf in enumerate(self.files):
            self.progress.emit(f"Analyzing {sf.name} ({i+1}/{len(self.files)})...")
            try:
                row, zone_rows, _ = self._analyze_one(sf)
                results["sessions"].append(row)
                results["px_per_cm"][sf.name] = row.get("__px_per_cm", 0)
                if row.get("__excluded"):
                    results["excluded"].append(
                        {"File": sf.name, "reason": row.get("Note", "no pose"),
                         "Stage": row.get("Stage", "")})
            except Exception as e:
                results["errors"].append(f"{sf.name}: {e}")
                logger.error(f"Analysis error for {sf.name}: {e}", exc_info=True)

        results["figures"] = self._figures
        results["summary_df"] = (pd.DataFrame(results["sessions"])
                                 if results["sessions"] else pd.DataFrame())
        self.progress.emit("Done.")
        self.finished.emit(results)

    def _analyze_one(self, sf: "SessionFile"):
        """Analyze one session via the offline_analysis engine.

        Returns ``(row, zone_rows, None)``. Metadata columns are interleaved
        right after Subject/Date/Stage; time-bin columns (when enabled) are
        appended to the same row. A live matplotlib figure is built and
        collected in ``self._figures`` when any plot is selected.

        Whole-session, single-row: the Verify preview and the columns dialog
        need one representative row per file. The real Run goes through
        :class:`PipelineWorker` → :func:`source.analysis.pipeline.run_all`,
        which emits one row per protocol stage.
        """
        from tools.offline_analysis.engine import offline_analysis as oa
        ep = self._engine_params()
        sess = oa.parse_txt(sf.txt_path)
        tr = oa.build_track(sess, ep)
        erow = oa.session_metrics(sess, tr, ep)

        subject = sf.subject or erow.get("Subject", "") or sess.stem
        row: Dict[str, Any] = {
            "Subject": subject,
            "Date": erow.get("Date", "") or getattr(sf, "date", ""),
            "Start_time": erow.get("Start_time", ""),
            "Stage": erow.get("Stage", "") or getattr(sf, "header", {}).get("stage", ""),
        }
        for k, v in self._lookup_metadata(subject).items():   # sex, strain, …
            row[str(k)] = v
        for k, v in erow.items():                             # engine metrics
            if k not in row:
                row[k] = v
        row["__file"] = sf.name
        row["__px_per_cm"] = erow.get("px_per_cm", 0)
        if not tr.has_pose:
            row["__excluded"] = True

        if oa.wants_bins(ep) and tr.has_pose:                 # bins → wide columns
            for k, v in oa.time_bins(sess, tr, ep).items():
                if k not in ("Subject", "Stage", "File"):
                    row[k] = v

        if oa._selected_figures(ep) and tr.has_pose:          # live figure
            try:
                fig = oa.build_figure(sess, tr, ep)
                if fig is not None:
                    self._figures.append(fig)
            except Exception as e:
                logger.warning(f"figure build failed for {sf.name}: {e}")

        return row, [], None



# ── Option Dialogs (proper QDialog with title bar) ───────────────



class _OptionDialog(QDialog):
    """Base class for option dialogs, proper window with OK button."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(_THEME.STYLE)
        self.setMinimumWidth(320)
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(12, 12, 12, 12)
        self._root.setSpacing(8)

    def _add_ok_button(self):
        from PySide6.QtWidgets import QDialogButtonBox
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bb.accepted.connect(self.accept)
        bb.setStyleSheet(f"QPushButton {{ background: {_THEME.OK_BG};"
                         "QPushButton:hover { background: #2ea043; }")
        self._root.addWidget(bb)


class ColumnsDialog(_OptionDialog):
    """Curate output columns, toggle inclusion, rename, and reorder.

    Feeds ``opts['columns'] = {'select': [orig,...ordered], 'rename': {orig:new}}``
    which the engine applies to the exported workbook and the tab applies to the
    results table. Identity columns (Subject/Date/Stage/…) are always kept and so
    are shown pinned and non-removable. Reordering is by Move Up / Move Down.
    """

    def __init__(self, opts: dict, columns: List[str], parent=None):
        super().__init__("Columns", parent)
        self._opts = opts
        self.setMinimumSize(420, 480)
        from tools.offline_analysis.engine.offline_analysis import ID_COLUMNS
        self._id_cols = set(ID_COLUMNS)

        cfg = opts.get("columns") or {"select": [], "rename": {}}
        select = [c for c in (cfg.get("select") or []) if c in columns]
        rename = cfg.get("rename") or {}
        # ordering: previously-selected first (in saved order), then the rest
        ordered = select + [c for c in columns if c not in select]
        # if nothing was ever selected, everything starts checked
        checked_default = not select

        info = QLabel("Drag blocks to reorder · click to select (Ctrl/Shift for many) · "
                      "double-click to rename · tick to include. Identity columns stay pinned.")
        info.setWordWrap(True)
        info.setObjectName("field")
        self._root.addWidget(info)

        self._list = QtWidgets.QListWidget()
        # draggable "lego blocks": internal move + multi-select + spaced pill rows
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.setSpacing(2)
        # Draggable pill rows, geometry only; colours come from the theme.
        self._list.setStyleSheet(
            f"QListWidget{{background:{_THEME.WELL};color:{_THEME.INK};"
            f"border:1px solid {_THEME.LINE};border-radius:4px;outline:none;}}"
            f"QListWidget::item{{background:{_THEME.RAISED};"
            f"border:1px solid {_THEME.LINE3};border-radius:4px;"
            f"margin:2px 3px;padding:7px 9px;}}"
            f"QListWidget::item:selected{{background:{_THEME.SEL};"
            f"border-color:{_THEME.ACCENT};color:#d8d8ff;}}")
        for col in ordered:
            it = QtWidgets.QListWidgetItem(rename.get(col, col))
            it.setData(Qt.ItemDataRole.UserRole, col)          # original name
            flags = it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
            it.setFlags(flags)
            is_id = col in self._id_cols
            it.setCheckState(Qt.CheckState.Checked
                             if (is_id or checked_default or col in select)
                             else Qt.CheckState.Unchecked)
            if is_id:
                it.setForeground(Qt.GlobalColor.gray)
                it.setToolTip("identity column, always included")
            self._list.addItem(it)
        self._root.addWidget(self._list, 1)

        btns = QHBoxLayout()
        for label, slot in [("▲", lambda: self._move(-1)),
                            ("▼", lambda: self._move(1)),
                            ("Check", lambda: self._set_all(True)),
                            ("Uncheck", lambda: self._set_all(False))]:
            b = QPushButton(label); b.clicked.connect(slot); btns.addWidget(b)
        btns.addStretch()
        self._root.addLayout(btns)
        self._add_ok_button()

    def _move(self, delta: int):
        row = self._list.currentRow()
        if row < 0:
            return
        new = row + delta
        if not (0 <= new < self._list.count()):
            return
        it = self._list.takeItem(row)
        self._list.insertItem(new, it)
        self._list.setCurrentRow(new)

    def _set_all(self, on: bool):
        # apply to the selected blocks, or to everything if nothing is selected
        sel = self._list.selectedItems()
        items = sel if sel else [self._list.item(i) for i in range(self._list.count())]
        for it in items:
            if it.data(Qt.ItemDataRole.UserRole) in self._id_cols:
                continue                          # identity always on
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def apply(self):
        select: List[str] = []
        rename: Dict[str, str] = {}
        for i in range(self._list.count()):
            it = self._list.item(i)
            orig = it.data(Qt.ItemDataRole.UserRole)
            new = it.text().strip()
            if it.checkState() == Qt.CheckState.Checked or orig in self._id_cols:
                select.append(orig)
            if new and new != orig:
                rename[orig] = new
        self._opts["columns"] = {"select": select, "rename": rename}


# ── Retrack / Correction Options Dialog ──────────────────────────

# ── Run worker ───────────────────────────────────────────────────
#
# One worker for every kind of work. There is no "mode": the bundle's plan
# says which stages a recording needs, read, track, re-zone, measure, and
# this runs exactly that. What the Plan column promised and what happens are
# the same list, computed once.

class ValidateWorker(QThread):
    """Runs the checks in :mod:`source.analysis.validate` off the GUI thread.

    They parse every selected recording and measure it three more ways, which
    is seconds per file, long enough that doing it inline would freeze the
    window and read as a hang.

    It also runs the PARITY comparison when a recording has been tracked more
    than one way, which is what lets a rig operator answer "did switching to
    fp16 change my data?" without a terminal. The runs are already stored side
    by side in the bundle, so nothing is re-tracked, the comparison is over
    numbers that are on disk already.
    """

    finished = Signal(object)
    progress = Signal(str)

    def __init__(self, paths: List[str], params: dict):
        super().__init__()
        self.paths = list(paths)
        self.params = dict(params or {})

    def run(self):
        from tools.offline_analysis.engine import validate as _vd

        checks = list(_vd.definition_checks())
        for i, path in enumerate(self.paths):
            self.progress.emit(f"[{i + 1}/{len(self.paths)}] "
                               f"{os.path.basename(path)}")
            try:
                checks.extend(_vd.identity_checks(path, self.params))
            except Exception as e:                        # pragma: no cover
                checks.append(_vd.Check(f"{os.path.basename(path)}: readable",
                                        True, False, False, str(e)))
            try:
                checks.extend(_vd.parity_in_bundle(path))
            except Exception as e:                        # pragma: no cover
                logger.warning("parity check failed for %s: %s", path, e)
        self.finished.emit(checks)


class PipelineWorker(QThread):
    """Runs the plan for the selected recordings, off the GUI thread.

    Cancellable at frame granularity, so a mis-configured six-hour tracking
    run can be stopped without killing the application.
    """

    finished = Signal(object)
    progress = Signal(str)

    def __init__(self, bundles: list, metrics: Dict[str, bool], params: dict,
                 metadata_df: Optional[pd.DataFrame] = None,
                 metadata_id_col: Optional[str] = None,
                 dry_run: bool = False):
        super().__init__()
        self.bundles = bundles
        self.metrics = metrics
        self.params = params
        self.metadata_df = metadata_df
        self.metadata_id_col = metadata_id_col
        self.dry_run = dry_run
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    #: Set when figures were asked for and something stopped them, so the
    #: reason reaches the panel instead of an empty Plots tab.
    _figure_note = ""

    #: How many recordings' plots the Plots tab will hold. Beyond this the
    #: run says so rather than quietly showing a short list.
    FIGURE_CAP = 12

    def _figures_for(self, out, ep):
        """The plots the user ticked, and, when there are none, why.

        Run returned `[]` unconditionally: ticking Track plot, Heatmap, Zone
        timeline or Distance & velocity produced nothing at all, and the Plots
        tab stayed empty with no reason given. The figure builder was there
        the whole time; the OTHER worker, the one the panel no longer runs,
        was the only caller.

        The pipeline builds them now, because only it holds the session the
        numbers came from, with the edited zones and scale applied. Building
        them here instead plotted the file's original zones beside a workbook
        measured with the user's.
        """
        from tools.offline_analysis.engine import offline_analysis as oa

        if not oa._selected_figures(ep):
            return []                       # nothing ticked is not a failure
        figs = out.get("figures") or []
        if not figs:
            why = oa._MPL_ERROR or "no measured recording had poses to plot"
            self._figure_note = f"no plots were produced, {why}"
        elif out.get("figures_capped"):
            n = len(figs) + int(out["figures_capped"])
            # Not "Save Plots writes the rest", it writes what is on screen,
            # and the others were never drawn. Say what would actually get
            # them: a smaller selection.
            self._figure_note = (
                f"plots shown for {len(figs)} of {n} recordings, the numbers "
                f"cover all {n}; select fewer recordings to plot the others")
        return figs

    def run(self):
        from tools.offline_analysis.engine import offline_analysis as _oa
        from tools.offline_analysis.engine import pipeline as _pl

        bridge = AnalysisWorker([], self.metrics, self.params,
                                metadata_df=self.metadata_df,
                                metadata_id_col=self.metadata_id_col)
        ep = bridge._engine_params()
        ep["plots"] = False                 # figures are live objects, not PNGs
        # Asked for only when a panel is ticked: `build_figure` imports
        # matplotlib before it looks at the selection, and a run with no plots
        # should not pay for the import.
        ep["live_figures"] = self.FIGURE_CAP if _oa._selected_figures(ep) else 0

        out = _pl.run_all(self.bundles, ep, progress=self.progress.emit,
                          cancel=self._cancel, dry_run=self.dry_run)

        rows = out["summary"]
        if self.metadata_df is not None:
            for row in rows:
                extra = bridge._lookup_metadata(str(row.get("Subject", "")))
                for k, v in extra.items():
                    row.setdefault(str(k), v)

        results = {
            "sessions": rows,
            "transitions": out.get("transitions") or [],
            "summary_df": (pd.DataFrame(rows)[out["columns"]]
                           if rows else pd.DataFrame()),
            "figures": self._figures_for(out, ep),
            "errors": out["errors"],
            "warnings": out["warnings"],
            "excluded": out["excluded"],
            "px_per_cm": {r.get("Subject", ""): r.get("px_per_cm", 0)
                          for r in rows},
            "cancelled": self._cancel.is_set(),
        }
        if self._figure_note:
            results["warnings"] = list(results["warnings"]) + [self._figure_note]
        self.progress.emit("Cancelled." if self._cancel.is_set() else "Done.")
        self.finished.emit(results)


class _LogBridge:
    """Stands where the tab's own Log pane used to, and forwards to logging.

    The window carries a slide-out Log sidebar, and this tab arrived from
    pyBehaveTrack with a Log tab of its own, so the analyser had two logs,
    each holding half the story, and neither was the one you would look in
    first. Everything now goes through the `tools.offline_analysis` logger,
    which the sidebar already listens to.

    It keeps the ``append``/``clear`` shape of the QTextEdit it replaces, so
    the thirty-odd call sites read exactly as they did. ``clear`` does
    nothing: the sidebar's history belongs to the session, not to one view's
    idea of starting again.
    """

    def append(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            if line.strip():
                logger.info(line.rstrip())

    def clear(self) -> None:
        return


def _newest_pose_file(bundle) -> str:
    """The most recent pose file a retrack left in this bundle, or "".

    `SessionBundle.active_txt` trusts what session.json recorded; this is the
    fallback for when that record is gone but the work is not.
    """
    import glob

    try:
        found = glob.glob(os.path.join(bundle.dir, "poses", "*_video_data.txt"))
    except Exception:                                     # pragma: no cover
        return ""
    if not found:
        return ""
    return max(found, key=lambda p: os.path.getmtime(p))


class AnalyzeTab(QWidget):
    """Main analysis tab, load data, configure, retrack/correct, analyze, export.

    UI layout (top to bottom):
    - Row 1: Load buttons (Files, Folder, Project) + file count + Clear
    - Row 2: Processing mode + option dialogs (Measures, Zones, Retrack,
             Time Bins, Body Parts, Plots)
    - Row 3: Run button + status + export buttons (Excel, CSV, Save Plots)
    - Results area: QTabWidget with Results table, Plots, Log

    Processing modes:
    - **Analysis Only**: Compute metrics from existing tracking data.
    - **Correct + Analyze**: Apply pose corrections (swap resolver, speed
      filter, rolling median, one-euro filter) then analyze.
    - **DLC Retrack**: Re-run DLC inference on video, apply corrections,
      write corrected ``_retracked.txt``, then analyze.

    Each option button opens a proper QDialog (with title bar and OK button).
    The Zones dialog includes an "Edit Zones on Video..." button that opens a
    file-by-file zone editor with prev/next navigation across all loaded files.
    """

    #: Asked for when what this tab has to say IS the log, the numbers check
    #: writes its findings there, and used to switch to a Log tab of its own.
    #: The window answers by sliding the Log sidebar out.
    logRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sessions: List[SessionFile] = []
        self._worker: Optional[AnalysisWorker] = None
        #: The "Check the numbers" thread, when one is running.
        self._validator: Optional[ValidateWorker] = None
        self._current_figures: list = []
        self._current_summary_df: Optional[pd.DataFrame] = None
        #: Ordered zone pairs from the last run, for the export's own sheet.
        self._current_transitions: list = []
        self._excluded: list = []
        self._metadata_df: Optional[pd.DataFrame] = None
        # Column the user picked as Subject-ID on metadata load.
        self._metadata_id_col: Optional[str] = None
        self._metadata_path: str = ""
        self.setStyleSheet(_THEME.STYLE)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4); root.setSpacing(5)
        self.setAcceptDrops(True)          # drag files/folders onto the tab
        from PySide6.QtCore import QSettings
        self._settings = QSettings("pyBehaveTrack", "AnalyzeTab")
        self._last_dir = self._settings.value("last_dir", "", str) or ""  # remembered across sessions
        #: Derived state per recording, clock, space, tracker, stage hashes.
        #: The Plan column and the run both read it, so they cannot disagree.
        self._bundles: Dict[str, Any] = {}
        self._worker_cancel = None
        self._intent = INTENT_ANALYSE
        #: The tracker chosen in the dialog, applied when the intent is
        #: Re-track. Held here so switching away and back does not lose it.
        self._pending_track = None
        #: {region: [zone, ...]}, sub-zones reported as one. Empty = none.
        self._regions: Dict[str, list] = {}
        #: measure key -> [(form, widget)] for every threshold it owns.
        self._threshold_rows: Dict[str, list] = {}
        self._metric_checks: Dict[str, QCheckBox] = {}
        self._plot_checks: Dict[str, QCheckBox] = {}
        self._opts = {
            "columns": {"select": [], "rename": {}},
            "dlc_model_path": "", "dlc_gpu_id": "0",
            "dlc_resize": 1.0, "dlc_skip_frames": 1, "pcut": 0.55,
            "correct_swap": True, "swap_threshold": 0.3,
            "correct_speed": True, "max_speed_px_s": 3000,
            "correct_median": True,
            "correct_euro": True, "euro_min_cutoff": 1.0, "euro_beta": 0.007,
            "retrack_write_video": False, "retrack_write_txt": True,
            "retrack_overwrite": True, "output_dir": "",
        }

        # ── toolbar: counts · Metadata · Profile ──
        #
        # No loader here. pyBehaveTrack loads recordings from inside its tab,
        # and this window carries a picker above it, so Folder, Files and
        # Project each existed twice, one above the other, and so did Clear.
        # Loading recordings belongs to the picker, which is the thing that
        # spans the window; what is left here are the actions that attach
        # something to the ANALYSIS rather than choose what is analysed.
        top = QHBoxLayout(); top.setSpacing(4)

        self._file_count = QLabel("No files loaded")
        self._file_count.setObjectName("status")
        top.addWidget(self._file_count)
        self._meta_label = QLabel("")
        self._meta_label.setObjectName("note")
        top.addWidget(self._meta_label)
        top.addStretch()
        for text, slot in [("Metadata…", self._load_metadata),
                           ("Save profile…", self._save_profile),
                           ("Load profile…", self._load_profile)]:
            b = QPushButton(text)
            b.setObjectName("quiet")
            b.clicked.connect(slot); top.addWidget(b)
        root.addLayout(top)


        # ── main split ──
        #
        # Left owns the recordings AND everything that acts on them; right owns
        # "what to measure → run → results". Setup buttons belong beside the
        # rows they change, not in a strip at the top of the window.
        split = QSplitter(Qt.Orientation.Horizontal)
        # The left half shows EITHER the recordings or the zone editor. Not a
        # modal: a modal covers the readiness table and the ZONES & SCALE
        # readout, which are the two things the editor changes.
        self._left_stack = QtWidgets.QStackedWidget()
        self._left_stack.addWidget(self._build_files_panel())
        left, right = self._left_stack, self._build_right_panel()
        # Floors, so dragging the splitter cannot crush either half into
        # something unreadable: the table needs its Plan column, the measure
        # row needs its five toggles.
        # The recordings table is six narrow fact columns plus the Plan
        # column; it does not need half the window.
        left.setMinimumWidth(400)
        right.setMinimumWidth(430)
        split.addWidget(left)
        split.addWidget(right)
        # The left panel is a short list of facts, six narrow columns and one
        # row per recording. The right one carries every control and the
        # results, and was the half that had to scroll. Weight it accordingly;
        # the splitter is still there for anyone with long file names.
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 7)
        split.setSizes([540, 960])
        split.setChildrenCollapsible(False)
        self._main_split = split

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(0)
        bar.addWidget(split, 1)

        # Same idiom as the shell's Experiment Info / Error Log sidebars: a
        # narrow rotated tab on the edge. NOT the shell's CollapsibleSidebar
        # widget itself, that one is an overlay which collapses as soon as
        # you click outside it, and this is a panel you tick things in and
        # then press Run.
        from tools.offline_analysis.analyze.rotated_button import RotatedButton

        self._options_tab = RotatedButton(
            "OPTIONS", down=_THEME.ACCENT, hover=_THEME.LINE3,
            base=_THEME.PANEL)
        self._options_tab.setFixedWidth(20)
        self._options_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        # The ONE control for this panel, and it is always here. There used
        # to be a "Hide options" button as well, alone on a row of its own
        # above the panel: a whole row of height spent on a second way to do
        # what this tab already does, and the tab only ever worked one way.
        self._options_tab.setToolTip("Show or hide the options panel")
        self._options_tab.clicked.connect(self._toggle_options)
        bar.addWidget(self._options_tab)
        bar.addWidget(self._build_sidebar())
        root.addLayout(bar, 1)

        # Every workspace-specific control exists by now; show the right half.
        self._apply_workspace_ui()
        self._set_intent(self._intent, quiet=True)
        self._refresh_setup_group()
        self._sync_thresholds()
        self._sync_trim()
        self._sync_corrections()
        self._refresh_columns_preview()
        self._update_file_count()

    # ── the recordings on screen ─────────────────────────────────

    def _visible_sessions(self) -> List["SessionFile"]:
        """The recordings this tab shows. Everything the table, the plan and
        the run operate on is drawn from here."""
        return list(self._sessions)

    def _apply_workspace_ui(self):
        """Name the columns and the controls for the data this reads.

        pyBehaveTrack's tab carries a second workspace for ``*_pose3d.txt``
        multi-view reconstructions, solid zone volumes, height and rearing,
        a triangulator's own corrections. This rig records one video per
        session and reconstructs nothing, so that half is not hidden here, it
        is gone: a control with no data behind it is worse than no control.
        """
        self._file_table.setHorizontalHeaderLabels(
            ["Use", "Recording", "Video", "Clock", "Pose", "Space", "Plan"])
        self._btn_bodyparts.setToolTip(
            "Distance and zone occupancy follow the Centre keypoint.\n"
            "Tick Head / Tailbase to ADD per-part columns beside it.")
        self._zone_label.setText("report")
        self._refresh_zone_state()
        self._update_zone_summary()          # owns the button's own text

    # ── bundles ──────────────────────────────────────────────────
    def _bundle_for(self, sf: "SessionFile"):
        """The derived state for one recording, created on first use."""
        from tools.offline_analysis.engine.session_bundle import Bundle

        b = self._bundles.get(sf.txt_path)
        if b is None:
            # A video-only row keys on the video path, so open it as one,
            # handing an .mp4 to the session-file reader would find nothing
            # and report "no poses" for a recording that simply has none yet.
            path = sf.txt_path
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".m4v")):
                b = Bundle.open(video_path=path)
            else:
                b = Bundle.open(path)
            self._bundles[path] = b
        return b

    def _selected_bundles(self) -> list:
        return [self._bundle_for(sf) for sf in self._selected_sessions()]

    def _check_the_numbers(self):
        """Check the statistics, and these recordings, and say so in the Log.

        Two questions in one: do the measures still agree with arithmetic, and
        do the numbers this selection produces hold together, the bins adding
        up to the totals, no zone occupied for longer than the recording, the
        live rig and this workbook measuring the same path.
        """
        if getattr(self, "_validator", None) is not None:
            return                                    # one at a time
        paths = [b.active_txt() for b in self._selected_bundles()]
        paths = [p for p in paths if p and os.path.exists(p)]
        self.logRequested.emit()
        self._log.append("Checking the numbers"
                         + (f" on {len(paths)} recording(s)…" if paths else
                            ", no recordings selected, so the definitions only…"))
        ep = AnalysisWorker([], self._collect_metrics(),
                            self._collect_params())._engine_params()
        ep["plots"] = False
        self._validator = ValidateWorker(paths, ep)
        self._validator.progress.connect(self._on_progress)
        self._validator.finished.connect(self._on_checks_done)
        self._validator.start()

    def _on_checks_done(self, checks):
        # Same teardown as the run: this slot fires from inside the thread's
        # own `run()`, so dropping the reference here would destroy a live
        # QThread and abort the process without a traceback.
        worker = self._validator
        self._validator = None
        if worker is not None:
            if not worker.wait(5000):                     # pragma: no cover
                logger.warning("validation worker did not stop within 5 s")
            worker.deleteLater()

        failed = [c for c in checks if not c.ok and not c.skipped]
        skipped = [c for c in checks if c.skipped]
        for c in failed:
            self._log.append(f"  FAIL  {c.name}, expected {c.expected}, "
                             f"got {c.got}"
                             + (f"  ({c.note})" if c.note else ""))
        # The interesting passes are the ones carrying a measurement: a bare
        # "PASS" list is 300 lines nobody reads, but "live 20.57 m vs workbook
        # 18.43 m" is the reason someone opened this.
        for c in checks:
            if c.ok and not c.skipped and c.note:
                self._log.append(f"  note  {c.name}, {c.note}")
        ran = len(checks) - len(skipped)
        msg = (f"{ran - len(failed)}/{ran} checks passed"
               + (f", {len(skipped)} skipped" if skipped else ""))
        self._log.append(msg + ("" if not failed else
                                ", see the FAIL lines above"))
        self._status.setText(msg)

    def _clear_tracker(self):
        """Drop the tracking method and measure what is already there."""
        self._pending_track = None
        self._update_tracker_label()
        self._set_intent(INTENT_ANALYSE)
        self._status.setText(
            "Tracking method cleared, the recordings' own poses will be "
            "used.")

    # ── files panel ──────────────────────────────────────────────
    def _build_files_panel(self) -> QWidget:
        """Left panel: a VISIBLE, selectable table of loaded recordings."""
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
        hdr = QLabel("Recordings, drag files or a folder here")
        hdr.setObjectName("section")
        v.addWidget(hdr)
        self._excluded_paths: set = set()
        # Readiness, not settings: every column is a FACT about the file, and
        # the last one says what Run will do about it.
        self._file_table = QTableWidget(0, 7)
        self._file_table.setHorizontalHeaderLabels(
            ["Use", "Recording", "Video", "Clock", "Pose", "Space", "Plan"])
        self._file_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._file_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._file_table.setAlternatingRowColors(True)
        self._file_table.verticalHeader().setVisible(False)
        self._file_table.itemChanged.connect(self._on_file_item_changed)
        try:
            self._file_table.horizontalHeader().setStretchLastSection(True)
        except Exception:
            pass
        # Sizing: Recording and Plan are the two columns that carry meaning;
        # the four fact columns in between size to their content.
        #
        # Interactive, NOT ResizeToContents. Under ResizeToContents Qt
        # re-measures every row of a column on each `setItem`, so filling
        # thirty rows costs 1.45 s of text metrics, most of the time a click
        # takes. The columns are still sized to their content, once, by
        # `resizeColumnsToContents` after the rows are in.
        try:
            from PySide6.QtWidgets import QHeaderView
            hh = self._file_table.horizontalHeader()
            hh.setStretchLastSection(False)
            hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
            self._file_table.setColumnWidth(1, 205)
        except Exception as e:
            logger.debug("file table header sizing: %s", e)
        v.addWidget(self._file_table, 1)

        # ── actions ON the selected recordings ──
        #
        # These live here, beside the rows they change, and are ranked: the two
        # that fix a blocked recording are primary, selection is quiet, and the
        # occasional ones are behind "More". A flat strip of equal-weight
        # buttons says nothing about which one the operator needs.
        # Row one ACTS on the recordings; row two manages the LIST. Six
        # buttons on one line left each too narrow to read at the width the
        # left panel actually gets.
        self._file_actions_row = QWidget()
        row = QHBoxLayout(self._file_actions_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        # 2D: zones drawn on a video frame, and a single-video detector.
        # "Draw zones…" and "Set up tracker…" are NOT repeated here: they
        # already live in ZONES & SCALE and under the Re-track intent, and one
        # action under two names in three places reads as three actions. The
        # right-hand controls say what state the selection is in, so they own
        # these go.
        row.addStretch()
        v.addWidget(self._file_actions_row)

        self._file_list_row = QWidget()
        row = QHBoxLayout(self._file_list_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for text, slot in [("Check all", lambda: self._set_all_included(True)),
                           ("Uncheck all", lambda: self._set_all_included(False)),
                           ("Remove", self._remove_selected_files)]:
            b = QPushButton(text)
            b.setObjectName("quiet")
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()

        more = QPushButton("More  ▾")
        more.setObjectName("quiet")
        menu = QtWidgets.QMenu(more)
        menu.addAction("Frame rate…", self._set_frame_rate)
        self._act_clear_tracker = menu.addAction("Clear tracker",
                                                 self._clear_tracker)
        menu.addSeparator()
        # "Are these numbers right?" is a question about the recordings in
        # front of you, and it should not require knowing that a command-line
        # entry point exists. Same checks as `python -m source.analysis.validate`.
        self._act_validate = menu.addAction("Check the numbers…",
                                            self._check_the_numbers)
        more.setMenu(menu)
        self._more_menu = menu
        row.addWidget(more)
        v.addWidget(self._file_list_row)
        return w

    # ── right panel: config band + advanced + run + results ──────
    def _make_chip(self, label: str) -> QPushButton:
        c = QPushButton(label); c.setCheckable(True)
        c.setObjectName("chip")
        return c

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(5)

        # The options scroll; Run and the results do not. Four stacked groups
        # plus a results table need about 1,100 px, and the window opens at
        # 900, without this the run bar is pushed off the bottom.
        opts = QWidget()
        v = QVBoxLayout(opts)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(5)

        # The main column answers "what am I doing, to which recordings".
        # MEASURE, CORRECTIONS and INVESTIGATION are settings you visit and
        # leave, so they live behind edge tabs, the same idiom as the
        # shell's Error Log and Documentation sidebars.
        v.addWidget(self._build_setup_group())
        v.addWidget(self._build_intent())

        # ── MEASURE ──
        #
        # One band, no step numbers. The old ① ZONES / ② MEASURE badges
        # promised a sequence that was not the real one, setting up a
        # recording comes first, and picking which zones to REPORT is a filter
        # on the output, not a step. So: what to follow, what to compute,
        # which zones to include.
        # Folds like the others, the four option blocks have to behave the
        # same way or the two that do look like an accident.
        band = Section("MEASURE")
        self._measure_box = band
        bv = QVBoxLayout(band.body())
        bv.setContentsMargins(11, 9, 11, 9)
        bv.setSpacing(8)

        # Row 1 is the two pickers; row 2 is the toggles, which get the
        # whole width. Cramming both onto one line clipped the toggle labels
        # down to ":omoti" and "tearing".
        r1 = QHBoxLayout(); r1.setSpacing(8)
        r1.addWidget(self._field_label("body points"))
        self._bp_include: set = set()
        self._btn_bodyparts = QPushButton("Center  ▾")
        self._btn_bodyparts.setObjectName("quiet")
        self._bp_menu = StickyMenu(self._btn_bodyparts)
        self._btn_bodyparts.setMenu(self._bp_menu)
        self._btn_bodyparts.setToolTip(
            "Distance and zone occupancy follow the Centre keypoint.\n"
            "Tick Head / Tailbase to ADD per-part columns beside it.")
        r1.addWidget(self._btn_bodyparts)
        r1.addStretch()
        bv.addLayout(r1)

        # No "Advanced" fold. It was worth its cost when these settings were
        # stacked in the options column with everything else and the panel had
        # run out of height; in a sidebar of its own there is room, and a
        # threshold you cannot see is a threshold you do not know you have.
        # The rows still come and go with the measure that consumes them, so
        # what is on screen is only ever what is in play.
        self._btn_adv = QPushButton("Advanced  ▾")
        self._btn_adv.setCheckable(True)
        self._btn_adv.setChecked(True)
        self._btn_adv.hide()

        # Labelled for what it does to the OUTPUT: each extra point adds a
        # column set. "Follow" would read as "pick one point to track".
        self._lbl_points = QLabel("")
        self._lbl_points.setObjectName("field")
        self._lbl_points.setWordWrap(True)
        # Wraps AND yields: a hint that insists on its natural width makes the
        # section wider than the sidebar it lives in.
        self._lbl_points.setMinimumWidth(80)
        self._lbl_points.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Preferred)
        bv.addWidget(self._lbl_points)

        # Wraps: eight toggles in one fixed row need ~750 px, and at 125%
        # display scaling the panel has less than that, so the last of them
        # were CLIPPED off the edge with nothing to say they existed.
        from tools.offline_analysis.analyze.flow_layout import FlowLayout

        r2 = FlowLayout(spacing=6)
        self._chips: Dict[str, QPushButton] = {}
        for key, label, on, tip in [
            ("locomotion", "Locomotion", True, "Distance, mean and max speed."),
            ("zones", "Zones", True, "Time, entries and distance per zone."),
            ("immobility", "Immobility", True, "Immobile time and bouts."),
            # Freezing is a lab CONVENTION defined by a speed cut and a
            # duration, and the thresholds travel with the data file.
            ("freezing", "Freezing", False,
             "Movement suppression below a stricter speed cut."),
            ("timebins", "Time bins", False, "Split every measure into bins."),
            ("video", "Annotated video", False,
             "Write a clip of each recording with the pose, the zones and "
             "the task state drawn on it.\nRoughly doubles the time a run "
             "takes, so it is off unless you ask."),
        ]:
            c = self._make_chip(label); c.setChecked(on); c.setToolTip(tip)
            c.toggled.connect(self._on_measure_toggled)
            self._chips[key] = c; r2.addWidget(c)
        bv.addLayout(r2)

        # Directly under the switch that turns it on, not two folds away.
        bv.addWidget(self._build_bins_row())
        self._lbl_bins = QLabel("")
        self._lbl_bins.setObjectName("field")
        self._lbl_bins.setWordWrap(True)
        self._lbl_bins.setMinimumWidth(80)
        self._lbl_bins.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                     QtWidgets.QSizePolicy.Policy.Preferred)
        bv.addWidget(self._lbl_bins)

        bv.addWidget(self._build_arm_roles())

        # Behind the edge tabs, not in this column, see `_build_sidebar`.
        self._sidebar_pages = [
            ("ZONES", self._build_zone_group()),
            ("MEASURE", band),
            ("CORRECT", self._build_corrections()),
            ("OBJECTS", self._build_investigation()),
        ]

        # Advanced drawer (hidden by default). It belongs to the MEASURE
        # page, not to this column: the button that opens it moved into the
        # sidebar, and a drawer that opens somewhere the user is not looking
        #, behind the sidebar overlay, at that, is the same as one that
        # does not open.
        self._advanced = self._build_advanced()
        self._advanced.setVisible(True)
        bv.addWidget(self._advanced)

        # ── what you will get ──
        #
        # The toggles above are abstract until you have run once and read the
        # spreadsheet. This names the exact columns the current selection
        # produces, generated by the REAL emitter on a small stand-in session
        # so it cannot drift from the actual output.
        self._cols_preview = QLabel("")
        self._cols_preview.setObjectName("field")
        self._cols_preview.setWordWrap(True)
        self._cols_preview.setMaximumHeight(34)      # a reassurance, not a manifest
        self._cols_preview.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)

        # ── PLAN + RUN, one bar ──
        #
        # The plan and the button that executes it are one idea; two stacked
        # bars made them look like two.
        # Built here for reading order, added to the OUTER column below so
        # it stays put while the options scroll.
        run_bar = QFrame()
        run_bar.setObjectName("runBar")
        run_bar.setStyleSheet(f"QFrame#runBar{{background:{_THEME.SUNKEN};"
                              f"border:1px solid {_THEME.LINE};border-radius:4px;}}")
        rb = QHBoxLayout(run_bar)
        rb.setContentsMargins(11, 7, 11, 7); rb.setSpacing(9)
        self._btn_run = QPushButton("▶  Run plan"); self._btn_run.setFixedHeight(31)
        self._btn_run.setObjectName("run")
        self._btn_run.setEnabled(False); self._btn_run.clicked.connect(self._run_analysis)
        rb.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("Stop")
        self._btn_cancel.setFixedHeight(31)
        self._btn_cancel.setObjectName("stop")
        self._btn_cancel.setToolTip("Stop after the current frame. Poses already "
                                    "produced are kept.")
        self._btn_cancel.setVisible(False)
        self._btn_cancel.clicked.connect(self._cancel_run)
        rb.addWidget(self._btn_cancel)

        # Where the run's take-away copies land. Beside Run because it is a
        # property of the run, not a preference buried in Advanced, and
        # because without it the results exist only inside a hidden
        # `.pbanalysis` folder under a hash for a name.
        self._btn_outdir = QPushButton("Save results to…")
        self._btn_outdir.setFixedHeight(31)
        self._btn_outdir.setObjectName("quiet")
        self._btn_outdir.clicked.connect(self._choose_output_dir)
        rb.addWidget(self._btn_outdir)
        self._sync_output_dir()

        stack = QVBoxLayout(); stack.setSpacing(1)
        self._plan_label = QLabel("Load recordings to begin.")
        self._plan_label.setObjectName("plan")
        stack.addWidget(self._plan_label)
        self._status = QLabel("")
        self._status.setObjectName("status")
        stack.addWidget(self._status)
        rb.addLayout(stack, 1)

        self._btn_dry = QPushButton("Dry run")
        self._btn_dry.setObjectName("quiet")
        self._btn_dry.setToolTip(
            "Run every step except the detector, into a scratch area.\n"
            "How you check a 40-session setup before committing hours to it.")
        self._btn_dry.clicked.connect(lambda: self._run_analysis(dry_run=True))
        rb.addWidget(self._btn_dry)
        v.addStretch()
        self._opts_scroll = scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(opts)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # As-needed, NOT always-off. Always-off does not make content fit,
        # it makes the overflow invisible and unreachable, which is how a
        # 707 px overflow at 125% scaling went unnoticed.
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Always on, so a group cut off at the fold reads as "there is more
        # below" rather than as a clipped layout.
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setMinimumHeight(220)

        lower = QWidget()
        lv = QVBoxLayout(lower)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(5)
        lv.addWidget(self._cols_preview)
        lv.addWidget(run_bar)

        # Results
        self._results_tabs = QTabWidget()
        self._summary_table = QTableWidget()
        self._summary_table.setAlternatingRowColors(True)
        self._summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._summary_table.setSortingEnabled(True)
        self._results_tabs.addTab(self._summary_table, "Results")
        self._plot_widget = QWidget()
        p_lay = QVBoxLayout(self._plot_widget); p_lay.setContentsMargins(2, 2, 2, 2)
        self._plot_scroll = QScrollArea(); self._plot_scroll.setWidgetResizable(True)
        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._plot_scroll.setWidget(self._plot_container)
        p_lay.addWidget(self._plot_scroll)
        self._results_tabs.addTab(self._plot_widget, "Plots")
        # Verify: the tracking, drawn on the video, frame by frame. Every
        # silent-wrong-number defect this pipeline had is visible here in
        # seconds, a dt of 1 ms, a zone that never matches, px instead of cm.
        from tools.offline_analysis.analyze.verify_view import VerifyView
        self._verify = VerifyView()
        self._results_tabs.addTab(self._verify, "Verify")
        # No Log tab: the window has one Log and it is the sidebar.
        self._log = _LogBridge()

        lv.addWidget(self._results_tabs, 1)

        # Export lives with the results, and appears only once there are some.
        # Three permanently-greyed buttons in the run bar were furniture that
        # did nothing until the very end; in the tab-bar corner they squeezed
        # the tabs themselves.
        self._export_row = QWidget()
        ex = QHBoxLayout(self._export_row)
        ex.setContentsMargins(0, 2, 0, 0); ex.setSpacing(5)
        self._export_hint = QLabel("")
        self._export_hint.setObjectName("field")
        ex.addWidget(self._export_hint)
        ex.addStretch()
        for text, slot in [("Export Excel", self._export_excel),
                           ("Export CSV", self._export_csv),
                           ("Save Poses", self._export_poses),
                           ("Save Plots", self._save_plots)]:
            b = QPushButton(text)
            b.setObjectName("quiet")
            b.setEnabled(False)
            b.clicked.connect(slot)
            ex.addWidget(b)
            setattr(self, f"_btn_{text.split()[-1].lower()}", b)
        self._export_row.setVisible(False)
        lv.addWidget(self._export_row)

        # A splitter, not a fixed ratio: how much of the panel is options and
        # how much is results is the user's call, and it changes with what
        # they are doing.
        #
        # Results get the larger share. Options used to, back when this column
        # held MEASURE, CORRECTIONS and INVESTIGATION as well; those live in
        # the sidebar now, so the column holds two small groups and the rest
        # of its 3/5 was empty, a band of nothing above a video frame that
        # had been squeezed to make room for it.
        self._panel_split = split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(scroll)
        split.addWidget(lower)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setChildrenCollapsible(False)
        # Without this the results table's own minimum is ~470 px, which the
        # splitter must honour, so it silently overrides setSizes and the
        # options pane opens cut off in the middle of MEASURE.
        # Verify holds a video frame, a scrubber and two readout lines and
        # asks for 329 px; it was being given 324, and five pixels short is
        # not a scrollbar, it is the readout drawn ON TOP of the frame.
        # Enough for the scrubber and the readout with a usable frame
        # above them, not enough to squeeze the options column that feeds it.
        self._results_tabs.setMinimumHeight(260)
        lower.setMinimumHeight(340)
        # The opening split is set on first show, see `showEvent`. Setting
        # it here does nothing: the splitter re-lays out when it is shown and
        # discards sizes given before it has a geometry.
        # Three sections instead of six: the options no longer need two
        # thirds of the height, and the results table was the half being
        # squeezed for it.
        split.setSizes([360, 580])
        outer.addWidget(split, 1)
        return w

    # ── small typographic helpers ────────────────────────────────
    def _field_label(self, text: str) -> QLabel:
        """A quiet name for the control beside it, "follow", "report zones".
        Cryptic one-word labels like "breakdown" were the complaint."""
        lb = QLabel(text)
        lb.setObjectName("field")
        return lb

    def _adv_form(self, title: str):
        box = QFrame()
        box.setStyleSheet("QFrame{background:transparent;border:none;}")
        vv = QVBoxLayout(box); vv.setContentsMargins(0, 0, 0, 0); vv.setSpacing(3)
        t = QLabel(title.upper()); t.setObjectName("section")
        vv.addWidget(t)
        f = QFormLayout(); f.setContentsMargins(0, 0, 0, 0); f.setSpacing(3); f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        vv.addLayout(f)
        return box, f, vv

    # ── corrections ──────────────────────────────────────────────

    # ── intent ───────────────────────────────────────────────────
    #
    # What do you want done to these recordings? Three answers, and the user
    # says which rather than inferring it from which setup button they found.
    #
    # This is NOT the old "mode" selector. That one was stored in a key nobody
    # read while Run hard-coded DeepLabCut. This one sets state on the bundles,
    # so the Plan column and the run both follow from it; you can watch the
    # plan change as you pick.

    # ── the options panel folds away ─────────────────────────────

    def _toggle_options(self):
        """Fold the options away, or bring them back."""
        right = self._main_split.widget(1)
        self._set_options_shown(right.isHidden())

    def _set_options_shown(self, shown: bool, *, animate: bool = True):
        """Slide the options panel away, giving the window to the recordings.

        Animated like the shell's sidebars, and NOT built on the shell's
        `CollapsibleSidebar`: that one collapses on any click that is not on
        an interactive child, and an options panel is mostly labels, so it
        would shut itself as you used it.

        The width you dragged to is remembered, so bringing it back does not
        reset it.
        """
        right = self._main_split.widget(1)
        if shown and not right.isHidden():
            return
        if not shown:
            self._saved_split = self._main_split.sizes()

        target = 0
        if shown and getattr(self, "_saved_split", None):
            target = int(self._saved_split[1])
        target = target or max(620, right.minimumWidth())

        if not animate or not self.isVisible():
            right.setMaximumWidth(16777215)
            right.setVisible(shown)
            self._name_options_tab(shown)
            if shown and getattr(self, "_saved_split", None):
                self._main_split.setSizes(self._saved_split)
            return

        from PySide6.QtCore import QEasingCurve, QPropertyAnimation

        start = right.width() if not right.isHidden() else 0
        if shown:
            right.setMaximumWidth(0)
            right.setVisible(True)
        anim = QPropertyAnimation(right, b"maximumWidth", self)
        anim.setDuration(160)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        anim.setStartValue(start if shown else right.width())
        anim.setEndValue(target if shown else 0)

        def done():
            if shown:
                # Released, or the splitter can never be dragged again.
                right.setMaximumWidth(16777215)
                if getattr(self, "_saved_split", None):
                    self._main_split.setSizes(self._saved_split)
            else:
                right.setVisible(False)
                right.setMaximumWidth(16777215)
            self._name_options_tab(shown)

        anim.finished.connect(done)
        self._options_anim = anim              # keep it alive
        anim.start()

    def _name_options_tab(self, shown: bool) -> None:
        """Say which way the tab goes, since it now goes both."""
        self._options_tab.setToolTip(
            "Hide the options panel" if shown else "Show the options panel")

    # ── what this selection still needs ──────────────────────────

    # ── the settings sidebar ─────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        """MEASURE, CORRECTIONS and INVESTIGATION behind edge tabs.

        Three sections you set once and then leave alone, stacked in the same
        column as everything you touch on every run, is what made the panel
        need more height than the window has. They are settings, not steps.

        The idiom is the shell's Error Log and Documentation sidebars, a
        narrow rotated tab on the edge, but not its CollapsibleSidebar
        widget, which collapses on any click that is not on an interactive
        child and would shut itself as you used it.
        """
        from tools.offline_analysis.analyze.rotated_button import RotatedButton

        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # A child of the TAB, not of this row: it rolls OVER the content
        # like the shell's sidebars instead of taking width from the
        # splitter. Squeezing the options column to make room for a settings
        # page is how the panel ran out of width in the first place.
        self._sidebar_stack = QtWidgets.QStackedWidget(self)
        self._sidebar_stack.setObjectName("sidebarPanel")
        # A QStackedWidget does not paint a stylesheet background unless it
        # is told to, and this one FLOATS: without an opaque ground the
        # options column shows straight through the controls.
        self._sidebar_stack.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self._sidebar_stack.setMinimumWidth(300)
        self._sidebar_stack.setMaximumWidth(560)
        for _name, page in self._sidebar_pages:
            wrap = QScrollArea()
            wrap.setWidgetResizable(True)
            wrap.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            wrap.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            inner = QWidget()
            iv = QVBoxLayout(inner)
            iv.setContentsMargins(6, 6, 6, 6)
            # Always open in the sidebar: the tab already decides whether the
            # page is on screen, so a page that arrives folded is an empty box
            # in answer to a click that asked to see it.
            page.set_expanded(True)
            iv.addWidget(page)
            iv.addStretch()
            wrap.setWidget(inner)
            self._sidebar_stack.addWidget(wrap)
        self._sidebar_stack.hide()

        tabs = QWidget()
        tv = QVBoxLayout(tabs)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(3)
        self._sidebar_tabs = []
        for i, (name, _page) in enumerate(self._sidebar_pages):
            # A closed tab is a quiet raised edge with readable ink and a
            # hairline; the open one takes the accent. A closed tab in PANEL
            # would be the colour of the panel behind it, and four of them
            # would read as one dark strip with words down it.
            b = RotatedButton(name, down=_THEME.ACCENT, hover=_THEME.LINE3,
                              base=_THEME.RAISED, ink=_THEME.INK3,
                              ink_active="#ffffff", edge=_THEME.LINE3)
            b.setCheckable(True)
            # The size lives in the stylesheet (QPushButton#sidebarTab):
            # setFixedHeight loses to the stylesheet's recomputed minimum.
            b.setObjectName("sidebarTab")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"Show {name.title()} settings")
            b.clicked.connect(lambda _=False, k=i: self._toggle_sidebar(k))
            self._sidebar_tabs.append(b)
            tv.addWidget(b)
        tv.addStretch()
        row.addWidget(tabs)

        self._sidebar_open = -1
        return host

    def _toggle_sidebar(self, index: int):
        """Open a page, or close it if it is the one already showing."""
        if index == self._sidebar_open:
            self._sidebar_stack.hide()
            self._sidebar_open = -1
        else:
            self._sidebar_stack.setCurrentIndex(index)
            self._place_sidebar()
            self._sidebar_stack.show()
            self._sidebar_stack.raise_()
            self._sidebar_open = index
        for i, b in enumerate(self._sidebar_tabs):
            b.setChecked(i == self._sidebar_open)
            b.update()
        # Watch for a click elsewhere only while there is something to close.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            if self._sidebar_open >= 0:
                app.installEventFilter(self)
            else:
                app.removeEventFilter(self)

    def eventFilter(self, obj, event):
        """Close the open sidebar when the user clicks away from it.

        A settings panel that floats over the work is in the way until it is
        dismissed, and going back to the tab to dismiss it is a trip across
        the window for something the click you just made already implied.

        Tested on GEOMETRY, not on what was clicked. The shell's own
        CollapsibleSidebar closes on any click that is not on an interactive
        child, which means it shuts itself while you are using it, on a
        label, on a group's background, on the gap between two rows. This
        closes only for a press that lands outside the panel and outside the
        tab strip, so every click inside it, on anything at all, is safe.

        Popups, the sticky menus, a combo's list, a file dialog, are their
        own windows, so a click in one never reaches this test.
        """
        if (event.type() == QEvent.Type.MouseButtonPress
                and self._sidebar_open >= 0
                and not self._sidebar_stack.isHidden()):
            w = obj if isinstance(obj, QtWidgets.QWidget) else None
            if w is not None and w.window() is self.window():
                gp = event.globalPosition().toPoint()
                inside = self._sidebar_stack.rect().contains(
                    self._sidebar_stack.mapFromGlobal(gp))
                strip = self._sidebar_tabs[0].parentWidget()
                on_tab = strip is not None and strip.rect().contains(
                    strip.mapFromGlobal(gp))
                if not inside and not on_tab:
                    self._toggle_sidebar(self._sidebar_open)
        return super().eventFilter(obj, event)

    def _place_sidebar(self):
        """Park the panel against the tab strip, spanning the tab's height."""
        if not getattr(self, "_sidebar_tabs", None):
            return
        strip = self._sidebar_tabs[0].parentWidget()
        if strip is None:
            return
        left = strip.mapTo(self, strip.rect().topLeft()).x()
        w = min(self._sidebar_stack.maximumWidth(), max(300, left - 8))
        top = self._file_table.mapTo(self, self._file_table.rect().topLeft()).y()
        # The control for getting this panel out of the way is the OPTIONS
        # tab on the edge, which the panel parks against rather than over,
        # a panel that buries the control for moving it is one you cannot
        # get out of the way.
        self._sidebar_stack.setGeometry(left - w, top, w,
                                        max(200, self.height() - top - 6))

    def showEvent(self, e):
        """Give the options column its content height, once.

        Stretch factors alone left it 297px tall around 214px of content: a
        band of nothing between WHAT TO DO and the column preview, sitting
        directly above a video frame that had been squeezed to make room for
        it. Most of those settings live in the sidebar now, so the column no
        longer needs the larger share it was written for. The splitter still
        honours both minimums and the handle is still the user's; this only
        decides where it starts.
        """
        super().showEvent(e)
        if not getattr(self, "_split_sized", False):
            self._split_sized = True
            self._panel_split.setSizes(
                [self._opts_scroll.sizeHint().height(), 10000])

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._sidebar_stack.isHidden():
            self._place_sidebar()

    def _build_setup_group(self):
        """The things a recording is MISSING, each with the fix beside it.

        A panel that only reports "NO SCALE" tells you something is wrong and
        leaves you to find the cure somewhere else. Every line here is a
        button that supplies exactly what the line says is absent, and the
        whole group disappears when there is nothing to fix.
        """
        box = Section("SET UP")
        box.setObjectName("needs")
        v = QVBoxLayout(box.body())
        v.setSpacing(4)
        # In the body, not the header: a long title is what stops a section
        # from ever being narrower than its own name.
        lead = QLabel("What these recordings still need.")
        lead.setObjectName("field")
        lead.setWordWrap(True)
        v.addWidget(lead)

        self._need_rows = {}
        for key, label, tip, slot in (
            ("clock", "⏱  No frame rate, set it",
             "Without timestamps or a declared rate there is no seconds axis, "
             "so every speed and duration is meaningless.", self._set_frame_rate),
            ("pose", "◎  No poses, choose tracking",
             "Nothing has been tracked in this video yet. Pick DeepLabCut, "
             "SLEAP or the blob tracker to produce poses.",
             self._open_retrack_dialog),
            ("zones", "▱  No zones, draw them",
             "Zone occupancy, entries and latency need zones. Locomotion "
             "works without them.", self._edit_zones_on_video),
            ("scale", "📏  No scale, distances are in PIXELS",
             "A pixel distance cannot be compared between rigs, or published. "
             "Set the scale and everything switches to centimetres.",
             self._set_scale),
        ):
            b = QPushButton(label)
            b.setObjectName("need")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            b.hide()
            v.addWidget(b)
            self._need_rows[key] = b

        self._setup_box = box
        return box

    def _refresh_setup_group(self):
        """Show a line only while the thing it offers is actually missing."""
        if not hasattr(self, "_need_rows"):
            return
        bundles = self._selected_bundles()
        need = {k: False for k in self._need_rows}
        for b in bundles:
            if not b.clock.usable:
                need["clock"] = True
            if not b.has_poses and b.track is None:
                need["pose"] = True
            if not b.space.zone_names:
                need["zones"] = True
            if b.space.zone_names and not b.space.calibrated:
                need["scale"] = True
        for key, b in self._need_rows.items():
            b.setVisible(bool(need[key]) and bool(bundles))
        self._setup_box.setVisible(any(
            b.isVisibleTo(self._setup_box) for b in self._need_rows.values()))

    def _set_scale(self):
        """Ask for the scale, in whichever terms the user actually knows it.

        Nobody knows their rig in pixels per centimetre. They know how wide
        the arena is, or they can point at something in the frame, so those
        are the two ways in, and px/cm is only ever shown as the result.
        """
        targets = self._selected_bundles()
        if not targets:
            return
        from PySide6.QtWidgets import (QDialogButtonBox, QFormLayout,
                                       QRadioButton)

        b0 = targets[0]
        zone_names = sorted({z for b in targets for z in b.space.zone_names})

        dlg = QDialog(self)
        dlg.setWindowTitle("Set the scale")
        dlg.setStyleSheet(_THEME.STYLE)
        lay = QVBoxLayout(dlg)
        head = QLabel(
            f"Distances for {len(targets)} recording(s) are in PIXELS. Give "
            f"the tool one real-world length and they become centimetres.")
        head.setWordWrap(True)
        lay.addWidget(head)

        # 1. a zone of known size, the thing an experimenter actually knows
        r_zone = QRadioButton("A zone I drew is a known size")
        r_zone.setChecked(bool(zone_names))
        r_zone.setEnabled(bool(zone_names))
        lay.addWidget(r_zone)
        zf = QFormLayout()
        zone_combo = QComboBox()
        zone_combo.addItems(zone_names)
        zf.addRow("zone", zone_combo)
        span = QComboBox()
        span.addItems(["width", "height"])
        zf.addRow("its", span)
        cm = QDoubleSpinBox()
        cm.setRange(0.1, 1000.0)
        cm.setDecimals(2)
        cm.setValue(40.0)
        cm.setSuffix(" cm")
        zf.addRow("measures", cm)
        lay.addLayout(zf)

        # 2. straight in, for anyone who calibrated elsewhere
        r_px = QRadioButton("I know the scale in pixels per centimetre")
        r_px.setChecked(not zone_names)
        lay.addWidget(r_px)
        pf = QFormLayout()
        ppc = QDoubleSpinBox()
        ppc.setRange(0.01, 1000.0)
        ppc.setDecimals(3)
        ppc.setValue(b0.space.px_per_cm or 10.0)
        ppc.setSuffix(" px/cm")
        pf.addRow("scale", ppc)
        lay.addLayout(pf)

        # 3. the interactive one, which owns the frame
        draw = QPushButton("…or draw a scale line on the video")
        draw.setObjectName("setup")
        draw.setToolTip("Opens the zone editor. Add a “scale” line across "
                        "something you know the length of.")
        lay.addWidget(draw)

        result = QLabel("")
        result.setObjectName("field")
        result.setWordWrap(True)
        lay.addWidget(result)

        def preview():
            val = self._scale_from_dialog(targets, r_zone.isChecked(),
                                          zone_combo.currentText(),
                                          span.currentText(), cm.value(),
                                          ppc.value())
            result.setText(
                f"→ {val:.2f} px/cm" if val > 0 else
                "→ that zone has no measurable extent on this frame")

        for w in (r_zone, r_px):
            w.toggled.connect(lambda *_: preview())
        for w in (zone_combo, span):
            w.currentIndexChanged.connect(lambda *_: preview())
        for w in (cm, ppc):
            w.valueChanged.connect(lambda *_: preview())
        preview()

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        def go_draw():
            dlg.reject()
            self._edit_zones_on_video()
        draw.clicked.connect(go_draw)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        val = self._scale_from_dialog(targets, r_zone.isChecked(),
                                      zone_combo.currentText(),
                                      span.currentText(), cm.value(),
                                      ppc.value())
        if val <= 0:
            QtWidgets.QMessageBox.warning(
                self, "No scale", "That did not give a usable scale.")
            return
        for b in targets:
            b.set_px_per_cm(val)
        self._refresh_file_tree()
        self._status.setText(
            f"Scale set to {val:.2f} px/cm for {len(targets)} recording(s), "
            f"distances are now in centimetres.")

    @staticmethod
    def _scale_from_dialog(targets, by_zone, zone_name, span, cm_value, ppc):
        """px/cm from whichever half of the dialog is active.

        Kept apart from the widgets so the arithmetic can be tested without
        opening anything.
        """
        if not by_zone:
            return float(ppc)
        if not zone_name or cm_value <= 0:
            return 0.0
        for b in targets:
            for z in b.space.zones:
                if z.get("name") != zone_name:
                    continue
                pts = z.get("points") or []
                if len(pts) < 2:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                extent = (max(xs) - min(xs)) if span == "width" else \
                         (max(ys) - min(ys))
                if extent > 0:
                    return extent / float(cm_value)
        return 0.0

    def _build_intent(self):
        box = Section("WHAT TO DO")
        v = QVBoxLayout(box.body())
        v.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(0)
        self._intent_buttons: Dict[str, QPushButton] = {}
        for intent, label, tip in (
            (INTENT_ANALYSE, "Analyse",
             "Measure the poses already in each recording.\n"
             "Corrections are applied while measuring, as always."),
            (INTENT_CORRECT, "Correct",
             "Write the corrections out as their own pose file first, then "
             "measure that.\nThe correction becomes a real file you can open "
             "in Verify or hand to another tool."),
            (INTENT_RETRACK, "Re-track",
             "Track the video again with DeepLabCut, SLEAP or the simple "
             "blob tracker,\nthen measure the poses that produces."),
        ):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setObjectName("seg")
            b.setMinimumWidth(108)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, i=intent: self._set_intent(i))
            self._intent_buttons[intent] = b
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)

        self._intent_hint = QLabel("")
        self._intent_hint.setObjectName("field")
        self._intent_hint.setWordWrap(True)
        v.addWidget(self._intent_hint)

        # The tracker chooser lives inside the intent that needs it, rather
        # than as a button somewhere else that you have to know about.
        self._retrack_row = QWidget()
        rr = QHBoxLayout(self._retrack_row)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.setSpacing(6)
        self._lbl_tracker = QLabel("no tracking method chosen")
        self._lbl_tracker.setObjectName("value")
        btn = QPushButton("Choose tracking method…")
        btn.setObjectName("setup")
        btn.setToolTip("DeepLabCut, SLEAP or the simple blob tracker, the "
                       "same three the live pipeline uses,\nwith the same "
                       "settings.")
        btn.clicked.connect(self._open_retrack_dialog)
        rr.addWidget(btn)
        rr.addWidget(self._lbl_tracker, 1)
        v.addWidget(self._retrack_row)

        # Drawing zones sits with the other things you DO to a recording,
        # under the intent that decides what happens to it. The ZONES page in
        # the sidebar still reports what space the selection is in; this is
        # the door to the editor, where the doing lives.
        zrow = QHBoxLayout()
        zrow.setSpacing(6)
        self._btn_zones_here = QPushButton("✎  Draw / edit zones on video")
        self._btn_zones_here.setObjectName("setup")
        self._btn_zones_here.setToolTip(
            "Opens the recording's own first frame in a window of its own, "
            "and steps through the selection.")
        self._btn_zones_here.clicked.connect(self._edit_zones_on_video)
        zrow.addWidget(self._btn_zones_here)
        self._lbl_zones_here = QLabel("")
        self._lbl_zones_here.setObjectName("field")
        self._lbl_zones_here.setWordWrap(True)
        self._lbl_zones_here.setMinimumWidth(80)
        self._lbl_zones_here.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred)
        zrow.addWidget(self._lbl_zones_here, 1)
        v.addLayout(zrow)

        self._intent_box = box
        return box

    def _set_intent(self, intent: str, *, quiet: bool = False):
        """Adopt an intent and push it onto every selected recording."""
        if intent not in (INTENT_ANALYSE, INTENT_CORRECT, INTENT_RETRACK):
            return
        self._intent = intent
        for key, b in self._intent_buttons.items():
            b.blockSignals(True)
            b.setChecked(key == intent)
            b.blockSignals(False)

        self._retrack_row.setVisible(intent == INTENT_RETRACK)
        self._intent_hint.setText({
            INTENT_ANALYSE:
                "Measures the poses each recording already has. Nothing is "
                "written except the results.",
            INTENT_CORRECT:
                "Writes a corrected pose file into the recording's own "
                "analysis folder, then measures that. Your raw data is not "
                "touched.",
            INTENT_RETRACK:
                "Tracks the video again from scratch. This is the "
                "expensive one, minutes to hours, not seconds.",
        }[intent])

        if not quiet:
            self._apply_intent()

    def _apply_intent(self):
        """Set the bundle state the chosen intent implies.

        Analyse and Correct both clear any tracker, so switching back from
        Re-track does not leave a six-hour job silently queued in the plan.
        """
        for b in self._selected_bundles():
            if self._intent == INTENT_RETRACK:
                b.set_write_corrected(False)
                if self._pending_track is not None:
                    b.set_track(self._pending_track)
            else:
                b.set_track(None)
                b.set_write_corrected(self._intent == INTENT_CORRECT)
        self._refresh_file_tree()

    def _update_tracker_label(self):
        spec = self._pending_track
        if spec is None:
            self._lbl_tracker.setText("no tracking method chosen")
            return
        name = os.path.basename(spec.model_path.rstrip("/\\")) or ", "
        self._lbl_tracker.setText(
            f"{spec.backend}"
            + (f" · {name}" if spec.needs_model else " · no model needed"))

    # ── zones ────────────────────────────────────────────────────

    def _build_zone_group(self):
        """Zones and scale, with the draw action stated as an action.

        "Set up space…" did open the real editor, but nothing said it was a
        draw-on-the-video tool, so the interactive part was effectively
        invisible.
        """
        box = Section("ZONES  &  SCALE")
        v = QVBoxLayout(box.body())
        v.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._btn_draw = QPushButton("✎  Draw / edit zones on video")
        self._btn_draw.setObjectName("setup")
        self._btn_draw.setToolTip(
            "Opens the recording's own first frame and lets you draw zones "
            "directly on it:\nclick to place a polygon, drag to move, "
            "Ctrl+drag to rotate.\n\nAdd a scale line and the results switch "
            "from pixels to centimetres. Applies to the whole selection, and "
            "re-zones in seconds; it does NOT re-run tracking.")
        self._btn_draw.clicked.connect(self._edit_zones_on_video)
        row.addWidget(self._btn_draw)
        self._zone_state = QLabel("")
        self._zone_state.setObjectName("value")
        # Four zone names plus a scale is ~1000 px on one line, which forced
        # the whole panel that wide and pushed it off the window.
        self._zone_state.setWordWrap(True)
        self._zone_state.setMinimumWidth(80)
        self._zone_state.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Preferred)
        row.addStretch()
        v.addLayout(row)
        # Its own line. Beside the button it forced the section 24px wider
        # than the sidebar that holds it, and it wraps to any width anyway.
        v.addWidget(self._zone_state)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self._zone_label = _THEME.field("report")
        row2.addWidget(self._zone_label)
        self._zones_include: set = set()               # empty = all zones
        self._btn_zones = QPushButton("all zones  ▾")
        self._btn_zones.setObjectName("quiet")
        self._zone_menu = StickyMenu(self._btn_zones)
        self._btn_zones.setMenu(self._zone_menu)
        self._btn_zones.setToolTip(
            "Which zones get columns in the results. This filters the OUTPUT "
            ", to change the zones themselves, draw them.")
        row2.addWidget(self._btn_zones)
        self._zone_summary = QLabel("")
        self._zone_summary.setObjectName("field")
        self._zone_summary.setWordWrap(True)
        self._zone_summary.setMinimumWidth(80)
        self._zone_summary.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Preferred)
        row2.addWidget(self._zone_summary, 1)
        row2.addStretch()
        v.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(6)
        self._btn_regions = QPushButton("Regions…")
        self._btn_regions.setObjectName("quiet")
        self._btn_regions.setToolTip(
            "Report several zones as one, Open_1 + Open_2 → Open.\n"
            "Time, entries and distance are summed over the group; the "
            "individual zones keep their own columns too.")
        self._btn_regions.clicked.connect(self._edit_regions)
        row3.addWidget(self._btn_regions)
        self._lbl_regions = QLabel("none")
        self._lbl_regions.setObjectName("value")
        self._lbl_regions.setWordWrap(True)
        self._lbl_regions.setMinimumWidth(80)
        self._lbl_regions.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                        QtWidgets.QSizePolicy.Policy.Preferred)
        row3.addWidget(self._lbl_regions, 1)
        v.addLayout(row3)

        self._zone_box = box
        return box

    def _refresh_zone_state(self):
        """Say what space the selection is in, beside the button that changes it."""
        if not hasattr(self, "_zone_state"):
            return
        bundles = self._selected_bundles()
        if not bundles:
            self._zone_state.setText("")
            return
        self._btn_draw.setEnabled(True)
        # Named as the results will name them, like the picker: one zone must
        # not be `Center` here and `Center_arm` in the sheet.
        rename = self._zone_rename()
        names = sorted({rename.get(z, z)
                        for b in bundles for z in b.space.zone_names})
        scales = {round(b.space.px_per_cm, 2) for b in bundles if b.space.calibrated}
        if not names:
            self._zone_state.setText("no zones yet, draw some to measure them")
            return
        txt = f"{len(names)} zone(s): " + ", ".join(names[:4])
        if len(names) > 4:
            txt += f" +{len(names) - 4}"
        if not scales:
            txt += "   ·   NO SCALE, distances in pixels"
        elif len(scales) == 1:
            txt += f"   ·   {next(iter(scales)):.1f} px/cm"
        else:
            txt += "   ·   scales differ across the selection"
        self._zone_state.setText(txt)
        if hasattr(self, "_lbl_zones_here"):
            self._lbl_zones_here.setText(txt)

    def _edit_regions(self):
        """Group sub-zones into the regions a paper reports.

        An arena is tiled with whatever sub-zones make the mesh work; the
        methods section says "time in the open arms". The aggregation for this
        already existed in `space.py` and nothing called it, so every user was
        adding sub-zone columns up by hand in a spreadsheet.
        """
        from PySide6.QtWidgets import (QDialogButtonBox, QListWidget,
                                       QListWidgetItem)

        from tools.offline_analysis.engine import space as _sp

        zones = self._detect_zone_names()
        if not zones:
            QtWidgets.QMessageBox.information(
                self, "No zones",
                "Draw some zones first, a region is a group of them.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Regions")
        dlg.setStyleSheet(_THEME.STYLE)
        dlg.resize(560, 460)
        lay = QVBoxLayout(dlg)
        head = QLabel(
            "A region is several zones reported as one. Time, entries and "
            "distance are summed over the group, and a frame in two of its "
            "zones is counted once. The individual zones keep their own "
            "columns either way.")
        head.setWordWrap(True)
        lay.addWidget(head)

        lst = QListWidget()
        lst.setToolTip("Tick the zones that belong to this region.")
        lay.addWidget(lst, 1)

        row = QHBoxLayout()
        row.addWidget(_THEME.field("region"))
        pick = QComboBox()
        pick.setEditable(True)
        pick.setMinimumWidth(180)
        row.addWidget(pick, 1)
        btn_del = QPushButton("Remove")
        btn_del.setObjectName("quiet")
        row.addWidget(btn_del)
        lay.addLayout(row)

        suggest = QPushButton("Suggest from the zone names")
        suggest.setObjectName("setup")
        suggest.setToolTip(
            "Open_1 and Open_2 become Open. Only proposes, nothing is "
            "grouped until you accept it.")
        lay.addWidget(suggest)

        note = QLabel("")
        note.setObjectName("field")
        note.setWordWrap(True)
        lay.addWidget(note)

        # Work on a copy; Cancel must leave the run untouched.
        draft = {k: list(v) for k, v in (self._regions or {}).items()}
        current = {"name": ""}

        def refresh_pick():
            pick.blockSignals(True)
            pick.clear()
            pick.addItems(sorted(draft))
            if current["name"] in draft:
                pick.setCurrentText(current["name"])
            pick.blockSignals(False)

        def refresh_list():
            lst.blockSignals(True)
            lst.clear()
            members = set(draft.get(current["name"], []))
            for z in zones:
                it = QListWidgetItem(z)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if z in members
                                 else Qt.CheckState.Unchecked)
                lst.addItem(it)
            lst.blockSignals(False)
            describe()

        def describe():
            if not draft:
                note.setText("No regions, only the individual zones will be "
                             "reported.")
                return
            note.setText("   ·   ".join(
                f"{k} = {len(v)} zone(s)" for k, v in sorted(draft.items())))

        def on_pick(text):
            current["name"] = text.strip()
            if current["name"] and current["name"] not in draft:
                draft[current["name"]] = []
            refresh_list()

        def on_item(item):
            name = current["name"]
            if not name:
                return
            members = set(draft.get(name, []))
            if item.checkState() == Qt.CheckState.Checked:
                members.add(item.text())
            else:
                members.discard(item.text())
            draft[name] = sorted(members)
            describe()

        def on_remove():
            draft.pop(current["name"], None)
            current["name"] = ""
            refresh_pick()
            refresh_list()

        def on_suggest():
            for name, parts in _sp.suggest_regions(zones).items():
                draft.setdefault(name, list(parts))
            if not draft:
                note.setText("Nothing to suggest, no zone names share a stem "
                             "(Open_1 / Open_2).")
                return
            refresh_pick()
            if not current["name"]:
                current["name"] = sorted(draft)[0]
                pick.setCurrentText(current["name"])
            refresh_list()

        pick.editTextChanged.connect(on_pick)
        lst.itemChanged.connect(on_item)
        btn_del.clicked.connect(on_remove)
        suggest.clicked.connect(on_suggest)

        refresh_pick()
        if draft:
            current["name"] = sorted(draft)[0]
            pick.setCurrentText(current["name"])
        refresh_list()

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # A region with no zones would emit nothing; drop it rather than keep
        # a name that silently does nothing.
        self._regions = {k: v for k, v in draft.items() if v}
        self._refresh_region_label()
        self._refresh_columns_preview()
        self._refresh_plan_strip()
        self._status.setText(
            f"{len(self._regions)} region(s) defined."
            if self._regions else "No regions, zones are reported singly.")

    def _refresh_region_label(self):
        if not hasattr(self, "_lbl_regions"):
            return
        if not self._regions:
            self._lbl_regions.setText("none")
            self._lbl_regions.setToolTip("")
            return
        names = sorted(self._regions)
        self._lbl_regions.setText(", ".join(names[:3])
                                  + (f" +{len(names) - 3}" if len(names) > 3
                                     else ""))
        self._lbl_regions.setToolTip(
            "\n".join(f"{k} = " + ", ".join(v)
                       for k, v in sorted(self._regions.items())))

    # ── investigation ────────────────────────────────────────────

    def _build_investigation(self):
        """Sniffing at an object: where the NOSE is, and where it points.

        Time in an interaction area is not investigation. An animal walking
        past a cup is in the area and is not investigating it, which is why
        the measure is head-in-zone AND oriented toward the object.

        All three settings existed in `DEFAULT_INTERACTION` and nothing in the
        app set any of them, and, worse, nothing could MARK a zone as an
        object, so on a rig whose zones are called Object1 rather than
        Object1_IA the analysis could never run at all.
        """
        box = Section("INVESTIGATION", expanded=False)
        v = QVBoxLayout(box.body())
        v.setSpacing(5)

        lead = QLabel(
            "Sniffing and orientation at an object. Scored on the HEAD, and "
            "only while it points at the object, time in the area alone is "
            "not investigation.")
        lead.setObjectName("field")
        lead.setWordWrap(True)
        v.addWidget(lead)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(_THEME.field("objects"))
        self._btn_objects = QPushButton("none  ▾")
        self._btn_objects.setObjectName("quiet")
        self._obj_menu = StickyMenu(self._btn_objects)
        self._btn_objects.setMenu(self._obj_menu)
        self._btn_objects.setToolTip(
            "Which zones are objects to be investigated, rather than places "
            "to be in.\nEach one gets investigation time and entries beside "
            "its ordinary occupancy.")
        row.addWidget(self._btn_objects)
        row.addStretch()
        v.addLayout(row)

        from PySide6.QtWidgets import QFormLayout

        f = QFormLayout()
        f.setSpacing(5)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        f.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._w_angle = QDoubleSpinBox()
        self._w_angle.setRange(5.0, 180.0)
        self._w_angle.setDecimals(0)
        self._w_angle.setSuffix("°")
        self._w_angle.setValue(45.0)
        self._w_angle.setToolTip(
            "How far off the object the head may point and still count as "
            "facing it.\n45° is the usual convention; 30° is stricter, 90° "
            "counts anything not turned away.")
        f.addRow("Facing within", self._w_angle)

        self._w_confirm = QDoubleSpinBox()
        self._w_confirm.setRange(0.0, 2000.0)
        self._w_confirm.setDecimals(0)
        self._w_confirm.setSingleStep(50.0)
        self._w_confirm.setSuffix(" ms")
        self._w_confirm.setValue(100.0)
        self._w_confirm.setToolTip(
            "The head must stay in the area this long before the visit "
            "counts.\nRejects the animal merely passing the object on its "
            "way somewhere else.")
        f.addRow("Settle for", self._w_confirm)

        self._w_target = QComboBox()
        for label, key in (("the object's outward face", "outer_edge"),
                           ("the nearest edge", "nearest_edge"),
                           ("the area's centre", "centroid")):
            self._w_target.addItem(label, key)
        self._w_target.setToolTip(
            "What the animal is judged to be facing.\n"
            "OUTWARD FACE suits an object standing against the arena wall, "
            "the object is at the area's outer edge and that is what the "
            "animal orients to.\n"
            "THE AREA'S CENTRE suits a free-standing object with the area "
            "drawn around it.\n"
            "NEAREST EDGE is the loosest, for arenas where neither holds.")
        f.addRow("Facing", self._w_target)

        self._w_reach = QDoubleSpinBox()
        self._w_reach.setRange(0.0, 500.0)
        self._w_reach.setDecimals(0)
        self._w_reach.setSingleStep(5.0)
        self._w_reach.setSuffix(" px")
        self._w_reach.setValue(50.0)
        self._w_reach.setToolTip(
            "How close to the object's face the nose must reach before the "
            "visit counts at all.\nAn interaction area is drawn big enough "
            "for the animal to stand in, so being inside it is not the same "
            "as being AT the object, sitting at the back of the area facing "
            "the right way would otherwise score as investigation.\n"
            "0 counts anywhere inside the area.")
        f.addRow("Nose within", self._w_reach)

        self._w_extend = QDoubleSpinBox()
        self._w_extend.setRange(0.0, 200.0)
        self._w_extend.setDecimals(0)
        self._w_extend.setSingleStep(5.0)
        self._w_extend.setSuffix(" px")
        self._w_extend.setValue(15.0)
        self._w_extend.setToolTip(
            "How far past the ends of the drawn edge the object's face is "
            "taken to continue.\nThe object is not confined to the width of "
            "the area drawn in front of it: without this, a nose at the "
            "corner measures its angle to the edge's END POINT, off to one "
            "side, and reads as facing away from something it is nose-to.")
        f.addRow("Face extends", self._w_extend)
        v.addLayout(f)

        self._lbl_investigation = QLabel("")
        self._lbl_investigation.setObjectName("field")
        self._lbl_investigation.setWordWrap(True)
        v.addWidget(self._lbl_investigation)

        for w in (self._w_angle, self._w_confirm, self._w_reach,
                  self._w_extend):
            w.valueChanged.connect(self._apply_investigation)
        self._w_target.currentIndexChanged.connect(self._apply_investigation)

        self._investigation_box = box
        self._object_zones: set = set()
        return box

    def _refresh_object_menu(self):
        """Offer this dataset's own zones as candidate objects."""
        if not hasattr(self, "_obj_menu"):
            return
        names = self._detect_zone_names()
        self._obj_menu.clear()
        # As in the zone picker: a profile saved before these names were the
        # analysed ones holds the recorded spelling, and intersecting without
        # translating it would unmark the object the user had marked.
        stale = self._zone_rename()
        self._object_zones = {stale.get(z, z) for z in self._object_zones}
        self._object_zones &= set(names)
        for z in names:
            a = self._obj_menu.addAction(z)
            a.setCheckable(True)
            a.setChecked(z in self._object_zones)
            a.toggled.connect(
                lambda on, name=z: self._on_object_toggled(name, on))
        self._update_object_label()

    def _on_object_toggled(self, name: str, on: bool):
        if on:
            self._object_zones.add(name)
        else:
            self._object_zones.discard(name)
        self._update_object_label()
        self._apply_investigation()

    def _update_object_label(self):
        if not hasattr(self, "_btn_objects"):
            return
        got = sorted(self._object_zones)
        self._btn_objects.setText(
            (", ".join(got[:2]) + (f" +{len(got) - 2}" if len(got) > 2 else "")
             if got else "none") + "  ▾")
        self._describe_investigation()

    def _investigation_cfg(self) -> dict:
        return {"angle_threshold_deg": float(self._w_angle.value()),
                "confirm_ms": float(self._w_confirm.value()),
                "target": self._w_target.currentData() or "outer_edge",
                "inner_offset_px": float(self._w_reach.value()),
                "line_extend_px": float(self._w_extend.value())}

    def _apply_investigation(self, *_):
        """Mark the object zones and push the settings onto every selection."""
        if not hasattr(self, "_object_zones"):
            return
        cfg = self._investigation_cfg()
        for b in self._selected_bundles():
            b.set_interaction(cfg, self._object_zones)
        self._describe_investigation()
        self._refresh_file_tree()
        self._refresh_columns_preview()

    def _describe_investigation(self):
        if not hasattr(self, "_lbl_investigation"):
            return
        got = sorted(self._object_zones)
        if not got:
            self._lbl_investigation.setText(
                "No objects marked, no investigation columns. Ordinary zone "
                "time and entries are unaffected.")
            return
        cols = ", ".join(f"{z}_investigation_s" for z in got[:2])
        extra = " (+2 more each)" if len(got) > 2 else ""
        tail = ("  A discrimination index is added for exactly two objects."
                if len(got) == 2 else "")
        self._lbl_investigation.setText(
            f"{len(got)} object(s) → {cols}{extra}.{tail}")

    def _build_arm_roles(self):
        """Which arm is novel, which familiar, which the animal started in.

        `_ymaze` returns immediately unless one of the first two is set, and
        nothing in the app set either, so novel-arm preference, the
        discrimination index, the latency to the novel arm and spontaneous
        alternation could not be produced at all, on a rig whose whole
        purpose is a Y-maze.

        Roles, not zones: the same three arms mean different things in the
        sample and test phases, and only the experimenter knows which.
        """
        host = QWidget()
        f = QtWidgets.QFormLayout(host)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(4)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        f.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._arm_combos = {}
        for key, label, tip in (
            ("novel_arm", "novel arm",
             "The arm that was blocked during the sample phase. Drives novel "
             "preference, the discrimination index and the latency to it."),
            ("familiar_arm", "familiar arm",
             "The arm that was open throughout, the comparison for novel "
             "preference."),
            ("start_arm", "start arm",
             "Where the animal was placed. Only used to complete the triple "
             "for spontaneous alternation."),
        ):
            c = QComboBox()
            c.setMinimumWidth(120)
            c.setToolTip(tip)
            c.currentIndexChanged.connect(self._on_arm_role_changed)
            self._arm_combos[key] = c
            f.addRow(label, c)

        self._lbl_arms = QLabel("")
        self._lbl_arms.setObjectName("field")
        self._lbl_arms.setWordWrap(True)
        f.addRow("", self._lbl_arms)
        self._arm_host = host
        return host

    def _refresh_arm_roles(self):
        """Offer this dataset's own zones as the three roles."""
        if not hasattr(self, "_arm_combos"):
            return
        names = self._detect_zone_names()
        for c in self._arm_combos.values():
            keep = c.currentText()
            c.blockSignals(True)
            c.clear()
            c.addItem(", ", "")
            for z in names:
                c.addItem(z, z)
            i = c.findData(keep)
            c.setCurrentIndex(i if i > 0 else 0)
            c.blockSignals(False)
        self._arm_host.setVisible(bool(names))
        self._describe_arms()

    def _on_arm_role_changed(self, *_):
        self._describe_arms()
        self._refresh_columns_preview()
        self._refresh_plan_strip()

    def _arm_roles(self) -> dict:
        if not hasattr(self, "_arm_combos"):
            return {"novel_arm": "", "familiar_arm": "", "start_arm": ""}
        return {k: (c.currentData() or "") for k, c in self._arm_combos.items()}

    def _describe_arms(self):
        if not hasattr(self, "_lbl_arms"):
            return
        r = self._arm_roles()
        if not (r["novel_arm"] or r["familiar_arm"]):
            self._lbl_arms.setText(
                "No roles set, no novel-arm columns. Ordinary per-zone time "
                "and entries are unaffected.")
            return
        bits = []
        if r["novel_arm"]:
            bits.append("NovelArm_time_s, _entries, _latency_s")
        if r["novel_arm"] and r["familiar_arm"]:
            bits.append("NovelArm_preference_pct, Discrimination_index")
        if sum(1 for v in r.values() if v) >= 3:
            bits.append("Spontaneous_alternation_pct")
        self._lbl_arms.setText("You will also get: " + "; ".join(bits) + ".")

    def _build_corrections(self):
        """Detector-error corrections, applied to EVERY analysis.

        These are not something you opt into. ``build_track`` runs them on
        every session it measures, so keeping them behind an Advanced drawer
        meant the numbers were being corrected in ways the user could neither
        see nor set.

        Two were not settable at all. The swap resolver had no threshold, and
        it chose its head/body pair by POSITION in the keypoint list, right
        by luck for Head/Center/Tailbase, silently wrong for a model that
        lists its parts in any other order. Both are controls now.
        """
        from PySide6.QtWidgets import QFormLayout

        # Folded by default: these have working defaults, and someone opening
        # the tab is looking for Run, not for a One-Euro beta.
        box = Section("CORRECTIONS", expanded=False)
        outer_v = QVBoxLayout(box.body())
        outer_v.setContentsMargins(0, 0, 0, 0)
        outer_v.setSpacing(4)
        self._corrections_lead = lead = QLabel(
            "Applied to every analysis, whichever intent you pick.")
        lead.setObjectName("field")
        lead.setWordWrap(True)
        outer_v.addWidget(lead)
        form_host = QWidget()
        outer_v.addWidget(form_host)
        f = QFormLayout(form_host)
        f.setSpacing(5)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Label above field when the panel is narrow, instead of demanding
        # the width of both side by side. This is what lets the corrections
        # live in a sidebar at a scaled font.
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        f.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._w_pcut = QDoubleSpinBox()
        self._w_pcut.setRange(0.0, 1.0)
        self._w_pcut.setSingleStep(0.05)
        self._w_pcut.setDecimals(2)
        self._w_pcut.setValue(0.55)
        self._w_pcut.setToolTip(
            "A keypoint below this confidence is treated as MISSING for that "
            "frame rather than as a position. Gaps contribute no distance.")
        f.addRow("Confidence at least", self._w_pcut)

        self._w_swap = QCheckBox("Head/tail swap resolver")
        self._w_swap.setChecked(True)
        self._w_swap.setToolTip(
            "Detectors sometimes flip which keypoint is the head and which is "
            "the tail, and the flip persists until they flip back. The "
            "resolver undoes that: a crossing must both restore continuity "
            "and put the body chain the right way round." + chr(10) + chr(10)
            + "The pair is picked by NAME, the two ENDS of the animal, which "
            "is what makes this safe to leave on. The first two keypoints in "
            "file order are Snout and Head on a SLEAP model, one end against "
            "itself. Comparing two points "
            "a centimetre apart made it fire on ordinary fast motion and "
            "lengthen the measured path." + chr(10) + chr(10)
            + "With no identifiable head AND tail part it does not run.")
        self._w_swap.toggled.connect(self._sync_corrections)
        f.addRow("", self._w_swap)

        # Wrapping: head, body and the threshold on one fixed line needed
        # 584 px, which does not fit the settings sidebar at a scaled font.
        from tools.offline_analysis.analyze.flow_layout import FlowLayout

        row = FlowLayout(spacing=4)
        self._w_swap_head = QComboBox()
        self._w_swap_body = QComboBox()
        for c in (self._w_swap_head, self._w_swap_body):
            c.setMinimumWidth(78)
            c.setToolTip(
                "Which two keypoints can be confused for one another.\n"
                "Left unset, the resolver takes the FIRST TWO in the file's "
                "own order, correct by luck, wrong in silence.")
        row.addWidget(_THEME.field("head"))
        row.addWidget(self._w_swap_head)
        row.addWidget(_THEME.field("body"))
        row.addWidget(self._w_swap_body)
        self._w_swap_thr = QDoubleSpinBox()
        self._w_swap_thr.setRange(0.05, 0.95)
        self._w_swap_thr.setDecimals(2)
        self._w_swap_thr.setSingleStep(0.05)
        self._w_swap_thr.setValue(0.30)
        self._w_swap_thr.setToolTip(
            "How much a swap must improve the cost before it is accepted. "
            "Higher is more conservative.")
        row.addWidget(_THEME.field("if better by"))
        row.addWidget(self._w_swap_thr)
        self._swap_row = QWidget()
        self._swap_row.setLayout(row)
        f.addRow("", self._swap_row)

        # The teleport clamp is a guard, not a preference: it always runs.
        self._w_maxspeed = QDoubleSpinBox()
        self._w_maxspeed.setRange(0, 20)
        self._w_maxspeed.setDecimals(2)
        self._w_maxspeed.setValue(3.8)
        self._w_maxspeed.setSuffix(" m/s")
        self._w_maxspeed.setToolTip(
            "A step faster than this is a detector jump, not an animal, and "
            "is frozen to the previous position.\n"
            "A mouse dart reaches about 3.3 m/s, so keep this high or real "
            "motion gets clipped.")
        f.addRow("Teleport above", self._w_maxspeed)

        self._w_median = QCheckBox("Rolling median gap-fill (5 frames)")
        self._w_median.setToolTip(
            "Fill isolated missing frames from their neighbours. Off by "
            "default, matching the reference pipeline.")
        f.addRow("", self._w_median)

        self._w_smooth = QCheckBox("One-Euro smoothing")
        self._w_smooth.setChecked(True)
        self._w_smooth.setToolTip(
            "Adaptive low-lag filter. Per-frame jitter otherwise inflates "
            "distance by roughly 12%; this removes it while preserving darts.")
        self._w_smooth.toggled.connect(self._sync_corrections)
        f.addRow("", self._w_smooth)

        row2 = FlowLayout(spacing=4)
        self._w_euro_mc = QDoubleSpinBox()
        self._w_euro_mc.setRange(0.1, 10.0)
        self._w_euro_mc.setDecimals(2)
        self._w_euro_mc.setValue(1.0)
        self._w_euro_mc.setToolTip("Lower is smoother, and laggier.")
        self._w_euro_b = QDoubleSpinBox()
        self._w_euro_b.setRange(0.0, 1.0)
        self._w_euro_b.setDecimals(4)
        self._w_euro_b.setSingleStep(0.001)
        self._w_euro_b.setValue(0.007)
        self._w_euro_b.setToolTip("How much fast motion is let through.")
        row2.addWidget(_THEME.field("cutoff"))
        row2.addWidget(self._w_euro_mc)
        row2.addWidget(_THEME.field("beta"))
        row2.addWidget(self._w_euro_b)
        self._euro_row = QWidget()
        self._euro_row.setLayout(row2)
        f.addRow("", self._euro_row)

        self._w_minmove = QDoubleSpinBox()
        self._w_minmove.setRange(0, 0.05)
        self._w_minmove.setSingleStep(0.001)
        self._w_minmove.setDecimals(4)
        self._w_minmove.setSuffix(" m")
        self._w_minmove.setToolTip(
            "Optional de-jitter on top: drop sub-threshold steps. 0 is the "
            "raw sum of pose displacements, which is what the reference "
            "pipeline reports.")
        f.addRow("Ignore steps under", self._w_minmove)

        self._smooth_note = QLabel("")
        self._smooth_note.setObjectName("note")
        self._smooth_note.setWordWrap(True)
        self._smooth_note.setVisible(False)
        f.addRow("", self._smooth_note)

        self._corrections_box = box
        return box

    def _sync_corrections(self, *_):
        """A correction's own settings show only while it is switched on."""
        if not hasattr(self, "_swap_row"):
            return
        self._swap_row.setVisible(self._w_swap.isChecked())
        self._euro_row.setVisible(self._w_smooth.isChecked())


    def _refresh_swap_parts(self):
        """Offer this dataset's actual keypoints as the swap pair.

        The pair is chosen by the ENGINE's own chooser, so the dialog and the
        analysis cannot disagree about which two points a swap is between,
        they used to, the dialog preferring a centre point and the analysis
        falling back to the first two in file order.
        """
        if not hasattr(self, "_w_swap_head"):
            return
        from tools.offline_analysis.engine.offline_analysis import (
            default_swap_pair)

        parts = self._detect_body_parts()
        head, body = default_swap_pair(parts)
        for combo, suggested in ((self._w_swap_head, head),
                                 (self._w_swap_body, body)):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(parts)
            if current in parts:
                combo.setCurrentText(current)
            elif suggested:
                combo.setCurrentText(suggested)
            combo.blockSignals(False)
        self._note_swap_pair(head, body)

    def _note_swap_pair(self, head, body):
        """Say which pair the resolver will use, or that it will not run."""
        if not hasattr(self, "_w_swap"):
            return
        if head and body:
            self._w_swap.setText(f"Head/tail swap resolver  ({head} ↔ {body})")
            self._w_swap.setEnabled(True)
        else:
            self._w_swap.setText("Head/tail swap resolver  (no head/tail pair)")
            # Nothing to compare: the engine leaves it off, so the box must
            # not suggest otherwise.
            self._w_swap.setEnabled(False)

    # ── measures: thresholds and the column preview ──────────────

    #: How the bins are described. The label is what the row reads as.
    BIN_MODES = (("every", "every"), ("count", "split into"),
                 ("edges", "at seconds"))

    def _build_bins_row(self):
        """How to cut the recording up, beside the switch that asks for it.

        One width was the only shape on offer, and it was two folds down in
        Advanced. Experiments ask three ways: "every minute" (habituation
        curves), "in thirds" (early/middle/late, whatever the session length),
        and "at 0, 120, 600" (before and after an event). All three produce
        the same ``{lo}-{hi}s_`` columns, so nothing downstream changes.
        """
        host = QWidget()
        h = QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(self._field_label("bins"))

        self._w_bin_mode = QComboBox()
        for key, label in self.BIN_MODES:
            self._w_bin_mode.addItem(label, key)
        self._w_bin_mode.setToolTip(
            "every, a fixed width, repeated to the end" \
            + chr(10) + "split into, N equal bins, whatever the run lasts" \
            + chr(10) + "at seconds, your own boundaries, comma separated")
        self._w_bin_mode.setMaximumWidth(150)
        self._w_bin_mode.currentIndexChanged.connect(self._sync_bins)
        h.addWidget(self._w_bin_mode)

        self._w_bin_every = QSpinBox()
        self._w_bin_every.setRange(1, 36000); self._w_bin_every.setValue(60)
        self._w_bin_every.setSuffix(" s")
        self._w_bin_every.setMaximumWidth(150)
        h.addWidget(self._w_bin_every)

        self._w_bin_count = QSpinBox()
        self._w_bin_count.setRange(2, 500); self._w_bin_count.setValue(3)
        self._w_bin_count.setSuffix(" bins")
        self._w_bin_count.setMaximumWidth(150)
        h.addWidget(self._w_bin_count)

        self._w_bin_edges = QLineEdit()
        self._w_bin_edges.setPlaceholderText("0, 120, 300")
        self._w_bin_edges.setToolTip(
            "Boundaries in seconds. 0 is added at the front and the end of "
            "the recording at the back, so \"120\" means two bins.")
        self._w_bin_edges.setMinimumWidth(80)
        self._w_bin_edges.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                        QtWidgets.QSizePolicy.Policy.Fixed)
        h.addWidget(self._w_bin_edges, 1)
        h.addStretch()

        for w in (self._w_bin_every, self._w_bin_count):
            w.valueChanged.connect(self._describe_bins)
        self._w_bin_edges.textChanged.connect(self._describe_bins)

        self._bins_host = host
        host.setVisible(False)          # the switch is off until it is not
        return host

    def _bin_edges(self):
        """The typed boundaries, as seconds. Unreadable text is no boundary."""
        out = []
        for piece in (self._w_bin_edges.text() or "").replace(";", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                out.append(float(piece))
            except ValueError:
                continue
        return sorted({v for v in out if v >= 0})

    def _bin_params(self) -> dict:
        """The three keys the analysis reads, only one of them ever set."""
        if not self._chips["timebins"].isChecked():
            return {"bin_size_s": 0.0, "bin_count": 0, "bin_edges_s": []}
        mode = self._w_bin_mode.currentData() or "every"
        if mode == "count":
            return {"bin_size_s": 0.0, "bin_count": int(self._w_bin_count.value()),
                    "bin_edges_s": []}
        if mode == "edges":
            return {"bin_size_s": 0.0, "bin_count": 0,
                    "bin_edges_s": self._bin_edges()}
        return {"bin_size_s": float(self._w_bin_every.value()),
                "bin_count": 0, "bin_edges_s": []}

    def _sync_bins(self, *_):
        """Show the switch's own row, and only the field its mode uses."""
        if not hasattr(self, "_w_bin_mode"):
            return
        on = self._chips["timebins"].isChecked()
        self._bins_host.setVisible(on)
        mode = self._w_bin_mode.currentData() or "every"
        self._w_bin_every.setVisible(on and mode == "every")
        self._w_bin_count.setVisible(on and mode == "count")
        self._w_bin_edges.setVisible(on and mode == "edges")
        self._describe_bins()

    def _describe_bins(self, *_):
        """Name the windows the current setting produces, on a real length."""
        if not hasattr(self, "_lbl_bins"):
            return
        if not self._chips["timebins"].isChecked():
            self._lbl_bins.setText("")
            return
        from tools.offline_analysis.engine import offline_analysis as oa

        secs = self._typical_duration_s()
        spans = oa.bin_spans(secs, self._bin_params())
        if not spans:
            self._lbl_bins.setText("No bins, nothing to split on.")
            return
        # The names the COLUMNS will carry, not a second set invented here.
        tags = oa.bin_tags(spans, self._bin_params())
        shown = ", ".join(tags[:4])
        more = f" … +{len(spans) - 4} more" if len(spans) > 4 else ""
        if tags and tags[0].startswith("bin"):
            # An ordinal does not depend on how long anything ran, so quoting
            # a duration here would be noise, and a guessed one, misleading.
            self._lbl_bins.setText(
                f"{shown}{more}, the same columns however long each "
                f"recording is. The seconds ride along beside them.")
            return
        # Say so when the length is a guess: "5 bins over 300 s" reads as a
        # fact about these recordings, and before they are read it is not one.
        how = ("" if getattr(self, "_duration_is_real", False)
               else " (assuming 300 s, run once and this uses the real one)")
        self._lbl_bins.setText(
            f"{len(spans)} bin(s) over {int(secs)} s: {shown}{more}{how}")

    def _typical_duration_s(self) -> float:
        """How long the selected recordings run, for the bins preview.

        The preview is worthless against a made-up 300 s when the rig records
        ten minutes, so it uses the shortest selected recording, the one that
        decides whether the last bin is a stub.
        """
        self._duration_is_real = False
        best = 0.0
        for sf in self._selected_sessions():
            b = self._bundles.get(sf.txt_path)
            d = float(getattr(getattr(b, "clock", None), "duration_s", 0) or 0)
            if d > 0:
                best = d if best == 0 else min(best, d)
        self._duration_is_real = best > 0
        return best or 300.0

    def _reg_row(self, measure: str, form, label: str, widget):
        """Add a threshold row and record which measure consumes it.

        A threshold for a measure you have switched off is a number you cannot
        act on. Registering the row lets the panel shrink to the settings that
        are actually in play, which is most of what made it forbidding.
        """
        form.addRow(label, widget)
        self._threshold_rows.setdefault(measure, []).append((form, widget))

    def _sync_trim(self, *_):
        """The window's fields show only while it is switched on."""
        if not hasattr(self, "_w_trim_from"):
            return
        on = self._w_trim.isChecked()
        for w in (self._w_trim_from, self._w_trim_to):
            w.setVisible(on)
            lbl = (self._trim_form.labelForField(w)
                   if hasattr(self, "_trim_form") else None)
            if lbl is not None:
                lbl.setVisible(on)
        self._on_trim_changed()

    def _on_trim_changed(self, *_):
        self._refresh_columns_preview()
        self._refresh_plan_strip()

    def _sync_thresholds(self):
        """Show each threshold only while its measure is ticked."""
        for measure, rows in self._threshold_rows.items():
            chip = self._chips.get(measure)
            # `isHidden`, not `isVisibleTo`: the chips live in the settings
            # sidebar, which is closed at rest, so `isVisibleTo` was False for
            # every one of them and every threshold hid itself the moment the
            # sidebar was shut. `isHidden` asks the only question that
            # matters, did the WORKSPACE hide this measure.
            on = bool(chip and chip.isChecked() and not chip.isHidden())
            for form, widget in rows:
                widget.setVisible(on)
                label = form.labelForField(widget)
                if label is not None:
                    label.setVisible(on)

    def _on_measure_toggled(self, _checked: bool = False):
        """A measure changed: the thresholds it owns, the columns it produces
        and the plan's measure key all follow from it."""
        self._sync_thresholds()
        self._sync_bins()
        self._refresh_columns_preview()
        self._refresh_plan_strip()

    def _preview_session(self):
        """A ten-frame stand-in carrying this dataset's shape.

        Real body-part names, real zone names, the right dimensionality, so
        the emitter produces exactly the columns a real run will, without
        parsing a nine-thousand-frame file on every click.
        """
        from tools.offline_analysis.engine import offline_analysis as oa

        parts = self._detect_body_parts() or ["Center"]
        zone_names = self._detect_zone_names()
        n = 10
        sess = oa.Session(path="preview", stem="preview", subject="preview",
                          stage="preview")
        sess.ts_ms = np.arange(n, dtype=float) * 50.0
        sess.px_per_cm = 10.0
        sess.resolution = (640, 480)
        sess.body_parts = list(parts)
        sess.kp = {p: {"x": np.linspace(10, 60, n),
                       "y": np.linspace(10, 60, n),
                       "conf": np.ones(n)} for p in parts}
        sess.zones = [{"name": z, "type": "polygon", "coord_space": "pixel",
                       "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}
                      for z in zone_names]
        sess.location = [(zone_names[i % len(zone_names)] if zone_names else "")
                         for i in range(n)]
        sess.stage_series = ["preview"] * n
        return sess

    def _refresh_columns_preview(self):
        """Name the columns this selection will produce, before it is run."""
        if not hasattr(self, "_cols_preview"):
            return
        from tools.offline_analysis.engine import offline_analysis as oa

        try:
            ep = AnalysisWorker([], self._collect_metrics(),
                                self._collect_params())._engine_params()
            rows = oa.session_rows(self._preview_session(), ep)
            cols = oa.measure_columns(oa.order_row_columns(rows))
        except Exception as e:                      # never break the panel
            logger.debug("column preview: %s", e)
            self._cols_preview.setText("")
            return
        if not cols:
            self._cols_preview.setText(
                "No measures selected, the result would be identity columns "
                "only.")
            self._cols_preview.setToolTip("")
            return
        head = ", ".join(cols[:8])
        more = f"  … +{len(cols) - 8} more (hover)" if len(cols) > 8 else ""
        self._cols_preview.setText(
            f"You will get {len(cols)} measure columns:  {head}{more}")
        self._cols_preview.setToolTip("\n".join(cols))

    def _build_advanced(self) -> QWidget:
        outer = QFrame()
        outer.setObjectName("advDrawer")
        outer.setStyleSheet(f"QFrame#advDrawer{{background:{_THEME.SUNKEN};"
                            f"border:1px solid {_THEME.LINE};border-radius:4px;}}")
        grid = QtWidgets.QGridLayout(outer); grid.setContentsMargins(10, 9, 10, 9)
        # Stacked, not side by side: this drawer lives in a 560px sidebar,
        # and three columns of forms needed nearly a thousand.
        grid.setHorizontalSpacing(20); grid.setVerticalSpacing(10)

        # De-jitter and smoothing are NOT here: they apply to every analysis,
        # so they belong in the always-visible CORRECTIONS group rather than
        # behind a fold, see `_build_corrections`.

        # Immobility / freezing
        box, f, _ = self._adv_form("Immobility / freezing")
        self._w_freezethr = QDoubleSpinBox(); self._w_freezethr.setRange(0, 1); self._w_freezethr.setDecimals(3); self._w_freezethr.setSingleStep(0.005); self._w_freezethr.setValue(0.020); self._w_freezethr.setSuffix(" m/s")
        self._reg_row("immobility", f, "Immobile <", self._w_freezethr)
        self._w_freezestrict = QDoubleSpinBox(); self._w_freezestrict.setRange(0, 1); self._w_freezestrict.setDecimals(3); self._w_freezestrict.setSingleStep(0.005); self._w_freezestrict.setValue(0.005); self._w_freezestrict.setSuffix(" m/s")
        self._w_freezestrict.setToolTip("Stricter cut for FREEZING (respiration-only). 0.005 m/s = 0.5 cm/s.")
        self._reg_row("freezing", f, "Freeze <", self._w_freezestrict)
        self._w_freezemin = QSpinBox(); self._w_freezemin.setRange(100, 30000); self._w_freezemin.setValue(1000); self._w_freezemin.setSuffix(" ms")
        self._reg_row("immobility", f, "Min bout", self._w_freezemin)
        self._w_dwell = QDoubleSpinBox(); self._w_dwell.setRange(0, 10); self._w_dwell.setSingleStep(0.05); self._w_dwell.setDecimals(2); self._w_dwell.setValue(0.20); self._w_dwell.setSuffix(" s")
        self._reg_row("zones", f, "Min dwell", self._w_dwell)
        grid.addWidget(box, 0, 0)

        self._adv_scale_box, f, _ = self._adv_form("Scale")
        box = self._adv_scale_box
        self._w_ppc = QDoubleSpinBox(); self._w_ppc.setRange(0, 1000); self._w_ppc.setDecimals(2)
        self._w_ppc.setSpecialValueText("auto"); self._w_ppc.setToolTip("Pixels/cm. 0 = read each file's own scale.")
        f.addRow("px / cm", self._w_ppc)
        # Bin size is NOT here: it belongs beside the Time bins switch that
        # turns it on, see `_build_bins_row`. Two folds deep, it can only
        # offer one shape of bin and nobody finds it.
        grid.addWidget(box, 1, 0)


        # Output
        box, f, vv = self._adv_form("Output")
        self._w_plots: Dict[str, QCheckBox] = {}
        for key, label in [("trajectory", "Track plot"), ("heatmap", "Heatmap"),
                           ("zone_bar", "Zone timeline"), ("speed_profile", "Distance && velocity")]:
            c = QCheckBox(label); self._w_plots[key] = c; vv.addWidget(c)
        self._w_trim = QCheckBox("Only part of each recording")
        self._w_trim.setToolTip(
            "Measure a window instead of the whole session, the first "
            "five minutes of an open field, say." + chr(10) +
            "The window is relative to the START of the recording (or of the "
            "stage, when rows are split by stage).")
        self._w_trim.toggled.connect(self._sync_trim)
        vv.addWidget(self._w_trim)

        self._w_trim_from = QDoubleSpinBox()
        self._w_trim_from.setRange(0.0, 86400.0)
        self._w_trim_from.setDecimals(0)
        self._w_trim_from.setSuffix(" s")
        self._w_trim_from.setToolTip("Skip this much from the start.")
        tf = QFormLayout()
        tf.setContentsMargins(16, 0, 0, 0)
        tf.setSpacing(3)
        tf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        vv.addLayout(tf)
        self._trim_form = tf
        tf.addRow("From", self._w_trim_from)

        self._w_trim_to = QDoubleSpinBox()
        self._w_trim_to.setRange(0.0, 86400.0)
        self._w_trim_to.setDecimals(0)
        self._w_trim_to.setValue(300.0)
        self._w_trim_to.setSuffix(" s")
        self._w_trim_to.setSpecialValueText("to the end")
        self._w_trim_to.setToolTip(
            "Stop here. Zero means run to the end of the recording.")
        tf.addRow("To", self._w_trim_to)
        for w in (self._w_trim_from, self._w_trim_to):
            w.valueChanged.connect(self._on_trim_changed)
        self._w_trim_from.hide()
        self._w_trim_to.hide()

        self._w_split = QCheckBox("One row per protocol stage")
        self._w_split.setChecked(True)
        self._w_split.setToolTip(
            "A protocol with fourteen stages is fourteen result rows.\n"
            "Untick to collapse a recording to a single whole-session row; "
            "a recording with no stages is unaffected either way.")
        vv.addWidget(self._w_split)

        self._w_heatmap_bins = QSpinBox()
        self._w_heatmap_bins.setRange(10, 400)
        self._w_heatmap_bins.setValue(40)
        self._w_heatmap_bins.setToolTip(
            "Grid resolution of the occupancy heatmap, per axis.")
        f.addRow("Heatmap bins", self._w_heatmap_bins)

        self._w_adv_transitions = QCheckBox("Zone transitions")
        self._w_adv_transitions.setChecked(True)
        self._w_adv_transitions.setToolTip(
            "Transitions_total in the summary, and a Transitions sheet in the "
            "export naming which zone led to which.\n"
            "A Y-maze alternation, a place bias and a perseverative loop all "
            "live in the ordered pairs and in none of the summary columns.")
        vv.addWidget(self._w_adv_transitions)
        self._w_adv_zonedist = QCheckBox("Per-zone exits / mean visit"); vv.addWidget(self._w_adv_zonedist)
        bc = QPushButton("Choose / rename columns…")
        bc.setObjectName("quiet")
        bc.clicked.connect(self._open_columns_dialog); vv.addWidget(bc)
        grid.addWidget(box, 2, 0)

        # A number is three to seven characters wide. Let each field stretch
        # the width of the panel and the drawer reads as a stack of bars with
        # one digit lost at the end of each.
        for w in outer.findChildren(QtWidgets.QAbstractSpinBox):
            w.setMaximumWidth(150)
        return outer

    def add_videos(self, paths):
        """Load bare videos, no sidecar, nothing tracked yet.

        The way in for a session recorded without tracking. No stub file is
        written: the bundle carries the recording's derived state, so a video
        with no data file is a first-class row whose Plan column says exactly
        what it still needs.

        Public because the picker above the view owns every way of getting
        recordings in. This was `_load_videos_to_retrack`, which opened its
        own dialog from a "+ Load" menu, the menu was dropped in the port and
        took the only route to bare videos with it, leaving this method in the
        file wired to nothing.
        """
        if paths:
            self._add_videos([str(p) for p in paths])

    # ── drag & drop ──────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        txts, vids = [], []
        for p in paths:
            if os.path.isdir(p):
                t, v = self._scan_folder(p)
                txts += t
                vids += v
            elif p.lower().endswith(".txt"):
                txts.append(p)
            elif p.lower().endswith(VIDEO_EXTS):
                vids.append(p)
        # Dropping a video used to pop a box telling you to go and use a menu
        # item instead, the app knew exactly what you had given it.
        if txts:
            self._add_files(sorted(set(txts)))
        if vids:
            self._add_videos(sorted(set(vids)))
        e.acceptProposedAction()

    def _remove_selected_files(self):
        """Remove the highlighted rows.

        Row indices are into the VISIBLE workspace, so they are mapped back to
        paths before removal, indexing `_sessions` directly would delete a
        different recording whenever the other workspace held any.
        """
        rows = {ix.row() for ix in self._file_table.selectedIndexes()}
        if not rows:
            return
        shown = self._visible_sessions()
        doomed = {shown[r].txt_path for r in rows if 0 <= r < len(shown)}
        self._sessions = [sf for sf in self._sessions
                          if sf.txt_path not in doomed]
        for p in doomed:
            self._bundles.pop(p, None)
        self._refresh_file_tree()

    def _first_frame(self, sf: "SessionFile"):
        """The frame zones are drawn on: this recording's own, never a live
        camera; this is offline analysis."""
        import cv2

        if not (sf.video_path and os.path.exists(sf.video_path)):
            return None
        cap = cv2.VideoCapture(sf.video_path)
        try:
            ok, fr = cap.read()
        finally:
            cap.release()
        return fr if ok else None

    # ── zones, as a panel you stay in ────────────────────────────

    def _zone_panel(self):
        """The editor, built on first use and kept afterwards."""
        if getattr(self, "_zonepanel", None) is None:
            from tools.offline_analysis.analyze.zone_workbench import ZoneWorkbench

            self._zonepanel = ZoneWorkbench(
                frame_for=self._frame_for_key,
                zones_for=self._zones_for_key,
                apply_to=self._apply_space_by_key,
                scale_for=self._scale_for_key,
                size_for=self._frame_size_for_key,
                zoned=self.zoned_keys,
                write_files=self.write_zones_into_files,
                parent=self)
            self._zonepanel.finished.connect(self._close_zone_panel)
            self._zonepanel.applied.connect(self._on_zones_applied)
            self._left_stack.addWidget(self._zonepanel)
        return self._zonepanel

    def _session_for_key(self, key: str):
        return next((sf for sf in self._sessions if sf.txt_path == key), None)

    def _frame_for_key(self, key: str):
        sf = self._session_for_key(key)
        return self._first_frame(sf) if sf is not None else None

    def _zones_for_key(self, key: str) -> list:
        """The zones this recording is CURRENTLY using, in PIXELS.

        Always the bundle's space, never the raw header. Zones are stored
        NORMALIZED in the file, and the editor draws in pixels, handing it
        the header's own numbers put all four zones inside the top-left
        1x1 pixel of a 360x288 frame. Their names appeared in the zone list,
        so it looked like they had loaded, and the video showed nothing.

        `space_from_header` already does the conversion, and the same object
        carries any edit made this session, so there is one path and it is
        the correct one.
        """
        sf = self._session_for_key(key)
        if sf is None:
            return []
        return [dict(z) for z in self._bundle_for(sf).space.zones]

    def _frame_size_for_key(self, key: str):
        """The frame the recording's zones are expressed against.

        Needed when the video is missing: the zones are still meaningful, and
        without a size they would be drawn and written against whichever
        recording happened to be on screen before.
        """
        sf = self._session_for_key(key)
        if sf is None:
            return (0, 0)
        return tuple(self._bundle_for(sf).space.frame_size or (0, 0))

    def _scale_for_key(self, key: str) -> float:
        """px/cm this recording is currently using, scale line or not."""
        sf = self._session_for_key(key)
        return self._bundle_for(sf).space.px_per_cm if sf is not None else 0.0

    def _apply_space_by_key(self, zone_dicts: list, keys: list,
                            frame_size) -> int:
        targets = [sf for sf in (self._session_for_key(k) for k in keys)
                   if sf is not None]
        if not targets:
            return 0
        return self._apply_space(zone_dicts, targets, frame_size)

    def zoned_keys(self) -> List[str]:
        """Loaded recordings that currently carry zones, in table order.

        What "write these into the files" would touch, so the panel can say
        the number before asking.
        """
        out: List[str] = []
        for sf in self._visible_sessions():
            try:
                if self._bundle_for(sf).space.zone_names:
                    out.append(sf.txt_path)
            except Exception:
                continue
        return out

    @staticmethod
    def _reference_frame(bundle) -> Tuple[int, int]:
        """The frame these zones are drawn on, in pixels.

        The header is asked first, then the video itself. A recorder that
        wrote ``"resolution":"0x0"``, several did, leaves the header saying
        nothing, and the video is the recording's own answer to the same
        question.
        """
        w, h = (bundle.space.frame_size or (0, 0))
        if w and h:
            return int(w), int(h)
        video = getattr(bundle, "video_path", "")
        if video and os.path.exists(video):
            from tools.offline_analysis.engine.clock import probe_video

            probe = probe_video(video)
            if probe.ok and probe.width and probe.height:
                return int(probe.width), int(probe.height)
        return 0, 0

    def write_zones_into_files(self, keys) -> Tuple[int, List[str]]:
        """Write each recording's zones into its OWN data file.

        The analyser's own copy lives in the bundle beside the recording, and
        that is the safe default, nothing the operator does while analysing
        can damage the data. It is not sufficient, though: carry the recording
        to another machine, or hand it to a colleague, and the zones drawn for
        it are not in the file. This puts them where the rig would have put
        them, in the recording's own header block.

        Zones are written NORMALIZED, with the frame size beside them, which
        is the form the rig writes and the form that survives a change of
        resolution. Everything else in the file, including every per-frame
        row, is copied through untouched.

        Returns ``(written, failures)`` rather than raising: one unwritable
        file in a cohort must not lose the rest.
        """
        from tools.offline_analysis import video_data_schema as vds
        from tools.offline_analysis.vendor.zone_coords import to_norm

        written, failures = 0, []
        for key in keys:
            sf = self._session_for_key(key)
            if sf is None:
                continue
            path = sf.txt_path or ""
            if not path.lower().endswith(".txt") or not os.path.exists(path):
                failures.append(f"{os.path.basename(path) or key}: not a data file")
                continue
            bundle = self._bundle_for(sf)
            frame = self._reference_frame(bundle)
            if not (frame[0] and frame[1]):
                # Zones are shapes ON a frame. Writing them with no frame size
                # beside them would store numbers nothing can interpret, and
                # this recording's header declares 0x0, which is exactly the
                # case that would go wrong silently.
                failures.append(
                    f"{os.path.basename(path)}: frame size unknown (the "
                    f"header says 0x0 and the video could not be measured), "
                    f"so the zones have nothing to be relative to")
                continue
            zones = []
            for z in bundle.space.zones or []:
                z = dict(z)
                points = z.get("points") or []
                if points:
                    z["points"] = to_norm(points, frame[0], frame[1])
                    z["coord_space"] = "normalized"
                    z["shape_dim"] = [int(frame[0]), int(frame[1])]
                zones.append(z)
            try:
                vds.write_zone_config(path, zones)
                written += 1
                self._log.append(
                    f"{bundle.stem}: {len(bundle.space.zone_names)} zone(s) "
                    f"written into {os.path.basename(path)}")
            except Exception as e:
                logger.exception("writing zones into %s failed", path)
                failures.append(f"{os.path.basename(path)}: {e}")
        if written:
            # The file now says what the bundle says, so a re-read agrees.
            for key in keys:
                self._bundles.pop(key, None)
            self._refresh_file_tree()
        return written, failures

    def _on_zones_applied(self, n: int):
        self._refresh_file_tree()
        self._status.setText(
            f"Zones written to {n} recording(s), Run will re-zone them "
            f"(seconds; the poses are not touched).")

    def _close_zone_panel(self):
        self._left_stack.setCurrentIndex(0)
        self._refresh_file_tree()

    def _edit_zones_on_video(self):
        """Open the zone editor in a window of its own.

        It needs the recording's frame at a workable size, a zone list and a
        shape palette; the settings sidebar is 560 px wide and the options
        column narrower still, so a window is the only place it fits. It
        navigates the recordings itself, so this is not the modal-per-file
        arrangement it replaced.

        NON-modal, deliberately. `exec()` blocks its caller, which is wrong
        twice over: you cannot consult the recordings table while drawing on
        one of them, and a headless test that calls this would hang forever
        with nothing to dismiss the dialog.
        """
        shown = self._visible_sessions()
        if not shown:
            QtWidgets.QMessageBox.information(self, "No Data",
                                              "Load recordings first.")
            return
        rows = sorted({ix.row() for ix in self._file_table.selectedIndexes()
                       if 0 <= ix.row() < len(shown)})
        start = rows[0] if rows else 0

        dlg = getattr(self, "_zone_window", None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Zones and scale")
            dlg.setStyleSheet(_THEME.STYLE)
            dlg.resize(1150, 760)
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(6, 6, 6, 6)
            panel = self._zone_panel()
            lay.addWidget(panel)
            # It was built inside a QStackedWidget, which hides every page but
            # the current one, reparenting does not clear that, so the window
            # came up blank.
            panel.show()
            # Its own Back button closes the window instead of swapping a
            # stack, and closing refreshes what the edit changed.
            panel.finished.connect(dlg.close)
            dlg.finished.connect(lambda _=0: self._refresh_file_tree())
            self._zone_window = dlg

        self._zone_panel().set_recordings(
            [(sf.name or sf.subject, sf.txt_path) for sf in shown], start)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def _apply_space(self, zone_dicts: list, targets: list,
                     frame_size: Tuple[int, int]) -> int:
        """Push an edited zone set into the targets' bundles.

        A scale LINE among the zones recalibrates the recording; that is how
        a session with no scale gains one. But most recordings carry their
        px/cm in the header and no scale line at all, and for those the line
        is simply absent, not zero: drawing a zone must not decalibrate a
        recording that was already calibrated.

        It did. `px_per_cm_from_zones` returns 0 when there is no scale zone,
        and that 0 was written straight over the header's value, so opening
        the zone editor and touching anything silently turned every distance
        into pixels and then reported "No scale" for a recording that had one.
        """
        from tools.offline_analysis.engine import space as _sp

        pixel_zones = _sp.zones_to_pixel(zone_dicts or [], frame_size)
        drawn = _sp.px_per_cm_from_zones(pixel_zones, frame_size)
        applied = 0
        lost = False
        for t in targets:
            b = self._bundle_for(t)
            # The drawn line wins; otherwise keep whatever this recording was
            # already calibrated at.
            ppc = drawn or b.space.px_per_cm
            model = _sp.SpaceModel(frame_size=frame_size, zones=pixel_zones,
                                   px_per_cm=float(ppc or 0.0),
                                   origin="edited")
            b.set_space(model)
            applied += 1
            lost = lost or not ppc
        if drawn:
            self._log.append(
                f"Scale line: {drawn:.2f} px/cm applied to {applied} "
                f"recording(s).")
        elif lost:
            self._log.append(
                "No scale line, and these recordings declare none, distances "
                "will be reported in PIXELS. Draw a scale zone to get "
                "centimetres.")
        return applied

    def _open_retrack_dialog(self):
        """Choose the detector, and ONLY the detector.

        This used to open the recording pipeline's TrackingConfigDialog,
        1,700 lines and two tabs, one of which is a zone editor, plus event
        triggers, coordinate mapping and MCU-push gates. Clicking "choose a
        tracking method" therefore put a zone editor on screen, and offered
        settings that mean nothing for a file recorded weeks ago: there is no
        MCU to push to and no trigger left to fire.

        `OfflineTrackerDialog` asks the one question and emits the same config
        shape, so `_spec_from_tracking_config` is still the only translation.
        The live dialog is untouched.
        """
        from tools.offline_analysis.engine.retrack2d import BACKEND_BLOB
        from tools.offline_analysis.analyze.offline_tracker import OfflineTrackerDialog

        sel = self._selected_sessions() or self._sessions
        if not sel:
            QtWidgets.QMessageBox.information(self, "No Data",
                                              "Load recordings first.")
            return
        cfg = self._opts.get("_tracking_config", {})
        dlg = OfflineTrackerDialog(cfg, parent=self, theme=_THEME.STYLE)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        tc = dlg.get_tracking_config()
        self._opts["_tracking_config"] = tc
        spec = self._spec_from_tracking_config(tc)

        # Refuse now, with the reason, rather than six hours into a run.
        problems = spec.validate()
        if problems:
            QtWidgets.QMessageBox.warning(
                self, "Tracker not usable",
                "This tracker cannot run here:\n\n• " + "\n• ".join(problems)
                + ("\n\nThe blob detector needs no model and no GPU."
                   if spec.backend != BACKEND_BLOB else ""))
            return

        # The choice is remembered, and applied by the intent. Picking a
        # tracker is not the same as asking for a six-hour run.
        self._pending_track = spec
        self._update_tracker_label()
        self._set_intent(INTENT_RETRACK)

        # A block here read zones back out of the dialog, from the days when
        # this opened the recording pipeline's editor. `OfflineTrackerDialog`
        # has no zones and there was no `frame` in scope either, so it raised
        # a NameError into a `logger.debug` on every single use. Zones are
        # drawn with "Draw / edit zones on video"; deleted rather than
        # revived, because a tracker dialog is not where you edit an arena.

        self._refresh_file_tree()
        self._status.setText(
            f"Tracking method set to {spec.backend} for {len(sel)} "
            f"recording(s).")

    @staticmethod
    def _spec_from_tracking_config(tc: dict):
        """The recording pipeline's tracking config → an offline TrackSpec.

        One translation, so the offline run uses the same detector settings the
        live pipeline would have used.
        """
        from tools.offline_analysis.engine.retrack2d import TrackSpec

        method = tc.get("method", "background_subtraction")
        dlc = tc.get("dlc", {}) or {}
        sleap = tc.get("sleap", {}) or {}
        if method == "deeplabcut":
            model, conf = dlc.get("model_path", ""), dlc.get("confidence", 0.55)
            resize = float(dlc.get("resize", 1.0) or 1.0)
        elif method == "sleap":
            model, conf = sleap.get("model_path", ""), sleap.get("confidence", 0.55)
            resize = 1.0
        else:
            model, conf, resize = "", 0.0, 1.0
        blob = {k: tc[k] for k in (
            "threshold", "min_area", "max_area", "detect_dark", "use_clahe",
            "clahe_clip_limit", "clahe_tile_size", "use_adaptive_threshold",
            "use_illumination_norm", "blur_kernel_size", "blur_mode", "bg_mode",
            "open_kernel_size", "close_kernel_size") if k in tc}
        # Carry the declared body-part names, exactly as the live pipeline
        # does in `tracking_controller._init_pose_model`.
        parts = (dlc if method == "deeplabcut" else sleap).get("body_parts") or []
        if parts:
            blob["body_parts"] = list(parts)
        # How the frame becomes the model's input, and which engine runs it.
        # Carried in `params` deliberately: `TrackSpec.fingerprint()` hashes
        # that dict whole, so a retrack under a different mode or engine gets
        # its own stored result instead of being served the old one. A field on
        # the dataclass would NOT be hashed, the trap the plan calls D2.
        for src_key, dst_key in (("pose_input_mode", "input_mode"),
                                 ("pose_input_w", "input_w"),
                                 ("pose_input_h", "input_h"),
                                 ("pose_crop_conf_min", "crop_conf_min"),
                                 ("pose_crop_good_min", "crop_good_min"),
                                 ("pose_crop_reacquire", "crop_reacquire")):
            if tc.get(src_key) not in (None, ""):
                blob[dst_key] = tc[src_key]
        pose_sub = dlc if method == "deeplabcut" else sleap
        for key in ("runtime", "device", "precision", "centroid_model_path"):
            if pose_sub.get(key):
                blob[key] = pose_sub[key]
        if method == "deeplabcut" and pose_sub.get("model_type"):
            blob["dlc_model_type"] = pose_sub["model_type"]
        return TrackSpec(backend=method, model_path=model,
                         confidence=float(conf or 0.0), resize=resize,
                         n_instances=int(tc.get("n_animals", 1) or 1),
                         params=blob)

    def _detect_zone_names(self) -> List[str]:
        """Zone names across loaded sessions, AS THE ANALYSIS WILL NAME THEM.

        A zone whose name collides with a body point is renamed when the file
        is parsed, the Y-maze's `Center`, beside a body point of the same
        name, becomes `Center_arm`. Listing the recorded name here meant the
        picker offered a zone that no column, plot or filter would ever answer
        to: the filter holds every zone by default, so `Center` matched
        nothing and the choice point was silently dropped from every result.
        """
        rename = self._zone_rename()
        return [rename.get(n, n) for n in self._recorded_zone_names()]

    def _recorded_zone_names(self) -> List[str]:
        """Non-scale zone names exactly as the loaded files spell them."""
        names: List[str] = []
        for sf in self._visible_sessions():
            for z in getattr(sf, "zones", []) or []:
                z = str(z).strip()
                if z and z.lower() not in ("scale", "na", "none") and z not in names:
                    names.append(z)
        return names

    def _zone_rename(self) -> Dict[str, str]:
        """``{recorded name: analysed name}`` for this dataset.

        Built from ALL of the dataset's zones, because the tag a renamed zone
        takes follows its siblings, `Center` beside `Home_arm` becomes
        `Center_arm`, and beside `nest` it becomes `Center_zone`. Asking the
        rule about one name in isolation gets a different answer.
        """
        from tools.offline_analysis.engine import offline_analysis as oa

        key = self._dataset_key()
        memo = getattr(self, "_rename_memo", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        rename = oa.zone_rename_map(self._recorded_zone_names(),
                                    self._detect_body_parts())
        self._rename_memo = (key, rename)
        return rename

    def _refresh_zone_menu(self):
        """Rebuild the Zones checklist from loaded data (default: all selected)."""
        zones = self._detect_zone_names()
        self._zone_menu.clear()
        if not self._zones_include:
            self._zones_include = set(zones)          # first load → all on
        else:
            # A profile saved before the picker used the analysed names holds
            # the recorded ones; intersecting without translating them would
            # untick the very zones the user had chosen.
            stale = self._zone_rename()
            self._zones_include = {stale.get(z, z) for z in self._zones_include}
            self._zones_include &= set(zones)
        for z in zones:
            a = self._zone_menu.addAction(z); a.setCheckable(True)
            a.setChecked(z in self._zones_include)
            a.toggled.connect(lambda on, name=z: self._on_zone_action(name, on))
        self._update_zone_summary()

    def _on_zone_action(self, name: str, on: bool):
        if on:
            self._zones_include.add(name)
        else:
            self._zones_include.discard(name)
        self._update_zone_summary()

    def _update_zone_summary(self):
        """The button says the state; the label beside it says the total.

        A button reading "Select zones ▾" is an instruction rather than a
        value, and then nothing on screen says which zones you actually get.
        """
        total = len(self._detect_zone_names())
        sel = len(self._zones_include)
        noun = "zones"
        if not total:
            self._btn_zones.setText(f"all {noun}  ▾")
            self._zone_summary.setText(f"no {noun} in these recordings")
            return
        if sel >= total or sel == 0:
            self._btn_zones.setText(f"all {noun}  ▾")
            self._zone_summary.setText(f"({total})")
        else:
            names = sorted(self._zones_include)
            head = names[0] if sel == 1 else f"{sel} zones"
            self._btn_zones.setText(f"{head}  ▾")
            self._zone_summary.setText(f"of {total}")

    def _populate_dynamic_combos(self):
        """After loading, fill the zone + body-point pickers from the data."""
        self._refresh_zone_menu()
        self._refresh_bodypart_menu()

    def _refresh_bodypart_menu(self):
        """Rebuild the OPTIONAL per-part breakdown list. The primary distance/zone
        always uses the whole-body centroid (like the reference); ticking points here
        just ADDS per-part columns (Head_Distance_m…). Default: none ticked."""
        parts = self._detect_body_parts()
        self._bp_menu.clear()
        self._bp_include &= set(parts)             # drop parts not in this dataset
        for p in parts:
            a = self._bp_menu.addAction(p); a.setCheckable(True)
            a.setChecked(p in self._bp_include)
            a.toggled.connect(lambda on, name=p: self._on_bodypart_action(name, on))
        self._update_bodypart_label()

    def _on_bodypart_action(self, name: str, on: bool):
        if on:
            self._bp_include.add(name)
        else:                                       # may empty → centroid only
            self._bp_include.discard(name)
        self._update_bodypart_label()

    def _zone_point(self) -> str:
        """Primary point for distance/zone = the Centre keypoint (body centre). It
        tracks translation only; the multi-KP centroid would add head/tail rotation."""
        return "Center"

    def _update_bodypart_label(self):
        extra = sorted(p for p in self._bp_include if p != "Center")
        txt = "Center" if not extra else "Center + " + ", ".join(extra)
        self._btn_bodyparts.setText(f"{txt}  ▾")
        if not hasattr(self, "_lbl_points"):
            return
        if not extra:
            self._lbl_points.setText(
                "Distance and zone occupancy follow the centre point. Tick "
                "another to ADD its own distance, speed and per-zone columns "
                "beside it.")
        else:
            self._lbl_points.setText(
                "Also measuring " + ", ".join(extra) + ", each adds "
                + ", ".join(f"{p}_Distance_m" for p in extra[:2])
                + (" and more" if len(extra) > 2 else "") + ".")

    # ── Dialog openers ───────────────────────────────────────────

    def _open_columns_dialog(self):
        """Curate output columns. Uses the current result's columns when a run
        has completed, otherwise dry-previews one loaded file to discover them."""
        cols = self._available_columns()
        if not cols:
            QtWidgets.QMessageBox.information(
                self, "No Columns",
                "Load data and Run once (or load at least one file) so the "
                "available measure columns can be discovered.")
            return
        dlg = ColumnsDialog(self._opts, cols, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.apply()
            # re-render the current results with the new curation, if any
            if self._current_summary_df is not None:
                self._populate_table(self._summary_table, self._current_summary_df)

    def _available_columns(self) -> List[str]:
        """Column names available for curation, from the last result if present,
        else from a fast single-file engine preview of the first session."""
        if self._current_summary_df is not None:
            return [c for c in self._current_summary_df.columns
                    if not str(c).startswith("__")]
        pool = self._selected_sessions() or self._sessions
        if not pool:
            return []
        try:
            worker = AnalysisWorker(pool[:1], self._collect_metrics(),
                                    self._collect_params())
            row, _, _ = worker._analyze_one(pool[0])
            return [c for c in row if not str(c).startswith("__")]
        except Exception as e:
            logger.warning(f"column preview failed: {e}")
            return []

    def _dataset_key(self) -> tuple:
        """Identity of what is loaded, for the answers derived from it.

        Body parts and zone names cannot change unless the set of recordings
        does, or a file on disk does. Both are asked for many times per
        refresh, so the answer is computed once per set and kept until the set
        moves.
        """
        out = []
        for sf in self._visible_sessions():
            p = sf.txt_path or ""
            try:
                st = os.stat(p)
                out.append((p, st.st_mtime_ns, st.st_size))
            except OSError:
                out.append((p, 0, 0))
        return tuple(out)

    def _detect_body_parts(self):
        """The dataset's keypoint names, or ``["center"]`` when it has none.

        Read from the header where the recording declares it and from a few
        rows where it does not (:func:`body_parts_of`), then memoised per
        loaded set. It used to parse every frame of every recording into a
        DataFrame, on each of the ten calls a single refresh makes: opening
        one file cost fifteen seconds and a folder took minutes.
        """
        key = self._dataset_key()
        memo = getattr(self, "_parts_memo", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        parts = []
        for sf in self._visible_sessions():
            parts = body_parts_of(sf.txt_path)
            if parts:
                break
        parts = parts or ["center"]                       # fallback
        self._parts_memo = (key, parts)
        return parts

    # ── Data loading ────────────────────────────────────────────

    # ── the project selector above the tabs ──────────────────────────
    #
    # This rig's analyser has a session picker over both tabs; pyBehaveTrack
    # loads files from inside the tab instead. Two methods bridge them, and
    # they route through the tab's OWN loader rather than reaching into its
    # table, so a pushed session and a session loaded by the tab's own button
    # arrive by the same path and cannot end up in different states.

    def set_project_context(self, ctx):
        """Remember which project the selector opened.

        Used for the recording paths it carries; the tab keeps its own view of
        what is loaded, so nothing else changes here.
        """
        self._project_ctx = ctx
        directory = getattr(ctx, "project_dir", "") or ""
        if directory:
            logger.info("analyze tab: project context %s", directory)

    def set_top_sessions(self, sessions):
        """Load what the picker pushed.

        The picker's rows carry a data file, a video, or both. Anything with
        neither is skipped rather than added as an empty row: a recording that
        cannot be read is not a recording, and the Plan column would have
        nothing to say about it.
        """
        if sessions is not None and len(sessions) == 0:
            # An EMPTY push is the picker's Clear, and that is now the only
            # Clear in the window, this tab used to carry a second one. A
            # non-empty push with nothing readable in it is a different thing
            # and is handled below: it leaves what is loaded alone.
            self._clear_all()
            return
        paths: List[str] = []
        for sess in (sessions or []):
            for attr in ("txt_path", "video_data_path", "video_path"):
                value = getattr(sess, attr, None) or (
                    sess.get(attr) if isinstance(sess, dict) else None)
                if value and str(value) not in paths and os.path.exists(str(value)):
                    paths.append(str(value))
                    break
        if not paths:
            logger.info("analyze tab: the picker pushed nothing readable")
            return
        self._add_paths(paths)
        logger.info("analyze tab: %d recording(s) pushed from the picker",
                    len(paths))

    def _add_paths(self, paths: List[str]):
        """Route a mixed pile of paths to the reader that fits each one."""
        txts = [p for p in paths if p.lower().endswith(".txt")]
        vids = [p for p in paths if p.lower().endswith(VIDEO_EXTS)]
        if txts:
            self._add_files(txts)
        if vids:
            self._add_videos(vids)
        if not txts and not vids:
            QtWidgets.QMessageBox.information(
                self, "Nothing to load",
                "No tracking data files or videos in that selection.")

    @staticmethod
    def _scan_folder(folder: str):
        """Data files, plus the videos that do not already have one.

        A video whose sidecar is being loaded anyway must not arrive twice,
        once as a recording and once as a bare video with no poses.
        """
        txts = sorted(glob.glob(os.path.join(folder, "**", "*_video_data.txt"),
                                recursive=True))
        if not txts:
            txts = sorted(glob.glob(os.path.join(folder, "**", "*.txt"),
                                    recursive=True))
        vids = []
        for ext in VIDEO_EXTS:
            vids += glob.glob(os.path.join(folder, "**", "*" + ext),
                              recursive=True)
        from tools.offline_analysis.engine.session_bundle import find_txt_for_video

        loose = sorted(v for v in vids if not find_txt_for_video(v))
        return sorted(set(txts)), loose

    def autoload_project(self, project_name: str = "", project_dir: str = "") -> int:
        """Populate the file list with a project's recordings, no dialog.

        Called when the analysis window opens with a project loaded so the user
        lands on all of that project's recordings ready to analyze/retrack.
        Recordings are written to ``<repo>/data/<project_name>/<ddmmyy>/``; the
        project folder itself may also hold retracked / manually-dropped .txt
        files. Both roots are scanned; ``_add_files`` de-dupes, so calling this
        again (e.g. after switching projects) only adds what's new. Returns the
        number of files found.
        """
        from tools.offline_analysis.analyze.paths import DATA_DIR

        roots = []
        if project_name:
            roots.append(DATA_DIR / project_name)
        if project_dir:
            roots.append(Path(project_dir))

        found: List[str] = []
        for root in roots:
            try:
                if root and Path(root).is_dir():
                    found += glob.glob(
                        os.path.join(str(root), "**", "*_video_data.txt"),
                        recursive=True)
            except Exception:
                continue
        found = sorted({os.path.abspath(f) for f in found})

        if found:
            self._add_files(found)
            self._log.append(
                f"Auto-loaded {len(found)} recording(s) for project "
                f"'{project_name or project_dir}'.")
        else:
            self._log.append(
                f"No recordings found yet for project "
                f"'{project_name or project_dir}'.")
        return len(found)

    def _add_videos(self, video_paths: List[str]):
        """Add bare videos as first-class recordings.

        No stub file is written next to the user's data. The old version wrote
        a `B`/`M`/`V` header that the analysis parser ignored entirely, so the
        session analysed to nothing, and it copied zones out of whatever
        project happened to be open, i.e. another rig's geometry.

        A video with no sidecar simply has no clock, no poses and no space yet;
        its Plan cell says which of those to supply.
        """
        from tools.offline_analysis.engine.session_bundle import Bundle, find_txt_for_video

        added = 0
        for vp in video_paths:
            vp = os.path.abspath(vp)
            txt = find_txt_for_video(vp)
            key = txt or vp
            if any(sf.txt_path == key for sf in self._sessions):
                continue
            bundle = Bundle.open(txt_path=txt, video_path=vp)
            self._bundles[key] = bundle
            sf = (parse_txt_header(txt) if txt
                  else SessionFile(txt_path=vp, video_path=vp,
                                   subject=bundle.stem))
            sf.txt_path = key
            sf.video_path = vp
            self._sessions.append(sf)
            added += 1
            if not bundle.clock.usable:
                self._log.append(
                    f"{bundle.stem}: frame rate unknown, set it with "
                    f"“Frame rate…” before running.")
        if video_paths:
            self._remember_dir(video_paths[-1])
        self._refresh_file_tree()
        self._log.append(f"Added {added} video(s) ({len(self._sessions)} total)")

    def _set_frame_rate(self):
        """Tell the tool the frame rate of videos that do not declare one.

        Rank 4 of the clock: an assumption, recorded as such, so every
        per-second measure downstream can be labelled honestly.
        """
        targets = self._selected_bundles()
        if not targets:
            return
        current = next((b.clock.fps for b in targets if b.clock.fps), 25.0)
        fps, ok = QtWidgets.QInputDialog.getDouble(
            self, "Frame rate",
            "Frames per second for the selected recording(s).\n"
            "Used only when the video does not carry timestamps of its own:",
            float(current or 25.0), 0.1, 1000.0, 2)
        if not ok:
            return
        for b in targets:
            b.set_user_fps(fps)
        self._refresh_file_tree()
        self._status.setText(f"Frame rate set to {fps:g} fps "
                             f"for {len(targets)} recording(s).")

    def _load_metadata(self):
        """Load a subject-metadata Excel file (sex, strain, weight, …).

        Columns are matched to subjects via a Subject_ID / subject_id /
        Subject / Mouse_ID column. All other columns are merged into the
        results table next to the per-session metrics.
        """
        default = self._default_data_dir()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Metadata File", default,
            "Spreadsheets (*.xlsx *.xls *.csv);;All Files (*)")
        if not filename:
            return
        try:
            if filename.lower().endswith(".csv"):
                df = pd.read_csv(filename)
            else:
                excel = pd.ExcelFile(filename)
                if len(excel.sheet_names) > 1:
                    sheet, ok = QtWidgets.QInputDialog.getItem(
                        self, "Select Sheet",
                        f"{len(excel.sheet_names)} sheets found, choose one:",
                        excel.sheet_names, 0, False)
                    if not ok:
                        return
                    df = pd.read_excel(filename, sheet_name=sheet)
                else:
                    df = pd.read_excel(filename)
            df.columns = df.columns.str.strip()
            # Ask the user which column holds subject IDs. We don't
            # auto-detect any more, comparing against hardcoded column
            # names misfired on sheets with odd headers (e.g. MouseID).
            from tools.offline_analysis.analyze.subject_column import pick_subject_column
            id_col = pick_subject_column(df, parent=self)
            if id_col is None:
                self._log.append("Metadata load cancelled.")
                return
            # Coerce to string so matches against recording subject IDs
            # are exact regardless of int/str storage in the sheet.
            df[id_col] = df[id_col].astype(str)
            self._metadata_df = df
            self._metadata_id_col = id_col
            self._metadata_path = filename
            n_rows = len(df)
            n_cols = len([c for c in df.columns if c != id_col])
            self._meta_label.setText(
                f"Metadata: {os.path.basename(filename)} "
                f"({n_rows} subjects, {n_cols} fields, id={id_col!r})")
            self._log.append(
                f"Loaded metadata: {filename} ({n_rows} rows, id={id_col!r})")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Metadata Load Error", f"Failed to load metadata:\n{e}")
            logger.error(f"Metadata load failed: {e}", exc_info=True)

    def _add_files(self, txt_paths: List[str]):
        """Parse headers via ``parse_txt_header()`` and add to the session list.

        Skips files already loaded (deduplication by path).  Refreshes the
        file tree and updates the Run button state after adding.
        """
        existing = {sf.txt_path for sf in self._sessions}
        added = 0
        for path in txt_paths:
            if path in existing:
                continue
            sf = parse_txt_header(path)
            self._sessions.append(sf)
            self._bundles.pop(path, None)      # re-read state from disk
            existing.add(path)
            added += 1
        if txt_paths:                       # remember the folder for next Load dialog
            self._remember_dir(txt_paths[-1])

        self._refresh_file_tree()
        self._log.append(
            f"Added {added} files ({len(self._sessions)} total)")
        # Run's enabled state belongs to the plan, not to "are there files",
        # `_refresh_file_tree` has already set it. Re-enabling here would offer
        # Run for a selection where every recording is blocked.

    def _remember_dir(self, path: str):
        """Persist the last-used folder across sessions (opens there next time)."""
        d = path if os.path.isdir(path) else os.path.dirname(path)
        if d and os.path.isdir(d):
            self._last_dir = d
            try:
                self._settings.setValue("last_dir", d)
            except Exception:
                pass

    def _refresh_file_tree(self):
        """Repopulate the readiness table and the plan strip."""
        tbl = self._file_table
        tbl.blockSignals(True)                     # don't fire itemChanged while rebuilding
        # Painting is off for the rebuild. Every setItem on a VISIBLE table
        # schedules its own repaint, and this runs on every click: with thirty
        # recordings a refresh cost two seconds shown against 130 ms hidden,
        # and all of the difference was paint.
        tbl.setUpdatesEnabled(False)
        shown = self._visible_sessions()
        tbl.setRowCount(len(shown))
        status_counts: Dict[str, int] = {}
        for r, sf in enumerate(shown):
            status = (sf.header or {}).get("_project_status", "")
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1
            chk = QTableWidgetItem()
            chk.setFlags((chk.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                         & ~Qt.ItemFlag.ItemIsEditable)
            chk.setCheckState(Qt.CheckState.Unchecked
                              if sf.txt_path in self._excluded_paths
                              else Qt.CheckState.Checked)
            chk.setToolTip(sf.txt_path)
            tbl.setItem(r, 0, chk)

            try:
                b = self._bundle_for(sf)
                ready = b.readiness()
                tips = self._readiness_tooltips(b)
            except Exception as e:                        # never let one bad file
                logger.warning("readiness for %s: %s", sf.txt_path, e)
                ready = {"recording": sf.name, "video": "?", "clock": "?",
                         "pose": "?", "space": "?", "plan": f"unreadable: {e}"}
                tips = {}
            # The px/cm belongs to the whole selection and is stated in
            # ZONES & SCALE on the right, so repeating it per row costs ~85 px
            # of the column the Plan needs. The tooltip still carries it. The
            # "· px" no-scale warning stays: that one is not a duplicate, it
            # is the difference between centimetres and pixels.
            space = _PX_PER_CM_CELL.sub("", str(ready["space"]))
            cells = [ready["recording"], ready["video"], ready["clock"],
                     ready["pose"], space, ready["plan"]]
            for c, text in enumerate(cells, start=1):
                it = QTableWidgetItem(str(text))
                it.setToolTip(tips.get(c, sf.txt_path))
                if c == 6:                               # the Plan cell
                    it.setForeground(QtGui.QColor(
                        "#e46a6a" if str(text).startswith("BLOCKED") else
                        "#8bd4a8" if str(text) == "up to date" else "#f0c988"))
                elif ", " in str(text):
                    it.setForeground(QtGui.QColor("#6e7681"))
                tbl.setItem(r, c, it)
        # Column widths are measured from every cell's text, so doing it on
        # each refresh is the second half of that two seconds. The widths only
        # need to change when the set of recordings does, a re-read of the
        # same rows says the same words.
        if len(shown) != getattr(self, "_sized_for_rows", -1):
            try:
                tbl.resizeColumnsToContents()
                self._sized_for_rows = len(shown)
            except Exception:
                pass
        tbl.setUpdatesEnabled(True)
        tbl.blockSignals(False)
        self._populate_dynamic_combos()
        self._refresh_swap_parts()
        self._refresh_zone_state()
        self._refresh_object_menu()
        self._refresh_arm_roles()
        self._refresh_region_label()
        self._refresh_setup_group()
        self._update_file_count(status_counts)
        self._refresh_columns_preview()   # zone names changed -> columns did
        self._refresh_plan_strip()

    def _readiness_tooltips(self, bundle) -> Dict[int, str]:
        """Where each fact came from. A cell the user cannot interrogate is a
        cell they have to take on trust."""
        clk = bundle.clock
        tip_clock = (f"{clk.source} ({clk.confidence})\n{clk.note}\n"
                     f"{clk.n_frames} rows, "
                     f"{'variable' if clk.vfr else 'constant'} rate")
        if clk.warnings:
            tip_clock += "\n\n! " + "\n! ".join(clk.warnings)
        tracker = (bundle.header.tracker if bundle.header else {}) or {}
        tip_pose = (f"backend: {tracker.get('backend', 'unknown')}\n"
                    f"model: {tracker.get('model_id') or ', '}"
                    if tracker else "no tracker declared in the header")
        if bundle.track is not None:
            tip_pose += f"\n\nwill retrack with: {bundle.track.backend}"
        sp = bundle.space
        tip_space = (f"{len(sp.zone_names)} zone(s), origin: {sp.origin}\n"
                     + (f"scale: {sp.px_per_cm:.2f} px/cm"
                        if sp.calibrated else
                        "NO SCALE, distances will be in pixels"))
        steps = bundle.plan()
        tip_plan = "\n".join(f"{s.label}: {s.why}" for s in steps) or \
            "everything is up to date"
        return {2: bundle.video_path or "no video found",
                3: tip_clock, 4: tip_pose, 5: tip_space, 6: tip_plan}

    def _measure_params(self) -> dict:
        """The measure-stage key, computed exactly as the run computes it.

        `analysis.pipeline` hashes this subset to decide whether metrics are
        out of date. The plan strip must hash the same thing or it describes a
        different run from the one the button performs.
        """
        from tools.offline_analysis.engine import pipeline as _pl

        try:
            ep = AnalysisWorker([], self._collect_metrics(),
                                self._collect_params())._engine_params()
        except Exception as e:                    # never break the strip
            logger.debug("measure params: %s", e)
            return {}
        return _pl._measure_key(ep)

    def _refresh_plan_strip(self):
        """The one-line truth about what Run is about to do."""
        if not hasattr(self, "_plan_label"):
            return
        from tools.offline_analysis.engine import session_bundle as _sb

        bundles = self._selected_bundles()
        if not bundles:
            self._plan_label.setText("")
            self._btn_run.setEnabled(False)
            return
        # The SAME measure parameters the run will use. `summarise` hashes
        # them to decide whether the measure stage is stale, so passing
        # nothing here made the strip and the run disagree: the strip
        # announced "3x measure" and the run correctly found nothing to do.
        s = _sb.summarise(bundles, self._measure_params())
        parts = [f"{ln['n']}× {ln['stage']} ({ln['label']})" for ln in s["steps"]]
        if parts:
            text = " · ".join(parts)
            if s["blocked"]:
                text += f"   ✗ {len(s['blocked'])} blocked and will be skipped"
        elif s["blocked"]:
            # "nothing to do, all up to date" beside a blocked count is a
            # contradiction: nothing is up to date, nothing can run.
            first = s["blocked"][0][1]
            text = (f"✗ nothing can run, {len(s['blocked'])} blocked "
                    f"({first})" if len(s["blocked"]) > 1
                    else f"✗ nothing can run, {first}")
        else:
            text = "already measured, Run shows the results"
        self._plan_label.setText(f"{len(bundles)} selected · {text}")
        # objectName, not an inline colour: the theme owns both cases.
        self._plan_label.setObjectName(
            "planBlocked" if (s["blocked"] and not parts) else "plan")
        self._plan_label.style().polish(self._plan_label)
        self._plan_label.setToolTip(
            "\n".join(f"{stem}: {why}" for stem, why in s["blocked"])
            or "no blocked recordings")
        # Runnable whenever something is selected and not everything is
        # blocked, NOT only when there is fresh work. Reopening the app on
        # recordings measured yesterday planned no steps, so Run was greyed
        # out and there was no way left to see the numbers: the results
        # existed only as a side effect of doing the work again. Pressing Run
        # on an up-to-date selection costs about a second per file and shows
        # what is already there.
        runnable = bool(s["steps"]) or not (s["blocked"] and not parts)
        self._btn_run.setEnabled(runnable)
        self._btn_dry.setEnabled(runnable)
        # Saving the corrected poses depends on the SELECTION, not on a run
        # having just finished, the files survive between sessions.
        if hasattr(self, "_btn_poses"):
            self._btn_poses.setEnabled(self._have_derived_poses())

        # Point at the control that unblocks the selection, so a blocked row
        # is a signpost rather than a dead end. Both controls live in the
        # sidebar now, the left panel used to carry a second copy of each,
        # so the highlight goes on the Re-track intent and the zone button.
        needs_tracker = any("tracker" in why for _stem, why in s["blocked"])
        btn = self._intent_buttons.get(INTENT_RETRACK)
        if btn is not None:
            # A dynamic property the stylesheet reacts to, so the highlight
            # lives with the theme rather than as a second inline style.
            btn.setProperty("wanted", "true" if needs_tracker else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        has_video = any(b.has_video for b in bundles)
        if hasattr(self, "_btn_draw"):
            self._btn_draw.setEnabled(
                has_video or any(b.space.zones for b in bundles))
            self._btn_draw.setToolTip(
                self._btn_draw.toolTip().split(chr(10) * 2)[0] if has_video
                else "No video in this selection; there is no frame to "
                     "draw zones on.")

    def _update_file_count(self, status_counts: Optional[dict] = None):
        """What is loaded, and how much of it Run would touch."""
        shown = self._visible_sessions()
        sel = len(self._selected_sessions())
        if not shown:
            self._file_count.setText("No recordings loaded")
            return
        summary = f"{sel} of {len(shown)} selected"
        if status_counts:
            summary += "  |  " + ", ".join(
                f"{v} {k}" for k, v in status_counts.items())
        self._file_count.setText(summary)

    def _selected_sessions(self) -> List[SessionFile]:
        """Ticked recordings in THIS workspace, what Run analyzes.

        Workspace-scoped deliberately: a run must never quietly reach into
        recordings the user cannot currently see.
        """
        return [sf for sf in self._visible_sessions()
                if sf.txt_path not in self._excluded_paths]

    def _on_file_item_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        row = item.row()
        shown = self._visible_sessions()
        if not (0 <= row < len(shown)):
            return
        path = shown[row].txt_path
        if item.checkState() == Qt.CheckState.Checked:
            self._excluded_paths.discard(path)
        else:
            self._excluded_paths.add(path)
        self._update_file_count()
        self._refresh_plan_strip()

    def _set_all_included(self, include: bool):
        """Applies to the visible workspace only, see `_selected_sessions`."""
        paths = {sf.txt_path for sf in self._visible_sessions()}
        if include:
            self._excluded_paths -= paths
        else:
            self._excluded_paths |= paths
        self._refresh_file_tree()

    def _default_data_dir(self) -> str:
        last = getattr(self, "_last_dir", "")     # remember where you were last
        if last and os.path.isdir(last):
            return last
        try:
            from tools.offline_analysis.analyze.paths import DATA_DIR
            if DATA_DIR.exists():
                return str(DATA_DIR)
        except Exception:
            pass
        return os.getcwd()

    # ── Analysis ────────────────────────────────────────────────

    def _collect_metrics(self) -> Dict[str, bool]:
        """Construct chips → the engine's fine-grained metric flags."""
        loco = self._chips["locomotion"].isChecked()
        zones = self._chips["zones"].isChecked()
        m = {
            "distance": loco, "speed": loco,
            "zone_time": zones, "zone_entries": zones, "latency": zones,
            "zone_distance": zones,   # distance travelled inside each zone/arm
            "immobility": self._chips["immobility"].isChecked(),
            "zone_transitions": self._w_adv_transitions.isChecked(),
            "zone_advanced": self._w_adv_zonedist.isChecked(),
        }
        m["freezing"] = self._chips["freezing"].isChecked()
        for k, c in self._w_plots.items():
            m[k] = c.isChecked()
        return m

    def _collect_params(self) -> dict:
        """Thresholds/curation read from the inline widgets, the single source
        the engine bridge (:meth:`AnalysisWorker._engine_params`) consumes, so Run
        and the column preview stay identical. No paradigm labels: objective only."""
        return {
            "pixels_per_cm": self._w_ppc.value(),
            # UI is in m/s and m; the engine works in cm internally → ×100
            "freeze_threshold": self._w_freezethr.value() * 100.0,     # immobility cut (cm/s)
            "freeze_strict_cm_s": self._w_freezestrict.value() * 100.0,  # freezing cut (cm/s)
            "freeze_min_ms": self._w_freezemin.value(),
            "immobility_thresh": self._w_freezethr.value() * 100.0,
            "min_dwell_time": self._w_dwell.value(),
            "time_bin_enabled": self._chips["timebins"].isChecked(),
            # One control, three shapes. `_bin_params` returns exactly the
            # three keys the analysis reads, with only one of them set.
            **self._bin_params(),
            # zone occupancy uses ONE point: Center if ticked, else the first ticked
            "body_part_primary": self._zone_point(),
            "body_parts_zone": [self._zone_point()],
            # every ticked body point gets its own {part}_Distance_m columns
            "body_parts_multi": sorted(self._bp_include),
            # advanced zone measures (mean visit + per-zone distance)
            "zone_advanced": self._w_adv_zonedist.isChecked(),
            # tracking-error corrections + de-jitter / clamp (UI in m → engine cm)
            # Corrections, applied to EVERY analysis by `build_track`.
            "pcut": self._w_pcut.value(),
            "swap_correct": self._w_swap.isChecked(),
            "swap_cost_threshold": self._w_swap_thr.value(),
            "head_part": self._w_swap_head.currentText(),
            "swap_body_part": self._w_swap_body.currentText(),
            "rolling_median": self._w_median.isChecked(),
            "smooth": self._w_smooth.isChecked(),
            "euro_min_cutoff": self._w_euro_mc.value(),
            "euro_beta": self._w_euro_b.value(),
            "min_move_cm": self._w_minmove.value() * 100.0,
            "max_speed_cm_s": self._w_maxspeed.value() * 100.0,
            # output column curation (select / rename)
            "columns": self._opts.get("columns", {"select": [], "rename": {}}),
            # which zones to report ([] = all)
            "zones_include": sorted(getattr(self, "_zones_include", set())),
            # sub-zones reported as one ({} = none)
            "regions": {k: list(v) for k, v in (self._regions or {}).items()},
            # Y-maze roles, `_ymaze` produces nothing without them.
            **self._arm_roles(),
            # one row per protocol stage, or one for the whole session
            "split_stages": self._w_split.isChecked(),
            # Where the run's readable copies go ("" = only the cache).
            "output_dir": self._opts.get("output_dir", ""),
            "write_video": self._chips["video"].isChecked(),
            "trim_start_s": (float(self._w_trim_from.value())
                             if self._w_trim.isChecked() else 0.0),
            "trim_end_s": (float(self._w_trim_to.value())
                           if self._w_trim.isChecked() else 0.0),
            "heatmap_bins": int(self._w_heatmap_bins.value()),
        }

    # ── Analysis profiles (save / load the whole config as JSON) ──────────
    def _config_dict(self) -> dict:
        """The full analysis config as a plain dict, one JSON per project."""
        return {
            "measures": {k: c.isChecked() for k, c in self._chips.items()},
            "plots": {k: c.isChecked() for k, c in self._w_plots.items()},
            "zone_transitions": self._w_adv_transitions.isChecked(),
            "zone_advanced": self._w_adv_zonedist.isChecked(),
            "px_per_cm": self._w_ppc.value(),
            "body_parts": sorted(self._bp_include),
            "min_move_cm": self._w_minmove.value(),
            "max_speed_cm_s": self._w_maxspeed.value(),
            "swap_correct": self._w_swap.isChecked(),
            "smooth": self._w_smooth.isChecked(),
            "immobility_cm_s": self._w_freezethr.value(),
            "freeze_cm_s": self._w_freezestrict.value(),
            "min_bout_ms": self._w_freezemin.value(),
            "min_dwell_s": self._w_dwell.value(),
            "bin_mode": self._w_bin_mode.currentData(),
            "bin_every_s": self._w_bin_every.value(),
            "bin_count_n": self._w_bin_count.value(),
            "bin_edges_text": self._w_bin_edges.text(),
            "zones_include": sorted(getattr(self, "_zones_include", set())),
            "regions": {k: list(v) for k, v in (self._regions or {}).items()},
            "columns": self._opts.get("columns", {"select": [], "rename": {}}),
        }

    def _apply_config(self, d: dict):
        """Apply a loaded profile dict back onto the widgets."""
        # Regions are part of the analysis, so a profile that defines them and
        # does not restore them would silently change the columns.
        self._regions = {str(k): list(v)
                         for k, v in (d.get("regions") or {}).items() if v}
        self._refresh_region_label()
        for k, v in (d.get("measures") or {}).items():
            if k in self._chips:
                self._chips[k].setChecked(bool(v))
        for k, v in (d.get("plots") or {}).items():
            if k in self._w_plots:
                self._w_plots[k].setChecked(bool(v))
        self._w_adv_transitions.setChecked(bool(d.get("zone_transitions", True)))
        self._w_adv_zonedist.setChecked(bool(d.get("zone_advanced", False)))
        for key, w in [("px_per_cm", self._w_ppc), ("min_move_cm", self._w_minmove),
                       ("max_speed_cm_s", self._w_maxspeed), ("immobility_cm_s", self._w_freezethr),
                       ("freeze_cm_s", self._w_freezestrict), ("min_dwell_s", self._w_dwell)]:
            if key in d:
                try: w.setValue(float(d[key]))
                except Exception: pass
        if "min_bout_ms" in d:
            self._w_freezemin.setValue(int(d["min_bout_ms"]))
        if "bin_every_s" in d:
            self._w_bin_every.setValue(int(d["bin_every_s"]))
        if "bin_count_n" in d:
            self._w_bin_count.setValue(int(d["bin_count_n"]))
        if "bin_edges_text" in d:
            self._w_bin_edges.setText(str(d["bin_edges_text"]))
        if "bin_mode" in d:
            i = self._w_bin_mode.findData(d["bin_mode"])
            if i >= 0:
                self._w_bin_mode.setCurrentIndex(i)
        elif "bin_size_s" in d:                 # profiles written before modes
            self._w_bin_every.setValue(max(1, int(d["bin_size_s"] or 60)))
        if "smooth" in d:
            self._w_smooth.setChecked(bool(d["smooth"]))
        if "swap_correct" in d:
            self._w_swap.setChecked(bool(d["swap_correct"]))
        if d.get("body_part"):
            pass
        if isinstance(d.get("body_parts"), list) and d["body_parts"]:
            self._bp_include = set(d["body_parts"])
            if hasattr(self, "_bp_menu"):
                self._refresh_bodypart_menu()
        if isinstance(d.get("columns"), dict):
            self._opts["columns"] = d["columns"]
        if isinstance(d.get("zones_include"), list):
            self._zones_include = set(d["zones_include"])
            if hasattr(self, "_zone_menu"):
                self._refresh_zone_menu()

    def _save_profile(self):
        from tools.offline_analysis.analyze.paths import dialog_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Save analysis profile",
            os.path.join(dialog_dir("analysis"), "analysis_profile.json"), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._config_dict(), f, indent=2)
            self._status.setText(f"Saved profile: {os.path.basename(path)}")
            self._log.append(f"Saved analysis profile → {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Profile Error", str(e))

    def _load_profile(self):
        from tools.offline_analysis.analyze.paths import dialog_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Load analysis profile", dialog_dir("analysis"), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._apply_config(json.load(f))
            self._status.setText(f"Loaded profile: {os.path.basename(path)}")
            self._log.append(f"Loaded analysis profile ← {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Profile Error", str(e))

    def _run_analysis(self, dry_run: bool = False):
        """Do what the Plan column says, for every ticked recording.

        There is no mode to dispatch on. Each recording's plan lists the stages
        it needs, read, track, re-zone, measure, and the worker runs exactly
        that list. A recording whose plan is blocked is reported with the
        reason and skipped; it never fails silently or half-way.
        """
        if self._worker is not None:
            return
        sessions = self._selected_sessions()
        if not sessions:
            QtWidgets.QMessageBox.information(
                self, "No files selected",
                "Tick at least one recording in the Use column.")
            return

        from tools.offline_analysis.engine import session_bundle as _sb

        bundles = self._selected_bundles()
        plan = _sb.summarise(bundles)
        if plan["blocked"] and not plan["steps"]:
            QtWidgets.QMessageBox.warning(
                self, "Nothing can run",
                "Every selected recording is blocked:\n\n• "
                + "\n• ".join(f"{stem}: {why}" for stem, why in plan["blocked"][:8]))
            return
        if plan["blocked"]:
            self._log.append(
                f"{len(plan['blocked'])} recording(s) blocked and will be skipped:")
            for stem, why in plan["blocked"]:
                self._log.append(f"  • {stem}: {why}")

        metrics = self._collect_metrics()
        params = self._collect_params()

        self._btn_run.setEnabled(False)
        self._btn_cancel.setVisible(True)
        self._status.setText("Running...")
        self._log.clear()
        self._log.append(("DRY RUN, " if dry_run else "")
                         + f"{len(bundles)} recording(s): "
                         + " · ".join(f"{ln['n']}× {ln['stage']}"
                                      for ln in plan["steps"]))
        self._log.append(f"Metrics: {[k for k, v in metrics.items() if v]}")

        self._worker = PipelineWorker(
            bundles, metrics, params, metadata_df=self._metadata_df,
            metadata_id_col=self._metadata_id_col, dry_run=dry_run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _refresh_verify(self):
        """Point the Verify tab at the poses the results were computed from,
        the retracked file when there is one, otherwise the recording's own."""
        if not hasattr(self, "_verify"):
            return
        entries = []
        for sf in self._selected_sessions():
            try:
                b = self._bundle_for(sf)
                txt = b.active_txt()
                if txt == b.txt_path:
                    # The stage lost its artifact, a session.json written by
                    # a run that was killed, or one copied without it. The
                    # poses are still on disk; verifying the recording's own
                    # empty pose column instead is how you end up staring at a
                    # video with no overlay and no idea why.
                    found = _newest_pose_file(b)
                    if found:
                        txt = found
                tag = " (retracked)" if txt != b.txt_path else ""
                entries.append((f"{b.stem}{tag}", txt, b.video_path))
            except Exception as e:
                logger.debug("verify entry for %s: %s", sf.txt_path, e)
        self._verify.set_recordings(entries)

    def _cancel_run(self):
        """Stop after the current frame; whatever was produced is kept."""
        w = self._worker
        if w is not None and hasattr(w, "cancel"):
            w.cancel()
            self._status.setText("Stopping…")
            self._btn_cancel.setEnabled(False)

    def _on_progress(self, msg: str):
        self._status.setText(msg)
        self._log.append(msg)

    def _on_finished(self, results: dict):
        """Handle analysis completion, populate Results and Plots tabs."""
        # The worker emits `finished` from INSIDE run(), so this slot executes
        # while the thread is still on the stack. Simply reassigning
        # `self._worker = None` drops the last Python reference, CPython
        # destroys a still-running QThread, and Qt aborts the process with
        # "QThread: Destroyed while thread is still running", an intermittent,
        # traceback-free exit that only shows up under real runs.
        worker = self._worker
        self._worker = None
        if worker is not None:
            if not worker.wait(5000):     # run() is one return away; instant
                logger.warning("analysis worker did not stop within 5 s")
            worker.deleteLater()          # let Qt free it on the event loop

        self._btn_run.setEnabled(True)
        self._btn_cancel.setVisible(False)
        self._btn_cancel.setEnabled(True)
        self._excluded = results.get("excluded", [])

        if results.get("errors"):
            for err in results["errors"]:
                self._log.append(f"ERROR: {err}")
        # Warnings are the honesty channel: an implausible clock, a row/frame
        # mismatch, an uncalibrated session. They must reach the user next to
        # the numbers they qualify, not only in a tooltip.
        for warn in results.get("warnings", []):
            self._log.append(f"! {warn}")
        if self._excluded:
            self._log.append(
                f"Skipped {len(self._excluded)} recording(s), reasons above "
                "and in the Excluded sheet on export.")
        if results.get("cancelled"):
            self._log.append("Run cancelled, completed work was kept.")
        self._refresh_file_tree()
        self._refresh_verify()

        # Single flat results table (like AnyMaze export)
        self._current_transitions = results.get("transitions") or []
        summary_df = results.get("summary_df")
        if summary_df is not None and not summary_df.empty:
            self._current_summary_df = summary_df
            self._populate_table(self._summary_table, summary_df)
            self._btn_excel.setEnabled(True)
            self._btn_csv.setEnabled(True)
            # Only when there is actually a corrected or retracked stream to
            # save, offered against a plain Analyse run it would be a button
            # whose only answer is "nothing to save".
            self._btn_poses.setEnabled(self._have_derived_poses())
            # The same rule the column preview promises with, so "you will get
            # 30 measure columns" and what the run reports are the same count.
            from tools.offline_analysis.engine.offline_analysis import measure_columns
            n_cols = len(summary_df.columns)
            n_meas = len(measure_columns(list(summary_df.columns)))
            n_subj = summary_df["Subject"].nunique() if "Subject" in summary_df else 0
            self._export_row.setVisible(True)
            self._export_hint.setText(
                f"{len(summary_df)} rows \u00d7 {n_meas} measures"
                + (f" \u00b7 {n_subj} animals" if n_subj else ""))
            self._status.setText(
                f"{len(summary_df)} rows \u00d7 {n_meas} measures")
            self._log.append(f"\nResults: {len(summary_df)} rows, {n_cols} columns "
                             f"({n_meas} measures)")
        else:
            self._status.setText("No results")
            self._export_row.setVisible(False)
            self._current_summary_df = None


        # Plots
        figures = results.get("figures", [])
        self._clear_plots()
        if figures:
            self._display_plots(figures)
            self._btn_plots.setEnabled(True)

        self._results_tabs.setCurrentIndex(0)

    # ── Table display ───────────────────────────────────────────

    def _curate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the user's column selection + rename (from the Columns dialog)."""
        from tools.offline_analysis.engine.offline_analysis import apply_column_config
        cfg = self._opts.get("columns") or {}
        if cfg.get("select") or cfg.get("rename"):
            df = apply_column_config(df, cfg)
        return df

    #: Identity columns that only exist when the session metadata supplies
    #: them. An experiment with no Group/Sex/Cage should not be looking at
    #: four empty columns, but the EXPORT keeps them, so a cohort's
    #: workbooks still stack on a common set.
    _OPTIONAL_ID_COLUMNS = ("Group", "Sex", "Cage", "Subgroup")

    def _populate_table(self, table: QTableWidget, df: pd.DataFrame):
        # Drop internal bookkeeping columns then apply the user's column
        # curation (select / rename) so the table matches the export exactly.
        drop_cols = {c for c in df.columns
                     if str(c).startswith("__") or c in ("Frames", "Video", "px_per_cm")}
        # Hide identity columns this experiment never filled in, on screen
        # only. Nothing is lost: they are still written on export.
        for col in self._OPTIONAL_ID_COLUMNS:
            if col in df.columns and not df[col].astype(str).str.strip().any():
                drop_cols.add(col)
        if drop_cols:
            df = df.drop(columns=list(drop_cols), errors="ignore")
        df = self._curate(df)

        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                text = "" if pd.isna(val) else str(val)
                item = QTableWidgetItem(text)
                # Right-align numbers
                if isinstance(val, (int, float)) and not pd.isna(val):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(i, j, item)

        table.resizeColumnsToContents()

    # ── Plot display ────────────────────────────────────────────

    def _clear_plots(self):
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # The figures are the panel's own (see `offline_analysis.new_figure`):
        # nothing global holds them, so dropping the list is all that frees
        # them. Detach each canvas from its figure first, the widgets are
        # only queued for deletion here, and a canvas that outlives the list
        # is the one thing that could still draw from a figure being freed.
        for fig in self._current_figures:
            canvas = getattr(fig, "canvas", None)
            if hasattr(canvas, "setParent"):              # the Qt one, if wrapped
                canvas.setParent(None)
        self._current_figures = []

    def _display_plots(self, figures: list):
        self._clear_plots()
        self._current_figures = figures

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

            for i, fig in enumerate(figures):
                fig.set_size_inches(10, 5)
                fig.tight_layout()

                label = QLabel(f"Plot {i + 1} of {len(figures)}")
                label.setObjectName("section")
                self._plot_layout.addWidget(label)

                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(360)
                toolbar = NavToolbar(canvas, self)
                self._plot_layout.addWidget(toolbar)
                self._plot_layout.addWidget(canvas)
                canvas.draw()

            self._plot_layout.addStretch()
            self._results_tabs.setCurrentWidget(self._plot_widget)

        except ImportError:
            self._log.append("matplotlib Qt backend not available for interactive plots")

    # ── Export ───────────────────────────────────────────────────

    def _export_df(self) -> Optional[pd.DataFrame]:
        """Return the summary DataFrame stripped of internal/junk columns."""
        if self._current_summary_df is None:
            return None
        df = self._current_summary_df
        drop_cols = [c for c in df.columns
                     if str(c).startswith("__") or c in ("Frames", "Video", "px_per_cm")]
        df = df.drop(columns=drop_cols, errors="ignore")
        return self._curate(df)

    def _export_excel(self):
        if self._current_summary_df is None:
            return
        from tools.offline_analysis.analyze.paths import dialog_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel", os.path.join(dialog_dir("analysis"), "analysis.xlsx"),
            "Excel (*.xlsx)")
        if not path:
            return
        try:
            # Build the engine result structure so the export goes through the
            # SAME writer as the CLI: Meta sheet first (all params used),
            # Session_Summary (curated), Excluded, + a _settings.json sidecar.
            from tools.offline_analysis.engine import offline_analysis as oa
            raw = self._current_summary_df
            records = [{k: v for k, v in r.items()
                        if not str(k).startswith("__") and k != "px_per_cm"}
                       for r in raw.to_dict("records")]
            ep = AnalysisWorker(self._visible_sessions(), self._collect_metrics(),
                                self._collect_params())._engine_params()
            res = {"summary": records, "bins": [], "excluded": self._excluded,
                   "transitions": self._current_transitions,
                   "params": ep, "n_files": len(records), "n_jobs": 1}
            oa.write_workbook(res, path)
            sheets = ["Meta", "Session_Summary"]
            if len(records) > 1:
                sheets.append("Group_Summary")
            if self._current_transitions:
                sheets.append("Transitions")
            if self._excluded:
                sheets.append("Excluded")
            self._log.append(f"Exported to {path} ({' + '.join(sheets)})")
            self._status.setText(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(e))

    def _export_csv(self):
        df = self._export_df()
        if df is None:
            return
        from tools.offline_analysis.analyze.paths import dialog_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", os.path.join(dialog_dir("analysis"), "analysis.csv"),
            "CSV (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            self._log.append(f"Exported to {path}")
            self._status.setText(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(e))

    def _have_derived_poses(self) -> bool:
        """Whether any selected recording has poses it did not arrive with."""
        for sf in self._selected_sessions():
            try:
                b = self._bundle_for(sf)
            except Exception:
                continue
            src = b.active_txt()
            if src and src != b.txt_path and os.path.exists(src):
                return True
        return False

    def _export_poses(self):
        """Copy the corrected / retracked pose files out of the bundle.

        A Correct or Re-track run writes a real session file, same format,
        same header, a record of what produced it, but it writes it
        inside the recording's `.pbanalysis` folder, which is a cache the
        user has no reason to browse and every reason to delete. So the one
        durable product of correcting or retracking was invisible: you could
        analyse with it, and not have it.

        This puts it beside the recording, under its own name, where it is a
        file like any other: openable in Verify, loadable here, readable by
        anything else that speaks the format.
        """
        pairs = []
        for sf in self._selected_sessions():
            try:
                b = self._bundle_for(sf)
            except Exception:
                continue
            src = b.active_txt()
            if src and src != b.txt_path and os.path.exists(src):
                pairs.append((b, src))
        if not pairs:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save",
                "None of the selected recordings has corrected or retracked "
                "poses yet.\n\nChoose Correct or Re-track in WHAT TO DO and "
                "run, then save them from here.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Save pose files to",
            os.path.dirname(pairs[0][0].txt_path or "") or "")
        if not folder:
            return

        import shutil

        written, failed = [], []
        for b, src in pairs:
            name = os.path.basename(src)
            # The bundle names its outputs by a hash key; out here the file
            # has to say which recording it belongs to.
            if b.stem and not name.startswith(b.stem):
                tag = "corrected" if "corrected" in name else "retracked"
                name = f"{b.stem}_{tag}_video_data.txt"
            dest = os.path.join(folder, name)
            try:
                if os.path.abspath(dest) != os.path.abspath(src):
                    shutil.copy2(src, dest)
                written.append(os.path.basename(dest))
            except Exception as e:
                failed.append(f"{b.stem}: {e}")

        for w in written:
            self._log.append(f"Saved poses: {w}")
        self._status.setText(f"Saved {len(written)} pose file(s)")
        msg = f"{len(written)} file(s) written to{chr(10)}{folder}"
        if failed:
            msg += f"{chr(10)}{chr(10)}Could not write:{chr(10)}" + chr(10).join(failed)
        QtWidgets.QMessageBox.information(self, "Saved", msg)

    def _choose_output_dir(self):
        """Pick where this run's readable copies go, or clear it."""
        # Imported here like every other caller in this file. Without it the
        # button raised NameError the moment it was clicked, the one call site
        # of six that had no import.
        from tools.offline_analysis.analyze.paths import dialog_dir

        start = self._opts.get("output_dir") or dialog_dir("analysis")
        folder = QFileDialog.getExistingDirectory(
            self, "Save results to", start)
        if not folder:
            return
        self._opts["output_dir"] = folder
        self._sync_output_dir()
        self._log.append(f"Results will also be written to {folder}")

    def _sync_output_dir(self):
        """Show the chosen folder on the button, or offer to choose one."""
        folder = self._opts.get("output_dir") or ""
        if folder:
            self._btn_outdir.setText("Results → " + os.path.basename(
                folder.rstrip("\\/")) or folder)
            self._btn_outdir.setToolTip(
                "Every recording's retracked poses and figures are copied "
                f"here, named after the recording:{chr(10)}{folder}"
                f"{chr(10)}{chr(10)}Click to change.")
        else:
            self._btn_outdir.setText("Save results to…")
            self._btn_outdir.setToolTip(
                "Choose a folder for the results.\n\nWithout one, a run's "
                "output stays in the hidden .pbanalysis cache beside each "
                "recording, under a hash for a name.")

    def _save_plots(self):
        if not self._current_figures:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save Plots To")
        if not folder:
            return
        try:
            for i, fig in enumerate(self._current_figures):
                fig.savefig(os.path.join(folder, f"{_plot_filename(fig, i)}.png"),
                            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            self._log.append(f"Saved {len(self._current_figures)} plots to {folder}")
            self._status.setText(f"Saved {len(self._current_figures)} plots")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", str(e))

    # ── Clear ───────────────────────────────────────────────────

    def _clear_all(self):
        self._current_transitions = []
        self._sessions.clear()
        self._excluded_paths.clear()
        self._bundles.clear()
        self._file_table.setRowCount(0)
        self._file_count.setText("No files loaded")
        self._summary_table.clear()
        self._summary_table.setRowCount(0)
        self._summary_table.setColumnCount(0)
        self._clear_plots()
        self._log.clear()
        self._status.setText("")
        self._excluded = []
        self._current_summary_df = None
        self._btn_run.setEnabled(False)
        self._btn_excel.setEnabled(False)
        self._btn_csv.setEnabled(False)
        self._btn_plots.setEnabled(False)
        self._export_row.setVisible(False)
        self._plan_label.setText("Load recordings to begin.")
