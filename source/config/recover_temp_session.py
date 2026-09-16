"""recover_temp_session.py - relocate dry-run / temp recordings into the
permanent data tree, exactly as a real recording would have saved them.

A "DRY Start" (Record clicked with an empty subject_id) writes a flat,
overwritten set of files into the GLOBAL temp dir::

    <code>/data/temp/Box<N>.tsv
    <code>/data/temp/video_Box<N>.{mp4,avi}
    <code>/data/temp/video_data_Box<N>.txt

A real record instead fans out per project/task/date with subject-stamped
names. This tool bridges the two: it recovers the original record-click
timestamp and the task family EMBEDDED in those files, asks for the project
+ cohort the SAME WAY the main app does (file-pick dialogs, not typing), and
copies each session file to::

    <data_root>/[<project>/][<task>/]<YYYY-MM-DD>/mcu/<subject>-Box<N>-<ts>.tsv
    <data_root>/[<project>/][<task>/]<YYYY-MM-DD>/video/<subject>-Box<N>-<ts>.<ext>
    <data_root>/[<project>/][<task>/]<YYYY-MM-DD>/video/<subject>-Box<N>-<ts>_video_data.txt

…then writes the same runs-index row + config snapshot a regular save does.

By default it opens a small single-window GUI: a table of the TSV-anchored
sessions (tick the ones to recover, give each a Subject ID by Load Project /
Load Metadata or by typing), an Export button that enables once IDs are
present and warns before writing outside a project. Pass ``--cli`` to drive
it from the terminal instead.

Design contract: this script IMPORTS standard builders from the pipeline
(``build_session_stem``, ``open_run``/``close_run``, ``snapshot_config_to_source``,
``load_experiment``, ``MetadataManager``); the pipeline never imports it.

Run it::

    recover_temp_session.bat                         (double-click - GUI)
    python -m source.config.recover_temp_session     (GUI)
    python -m source.config.recover_temp_session --cli
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Make ``source.*`` importable whether run as `-m` from the repo root or as a
# bare file path (parents[2] of source/config/<this>.py is the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import source.paths as paths                                  # noqa: E402
from source.config.experiment import (                        # noqa: E402
    load_experiment, _default_data_dir_for, snapshot_config_to_source,
)
from source.datetime_formats import format_date_dir           # noqa: E402
from source.video.recording import build_session_stem         # noqa: E402
from source.gui.project_workflow import open_run, close_run   # noqa: E402

_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
_HEADER_TS_FMT = "%Y-%m-%d %H:%M:%S"
_CFG_NAME = "experiment_config.json"


# ─────────────────────────────────────────────────────────── temp scanning
def scan_temp(temp_dir: Path) -> dict[int, dict]:
    """Find sessions anchored on the MCU TSV. Returns {N: {tsv[, txt]}}.

    The TSV is the anchor, a box without a Box<N>.tsv is not a session.
    Video is read from the TSV header (see _gather), not globbed. The
    frame-log sibling video_data_Box<N>.txt is attached if present."""
    found: dict[int, dict] = {}
    if not temp_dir.is_dir():
        return found
    for p in sorted(temp_dir.iterdir()):
        if not p.is_file():
            continue
        m = re.fullmatch(r"Box(\d+)\.tsv", p.name)
        if m:
            found.setdefault(int(m.group(1)), {})["tsv"] = p
            continue
        m = re.fullmatch(r"video_data_Box(\d+)\.txt", p.name)
        if m:
            found.setdefault(int(m.group(1)), {})["txt"] = p
    return found


def parse_header(tsv_path: Path) -> dict:
    """Read every session fact from the TSV header: start_time, task_name,
    subject_id, completion (an ``end_time`` line), an approx data-row count,
    and the video declaration (``video_recorded`` / ``video_file``, or the
    ``video`` JSON block). Tab-separated ``time<TAB>type<TAB>subtype<TAB>content``
    with a float time column. Best-effort, never raises."""
    info: dict = {"start_time": None, "task_name": "", "subject_id": "",
                  "rows": 0, "completed": False,
                  "video_recorded": False, "video_file": ""}
    try:
        with open(tsv_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                rtype = parts[1] if len(parts) > 1 else ""
                subtype = parts[2] if len(parts) > 2 else ""
                content = parts[-1] if len(parts) > 1 else ""
                if rtype == "info":
                    if subtype == "start_time" and info["start_time"] is None:
                        m = _TS_RE.search(content)
                        if m:
                            info["start_time"] = datetime.strptime(
                                m.group(0), _HEADER_TS_FMT)
                    elif subtype == "task_name" and not info["task_name"]:
                        info["task_name"] = content
                    elif subtype == "subject_id" and not info["subject_id"]:
                        info["subject_id"] = content
                    elif subtype == "end_time":
                        info["completed"] = True
                    elif subtype == "video_recorded":
                        info["video_recorded"] = content.strip().lower() == "true"
                    elif subtype == "video_file" and not info["video_file"]:
                        info["video_file"] = content.strip()
                    elif subtype == "video":
                        # JSON block: {"recorded":bool,"video_name":...}
                        try:
                            import json
                            v = json.loads(content)
                            info["video_recorded"] = bool(
                                v.get("recorded", info["video_recorded"]))
                            if not info["video_file"]:
                                info["video_file"] = v.get("video_name") or ""
                        except Exception:
                            pass
                elif rtype and rtype != "type":
                    info["rows"] += 1          # event / state / print / variable
    except Exception:
        pass
    return info


def parse_txt(txt_path: Path) -> dict:
    """Date + task family embedded in a _video_data.txt header:
    ``#session_start <ts>`` and ``#pycontrol {"task":"Family\\Name"}``."""
    out: dict = {"session_start": None, "task_family": ""}
    try:
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                low = line.lower()
                if out["session_start"] is None and "session_start" in low:
                    m = _TS_RE.search(line)
                    if m:
                        out["session_start"] = datetime.strptime(
                            m.group(0), _HEADER_TS_FMT)
                if not out["task_family"] and '"task"' in low:
                    m = re.search(r'"task"\s*:\s*"([^"]+)"', line)
                    if m:
                        segs = [s for s in re.split(r'[\\/]+', m.group(1)) if s]
                        out["task_family"] = segs[-2] if len(segs) >= 2 else ""
    except Exception:
        pass
    return out


def check_session_files(files: dict, header: dict) -> list[str]:
    """'Check the data' - human-readable findings/warnings per box."""
    notes: list[str] = []
    tsv = files.get("tsv")
    if tsv is None:
        notes.append("WARNING: no MCU .tsv for this box")
    else:
        notes.append(f"tsv {tsv.stat().st_size} B, ~{header.get('rows', 0)} rows"
                     + (", completed" if header.get("completed")
                        else ", NO end-marker (may be truncated)"))
        if header.get("start_time") is None:
            notes.append("WARNING: no start_time header - will use file mtime")
    # video, from the TSV's declaration, not the folder.
    if header.get("video_recorded"):
        vf = header.get("video_file") or "?"
        vid = files.get("video")
        if vid is not None:
            sz = vid.stat().st_size
            notes.append(f"video (TSV declares '{vf}') present, {sz} B"
                         + ("" if sz > 0 else "  WARNING: empty!"))
            _probe_video(vid, notes)
        else:
            notes.append(f"WARNING: TSV declares video '{vf}' but it is "
                         f"missing from temp")
    elif header:
        notes.append("TSV: no video recorded for this box")
    txt = files.get("txt")
    if txt is not None:
        sz = txt.stat().st_size
        notes.append(f"video_data.txt {sz} B"
                     + ("" if sz > 0 else "  WARNING: empty!"))
    return notes


def _probe_video(vid: Path, notes: list[str]) -> None:
    try:
        import cv2
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            notes.append("  WARNING: video not openable by OpenCV")
        else:
            notes.append(f"  video opens OK, ~{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
        cap.release()
    except Exception:
        pass


def read_cohort(path: Path) -> dict[int, str]:
    """Read a cohort Excel/CSV into {SetupID: Subject}. Mirrors the GUI's
    contract: required columns are 'Subject' and 'SetupID'."""
    import pandas as pd
    p = str(path)
    df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)
    cols = {str(c).strip(): c for c in df.columns}
    if "Subject" not in cols or "SetupID" not in cols:
        raise ValueError("cohort metadata needs columns 'Subject' and 'SetupID'")
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            sid = int(row[cols["SetupID"]])
        except (ValueError, TypeError):
            continue
        subj = str(row[cols["Subject"]]).strip()
        if subj and subj.lower() != "nan":
            out.setdefault(sid, subj)
    return out


# ─────────────────────────────────────────────────────────── path building
def session_dirs(cfg, task_seg: str, dt: datetime) -> tuple[Path, Path]:
    """Replicate SetupWidget.get_session_dirs() - return (mcu_dir, video_dir).
    <base>/[<task>/]<YYYY-MM-DD>/{mcu,video}, base = project data dir (already
    includes project name) or global <code>/data when no project."""
    date_str = format_date_dir(dt)
    if cfg is not None:
        data_dir = getattr(cfg.meta, "data_dir", "") or ""
        if data_dir:
            base = Path(data_dir)
            if not base.is_absolute():
                base = Path(paths.top_dir) / base
        else:
            base = Path(_default_data_dir_for(cfg.meta.project))
    else:
        base = Path(paths.top_dir) / "data"
    root = base / task_seg / date_str if task_seg else base / date_str
    return root / "mcu", root / "video"


def _dest_for(plan: dict, kind: str):
    stem = plan["stem"]
    if kind == "tsv":
        return plan["mcu_dir"] / f"{stem}.tsv"
    if kind == "video":
        return plan["video_dir"] / f"{stem}{plan['vext']}"
    if kind == "txt":
        return plan["video_dir"] / f"{stem}_video_data.txt"
    return None


# ─────────────────────────────────────────────────────────── core (no I/O UI)
def build_plans(found, headers, txtinfo, cfg, subject_for, task_override=""):
    """Build one move-plan per box. ``subject_for(box) -> str`` is injected so
    the GUI (dialog) and CLI (prompt) reuse identical logic. Boxes whose
    subject_for returns blank are skipped."""
    plans = []
    for n in sorted(found):
        arts = found[n]
        hdr = headers.get(n, {})
        txt = txtinfo.get(n, {})
        # date: TSV start_time -> txt #session_start -> file mtime (last resort).
        dt = hdr.get("start_time") or txt.get("session_start")
        if dt is None:
            anyf = arts.get("tsv") or next(iter(arts.values()))
            dt = datetime.fromtimestamp(anyf.stat().st_mtime)
        task_seg = task_override or txt.get("task_family", "")
        subj = (subject_for(n) or "").strip()
        if not subj:
            continue
        mcu_dir, video_dir = session_dirs(cfg, task_seg, dt)
        plans.append({
            "box": n, "subject": subj, "dt": dt, "task": task_seg,
            "mcu_dir": mcu_dir, "video_dir": video_dir,
            "stem": build_session_stem(subj, n, dt), "src": arts,
            "vext": arts["video"].suffix if arts.get("video") else None,
        })
    return plans


def render_preview(plans, op_name) -> str:
    out = []
    for pl in plans:
        out.append(f"Box{pl['box']}  subject={pl['subject']}  "
                   f"task={pl['task'] or '(none)'}  "
                   f"ts={pl['dt'].strftime(_HEADER_TS_FMT)}")
        for kind, src in pl["src"].items():
            dst = _dest_for(pl, kind)
            if dst is None:
                continue
            ex = "   [EXISTS - will skip]" if dst.exists() else ""
            out.append(f"   {op_name} {src.name}\n      -> {dst}{ex}")
        out.append("")
    return "\n".join(out)


def execute_plans(plans, cfg, project_dir):
    """Copy each session file and (project mode) write the runs row + config
    snapshot, exactly as a real save does. Always copies, the temp
    originals are the safety net until the cleanup step removes them.
    Returns (n_files, log_lines)."""
    xfer = shutil.copy2
    n_ok = 0
    log: list[str] = []
    for pl in plans:
        pl["mcu_dir"].mkdir(parents=True, exist_ok=True)
        pl["video_dir"].mkdir(parents=True, exist_ok=True)
        moved = {}
        for kind, src in pl["src"].items():
            dst = _dest_for(pl, kind)
            if dst is None:
                continue
            if dst.exists():
                log.append(f"skip (exists): {dst.name}")
                continue
            try:
                xfer(str(src), str(dst))
                moved[kind] = dst
                n_ok += 1
            except Exception as e:
                log.append(f"FAILED {src.name}: {e}")
        if cfg is not None and project_dir is not None and "tsv" in moved:
            try:
                run_id = open_run(
                    project_dir, cfg, pl["box"], pl["subject"],
                    datetime_now=pl["dt"], date_iso=pl["dt"].strftime("%Y-%m-%d"))
                if run_id:
                    close_run(project_dir, run_id, files={
                        "mcu_tsv": str(moved.get("tsv") or ""),
                        "video_mp4": str(moved["video"]) if "video" in moved else None,
                    })
                    snapshot_config_to_source(cfg, project_dir)
                    log.append(f"runs row written: {run_id}")
            except Exception as e:
                log.append(f"runs-row write failed for Box{pl['box']}: {e}")
    return n_ok, log


def _clear_temp_for_completed(plans) -> int:
    """Delete the temp source files of every plan whose session files ALL landed at
    their destination, i.e. the session is now safely in the permanent tree.
    A partially-exported session (a file skipped because the dest already
    existed, or a failed copy) keeps its temp files. Returns sessions cleared."""
    cleared = 0
    for pl in plans:
        if not pl:
            continue
        dests = {k: _dest_for(pl, k) for k in pl["src"]}
        if not all(d is not None and d.exists() for d in dests.values()):
            continue                       # not fully present yet -> keep temp
        removed = False
        for kind, src in pl["src"].items():
            try:
                if src.exists() and src.resolve() != dests[kind].resolve():
                    src.unlink()
                    removed = True
            except Exception:
                pass
        cleared += 1 if removed else 0
    return cleared


def _gather(temp_dir: Path):
    """Anchor on the TSV: scan for Box<N>.tsv, parse each header, then attach
    the video the TSV DECLARES (video_recorded + video_file), never globbed.
    Drops boxes with no TSV (not a real session)."""
    found = scan_temp(temp_dir)
    headers, txtinfo = {}, {}
    for n in list(found):
        arts = found[n]
        if not arts.get("tsv"):
            del found[n]                      # no anchor -> not a session
            continue
        hdr = parse_header(arts["tsv"])
        headers[n] = hdr
        txtinfo[n] = parse_txt(arts["txt"]) if arts.get("txt") else {}
        # Video is whatever the TSV says it is, look for that exact file.
        vf = hdr.get("video_file")
        if hdr.get("video_recorded") and vf:
            vp = temp_dir / vf
            if vp.exists():
                arts["video"] = vp
            else:
                hdr["video_missing"] = vf     # declared but not on disk
    return found, headers, txtinfo


# ─────────────────────────────────────────────────────────── GUI (one window)
try:
    from PySide6 import QtCore, QtWidgets
    _HAS_QT = True
except Exception:
    _HAS_QT = False


if _HAS_QT:
    _ROLE_BOX = QtCore.Qt.ItemDataRole.UserRole
    _CHK = QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
    _RO = QtCore.Qt.ItemFlag.ItemIsEnabled
    _EDIT = (QtCore.Qt.ItemFlag.ItemIsEnabled
             | QtCore.Qt.ItemFlag.ItemIsSelectable
             | QtCore.Qt.ItemFlag.ItemIsEditable)

    class RecoverDialog(QtWidgets.QDialog):
        """One window: tick the TSV-anchored sessions, give each a Subject ID
        (Load Project / Load Metadata auto-fills by SetupID, or type it), then
        Export. Export warns before writing outside a project, and never
        overwrites an existing destination."""

        COLS = ("", "Box", "Date / time", "Task", "Files", "Subject ID", "State")

        def __init__(self, temp_dir: Path):
            super().__init__()
            from source.gui.metadata_manager import MetadataManager
            self.temp_dir = temp_dir
            self.cfg = None
            self.project_dir = None
            self.mgr = MetadataManager()
            self.setWindowTitle("Recover temp recordings")
            self.resize(940, 460)
            self._build()
            self.rescan()

        # High-contrast QSS so the cell editor, typed text, and buttons stay
        # readable on the dark theme.
        _QSS = """
        QDialog { background: #0f1419; }
        QLabel { color: #e6edf3; }
        QCheckBox { color: #cdd9e5; }
        QToolTip { color:#e6edf3; background:#1c2430; border:1px solid #2a323d; }
        QTableWidget {
            background:#11161d; color:#e6edf3; gridline-color:#2a323d;
            alternate-background-color:#171e27;
            border:1px solid #2a323d; border-radius:8px;
            selection-background-color:#22425f; selection-color:#ffffff;
        }
        QTableWidget::item { padding:4px 8px; color:#e6edf3; }
        QTableWidget::item:selected { background:#22425f; color:#ffffff; }
        QHeaderView::section {
            background:#1c2430; color:#cdd9e5; padding:6px 8px;
            border:none; border-bottom:1px solid #2a323d; font-weight:600;
        }
        QTableWidget QLineEdit {
            color:#ffffff; background:#0b0f14;
            border:1px solid #4cc2ff; border-radius:3px; padding:1px 4px;
            selection-background-color:#2d6da3; selection-color:#ffffff;
        }
        QPushButton {
            color:#e6edf3; background:#222c38; border:1px solid #3a4654;
            border-radius:7px; padding:0 14px; min-height:30px; font-weight:600;
        }
        QPushButton:hover { background:#2a3744; border-color:#4a5a6a; }
        QPushButton:disabled { color:#6e7681; background:#1a2129; border-color:#2a323d; }
        """

        def _build(self):
            self.setStyleSheet(self._QSS)
            try:
                from source.gui.style_builders import button_style
            except Exception:
                button_style = lambda *a, **k: ""   # noqa: E731 (base QSS covers it)
            hand = QtCore.Qt.CursorShape.PointingHandCursor

            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(20, 18, 20, 18)
            v.setSpacing(12)

            # ---- header ----
            title = QtWidgets.QLabel("Recover temp recordings")
            title.setStyleSheet("font-size: 17pt; font-weight: 700; color: #e6edf3;")
            v.addWidget(title)
            sub = QtWidgets.QLabel(
                "Tick the sessions to recover, give each a Subject ID "
                "(Load Project / Load Metadata to auto-fill, or type it), then Export.\n"
                f"temp:  {self.temp_dir}")
            sub.setStyleSheet("color: #9aa7b4; font-size: 11px;")
            v.addWidget(sub)

            # ---- toolbar ----
            top = QtWidgets.QHBoxLayout()
            top.setSpacing(8)
            self.proj_lbl = QtWidgets.QLabel("Project:  (none)")
            self.proj_lbl.setStyleSheet("color: #cdd9e5; font-weight: 600;")
            top.addWidget(self.proj_lbl)
            top.addStretch(1)
            for text, slot, variant in (
                    ("Load Project...", self._on_project, "info"),
                    ("Load Metadata...", self._on_metadata, "info"),
                    ("Rescan", self.rescan, "secondary")):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(slot)
                b.setStyleSheet(button_style(variant, height=30))
                b.setCursor(hand)
                top.addWidget(b)
            v.addLayout(top)

            # ---- table ----
            self.table = QtWidgets.QTableWidget(0, len(self.COLS))
            self.table.setHorizontalHeaderLabels(self.COLS)
            self.table.verticalHeader().setVisible(False)
            self.table.verticalHeader().setDefaultSectionSize(30)
            self.table.setShowGrid(False)
            self.table.setAlternatingRowColors(True)
            self.table.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            self.table.setCornerButtonEnabled(False)
            hh = self.table.horizontalHeader()
            hh.setHighlightSections(False)
            RM = QtWidgets.QHeaderView.ResizeMode
            for c in range(len(self.COLS)):
                hh.setSectionResizeMode(
                    c, RM.Stretch if c == 5 else RM.ResizeToContents)
            self.table.itemChanged.connect(self._on_changed)
            v.addWidget(self.table)

            # ---- footer ----
            bot = QtWidgets.QHBoxLayout()
            bot.setSpacing(8)
            hint = QtWidgets.QLabel(
                "Export copies into the data tree, then clears those sessions "
                "from temp.")
            hint.setStyleSheet("color: #9aa7b4; font-size: 11px;")
            bot.addWidget(hint)
            bot.addStretch(1)
            self.export_btn = QtWidgets.QPushButton("Export")
            self.export_btn.clicked.connect(self._on_export)
            self.export_btn.setStyleSheet(button_style("success", height=34))
            self.export_btn.setCursor(hand)
            self.export_btn.setMinimumWidth(170)
            close = QtWidgets.QPushButton("Close")
            close.clicked.connect(self.reject)
            close.setStyleSheet(button_style("secondary", height=34))
            close.setCursor(hand)
            bot.addWidget(self.export_btn)
            bot.addWidget(close)
            v.addLayout(bot)

        # ---- data / table ----
        def rescan(self):
            self.found, self.headers, self.txtinfo = _gather(self.temp_dir)
            self._fill_table()
            self._autofill()
            self._refresh_export()

        def _row_dt(self, n):
            hdr = self.headers[n]
            dt = hdr.get("start_time") or self.txtinfo.get(n, {}).get("session_start")
            if dt is None:
                a = self.found[n].get("tsv") or next(iter(self.found[n].values()))
                dt = datetime.fromtimestamp(a.stat().st_mtime)
            return dt

        def _fill_table(self):
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for n in sorted(self.found):
                hdr = self.headers[n]
                files = ", ".join(k for k in ("tsv", "video", "txt")
                                  if k in self.found[n])
                notes = check_session_files(self.found[n], hdr)
                if any("WARNING" in x for x in notes):
                    short = "check"
                elif not hdr.get("completed"):
                    short = "truncated"
                else:
                    short = "ok"
                vals = [str(n), self._row_dt(n).strftime(_HEADER_TS_FMT),
                        self.txtinfo.get(n, {}).get("task_family", "") or "(none)",
                        files]
                r = self.table.rowCount()
                self.table.insertRow(r)
                chk = QtWidgets.QTableWidgetItem()
                chk.setFlags(_CHK)
                chk.setCheckState(QtCore.Qt.CheckState.Checked)
                chk.setData(_ROLE_BOX, n)
                self.table.setItem(r, 0, chk)
                for c, val in enumerate(vals, start=1):
                    it = QtWidgets.QTableWidgetItem(val)
                    it.setFlags(_RO)
                    self.table.setItem(r, c, it)
                subj = QtWidgets.QTableWidgetItem("")
                subj.setFlags(_EDIT)
                self.table.setItem(r, 5, subj)
                st = QtWidgets.QTableWidgetItem(short)
                st.setFlags(_RO)
                st.setToolTip("\n".join(notes))
                self.table.setItem(r, 6, st)
            self.table.blockSignals(False)
            self.table.resizeColumnsToContents()

        def _autofill(self):
            if getattr(self.mgr, "metadata_df", None) is None:
                return
            self.table.blockSignals(True)
            for r in range(self.table.rowCount()):
                n = self.table.item(r, 0).data(_ROLE_BOX)
                cell = self.table.item(r, 5)
                if cell is not None and not cell.text().strip():
                    row = self.mgr.row_for_setup(n)
                    if row:
                        cell.setText(str(row.get("Subject") or "").strip())
            self.table.blockSignals(False)

        def _selected(self):
            out = []
            for r in range(self.table.rowCount()):
                chk = self.table.item(r, 0)
                if chk.checkState() == QtCore.Qt.CheckState.Checked:
                    out.append((chk.data(_ROLE_BOX),
                                self.table.item(r, 5).text().strip()))
            return out

        # ---- slots ----
        def _on_project(self):
            start = str(Path(paths.experiments_dir) / "projects")
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select project config", os.path.join(start, _CFG_NAME),
                f"Project ({_CFG_NAME});;JSON Files (*.json);;All Files (*)")
            if not chosen:
                return
            try:
                self.project_dir = Path(chosen).resolve().parent
                self.cfg = load_experiment(self.project_dir)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
                self.cfg = None
                self.project_dir = None
                return
            self.proj_lbl.setText(f"Project: {self.cfg.meta.project}")
            self.mgr.ensure_loaded(self.cfg, parent=self, project_dir=self.project_dir)
            self._autofill()
            self._refresh_export()

        def _on_metadata(self):
            if self.mgr.load_metadata(parent=self):
                self._autofill()
                self._refresh_export()

        def _on_changed(self, item):
            if item.column() in (0, 5):
                self._refresh_export()

        def _refresh_export(self, *_):
            valid = [t for t in self._selected() if t[1]]
            self.export_btn.setEnabled(bool(valid))
            self.export_btn.setText(
                "Export" + (f"  -  {len(valid)}" if valid else ""))

        def _on_export(self):
            sel = [t for t in self._selected() if t[1]]
            if not sel:
                return
            if self.cfg is None:
                if QtWidgets.QMessageBox.warning(
                        self, "Outside a project",
                        "These recordings will be exported OUTSIDE any project "
                        "(global data/ tree, no runs index).\n\nExport anyway?",
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No,
                        QtWidgets.QMessageBox.StandardButton.No
                ) != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            subj = dict(sel)
            f = {n: self.found[n] for n in subj}
            h = {n: self.headers[n] for n in subj}
            t = {n: self.txtinfo.get(n, {}) for n in subj}
            plans = build_plans(f, h, t, self.cfg, lambda b: subj.get(b, ""))
            # Copy into the permanent tree first (safe), then clear the temp
            # originals of any session whose every file verifiably landed.
            n_ok, log = execute_plans(plans, self.cfg, self.project_dir)
            cleared = _clear_temp_for_completed(plans)
            QtWidgets.QMessageBox.information(
                self, "Done",
                f"Exported {n_ok} file(s); cleared {cleared} temp session(s).\n\n"
                + "\n".join(log))
            self.rescan()


def run_gui(temp_dir: Path) -> int:
    if not _HAS_QT:
        print("PySide6 not available; falling back to --cli.")
        return run_cli(temp_dir)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    try:                                  # match the main app's dark theme
        from source.gui.styles import ThemeManager
        ThemeManager.apply_theme(app)
    except Exception:
        pass
    RecoverDialog(temp_dir).exec()
    return 0


# ─────────────────────────────────────────────────────────── CLI driver
def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def run_cli(temp_dir: Path) -> int:
    print(f"\n=== Recover temp recordings ===\ntemp dir: {temp_dir}\n")
    found, headers, txtinfo = _gather(temp_dir)
    if not found:
        print("No temp recordings found. Nothing to do.")
        return 0
    for n in sorted(found):
        have = ",".join(k for k in ("tsv", "video", "txt") if k in found[n])
        print(f"  Box{n}: [{have}]")
        for note in check_session_files(found[n], headers.get(n, {})):
            print(f"       - {note}")

    from source.config.experiment import list_projects
    projects = list_projects(paths.experiments_dir)
    print("\nChoose project:")
    for i, p in enumerate(projects, 1):
        print(f"  {i}. {p.name}")
    print(f"  {len(projects)+1}. [no project]")
    sel = _ask("project #")
    cfg = None
    project_dir = None
    cohort: dict[int, str] = {}
    if sel.isdigit() and 1 <= int(sel) <= len(projects):
        project_dir = projects[int(sel) - 1]
        cfg = load_experiment(project_dir)
        mfile = getattr(cfg.meta, "metadata_file", "") or ""
        mpath = project_dir / "metadata" / mfile if mfile else None
        if mpath and mpath.exists():
            try:
                cohort = read_cohort(mpath)
            except Exception as e:
                print(f"  ! cohort: {e}")

    def subject_for(box: int) -> str:
        return _ask(f"Box{box} mouse/subject ID", default=cohort.get(box, ""))

    plans = build_plans(found, headers, txtinfo, cfg, subject_for)
    if not plans:
        print("Nothing to recover.")
        return 0
    print("\n=== Preview ===\n" + render_preview(plans, "Export"))
    if _ask("Proceed to export? (y/N)").lower()[:1] != "y":
        print("Aborted - nothing changed.")
        return 0
    n_ok, log = execute_plans(plans, cfg, project_dir)
    for line in log:
        print("  " + line)
    cleared = _clear_temp_for_completed(plans)
    print(f"\nDone. Exported {n_ok} file(s); cleared {cleared} temp session(s).")
    return 0


# ─────────────────────────────────────────────────────────── entry
def main(argv=None) -> int:
    for _s in (sys.stdout, sys.stderr):           # Windows cmd is cp1252
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(
        description="Recover temp/dry-run recordings into the permanent data tree.")
    ap.add_argument("--cli", action="store_true",
                    help="terminal flow instead of the default GUI dialogs.")
    ap.add_argument("--temp-dir", default=None,
                    help="override the temp dir (default <code>/data/temp).")
    args = ap.parse_args(argv)
    temp_dir = (Path(args.temp_dir) if args.temp_dir
                else Path(paths.top_dir) / "data" / "temp")
    if args.cli:
        return run_cli(temp_dir)
    try:
        return run_gui(temp_dir)
    except Exception as e:
        print(f"GUI unavailable ({e}); falling back to --cli.")
        return run_cli(temp_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted - nothing changed.")
        raise SystemExit(130)
