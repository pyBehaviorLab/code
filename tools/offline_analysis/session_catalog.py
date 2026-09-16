"""Session catalog, read-only project + session discovery for the analyzer.

A *project* is the folder containing ``experiment_config.json``.
A *session* is one recording in one box for one subject on one date,
the record either comes from ``<project>/history/<YYYY-MM-DD>.json`` (the
authoritative source written by the rig GUI when the user clicks Record)
or, when history is missing, is reconstructed by globbing the data layout
``<code>/data/<project_name>/<ddmmyy>/{mcu,video}/`` and pairing files by
the ``<subject>-Box<n>-YYYY-MM-DD-HHMMSS`` naming convention.

CLI smoke test:
    python -m tools.offline_analysis.session_catalog <project_dir>
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session record
# ---------------------------------------------------------------------------

INTEGRITY_OK       = "OK"        # file present and hash matches history
INTEGRITY_MISSING  = "MISSING"   # history says it should be there, isn't
INTEGRITY_MODIFIED = "MODIFIED"  # present but hash mismatches history
INTEGRITY_UNLISTED = "UNLISTED"  # present but no history record (orphan)


@dataclass
class Session:
    """Canonical session record consumed by every analyzer view."""
    project_dir: Path
    date: str                  # YYYY-MM-DD (always ISO, regardless of disk layout)
    subject: str
    task: str = ""
    box: int = 0
    started_at: str = ""       # ISO timestamp from history (when known)
    run_id: str = ""           # YYYYMMDD_HHMMSS_box<N> (when known)
    mcu_path: Optional[Path] = None
    video_path: Optional[Path] = None
    video_header: Dict[str, str] = field(default_factory=dict)
    cohort_meta: Dict[str, str] = field(default_factory=dict)
    integrity: str = INTEGRITY_OK
    duration_s: float = 0.0

    # convenience
    def label(self) -> str:
        bits = [self.date]
        if self.subject: bits.append(self.subject)
        if self.task:    bits.append(self.task)
        if self.box:     bits.append(f"Box{self.box}")
        if self.started_at:
            t = self.started_at.split("T")[-1][:5] if "T" in self.started_at else ""
            if t: bits.append(t)
        return " · ".join(bits)


@dataclass
class ProjectContext:
    """Bundle returned by :func:`load_project_context`."""
    project_dir: Path
    project_name: str
    cohort_path: Optional[Path] = None
    cohort_df: object = None              # pandas.DataFrame when loaded
    sessions: List[Session] = field(default_factory=list)
    cfg: object = None                    # source.config.experiment.Config


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_project_context(cfg_path: str | Path) -> ProjectContext:
    """Load a project + discover its sessions. Read-only.

    ``cfg_path`` may point at the project folder or directly at
    ``experiment_config.json``. The function wraps the rig GUI's
    ``load_experiment_with_guard`` but does not retain the OCC guard,
    the analyzer never writes back to the project.
    """
    p = Path(cfg_path)
    if p.is_file():
        project_dir = p.parent
    else:
        project_dir = p

    # Standalone-friendly project read: parse experiment_config.json directly
    # instead of pulling source.config.experiment (which drags in the whole
    # rig Config dataclass + OCC guard machinery). The analyzer is read-only;
    # raw dict is enough.
    cfg: Optional[dict] = None
    cfg_path = project_dir / "experiment_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("experiment_config.json read failed: %s", e)

    project_name = ""
    if isinstance(cfg, dict):
        meta = cfg.get("meta") or {}
        project_name = str(meta.get("project") or "")
    if not project_name:
        project_name = project_dir.name

    cohort_path, cohort_df = _autoload_cohort(project_dir, cfg)
    sessions = discover_sessions(project_dir, project_name=project_name)
    _attach_cohort_meta(sessions, cohort_df)

    return ProjectContext(
        project_dir=project_dir,
        project_name=project_name,
        cohort_path=cohort_path,
        cohort_df=cohort_df,
        sessions=sessions,
        cfg=cfg,
    )


def _autoload_cohort(project_dir: Path, cfg) -> Tuple[Optional[Path], object]:
    """Resolve and load the cohort metadata file (Excel/CSV) for a project.

    Looks at ``cfg.meta.metadata_file`` first; falls back to anything inside
    ``<project>/metadata/`` if present.
    """
    rel = ""
    if isinstance(cfg, dict):
        rel = str(((cfg.get("meta") or {}).get("metadata_file") or "")).strip()
    cand: Optional[Path] = None
    if rel:
        cand = (project_dir / rel).resolve()
    if cand is None or not cand.exists():
        meta_dir = project_dir / "metadata"
        if meta_dir.is_dir():
            for ext in ("*.xlsx", "*.xls", "*.csv"):
                hits = sorted(meta_dir.glob(ext))
                if hits:
                    cand = hits[0]
                    break
    if cand is None or not cand.exists():
        return None, None
    try:
        import pandas as pd
        if cand.suffix.lower() == ".csv":
            df = pd.read_csv(cand)
        else:
            df = pd.read_excel(cand)
        return cand, df
    except Exception as e:
        logger.warning("cohort load failed (%s): %s", cand, e)
        return cand, None


def _attach_cohort_meta(sessions: List[Session], cohort_df) -> None:
    """Augment each session with the cohort row that matches its subject."""
    if cohort_df is None:
        return
    try:
        # Try canonical column names first
        for col in ("subject_id", "Subject ID", "subject", "Subject"):
            if col in cohort_df.columns:
                id_col = col
                break
        else:
            return
        index = {str(row[id_col]): {k: row[k] for k in cohort_df.columns}
                 for _, row in cohort_df.iterrows()}
        for s in sessions:
            row = index.get(s.subject)
            if row:
                s.cohort_meta = {str(k): str(v) for k, v in row.items()}
    except Exception as e:
        logger.debug("cohort attach failed: %s", e)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_MCU_NAME_RE = re.compile(
    r"^(?P<subject>.+?)-Box(?P<box>\d+)-"
    r"(?P<date>\d{4}-\d{2}-\d{2})-(?P<hms>\d{6})\.tsv$"
)
# Matches the current ``<stem>_video_data.txt`` and the v0 GUI's
# ``<subj>_video_data_-<date>.txt`` (``_video_data`` mid-name).
#
# ANCHORED, and the group allows underscores. Neither was true before, and
# both mattered: used with ``search`` on an unanchored pattern that excluded
# ``_``, the match started wherever the last underscore-free run happened to
# begin. ``Validation_4box_cctv_latency_box1-Box1-…_video_data.txt`` yielded
# ``box1-Box1-…``, and a project whose subjects carry underscores yielded the
# bare timestamp. Lazy ``+?`` keeps the v0 mid-name form resolving to the
# subject rather than swallowing the date after it.
_VID_NAME_RE = re.compile(
    r"^(?P<subject>.+?)_video_data.*\.txt$"
)


def _video_stem(name: str) -> str:
    """Filename minus ``_video_data…`` and the extension.

    What pairing compares. ``_MCU_NAME_RE``'s ``subject`` is only the part
    BEFORE ``-Box<n>-<date>-<hms>``, while a video filename carries the whole
    stem, so comparing the two could never match for any project. Comparing
    stem to stem is convention-agnostic and works for both layouts.
    """
    m = _VID_NAME_RE.match(name)
    return m.group("subject") if m else Path(name).stem
# Date folders under ``data/<project>/`` come in two spellings, because the
# recorder's changed at some point and both are still on disk side by side:
# ``270526`` (ddmmyy) and ``2026-06-09`` (ISO). Matching only the first made
# every recording made after the switch invisible to the analyser, the folder
# was there, full of sessions, and the table simply did not list them.
_DATA_DATE_RE = re.compile(r"^(\d{6}|\d{4}-\d{2}-\d{2})$")


def discover_sessions(project_dir: Path,
                      project_name: str = "") -> List[Session]:
    """Return all sessions for a project, sorted newest → oldest.

    Strategy:
      1. Walk ``history/<date>.json`` files (authoritative if present).
      2. Glob the data layout to find orphan files not in history.
      3. Stamp integrity badges per file (OK / MISSING / MODIFIED / UNLISTED).
    """
    project_dir = Path(project_dir)
    history_rows = _read_history(project_dir)
    data_files = _scan_data_files(project_dir, project_name)

    sessions: List[Session] = []
    seen_files: set = set()

    # 1) Rows from history
    for row in history_rows:
        s = _session_from_history_row(row, project_dir, data_files)
        sessions.append(s)
        if s.mcu_path:   seen_files.add(s.mcu_path)
        if s.video_path: seen_files.add(s.video_path)

    # 2) Orphan files not referenced by history
    for fp in data_files["mcu"]:
        if fp in seen_files:
            continue
        s = _session_from_orphan_mcu(fp, project_dir, data_files)
        if s is not None:
            sessions.append(s)
            if s.video_path:
                seen_files.add(s.video_path)
            seen_files.add(fp)

    for fp in data_files["video"]:
        if fp in seen_files:
            continue
        s = _session_from_orphan_video(fp, project_dir)
        if s is not None:
            sessions.append(s)
            seen_files.add(fp)

    sessions.sort(key=lambda s: (s.date, s.started_at, s.box), reverse=True)
    return sessions


def _read_history(project_dir: Path) -> List[Dict]:
    """Concatenate all ``history/<YYYY-MM-DD>.json`` files."""
    hist_dir = project_dir / "history"
    if not hist_dir.is_dir():
        return []
    rows: List[Dict] = []
    for jp in sorted(hist_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("history read failed %s: %s", jp, e)
            continue
        # File may be a single record or a list of records.
        if isinstance(data, list):
            rows.extend(d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            # Some flavours use {"runs": [...]} envelope
            if isinstance(data.get("runs"), list):
                rows.extend(d for d in data["runs"] if isinstance(d, dict))
            else:
                rows.append(data)
    return rows


def _scan_data_files(project_dir: Path,
                     project_name: str = "") -> Dict[str, List[Path]]:
    """Every recording belonging to this project, whatever the layout.

    The rig writes ``<repo>/data/<project>/<ddmmyy>/<session>.tsv`` with the
    videos in a ``video/`` folder beside them. Three things about that defeated
    the old walk, and all three failed silently, the tab showed an empty table
    for a folder holding thirteen hundred recordings:

    * the ``data`` directory is at the REPOSITORY root, and this climbed only
      as far as ``experiments/`` from ``experiments/projects/<name>``;
    * the ``.tsv`` sits directly in the date folder. It looked under a ``mcu/``
      subfolder, which nothing creates;
    * the data folder carries the SHORT project name (``aCG_RL``) where the
      project folder carries the long one (``DKLab_aCG_reversal_learning``),
      so neither of the two names it tried ever matched.

    So the search is widened at each step and says what it settled for, rather
    than requiring one exact shape and returning nothing when it is not found.
    """
    out: Dict[str, List[Path]] = {"mcu": [], "video": []}
    names = [n for n in (project_name, project_dir.name) if n]
    for date_dir in _date_dirs(project_dir, names):
        # Directly in the date folder is where the recorder puts them; the
        # mcu/ subfolder is kept for layouts that do have one.
        out["mcu"].extend(sorted(date_dir.glob("*.tsv")))
        if (date_dir / "mcu").is_dir():
            out["mcu"].extend(sorted((date_dir / "mcu").glob("*.tsv")))
        for folder in (date_dir / "video", date_dir):
            if not folder.is_dir():
                continue
            vids = set(folder.glob("*_video_data.txt"))
            vids |= set(folder.glob("*_video_data_*.txt"))
            out["video"].extend(sorted(vids))
    for kind in out:
        out[kind] = _unique(out[kind])
    return out


def _unique(paths: List[Path]) -> List[Path]:
    """Order-preserving de-duplication, a folder can be reached two ways."""
    seen: set = set()
    keep: List[Path] = []
    for p in paths:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            keep.append(p)
    return keep


def _data_dirs(project_dir: Path) -> List[Path]:
    """Every ``data`` directory above the project, nearest first."""
    found: List[Path] = []
    for parent in [project_dir, *project_dir.parents]:
        candidate = parent / "data"
        if candidate.is_dir() and candidate not in found:
            found.append(candidate)
    return found


def _date_dirs(project_dir: Path, names: List[str]) -> List[Path]:
    """The ``ddmmyy`` folders holding this project's recordings.

    Preference order, because the first match that yields anything is the most
    specific claim available: a folder named for the project, then date folders
    sitting directly under ``data/``, then, only when neither matched, every
    project folder there is. The last step is what rescues a project whose
    data folder is named differently from its project folder, and it says so,
    because offering another project's recordings silently would be worse than
    offering none.
    """
    for data_dir in _data_dirs(project_dir):
        for name in names:
            dated = _dated_children(data_dir / name)
            if dated:
                return dated
        dated = _dated_children(data_dir)
        if dated:
            return dated
    for data_dir in _data_dirs(project_dir):
        every: List[Path] = []
        for child in sorted(data_dir.iterdir()):
            if child.is_dir():
                every.extend(_dated_children(child))
        if every:
            logger.warning(
                "no data folder under %s is named for this project (tried %s), "
                "so every recording under it is listed. Rename the data folder "
                "to match the project to narrow this.",
                data_dir, " or ".join(names) or "the project name")
            return every
    return []


def _dated_children(folder: Path) -> List[Path]:
    """``ddmmyy`` subfolders of ``folder``, or empty when there are none."""
    if not folder.is_dir():
        return []
    return [d for d in sorted(folder.iterdir())
            if d.is_dir() and _DATA_DATE_RE.match(d.name)]


def _session_from_history_row(row: Dict, project_dir: Path,
                              data_files: Dict[str, List[Path]]) -> Session:
    run_id = row.get("run_id", "") or ""
    date_iso = _date_iso_from_run_id(run_id) if run_id else ""
    if not date_iso:
        ts = row.get("started_at", "")
        date_iso = ts.split("T")[0] if "T" in ts else ts[:10]
    subject = str(row.get("subject_id", "") or "")
    task = str(row.get("task_name", "") or "")
    box = int(row.get("box_id", 0) or 0)
    duration = float(row.get("duration_s", 0.0) or 0.0)
    started_at = str(row.get("started_at", "") or "")
    # File pairing
    mcu, vid = _find_files_for_history(row, subject, box, date_iso, data_files)
    integrity = INTEGRITY_OK
    files_block = row.get("data_files") or {}
    if isinstance(files_block, dict) and files_block:
        # Compare hashes/sizes from history.data_files against discovered file
        for kind, fp in (("mcu", mcu), ("video", vid)):
            spec = files_block.get(kind)
            if isinstance(spec, dict):
                expected_path = spec.get("path", "")
                if expected_path and fp is None:
                    integrity = INTEGRITY_MISSING
                    break
                # Read-old-data gate: pre-djb2 history rows carried a
                # per-file "sha256"; the rig no longer writes it, but the
                # analyzer still verifies those older sessions with it.
                if fp is not None and "sha256" in spec:
                    actual = _sha256(fp)
                    if actual and spec["sha256"] and actual != spec["sha256"]:
                        integrity = INTEGRITY_MODIFIED
                        break
            elif spec and fp is None:
                integrity = INTEGRITY_MISSING
                break
    return Session(
        project_dir=project_dir, date=date_iso, subject=subject,
        task=task, box=box, started_at=started_at, run_id=run_id,
        mcu_path=mcu, video_path=vid, integrity=integrity,
        duration_s=duration,
    )


def _find_files_for_history(row: Dict, subject: str, box: int, date_iso: str,
                            data_files: Dict[str, List[Path]]
                            ) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate MCU + video file for a history row. Tries explicit paths
    first, falls back to naming-convention match."""
    mcu_path: Optional[Path] = None
    vid_path: Optional[Path] = None
    files_block = row.get("data_files") or {}
    if isinstance(files_block, dict):
        for key in ("mcu", "mcu_log", "tsv"):
            v = files_block.get(key)
            if isinstance(v, dict) and v.get("path"):
                p = Path(v["path"])
                if p.is_file(): mcu_path = p
        for key in ("video", "video_data", "txt"):
            v = files_block.get(key)
            if isinstance(v, dict) and v.get("path"):
                p = Path(v["path"])
                if p.is_file(): vid_path = p
    if mcu_path is None:
        # Match by naming convention
        for fp in data_files["mcu"]:
            m = _MCU_NAME_RE.match(fp.name)
            if m and m.group("subject") == subject \
                    and int(m.group("box")) == box \
                    and m.group("date") == date_iso:
                mcu_path = fp
                break
    if vid_path is None and mcu_path is not None:
        # Stem to stem. The MCU file and its video share a stem in every
        # layout; comparing the MCU's ``subject`` (which stops before
        # ``-Box<n>-``) against a video's full stem never matched for any
        # project, which is why this heuristic had never once fired.
        want = mcu_path.stem
        for fp in data_files["video"]:
            if _video_stem(fp.name) == want:
                vid_path = fp
                break
    if vid_path is None:
        for fp in data_files["video"]:
            stem = _video_stem(fp.name)
            if not stem.startswith(subject):
                continue
            date_dir = fp.parent.parent.name
            if _iso_from_ddmmyy(date_dir) == date_iso:
                vid_path = fp
                break
    return mcu_path, vid_path


def _session_from_orphan_mcu(fp: Path, project_dir: Path,
                             data_files: Dict[str, List[Path]]
                             ) -> Optional[Session]:
    m = _MCU_NAME_RE.match(fp.name)
    if not m:
        return None
    subject = m.group("subject")
    box = int(m.group("box"))
    date_iso = m.group("date")
    hms = m.group("hms")
    started = f"{date_iso}T{hms[:2]}:{hms[2:4]}:{hms[4:6]}"
    # Pair with video if available. Stem to stem first, for the same reason
    # as above; the subject-prefix pass is the fallback for layouts where the
    # two names genuinely differ.
    vid = None
    for vp in data_files["video"]:
        if _video_stem(vp.name) == fp.stem:
            vid = vp
            break
    if vid is None:
        for vp in data_files["video"]:
            if not _video_stem(vp.name).startswith(subject):
                continue
            date_dir = vp.parent.parent.name
            if _iso_from_ddmmyy(date_dir) == date_iso:
                vid = vp
                break
    return Session(
        project_dir=project_dir, date=date_iso, subject=subject,
        box=box, started_at=started,
        mcu_path=fp, video_path=vid,
        integrity=INTEGRITY_UNLISTED,
    )


def _session_from_orphan_video(fp: Path, project_dir: Path) -> Optional[Session]:
    vm = _VID_NAME_RE.search(fp.name)
    if not vm:
        return None
    subject = vm.group("subject")
    date_dir = fp.parent.parent.name
    date_iso = _iso_from_ddmmyy(date_dir) or ""
    return Session(
        project_dir=project_dir, date=date_iso, subject=subject,
        video_path=fp, integrity=INTEGRITY_UNLISTED,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_iso_from_run_id(run_id: str) -> str:
    """run_id ``YYYYMMDD_HHMMSS_box<N>`` -> ``YYYY-MM-DD``."""
    if len(run_id) < 8:
        return ""
    ymd = run_id[:8]
    if not ymd.isdigit():
        return ""
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _iso_from_ddmmyy(name: str) -> Optional[str]:
    """A date folder's name -> ``YYYY-MM-DD``.

    Both spellings the recorder has used: ``ddmmyy`` (assumes 20xx) and an ISO
    date, which is already the answer.
    """
    if not _DATA_DATE_RE.match(name):
        return None
    if "-" in name:
        return name
    dd, mm, yy = name[:2], name[2:4], name[4:6]
    yyyy = f"20{yy}"
    return f"{yyyy}-{mm}-{dd}"


def _sha256(p: Optional[Path], *, chunk: int = 1 << 16) -> Optional[str]:
    if p is None or not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _print_table(sessions: List[Session]) -> None:
    rows = [("Date", "Subject", "Task", "Box", "Started", "Vid", "MCU", "Status")]
    for s in sessions:
        rows.append((
            s.date, s.subject, s.task, str(s.box) if s.box else "",
            (s.started_at.split("T")[-1][:8] if "T" in s.started_at else ""),
            "✓" if s.video_path else "–",
            "✓" if s.mcu_path else "–",
            s.integrity,
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        line = "  ".join(c.ljust(w) for c, w in zip(r, widths))
        print(line)
        if i == 0:
            print("  ".join("-" * w for w in widths))


def _cli(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m tools.offline_analysis.session_catalog <project_dir>")
        return 2
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ctx = load_project_context(argv[1])
    print(f"Project: {ctx.project_name}")
    print(f"Path:    {ctx.project_dir}")
    print(f"Cohort:  {ctx.cohort_path or '<none>'}")
    print(f"Sessions discovered: {len(ctx.sessions)}")
    if ctx.sessions:
        print()
        _print_table(ctx.sessions)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
