"""
Parser for ``*_video_data.txt`` session logs written by the live
tracking writer (source/communication/tracking_writer.py on the rig).
Reads them back into a SessionData object consumed by the offline
analyzer views.

Usage:
    from tools.offline_analysis.video_data_parser import (
        parse_video_data, find_video_data_files,
    )

    files = find_video_data_files("D:/path/to/data")
    session = parse_video_data(files[0])
    session.zones              # list of zone dicts
    session.frames_df          # pandas DataFrame, one row per captured frame
    session.events             # list of #event dicts (MCU mirror + zone triggers)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# v1 columns, the only format produced before #version 2 (2026-05-29).
DEFAULT_COLUMNS_V1: Tuple[str, ...] = (
    "frame_num", "capture_elapsed", "mcu_fw_capture_ms", "mcu_fw_pose_ms",
    "pose", "zone", "speed_pxs", "stage", "note",
)

# Back-compat alias for any external caller importing the old name.
DEFAULT_COLUMNS = DEFAULT_COLUMNS_V1

_KV_RE = re.compile(r"^#([a-zA-Z_][a-zA-Z0-9_]*)\s+(.*)$")


@dataclass
class SessionData:
    """Parsed _video_data.txt session."""
    filepath: str
    info: Dict = field(default_factory=dict)
    frames_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    events: List[Dict] = field(default_factory=list)

    @property
    def subject_id(self) -> str:
        pc = self.info.get("pycontrol", {})
        if isinstance(pc, dict):
            return str(pc.get("subject", ""))
        return ""

    @property
    def box_id(self) -> int:
        try:
            return int(self.info.get("box_id", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def zones(self) -> List[Dict]:
        """Zone list in the shape the zone-editor UI expects: [{name, type, points, ...}]."""
        zc = self.info.get("zone_config", {})
        zs = zc.get("zones", {}) if isinstance(zc, dict) else {}
        if isinstance(zs, dict):
            out: List[Dict] = []
            for name, vals in zs.items():
                if isinstance(vals, dict):
                    d = {"name": name}
                    d.update(vals)
                    out.append(d)
            return out
        if isinstance(zs, list):
            return list(zs)
        return []

    @property
    def scale(self) -> Dict:
        zc = self.info.get("zone_config", {})
        return zc.get("scale", {}) if isinstance(zc, dict) else {}

    @property
    def arena(self) -> Dict:
        zc = self.info.get("zone_config", {})
        return zc.get("arena", {}) if isinstance(zc, dict) else {}

    @property
    def resolution(self) -> Tuple[int, int]:
        cam = self.info.get("camera", {})
        res = cam.get("resolution", "") if isinstance(cam, dict) else ""
        if isinstance(res, str) and "x" in res:
            try:
                w, h = res.split("x", 1)
                return (int(w), int(h))
            except ValueError:
                pass
        return (0, 0)

    @property
    def fps(self) -> float:
        cam = self.info.get("camera", {})
        try:
            return float(cam.get("fps", 30.0)) if isinstance(cam, dict) else 30.0
        except (TypeError, ValueError):
            return 30.0

    @property
    def tracking_mode(self) -> str:
        tr = self.info.get("tracker", {})
        return tr.get("backend", "unknown") if isinstance(tr, dict) else "unknown"

    @property
    def speed_unit(self) -> str:
        """Unit of the ``speed_pxs`` / ``speed`` columns. The writer
        declares ``#units {"speed": "m/s"}`` when scale-calibrated and
        ``px/s`` otherwise; v0/v1 files predate ``#units`` → px/s."""
        units = self.info.get("units", {})
        if isinstance(units, dict):
            return str(units.get("speed") or "px/s")
        return "px/s"

    @property
    def duration_sec(self) -> float:
        df = self.frames_df
        if df.empty:
            return 0.0
        if "mcu_fw_capture_ms" in df:
            ts = df["mcu_fw_capture_ms"].dropna()
            if len(ts) >= 2:
                return float((ts.iloc[-1] - ts.iloc[0]) / 1000.0)
        # Board-less / dry-run sessions log "na" fw timestamps; fall back
        # to the elapsed clock column, which is populated for every frame.
        if "capture_elapsed" in df:
            secs = df["capture_elapsed"].map(_elapsed_to_sec).dropna()
            if len(secs) >= 2:
                return float(secs.iloc[-1] - secs.iloc[0])
        return 0.0

    @property
    def n_frames(self) -> int:
        return len(self.frames_df)


def parse_video_data(filepath: str) -> SessionData:
    """Parse a _video_data.txt file into SessionData.

    Auto-detects the format. The legacy **v0** files (written by the v0
    GUI) use ``B``/``M``/``V`` header lines followed by one JSON object
    per row; they are routed to the self-contained, removable
    :mod:`legacy_v0` module. The ``#``-header formats are v1 (nine columns
    ending in ``stage`` / ``note`` plus free-form ``#event …`` lines) and
    v2 (ten columns, frame, elapsed, frame_fw_ms, pose, zone, speed,
    state, events, pose_fw_ms, track, with state/event names folded into
    the row; ``track`` says whether the row's position was measured,
    coasted or unknown). **v3** is v2 with ``capture_host_ns`` inserted
    third: the raw host instant the frame was stamped with, which
    ``elapsed`` and ``frame_fw_ms`` are both lossy renderings of.

    v2 and v3 share a reader. Columns are resolved BY NAME from the
    ``#columns`` header, never by position, so inserting a column does not
    move any other one.

    The returned ``frames_df`` is always written under the v1 column
    NAMES (``frame_num``, ``mcu_fw_capture_ms``, ``speed_pxs``, …) so
    downstream tools (viewers, exporters) don't need to branch on
    version, including v0.
    """
    from . import legacy_v0  # local import keeps the v0 dependency one-directional
    if legacy_v0.is_v0_file(filepath):
        return legacy_v0.parse_v0(filepath)

    # The arena recorder writes tab-separated info rows and a ``# <names>``
    # column header, neither of which this reader can see: the info rows land
    # as data and the header is filed as a key. Routed out for the same reason
    # v0 is, and on the same terms, its own module, sniffed by content.
    from . import legacy_arena
    if legacy_arena.is_arena_file(filepath):
        return legacy_arena.parse_arena(filepath)

    info: Dict = {}
    events: List[Dict] = []
    data_rows: List[List[str]] = []
    col_names: List[str] = list(DEFAULT_COLUMNS_V1)

    in_zone_config = False
    zone_config_lines: List[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            # Multi-line zone_config JSON block
            if stripped == "#zone_config_begin":
                in_zone_config = True
                zone_config_lines = []
                continue
            if stripped == "#zone_config_end":
                in_zone_config = False
                try:
                    info["zone_config"] = json.loads("\n".join(zone_config_lines))
                except Exception as e:
                    logger.warning(f"zone_config parse error in {filepath}: {e}")
                continue
            if in_zone_config:
                zone_config_lines.append(line)
                continue

            # v1 event rows (MCU mirror + zone triggers)
            if line.startswith("#event"):
                ev = _parse_event_row(line)
                if ev:
                    events.append(ev)
                continue

            # Misc comment lines we skip
            if line.startswith("#resync") or line.startswith("#session_end") or line.startswith("# ="):
                continue

            # Column declaration
            if line.startswith("#columns"):
                m = _KV_RE.match(line)
                if m:
                    col_names = m.group(2).split()
                continue

            # Header #key value rows (JSON value or plain)
            if line.startswith("#"):
                m = _KV_RE.match(line)
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                if val.startswith("{") or val.startswith("["):
                    try:
                        info[key] = json.loads(val)
                        continue
                    except Exception:
                        pass
                info[key] = val
                continue

            # Data row, tab-separated
            parts = line.split("\t")
            if len(parts) < len(col_names):
                parts = parts + ["na"] * (len(col_names) - len(parts))
            data_rows.append(parts[: len(col_names)])

    # Detect version. ``#version`` is a header field (numeric str).
    try:
        version = int(str(info.get("version", "1")))
    except (TypeError, ValueError):
        version = 1
    # Belt and braces, a corrupt/missing ``#version`` header must not
    # demote a v2 file to v1: the per-frame fw-timestamp columns exist
    # only in the v2 ``#columns`` line, so their presence is a v2 signature.
    is_v2 = version >= 2 or any(
        n in col_names for n in ("frame_fw_ms", "fw_ms", "pose_fw_ms", "fw_pose_ms"))

    if is_v2:
        frames_df = _build_frames_df_v2(data_rows, col_names, events)
    else:
        frames_df = _build_frames_df_v1(data_rows, col_names)

    session = SessionData(filepath=filepath, info=info,
                          frames_df=frames_df, events=events)
    logger.info(
        f"Parsed {filepath} (v{version}): {session.n_frames} frames, "
        f"{len(events)} events, {len(session.zones)} zones"
    )
    return session


def _build_frames_df_v1(data_rows: List[List[str]],
                        col_names: List[str]) -> pd.DataFrame:
    """v1 → frames_df. Every row is a frame; columns are as declared."""
    if not data_rows:
        return pd.DataFrame(columns=list(col_names))
    df = pd.DataFrame(data_rows, columns=list(col_names))
    df["frame_num"] = pd.to_numeric(df["frame_num"], errors="coerce").fillna(0).astype(int)
    df["mcu_fw_capture_ms"] = pd.to_numeric(df["mcu_fw_capture_ms"], errors="coerce")
    df["mcu_fw_pose_ms"]    = pd.to_numeric(df["mcu_fw_pose_ms"], errors="coerce")
    df["speed_pxs"]         = pd.to_numeric(df["speed_pxs"], errors="coerce")

    xy = df["pose"].apply(_extract_first_xy)
    df["x"] = xy.apply(lambda t: t[0])
    df["y"] = xy.apply(lambda t: t[1])

    df["location"] = df["zone"].astype(str).where(df["zone"] != "na", "")

    # older aliases for callers that grew up on the old JSONL parser
    df["frames"]     = df["frame_num"]
    df["timestamps"] = df["mcu_fw_capture_ms"].fillna(0).astype(int)
    df["speed"]      = df["speed_pxs"].fillna(0.0)
    return df


def _build_frames_df_v2(data_rows: List[List[str]],
                        col_names: List[str],
                        events_out: List[Dict]) -> pd.DataFrame:
    """v2 → frames_df.

    Every row is one camera frame. State and event names that landed
    during the frame's interval are pipe-joined in ``state`` /
    ``events`` columns. The frames DataFrame is renamed to v1 column
    names so downstream tools don't need to branch on version; the
    state/event names are also exploded into ``events_out`` for
    backwards-compat with callers that consumed v1's interleaved
    ``#event`` lines.
    """
    if not data_rows:
        return pd.DataFrame(columns=list(DEFAULT_COLUMNS_V1))

    def _idx(*names):
        """First matching column index among ``names`` (handles the
        frame_fw_ms→fw_ms rename across file revisions)."""
        for n in names:
            if n in col_names:
                return col_names.index(n)
        raise ValueError(names[0])

    try:
        idx_frame   = _idx("frame")
        idx_elapsed = _idx("elapsed")
        idx_fw      = _idx("frame_fw_ms", "fw_ms")
        idx_pose    = _idx("pose")
        idx_zone    = _idx("zone")
        idx_state   = _idx("state")
    except ValueError as e:
        logger.warning("v2 header missing expected column: %s", e)
        return pd.DataFrame(columns=list(DEFAULT_COLUMNS_V1))
    # Optional columns. ``host_mono_ns`` was removed when fw timestamps
    # became raw ``pycboard.timestamp`` reads, older files may still
    # carry it, so read it when present.
    idx_fw_pose = (col_names.index("pose_fw_ms") if "pose_fw_ms" in col_names
                   else col_names.index("fw_pose_ms") if "fw_pose_ms" in col_names
                   else None)
    # v3 restored the raw host instant under its real name,
    # ``capture_host_ns``. Both spellings land in the same field: the value is
    # the same thing, ``host_clock.host_ns()`` read when the frame arrived, and
    # a caller should not have to know which era its file came from.
    idx_host_mono = (col_names.index("capture_host_ns")
                     if "capture_host_ns" in col_names
                     else col_names.index("host_mono_ns")
                     if "host_mono_ns" in col_names else None)
    # Columns the live writer no longer keeps. ``speed`` follows from the
    # coordinates and the scale, and the task's events are the MCU TSV's job;
    # files written before the cut still carry both, so they are read when
    # present rather than required.
    idx_speed = col_names.index("speed") if "speed" in col_names else None
    idx_events = col_names.index("events") if "events" in col_names else None
    # How old this row's pose is, and how much of that was the smoother.
    idx_infer = col_names.index("pose_lag_ms") if "pose_lag_ms" in col_names else None
    idx_filter = (col_names.index("filter_ms")
                  if "filter_ms" in col_names else None)
    # Per-keypoint zone membership, appended after ``track``. Absent from
    # files written before it existed, which ``_col`` handles.
    idx_part_zones = (col_names.index("part_zones")
                      if "part_zones" in col_names else None)
    idx_filled = (col_names.index("filled_parts")
                  if "filled_parts" in col_names else None)

    def _col(idx, default="na"):
        if idx is None:
            return [default] * len(data_rows)
        return [r[idx] if idx < len(r) else default for r in data_rows]

    # Walk every frame row, exploding any non-empty state/events
    # field into events_out for v1-compat callers. The folded value
    # is also preserved on the frame row itself under "state" / "events"
    # columns so v2-aware tools can read it without crawling events_out.
    for row in data_rows:
        if len(row) <= idx_state:
            continue
        try:
            fw_ms = float(row[idx_fw]) if row[idx_fw] != "na" else None
        except (TypeError, ValueError):
            fw_ms = None
        try:
            frame_num = int(row[idx_frame])
        except (TypeError, ValueError):
            frame_num = None
        state_field  = row[idx_state]
        events_field = (row[idx_events] if idx_events is not None
                        and idx_events < len(row) else "-")
        if state_field and state_field != "-":
            for name in state_field.split("|"):
                events_out.append({"kind": "S", "name": name,
                                   "fw_ms": fw_ms, "frame": frame_num})
        if events_field and events_field != "-":
            for name in events_field.split("|"):
                events_out.append({"kind": "E", "name": name,
                                   "fw_ms": fw_ms, "frame": frame_num})

    df = pd.DataFrame({
        "frame_num":         [r[idx_frame]   for r in data_rows],
        "capture_elapsed":   [r[idx_elapsed] for r in data_rows],
        "mcu_fw_capture_ms": [r[idx_fw]      for r in data_rows],
        # MCU fw time when the row's pose was inferred (na when no pose /
        # older files). Coerced below.
        "mcu_fw_pose_ms":    _col(idx_fw_pose),
        # The raw host instant the frame was stamped with, from v3's
        # ``capture_host_ns`` or the older ``host_mono_ns``. Every other clock
        # on the row is derived from it: ``elapsed`` truncates it to a
        # millisecond and ``mcu_fw_capture_ms`` rounds it AND clamps it
        # forward, so this is the only one that can be checked or recomputed.
        # It is also the only key that matches ACROSS boxes on a shared
        # camera, since ``frame_num`` counts from each box's own first frame
        # and the boxes start recording a beat apart.
        "host_mono_ns":      _col(idx_host_mono),
        "capture_host_ns":   _col(idx_host_mono),
        "pose":              [r[idx_pose]    for r in data_rows],
        "zone":              [r[idx_zone]    for r in data_rows],
        # ``speed_pxs`` keeps the v1 column NAME even when the file
        # actually stored m/s, ``SessionData.speed_unit`` says which.
        "speed_pxs":         _col(idx_speed, "na"),
        # New v2 columns surfaced directly to the frames_df.
        "state":             [r[idx_state]   for r in data_rows],
        "events":            _col(idx_events, "-"),
        # ``Snout:Left_poke|Head:Left_poke`` or "-". The ``zone`` column
        # above is the CENTROID's zone, which for small zones such as pokes
        # is empty while the snout is well inside one.
        "part_zones":        _col(idx_part_zones, "-"),
        # Keypoints predicted across a brief disappearance rather than
        # measured. Analysis that wants only measured points filters here.
        "filled_parts":      _col(idx_filled, "-"),
        # Milliseconds from this pose's frame being captured to its pose
        # existing, and the share of that the smoother took.
        "pose_lag_ms":          _col(idx_infer, "na"),
        "filter_ms":         _col(idx_filter, "na"),
        # v1 columns retired in v2, keep keys to avoid KeyError on
        # v1-only callers.
        "stage":             ["na"]          * len(data_rows),
        "note":              [""]            * len(data_rows),
    })
    df["frame_num"]         = pd.to_numeric(df["frame_num"], errors="coerce").fillna(0).astype(int)
    df["mcu_fw_capture_ms"] = pd.to_numeric(df["mcu_fw_capture_ms"], errors="coerce")
    df["mcu_fw_pose_ms"]    = pd.to_numeric(df["mcu_fw_pose_ms"], errors="coerce")
    df["host_mono_ns"]      = pd.to_numeric(df["host_mono_ns"], errors="coerce")
    # int64, not float: at a perf_counter epoch of a few days a float64 still
    # holds the nanoseconds, but the column is an instant and reads as one.
    df["capture_host_ns"]   = pd.to_numeric(df["capture_host_ns"],
                                            errors="coerce").astype("Int64")
    df["speed_pxs"]         = pd.to_numeric(df["speed_pxs"], errors="coerce")

    xy = df["pose"].apply(_extract_first_xy)
    df["x"] = xy.apply(lambda t: t[0])
    df["y"] = xy.apply(lambda t: t[1])

    df["location"] = df["zone"].astype(str).where(df["zone"] != "na", "")

    df["frames"]     = df["frame_num"]
    df["timestamps"] = df["mcu_fw_capture_ms"].fillna(0).astype(int)
    df["speed"]      = df["speed_pxs"].fillna(0.0)
    return df


def find_video_data_files(directory: str) -> List[str]:
    """Recursively enumerate ``*_video_data*.txt`` files under ``directory``.

    Covers both naming layouts: the current ``<stem>_video_data.txt`` and
    the v0 GUI's ``<subj>_video_data_-<date>.txt`` (where ``_video_data``
    is mid-name, not the suffix).
    """
    base = Path(directory)
    seen: Dict[str, Path] = {}
    for pattern in ("*_video_data.txt", "*_video_data_*.txt"):
        for p in base.rglob(pattern):
            seen[str(p)] = p
    results = list(seen.keys())
    results.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return results


# ---- internals --------------------------------------------------------


def _parse_event_row(line: str) -> Dict:
    """Parse '#event fw_ms=... type=... key=val ... frame=N' into a dict."""
    tokens = line.replace("\t", " ").split()
    if not tokens or tokens[0] != "#event":
        return {}
    out: Dict = {}
    for tok in tokens[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            # Try to coerce numerics for fw_ms / frame
            if k in ("fw_ms", "frame"):
                try:
                    out[k] = float(v) if k == "fw_ms" else int(v)
                    continue
                except ValueError:
                    pass
            out[k] = v
        else:
            # Free-form token (rare)
            out.setdefault("_extra", []).append(tok)
    return out


def _elapsed_to_sec(value) -> float | None:
    """``MM:SS.mmm`` / ``HH:MM:SS.mmm`` elapsed-clock string → seconds.

    Plain numeric strings pass through; ``na`` / empty → None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("na", "-"):
        return None
    try:
        parts = s.split(":")
        sec = float(parts[-1])
        if len(parts) >= 2:
            sec += int(parts[-2]) * 60
        if len(parts) >= 3:
            sec += int(parts[-3]) * 3600
        return sec
    except (TypeError, ValueError):
        return None


def _extract_first_xy(pose_str) -> Tuple[float, float]:
    """Extract (x, y) from the first body part of a JSON pose string."""
    if not isinstance(pose_str, str) or pose_str.strip() in ("", "na"):
        return (np.nan, np.nan)
    try:
        arr = json.loads(pose_str)
        if arr and isinstance(arr, list) and len(arr[0]) >= 2:
            return (float(arr[0][0]), float(arr[0][1]))
    except Exception:
        pass
    return (np.nan, np.nan)
