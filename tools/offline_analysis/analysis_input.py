"""
Format bridge, turn a parsed :class:`SessionData` (old ``B/M/V`` v0 or new
``#``-header v1/v2, both produced by the existing readers) into the exact input
buffers that :mod:`zone_stats_full` expects.

This is the ONLY module that knows how a session's columns map onto the analysis
inputs. Pose parsing, timestamp selection, state normalization and zone-shape
conversion all live here, so the format readers (``legacy_v0`` /
``video_data_parser``) stay format-pure and the analysis (``zone_stats_full``)
never branches on file format.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .zone_stats_full import (
    FastZoneAssigner,
    compute_zone_stats_df,
    write_zone_stats_excel,
)


# =============================================================================
# Canonical analysis input
# =============================================================================

@dataclass
class AnalysisInput:
    """Buffers consumed by ``compute_zone_stats_df`` (one entry per frame)."""
    stem: str
    times_buf: List[float]                      # ms (monotonic-unwrapped downstream)
    zone_buf: List[Optional[str]]               # body zone per frame
    state_buf: List[Optional[str]]              # state/stage per frame (None = drop)
    centers_buf: List[Optional[Tuple[float, float]]]  # body keypoint (x, y)
    head_buf: List[Optional[Tuple[float, float]]]     # head keypoint (x, y)
    zones_full: Dict[str, Any]                  # retracking shape {'zones':{...},'scale':...}
    file_meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Helpers
# =============================================================================

def clean_stem(filepath: str) -> str:
    """Output stem from a ``*_video_data*.txt`` path.

    ``D860_video_data_-2025-07-30-113426.txt`` -> ``D860-2025-07-30-113426``
    ``282-Box9-2026-06-05-103607_video_data.txt`` -> ``282-Box9-2026-06-05-103607``
    """
    base = os.path.basename(filepath)
    name = os.path.splitext(base)[0]
    for sep in ("_video_data_-", "_video_data-"):
        if sep in base:
            parts = base.split(sep)
            return f"{parts[0]}-{parts[-1].rsplit('.txt', 1)[0]}"
    if "_video_data" in name:
        return name.split("_video_data")[0].rstrip("-_") or name
    return name


def _parse_pose(pose_val) -> Optional[List[List[float]]]:
    """Parse a pose cell into ``[[x, y, conf], ...]`` or None.

    Accepts the JSON string stored in the ``pose`` column (v0/v1/v2) and, for
    safety, an already-decoded list.
    """
    if pose_val is None:
        return None
    if isinstance(pose_val, list):
        arr = pose_val
    else:
        s = str(pose_val).strip()
        if not s or s in ("na", "-", "nan", "None"):
            return None
        try:
            arr = json.loads(s)
        except Exception:
            return None
    if not isinstance(arr, list) or not arr:
        return None
    return arr


def _kp_xy(pose: Optional[List[List[float]]], idx: int) -> Optional[Tuple[float, float]]:
    if not pose or idx < 0 or idx >= len(pose):
        return None
    pt = pose[idx]
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (x, y)


def _elapsed_to_ms(val) -> float:
    """Parse a ``MM:SS.mmm`` (or ``SS.mmm``) capture_elapsed string to ms."""
    if val is None:
        return float("nan")
    s = str(val).strip()
    if not s or s in ("na", "-"):
        return float("nan")
    try:
        if ":" in s:
            mm, rest = s.split(":", 1)
            return (int(mm) * 60.0 + float(rest)) * 1000.0
        return float(s) * 1000.0
    except (TypeError, ValueError):
        return float("nan")


def _times_ms(df, fps: float) -> List[float]:
    """Per-frame timestamps in ms, with graceful fallbacks.

    Prefers the MCU-fw capture clock; falls back to ``capture_elapsed`` and
    finally to ``frame_num / fps``. Reset/segment handling is done by the
    analysis (timestamp unwrap), so per-segment clocks are fine here.
    """
    n = len(df)
    ts = None
    if "mcu_fw_capture_ms" in df.columns:
        ts = df["mcu_fw_capture_ms"].to_numpy(dtype=float)
    if ts is None or not np.any(np.isfinite(ts)):
        if "capture_elapsed" in df.columns:
            ts = np.array([_elapsed_to_ms(v) for v in df["capture_elapsed"]], dtype=float)
    if ts is None or not np.any(np.isfinite(ts)):
        if "timestamps" in df.columns:
            cand = df["timestamps"].to_numpy(dtype=float)
            if np.any(cand > 0):
                ts = cand
    if ts is None or not np.any(np.isfinite(ts)):
        f = fps if (fps and fps > 0) else 30.0
        frames = (df["frame_num"].to_numpy(dtype=float)
                  if "frame_num" in df.columns else np.arange(n, dtype=float))
        ts = frames / f * 1000.0
    # Replace residual NaNs by forward-fill so dt computation stays monotonic.
    out: List[float] = []
    last = 0.0
    for v in ts:
        if v is None or not np.isfinite(v):
            out.append(last)
        else:
            last = float(v)
            out.append(last)
    return out


def _norm_state(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("na", "-", "nan", "None"):
        return None
    if "State:" in s:
        s = s.split("State:", 1)[-1].strip()
    return s or None


def _state_buf(df) -> List[Optional[str]]:
    """Per-frame state. ``state`` (v0/v2) wins over ``stage`` (v1)."""
    col = None
    for cand in ("state", "stage"):
        if cand in df.columns:
            vals = [_norm_state(v) for v in df[cand]]
            if any(v is not None for v in vals):
                return vals
            if col is None:
                col = vals
    return col if col is not None else [None] * len(df)


def _zones_full_from_list(zone_list, scale, arena) -> Dict[str, Any]:
    """Build the retracking-shape zones dict from a list of zone dicts.

    ``{'zones': {name: {'values': [[x, y], ...]}}, 'scale': {...}, 'arena': {...}}``.
    The synthetic ``type:"scale"`` zone (injected by the v0 reader) is skipped.
    """
    zones_out: Dict[str, Dict] = {}
    for z in zone_list or []:
        if z.get("type") == "scale":
            continue
        name = z.get("name")
        if not name:
            continue
        pts = z.get("values") or z.get("points") or []
        if len(pts) >= 3:
            zones_out[name] = {"values": [[float(p[0]), float(p[1])] for p in pts]}
            for k in ("ia_inner_offset_px", "ia_line_extend_px", "color", "type"):
                if k in z:
                    zones_out[name][k] = z[k]
    return {
        "zones": zones_out,
        "scale": scale or {},
        "arena": arena or {},
    }


def _file_meta_from_session(session) -> Dict[str, Any]:
    """Map session header info to the keys the metadata parser reads."""
    info = session.info or {}
    v0_meta = info.get("_v0_meta")
    if isinstance(v0_meta, dict) and v0_meta:
        return dict(v0_meta)  # already 'Subject ID' / 'Group' / 'Start time' keyed
    pc = info.get("pycontrol", {}) if isinstance(info.get("pycontrol"), dict) else {}
    return {
        "Subject ID": pc.get("subject", "") or info.get("subject", ""),
        "Group": info.get("group", ""),
        "Sub Group": info.get("subgroup", ""),
        "Start time": info.get("session_start", "") or info.get("start_time", ""),
        "Experiment": pc.get("task", ""),
    }


def suggest_head_body_indices(session, default=(0, 1)) -> Tuple[int, int]:
    """Best-effort head/body keypoint indices from any available names.

    Falls back to ``default`` (head=0, body=1) when no names are known, which
    is the common case for the JSON pose lists (positional, unnamed).
    """
    names = None
    info = session.info or {}
    for key in ("body_parts", "bodyparts", "keypoint_names"):
        if isinstance(info.get(key), list) and info[key]:
            names = info[key]
            break
    tr = info.get("tracker")
    if names is None and isinstance(tr, dict) and isinstance(tr.get("bodyparts"), list):
        names = tr["bodyparts"]
    if not names:
        return default
    low = [str(n).lower() for n in names]
    head_keys = ["nose", "snout", "head", "ear", "face", "forehead"]
    body_keys = ["body", "center", "torso", "mid", "spine", "neck", "thorax", "back"]

    def find(keys):
        for k in keys:
            for i, nm in enumerate(low):
                if k in nm:
                    return i
        return None

    h = find(head_keys)
    b = find(body_keys)
    if h is None and b is None:
        return default
    if h is None:
        h = 0 if b != 0 else (1 if len(low) > 1 else 0)
    if b is None:
        b = 1 if (len(low) > 1 and h != 1) else (0 if h != 0 else h)
    if h == b and len(low) > 1:
        b = 0 if h != 0 else 1
    return h, b


# =============================================================================
# Canonicalize
# =============================================================================

def canonicalize_session(session, head_idx: int = 0, body_idx: int = 1,
                         zones=None) -> AnalysisInput:
    """Map a parsed SessionData onto the analysis input buffers.

    ``zones`` overrides the session's zone list (e.g. the Analyze UI's adjusted
    working copy); when None the session's own zones are used.

    Body zone is recomputed from the body keypoint with the same zone assigner
    the analysis uses for the head, so body/head are symmetric and independent
    of the live ``zone`` column (which is often ``na`` in new files). It falls
    back to the logged ``location`` when no pose/zones are available.
    """
    df = session.frames_df
    stem = clean_stem(session.filepath)
    zone_list = session.zones if zones is None else zones
    zones_full = _zones_full_from_list(zone_list, session.scale, session.arena)
    file_meta = _file_meta_from_session(session)

    if df is None or len(df) == 0:
        return AnalysisInput(stem, [], [], [], [], [], zones_full, file_meta)

    n = len(df)
    times_buf = _times_ms(df, getattr(session, "fps", 30.0))
    state_buf = _state_buf(df)

    pose_series = df["pose"] if "pose" in df.columns else [None] * n
    loc_series = (df["location"].tolist() if "location" in df.columns
                  else (df["zone"].tolist() if "zone" in df.columns else [None] * n))

    assigner = FastZoneAssigner(zones_full) if zones_full.get("zones") else None

    centers_buf: List[Optional[Tuple[float, float]]] = []
    head_buf: List[Optional[Tuple[float, float]]] = []
    zone_buf: List[Optional[str]] = []

    for i in range(n):
        pose = _parse_pose(pose_series.iloc[i] if hasattr(pose_series, "iloc") else pose_series[i])
        center = _kp_xy(pose, body_idx)
        head = _kp_xy(pose, head_idx)
        centers_buf.append(center)
        head_buf.append(head)

        body_zone = None
        if assigner is not None and center is not None:
            body_zone = assigner.locate(center[0], center[1])
        if body_zone is None:
            loc = loc_series[i]
            if loc is not None:
                s = str(loc).strip()
                body_zone = s if s and s not in ("na", "-", "nan", "None", "") else None
        zone_buf.append(body_zone)

    return AnalysisInput(stem, times_buf, zone_buf, state_buf,
                         centers_buf, head_buf, zones_full, file_meta)


# =============================================================================
# Convenience runners
# =============================================================================

_DEFAULT_PARAMS = dict(
    min_dwell_time=0.2,
    angle_threshold=45.0,
    ia_confirm_ms=100.0,
    ia_line_extend_px=15.0,
    ia_inner_offset_px=50.0,
    ia_parallel_tol_deg=15.0,
)


def analyze_session(session, *, head_idx: int = 0, body_idx: int = 1,
                    zones=None, params: Optional[Dict[str, Any]] = None,
                    min_dwell_time: float = 0.2, angle_threshold: float = 45.0,
                    ia_confirm_ms: float = 100.0, ia_line_extend_px: float = 15.0,
                    ia_inner_offset_px: float = 50.0, ia_parallel_tol_deg: float = 15.0):
    """Canonicalize + compute the (state, bin) stats DataFrame for a session."""
    ai = canonicalize_session(session, head_idx=head_idx, body_idx=body_idx, zones=zones)
    return compute_zone_stats_df(
        ai.stem, ai.times_buf, ai.zone_buf, ai.state_buf,
        zones_full=ai.zones_full, centers_buf=ai.centers_buf, head_buf=ai.head_buf,
        file_meta=ai.file_meta, min_dwell_time=min_dwell_time,
        angle_threshold=angle_threshold, ia_confirm_ms=ia_confirm_ms,
        ia_line_extend_px=ia_line_extend_px, ia_inner_offset_px=ia_inner_offset_px,
        ia_parallel_tol_deg=ia_parallel_tol_deg, params=params,
    )


def analyze_session_to_excel(session, output_dir: str, *, head_idx: int = 0,
                             body_idx: int = 1, params: Optional[Dict[str, Any]] = None,
                             **kwargs) -> Optional[str]:
    """Run the analysis and write ``<stem>_zone_stats.xlsx``. Returns the path."""
    ai = canonicalize_session(session, head_idx=head_idx, body_idx=body_idx)
    df = analyze_session(session, head_idx=head_idx, body_idx=body_idx,
                         params=params, **kwargs)
    if df is None or len(df) == 0:
        return None
    return write_zone_stats_excel(df, ai.stem, output_dir, params=params)


__all__ = [
    "AnalysisInput", "canonicalize_session", "suggest_head_body_indices",
    "analyze_session", "analyze_session_to_excel", "clean_stem",
]
