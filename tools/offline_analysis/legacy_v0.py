"""Legacy v0 ``_video_data`` support, SELF-CONTAINED and REMOVABLE.

The v0 GUI wrote a different file shape than the current rig:

  * Header lines ``B {meta}`` / ``M {zones,scale,arena}`` / ``V [y1,y2,x1,x2]``
    (crop box) / ``P {…}`` instead of ``#key value`` lines.
  * One JSON object per data row (``frames, mcu_frame, state_name,
    timestamps, speed, location, pose_array``) instead of TSV.
  * Multiple experiments concatenated (the ``frames`` counter resets).
  * No MCU-fw clock; ``timestamps`` is host ms (per segment).
  * Scale stored as ``{length(cm), values}`` (NOT a scale zone), and the
    paired video is named ``<subject>-<timestamp>.mp4``, the txt's
    ``_video_data_-`` chunk is dropped, so the current suffix-strip video
    finder can't pair it.

ALL of that lives here. ``video_data_parser`` and ``view_video`` each
dispatch to this module in one line for v0 files; deleting this file plus
those two lines drops v0 support cleanly once v0 data is retired.

Mirrors the loaders in the user's ``retracking_pyqt6_fast_v1.py``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# v1 column names, the canonical frames_df schema every analyzer view
# consumes. v0 is normalized into these so nothing downstream branches.
_V1_COLUMNS: Tuple[str, ...] = (
    "frame_num", "capture_elapsed", "mcu_fw_capture_ms", "mcu_fw_pose_ms",
    "pose", "zone", "speed_pxs", "stage", "note",
)

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# ---- detection ----------------------------------------------------------

def is_v0_file(filepath: str) -> bool:
    """True when the first non-empty line is a v0 ``B``/``M``/``V``/``P``
    header or a bare JSON object row. The current ``#``-header formats
    (v1/v2) open with a ``# ===`` banner, so they return False."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                return s[:2] in ("B ", "M ", "V ", "P ") or s.startswith("{")
    except Exception:
        return False
    return False


# ---- scale --------------------------------------------------------------

def _v0_scale_to_zone(scale: Optional[Dict]) -> Optional[Dict]:
    """Convert a v0 ``{length(cm), values}`` scale into a ``type:"scale"``
    zone (two endpoint ``points`` + ``scale_length`` in cm) so the
    analyzer's existing ``_detect_scale`` resolves it without changes.

    ``values`` forms (matching retracking_pyqt6_fast_v1._pix_per_cm...):
      * nested ``[[x1,y1],[x2,y2]]``      -> those two points
      * flat 4 ``[x1,y1,x2,y2]``          -> (x1,y1),(x2,y2)
      * flat 2 ``[x1,x2]``                -> (x1,0),(x2,0)  (horizontal bar)
    """
    if not isinstance(scale, dict):
        return None
    length = scale.get("length")
    vals = scale.get("values")
    if length is None or vals is None:
        return None
    try:
        length_cm = float(length)
    except (TypeError, ValueError):
        return None
    if length_cm <= 0:
        return None
    pts: Optional[List[List[float]]] = None
    if isinstance(vals, (list, tuple)) and len(vals) >= 2:
        try:
            if isinstance(vals[0], (list, tuple)) and len(vals[0]) >= 2:
                pts = [[float(vals[0][0]), float(vals[0][1])],
                       [float(vals[1][0]), float(vals[1][1])]]
            elif len(vals) >= 4:
                pts = [[float(vals[0]), float(vals[1])],
                       [float(vals[2]), float(vals[3])]]
            elif len(vals) == 2:
                pts = [[float(vals[0]), 0.0], [float(vals[1]), 0.0]]
        except (TypeError, ValueError):
            pts = None
    if not pts:
        return None
    if math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) <= 0:
        return None
    return {"type": "scale", "points": pts,
            "scale_length": length_cm, "scale_unit": "cm"}


def _v0_zone_config(m: Dict) -> Dict:
    """Map a v0 ``M`` block to the nested zone_config the UI expects.

    v0 stores polygon vertices under ``values`` with ``points: null``; the
    zone editor reads ``points``, so copy them across. The scale is
    injected as a synthetic ``type:"scale"`` zone (stats views skip it)."""
    zones_in = m.get("zones", {}) or {}
    zones_out: Dict[str, Dict] = {}
    if isinstance(zones_in, dict):
        for name, z in zones_in.items():
            if not isinstance(z, dict):
                continue
            zz = dict(z)
            if not zz.get("points") and zz.get("values"):
                zz["points"] = zz["values"]
            zones_out[name] = zz
    scale_zone = _v0_scale_to_zone(m.get("scale"))
    if scale_zone is not None and "scale" not in zones_out:
        zones_out["scale"] = scale_zone
    return {
        "zones": zones_out,
        "scale": m.get("scale", {}) or {},
        "arena": m.get("arena", {}) or {},
    }


# ---- parse --------------------------------------------------------------

def parse_v0(filepath: str):
    """Parse a v0 ``_video_data`` file into the same SessionData shape as
    v1/v2 so every analyzer view works unchanged."""
    from .video_data_parser import SessionData  # local import avoids cycle

    info: Dict = {"version": "0", "sync": {"source": "v0_host_ms"}}
    events: List[Dict] = []
    rows: List[Dict] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("B "):
                try:
                    b = json.loads(line[2:])
                    info["pycontrol"] = {
                        "subject": str(b.get("Subject ID", "")),
                        "task": str(b.get("Experiment", "")),
                    }
                    info["experimenter"] = b.get("Experimenter", "")
                    info["group"] = b.get("Group", "")
                    info["session_start"] = b.get("Start time", "")
                    info["_v0_meta"] = b
                except Exception as e:
                    logger.warning("v0 B-block parse error in %s: %s", filepath, e)
                continue
            if line.startswith("M "):
                try:
                    info["zone_config"] = _v0_zone_config(json.loads(line[2:]))
                except Exception as e:
                    logger.warning("v0 M-block parse error in %s: %s", filepath, e)
                continue
            if line.startswith("V "):
                try:
                    info["video_region"] = json.loads(line[2:])
                except Exception:
                    pass
                continue
            if line.startswith("P "):
                continue  # extra v0 metadata, not used by the analyzer
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
                continue

    frames_df = _build_frames_df_v0(rows, events)
    session = SessionData(filepath=filepath, info=info,
                          frames_df=frames_df, events=events)
    logger.info("Parsed %s (v0): %d frames, %d events, %d zones",
                filepath, session.n_frames, len(events), len(session.zones))
    return session


def _build_frames_df_v0(rows: List[Dict],
                        events_out: List[Dict]) -> pd.DataFrame:
    """v0 JSON rows -> frames_df under the v1 column NAMES.

    Handles segment resets (``frames`` dropping marks a new bout ->
    ``segment`` column) and derives state-change entries into
    ``events_out``. v0 has no MCU-fw clock, so ``mcu_fw_capture_ms`` holds
    the host ``timestamps`` (per-segment ms), callers read
    ``info['sync']`` to know it's not a true MCU clock.
    """
    if not rows:
        return pd.DataFrame(columns=list(_V1_COLUMNS))

    segs, frame_nums, times, speeds = [], [], [], []
    zones, states, poses, xs, ys = [], [], [], [], []
    seg = 0
    prev_frame = None
    last_state = None

    for r in rows:
        try:
            fn = int(r.get("frames", 0))
        except (TypeError, ValueError):
            fn = 0
        if prev_frame is not None and fn < prev_frame:
            seg += 1
        prev_frame = fn

        ts = r.get("timestamps")
        st_raw = r.get("state_name")
        st = ""
        if isinstance(st_raw, str):
            st = st_raw.split("State:", 1)[-1].strip() if "State:" in st_raw \
                else st_raw.strip()
        pose_arr = r.get("pose_array")
        pose_str = json.dumps(pose_arr) if pose_arr else "na"
        x = y = np.nan
        if isinstance(pose_arr, list) and pose_arr and len(pose_arr[0]) >= 2:
            try:
                x, y = float(pose_arr[0][0]), float(pose_arr[0][1])
            except (TypeError, ValueError):
                x = y = np.nan

        if st and st != last_state:
            events_out.append({
                "kind": "S", "name": st,
                "fw_ms": float(ts) if ts is not None else None,
                "frame": fn,
            })
            last_state = st

        loc = r.get("location")
        segs.append(seg)
        frame_nums.append(fn)
        times.append(ts)
        speeds.append(r.get("speed"))
        zones.append(loc if loc else "na")
        states.append(st if st else "-")
        poses.append(pose_str)
        xs.append(x)
        ys.append(y)

    df = pd.DataFrame({
        "frame_num":         frame_nums,
        "capture_elapsed":   ["" for _ in rows],
        "mcu_fw_capture_ms": times,
        "mcu_fw_pose_ms":    [np.nan] * len(rows),
        "pose":              poses,
        "zone":              zones,
        "speed_pxs":         speeds,
        "state":             states,
        "events":            ["-"] * len(rows),
        "stage":             ["na"] * len(rows),
        "note":              [""] * len(rows),
        "segment":           segs,
    })
    df["frame_num"]         = pd.to_numeric(df["frame_num"], errors="coerce").fillna(0).astype(int)
    df["mcu_fw_capture_ms"] = pd.to_numeric(df["mcu_fw_capture_ms"], errors="coerce")
    df["speed_pxs"]         = pd.to_numeric(df["speed_pxs"], errors="coerce")
    df["x"] = xs
    df["y"] = ys
    df["location"] = df["zone"].astype(str).where(df["zone"] != "na", "")
    df["frames"]     = df["frame_num"]
    df["timestamps"] = df["mcu_fw_capture_ms"].fillna(0).astype(int)
    df["speed"]      = df["speed_pxs"].fillna(0.0)
    df["frame_idx"]  = range(1, len(df) + 1)
    return df


# ---- video pairing ------------------------------------------------------

def _v0_video_stem(txt_name: str) -> Tuple[str, str]:
    """Return ``(stem, timestamp)`` for the video paired with a v0 txt.

    Port of retracking_pyqt6_fast_v1.parse_video_name_from_txt, the v0
    video drops the ``_video_data_-`` chunk:
    ``D860_video_data_-2025-07-30-113426.txt`` -> ``D860-2025-07-30-113426``.
    """
    base = os.path.basename(txt_name)
    for sep in ("_video_data_-", "_video_data-"):
        if sep in base:
            parts = base.split(sep)
            subject = parts[0]
            ts = parts[-1].rsplit(".txt", 1)[0]
            return f"{subject}-{ts}", ts
    name_no_ext = os.path.splitext(base)[0]
    if "_video_data" in name_no_ext:
        stem = name_no_ext.split("_video_data")[0].rstrip("-_")
        return stem, (stem.split("-")[-1] if "-" in stem else stem)
    if "-" in name_no_ext:
        return name_no_ext, name_no_ext.split("-")[-1]
    return name_no_ext, name_no_ext


def find_video_v0(txt_path: str) -> str:
    """Find the video paired with a v0 txt. Returns "" when none found.

    Searches the txt dir, its ``video/`` subdir, the parent, and the
    parent's ``video/``, trying exact stem, then substring (either way),
    then a timestamp match (port of find_video_path)."""
    txt_dir = os.path.dirname(txt_path)
    stem, ts = _v0_video_stem(txt_path)

    search_dirs = [
        txt_dir,
        os.path.join(txt_dir, "video"),
        os.path.dirname(txt_dir),
        os.path.join(os.path.dirname(txt_dir), "video"),
    ]
    seen = set()
    for d in search_dirs:
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        # 1. Exact stem match (both cases).
        for ext in _VIDEO_EXTS:
            for e in (ext, ext.upper()):
                p = os.path.join(d, stem + e)
                if os.path.isfile(p):
                    return p
        # 2. Substring match either direction.
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for f in names:
            if not any(f.lower().endswith(e) for e in _VIDEO_EXTS):
                continue
            b = os.path.splitext(f)[0].lower()
            if stem.lower() in b or b in stem.lower():
                return os.path.join(d, f)
        # 3. Timestamp match.
        if ts:
            for f in names:
                if not any(f.lower().endswith(e) for e in _VIDEO_EXTS):
                    continue
                if ts.lower() in f.lower():
                    return os.path.join(d, f)
    return ""


# ---- body-part names (v0 carries none, the view asks the user) ---------

# Removable legacy state: maps a v0 txt path -> the body-part names the
# user supplied for its (unnamed) pose_array points.
_BP_REGISTRY: Dict[str, List[str]] = {}


def set_body_parts(txt_path: str, names: List[str]) -> None:
    _BP_REGISTRY[os.path.abspath(txt_path)] = [str(n) for n in names]


def get_body_parts(txt_path: str) -> Optional[List[str]]:
    return _BP_REGISTRY.get(os.path.abspath(txt_path))


def peek_pose_len(txt_path: str) -> int:
    """Number of points in the first v0 ``pose_array`` (0 if none)."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if s.startswith("{"):
                    pa = json.loads(s).get("pose_array")
                    return len(pa) if isinstance(pa, list) else 0
    except Exception:
        pass
    return 0


def _v0_zone_list(m: Dict) -> List[Dict]:
    """List of zone dicts (behavioural + the scale zone) in the shape
    ``view_video`` expects (``_zones_to_list`` / ``_px_per_cm_from_zones``
    both accept a flat list with ``points`` + ``type``)."""
    out: List[Dict] = []
    zones_in = m.get("zones", {}) or {}
    if isinstance(zones_in, dict):
        for name, z in zones_in.items():
            if not isinstance(z, dict):
                continue
            pts = z.get("points") or z.get("values") or []
            out.append({"name": name, "type": z.get("type", "polygon"),
                        "points": pts, "values": pts})
    sz = _v0_scale_to_zone(m.get("scale"))
    if sz is not None:
        sz = dict(sz)
        sz["name"] = "scale"
        out.append(sz)
    return out


def load_frames_for_view(txt_path: str):
    """v0 -> ``(header, zones_config, frames_df, body_parts)`` matching
    ``view_video._load_session_frames`` so the main analysis view (scale,
    stage, body-part picker) works on v0 files.

    Pose lists are converted to ``{name: [x,y,conf]}`` dicts using the
    user-registered body-part names (positional ``bodypart_N`` fallback),
    so the body-part selector behaves as for new files. ``state_name``
    becomes the per-frame ``stage`` column.
    """
    header: Dict = {}
    zones_list: List[Dict] = []
    rows: List[Dict] = []
    video_region = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("B "):
                try:
                    header.update(json.loads(line[2:]))
                except Exception:
                    pass
                continue
            if line.startswith("M "):
                try:
                    zones_list = _v0_zone_list(json.loads(line[2:]))
                except Exception:
                    pass
                continue
            if line.startswith("V "):
                try:
                    video_region = json.loads(line[2:])
                except Exception:
                    pass
                continue
            if line.startswith("P "):
                continue
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
                continue

    npts = 0
    for r in rows:
        pa = r.get("pose_array")
        if isinstance(pa, list):
            npts = max(npts, len(pa))
    names = get_body_parts(txt_path) or [f"bodypart_{i + 1}" for i in range(npts)]

    frame_rows: List[Dict] = []
    for r in rows:
        try:
            ts = float(r.get("timestamps") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        pa = r.get("pose_array")
        pose = None
        if isinstance(pa, list) and pa:
            pose = {}
            for i, pt in enumerate(pa):
                nm = names[i] if i < len(names) else f"bodypart_{i + 1}"
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pose[nm] = [float(pt[0]), float(pt[1]),
                                float(pt[2]) if len(pt) > 2 else 1.0]
        st_raw = r.get("state_name")
        stage = ""
        if isinstance(st_raw, str):
            stage = st_raw.split("State:", 1)[-1].strip() if "State:" in st_raw \
                else st_raw.strip()
        try:
            spd = float(r.get("speed") or 0.0)
        except (TypeError, ValueError):
            spd = 0.0
        frame_rows.append({
            "frame": int(r.get("frames") or 0),
            "desired_ts": ts, "actual_ts": ts, "timestamp_ms": ts,
            "speed": spd, "location": r.get("location") or "",
            "pose": pose, "stage": stage,
        })

    df = pd.DataFrame(frame_rows) if frame_rows else pd.DataFrame(
        columns=["frame", "desired_ts", "actual_ts", "timestamp_ms",
                 "speed", "location", "pose", "stage"])

    header["body_parts"] = names
    if isinstance(video_region, list) and len(video_region) == 4:
        y1, y2, x1, x2 = video_region
        header.setdefault("resolution",
                          f"{int(abs(x2 - x1))}x{int(abs(y2 - y1))}")
    if not header.get("stage"):
        stages = [fr["stage"] for fr in frame_rows if fr["stage"]]
        if stages:
            header["stage"] = stages[0]
    return header, zones_list, df, names


__all__ = [
    "is_v0_file", "parse_v0", "find_video_v0",
    "set_body_parts", "get_body_parts", "peek_pose_len",
    "load_frames_for_view",
]
