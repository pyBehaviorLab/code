"""Reader for the arena recorder's ``_video_data.txt``, a third on-disk shape.

Self-contained and removable, like :mod:`legacy_v0`: ``parse_video_data``
sniffs and delegates here, and nothing in the current pipeline's path changes.

The format differs from v1/v2/v3 in four ways that each break the main reader,
which is why it needs its own function rather than another branch:

1. Header rows are TAB-separated ``<t>\\tinfo\\t<key>\\t<value>`` lines with no
   leading ``#``, so the main reader consumes them as data rows.
2. The column header is ``# frame_number\\tcam_frame_id\\t…`` (hash-SPACE), not
   ``#columns …``. The main reader files it away as an info key called
   ``frame_number`` and never learns the real column names.
3. ``pose_array`` is a JSON OBJECT keyed by body part, ``{"Head": [x, y, c]}``,
   where every other format writes a list. ``analysis_input._parse_pose``
   returns None for a dict, so centroids, head positions and zone assignment
   would all come back empty even once the file parsed.
4. ``actual_ts`` is already milliseconds. ``_elapsed_to_ms`` reads a bare
   number as SECONDS, so routing it through ``capture_elapsed`` would make
   every time bin 1000x too long. It is written as ``MM:SS.mmm`` here so the
   existing parser produces the right milliseconds.

Output uses the canonical v1 column names, so every downstream readout,
exporter and viewer works unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

#: The column header this format uses. Hash, whitespace, then tab-separated
#: names starting with the two that identify it.
_COLS_RE = re.compile(r"^#\s*frame_number\b")

#: ``<seconds>\tinfo\t<key>\t<value>``, the header rows.
_INFO_RE = re.compile(r"^[\d.]+\t(info|print|variable)\t([^\t]*)\t?(.*)$")

#: How many lines to read when sniffing. The column header sits after the
#: info block, which is a dozen or so lines in the files seen so far; 200 is
#: generous without reading a 100 MB session to answer "is this mine?".
_SNIFF_LINES = 200


def is_arena_file(filepath: str) -> bool:
    """True when this file is the arena recorder's shape.

    Keyed on the column header, which is the one line no other format has.
    Deliberately NOT keyed on the filename: names in this project carry
    underscores and a ddmmyyhhmmss stamp, but that is a convention, and a
    convention is a weaker signal than the header the writer actually emits.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= _SNIFF_LINES:
                    return False
                if _COLS_RE.match(line):
                    return "cam_frame_id" in line
        return False
    except OSError:
        return False


def _ms_to_elapsed(ms: float) -> str:
    """Milliseconds -> ``MM:SS.mmm``, the shape ``_elapsed_to_ms`` expects.

    Written rather than passing the raw number because that helper multiplies
    a bare value by 1000, reading it as seconds.
    """
    try:
        total = float(ms)
    except (TypeError, ValueError):
        return "na"
    if total != total:                        # NaN
        return "na"
    total = max(total, 0.0)
    mm = int(total // 60000)
    rest = (total - mm * 60000) / 1000.0
    return f"{mm:02d}:{rest:06.3f}"


def _pose_dict_to_list(cell: str, order: List[str]) -> str:
    """``{"Head": [x, y, c], ...}`` -> ``[[x, y, c], ...]`` in body-part order.

    Order comes from the file's own ``body_parts`` header, so index 0 is the
    part the analysis calls head/body by index, exactly as in every other
    format. Without this the analysis sees a dict, gives up, and reports no
    position for any frame.
    """
    s = (cell or "").strip()
    if not s or s in ("none", "na", "-", "nan", "None"):
        return "na"
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return "na"
    if isinstance(obj, list):
        return s                              # already the canonical shape
    if not isinstance(obj, dict):
        return "na"
    names = order or list(obj.keys())
    out: List[List[float]] = []
    for name in names:
        pt = obj.get(name)
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                out.append([float(pt[0]), float(pt[1]),
                            float(pt[2]) if len(pt) > 2 else 1.0])
                continue
            except (TypeError, ValueError):
                pass
        out.append([float("nan"), float("nan"), 0.0])
    return json.dumps(out)


def _read(filepath: str) -> Tuple[Dict, List[str], List[List[str]]]:
    info: Dict = {}
    col_names: List[str] = []
    rows: List[List[str]] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if col_names and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) < len(col_names):
                    parts += ["na"] * (len(col_names) - len(parts))
                rows.append(parts[:len(col_names)])
                continue
            if _COLS_RE.match(line):
                col_names = [c.strip() for c in
                             line.lstrip("#").strip().split("\t") if c.strip()]
                continue
            if line.startswith("Zones\t"):
                try:
                    info["zones_raw"] = json.loads(line.split("\t", 1)[1])
                except (ValueError, IndexError):
                    logger.warning("arena reader: unreadable Zones line")
                continue
            m = _INFO_RE.match(line)
            if m and m.group(1) == "info":
                key, val = m.group(2).strip(), m.group(3).strip()
                if val.startswith("{") or val.startswith("["):
                    try:
                        info[key] = json.loads(val)
                        continue
                    except ValueError:
                        pass
                info[key] = val
    return info, col_names, rows


def parse_arena(filepath: str):
    """Parse an arena ``_video_data.txt`` into the shared ``SessionData``."""
    from .video_data_parser import SessionData          # avoid a cycle

    info, col_names, rows = _read(filepath)
    body_parts = info.get("body_parts") or []
    if isinstance(body_parts, str):
        try:
            body_parts = json.loads(body_parts)
        except ValueError:
            body_parts = [p.strip() for p in body_parts.split(",") if p.strip()]

    # Surface the header under the keys SessionData's properties read, so
    # resolution / fps / zones behave as they do for every other format.
    cam = {}
    if info.get("resolution"):
        cam["resolution"] = info["resolution"]
    if info.get("target_fps"):
        cam["fps"] = info["target_fps"]
    if cam:
        info.setdefault("camera", cam)
    if "zones_raw" in info:
        info.setdefault("zone_config", {"zones": info["zones_raw"]})
    info.setdefault("body_parts", body_parts)
    info.setdefault("version", "arena")

    if not rows or not col_names:
        logger.warning("arena reader: %s has no data rows", filepath)
        return SessionData(filepath=filepath, info=info,
                           frames_df=pd.DataFrame(), events=[])

    src = pd.DataFrame(rows, columns=col_names)

    def num(name):
        return (pd.to_numeric(src[name], errors="coerce")
                if name in src.columns else pd.Series([float("nan")] * len(src)))

    ts_ms = num("actual_ts")
    df = pd.DataFrame()
    df["frame_num"] = num("frame_number").fillna(0).astype(int)
    df["capture_elapsed"] = [_ms_to_elapsed(v) for v in ts_ms]
    # No MCU clock exists in this format. Left empty rather than filled with
    # the host clock, so nothing downstream mistakes one for the other;
    # _times_ms falls through to capture_elapsed, which is what carries it.
    df["mcu_fw_capture_ms"] = float("nan")
    df["mcu_fw_pose_ms"] = float("nan")
    df["pose"] = [_pose_dict_to_list(v, body_parts)
                  for v in (src["pose_array"] if "pose_array" in src.columns
                            else [""] * len(src))]
    loc = (src["location"].astype(str) if "location" in src.columns
           else pd.Series([""] * len(src)))
    df["zone"] = loc.where(~loc.isin(["none", "na", "-", "nan", "None"]), "na")
    df["location"] = df["zone"].where(df["zone"] != "na", "")
    df["speed_pxs"] = num("speed")
    df["stage"] = (src["stage"].astype(str) if "stage" in src.columns
                   else pd.Series([""] * len(src)))
    df["note"] = ""

    # This format's own columns, kept so they are not silently lost.
    for extra in ("cam_frame_id", "pose_cam_frame_id", "pose_lat_frames",
                  "pose_age_ms"):
        if extra in src.columns:
            df[extra] = num(extra)

    # v1 aliases.
    df["frames"] = df["frame_num"]
    df["timestamps"] = ts_ms.fillna(0).astype("int64")
    df["speed"] = df["speed_pxs"].fillna(0.0)

    session = SessionData(filepath=filepath, info=info, frames_df=df, events=[])
    logger.info("Parsed %s (arena): %d frames, %d zones, body_parts=%s",
                os.path.basename(filepath), len(df),
                len(session.zones), body_parts)
    return session
