r"""
Full video zone-statistics analysis, ported VERBATIM from
``retracking_pyqt6_fast_v1_2.py`` (the analysis half only; no Qt, no
multiprocessing, no DLC inference).

The script it was ported from is not in this repo. It is archived at
``D:\cKet_retrack\OldScrit_tracking\retracking_pyqt6_fast_v1_2 - from
pyBehaviorLab repo.py``, note the suffix: the plain ``_v1_2.py`` beside it
in that folder is an earlier, shorter file and is NOT the one this was ported
from.

This is the single source of truth for the *proper* analysis: frames are
grouped by state, each state is split into fixed-length time bins, and one row
is emitted per ``(state, bin)`` with REGIONS_HIERARCHY-aggregated body/head/angle
metrics, per-region distance, immobile time, and the full IA facing machinery.

The format readers (``legacy_v0`` for old ``B/M/V`` files, ``video_data_parser``
for new ``#``-header files) feed this through ``analysis_input.canonicalize_session``
so the analysis itself never branches on file format.

Do not add format-specific logic here. Keep this module byte-aligned with the
retracking implementation so numbers match a retracking run on the same file.
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import cv2
    _CV2_OK = True
except Exception:  # pragma: no cover - cv2 is a core project dep
    _CV2_OK = False


# =============================================================================
# REGIONS HIERARCHY - Maps main regions to sub-zones for Excel aggregation
# =============================================================================

REGIONS_HIERARCHY = {
    'CenterZone': [
        'Social_Buffer', 'Social_Open_Corner', 'Familiar_Social_Corner', 'Open_Buffer',
        'Open_Novel_Corner', 'Novel_Buffer', 'Novel_Object1_Corner', 'Object1_Buffer',
        'Object1_Object2_Corner', 'Object2_Buffer', 'Object2_Familiar_Corner', 'Familiar_Buffer'
    ],
    'Center': ['CenterZone'],
    'Social_IA': ['Social_IA'],
    'Social_Entry': ['Social_Entry'],
    'OpenEntry': ['OpenEntry'],
    'OpenArm': ['OpenArm'],
    'NovelArm': ['NovelArm'],
    'Object1_IA': ['Object1_IA'],
    'Object1_Entry': ['Object1_Entry'],
    'Object2_IA': ['Object2_IA'],
    'Object2_Entry': ['Object2_Entry'],
    'FamiliarArm': ['FamiliarArm']
}


# =============================================================================
# METADATA PARSING
# =============================================================================

def parse_metadata_from_filename(stem: str) -> Dict[str, str]:
    """Parse metadata from filename (group, mouseID, expt_date, time)."""
    import re

    result = {'group': '', 'mouseID': '', 'expt_date': '', 'time': ''}

    parts = [p for p in stem.replace('-', '_').split('_') if p]

    date_pattern = re.compile(r'(\d{4}[-_]?\d{2}[-_]?\d{2})|(\d{2}[-_]?\d{2}[-_]?\d{4})')
    time_pattern = re.compile(r'(\d{2}[-_:]?\d{2}[-_:]?\d{2})')
    mouse_pattern = re.compile(r'(mouse|m|M)[-_]?(\d+)', re.IGNORECASE)
    simple_mouse_pattern = re.compile(r'^[A-Za-z]*\d+$')

    for part in parts:
        if date_pattern.match(part):
            result['expt_date'] = part
        elif time_pattern.match(part) and len(part) >= 6 and not date_pattern.match(part):
            result['time'] = part
        elif mouse_pattern.match(part):
            result['mouseID'] = part
        elif simple_mouse_pattern.match(part) and not result['mouseID']:
            result['mouseID'] = part

    for i in range(len(parts) - 2):
        if (re.fullmatch(r'\d{4}', parts[i]) and
            re.fullmatch(r'\d{2}', parts[i + 1]) and
            re.fullmatch(r'\d{2}', parts[i + 2])):
            result['expt_date'] = f"{parts[i]}-{parts[i + 1]}-{parts[i + 2]}"
            break

    if not result['group'] and len(parts) >= 1:
        ignore = {'video', 'data', 'videodata'}
        for part in parts:
            if part == result['mouseID']:
                continue
            if date_pattern.match(part) or time_pattern.match(part):
                continue
            if part.lower() in ignore:
                continue
            result['group'] = part
            break
    if not result['mouseID'] and len(parts) >= 2:
        result['mouseID'] = parts[1]
    if not result['expt_date'] and len(parts) >= 3:
        result['expt_date'] = parts[2]
    if not result['time'] and len(parts) >= 4:
        result['time'] = parts[3]

    return result


def parse_metadata_from_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Parse metadata from B-line meta dict."""
    import re

    result = {
        'group': '', 'mouseID': '', 'expt_date': '', 'time': '',
        'start_time': '', 'subgroup': ''
    }
    if not meta:
        return result

    def pick(keys):
        for k in keys:
            v = meta.get(k)
            if v:
                return str(v)
        return ''

    result['mouseID'] = pick(['Subject ID', 'SubjectID', 'Mouse ID', 'MouseID', 'ID'])
    result['group'] = pick(['Group'])
    result['subgroup'] = pick(['Sub Group', 'SubGroup'])

    start_time = pick(['Start time', 'Start_time', 'StartTime', 'DateTime', 'Date'])
    result['start_time'] = start_time
    if start_time:
        date_match = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', start_time)
        time_match = re.search(r'(\d{2})[:\-](\d{2})[:\-](\d{2})', start_time)
        if date_match:
            result['expt_date'] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        if time_match:
            result['time'] = f"{time_match.group(1)}:{time_match.group(2)}:{time_match.group(3)}"

    return result


def merge_metadata(primary: Dict[str, str], fallback: Dict[str, str]) -> Dict[str, str]:
    """Prefer primary values, fill missing from fallback."""
    merged = dict(primary)
    for key in ('group', 'mouseID', 'expt_date', 'time', 'start_time', 'subgroup'):
        if not merged.get(key):
            merged[key] = fallback.get(key, '')
    return merged


def _pix_per_cm_from_scale_info(scale_info: Any) -> Optional[float]:
    if scale_info is None:
        return None
    if isinstance(scale_info, (int, float)):
        return float(scale_info)
    if isinstance(scale_info, dict):
        for key in ('pix_per_cm', 'pixels_per_cm', 'px_per_cm', 'pixpercm'):
            if key in scale_info:
                try:
                    return float(scale_info[key])
                except Exception:
                    pass
        if 'length' in scale_info and 'values' in scale_info:
            try:
                length_cm = float(scale_info.get('length', 0))
                vals = scale_info.get('values')
                if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                    if isinstance(vals[0], (list, tuple)) and len(vals[0]) >= 2:
                        x1, y1 = float(vals[0][0]), float(vals[0][1])
                        x2, y2 = float(vals[1][0]), float(vals[1][1])
                    elif len(vals) >= 4:
                        x1, y1, x2, y2 = map(float, vals[:4])
                    elif len(vals) == 2:
                        x1, x2 = map(float, vals[:2])
                        y1 = y2 = 0.0
                    else:
                        return None
                    pix_dist = math.hypot(x2 - x1, y2 - y1)
                    if pix_dist > 0 and length_cm > 0:
                        return pix_dist / length_cm
            except Exception:
                return None
    if isinstance(scale_info, (list, tuple)) and len(scale_info) >= 4:
        try:
            x1, y1, x2, y2 = map(float, scale_info[:4])
            length_cm = 22.0
            pix_dist = math.hypot(x2 - x1, y2 - y1)
            if pix_dist > 0:
                return pix_dist / length_cm
        except Exception:
            return None
    return None


def extract_scale_m_per_px(zones_full: Optional[Dict], file_meta: Optional[Dict[str, Any]]) -> float:
    """Return meters-per-pixel based on scale info when available."""
    pix_per_cm = None
    if zones_full:
        try:
            scale_info = zones_full.get('scale') or zones_full.get('Scale')
            pix_per_cm = _pix_per_cm_from_scale_info(scale_info)
        except Exception:
            pix_per_cm = None
    if pix_per_cm is None and file_meta:
        for key in ('scale', 'Scale', 'pix_per_cm', 'pixels_per_cm', 'px_per_cm'):
            if key in file_meta:
                pix_per_cm = _pix_per_cm_from_scale_info(file_meta.get(key))
                if pix_per_cm:
                    break
    if pix_per_cm and pix_per_cm > 0:
        return 0.01 / float(pix_per_cm)
    return 1.0


# =============================================================================
# IA ZONE GEOMETRY & ANGLE CALCULATION
# =============================================================================

def get_zone_center(zone_pts: List) -> Tuple[float, float]:
    if not zone_pts or len(zone_pts) < 3:
        return (0.0, 0.0)
    pts = np.array(zone_pts, dtype=np.float32)
    return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))


def get_arena_center(zroot: Dict) -> Tuple[float, float]:
    if not zroot:
        return (0.0, 0.0)
    centers = []
    for z in zroot.values():
        pts = z.get('values', [])
        if len(pts) >= 3:
            centers.append(get_zone_center(pts))
    if not centers:
        return (0.0, 0.0)
    arr = np.array(centers, dtype=np.float32)
    return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))


def get_outermost_edge_line(zone_pts: List,
                            reference_center: Tuple[float, float],
                            parallel_tol_deg: float = 15.0) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not zone_pts or len(zone_pts) < 2:
        return None

    pts = np.array(zone_pts, dtype=np.float32)
    if len(pts) < 2:
        return None

    rx, ry = reference_center
    edges = []
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
        mx = 0.5 * (p1[0] + p2[0])
        my = 0.5 * (p1[1] + p2[1])
        dist = float((mx - rx) ** 2 + (my - ry) ** 2)
        edges.append({
            'p1': (float(p1[0]), float(p1[1])),
            'p2': (float(p2[0]), float(p2[1])),
            'angle': angle,
            'length': length,
            'dist': dist
        })

    if not edges:
        return None

    groups = []
    for e in edges:
        placed = False
        for g in groups:
            if abs(e['angle'] - g['angle']) <= parallel_tol_deg:
                g['edges'].append(e)
                g['max_len'] = max(g['max_len'], e['length'])
                placed = True
                break
        if not placed:
            groups.append({'angle': e['angle'], 'edges': [e], 'max_len': e['length']})

    groups.sort(key=lambda g: g['max_len'], reverse=True)
    best_group = groups[0]

    best_edge = max(best_group['edges'], key=lambda e: e['dist'])
    return best_edge['p1'], best_edge['p2']


def extend_line(p1: Tuple[float, float], p2: Tuple[float, float], extend_px: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return p1, p2
    ux = dx / length
    uy = dy / length
    return (x1 - ux * extend_px, y1 - uy * extend_px), (x2 + ux * extend_px, y2 + uy * extend_px)


def build_ia_reference_lines(zone_pts: List,
                             arena_center: Tuple[float, float],
                             line_extend_px: float = 15.0,
                             inner_offset_px: float = 50.0,
                             parallel_tol_deg: float = 15.0
                             ) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
                                        Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
                                        Optional[Tuple[float, float]]]:
    edge = get_outermost_edge_line(zone_pts, arena_center, parallel_tol_deg=parallel_tol_deg)
    if not edge:
        return None, None, None

    outer_line = extend_line(edge[0], edge[1], line_extend_px)
    (x1, y1), (x2, y2) = outer_line
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return outer_line, None, None

    nx = -dy / length
    ny = dx / length
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)
    vx = arena_center[0] - mx
    vy = arena_center[1] - my
    if nx * vx + ny * vy < 0:
        nx *= -1.0
        ny *= -1.0

    inner_line = ((x1 + nx * inner_offset_px, y1 + ny * inner_offset_px),
                  (x2 + nx * inner_offset_px, y2 + ny * inner_offset_px))
    return outer_line, inner_line, (nx, ny)


def get_zone_ia_inner_offset(zone_data: Dict, default_px: float) -> float:
    try:
        return float(zone_data.get('ia_inner_offset_px', default_px))
    except Exception:
        return float(default_px)


def get_zone_ia_line_extend(zone_data: Dict, default_px: float) -> float:
    try:
        return float(zone_data.get('ia_line_extend_px', default_px))
    except Exception:
        return float(default_px)


def point_is_past_line(point: Tuple[float, float],
                       line_p1: Tuple[float, float],
                       normal_unit: Tuple[float, float],
                       toward_center: bool = True) -> bool:
    px, py = point
    x1, y1 = line_p1
    nx, ny = normal_unit
    side = (px - x1) * nx + (py - y1) * ny
    return side >= 0.0 if toward_center else side <= 0.0


def compute_facing_angle_to_point(head_xy: Tuple[float, float], body_xy: Tuple[float, float],
                                  target_xy: Tuple[float, float]) -> Optional[float]:
    if head_xy is None or body_xy is None or target_xy is None:
        return None
    hx, hy = head_xy
    bx, by = body_xy
    tx, ty = target_xy
    if not all(np.isfinite([hx, hy, bx, by, tx, ty])):
        return None

    vx = hx - bx
    vy = hy - by
    vmag = np.hypot(vx, vy)
    if vmag < 1e-6:
        return None
    vx /= vmag
    vy /= vmag

    ux = tx - hx
    uy = ty - hy
    umag = np.hypot(ux, uy)
    if umag < 1e-6:
        return None
    ux /= umag
    uy /= umag

    dot = np.clip(vx * ux + vy * uy, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def closest_point_on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float]]:
    ex = x2 - x1
    ey = y2 - y1
    denom = ex * ex + ey * ey
    if denom < 1e-6:
        return None
    t = ((px - x1) * ex + (py - y1) * ey) / denom
    t = max(0.0, min(1.0, t))
    return (x1 + t * ex, y1 + t * ey)


def compute_facing_angle_to_edge(head_xy: Tuple[float, float], body_xy: Tuple[float, float],
                                 edge_p1: Tuple[float, float], edge_p2: Tuple[float, float]) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    if head_xy is None or body_xy is None or edge_p1 is None or edge_p2 is None:
        return None, None
    hx, hy = head_xy
    if not all(np.isfinite([hx, hy, edge_p1[0], edge_p1[1], edge_p2[0], edge_p2[1]])):
        return None, None

    cp = closest_point_on_segment(hx, hy, edge_p1[0], edge_p1[1], edge_p2[0], edge_p2[1])
    if cp is None:
        return None, None

    angle = compute_facing_angle_to_point(head_xy, body_xy, cp)
    return angle, cp


def get_target_zone_name(ia_zone_name: str) -> str:
    if '_IA' in ia_zone_name:
        return ia_zone_name.replace('_IA', '')
    return ia_zone_name


def _zone_name_tokens(name: str) -> List[str]:
    parts = [p for p in name.replace('-', '_').split('_') if p]
    return [p.lower() for p in parts if p.upper() != 'IA']


def resolve_ia_target_centroid(zroot: Dict, ia_zone_name: str) -> Optional[Tuple[float, float]]:
    if not zroot or ia_zone_name not in zroot:
        return None

    ia_pts = zroot.get(ia_zone_name, {}).get('values', [])
    if len(ia_pts) < 3:
        return None
    ia_centroid = get_zone_center(ia_pts)

    target_name = get_target_zone_name(ia_zone_name)
    if target_name in zroot:
        return get_zone_center(zroot[target_name].get('values', []))

    ia_tokens = _zone_name_tokens(ia_zone_name)
    candidates = []
    for zn, zdata in zroot.items():
        if '_IA' in zn or zn == ia_zone_name:
            continue
        zn_tokens = _zone_name_tokens(zn)
        if all(t in zn_tokens for t in ia_tokens):
            candidates.append(zn)

    non_ia = [zn for zn in zroot.keys() if '_IA' not in zn and zn != ia_zone_name]
    if not candidates and not non_ia:
        return ia_centroid

    def dist_to_ia(zn: str) -> float:
        pts = zroot.get(zn, {}).get('values', [])
        if len(pts) < 3:
            return float('inf')
        c = get_zone_center(pts)
        dx = c[0] - ia_centroid[0]
        dy = c[1] - ia_centroid[1]
        return dx * dx + dy * dy

    if candidates:
        best = min(candidates, key=dist_to_ia)
        return get_zone_center(zroot[best].get('values', []))

    best = min(non_ia, key=dist_to_ia)
    return get_zone_center(zroot[best].get('values', []))


def count_zone_entries_with_dwell(in_zone_array: np.ndarray, time_intervals: np.ndarray,
                                   min_dwell_time: float = 0.2) -> int:
    if len(in_zone_array) < 2:
        return 0

    entries = 0
    in_zone = False
    dwell_time = 0.0
    entry_valid = False

    for i in range(len(in_zone_array)):
        if in_zone_array[i]:
            if not in_zone:
                in_zone = True
                dwell_time = time_intervals[i] if i < len(time_intervals) else 0.0
                entry_valid = False
            else:
                dwell_time += time_intervals[i] if i < len(time_intervals) else 0.0
                if dwell_time >= min_dwell_time and not entry_valid:
                    entries += 1
                    entry_valid = True
        else:
            in_zone = False
            dwell_time = 0.0
            entry_valid = False

    return entries


# =============================================================================
# ZONE ASSIGNMENT
# =============================================================================

def zones_root(z: Dict) -> Dict:
    return z.get('zones', z)


def _point_in_polygon(px, py, polygon_pts) -> bool:
    """Ray-casting fallback when cv2 is unavailable."""
    n = len(polygon_pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_pts[i][0], polygon_pts[i][1]
        xj, yj = polygon_pts[j][0], polygon_pts[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class FastZoneAssigner:
    """Optimized zone assignment with spatial indexing (cv2 with pure-Python fallback)."""

    def __init__(self, zones_full: Dict):
        self.zone_names: List[str] = []
        self.zone_polys: List[np.ndarray] = []
        self.zone_bounds: List[List[float]] = []

        zroot = zones_root(zones_full)
        for name, z in zroot.items():
            pts = z.get('values', [])
            if len(pts) >= 3:
                arr = np.array(pts, dtype=np.float32)
                self.zone_names.append(name)
                self.zone_polys.append(arr)
                self.zone_bounds.append([
                    float(np.min(arr[:, 0])), float(np.min(arr[:, 1])),
                    float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))
                ])

        self.bounds_arr = np.array(self.zone_bounds, dtype=np.float32) if self.zone_bounds else np.zeros((0, 4))

    def locate(self, x: float, y: float) -> Optional[str]:
        if np.isnan(x) or np.isnan(y):
            return None

        for i, (name, poly) in enumerate(zip(self.zone_names, self.zone_polys)):
            if not (self.bounds_arr[i, 0] <= x <= self.bounds_arr[i, 2] and
                    self.bounds_arr[i, 1] <= y <= self.bounds_arr[i, 3]):
                continue

            if _CV2_OK:
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    return name
            else:
                if _point_in_polygon(float(x), float(y), poly):
                    return name

        return None

    def locate_batch(self, centers: List[Tuple[float, float]]) -> List[Optional[str]]:
        return [self.locate(c[0], c[1]) if c else None for c in centers]


# =============================================================================
# TIME-BIN HELPERS
# =============================================================================

def _format_bin_label(idx: int, bin_sec: float) -> str:
    """Human-readable label for a time bin (e.g. '0-1min' or '0-30s')."""
    lo = idx * bin_sec
    hi = (idx + 1) * bin_sec
    if bin_sec >= 60 and abs(bin_sec % 60) < 1e-9:
        return f'{int(round(lo / 60))}-{int(round(hi / 60))}min'
    return f'{lo:g}-{hi:g}s'


def split_frames_into_time_bins(frames: List[Dict], bin_sec: float) -> List[Tuple[str, List[Dict]]]:
    """Split a state's frames into fixed-length time bins.

    Each frame is expected to carry a 't_adj' timestamp (ms, monotonic, set by
    the caller). Bin index is computed from elapsed time since the first frame
    of the state, so binning is relative to the start of each state/stage.
    """
    if not bin_sec or bin_sec <= 0 or not frames:
        return [('', frames)]
    bin_ms = bin_sec * 1000.0
    start = frames[0].get('t_adj', frames[0].get('timestamp', 0.0))
    bins: Dict[int, List[Dict]] = {}
    for f in frames:
        t = f.get('t_adj', f.get('timestamp', 0.0))
        idx = int((t - start) // bin_ms)
        if idx < 0:
            idx = 0
        bins.setdefault(idx, []).append(f)
    return [(_format_bin_label(idx, bin_sec), bins[idx]) for idx in sorted(bins)]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def compute_zone_stats_df(stem: str, times_buf: List[float],
                          zone_buf: List[Optional[str]], state_buf: List[Optional[str]],
                          zones_full: Optional[Dict] = None,
                          centers_buf: List = None, head_buf: List = None,
                          min_dwell_time: float = 0.2,
                          zone_assigner: 'FastZoneAssigner' = None,
                          file_meta: Optional[Dict[str, Any]] = None,
                          angle_threshold: float = 45.0,
                          ia_confirm_ms: float = 100.0,
                          ia_line_extend_px: float = 15.0,
                          ia_inner_offset_px: float = 50.0,
                          ia_parallel_tol_deg: float = 15.0,
                          params: Optional[Dict[str, Any]] = None):
    """
    Calculate zone duration statistics with angle time for IA zones, returning
    the ordered results DataFrame (or None). One row per (state, time-bin).

    Uses REGIONS_HIERARCHY for aggregation. Angle time = time IN zone AND facing
    zone edge. Includes head entries/duration and metadata columns. This is the
    pure-compute entry point used by both the Analyze UI (renders the frame) and
    :func:`calculate_zone_statistics_full` (writes the frame to Excel).
    """
    try:
        import pandas as pd

        if not times_buf or len(times_buf) < 2:
            return None

        metadata_from_meta = parse_metadata_from_meta(file_meta)
        metadata_from_name = parse_metadata_from_filename(stem)
        metadata = merge_metadata(metadata_from_meta, metadata_from_name)
        scale_m_per_px = extract_scale_m_per_px(zones_full, file_meta)

        n_frames = len(times_buf)
        zroot = zones_full.get('zones', zones_full) if zones_full else {}
        zone_names = list(zroot.keys()) if zroot else []

        if zone_assigner is None and zones_full:
            zone_assigner = FastZoneAssigner(zones_full)

        frame_data = []
        for i in range(n_frames):
            head = head_buf[i] if head_buf and i < len(head_buf) else None
            head_zone = None
            if head and zone_assigner:
                hx, hy = head[0], head[1]
                if np.isfinite(hx) and np.isfinite(hy):
                    head_zone = zone_assigner.locate(hx, hy)

            fd = {
                'timestamp': times_buf[i],
                'zone': zone_buf[i] if i < len(zone_buf) else None,
                'head_zone': head_zone,
                'state': state_buf[i] if i < len(state_buf) else None,
                'center': centers_buf[i] if centers_buf and i < len(centers_buf) else None,
                'head': head,
            }
            frame_data.append(fd)

        # Calculate time intervals (dt) in seconds; unwrap timestamps on reset.
        raw_ts = []
        for f in frame_data:
            try:
                raw_ts.append(float(f.get('timestamp', 0)))
            except Exception:
                raw_ts.append(0.0)

        pos_diffs = [raw_ts[i] - raw_ts[i - 1] for i in range(1, len(raw_ts)) if raw_ts[i] > raw_ts[i - 1]]
        median_dt_ms = float(np.median(pos_diffs)) if pos_diffs else 0.0

        offset = 0.0
        prev_adj = None
        for i, f in enumerate(frame_data):
            raw = raw_ts[i]
            if prev_adj is None:
                adj = raw
                dt = 0.0
            else:
                if raw + offset < prev_adj:
                    offset += (prev_adj - (raw + offset)) + (median_dt_ms if median_dt_ms > 0 else 0.0)
                adj = raw + offset
                dt = (adj - prev_adj) / 1000.0
                if dt < 0:
                    dt = 0.0
            f['dt'] = dt
            f['t_adj'] = adj
            prev_adj = adj

        zone_centroids = {}
        for zn in zone_names:
            zone_centroids[zn] = get_zone_center(zroot[zn].get('values', []))
        arena_center = get_arena_center(zroot)
        ia_edge_lines = {}
        ia_inner_lines = {}
        ia_line_normals = {}
        for zn in zone_names:
            if '_IA' in zn:
                zdata = zroot.get(zn, {})
                ia_offset = get_zone_ia_inner_offset(zdata, ia_inner_offset_px)
                ia_extend = get_zone_ia_line_extend(zdata, ia_line_extend_px)
                outer_line, inner_line, normal = build_ia_reference_lines(
                    zroot[zn].get('values', []),
                    arena_center,
                    line_extend_px=ia_extend,
                    inner_offset_px=ia_offset,
                    parallel_tol_deg=ia_parallel_tol_deg
                )
                ia_edge_lines[zn] = outer_line
                ia_inner_lines[zn] = inner_line
                ia_line_normals[zn] = normal

        time_bin_sec = 0.0
        if params and params.get('time_bin_enabled'):
            try:
                time_bin_sec = max(float(params.get('time_bin_min', 0.0)), 0.0) * 60.0
            except (TypeError, ValueError):
                time_bin_sec = 0.0

        # Group by state
        state_groups = {}
        for fd in frame_data:
            state = fd.get('state') or 'unknown'
            if state not in state_groups:
                state_groups[state] = []
            state_groups[state].append(fd)

        iter_groups = []
        for state_name, state_frames in state_groups.items():
            if not state_frames or state_name == 'unknown':
                continue
            for bin_label, bin_frames in split_frames_into_time_bins(state_frames, time_bin_sec):
                if bin_frames:
                    iter_groups.append((state_name, bin_label, bin_frames))

        rows = []

        def _sum_distance_in_mask(distances_m: np.ndarray, mask: np.ndarray) -> float:
            if distances_m.size < 2:
                return 0.0
            mask = mask.astype(bool)
            mask_shift = mask & np.roll(mask, 1)
            mask_shift[0] = False
            return float(np.nansum(distances_m[mask_shift]))

        for state_name, bin_label, frames in iter_groups:

            time_intervals = np.array([f.get('dt', 0) for f in frames], dtype=np.float64)
            total_duration = float(np.sum(time_intervals))

            centers = [f.get('center') for f in frames]
            coords_px = np.array(
                [(c[0], c[1]) if c else (np.nan, np.nan) for c in centers],
                dtype=np.float64
            )
            coords_m = coords_px * float(scale_m_per_px)
            dx = np.diff(coords_m[:, 0], prepend=np.nan)
            dy = np.diff(coords_m[:, 1], prepend=np.nan)
            distances_m = np.sqrt(dx * dx + dy * dy)
            total_distance_m = float(np.nansum(distances_m))
            speeds_mps = np.divide(
                distances_m, time_intervals,
                out=np.full_like(distances_m, np.nan),
                where=time_intervals > 0
            )
            max_speed_mps = float(np.nanmax(speeds_mps)) if np.any(np.isfinite(speeds_mps)) else 0.0
            mean_speed_mps = float(np.nanmean(speeds_mps)) if np.any(np.isfinite(speeds_mps)) else 0.0
            immobile_mask = (speeds_mps < 0.01) & np.isfinite(speeds_mps) & (time_intervals > 0)
            immobile_time_s = float(np.sum(immobile_mask * time_intervals))
            center_distance_m = 0.0
            novel_distance_m = 0.0
            familiar_distance_m = 0.0

            zone_membership_body = {}
            for zone_name in zone_names:
                zone_membership_body[zone_name] = np.array(
                    [f.get('zone') == zone_name for f in frames], dtype=bool
                )

            zone_membership_head = {}
            for zone_name in zone_names:
                zone_membership_head[zone_name] = np.array(
                    [f.get('head_zone') == zone_name for f in frames], dtype=bool
                )

            angle_times = {}
            angle_entries = {}
            ia_confirm_s = max(ia_confirm_ms, 0.0) / 1000.0
            FACING_CONFIRM_MS = max(ia_confirm_ms, 0.0)
            zone_membership_head_confirmed = {}
            for zone_name in zone_names:
                if '_IA' not in zone_name:
                    continue

                zc = resolve_ia_target_centroid(zroot, zone_name) or zone_centroids.get(zone_name)
                if not zc:
                    continue

                angle_time = 0.0
                angle_entry_count = 0
                head_in_zone = zone_membership_head.get(zone_name, np.zeros(len(frames), dtype=bool))
                inner_line = ia_inner_lines.get(zone_name)
                line_normal = ia_line_normals.get(zone_name)
                prev_facing_confirmed = False
                facing_start_time = None
                edge_line = ia_edge_lines.get(zone_name)
                confirmed_mask = np.zeros(len(frames), dtype=bool)
                in_zone_time = 0.0
                in_zone_confirmed = False

                for i, f in enumerate(frames):
                    is_in_zone = head_in_zone[i]
                    head = f.get('head')
                    body = f.get('center')
                    timestamp = f.get('t_adj', f.get('timestamp', 0))

                    crossed = True
                    if is_in_zone and inner_line and line_normal and head:
                        crossed = point_is_past_line((head[0], head[1]), inner_line[0], line_normal, toward_center=False)
                    is_in_zone = is_in_zone and crossed

                    if is_in_zone:
                        in_zone_time += time_intervals[i] if i < len(time_intervals) else 0.0
                    else:
                        in_zone_time = 0.0
                        in_zone_confirmed = False
                    if not in_zone_confirmed and in_zone_time >= ia_confirm_s:
                        in_zone_confirmed = True
                    confirmed_mask[i] = in_zone_confirmed

                    is_facing = False
                    if head and body and in_zone_confirmed and edge_line:
                        ang_edge, _ = compute_facing_angle_to_edge(
                            (head[0], head[1]), (body[0], body[1]),
                            edge_line[0], edge_line[1]
                        )
                        if ang_edge is not None and ang_edge <= angle_threshold:
                            is_facing = True

                    facing_confirmed = False
                    if is_facing:
                        if facing_start_time is None:
                            facing_start_time = timestamp
                        elif timestamp - facing_start_time >= FACING_CONFIRM_MS:
                            facing_confirmed = True
                    else:
                        facing_start_time = None

                    if in_zone_confirmed and facing_confirmed and not prev_facing_confirmed:
                        angle_entry_count += 1

                    if in_zone_confirmed and is_facing:
                        angle_time += time_intervals[i] if i < len(time_intervals) else 0.0

                    prev_facing_confirmed = facing_confirmed

                angle_times[zone_name] = angle_time
                angle_entries[zone_name] = angle_entry_count
                zone_membership_head_confirmed[zone_name] = confirmed_mask

            start_time = metadata.get('time', '') or metadata.get('start_time', '')
            row = {
                'MouseID': metadata.get('mouseID', ''),
                'Group': metadata.get('group', ''),
                'StartTime': start_time,
                'ExptDate': metadata.get('expt_date', ''),
                'subgroup': metadata.get('subgroup', ''),
                'State': state_name,
                'Bin': bin_label,
                'StateDur_s': int(math.ceil(total_duration)),
                'Distance_m': round(total_distance_m, 5),
                'Max_speed': round(max_speed_mps, 4),
            }

            covered_zones = set()
            for main_region, sub_zones in REGIONS_HIERARCHY.items():
                matched = [sz for sz in sub_zones if sz in zone_names]
                if not matched:
                    continue
                covered_zones.update(matched)

                combined_body = np.zeros(len(frames), dtype=bool)
                for sz in matched:
                    combined_body |= zone_membership_body.get(sz, np.zeros(len(frames), dtype=bool))
                time_body = float(np.sum(time_intervals[combined_body])) if np.any(combined_body) else 0.0
                entries_body = count_zone_entries_with_dwell(combined_body, time_intervals, min_dwell_time)
                if main_region != 'CenterZone':
                    row[f'{main_region}_time_body'] = round(time_body, 3)
                    row[f'{main_region}_entries_body'] = entries_body
                if main_region == 'Center':
                    center_distance_m = _sum_distance_in_mask(distances_m, combined_body)
                elif main_region == 'NovelArm':
                    novel_distance_m = _sum_distance_in_mask(distances_m, combined_body)
                elif main_region == 'FamiliarArm':
                    familiar_distance_m = _sum_distance_in_mask(distances_m, combined_body)

                combined_head = np.zeros(len(frames), dtype=bool)
                for sz in matched:
                    if '_IA' in sz:
                        combined_head |= zone_membership_head_confirmed.get(sz, np.zeros(len(frames), dtype=bool))
                    else:
                        combined_head |= zone_membership_head.get(sz, np.zeros(len(frames), dtype=bool))
                time_head = float(np.sum(time_intervals[combined_head])) if np.any(combined_head) else 0.0
                entries_head = count_zone_entries_with_dwell(combined_head, time_intervals, min_dwell_time)
                row[f'{main_region}_time_head'] = round(time_head, 3)
                row[f'{main_region}_entries_head'] = entries_head

                if '_IA' in main_region:
                    total_angle_time = sum(angle_times.get(sz, 0) for sz in matched)
                    total_angle_entries = sum(angle_entries.get(sz, 0) for sz in matched)
                    row[f'{main_region}_time_angle'] = round(total_angle_time, 3)
                    if main_region == 'Object1_IA':
                        row['Object1_IA-entries_angle'] = total_angle_entries
                    else:
                        row[f'{main_region}_entries_angle'] = total_angle_entries

            row['Distance_Center_m'] = round(center_distance_m, 5)
            row['Distance_NovelArm_m'] = round(novel_distance_m, 5)
            row['Distance_FamiliarArm_m'] = round(familiar_distance_m, 5)

            for zone_name in zone_names:
                if zone_name in covered_zones:
                    continue
                body_mask = zone_membership_body.get(zone_name, np.zeros(len(frames), dtype=bool))
                time_body = float(np.sum(time_intervals[body_mask])) if np.any(body_mask) else 0.0
                entries_body = count_zone_entries_with_dwell(body_mask, time_intervals, min_dwell_time)
                row[f'{zone_name}_time_body'] = round(time_body, 3)
                row[f'{zone_name}_entries_body'] = entries_body

                if '_IA' in zone_name:
                    head_mask = zone_membership_head_confirmed.get(zone_name, np.zeros(len(frames), dtype=bool))
                else:
                    head_mask = zone_membership_head.get(zone_name, np.zeros(len(frames), dtype=bool))
                time_head = float(np.sum(time_intervals[head_mask])) if np.any(head_mask) else 0.0
                entries_head = count_zone_entries_with_dwell(head_mask, time_intervals, min_dwell_time)
                row[f'{zone_name}_time_head'] = round(time_head, 3)
                row[f'{zone_name}_entries_head'] = entries_head

                if '_IA' in zone_name:
                    row[f'{zone_name}_angle_time'] = round(angle_times.get(zone_name, 0.0), 3)
                    row[f'{zone_name}_angle_entries'] = angle_entries.get(zone_name, 0)

            rows.append(row)

        if not rows:
            return None

        result_df = pd.DataFrame(rows)
        desired_cols = [
            'MouseID', 'Group', 'ExptDate', 'StartTime', 'subgroup', 'State', 'Bin',
            'StateDur_s', 'Distance_m', 'Max_speed',
            'OpenArm_time_body', 'OpenEntry_time_body', 'Object1_IA_time_head',
            'Object2_IA_time_head', 'Social_IA_time_head', 'FamiliarArm_time_body',
            'NovelArm_time_body', 'Object1_IA_entries_head', 'Object2_IA_entries_head',
            'Social_IA_entries_head', 'Object1_IA_time_angle', 'Object2_IA_time_angle',
            'Object1_IA-entries_angle', 'Object2_IA_entries_angle',
            'Social_IA_time_angle', 'Social_IA_entries_angle',
            'Distance_Center_m', 'Distance_NovelArm_m', 'Distance_FamiliarArm_m',
            'Center_entries_body', 'Center_time_body', 'Center_entries_head', 'Center_time_head',
            'Social_IA_entries_body', 'Social_IA_time_body',
            'Social_Entry_entries_body', 'Social_Entry_time_body', 'Social_Entry_entries_head', 'Social_Entry_time_head',
            'OpenEntry_entries_body', 'OpenEntry_entries_head', 'OpenEntry_time_head',
            'OpenArm_entries_body', 'OpenArm_entries_head', 'OpenArm_time_head',
            'NovelArm_entries_body', 'NovelArm_entries_head', 'NovelArm_time_head',
            'Object1_IA_entries_body', 'Object1_IA_time_body',
            'Object1_Entry_entries_body', 'Object1_Entry_time_body', 'Object1_Entry_entries_head', 'Object1_Entry_time_head',
            'Object2_IA_entries_body', 'Object2_IA_time_body',
            'Object2_Entry_entries_body', 'Object2_Entry_time_body', 'Object2_Entry_entries_head', 'Object2_Entry_time_head',
            'FamiliarArm_entries_body', 'FamiliarArm_entries_head', 'FamiliarArm_time_head'
        ]
        text_cols = {'MouseID', 'Group', 'ExptDate', 'StartTime', 'subgroup', 'State', 'Bin'}
        for col in desired_cols:
            if col not in result_df.columns:
                result_df[col] = '' if col in text_cols else 0.0
        ordered_existing = [c for c in desired_cols if c in result_df.columns]
        trailing_cols = [c for c in result_df.columns if c not in desired_cols]
        result_df = result_df[ordered_existing + trailing_cols]
        return result_df

    except Exception as e:
        print(f"Statistics error: {e}")
        import traceback
        traceback.print_exc()
        return None


def write_zone_stats_excel(result_df, stem: str, output_dir: str,
                           params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Write a results DataFrame to ``<stem>_zone_stats.xlsx`` (+ params sheet).

    Returns the written path, or None when there is nothing to write.
    """
    try:
        import pandas as pd
    except Exception:
        return None
    if result_df is None or len(result_df) == 0:
        return None
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f'{stem}_zone_stats.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        result_df.to_excel(writer, index=False, sheet_name='Zone Statistics')
        if params:
            param_rows = []
            for key, val in params.items():
                if isinstance(val, (dict, list, tuple)):
                    try:
                        val_out = json.dumps(val)
                    except Exception:
                        val_out = str(val)
                else:
                    val_out = str(val)
                param_rows.append({'param': str(key), 'value': val_out})
            params_df = pd.DataFrame(param_rows)
            params_df.to_excel(writer, index=False, sheet_name='params')
    return excel_path


def calculate_zone_statistics_full(stem: str, times_buf: List[float],
                                    zone_buf: List[Optional[str]], state_buf: List[Optional[str]],
                                    output_dir: str, zones_full: Optional[Dict] = None,
                                    centers_buf: List = None, head_buf: List = None,
                                    min_dwell_time: float = 0.2,
                                    zone_assigner: 'FastZoneAssigner' = None,
                                    file_meta: Optional[Dict[str, Any]] = None,
                                    angle_threshold: float = 45.0,
                                    ia_confirm_ms: float = 100.0,
                                    ia_line_extend_px: float = 15.0,
                                    ia_inner_offset_px: float = 50.0,
                                    ia_parallel_tol_deg: float = 15.0,
                                    params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Compute zone statistics and write ``<stem>_zone_stats.xlsx``.

    Drop-in equivalent of the retracking function: same args (incl. positional
    ``output_dir``), same Excel output, returns the written path or None.
    """
    result_df = compute_zone_stats_df(
        stem, times_buf, zone_buf, state_buf, zones_full=zones_full,
        centers_buf=centers_buf, head_buf=head_buf, min_dwell_time=min_dwell_time,
        zone_assigner=zone_assigner, file_meta=file_meta,
        angle_threshold=angle_threshold, ia_confirm_ms=ia_confirm_ms,
        ia_line_extend_px=ia_line_extend_px, ia_inner_offset_px=ia_inner_offset_px,
        ia_parallel_tol_deg=ia_parallel_tol_deg, params=params,
    )
    if result_df is None or len(result_df) == 0:
        return None
    return write_zone_stats_excel(result_df, stem, output_dir, params=params)
