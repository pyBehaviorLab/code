#!/usr/bin/env python3
r"""
Offline behavioural analysis for pyBehaveTrack / pymazeRetrack video_data.txt.

A cohesive, parallelised, plot-producing analyzer that matches the methodology of
D:\pymazeRetrack\retracking_pyqt6_fast_v1.py and the conventions of EthoVision XT /
ANY-maze (see ANALYSIS_PLAN.md).

Pipeline (per file, computed ONCE into a Track, reused by metrics/bins/plots):
    parse → confidence-gate → jump-clamp (teleports) → [optional light smooth]
          → minimum-distance-moved filter (de-jitter WITHOUT killing fast darts)
          → dt timeline → distance / velocity / zones / entries / immobility

Key correctness choices:
  * Velocity/distance use a **minimum-distance-moved** filter (EthoVision "Input
    Filter"), NOT heavy smoothing, mice are fast, real darts are preserved
    (peak velocity kept; only sub-threshold jitter dropped).
  * Zone visits are **dwell-debounced** (min visit duration): no bogus flicker entries.
  * Immobility is proper **bouts** (min duration), not per-frame counting.

Outputs an .xlsx whose FIRST sheet is `Meta` (every parameter used), plus
`Session_Summary`, `Time_Bins`, `Excluded`, and per-session plot PNGs.

CLI:  python -m source.analysis.offline_analysis DATA_DIR --bin 60 \
          --novel Right_arm --familiar Left_arm --plots -o analysis.xlsx
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Every statistic this module reports is DEFINED there, once, as a pure
# function over arrays. This file decides which columns to emit and what to
# call them; `measures` decides what the numbers mean. Run
# `python -m source.analysis.validate` to check them against arithmetic.
from tools.offline_analysis.engine import measures as ms
from tools.offline_analysis.engine import refine as _rf

VERSION = "1.0"

# Fast JSON for the per-frame pose_array (the parse hot-path). orjson is ~5-10×
# faster than stdlib json for the 9000 records/file; install it for a big speedup.
try:
    import orjson as _oj

    def _loads(s):
        try:
            return _oj.loads(s)
        except Exception:
            return None
except Exception:                                          # pragma: no cover
    import json as _json

    def _loads(s):
        try:
            return _json.loads(s)
        except Exception:
            return None

# ─────────────────────────────── parameters ──────────────────────────────────

DEFAULTS: Dict[str, Any] = {
    # trajectory
    "pcut": 0.55,                 # pose confidence cutoff, keypoints below 55% are treated
                                  # as missing (excluded) for that frame
    "body_part": "Center",        # primary point for distance/zone = the body CENTRE keypoint.
                                  # It tracks translation only; the multi-KP centroid also
                                  # picks up head/tail rotation (~2 m more) which is NOT travel.
                                  # "" would use the centroid of all valid keypoints instead.
    "body_parts_multi": [],       # >1 part → per-part locomotion columns ({part}_Distance_m…)
    # ── tracking-error corrections (fix detector mistakes; NOT smoothing) ──
    "swap_correct": True,         # head/tail identity-swap resolver. ON, and the pair it
                                  # compares is chosen by NAME (default_swap_pair): the two ENDS
                                  # of the animal. The pair must be the two ENDS: names[0]/names[1]
                                  # is Snout and Head on a SLEAP model, one end against itself,
                                  # which false-positives on fast frames and inflates distance. With
                                  # no identifiable head AND body part the pair is ("", "") and the
                                  # resolver stays off rather than guess.
    "swap_cost_threshold": 0.3,   # relative displacement drop that flags a swap
    "head_part": "",              # keypoint treated as "head" ("" → 1st keypoint)
    "swap_body_part": "",         # keypoint treated as "body" ("" → 2nd keypoint)
    "max_speed_cm_s": 380.0,      # outlier clamp ≈ 3000 px/s (reference default); a
                                  # mouse dart can hit ~330 cm/s, so 200 clipped real motion
    # ── trace cleaning (see engine/refine.py) ──
    # ON, because a trace nobody cleaned is not a neutral starting point: it
    # is one with the model's worst frames still in it. Modelled on refineDLC
    # and adaptive where that paper is fixed, per-keypoint confidence rather
    # than one global cutoff, a jump limit taken from the recording's own
    # displacement statistics rather than 30 px, and a gap limit in seconds
    # rather than frames.
    "refine": True,
    "refine_percentile": 5.0,     # per-keypoint confidence percentile to drop
    "refine_min_confidence": 0.1,  # absolute floor under the percentile
    "refine_confidence_ceiling": 0.6,   # never gate above this
    "refine_jump_mads": 6.0,      # robust deviations above the median step
    "refine_max_gap_s": 0.25,     # longest hole worth interpolating
    "refine_interpolate": True,
    # ── across keypoints, and forwards+backwards in time (pose_repair.py) ──
    # The stages the two above cannot reach. Confidence and the jump limit both
    # pass a keypoint that is certain and in the WRONG PLACE; the body shape
    # catches it. And a left/right swap keeps the pose plausible, so only the
    # shape sees that too.
    "refine_reconstruct": True,   # place a missing part from the rest of the body
    "refine_mistrack_residual": 0.35,   # body lengths from the fitted shape
    "refine_fix_swaps": True,     # exchange a mirror pair that is on the wrong side
    # ``anatomical`` decides side from the body axis in each frame and resolves
    # the run along time, so a majority of swapped frames cannot outvote it.
    # ``shape`` is the older per-frame test against a canonical shape learned
    # from the same trace, and is kept for comparison.
    #
    # Measured against known truth: with 40% of a session swapped, anatomical
    # recovers 100% and shape 3.9%; at 70% shape recovers none and corrupts 270
    # good frames. On a clean trace they tie at 100%, and there anatomical is
    # 1.581 px mean error against shape's 1.465. A tenth of a pixel is inside
    # the keypoint noise; a run identified back-to-front is not, which is why
    # this is the default.
    "refine_swap_method": "auto",
    "refine_swap_transition_cost": 0.15,  # cost of changing side between frames
    "refine_swap_margin": 0.6,    # ``shape`` only: how much better before believing it
    "refine_robust_rounds": 3,    # fit-then-reject passes; one per frame each
    # ``rts`` = forward Kalman + backward smoother, weighted by each
    # detection's own confidence. ``savgol`` is the older centred window,
    # ``none`` disables. Measured against a synthetic trace with known truth:
    # median error 0.58 px against 0.77 for savgol, worst 11.2 px against 50.1.
    "refine_smoother": "rts",
    "refine_measure_px": 2.0,     # localisation noise at confidence 1.0
    "refine_process_scale": 1.0,  # multiplier on the trace's own acceleration
    "refine_min_anchors": 3,      # fewest visible parts a frame can be fitted from
    "refine_max_residual": 0.25,  # body lengths; beyond it the frame is not fitted
    "refine_smooth": True,
    "refine_smooth_window": 7,    # frames; only the savgol smoother reads these
    "refine_smooth_order": 2,
    "min_move_cm": 0.0,           # OPT-IN extra de-jitter on top of One-Euro; 0 = off
    "rolling_median": False,      # optional 5-frame median gap-fill/de-spike
    "smooth": True,               # One-Euro jitter filter ON: removes the ~12% per-frame jitter
                                  # inflation on the Centre point while preserving fast darts
    "euro_min_cutoff": 1.0,       # One-Euro cutoff (reference default), lower = more smoothing
    "euro_beta": 0.007,           # One-Euro speed coupling (reference default)
    # zone / behaviour
    "min_dwell_s": 0.20,          # min visit duration for a counted entry
    "immobility_cm_s": 2.0,       # mobile/immobile speed cut
    "freeze_cm_s": 0.5,           # stricter freezing speed cut (respiration-only)
    "min_freeze_s": 1.0,          # min duration for an immobility/freezing bout
    # binning, three ways to ask for the same thing, because experiments
    # ask for it three ways: "every minute", "in thirds", and "the two minutes
    # before the door opened and the eight after". Precedence: edges, then
    # count, then width.
    # OFF by default. Bins multiply out against every zone metric, so a
    # 60-second width turned a 33-column result into 97 columns, 64 of them
    # `0-60s_Left_poke_entries` and its kin, and on a full hour it would be
    # about 460. Nobody who wanted twelve numbers asked for that, and the
    # columns that answer the question were buried among them. The machinery
    # is untouched and the control still offers it; only the default moved.
    "bin_size_s": 0.0,            # fixed width in seconds (0 = no binning)
    "bin_count": 0,               # >0: this many EQUAL bins over the recording
    "bin_edges_s": (),            # explicit boundaries in seconds, e.g. (0,120,600)
    # y-maze roles
    "novel_arm": "",
    "familiar_arm": "",
    "start_arm": "",
    # scale / io
    "px_per_cm_override": 0.0,    # 0 → use file header
    "plots": True,
    "n_jobs": 0,                  # 0 → auto (cpu-1)
    # ── metric selection (which groups get computed/exported) ──
    "metrics": {
        "locomotion": True,       # distance (m), mean/max speed
        "immobility": True,       # immobile time + bouts
        "freezing": False,        # stricter freezing time, episodes, latency
        "transitions": True,      # zone transition count
        "zone_time": True,        # per-zone time_s
        "zone_entries": True,     # per-zone entries
        "zone_latency": False,    # per-zone first-entry latency
        "zone_distance": True,    # per-zone distance travelled INSIDE each zone (m)
        "zone_advanced": False,   # per-zone exits + mean_visit (distance is its own flag)
        "ymaze": False,           # (disabled, objective measures only)
        "head_zones": True,       # per-zone occupancy of the HEAD point, when the
                                  # pose has one. Investigation is scored on the
                                  # nose; locomotion on the centroid.
        "interaction": True,      # object/social interaction zones: time and
                                  # entries while ORIENTED at the target. Only
                                  # fires when such zones exist.
    },
    "zones_include": [],          # which zones to report; [] = all
    "regions": {},                # {region: [zone, ...]}, sub-zones combined
                                  # into what a paper reports. Empty = none.
    # ── staging ──
    # A fourteen-stage protocol is fourteen rows. Splitting happens only when
    # the recording actually carries two or more named stages, so a session
    # with no protocol is unaffected.
    "split_stages": True,
    # How a recording's states become rows. "auto" groups by state NAME when
    # the states interleave, a trial-structured task returns to `choice_state`
    # hundreds of times, and each visit is far too short to measure on its own,
    # and keeps contiguous spans for a protocol whose phases run once each.
    # "grouped" and "spans" force it either way. See `group_state_rows`.
    "state_rows": "auto",         # auto | grouped | spans
    # Analyse only part of each recording (or of each stage, when rows
    # are split by stage). 0/0 is the whole thing, which is the default
    # and what every existing result was produced with.
    "trim_start_s": 0.0,
    "trim_end_s": 0.0,
    # Interaction scoring (see source/analysis/space.py). Empty = defaults:
    # 45 deg facing cone, 100 ms confirmation, nearest edge of the zone.
    "interaction": {},
    # ── figure selection (which plots to render) ──
    "figures": {
        "track": True, "heatmap": True, "zone_timeline": True,
        "distance_velocity": False, "group": False,
    },
    "heatmap_bins": 40,
    # ── output column curation (empty = all, natural order) ──
    "columns": {
        "select": [],             # keep only these output columns (in this order); [] = all
        "rename": {},             # {orig: new} applied to output column names
    },
}


def _want(params: dict, key: str, default: bool = False) -> bool:
    """Whether a metric group is enabled (auto-on Y-maze when arm roles set)."""
    m = params.get("metrics") or DEFAULTS["metrics"]
    if key == "ymaze":
        return bool(m.get("ymaze")) or bool(params.get("novel_arm") or params.get("familiar_arm"))
    return bool(m.get(key, default))


# ──────────────────────────────── parsing ────────────────────────────────────

@dataclass
class Session:
    path: str
    stem: str
    subject: str = ""
    date: str = ""
    start_time: str = ""          # HH:MM:SS of the stage start
    stage: str = ""
    px_per_cm: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    body_parts: List[str] = field(default_factory=list)
    zones: List[dict] = field(default_factory=list)
    ts_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    location: List[str] = field(default_factory=list)
    kp: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    #: The protocol stage each frame belongs to. The recorder writes it per
    #: row, and a 14-stage protocol is 14 result rows, not one, which is how
    #: every behavioural paper reports it.
    stage_series: List[str] = field(default_factory=list)
    #: ``"file"`` when the timestamps are the recorded acquisition times,
    #: ``"index"`` when the file had no timestamp column and these are frame
    #: indices. Never conflate the two: one is milliseconds, one is not.
    ts_source: str = "file"
    #: Where the camera's timestamps came from (``device``/``host``/…), as
    #: declared by the recorder.
    frame_clock: str = ""
    #: Backend, model and identity method that produced the poses.
    tracker: Dict[str, Any] = field(default_factory=dict)
    #: The parsed header, for callers that need a field this dataclass omits.
    header: Any = None
    dialect: str = ""


# body-part keywords a zone name could collide with (case-insensitive). A zone
# literally named one of these (e.g. "Center") would produce columns that clash
# with the body point's own columns, so we tag such zones like their siblings.
_BODYPART_KEYWORDS = {"center", "centre", "head", "tail", "tailbase", "nose",
                      "snout", "body", "neck", "back", "middle"}


def zone_rename_map(names: Sequence[str],
                    body_parts: Sequence[str]) -> Dict[str, str]:
    """``{recorded name: analysed name}`` for zones that collide with a body part.

    THE one statement of the rule, because a caller that does not know about it
    is looking at a different set of zone names from the one every column,
    plot and filter uses. The Analyze panel was such a caller: it listed the
    recording's `Center` while the analysis called it `Center_arm`, so the
    zone filter, which holds every zone by default, matched three of the
    four and the Y-maze's choice point was dropped from every result.
    """
    if not names:
        return {}
    collide = set(str(p).lower() for p in body_parts) | _BODYPART_KEYWORDS
    suffix = "_arm" if any(str(n).lower().endswith("_arm") for n in names) else "_zone"
    rename: Dict[str, str] = {}
    existing = set(str(n) for n in names)
    for n in names:
        base = str(n)
        if base.lower() in collide and not base.lower().endswith(("_arm", "_zone")):
            new = base + suffix
            if new not in existing:                # never create a duplicate
                rename[base] = new
                existing.add(new)
    return rename


def _disambiguate_zones(s: "Session") -> None:
    """Rename any zone whose name collides with a body-part name so zone columns and
    body-point columns never clash (user rule: a zone "Center" alongside a body point
    "Center" → tag the ZONE). The tag matches the siblings' convention: if other zones
    end in "_arm" (Y-maze) the zone becomes e.g. "Center_arm"; otherwise "_zone".
    Applied to both the zone list and the per-frame `location` strings so every
    downstream metric/plot uses the disambiguated name consistently."""
    names = [z.get("name") for z in s.zones
             if isinstance(z, dict) and z.get("type") != "scale" and z.get("name")]
    rename = zone_rename_map(names, s.body_parts)
    if not rename:
        return
    for z in s.zones:
        if isinstance(z, dict) and z.get("name") in rename:
            z["name"] = rename[z["name"]]
    s.location = [rename.get(loc, loc) for loc in s.location]



def parse_txt(path: str) -> Session:
    """Parse a session file into a :class:`Session`.

    Reading goes through :mod:`source.core.video_data_schema`, the one reader,
    so every dialect (the current v4 columns, the legacy ``actual_ts``
    header, the legacy ``B``/``M``/``V`` lines, and pymazeRetrack's one-JSON-
    record-per-line) arrives here under the same canonical column names.

    The timestamp is never silently invented. If the file carries no timestamp
    column at all, ``ts_source`` says ``"index"`` and the values are frame
    indices, which :mod:`source.analysis.clock` then refuses to treat as
    milliseconds. The old reader defaulted to the row counter in silence, and a
    thirty-minute session measured thirty-six seconds.
    """
    from tools.offline_analysis import video_data_schema as vds

    stem = (os.path.splitext(os.path.basename(path))[0]
            .replace("_video_data", ""))
    s = Session(path=path, stem=stem)

    hdr = vds.VideoDataHeader.parse(path)
    s.header = hdr
    s.dialect = hdr.dialect
    s.frame_clock = hdr.frame_clock
    s.tracker = dict(hdr.tracker)
    s.subject = hdr.subject_id
    s.stage = hdr.stage
    s.resolution = hdr.resolution
    s.body_parts = list(hdr.body_parts)
    s.zones = vds.zones_as_list(hdr.zones)

    start = hdr.start_time
    if start:
        s.date = start[:10]
        if "T" in start:
            s.start_time = start.split("T", 1)[1][:8]
        elif " " in start:
            s.start_time = start.split(" ", 1)[1][:8]

    s.px_per_cm = hdr.px_per_cm

    cols = set(hdr.index)
    ts_col = "frame_ts_ms"
    pose_col = "pose_array"
    s.ts_source = "file" if ts_col in cols else "index"

    ts: List[float] = []
    loc: List[str] = []
    stages: List[str] = []
    # Keypoints are indexed BY FRAME (not by occurrence): a body part absent
    # from some rows, a keypoint occluded on those frames, must land at its
    # true frame index, with NaN where missing. Appending per occurrence and
    # end-padding would silently shift the whole trajectory.
    kp_by_frame: Dict[str, Dict[int, Tuple[float, float, float]]] = {}

    for row in vds.iter_rows(path, hdr):
        fi = len(ts)
        t = vds.parse_float(row.get(ts_col), math.nan)
        if math.isnan(t):
            t = float(fi)
        ts.append(t)
        loc.append("" if vds.is_na(row.get("location")) else str(row["location"]))
        stages.append("" if vds.is_na(row.get("stage")) else str(row["stage"]).strip())

        pose = _loads(row.get(pose_col, "")) if not vds.is_na(row.get(pose_col)) else None
        if isinstance(pose, dict):
            for bp, v in pose.items():
                if not (isinstance(v, (list, tuple)) and len(v) >= 2):
                    continue
                kp_by_frame.setdefault(bp, {})[fi] = (
                    float(v[0]), float(v[1]),
                    float(v[2]) if len(v) > 2 else 1.0)

    s.stage_series = stages
    if not s.stage and stages:
        named = [x for x in stages if x]
        if named:
            s.stage = named[0]
    N = len(ts)
    s.ts_ms = np.asarray(ts, float)
    s.location = loc

    for bp, fmap in kp_by_frame.items():
        x = np.full(N, math.nan); y = np.full(N, math.nan); c = np.full(N, math.nan)
        for i, (xi, yi, ci) in fmap.items():
            x[i], y[i], c[i] = xi, yi, ci
        s.kp[bp] = {"x": x, "y": y, "conf": c}
    if not s.body_parts:
        s.body_parts = list(s.kp.keys())

    _disambiguate_zones(s)
    if not s.start_time:
        # fallback: trailing DDMMYYHHMMSS stamp in the stem (…_070726090107)
        import re
        mt = re.search(r"(\d{12})(?:_video_data)?$", s.stem)
        if mt:
            hhmmss = mt.group(1)[6:]            # last 6 of DDMMYY+HHMMSS
            s.start_time = f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
    return s


# ─────────────────────────── signal processing ───────────────────────────────

def dt_seconds(ts_ms: np.ndarray) -> np.ndarray:
    """Per-frame dt (s), monotonic-unwrapped across resets, clamped ≥0."""
    if len(ts_ms) == 0:
        return np.empty(0)
    d = np.diff(ts_ms, prepend=ts_ms[0])
    if np.all(d >= 0):                       # already monotonic (common), vectorised
        d[d < 0] = 0.0
        return d / 1000.0
    med = np.median(d[d > 0]) if np.any(d > 0) else 33.0
    adj = np.empty_like(ts_ms)
    offset = 0.0; prev = None
    for i, t in enumerate(ts_ms):
        v = t + offset
        if prev is not None and v < prev:
            offset += (prev - v) + med
            v = t + offset
        adj[i] = v; prev = v
    out = np.diff(adj, prepend=adj[0]) / 1000.0
    out[out < 0] = 0.0
    return out


def _swap_anchor(kx: dict, ky: dict, head: str, body: str):
    """A third keypoint that can say which of two labels is which.

    A mouse is a chain: head, centre, tail base. The head is farther from the
    tail base than the centre is, in every frame, whatever the animal is
    doing -- so a keypoint outside the confusable pair settles the question
    outright, with no reference to the previous frame and nothing to
    false-positive on when the animal turns.

    Returns ``(name, head_is_farther, consistency)`` for the most decisive
    such keypoint, or ``None`` when none of them is decisive enough to trust.
    """
    best = None
    for a in kx:
        if a in (head, body):
            continue
        dh = np.hypot(kx[head] - kx[a], ky[head] - ky[a])
        db = np.hypot(kx[body] - kx[a], ky[body] - ky[a])
        ok = np.isfinite(dh) & np.isfinite(db)
        if ok.sum() < 20:
            continue
        # MEDIANS, not the fraction of frames. Counting frames measures how
        # often the chain currently holds, which a swapped recording drags
        # down, so the anchor was rejected on exactly the data it exists
        # for: a quarter of frames crossed scored 0.75 and failed a 0.90 bar.
        # The medians say how far apart the two keypoints sit from this one,
        # which a swap exchanges but does not blur.
        mh, mb = float(np.median(dh[ok])), float(np.median(db[ok]))
        if min(mh, mb) <= 0:
            continue
        ratio = max(mh, mb) / min(mh, mb)
        if best is None or ratio > best[2]:
            best = (a, mh > mb, ratio)
    # Below this the two keypoints are the same distance from the candidate,
    # so it cannot tell them apart and guessing from it would be worse than
    # the displacement test it replaces.
    if best is not None and best[2] >= 1.25:
        return best
    return None


def _resolve_swaps(kx: dict, ky: dict, kc: dict, head: str, body: str,
                   threshold: float) -> int:
    """Head/body identity-swap RESOLUTION.

    Detectors flip which keypoint is the head and which the body. Resolving
    that by displacement alone was wrong in two ways, both found by running
    it over real recordings rather than by reasoning about it:

    1. A FLIP PERSISTS. Once the labels cross they stay crossed until the
       detector crosses them back. Flagging frames independently against the
       ORIGINAL previous frame marks the two CROSSINGS and nothing between,
       so a 51-frame swap had 2 frames corrected, all 51 left wrong, and a
       21 px discontinuity introduced where the animal moved 1 px per frame.

    2. A SWAP AND A SHARP TURN LOOK THE SAME to a displacement test, for
       keypoints a body-length apart. On a real Y-maze recording it called
       31.5% of frames swapped and inflated distance by 23%.

    So a crossing must now satisfy BOTH tests, because each alone has a
    failure mode the other does not:

    * displacement -- exchanging the labels restores continuity. Alone it
      cannot tell a swap from the animal turning on the spot.
    * the body chain -- a mouse is head, centre, tail base in that order, so
      a third keypoint says which label belongs where without reference to
      any other frame. Alone it cannot tell a swap from a curled or rearing
      animal, whose head really is near its tail: on the same recording the
      chain was violated on 1.6% of frames and correcting them made the
      centre's path LONGER, which is not what fixing a swap does.

    Requiring both, on that recording, leaves 0.10% of frames -- and even
    those do not improve it, which is the measured reason `swap_correct`
    defaults to OFF. It is for data whose detector genuinely swaps.

    Mutates kx/ky/kc for `head` and `body` in place; returns the number of
    frames whose labels were exchanged.
    """
    hx, hy, bx, by = kx[head], ky[head], kx[body], ky[body]
    n = len(hx)
    anchor = _swap_anchor(kx, ky, head, body)
    ax = ay = None
    if anchor is not None:
        name, head_is_farther, _c = anchor
        ax, ay = kx[name], ky[name]

    def chain_ok(h_xy, b_xy, i) -> bool:
        """Whether this labelling puts the chain the right way round."""
        if ax is None or not (np.isfinite(ax[i]) and np.isfinite(ay[i])):
            return True                      # no anchor: nothing to object to
        dh = math.hypot(h_xy[0] - ax[i], h_xy[1] - ay[i])
        db = math.hypot(b_xy[0] - ax[i], b_xy[1] - ay[i])
        return (dh > db) if head_is_farther else (dh < db)

    swapped = np.zeros(n, dtype=bool)
    state = False                      # True: this frame's labels are crossed
    prev = None                        # previous valid frame, already corrected
    for i in range(n):
        if not (np.isfinite(hx[i]) and np.isfinite(bx[i])
                and np.isfinite(hy[i]) and np.isfinite(by[i])):
            swapped[i] = state         # a gap does not undo a crossing
            continue
        here = ((hx[i], hy[i]), (bx[i], by[i]))
        keep = (here[1], here[0]) if state else here
        toggle = (keep[1], keep[0])
        if prev is not None:
            ph, pb = prev
            cost_keep = (math.hypot(keep[0][0] - ph[0], keep[0][1] - ph[1])
                         + math.hypot(keep[1][0] - pb[0], keep[1][1] - pb[1]))
            cost_tog = (math.hypot(toggle[0][0] - ph[0], toggle[0][1] - ph[1])
                        + math.hypot(toggle[1][0] - pb[0], toggle[1][1] - pb[1]))
            continuity = (cost_keep > 0
                          and (cost_keep - cost_tog) / cost_keep > threshold)
            # The chain has to agree, and has to agree BECAUSE of the
            # toggle: a frame whose chain is wrong either way is a posture,
            # not a label error, and exchanging the labels cannot repair it.
            # With no anchor there is no chain to consult and no objection to
            # raise, requiring one there vetoed every toggle, which turned
            # the two-keypoint fallback off entirely.
            physical = ax is None or (
                chain_ok(toggle[0], toggle[1], i)
                and not chain_ok(keep[0], keep[1], i))
            if continuity and physical:
                state = not state
                keep = toggle
        swapped[i] = state
        prev = keep

    idx = np.where(swapped)[0]
    if idx.size:                              # x, y AND confidence together
        for a, b in ((kx[head], kx[body]), (ky[head], ky[body]),
                     (kc[head], kc[body])):
            tmp = a[idx].copy(); a[idx] = b[idx]; b[idx] = tmp
    return int(idx.size)


def _rolling_median(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred rolling median over VALID (non-NaN) values in the window, mirrors
    retracking_pyqt6_fast_v1_2.batch_rolling_median_numba. Fills isolated NaN from
    neighbours and de-spikes; NaN only survives where the whole window is NaN."""
    n = len(x)
    out = np.array(x, float)
    half = window // 2
    for i in range(n):
        seg = x[max(0, i - half): min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


def _one_euro(x: np.ndarray, t_s: np.ndarray, min_cutoff: float, beta: float,
              d_cutoff: float = 1.0) -> np.ndarray:
    """Casiez One-Euro low-lag jitter filter (NaN-safe). Optional/gentle only."""
    out = np.array(x, float)
    x_prev = dx_prev = t_prev = None
    for i in range(len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            continue
        if x_prev is None:
            x_prev, dx_prev, t_prev = xi, 0.0, t_s[i]; out[i] = xi; continue
        te = t_s[i] - t_prev
        if te <= 0:
            out[i] = x_prev; continue
        a = lambda c: (2 * math.pi * c * te) / (2 * math.pi * c * te + 1)
        dx = (xi - x_prev) / te
        dx_hat = a(d_cutoff) * dx + (1 - a(d_cutoff)) * dx_prev
        ac = a(min_cutoff + beta * abs(dx_hat))
        x_hat = ac * xi + (1 - ac) * x_prev
        out[i] = x_hat
        x_prev, dx_prev, t_prev = x_hat, dx_hat, t_s[i]
    return out


@dataclass
class Track:
    """Processed trajectory + timeline, computed ONCE per file, reused everywhere."""
    cx: np.ndarray                 # body point x (px), teleport-clamped, optional smooth
    cy: np.ndarray
    dt: np.ndarray                 # per-frame dt (s)
    t_rel: np.ndarray              # stage-relative time (s), 0-based
    step_cm: np.ndarray            # per-frame distance (cm), min-move filtered
    speed_cm_s: np.ndarray         # per-frame speed (cm/s)
    duration_s: float
    ppc: float                     # px per cm (0 = uncalibrated)
    calibrated: bool
    has_pose: bool


def build_track(sess: Session, params: dict) -> Track:
    """Build a processed trajectory."""
    return _build_track_core(sess, params)


def _build_track_core(sess: Session, params: dict) -> Track:
    """Gate → jump-clamp → optional smooth → min-move filter → distance/velocity."""
    N = len(sess.ts_ms)
    if N == 0:                                    # empty/na file, no rows to track
        z = np.empty(0)
        ppc = params["px_per_cm_override"] or sess.px_per_cm
        return Track(cx=z, cy=z, dt=z, t_rel=z, step_cm=z, speed_cm_s=z,
                     duration_s=0.0, ppc=ppc, calibrated=ppc > 0, has_pose=False)
    dt = dt_seconds(sess.ts_ms)
    t_s = sess.ts_ms / 1000.0
    t_rel = (sess.ts_ms - sess.ts_ms[0]) / 1000.0 if N else np.empty(0)
    duration = float(t_rel[-1]) if N else 0.0
    ppc = params["px_per_cm_override"] or sess.px_per_cm
    calibrated = ppc > 0
    ppc_eff = ppc if calibrated else 1.0     # px treated as "cm" when uncalibrated

    # ── tracking-error corrections on the RAW keypoints, BEFORE the confidence
    #    gate and body-point selection. These fix detector MISTAKES (head/body
    #    identity swaps, teleport outliers, missing pose); they do NOT smooth.
    #    Work on copies so the Session stays pristine (reused across body points).
    kx = {bp: np.asarray(a["x"], float).copy() for bp, a in sess.kp.items()}
    ky = {bp: np.asarray(a["y"], float).copy() for bp, a in sess.kp.items()}
    kc = {bp: np.asarray(a["conf"], float).copy() for bp, a in sess.kp.items()}
    names = list(sess.kp.keys())

    # (1) head/body swap resolution, cost/anchor based (reference: swap FIRST).
    #
    # The fallback is DEFAULTS, not a literal: this said True while DEFAULTS
    # and the per-body-part builder below both said False, so a caller that
    # omitted the key got the resolver running over the primary track and NOT
    # over the per-part tracks, two answers from one run.
    if params.get("swap_correct", DEFAULTS["swap_correct"]) and len(names) >= 2:
        _h, _b = default_swap_pair(names)
        h = params.get("head_part") or _h
        b = params.get("swap_body_part") or _b
        if h in kx and b in kx and h != b:
            _resolve_swaps(kx, ky, kc, h, b, float(params.get("swap_cost_threshold", 0.3)))

    # (1b) the confidence gate, and then cleaning, in that order, because the
    # reverse undoes the cleaning.
    #
    # The gate runs FIRST and ONCE. Downstream of `refine_all` it would undo
    # it: refine interpolates a hole and the gate immediately punches it back
    # out, because the gate tests the confidence of a detection refine has
    # already decided not to trust. Anything refine recovers has no confidence
    # to be gated on at all, so a gate below it discards exactly the frames
    # the recovery exists to restore.
    pcut = params["pcut"]
    for bp in names:
        below = ~(kc[bp] >= pcut)
        kx[bp][below] = np.nan
        ky[bp][below] = np.nan

    # (1c) refineDLC-style cleaning, per keypoint, ON by default: each part
    # gated against its OWN confidence distribution as well, frame-to-frame
    # jumps rejected against a robust limit taken from the recording itself,
    # short holes interpolated from both sides while the long ones stay open,
    # whatever is still missing placed from the rest of the animal, and the
    # result smoothed with a centred window. See `engine.refine`.
    _refined, _reports = _rf.refine_all(
        {bp: {"x": kx[bp], "y": ky[bp], "conf": kc[bp]} for bp in names},
        t_s, params)
    for bp in names:
        kx[bp], ky[bp] = _refined[bp]["x"], _refined[bp]["y"]

    # pick/derive the body point from the cleaned keypoints
    xs = [kx[bp] for bp in names]
    ys = [ky[bp] for bp in names]
    want = params.get("body_part") or ""
    if want and want in kx:
        cx, cy = kx[want], ky[want]
    elif xs:
        # A frame where every keypoint fell below the gate is all-NaN, and
        # `nanmean` of nothing is a documented NaN, the right answer, with a
        # RuntimeWarning attached. Suppressed rather than silenced globally:
        # the warning fired on every run of every model and buried the log,
        # while saying only "this frame had no usable keypoints", which the
        # NaN already says to everything downstream.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice",
                                    RuntimeWarning)
            cx = np.nanmean(np.vstack(xs), axis=0)
            cy = np.nanmean(np.vstack(ys), axis=0)
    else:
        cx = np.full(N, np.nan); cy = np.full(N, np.nan)

    has_pose = np.isfinite(cx).sum() >= 2

    # teleport/outlier correction, reject supra-threshold jumps. CRITICAL: each
    # frame is judged against the ORIGINAL (unmodified) neighbours, exactly like
    # retracking_pyqt6_fast_v1_2.batch_speed_correction_numba. Judging against the
    # already-corrected value would let a single fast dart cascade and freeze the
    # WHOLE track to one point (the 306/309 "0 m distance" bug).
    max_px_s = params["max_speed_cm_s"] * ppc_eff
    ox, oy = cx.copy(), cy.copy()                 # pristine originals for the test
    for i in range(1, N):
        if not (np.isfinite(ox[i]) and np.isfinite(ox[i - 1])):
            continue
        d = dt[i]
        if d > 0 and math.hypot(ox[i] - ox[i - 1], oy[i] - oy[i - 1]) / d > max_px_s:
            # MISSING, not "stationary". Freezing the outlier to the previous
            # position removed the jump on the way IN and then re-added it on
            # the way OUT, because the next frame is back on the real track:
            # 49 -> 349 -> 350 clamped to 49 -> 49 -> 350 still walks 301 px,
            # so a 600 cm/s excursion cost the distance nothing at all. A gap
            # diffs to NaN at both ends and contributes zero, which is what a
            # frame that failed the plausibility test is worth.
            cx[i], cy[i] = np.nan, np.nan

    # optional rolling-median gap fill (reference use_rolling_median, window 5),
    # fills isolated NaN from valid neighbours; OFF by default like the reference.
    if params.get("rolling_median"):
        cx = _rolling_median(cx, 5)
        cy = _rolling_median(cy, 5)

    if params.get("smooth"):
        cx = _one_euro(cx, t_s, params["euro_min_cutoff"], params["euro_beta"])
        cy = _one_euro(cy, t_s, params["euro_min_cutoff"], params["euro_beta"])

    # per-frame step. NaN frames (gated-out / gaps) → NaN diff → nan_to_num 0, so a
    # gap contributes 0 and is not teleported across (reference: np.nansum). The
    # minimum-distance-moved de-jitter is OPT-IN (min_move_cm default 0) so, like
    # the reference, distance is the raw sum of pose displacements.
    step_px = np.hypot(np.diff(cx, prepend=cx[0]), np.diff(cy, prepend=cy[0]))
    step_px[0] = 0.0
    step_cm = step_px / ppc_eff
    step_cm = np.nan_to_num(step_cm, nan=0.0)
    min_move = params["min_move_cm"]
    if min_move > 0:
        step_cm = np.where(step_cm < min_move, 0.0, step_cm)
    speed_cm_s = np.divide(step_cm, dt, out=np.zeros_like(step_cm), where=dt > 0)

    return Track(cx=cx, cy=cy, dt=dt, t_rel=t_rel, step_cm=step_cm,
                 speed_cm_s=speed_cm_s, duration_s=duration, ppc=ppc,
                 calibrated=calibrated, has_pose=has_pose)


def corrected_keypoints(sess: Session, params: dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Every keypoint after the detector-error corrections, ready to write out.

    ``build_track`` applies these to produce ONE trajectory and keeps the
    result in memory. This applies the same corrections to EVERY body part and
    hands them back, so a corrected pose stream can be written to disk and
    then looked at, in Verify, or by any other tool.

    The order matches ``_build_track_core`` exactly, because the point is that
    a corrected file measures the same as the in-memory correction did:

      1. head/body swap resolution across the named pair
      2. confidence gate, below the cut a keypoint is ABSENT, not a position
      3. teleport clamp, judged against the ORIGINAL neighbours
      4. optional rolling-median gap fill
      5. optional One-Euro smoothing

    Returns ``{body_part: {"x", "y", "conf"}}`` with NaN where a keypoint is
    absent, which is what the writer turns into a missing entry.
    """
    n = len(sess.ts_ms)
    if n == 0 or not sess.kp:
        return {}

    dt = dt_seconds(sess.ts_ms)
    t_s = sess.ts_ms / 1000.0
    ppc = params.get("px_per_cm_override") or sess.px_per_cm
    ppc_eff = ppc if ppc > 0 else 1.0

    kx = {bp: np.asarray(a["x"], float).copy() for bp, a in sess.kp.items()}
    ky = {bp: np.asarray(a["y"], float).copy() for bp, a in sess.kp.items()}
    kc = {bp: np.asarray(a["conf"], float).copy() for bp, a in sess.kp.items()}
    names = list(sess.kp.keys())

    # (1) swap resolution, the named pair, falling back to the first two.
    if params.get("swap_correct", DEFAULTS["swap_correct"]) and len(names) >= 2:
        _h, _b = default_swap_pair(names)
        h = params.get("head_part") or _h
        b = params.get("swap_body_part") or _b
        if h in kx and b in kx and h != b:
            _resolve_swaps(kx, ky, kc, h, b,
                           float(params.get("swap_cost_threshold", 0.3)))

    pcut = params.get("pcut", 0.55)
    max_px_s = params.get("max_speed_cm_s", 380.0) * ppc_eff
    gated: Dict[str, Dict[str, np.ndarray]] = {}

    for bp in names:
        # (2) gate: below the cut the keypoint is missing for that frame.
        keep = kc[bp] >= pcut
        x = np.where(keep, kx[bp], np.nan)
        y = np.where(keep, ky[bp], np.nan)

        # (3) teleport clamp, judged against the PRISTINE neighbours so one
        #     fast dart cannot cascade and freeze the whole track.
        ox, oy = x.copy(), y.copy()
        for i in range(1, n):
            if not (np.isfinite(ox[i]) and np.isfinite(ox[i - 1])):
                continue
            d = dt[i]
            if d > 0 and math.hypot(ox[i] - ox[i - 1],
                                    oy[i] - oy[i - 1]) / d > max_px_s:
                x[i], y[i] = x[i - 1], y[i - 1]

        if params.get("rolling_median"):
            x = _rolling_median(x, 5)
            y = _rolling_median(y, 5)

        gated[bp] = {"x": x, "y": y, "conf": np.where(keep, kc[bp], np.nan)}

    # (4) the same cleaning `build_track` measures through, interpolation
    #     across short holes, reconstruction of what is still missing from the
    #     rest of the animal, and a centred smoother.
    #
    #     ONE chain, shared with `build_track`, so the file on disk measures
    #     the same as the report beside it. Stopping at step 3 here instead
    #     leaves the written recording merely gated while the reported numbers
    #     come from a cleaned trace, on this rig a 14-30% gap in every
    #     keypoint, invisible because both halves look internally consistent.
    cleaned, _reports = _rf.refine_all(gated, t_s, params)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for bp in names:
        c = cleaned.get(bp, gated[bp])
        out[bp] = {"x": c["x"], "y": c["y"], "conf": gated[bp]["conf"]}
        if "filled" in c:
            # Carried through so a reader can tell a computed position from a
            # measured one. Nothing that is written out may look like a
            # detection when it is an inference from the other keypoints.
            out[bp]["filled"] = c["filled"]
    return out


# ───────────────────────────── core algorithms ───────────────────────────────

#: The bout and dwell detectors, and the rest of the statistics, live in
#: `measures`. They are re-exported here because scripts, the HTML report and
#: a good deal of the test suite call them as `oa.dwell_entries(...)`: and a
#: name that moves is a name that gets re-implemented.
dwell_entries = ms.dwell_entries
freeze_bouts = ms.freeze_bouts


def zone_names(sess: Session) -> List[str]:
    out = [z["name"] for z in sess.zones
           if isinstance(z, dict) and z.get("type") != "scale" and z.get("name")]
    if not out:
        out = sorted({z for z in sess.location if z and z not in ("", "na", "none", "None")})
    return out


def _region_parts(regions, zones):
    """The zones each region actually covers, in this recording.

    A region whose parts are all absent is dropped rather than reported as
    zeros, a column of zeros reads as "the animal never went there", which is
    a different claim from "this arena has no such region".
    """
    out = {}
    for region, parts in (regions or {}).items():
        present = [z for z in parts if z in zones]
        if present:
            out[str(region)] = present
    return out


def _emit_regions(row: dict, sess: "Session", tr: "Track", params: dict,
                  zones, N: int, multi=(), part_tracks=None):
    """Sub-zones combined into the regions a paper actually reports.

    An experimenter tiles an elevated-plus maze with ``Open_1``/``Open_2`` so
    the mesh covers the arena; the methods section says "time in the open
    arms". The combined mask is the OR of the parts, so a frame in two of them
    is counted once and region time can never exceed the session.

    The region's own name is used verbatim, so ``Open`` yields ``Open_time_s``
    beside the parts' ``Open_1_time_s``, both are available, and which one is
    reported is the user's call, not ours.
    """
    regions = _region_parts(params.get("regions"), zones)
    if not regions:
        return
    want_time = _want(params, "zone_time", True)
    want_ent = _want(params, "zone_entries", True)
    want_lat = _want(params, "zone_latency", False)
    want_dist = _want(params, "zone_distance", True)

    for region, parts in regions.items():
        inz = np.zeros(N, dtype=bool)
        for z in parts:
            m = _zone_mask(sess, z, N)
            inz |= (m if len(m) == N else np.resize(m, N))
        e, _ex, lat, tt, _visits = dwell_entries(inz, tr.dt,
                                                 params["min_dwell_s"])
        if want_time:
            row[f"{region}_time_s"] = tt
        if want_ent:
            row[f"{region}_entries"] = e
        if want_lat:
            row[f"{region}_latency_s"] = lat if not math.isnan(lat) else np.nan
        if want_dist:
            _emit_zone_distance(row, tr, inz, region, "")
            for part in multi:
                _emit_zone_distance(row, (part_tracks or {})[part], inz,
                                    region, part)


def _zone_mask(sess: Session, zone: str, n: int) -> np.ndarray:
    """Per-frame in-zone boolean, from the single recorded ``location``."""
    return np.asarray(sess.location, dtype=object) == zone


# ──────────────────────────── metric assembly ────────────────────────────────

def _emit_locomotion(row: dict, tr: "Track", prefix: str):
    """Distance + mean/max speed for one track, in metres (px when uncalibrated).
    Values are UNROUNDED (full float precision), only Duration_s is rounded."""
    dist = ms.distance_cm(tr.step_cm)
    mean = ms.mean_speed_cm_s(tr.step_cm, tr.dt)
    peak = ms.peak_speed_cm_s(tr.speed_cm_s)
    if tr.calibrated:
        row[f"{prefix}Distance_m"] = dist / 100.0
        row[f"{prefix}Mean_Speed_m_s"] = mean / 100.0
        row[f"{prefix}Max_Speed_m_s"] = peak / 100.0
    else:
        row[f"{prefix}Distance_px"] = dist
        row[f"{prefix}Mean_Speed_px_s"] = mean
        row[f"{prefix}Max_Speed_px_s"] = peak


def _emit_zone_distance(row: dict, tr: "Track", inside: np.ndarray, zone: str, part: str):
    """Distance travelled by one body point WHILE inside `zone`, in metres (px when
    uncalibrated). With multiple body points ticked, `part` names the point:
    ``Left_arm_Center_distance_m``; with a single point `part` is empty →
    ``Left_arm_distance_m``. Zone/body-point name clashes are handled upstream by
    tagging the ZONE (``Center`` → ``Center_arm``), so no body_ prefix is needed.

    ATTRIBUTION: each per-frame step (displacement from the previous frame) is
    credited to the zone the animal is in AT THAT FRAME (arrival). Because every
    frame has exactly one location, the per-zone distances + the distance spent
    outside all zones sum EXACTLY to the total distance, no boundary-crossing
    travel is silently dropped. (The reference's both-endpoints rule discarded it.)"""
    tag = f"{part}_" if part else ""
    dist = ms.distance_cm(tr.step_cm, inside)
    if tr.calibrated:
        row[f"{zone}_{tag}distance_m"] = dist / 100.0
    else:
        row[f"{zone}_{tag}distance_px"] = dist


def _round4(row: dict) -> dict:
    """Round every float value to 4 decimal places, in place. Ints (e.g. Duration_s,
    counts) and non-numerics are left untouched; NaN stays NaN."""
    for k, v in row.items():
        if isinstance(v, (float, np.floating)) and not isinstance(v, bool):
            row[k] = round(float(v), 4)
    return row


def session_metrics(sess: Session, tr: Track, params: dict) -> dict:
    row: dict = {"Subject": sess.subject or sess.stem, "Date": sess.date,
                 "Start_time": sess.start_time, "Stage": sess.stage,
                 "File": os.path.basename(sess.path),
                 "Duration_s": int(math.ceil(tr.duration_s)), "px_per_cm": tr.ppc}
    if not tr.has_pose:
        # A session recorded without tracking is not necessarily unanalysable.
        # The rig writes which zone the animal was in on every frame, and zone
        # time, entries, latency and transitions are all computable from that
        # column alone. What genuinely needs coordinates, distance, speed,
        # immobility, freezing, distance travelled inside a zone, is switched
        # off here, once, so every `_want` below and in `_emit_regions` agrees
        # about what this recording can answer.
        recorded_zoning = any(
            z and str(z).strip().lower() not in ("na", "none", "")
            for z in (sess.location or ()))
        if not recorded_zoning:
            row["Note"] = "no pose data (na), geometry/scale only"
            return _round4(row)
        row["Note"] = ("no pose data, zone occupancy from the recording's "
                       "own location column; distance and speed need "
                       "coordinates")
        params = {**params, "metrics": {
            **(params.get("metrics") or DEFAULTS["metrics"]),
            "locomotion": False, "immobility": False, "freezing": False,
            "zone_distance": False, "zone_advanced": False,
            "head_zones": False, "interaction": False}}
    # Per-part tracks built ONCE here and reused for both whole-session locomotion
    # and per-zone distance. `multi` = the body points the user ticked (Head/Center/
    # Tailbase…); empty → single primary point already in `tr`.
    # `primary` = the body point that produces the single cumulative Distance_m. A ticked
    # breakdown point equal to the primary is skipped so it is never reported twice.
    primary = params.get("body_part") or ""
    multi = [p for p in (params.get("body_parts_multi") or []) if p in sess.kp and p != primary]
    part_tracks = {p: build_track(sess, {**params, "body_part": p}) for p in multi}

    # Everything in METRES when calibrated (px when not). No cm anywhere.
    if _want(params, "locomotion", True):
        # PRIMARY trajectory (the Centre keypoint by default) → the one cumulative Distance_m.
        _emit_locomotion(row, tr, "")
        # OPTIONAL per-part breakdown for OTHER ticked points → Head_Distance_m, Tailbase_Distance_m…
        for part in multi:
            _emit_locomotion(row, part_tracks[part], f"{part}_")

    if _want(params, "immobility", True):
        imt, ibouts, _ = freeze_bouts(tr.speed_cm_s, tr.dt, params["immobility_cm_s"], params["min_freeze_s"])
        row["Immobile_time_s"] = imt
        row["Immobile_bouts"] = ibouts

    # Freezing, stricter than immobility (ANY-maze: movement suppression except
    # respiration). Own (lower) speed cut; reports episodes + onset latency.
    if _want(params, "freezing", False):
        fzt, fzb, fzl = freeze_bouts(tr.speed_cm_s, tr.dt, params["freeze_cm_s"], params["min_freeze_s"])
        row["Freezing_time_s"] = fzt
        row["Freezing_episodes"] = fzb
        row["Freezing_latency_s"] = fzl if not math.isnan(fzl) else np.nan

    N = len(sess.ts_ms)
    zones = zone_names(sess)
    include = params.get("zones_include")            # None/empty → all zones
    if include:
        zones = [z for z in zones if z in include]

    if _want(params, "transitions", True):
        row["Transitions_total"] = ms.transitions_total(sess.location, include)

    want_time = _want(params, "zone_time", True)
    want_ent = _want(params, "zone_entries", True)
    want_lat = _want(params, "zone_latency", False)
    want_dist = _want(params, "zone_distance", True)
    want_adv = _want(params, "zone_advanced", False)
    if want_time or want_ent or want_lat or want_dist or want_adv:
        for z in zones:
            inz = _zone_mask(sess, z, N)
            e, ex, lat, tt, visits = dwell_entries(inz, tr.dt, params["min_dwell_s"])
            if want_time:
                row[f"{z}_time_s"] = tt
            if want_ent:
                row[f"{z}_entries"] = e
            if want_lat:
                row[f"{z}_latency_s"] = lat if not math.isnan(lat) else np.nan
            if want_dist:
                # distance travelled INSIDE this zone/arm: primary (centroid) always,
                # plus one column per ticked body point when a breakdown is requested.
                _emit_zone_distance(row, tr, inz, z, "")
                for part in multi:
                    _emit_zone_distance(row, part_tracks[part], inz, z, part)
            if want_adv:
                row[f"{z}_exits"] = ex
                row[f"{z}_mean_visit_s"] = float(np.mean(visits)) if visits else 0.0

        _emit_regions(row, sess, tr, params, zones, N, multi, part_tracks)

    if _want(params, "ymaze"):
        _ymaze(row, sess, tr, params)
    return _round4(row)


def _ymaze(row: dict, sess: Session, tr: Track, params: dict):
    novel, fam, start = params["novel_arm"], params["familiar_arm"], params["start_arm"]
    if not (novel or fam):
        return
    loc = np.asarray(sess.location, dtype=object)
    tn = ms.time_in_s(loc == novel, tr.dt) if novel else 0.0
    tf = ms.time_in_s(loc == fam, tr.dt) if fam else 0.0
    if novel:
        row["NovelArm_time_s"] = round(tn, 1)
        e, ex, lat, tt, v = dwell_entries(loc == novel, tr.dt, params["min_dwell_s"])
        row["NovelArm_entries"] = e
        row["NovelArm_latency_s"] = round(lat, 1) if not math.isnan(lat) else np.nan
    if fam:
        row["FamiliarArm_time_s"] = round(tf, 1)
    if novel and fam and (tn + tf) > 0:
        row["NovelArm_preference_pct"] = round(100 * tn / (tn + tf), 1)
        row["Discrimination_index"] = round((tn - tf) / (tn + tf), 3)
    arms = [a for a in (novel, fam, start) if a]
    seq = []; prev = None
    for z in sess.location:
        if z in arms and z != prev:
            seq.append(z)
        if z in arms:
            prev = z
    if len(seq) >= 3:
        alts = sum(1 for i in range(2, len(seq)) if len({seq[i], seq[i-1], seq[i-2]}) == 3)
        row["Arm_entries"] = len(seq)
        row["Spontaneous_alternation_pct"] = round(100 * alts / (len(seq) - 2), 1)


# ──────────────────── identity, staging and the head point ───────────────────
#
# A protocol with fourteen stages is fourteen result rows, not one. Every
# behavioural paper reports per-phase measures, habituation vs test, baseline
# vs shock, sample vs choice, and a session-wide mean of those is not a
# number anyone can use. The recorder writes the stage on every row; this is
# where that finally becomes the unit of analysis.

#: Columns that identify WHICH animal, in WHICH group, on WHICH day, in WHICH
#: phase. They lead every table so a cohort's workbooks stack into one sheet
#: that a stats package can read without reshaping.
IDENTITY_COLUMNS = ("Subject", "Group", "Sex", "Cage", "Subgroup",
                    "ExptDate", "StartTime", "Stage")

_META_ALIASES = {
    "group": "Group", "grp": "Group", "sex": "Sex", "cage": "Cage",
    "subgroup": "Subgroup", "sub_group": "Subgroup", "run": "Subgroup",
    "genotype": "Group", "treatment": "Group",
}


def _filename_metadata(stem: str) -> Dict[str, str]:
    """Date, time and a subject guess from the file name.

    Rigs name files by convention rather than by contract, so this is a
    fallback under the header's own metadata, never an override of it.
    """
    import re

    out: Dict[str, str] = {}
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", stem)
    if m:
        out["ExptDate"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(?:^|[-_])(\d{2})(\d{2})(\d{2})(?:$|[-_])", stem[10:] or stem)
    if m:
        out["StartTime"] = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    head = re.split(r"[-_]", stem)[0].strip()
    if head:
        out["Subject"] = head
    return out


def identity_columns(sess: Session, params: dict) -> Dict[str, Any]:
    """The leading columns of every row, header first and file name second."""
    meta: Dict[str, Any] = {}
    hdr = getattr(sess, "header", None)
    raw = dict(getattr(hdr, "metadata", {}) or {}) if hdr is not None else {}
    for k, v in raw.items():
        key = _META_ALIASES.get(str(k).strip().lower())
        if key and str(v).strip():
            meta[key] = v
    guess = _filename_metadata(sess.stem)
    row: Dict[str, Any] = {}
    row["Subject"] = sess.subject or guess.get("Subject", "") or sess.stem
    for col in ("Group", "Sex", "Cage", "Subgroup"):
        row[col] = meta.get(col, "")
    row["ExptDate"] = sess.date or guess.get("ExptDate", "")
    row["StartTime"] = sess.start_time or guess.get("StartTime", "")
    row["Stage"] = sess.stage
    return row


def trim_span(sess: "Session", lo: int, hi: int, params: dict):
    """Narrow a span to the requested time window.

    The window is measured from the START of the span, so "the first five
    minutes" means the same thing whether it is applied to a whole recording
    or to each stage of a protocol, which is what a methods section means by
    it. Returns the span unchanged when no window is set.
    """
    start = float(params.get("trim_start_s", 0.0) or 0.0)
    end = float(params.get("trim_end_s", 0.0) or 0.0)
    if start <= 0 and end <= 0:
        return lo, hi
    ts = np.asarray(sess.ts_ms, float)
    if ts.size == 0 or lo >= hi:
        return lo, hi
    t0 = ts[lo]
    a, b = lo, hi
    if start > 0:
        want = t0 + start * 1000.0
        a = int(np.searchsorted(ts[lo:hi], want, side="left")) + lo
    if end > 0:
        want = t0 + end * 1000.0
        b = int(np.searchsorted(ts[lo:hi], want, side="right")) + lo
    a = max(lo, min(a, hi))
    b = max(a, min(b, hi))
    return a, b


def stage_spans(sess: Session) -> List[Tuple[str, int, int]]:
    """Contiguous ``(stage, lo, hi)`` runs over the frames.

    Contiguous, not grouped-by-name: a protocol that returns to a stage runs
    it twice, and merging those would average two different bits of behaviour
    into one row.
    """
    stages = getattr(sess, "stage_series", None) or []
    n = len(sess.ts_ms)
    if not stages or len(stages) < n:
        return [(sess.stage, 0, n)]
    spans: List[Tuple[str, int, int]] = []
    lo = 0
    for i in range(1, n + 1):
        if i == n or stages[i] != stages[lo]:
            spans.append((stages[lo], lo, i))
            lo = i
    return spans


def distinct_stages(sess: Session) -> List[str]:
    return sorted({s for s, _, _ in stage_spans(sess) if s})


def states_interleave(sess: Session, *, revisits: int = 3) -> bool:
    """Whether this recording's states are visited repeatedly, not once each.

    A protocol has phases: habituation, then trial, then probe, each entered
    once, each lasting minutes, and merging two of them would average two
    different bits of behaviour. A trial-structured task has states: it
    returns to ``choice_state`` four hundred times for a fraction of a second
    apiece, and reporting four hundred rows of half a second is no more useful
    than reporting one row of the whole hour.

    They are told apart by revisits, because that is the thing that actually
    differs, not by duration, which varies by assay.
    """
    named = [(s, lo, hi) for s, lo, hi in stage_spans(sess) if s]
    distinct = {s for s, _, _ in named}
    if not distinct:
        return False
    return len(named) >= revisits * len(distinct)


def state_index(sess: Session) -> "Dict[str, np.ndarray]":
    """``{state: frame indices}``, states in the order they first appear.

    First-appearance order rather than alphabetical: the rows then read in the
    order the animal met them, which is how the task is described.
    """
    order: List[str] = []
    found: Dict[str, List[int]] = {}
    for i, name in enumerate(getattr(sess, "stage_series", None) or []):
        if not name:
            continue
        if name not in found:
            found[name] = []
            order.append(name)
        found[name].append(i)
    return {name: np.asarray(found[name], dtype=int) for name in order}


def subset_session(sess: Session, idx: "np.ndarray", stage: str = "") -> Session:
    """A Session over an arbitrary set of frames, in their recorded order.

    ``slice_session``'s non-contiguous sibling. A grouped state is every frame
    the animal spent in it, scattered through the recording, so a contiguous
    ``[lo:hi]`` cannot express it.
    """
    from dataclasses import fields as _dc_fields

    idx = np.asarray(idx, dtype=int)
    out = type(sess)(**{f.name: getattr(sess, f.name) for f in _dc_fields(Session)})
    out.ts_ms = np.asarray(sess.ts_ms)[idx]
    out.location = [sess.location[i] for i in idx]
    series = getattr(sess, "stage_series", None) or []
    out.stage_series = [series[i] for i in idx] if series else []
    out.stage = stage or sess.stage
    out.kp = {bp: {k: np.asarray(v)[idx] for k, v in ch.items()}
              for bp, ch in sess.kp.items()}
    return out


def subset_track(tr: Track, idx: "np.ndarray") -> Track:
    """The same frames of an ALREADY-BUILT track, intervals carried, not redone.

    This is the whole reason grouped states are computed this way rather than
    by re-tracking a subset. ``dt[i]`` is the interval the animal spent
    arriving at frame ``i``; select those intervals and they sum to the time
    spent in the state. Recompute them from the subset's own timestamps
    instead and every gap between two visits, whole minutes of the animal
    doing something else entirely, is counted as time inside the state.

    ``t_rel`` becomes elapsed time WITHIN the state, so a time bin over a
    grouped row means "the first minute spent in this state", which is the
    only reading of it that makes sense.
    """
    idx = np.asarray(idx, dtype=int)
    dt = np.asarray(tr.dt)[idx]
    return Track(
        cx=np.asarray(tr.cx)[idx], cy=np.asarray(tr.cy)[idx], dt=dt,
        t_rel=np.cumsum(dt) - (dt[0] if dt.size else 0.0),
        step_cm=np.asarray(tr.step_cm)[idx],
        speed_cm_s=np.asarray(tr.speed_cm_s)[idx],
        duration_s=float(np.nansum(dt)), ppc=tr.ppc,
        calibrated=tr.calibrated, has_pose=tr.has_pose)


def slice_session(sess: Session, lo: int, hi: int, stage: str = "") -> Session:
    """A Session over frames ``[lo, hi)``, the unit one result row covers."""
    from dataclasses import fields as _dc_fields

    kind = type(sess)
    out = kind(**{f.name: getattr(sess, f.name) for f in _dc_fields(Session)})
    out.ts_ms = sess.ts_ms[lo:hi]
    out.location = list(sess.location[lo:hi])
    out.stage_series = list((getattr(sess, "stage_series", None) or [])[lo:hi])
    out.stage = stage or sess.stage
    out.kp = {bp: {k: v[lo:hi] for k, v in ch.items()}
              for bp, ch in sess.kp.items()}
    return out


#: Names a head keypoint goes by. Investigation is scored on the nose, not the
#: centroid, in every assay that scores it at all.
HEAD_ALIASES = ("head", "nose", "snout", "headcentre", "head_center")

#: The far end. A swap is a head/TAIL confusion, so the resolver needs the two
#: ends of the animal, in this order of preference.
BODY_ALIASES = ("tail_base", "tailbase", "tail", "tail1", "body",
                "mid_back", "center", "centre", "mouse_center", "back")


def default_swap_pair(names):
    """The two keypoints a swap resolver should compare, chosen by NAME.

NOT ``names[0]`` and ``names[1]``, the first two in the file's own order.
    For a six-node SLEAP model those are ``Snout`` and ``Head``: two points a
    centimetre apart at the SAME end of the animal, whose frame-to-frame
    displacements are near-identical, so the swap test compares noise and fires
    on ordinary fast motion, which lengthens the measured path rather than
    correcting it.

    Picking by name gives the two ENDS, ``Snout`` and ``Tail_Base``, which
    is the confusion that actually happens. Returns ``("", "")`` when the part
    names do not identify both ends, and the caller then leaves the resolver
    off rather than guessing.
    """
    def _find(aliases):
        lowered = {str(n).strip().lower().replace(" ", "_"): n for n in names}
        for alias in aliases:
            if alias in lowered:
                return lowered[alias]
        return ""

    head = _find(HEAD_ALIASES)
    body = _find(BODY_ALIASES)
    if not head or not body or head == body:
        return ("", "")
    return (head, body)


def head_part_of(sess: Session, params: dict) -> str:
    want = params.get("head_part") or ""
    if want and want in sess.kp:
        return want
    for bp in sess.kp:
        if bp.strip().lower().replace(" ", "_") in HEAD_ALIASES:
            return bp
    return ""


def _space_with_analysed_names(space, sess: Session):
    """The given arena, with its zones named the way this analysis names them.

    :func:`_disambiguate_zones` renames a colliding zone in the SESSION. The
    bundle hands its own :class:`SpaceModel` over for measuring, and that kept
    the recorded name, so the head-point and investigation columns for such a
    zone came out under a name nothing else used: on the Y-maze recordings the
    body columns said ``Center_arm`` while the head columns said ``Center``,
    and the zone filter (which holds every zone) dropped the latter outright.

    Returns the space unchanged when no zone collides, so the common case
    copies nothing.
    """
    import copy as _copy

    names = [z.get("name") for z in space.zones
             if isinstance(z, dict) and z.get("name")]
    rename = zone_rename_map(names, sess.body_parts)
    if not rename:
        return space
    out = _copy.copy(space)
    out.zones = [{**z, "name": rename[z["name"]]}
                 if isinstance(z, dict) and z.get("name") in rename else z
                 for z in space.zones]
    return out


def space_of(sess: Session, params: dict):
    """The arena this session is measured in, as a
    :class:`~source.analysis.space.SpaceModel`."""
    from tools.offline_analysis.engine import space as _sp

    override = params.get("space")
    if override is not None:
        return _space_with_analysed_names(override, sess)
    fs = sess.resolution if sess.resolution and sess.resolution[0] else (0, 0)
    zones = _sp.zones_to_pixel(sess.zones, fs)
    ppc = params.get("px_per_cm_override") or sess.px_per_cm
    return _sp.SpaceModel(frame_size=fs, zones=zones, px_per_cm=float(ppc or 0),
                          interaction=params.get("interaction") or {})


def _point_series(sess: Session, part: str, params: dict):
    """(x, y) for one keypoint, confidence-gated like the primary track."""
    ch = sess.kp.get(part)
    n = len(sess.ts_ms)
    if not ch:
        return np.full(n, np.nan), np.full(n, np.nan)
    pcut = params["pcut"]
    keep = ch["conf"] >= pcut
    return (np.where(keep, ch["x"], np.nan), np.where(keep, ch["y"], np.nan))


def _emit_head_and_interaction(row: dict, sess: Session, tr: Track,
                               params: dict) -> None:
    """Head-point zone occupancy, and investigation of interaction zones.

    Two things the live pipeline cannot produce and every object/social assay
    needs: where the *nose* was, and whether the animal was oriented toward
    the object while it was there. Time in an interaction area is not
    investigation on its own, an animal walking past a cup is in the area and
    is not investigating it.
    """
    from tools.offline_analysis.engine import space as _sp

    head = head_part_of(sess, params)
    if not head:
        return
    space = space_of(sess, params)
    if not space.zones:
        return
    hx, hy = _point_series(sess, head, params)
    want_head = _want(params, "head_zones", True)
    want_ia = _want(params, "interaction", True)
    if not (want_head or want_ia):
        return

    res = _sp.rezone(space, body_xy=(tr.cx, tr.cy), head_xy=(hx, hy),
                     dt=tr.dt, min_dwell_s=params["min_dwell_s"])
    include = params.get("zones_include")
    for zname in res.names:
        if include and zname not in include:
            continue
        if want_head:
            m = res.head.get(zname)
            if m is not None:
                row[f"{zname}_time_head_s"] = round(ms.time_in_s(m, tr.dt), 3)
                row[f"{zname}_entries_head"] = _sp.count_entries(
                    m, tr.dt, params["min_dwell_s"])
    if not want_ia:
        return
    for zname, ia in res.interaction.items():
        if include and zname not in include:
            continue
        row[f"{zname}_investigation_s"] = round(ia.facing_time_s, 3)
        row[f"{zname}_investigation_entries"] = ia.facing_entries
        row[f"{zname}_head_in_zone_s"] = round(ia.time_s, 3)

    # Discrimination index over exactly two interaction zones, the standard
    # novel-object measure, and meaningless with any other number of objects.
    ias = [z for z in res.interaction if not include or z in include]
    if len(ias) == 2:
        a, b = sorted(ias)
        ta = res.interaction[a].facing_time_s
        tb = res.interaction[b].facing_time_s
        if (ta + tb) > 0:
            row["Investigation_total_s"] = round(ta + tb, 3)
            row["Discrimination_index"] = round((tb - ta) / (ta + tb), 4)
            row["Preference_pct"] = round(100 * tb / (ta + tb), 2)


def _merge_bins(row: dict, sess: Session, tr: Track, params: dict) -> None:
    """Put this row's time bins into the row.

    Here, and not returned beside the rows for a caller to merge: a caller
    that forgets computes every bin and drops all of them, which is what
    ticking "Time bins" then looks like. Merging at the point the row is built
    means a row cannot be emitted without them.
    """
    if not wants_bins(params):
        return
    for k, v in time_bins(sess, tr, params).items():
        if k not in ("Subject", "Stage", "File"):
            row[k] = v


def _wants_grouped_states(sess: Session, params: dict) -> bool:
    """Whether to report one row per state NAME rather than per span."""
    mode = str(params.get("state_rows", "auto") or "auto").lower()
    if not bool(params.get("split_stages", True)):
        return False               # the operator asked for the whole session
    if mode == "grouped":
        return True
    if mode == "spans":
        return False
    return states_interleave(sess)


def group_state_rows(sess: Session, params: dict,
                     ident: Optional[dict] = None,
                     drop: Sequence[str] = ()) -> List[dict]:
    """One row per state NAME, every frame in it, wherever it occurred.

    The model this analysis is actually read by, and the one the rig's own
    recordings need: a trial-structured task enters ``choice_state`` four
    hundred times for a fraction of a second each, so per-visit rows are
    unmeasurable and a single whole-session row mixes the choice with the
    inter-trial interval.

    Every metric comes from the FULL recording's track, selected by frame
    (see :func:`subset_track`), so the per-state durations sum to the session
    duration and the per-state distances sum to the session distance. That is
    the property that makes these rows trustworthy, and phase 11 asserts it.
    """
    ident = identity_columns(sess, params) if ident is None else ident
    lo, hi = trim_span(sess, 0, len(sess.ts_ms), params)
    if (lo, hi) != (0, len(sess.ts_ms)):
        sess = slice_session(sess, lo, hi, sess.stage)
    by_state = state_index(sess)
    if not by_state:
        return []
    tr = build_track(sess, params)
    rows: List[dict] = []
    skipped: List[str] = []
    for name, idx in by_state.items():
        if idx.size < 2:
            # One frame carries no interval, so nothing about it is
            # measurable. Named below rather than dropped in silence, a
            # state 2 ms long is entered hundreds of times and still never
            # sampled, and its absence should not look like a bug.
            skipped.append(f"{name} ({idx.size} frame)")
            continue
        sub = subset_session(sess, idx, name)
        sub_tr = subset_track(tr, idx)
        row = session_metrics(sub, sub_tr, params)
        _emit_head_and_interaction(row, sub, sub_tr, params)
        merged = {**ident, **{k: v for k, v in row.items() if k not in drop}}
        merged["Stage"] = name
        merged["State_frames"] = int(idx.size)
        _merge_bins(merged, sub, sub_tr, params)
        rows.append(_round4(merged))
    if skipped:
        logger.warning(
            "these states hold too few frames to measure and have no row: %s. "
            "A state shorter than the interval between frames is entered "
            "without ever being photographed.", ", ".join(skipped))
    return rows


def session_rows(sess: Session, params: dict) -> List[dict]:
    """One row per protocol stage, the shape results are reported in.

    A single-stage recording yields exactly one row, so nothing changes for
    sessions that have no protocol.
    """
    ident = identity_columns(sess, params)
    # `session_metrics` emits its own Subject/Date/Start_time. The identity
    # block supersedes them (it also reads the header's group/sex/cage), so
    # the older names are dropped rather than left beside their twins.
    drop = ("Date", "Start_time")
    spans = stage_spans(sess)
    named = [(s, lo, hi) for s, lo, hi in spans if s]
    if _wants_grouped_states(sess, params):
        rows = group_state_rows(sess, params, ident, drop)
        if rows:
            return rows
    split = bool(params.get("split_stages", True)) and len(named) >= 2

    if not split:
        a, b = trim_span(sess, 0, len(sess.ts_ms), params)
        if (a, b) != (0, len(sess.ts_ms)):
            sess = slice_session(sess, a, b, sess.stage)
        tr = build_track(sess, params)
        row = session_metrics(sess, tr, params)
        _emit_head_and_interaction(row, sess, tr, params)
        out = {**ident, **{k: v for k, v in row.items() if k not in drop}}
        out["Stage"] = sess.stage or ident.get("Stage", "")
        _merge_bins(out, sess, tr, params)
        return [_round4(out)]

    rows: List[dict] = []
    for stage, lo, hi in named:
        # Per stage, so "the first five minutes" means the first five minutes
        # OF EACH PHASE, which is what a protocol's methods section means.
        lo, hi = trim_span(sess, lo, hi, params)
        if hi - lo < 2:
            continue
        sub = slice_session(sess, lo, hi, stage)
        tr = build_track(sub, params)
        row = session_metrics(sub, tr, params)
        _emit_head_and_interaction(row, sub, tr, params)
        merged = {**ident, **{k: v for k, v in row.items() if k not in drop}}
        merged["Stage"] = stage
        merged["Stage_start_s"] = round(float(sess.ts_ms[lo] - sess.ts_ms[0]) / 1000.0, 3)
        # Bins are per ROW, so on a split recording "0-60s" is the first
        # minute OF THAT STAGE, the same thing the trim window means.
        _merge_bins(merged, sub, tr, params)
        rows.append(_round4(merged))
    if rows:
        return rows
    # Splitting found stage names but no stage long enough to measure, so it
    # would return NOTHING for a recording full of data. That happens on every
    # session from this rig: its `state` column is the MCU state machine's
    # state for that frame's interval, pipe-joined, blank between, changing
    # every frame or two, not a protocol phase that lasts minutes. Every span
    # is one frame long, every span is skipped, and a real hour of tracking
    # analysed to an empty table with no error.
    #
    # The whole session is the honest answer when the parts are unmeasurable.
    return session_rows(sess, {**params, "split_stages": False})


def order_row_columns(rows: List[dict]) -> List[str]:
    """Identity first, then the session block, then everything else.

    A stable order is not cosmetic: a cohort of workbooks only stacks into one
    analysable sheet if every file puts the same column in the same place.
    """
    session_block = ("File", "Duration_s", "Stage_start_s", "px_per_cm",
                     "Distance_m", "Distance_px", "Mean_speed_m_s",
                     "Max_speed_m_s", "Note")
    seen: List[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    lead = [c for c in IDENTITY_COLUMNS if c in seen]
    lead += [c for c in session_block if c in seen and c not in lead]
    return lead + [c for c in seen if c not in lead]


def _bint(v: float) -> str:
    """Whole seconds when they are whole, so 0-60s does not become 0.0-60.0s."""
    return str(int(v)) if float(v).is_integer() else ("%g" % v)


def bin_spans(duration_s: float, params: dict):
    """The windows `params` asks for. The rule itself is `measures.bin_spans`;
    this only reads the three keys the UI and the CLI speak."""
    return ms.bin_spans(duration_s,
                        edges=params.get("bin_edges_s") or None,
                        count=int(params.get("bin_count", 0) or 0),
                        width=float(params.get("bin_size_s", 0) or 0))


def bin_tags(spans, params: dict):
    """The name of each window, as it appears in the column.

    Seconds for a fixed width or explicit edges, "0-60s" means the same
    thing in every recording. But "split into 3" means the first THIRD, and a
    five-minute session and a two-minute one have different thirds: naming
    those by seconds gives `0-100s_Distance_m` in one row and `0-40s_` in the
    next, so the one mode whose whole purpose is comparing across unequal
    recordings produced columns that could not be compared. By ordinal, they
    line up.
    """
    if int(params.get("bin_count", 0) or 0) > 0 and not params.get("bin_edges_s"):
        n = len(spans)
        return [f"bin{i + 1}of{n}" for i in range(n)]
    return [f"{_bint(lo)}-{_bint(hi)}s" for lo, hi in spans]


def wants_bins(params: dict) -> bool:
    """Whether ANY of the three ways of asking was used."""
    return bool(params.get("bin_edges_s")
                or int(params.get("bin_count", 0) or 0) > 0
                or float(params.get("bin_size_s", 0) or 0) > 0)


def time_bins(sess: Session, tr: Track, params: dict) -> dict:
    """Per-bin columns: {lo}-{hi}s_{metric}. Wide (one row per session)."""
    out: dict = {"Subject": sess.subject or sess.stem, "Stage": sess.stage,
                 "File": os.path.basename(sess.path)}
    if not tr.has_pose or tr.duration_s <= 0:
        return out
    spans = bin_spans(tr.duration_s, params)
    if not spans:
        return out
    calib = tr.calibrated
    N = len(sess.ts_ms)
    zones = zone_names(sess)
    zmasks = {z: _zone_mask(sess, z, N) for z in zones}
    include = params.get("zones_include")
    if include:
        zones = [z for z in zones if z in include]
    tags = bin_tags(spans, params)
    for (lo, hi), tag in zip(spans, tags):
        m = ms.bin_mask(tr.t_rel, lo, hi)
        if not m.any():
            continue
        # The SAME functions the session row uses, a bin is the session over
        # fewer frames, and when the two were written out separately the mean
        # speed ended up with two different denominators.
        dist = ms.distance_cm(tr.step_cm, m)
        mean = ms.mean_speed_cm_s(tr.step_cm, tr.dt, m)
        if calib:
            out[f"{tag}_Distance_m"] = dist / 100.0
            out[f"{tag}_Mean_Speed_m_s"] = mean / 100.0
        else:
            out[f"{tag}_Distance_px"] = dist
            out[f"{tag}_Mean_Speed_px_s"] = mean
        for z in zones:
            zm = zmasks[z]
            e, ex, lat, tt, v = dwell_entries(zm & m, tr.dt, params["min_dwell_s"])
            out[f"{tag}_{z}_time_s"] = ms.time_in_s(m & zm, tr.dt)
            out[f"{tag}_{z}_entries"] = e
        # Partial means "too little of this window was recorded to compare it
        # with its siblings", not "the window ends one frame past the data".
        # `hi > tr.duration_s` marked the last bin of essentially every
        # recording: a 299.8 s session binned at 60 s has a final window
        # covering 59.8 s of 60, and flagging that made the flag meaningless.
        if ms.is_partial(ms.duration_s(tr.dt, m), hi - lo):
            out[f"{tag}_partial"] = 1
        # An ordinal name says nothing about WHEN. Carry the seconds beside
        # it so a figure can still be drawn against real time.
        if tag.startswith("bin"):
            out[f"{tag}_start_s"] = float(lo)
            out[f"{tag}_end_s"] = float(hi)
    return _round4(out)


# ──────────────────────────────── plots ──────────────────────────────────────

_MPL = None  # cache: None=unchecked, False=unavailable, (plt, LineCollection)=ready


#: Why matplotlib is unavailable, when it is. Distinguishing *absent* from
#: *broken* matters: one is "pip install matplotlib", the other is a broken
#: backend, and collapsing them into one silent False told you neither.
_MPL_ERROR: str = ""


def new_figure(*args, **kw):
    """A figure that belongs to its caller and to nobody else.

    NOT ``plt.subplots``. pyplot keeps a global registry of every figure it
    makes, so one built for the Plots tab lived until someone closed it,
    and, far worse, it was built on the WORKER thread while the GUI thread
    painted. That is a global structure mutated from two threads: the app
    died with an access violation inside a garbage collection, reproducibly,
    on the second recording of a run with plots ticked.

    A bare ``Figure`` has no registry and no manager. ``savefig`` still works
    (it attaches an Agg canvas on demand), which is all the CLI needs.
    """
    from matplotlib.figure import Figure

    return Figure(*args, **kw)


def _get_mpl():
    """Import matplotlib once per process (cached). Never imported on no-plot
    runs; imported once, not per file, when plotting.

    Returns ``(plt, LineCollection)`` or ``False``. On failure the reason is
    kept in :data:`_MPL_ERROR` so the caller can report it rather than
    producing no figures and saying nothing.

    ``plt`` here is used only for its stateless helpers, colour maps, norms.
    Figures come from :func:`new_figure`; see there for why.
    """
    global _MPL, _MPL_ERROR
    if _MPL is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            _MPL = (plt, LineCollection)
        except ImportError as e:
            _MPL, _MPL_ERROR = False, f"matplotlib is not installed ({e})"
        except Exception as e:
            _MPL, _MPL_ERROR = False, f"matplotlib failed to initialise: {e}"
    return _MPL


def plot_backend_status(params: dict) -> Tuple[str, str]:
    """``(state, detail)`` for the PNG figure path.

    ``state`` is one of ``ok`` / ``disabled`` / ``none_selected`` /
    ``unavailable``. A run that asks for plots and makes none must be able to
    say which of those happened, silently returning an empty list is how you
    get an analysis that "succeeded" with nothing to look at.
    """
    if not params.get("plots", True):
        return "disabled", "plots were turned off for this run"
    if not _selected_figures(params) and not params.get("figures", {}).get("group"):
        return "none_selected", "no figure panels are selected"
    if _get_mpl() is False:
        return "unavailable", _MPL_ERROR or "matplotlib unavailable"
    return "ok", ""


def _selected_figures(params: dict) -> List[str]:
    figsel = params.get("figures") or DEFAULTS["figures"]
    order = ["track", "heatmap", "zone_timeline", "distance_velocity"]
    return [k for k in order if figsel.get(k)]


def _frame_dims(sess: Session, tr: "Track") -> Tuple[float, float]:
    """Recover the tracking-frame (ROI) pixel size so NORMALISED zones overlay the
    pose correctly. The pose lives in ROI pixels but the txt doesn't store the ROI
    size, and the animal usually explores only part of the arena, so max(pose) is
    NOT the frame. Fit pose px against the recorded-zone centroids (px ≈ W·norm),
    which recovers the true frame even from partial exploration; fall back to the
    observed pose extent."""
    zc: Dict[str, Tuple[float, float]] = {}
    for z in sess.zones:
        if not isinstance(z, dict) or z.get("type") == "scale":
            continue
        pts = z.get("points") or []
        if not pts:
            continue
        norm = z.get("coord_space") == "normalized" or max(max(abs(p[0]), abs(p[1])) for p in pts) <= 1.5
        if norm:
            zc[z.get("name")] = (float(np.mean([p[0] for p in pts])),
                                 float(np.mean([p[1] for p in pts])))
    cx, cy = tr.cx, tr.cy
    px, cxn, py, cyn = [], [], [], []
    for i, loc in enumerate(sess.location):
        if loc in zc and i < len(cx) and np.isfinite(cx[i]) and np.isfinite(cy[i]):
            px.append(cx[i]); cxn.append(zc[loc][0]); py.append(cy[i]); cyn.append(zc[loc][1])

    def _fit(pv, cv):
        pv, cv = np.asarray(pv, float), np.asarray(cv, float)
        d = float(np.sum(cv * cv))
        return float(np.sum(pv * cv) / d) if d > 1e-9 else 0.0

    W = _fit(px, cxn) if len(px) >= 20 else 0.0
    H = _fit(py, cyn) if len(py) >= 20 else 0.0
    obs_w = float(np.nanmax(cx)) if np.isfinite(cx).any() else 1.0
    obs_h = float(np.nanmax(cy)) if np.isfinite(cy).any() else 1.0
    return (max(W, obs_w) or 1.0), (max(H, obs_h) or 1.0)   # frame must contain the pose


def build_figure(sess: Session, tr: Track, params: dict):
    """Build an AnyMaze/Noldus-style matplotlib Figure with ONLY the panels the
    user selected (`params['figures']`), laid out in a grid sized to the count.
    Returns the Figure (caller owns it, display in a Qt canvas or savefig), or
    None if matplotlib is unavailable / no pose / nothing selected.

    Panels: track coloured by speed · occupancy heatmap · zone-over-time
    timeline · cumulative distance + velocity."""
    mpl = _get_mpl()
    if not mpl:
        return None
    plt, LineCollection = mpl
    if not tr.has_pose:
        return None
    panels = _selected_figures(params)
    if not panels:
        return None
    valid = np.isfinite(tr.cx) & np.isfinite(tr.cy)
    if valid.sum() < 2:
        return None
    x, y = tr.cx[valid], tr.cy[valid]
    fw, fh = _frame_dims(sess, tr)             # true ROI frame so zones overlay the pose
    unit = "cm" if tr.calibrated else "px"

    def draw_zones(ax):
        for z in sess.zones:
            if not isinstance(z, dict) or z.get("type") == "scale":
                continue
            pts = z.get("points") or []
            if len(pts) < 3:
                continue
            norm = z.get("coord_space") == "normalized" or max(max(abs(p[0]), abs(p[1])) for p in pts) <= 1.5
            poly = [(p[0] * fw, p[1] * fh) if norm else (p[0], p[1]) for p in pts]
            poly.append(poly[0])
            xs, ys = zip(*poly)
            ax.plot(xs, ys, color="#888", lw=1.0, alpha=0.7)
            cx0 = sum(p[0] for p in poly[:-1]) / (len(poly) - 1)
            cy0 = sum(p[1] for p in poly[:-1]) / (len(poly) - 1)
            ax.text(cx0, cy0, z.get("name", ""), color="#555", fontsize=7,
                    ha="center", va="center", alpha=0.8)

    def panel_track(ax):
        sp = tr.speed_cm_s[valid]
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        vmax = np.percentile(sp[sp > 0], 95) if np.any(sp > 0) else 1.0
        lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(0, max(vmax, 1e-3)))
        lc.set_array(sp[1:]); lc.set_linewidth(1.2)
        draw_zones(ax); ax.add_collection(lc)
        ax.set_xlim(0, fw); ax.set_ylim(fh, 0); ax.set_aspect("equal")
        ax.set_title("Track (colour = speed)"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(lc, ax=ax, fraction=0.046, label=f"speed ({unit}/s)")

    def panel_heatmap(ax):
        from matplotlib.colors import LinearSegmentedColormap
        # more time spent → RED (black background, red hotspots)
        cmap = LinearSegmentedColormap.from_list(
            "occ", ["#0a0a12", "#4a0a0a", "#a01010", "#ff2b2b"])
        nb = int(params.get("heatmap_bins", 40) or 40)
        # bin over the FULL frame so the image lines up with the zone overlay
        H, xe, ye = np.histogram2d(x, y, bins=nb, weights=tr.dt[valid],
                                   range=[[0, fw], [0, fh]])
        im = ax.imshow(H.T, origin="upper", extent=[0, fw, fh, 0],
                       cmap=cmap, aspect="equal", interpolation="gaussian")
        draw_zones(ax)
        ax.set_xlim(0, fw); ax.set_ylim(fh, 0)
        ax.set_title("Occupancy heatmap"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="time (s)")

    def panel_zone_timeline(ax):
        zones = zone_names(sess)
        cmap = plt.get_cmap("tab10")
        zc = {z: cmap(i % 10) for i, z in enumerate(zones)}
        loc = sess.location
        for i in range(len(tr.t_rel) - 1):
            z = loc[i]
            if z in zc:
                ax.axvspan(tr.t_rel[i], tr.t_rel[i + 1], color=zc[z], lw=0)
        ax.set_yticks([]); ax.set_xlim(0, tr.duration_s)
        ax.set_xlabel("time (s)"); ax.set_title("Zone occupancy over time")
        handles = [plt.Line2D([0], [0], color=zc[z], lw=6) for z in zones]
        ax.legend(handles, zones, loc="upper right", fontsize=7, ncol=2)

    def panel_distance_velocity(ax):
        cumd = np.cumsum(tr.step_cm)
        if tr.calibrated:
            cumd = cumd / 100.0; ylab = "cumulative distance (m)"
        else:
            ylab = "cumulative distance (px)"
        ax.plot(tr.t_rel, cumd, color="#1f77b4", label="distance")
        ax.set_xlabel("time (s)"); ax.set_ylabel(ylab, color="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(tr.t_rel, tr.speed_cm_s, color="#d62728", alpha=0.35, lw=0.6)
        ax2.set_ylabel(f"speed ({unit}/s)", color="#d62728")
        ax.set_title("Distance & velocity"); ax.set_xlim(0, tr.duration_s)

    drawers = {"track": panel_track, "heatmap": panel_heatmap,
               "zone_timeline": panel_zone_timeline,
               "distance_velocity": panel_distance_velocity}

    n = len(panels)
    cols = 1 if n == 1 else 2
    rows = int(math.ceil(n / cols))
    fig = new_figure(figsize=(6 * cols, 4.5 * rows))
    axarr = fig.subplots(rows, cols, squeeze=False)
    axes_flat = [axarr[r][c] for r in range(rows) for c in range(cols)]
    fig.suptitle(f"{sess.subject or sess.stem}, {sess.stage}",
                 fontsize=13, fontweight="bold")
    for ax, key in zip(axes_flat, panels):
        drawers[key](ax)
    for ax in axes_flat[n:]:                    # hide unused cells
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_plots(sess: Session, tr: Track, out_dir: str, params: dict) -> List[str]:
    """CLI/batch path: build the selected-panel figure and save it as a PNG.

    Returns ``[]`` when no figure could be made; :func:`plot_backend_status`
    says why.
    """
    out: List[str] = []
    fig = build_figure(sess, tr, params)
    if fig is not None:
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, f"{sess.stem}.png")
        fig.savefig(png, dpi=params.get("figure_dpi", 200))
        # No `plt.close`: the figure was never given to pyplot, so there is
        # nothing registered to close. Dropping the reference frees it.
        out.append(png)
    pdf = state_trajectory_pdf(sess, tr, out_dir, params)
    if pdf:
        out.append(pdf)
    return out


def state_trajectory_pdf(sess: Session, tr: Track, out_dir: str,
                         params: dict) -> str:
    """One page per state, the figure this analysis is actually read from.

    A single whole-session track is one tangle covering an hour; the same
    trajectory split by task state shows where the animal went while it was
    choosing and where it went while it was waiting, which is the comparison
    the experiment exists to make.

    Built by calling :func:`build_figure` once per state on a subset of the
    same track, so a page is drawn by exactly the code that draws the
    whole-session panel, and the per-state intervals are carried, never
    recomputed (see :func:`subset_track`).
    """
    if not params.get("state_pdf", True):
        return ""
    mpl = _get_mpl()
    if not mpl:
        return ""
    by_state = state_index(sess)
    if len(by_state) < 2:
        return ""            # nothing to compare, the single panel says it
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as e:                                # pragma: no cover
        logger.debug("no PDF backend (%s)", e)
        return ""

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sess.stem}_trajectory.pdf")
    pages = 0
    with PdfPages(path) as pdf:
        for name, idx in by_state.items():
            if idx.size < 2:
                continue
            sub = subset_session(sess, idx, name)
            sub_tr = subset_track(tr, idx)
            fig = build_figure(sub, sub_tr, params)
            if fig is None:
                continue
            seconds = float(np.nansum(sub_tr.dt))
            fig.suptitle(f"{sess.stem}, {name}   "
                         f"({idx.size} frames, {seconds:.0f} s)", fontsize=10)
            pdf.savefig(fig, dpi=params.get("figure_dpi", 200))
            pages += 1
    if not pages:
        os.remove(path)
        return ""
    logger.info("%s: %d-page trajectory PDF (%s)", sess.stem, pages,
                ", ".join(by_state))
    return path


# ──────────────────────────── parallel driver ────────────────────────────────

def _analyze_one(job: Tuple[str, dict, str]) -> dict:
    """Top-level worker (picklable), parse + track + metrics + bins + plots."""
    path, params, plot_dir = job
    try:
        sess = parse_txt(path)
        tr = build_track(sess, params)
        # One row per protocol stage. `rows[0]` is the whole session when the
        # recording has no stages, so every caller that wants a single summary
        # still gets one.
        rows = session_rows(sess, params)
        bins = time_bins(sess, tr, params) if wants_bins(params) else None
        trans = (transition_rows(sess, params)
                 if _want(params, "transitions", True) else [])
        plots = make_plots(sess, tr, plot_dir, params) if (params["plots"] and plot_dir) else []
        excluded = None
        if not tr.has_pose:
            excluded = {"File": os.path.basename(path),
                        "reason": (rows[0].get("Note") if rows else "no pose"),
                        "px_per_cm": sess.px_per_cm, "Stage": sess.stage}
        return {"summary": rows[0] if rows else None, "rows": rows,
                "bins": bins, "excluded": excluded, "plots": plots,
                "transitions": trans}
    except Exception as e:
        return {"summary": None, "rows": [], "bins": None,
                "excluded": {"File": os.path.basename(path), "reason": f"error: {e}"},
                "plots": [], "transitions": []}


def _input_record(path: str) -> dict:
    """Enough to prove which bytes produced a result.

    A workbook that cannot name its inputs cannot be reproduced, and a
    re-analysis six months later has no way to tell whether the file changed
    underneath it. Hash is truncated, this identifies a file, it is not a
    security boundary.
    """
    import hashlib
    out = {"file": os.path.basename(path), "path": os.path.abspath(path)}
    try:
        st = os.stat(path)
        out["bytes"] = st.st_size
        out["mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()[:16]
    except OSError as e:
        out["error"] = str(e)
    return out


#: The per-frame data files this analysis reads.
SESSION_GLOBS = ("*_video_data.txt",)


def collect_paths(inputs: List[str]) -> List[str]:
    """Every analysable session file under ``inputs``.

    Scanning is by :data:`SESSION_GLOBS`, so every entry point that takes a
    folder, CLI and tab alike, finds the same set of recordings.
    """
    paths: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            for pat in SESSION_GLOBS:
                paths += glob.glob(os.path.join(p, "**", pat), recursive=True)
        elif os.path.isfile(p):
            paths.append(p)
        else:
            paths += glob.glob(p, recursive=True)
    return sorted(set(paths))


def analyze_files(inputs: List[str], params: Optional[dict] = None,
                  plot_dir: str = "", progress=None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    paths = collect_paths(inputs)
    jobs = [(path, p, plot_dir) for path in paths]
    results: List[dict] = []
    n_jobs = p["n_jobs"] or max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, len(jobs)) or 1

    if n_jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        try:
            with mp.Pool(n_jobs) as pool:
                for i, r in enumerate(pool.imap_unordered(_analyze_one, jobs)):
                    results.append(r)
                    if progress:
                        progress(i + 1, len(jobs))
        except Exception:
            results = [_analyze_one(j) for j in jobs]     # fallback: serial
    else:
        for i, j in enumerate(jobs):
            results.append(_analyze_one(j))
            if progress:
                progress(i + 1, len(jobs))

    summary = [row for r in results for row in (r.get("rows") or [])]
    bins = [r["bins"] for r in results if r.get("bins")]
    excluded = [r["excluded"] for r in results if r.get("excluded")]
    plots = [pp for r in results for pp in r.get("plots", [])]
    transitions = [t for r in results for t in (r.get("transitions") or [])]
    # Stable order by file, then by where the stage started, so a protocol's
    # rows read in the order the animal experienced them.
    summary.sort(key=lambda d: (d.get("File", ""), d.get("Stage_start_s", 0.0)))
    bins.sort(key=lambda d: d.get("File", ""))
    state, detail = plot_backend_status(p)
    if state == "ok" and not plots and summary:
        # Asked for figures, backend fine, still nothing: the panels did
        # not apply to these sessions (degenerate tracks).
        state, detail = "no_figures", ("the selected panels produced nothing "
                                       "for these sessions")
    transitions.sort(key=lambda d: (d.get("File", ""), d.get("Stage", ""),
                                    d.get("From", ""), d.get("To", "")))
    return {"summary": summary, "bins": bins, "excluded": excluded,
            "plots": plots, "params": p, "n_files": len(paths), "n_jobs": n_jobs,
            "plot_state": state, "plot_detail": detail,
            "transitions": transitions,
            "inputs": [_input_record(path) for path in paths]}


def preview_one(path: str, params: Optional[dict] = None,
                plot_dir: str = "") -> dict:
    """Fast single-file analysis for the GUI live-preview (no pool overhead).
    Returns {summary, bins, plots, track, session} so the panel can show numbers
    AND the effect of de-jitter/thresholds immediately as the user tweaks them."""
    p = {**DEFAULTS, **(params or {})}
    sess = parse_txt(path)
    tr = build_track(sess, p)
    summary = session_metrics(sess, tr, p)
    bins = time_bins(sess, tr, p) if wants_bins(p) else None
    plots = make_plots(sess, tr, plot_dir, p) if (p["plots"] and plot_dir) else []
    return {"summary": summary, "bins": bins, "plots": plots,
            "raw_distance_cm": ms.distance_cm(tr.step_cm) if tr.has_pose else 0.0,
            "has_pose": tr.has_pose, "calibrated": tr.calibrated,
            "zones": zone_names(sess), "session": sess, "track": tr}


def ordered_columns(rows: List[dict]) -> List[str]:
    """Union of keys across rows, preserving first-seen order (stable for GUI)."""
    seen: List[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    return seen


# always-kept identity columns (never curated away, they identify the row)
ID_COLUMNS = IDENTITY_COLUMNS + ("Date", "Start_time", "File", "Duration_s")

#: Columns that describe the row rather than measure the animal. Bookkeeping,
#: not results: they are in every sheet whatever the user ticked.
NON_MEASURE_COLUMNS = ID_COLUMNS + ("Stage_start_s", "px_per_cm", "Note")


def measure_columns(cols: Sequence[str]) -> List[str]:
    """The columns that are an ANSWER, out of a result's full column list.

    The panel promised "30 measure columns" from this rule and then reported
    "37 measures" by counting the DataFrame's columns, identity, file name
    and scale included. Two numbers for the same run, neither of them wrong
    about what it counted, and no way to tell from the screen which was which.
    One definition, used by both.
    """
    return [c for c in cols if c not in NON_MEASURE_COLUMNS]


def apply_column_config(df, cfg: Optional[dict]):
    """Select (keep+order) then rename output columns. `cfg` = {'select':[...],
    'rename':{orig:new}}. Empty/absent select → keep all in natural order. ID
    columns are always retained. Operates on a pandas DataFrame, returns a copy."""
    if df is None or df.empty or not cfg:
        return df
    select = [c for c in (cfg.get("select") or []) if c]
    rename = cfg.get("rename") or {}
    out = df
    if select:
        keep = [c for c in ID_COLUMNS if c in out.columns]
        for c in select:
            if c in out.columns and c not in keep:
                keep.append(c)
        out = out[keep]
    if rename:
        out = out.rename(columns={k: v for k, v in rename.items() if v})
    return out


# ───────────────────────────── settings & output ─────────────────────────────

def save_settings(params: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_settings(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _meta_rows(res: dict) -> List[Tuple[str, Any]]:
    p = res["params"]
    import platform
    meta = [("tool", "pyBehaveTrack offline_analysis"), ("version", VERSION),
            ("analyzed_at", datetime.now().isoformat(timespec="seconds")),
            ("n_files", res.get("n_files")), ("n_analyzed", len(res["summary"])),
            ("n_excluded", len(res["excluded"])), ("n_jobs", res.get("n_jobs")),
            # Which figures were produced, or why none were. Without this the
            # workbook cannot distinguish "no plots wanted" from "plotting
            # silently unavailable on the machine that ran this".
            ("plot_state", res.get("plot_state", "")),
            ("plot_detail", res.get("plot_detail", "")),
            ("n_plots", len(res.get("plots", []))),
            # The environment. Two runs of the same data on different machines
            # can differ (numpy/scipy versions change filter output slightly),
            # and a result that cannot name its interpreter cannot be chased.
            ("python", platform.python_version()),
            ("platform", f"{platform.system()} {platform.machine()}"),
            ("numpy", getattr(__import__("numpy"), "__version__", "?"))]
    meta += [(k, (json.dumps(v) if isinstance(v, (dict, list)) else v))
             for k, v in sorted(p.items())]
    # Input record last: name, size, mtime and a short content hash per
    # file, so the result can be tied to the exact bytes it came from.
    for i, prov in enumerate(res.get("inputs") or [], 1):
        meta.append((f"input_{i}", json.dumps(prov)))
    return meta


def transition_rows(sess: "Session", params: dict) -> List[dict]:
    """Which zone the animal went to from which, and how often.

    ``Transitions_total`` says an animal changed zone 42 times. It cannot say
    whether those were Left↔Centre shuttles or a full tour of the maze, and
    that difference is the behaviour, a Y-maze alternation score, a
    place-preference bias and a perseverative loop all sit in this table and
    in none of the summary columns.

    Long form (one row per ordered pair) rather than N columns per zone: a
    10-zone arena is 90 pairs, which is a sheet, not a summary row. Pairs that
    never occurred are omitted rather than written as zeros.

    Self-transitions are impossible by construction, a run of the same zone
    is one visit, so From is never To.
    """
    include = params.get("zones_include")
    stages = stage_spans(sess)
    out: List[dict] = []
    for stage_name, i0, i1 in stages:
        counts: Dict[Tuple[str, str], int] = {}
        prev = None
        for z in list(sess.location)[i0:i1]:
            if not z or z in ("na", "none", "None"):
                continue
            if include and z not in include:
                continue
            if prev is not None and z != prev:
                counts[(prev, z)] = counts.get((prev, z), 0) + 1
            prev = z
        for (a, b), c in sorted(counts.items()):
            out.append({"File": os.path.basename(sess.path),
                        "Subject": sess.subject or "",
                        "Stage": stage_name, "From": a, "To": b, "Count": c})
    return out


def group_summary(summary: List[dict],
                  by: Sequence[str] = ("Group", "Stage")) -> List[dict]:
    """n, mean, SD and SEM per group for every numeric measure.

    A cohort workbook that stops at one row per animal still leaves the person
    who has to make the figure doing pivot tables by hand, and SEM computed
    in a spreadsheet is where the n gets it wrong when an animal is excluded.

    Grouping keys that are entirely blank are skipped, so an experiment with
    no Group column summarises by Stage rather than by "".
    """
    import pandas as pd

    if not summary:
        return []
    df = pd.DataFrame(summary)
    keys = [k for k in by
            if k in df.columns and df[k].astype(str).str.strip().any()]
    if not keys:
        return []
    num = [c for c in df.columns
           if c not in IDENTITY_COLUMNS and c not in ("File", "Stage")
           and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return []

    rows: List[dict] = []
    for vals, sub_df in df.groupby(keys, dropna=False):
        vals = vals if isinstance(vals, tuple) else (vals,)
        base = dict(zip(keys, [("" if pd.isna(v) else v) for v in vals]))
        base["n"] = int(len(sub_df))
        for c in num:
            col = sub_df[c].astype(float)
            mean = col.mean()
            sd = col.std(ddof=1)          # sample SD; a cohort is a sample
            cnt = int(col.notna().sum())
            base[f"{c}_mean"] = float(mean) if pd.notna(mean) else float("nan")
            base[f"{c}_SD"] = float(sd) if pd.notna(sd) else float("nan")
            # SEM uses the n that actually contributed, not the group size,
            # an excluded animal must not shrink the error bar.
            base[f"{c}_SEM"] = (float(sd) / (cnt ** 0.5)
                                if cnt > 1 and pd.notna(sd) else float("nan"))
            base[f"{c}_n"] = cnt
        rows.append(base)
    return rows


def write_workbook(res: dict, out_path: str):
    import pandas as pd
    cfg = (res.get("params") or {}).get("columns")
    sdf = apply_column_config(pd.DataFrame(res["summary"]), cfg)
    bdf = pd.DataFrame(res["bins"]) if res["bins"] else None
    tdf = (pd.DataFrame(res["transitions"])
           if res.get("transitions") else None)
    # `or None` on a DataFrame raises, its truthiness is ambiguous.
    grows = group_summary(res.get("summary") or [])
    gdf = pd.DataFrame(grows) if grows else None
    edf = pd.DataFrame(res["excluded"]) if res["excluded"] else None
    mdf = pd.DataFrame(_meta_rows(res), columns=["param", "value"])
    try:
        with pd.ExcelWriter(out_path) as w:
            mdf.to_excel(w, index=False, sheet_name="Meta")          # ALWAYS first
            sdf.to_excel(w, index=False, sheet_name="Session_Summary")
            if gdf is not None and not gdf.empty:
                gdf.to_excel(w, index=False, sheet_name="Group_Summary")
            if tdf is not None and not tdf.empty:
                tdf.to_excel(w, index=False, sheet_name="Transitions")
            if bdf is not None and not bdf.empty:
                bdf.to_excel(w, index=False, sheet_name="Time_Bins")
            if edf is not None and not edf.empty:
                edf.to_excel(w, index=False, sheet_name="Excluded")
    except Exception as e:
        base = os.path.splitext(out_path)[0]
        mdf.to_csv(base + "_meta.csv", index=False)
        sdf.to_csv(base + "_summary.csv", index=False)
        raise RuntimeError(f"xlsx write failed ({e}); wrote CSV fallback {base}_*.csv")
    sdf.to_csv(os.path.splitext(out_path)[0] + "_summary.csv", index=False)
    save_settings(res["params"], os.path.splitext(out_path)[0] + "_settings.json")


#: Scalar params the CLI exposes as --flags, derived from DEFAULTS rather than
#: hand-listed. Nested dicts (metrics/figures/columns) are set with --set
#: instead; ``n_jobs``/``plots`` have dedicated friendlier flags.
_CLI_SKIP = {"n_jobs", "plots", "metrics", "figures", "columns",
             "body_parts_multi", "zones_include"}


def _cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _add_default_args(ap) -> None:
    """One --flag per scalar DEFAULT.

    Generated, not hand-written: the previous CLI listed 13 of ~40 params, so
    most of them were reachable only by hand-writing a settings JSON. Driving
    this from DEFAULTS means a parameter added tomorrow is on the CLI tomorrow.
    """
    g = ap.add_argument_group("parameters (generated from DEFAULTS)")
    for key, val in sorted(DEFAULTS.items()):
        if key in _CLI_SKIP:
            continue
        flag = _cli_flag(key)
        if isinstance(val, bool):
            g.add_argument(flag, dest=key, action="store_true", default=None,
                           help=f"(default {val})")
            g.add_argument("--no-" + flag[2:], dest=key, action="store_false",
                           default=None, help=argparse.SUPPRESS)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            g.add_argument(flag, dest=key, type=type(val), default=None,
                           metavar=type(val).__name__.upper(),
                           help=f"(default {val})")
        elif isinstance(val, str):
            g.add_argument(flag, dest=key, type=str, default=None,
                           metavar="STR", help=f"(default {val!r})")


def _apply_set(params: dict, assignments: List[str]) -> None:
    """``--set metrics.freezing=true`` / ``--set figures.track=false``.

    The escape hatch for the nested groups, so nothing in DEFAULTS is
    unreachable from the command line.
    """
    for item in assignments or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except ValueError:
            val = raw
        node, *rest = key.split(".")
        if node not in DEFAULTS:
            raise SystemExit(f"--set: unknown parameter {node!r}")
        if not rest:
            params[node] = val
            continue
        target = params.setdefault(node, {})
        if not isinstance(target, dict):
            raise SystemExit(f"--set: {node!r} is not a group")
        target = dict(target)
        target[rest[0]] = val
        params[node] = target


#: Video suffixes the CLI accepts directly as inputs.
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def _wants_pipeline(a) -> bool:
    """Whether this invocation needs the bundle pipeline rather than the plain
    file scan: re-tracking, re-zoning, a supplied frame rate, or a bare video."""
    if a.track or a.zones or a.fps or a.scale_px_per_cm or a.dry_run:
        return True
    return any(str(p).lower().endswith(_VIDEO_EXTS) for p in a.inputs)


def _collect_bundle_inputs(inputs: List[str]) -> Tuple[List[str], List[str]]:
    """Split the inputs into session files and bare videos."""
    txts: List[str] = []
    vids: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            txts += collect_paths([p])
            for ext in _VIDEO_EXTS:
                vids += glob.glob(os.path.join(p, "**", "*" + ext), recursive=True)
        elif str(p).lower().endswith(_VIDEO_EXTS):
            vids.append(p)
        else:
            txts += collect_paths([p])
    return sorted(set(txts)), sorted(set(vids))


def _run_pipeline_cli(a, params: dict, plot_dir: str) -> dict:
    """The offline pipeline from the command line, the same code the tab runs."""
    from tools.offline_analysis.engine import pipeline as _pl
    from tools.offline_analysis.engine import space as _sp
    from tools.offline_analysis.engine.retrack2d import TrackSpec
    from tools.offline_analysis.engine.session_bundle import Bundle, find_txt_for_video

    txts, vids = _collect_bundle_inputs(a.inputs)
    claimed = {find_txt_for_video(v) for v in vids}
    bundles = [Bundle.open(t) for t in txts if t not in claimed]
    bundles += [Bundle.open(video_path=v) for v in vids]

    zone_dicts: List[dict] = []
    if a.zones:
        with open(a.zones, "r", encoding="utf-8") as f:
            from tools.offline_analysis import video_data_schema as vds
            zone_dicts = vds.zones_as_list(json.load(f))

    spec = None
    if a.track:
        spec = TrackSpec(backend=a.track, model_path=a.model,
                         confidence=float(params.get("pcut", 0.55)),
                         skip=max(1, int(a.skip)))
        problems = spec.validate()
        if problems:
            raise SystemExit("cannot track: " + "; ".join(problems))

    for b in bundles:
        if a.fps:
            b.set_user_fps(a.fps)
        if zone_dicts:
            fs = b.space.frame_size or (0, 0)
            pix = _sp.zones_to_pixel(zone_dicts, fs)
            ppc = a.scale_px_per_cm or _sp.px_per_cm_from_zones(pix, fs)
            b.set_space(_sp.SpaceModel(frame_size=fs, zones=pix,
                                       px_per_cm=float(ppc), origin="edited"))
        elif a.scale_px_per_cm:
            b.space.px_per_cm = float(a.scale_px_per_cm)
            b.set_space(b.space)
        if spec is not None:
            b.set_track(spec)

    p = dict(params)
    p["plots"] = bool(params.get("plots")) and bool(plot_dir)
    out = _pl.run_all(bundles, p, progress=lambda m: print("  " + m),
                      dry_run=a.dry_run)

    if a.publish:
        for b in bundles:
            try:
                print("  published:", b.publish())
            except ValueError:
                pass
    for w in out.get("warnings", []):
        print("  ! " + w)

    return {"summary": out["summary"], "bins": out["bins"],
            "excluded": out["excluded"] + [{"File": e} for e in out["errors"]],
            "plots": out["plots"], "params": p, "n_files": len(bundles),
            "n_jobs": 1, "plot_state": "ok" if out["plots"] else "off",
            "plot_detail": "", "inputs": [_input_record(b.active_txt())
                                          for b in bundles if b.active_txt()]}


def main(argv=None):
    # NOT ArgumentDefaultsHelpFormatter: every generated flag defaults to None
    # so that "unset" is distinguishable from "set to the default value", and
    # the formatter would print that None as if it were the default. The real
    # default is in each flag's help text.
    ap = argparse.ArgumentParser(
        description="Offline behavioural analysis")
    ap.add_argument("inputs", nargs="+", help="files, folders, or globs")
    ap.add_argument("-o", "--out", default="analysis.xlsx")
    ap.add_argument("--settings", default="",
                    help="load params from a saved settings.json")
    ap.add_argument("--set", dest="assignments", action="append", metavar="K=V",
                    help="set a nested param, e.g. --set metrics.freezing=true")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel workers (0=auto)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--html", default="", metavar="PATH",
                    help="also write a self-contained interactive HTML report "
                         "(default: alongside --out)")
    ap.add_argument("--no-html", action="store_true",
                    help="skip the HTML report")
    ap.add_argument("--dump-params", action="store_true",
                    help="print the effective parameters and exit")
    ap.add_argument("--validate", action="store_true",
                    help="check the statistics instead of running them: each "
                         "definition against a worked example, then the "
                         "identities that must hold on these recordings (the "
                         "bins add up to the totals, no zone is occupied for "
                         "longer than the session). Exits non-zero on a "
                         "failure. Same as python -m source.analysis.validate")
    # ── the bundle path: bare videos, re-tracking, re-zoning ──
    #
    # Everything the Analyze tab can do is reachable here, because both drive
    # the same `analysis.pipeline`. A headless rig or a batch script is not a
    # second-class citizen.
    g = ap.add_argument_group("tracking and geometry (offline pipeline)")
    g.add_argument("--track", metavar="BACKEND", default="",
                   help="re-run tracking: deeplabcut | sleap | blob")
    g.add_argument("--model", metavar="PATH", default="",
                   help="model directory for --track deeplabcut/sleap")
    g.add_argument("--fps", type=float, default=0.0,
                   help="frame rate for videos that carry no timestamps")
    g.add_argument("--zones", metavar="PATH", default="",
                   help="JSON zone file to analyse with (re-zones the poses)")
    g.add_argument("--scale-px-per-cm", type=float, default=0.0,
                   help="scale to use with --zones when they carry no scale line")
    g.add_argument("--skip", type=int, default=1,
                   help="track every Nth frame (gaps keep their true time)")
    g.add_argument("--publish", action="store_true",
                   help="also copy the retracked poses next to the original")
    g.add_argument("--dry-run", action="store_true",
                   help="run every step except the detector")
    _add_default_args(ap)
    a = ap.parse_args(argv)

    params = dict(DEFAULTS)
    if a.settings:
        params.update(load_settings(a.settings))
    # Only flags the user actually passed override the defaults/settings.
    for key in DEFAULTS:
        got = getattr(a, key, None)
        if got is not None and key not in _CLI_SKIP:
            params[key] = got
    _apply_set(params, a.assignments)
    params["n_jobs"] = a.jobs
    params["plots"] = not a.no_plots

    if a.dump_params:
        print(json.dumps(params, indent=2, sort_keys=True))
        return

    if a.validate:
        # Deliberately BEFORE any work: "are these numbers trustworthy" is a
        # question you ask about the recordings you are holding, not one you
        # answer after writing a workbook from them.
        from tools.offline_analysis.engine import validate as _vd

        raise SystemExit(_vd.report(_vd.run(_vd.expand(a.inputs))))

    out_dir = os.path.dirname(os.path.abspath(a.out)) or "."
    plot_dir = os.path.join(out_dir, "plots")

    def prog(i, n):
        print(f"\r  {i}/{n} files…", end="", flush=True)

    if _wants_pipeline(a):
        res = _run_pipeline_cli(a, params, plot_dir)
    else:
        res = analyze_files(a.inputs, params,
                            plot_dir=plot_dir if params["plots"] else "",
                            progress=prog)
    if res["n_files"] == 0:
        print(f"\nWARNING: no '*_video_data.txt' files "
              f"matched {a.inputs} (searched recursively). Point at your data "
              f"folder, e.g. data/<Project>. Nothing written.")
        return
    print(f"\nAnalyzed {len(res['summary'])} file(s) on {res['n_jobs']} "
          f"worker(s); {len(res['excluded'])} excluded; "
          f"{len(res['plots'])} plot(s).")

    # Say why there are no figures, rather than reporting "0 plot(s)" and
    # leaving the user to guess between "off", "none selected" and "matplotlib
    # is not installed on this machine".
    state, detail = res.get("plot_state", ""), res.get("plot_detail", "")
    if state and state != "ok":
        print(f"  NOTE: no PNG figures, {detail}")
        if state == "unavailable":
            print("        the HTML report below needs no plotting library.")

    write_workbook(res, a.out)
    print(f"Wrote {a.out}  (+ plots in {plot_dir})" if res["plots"]
          else f"Wrote {a.out}")

    if not a.no_html:
        from tools.offline_analysis.engine.report_html import write_report
        html = a.html or os.path.splitext(a.out)[0] + "_report.html"
        try:
            write_report(res, html, params)
            print(f"Wrote {html}")
        except Exception as e:
            print(f"  WARNING: HTML report failed: {e}")


if __name__ == "__main__":
    main()
