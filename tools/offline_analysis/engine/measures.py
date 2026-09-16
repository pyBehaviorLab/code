"""Every number the workbench reports, defined once.

A statistic written out where it is used drifts from itself. Distance spelled
``float(np.sum(step_cm[mask]))`` in the session row, again in the per-zone
breakdown, again in the regions, and again in the time bins; mean speed as
``sum(step)/duration`` in one and ``sum(step)/sum(dt)`` in another. Four
copies of a formula are four definitions, and the only way to find out they
had drifted was to compare two numbers in a spreadsheet and wonder.

So: one function per statistic, and everything that emits a column calls it.

Everything here is a **pure function over plain arrays**, no ``Session``, no
params dict, no column names, no rounding. That is what makes them checkable:
:mod:`tools.offline_analysis.engine.validate` runs each one against a case whose answer can
be worked out on paper, and prints expected against actual.

UNITS. Centimetres and seconds, because that is what the tracker produces.
The conversion to metres happens at the edge, in the emitter, where the column
name says ``_m``. Nothing here knows about metres.

THE FRAME CONVENTION. ``dt[i]`` is the time from frame ``i-1`` to frame ``i``,
and ``step_cm[i]`` the distance covered in it, both are credited to the frame
they ARRIVE at, and ``dt[0]`` is 0. So a mask over frames selects whole
intervals, and summing any quantity over a partition of the frames gives
exactly the total. That is why the time bins add up.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Mask = Optional[np.ndarray]


def _sel(a: np.ndarray, mask: Mask) -> np.ndarray:
    """``a`` restricted to ``mask``, or all of it when no mask is given.

    A length mismatch RAISES, as numpy's own boolean indexing does. Quietly
    truncating to the shorter of the two would answer a question nobody asked,
    the frames of one thing measured against the mask of another, and it
    would answer it plausibly, which is worse than not answering.
    """
    a = np.asarray(a)
    if mask is None:
        return a
    m = np.asarray(mask, dtype=bool)
    if len(m) != len(a):
        raise ValueError(f"mask of {len(m)} frames against {len(a)} values")
    return a[m]


# ── time ─────────────────────────────────────────────────────────────────────

def duration_s(dt: np.ndarray, mask: Mask = None) -> float:
    """Seconds of recording covered by the selected frames.

    With no mask this is the whole session, and equals ``t[-1] - t[0]``
    because ``dt[0]`` is 0.
    """
    return float(np.sum(_sel(dt, mask)))


#: A bin counts as partial when the data covers less than this share of the
#: window. It is not "shorter than the window by any amount": a 299.8 s
#: recording binned at 60 s has a final window covering 59.8 s, one frame
#: short of full, and flagging that marked the last bin of essentially every
#: recording, so the flag that should mean "half of this window is missing"
#: meant nothing at all.
BIN_PARTIAL_COVERAGE = 0.95


def is_partial(covered_s: float, width_s: float,
               threshold: float = BIN_PARTIAL_COVERAGE) -> bool:
    """Whether a window is too poorly covered to compare with its siblings."""
    if width_s <= 0:
        return False
    return float(covered_s) < threshold * float(width_s)


# ── locomotion ───────────────────────────────────────────────────────────────

def path_length(step: np.ndarray, mask: Mask = None) -> float:
    """Path length over the selected frames, in whatever unit ``step`` carries.

    Per-frame steps are summed as they are: gaps contribute 0 (the track
    builder puts 0 there rather than teleporting across them), so this never
    invents travel the tracker did not see.
    """
    return float(np.sum(_sel(step, mask)))


def mean_rate(step: np.ndarray, dt: np.ndarray, mask: Mask = None) -> float:
    """Path length over TIME COVERED, not over the number of frames.

    The session row divided by ``duration_s`` and the time bins by the summed
    ``dt``. For a whole session those are the same number; for anything less
    than a whole session they are not, and only one of them is the mean speed
    during the window.
    """
    t = duration_s(dt, mask)
    return path_length(step, mask) / t if t > 0 else 0.0


def peak_rate(values: np.ndarray, mask: Mask = None) -> float:
    """The fastest single frame. 0 for an empty selection."""
    v = _sel(values, mask)
    return float(np.max(v)) if len(v) else 0.0


#: The track works in centimetres, and its callers read better saying so.
#: The neutral names above are the one implementation, so a unit never gets
#: its own copy of the sum.
distance_cm = path_length
mean_speed_cm_s = mean_rate
peak_speed_cm_s = peak_rate


# ── zones ────────────────────────────────────────────────────────────────────

def time_in_s(in_zone: np.ndarray, dt: np.ndarray) -> float:
    """Time with the point inside the zone. Every frame counts, including the
    brief visits :func:`dwell_entries` refuses to call an entry, being
    somewhere and having ENTERED it are different questions."""
    return duration_s(dt, in_zone)


def dwell_entries(in_zone: np.ndarray, dt: np.ndarray, min_dwell: float):
    """EthoVision/ANY-maze minimum-visit-duration debounce.

    Returns ``(entries, exits, latency_s, total_time_s, visit_durations)``. A
    visit counts as an entry once it has lasted ``min_dwell``; the latency is
    the START of the first such visit, not the moment it qualified.
    """
    entries = exits = 0
    inz = False; dwell = 0.0; valid = False
    latency = math.nan; total = 0.0
    t_cum = np.concatenate(([0.0], np.cumsum(dt)))
    visits: List[float] = []
    visit_start = 0.0; visit_len = 0.0
    for i in range(len(in_zone)):
        if in_zone[i]:
            total += dt[i]
            if not inz:
                inz = True; dwell = dt[i]; valid = False
                visit_start = t_cum[i]; visit_len = dt[i]
            else:
                dwell += dt[i]; visit_len += dt[i]
                if dwell >= min_dwell - 1e-9 and not valid:
                    entries += 1; valid = True
                    if math.isnan(latency):
                        latency = visit_start
        else:
            if inz and valid:
                exits += 1; visits.append(visit_len)
            inz = False; dwell = 0.0; valid = False
    if inz and valid:
        visits.append(visit_len)
    return entries, exits, latency, total, visits


def transitions_total(location: Sequence[str],
                      include: Optional[Sequence[str]] = None) -> int:
    """How many times the animal moved from one zone to a DIFFERENT one.

    Frames with no zone are skipped rather than treated as a zone, so passing
    through unzoned floor between two arms is one transition, not two.
    ``include`` restricts it to the zones the user is reporting on.
    """
    prev = None
    seen = 0
    for z in location:
        if not z or z in ("na", "none", "None", "-"):
            continue
        if include and z not in include:
            continue
        if z != prev:
            seen += 1
        prev = z
    return max(0, seen - 1)


# ── bouts ────────────────────────────────────────────────────────────────────

def freeze_bouts(speed_cm_s: np.ndarray, dt: np.ndarray, thr: float,
                 min_s: float, return_spans: bool = False):
    """Immobility/freezing bouts: speed below ``thr`` continuously for ``min_s``.

    Returns ``(time_s, n_bouts, latency_to_first_s)``; the third value lets
    callers that want it (freezing) report onset latency, and immobility can
    ignore it.

    With ``return_spans=True`` a 4th element carries ``[[start_s, end_s], ...]``
    for the counted bouts. It exists so a plot can SHADE the exact bouts the
    workbook counted rather than re-detecting them with its own loop, two
    detectors drift, and a figure sitting next to the table it disagrees with
    is the most misleading place for that to happen.
    """
    total = 0.0; bouts = 0; run = 0.0; counted = False
    t = 0.0; run_start = 0.0; latency = math.nan
    spans: List[List[float]] = []
    for i in range(len(speed_cm_s)):
        if speed_cm_s[i] < thr:
            if run == 0.0:
                run_start = t
            run += dt[i]
            if run >= min_s - 1e-9:
                if not counted:
                    bouts += 1; counted = True; total += run
                    spans.append([run_start, t + dt[i]])
                    if math.isnan(latency):
                        latency = run_start
                else:
                    total += dt[i]
                    spans[-1][1] = t + dt[i]        # extend the open bout
        else:
            run = 0.0; counted = False
        t += dt[i]
    if return_spans:
        return total, bouts, latency, [[round(a, 3), round(b, 3)]
                                       for a, b in spans]
    return total, bouts, latency


# ── time windows ─────────────────────────────────────────────────────────────

def bin_spans(duration_s_: float, *, edges: Optional[Sequence[float]] = None,
              count: int = 0, width: float = 0.0) -> List[Tuple[float, float]]:
    """The ``[lo, hi)`` windows asked for, in seconds.

    Explicit edges win, then a count of equal bins, then a fixed width. An
    empty list means "not binned", the one answer every caller checks.

    Every boundary this DERIVES is a whole second. A recording is 299.8 s
    long, not 300, so "split into 4" produced 74.95 s bins and columns named
    ``0-74.95s_Distance_m``: unreadable alone, and different in every
    recording, so a cohort sheet could not line them up. Boundaries the user
    typed are left exactly as typed.
    """
    if duration_s_ <= 0:
        return []
    end = float(math.ceil(duration_s_))
    if edges:
        e = sorted({float(x) for x in edges if float(x) >= 0})
        # An open last edge is the common case: "0, 120" over a 600 s session
        # means 0-120 and 120-600, not "everything after 120 s is discarded".
        if e and e[0] > 0:
            e.insert(0, 0.0)
        if e and e[-1] < end:
            e.append(end)
        return [(e[i], e[i + 1]) for i in range(len(e) - 1) if e[i + 1] > e[i]]
    count = int(count or 0)
    if count > 0:
        cuts = [round(i * end / count) for i in range(count)] + [end]
        return [(float(cuts[i]), float(cuts[i + 1])) for i in range(count)
                if cuts[i + 1] > cuts[i]]
    width = float(width or 0)
    if width <= 0:
        return []
    nb = int(math.ceil(duration_s_ / width))
    return [(b * width, (b + 1) * width) for b in range(nb)]


def bin_mask(t_rel: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The frames inside one window, half-open, so the windows partition the
    session and every frame is counted exactly once."""
    t = np.asarray(t_rel)
    return (t >= lo) & (t < hi)
