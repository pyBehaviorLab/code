"""Cleaning a keypoint trace before anything measures it.

Modelled on refineDLC (Biology Methods and Protocols, 2025), which sets out
the stages a DeepLabCut trace needs before it is analysed: drop the detections
the model was not confident about, drop the ones that moved further than an
animal can, fill the short holes that leaves, and leave the long ones alone.

Three places this goes further than the published pipeline, each for a reason
that paper itself gives:

* **Per-keypoint thresholds, not one number.** refineDLC offers a global 0.5
  or the 5th percentile *per video*, and observes that "distal limb
  markers… were the least consistent among all body parts". A snout and a
  centre do not share a confidence distribution, so they do not share a
  threshold: each keypoint is gated against its own.

* **A jump limit in the animal's units, not pixels.** 30 px means one thing
  on a 360-px arena and another on a 1280-px one, and nothing at all at a
  different frame rate. The limit here is a robust multiple of the recording's
  own frame-to-frame displacement, measured with the interquartile range so
  a handful of wild outliers cannot raise the bar that is meant to catch
  them.

* **A gap limit in seconds, not frames.** "5 frames" is 0.17 s at 30 fps and
  0.25 s at 20; the thing that matters is how long the animal was unobserved,
  so that is what is bounded.

Everything reports what it did. A stage that removed a third of a keypoint's
detections and said nothing would be indistinguishable from a model that never
found it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from tools.offline_analysis.engine import pose_repair as _repair

logger = logging.getLogger(__name__)

#: Confidence percentile per keypoint below which detections are dropped.
#: refineDLC's per-video option, applied per keypoint.
DEFAULT_PERCENTILE = 5.0

#: Never gate above this, however poor a keypoint's distribution is. A part
#: the model is uniformly unsure about would otherwise have its 5th percentile
#: land at 0.9 and lose almost everything.
CONFIDENCE_CEILING = 0.6

#: An absolute floor under the percentile. A percentile rule cannot remove
#: more than that percentage of frames, so a recording where a fifth of one
#: keypoint's detections are near-zero keeps most of them: the 5th percentile
#: lands INSIDE the bad group and gates at 0.01. refineDLC offers a global
#: 0.5 as the alternative to its percentile; this uses both, with the floor
#: set low enough to remove only what is obviously not a detection.
DEFAULT_FLOOR = 0.1

#: How many robust deviations above the median step counts as impossible.
#: 1.5 is refineDLC's IQR multiplier; this is the MAD equivalent, which is the
#: same idea with a breakdown point that survives a run of bad frames.
DEFAULT_JUMP_MADS = 6.0

#: Longest hole worth filling, in seconds. Beyond this the animal was
#: genuinely unobserved and a straight line between the ends is invention.
DEFAULT_MAX_GAP_S = 0.25


@dataclass
class RefineReport:
    """What cleaning did to one keypoint, so it can be said out loud."""
    part: str = ""
    frames: int = 0
    dropped_confidence: int = 0
    dropped_jump: int = 0
    #: Confident detections dropped for sitting off the fitted body, the error
    #: neither of the two gates above can see.
    mistracked: int = 0
    filled: int = 0
    left_open: int = 0
    reconstructed: int = 0
    threshold: float = 0.0
    jump_limit_px: float = 0.0

    @property
    def usable(self) -> int:
        return (self.frames - self.dropped_confidence - self.dropped_jump
                - self.mistracked)

    def line(self) -> str:
        return (f"{self.part}: kept {self.usable}/{self.frames}"
                f" (conf<{self.threshold:.2f} dropped {self.dropped_confidence},"
                f" jump>{self.jump_limit_px:.0f}px dropped {self.dropped_jump},"
                f" filled {self.filled}, left open {self.left_open},"
                f" placed from the other keypoints {self.reconstructed})")


#: Fewest keypoints that have to be visible before a missing one is placed
#: from them. Three is the minimum a similarity transform is determined by;
#: below it the fit is exact by construction and tells you nothing about
#: whether it is right.
DEFAULT_MIN_ANCHORS = 3

#: How far the visible keypoints may sit from the fitted body shape before the
#: reconstruction is refused, as a multiple of the animal's own body length.
#: A frame where the anchors do not fit the shape is one where the model was
#: confused about the whole animal, not just the missing part.
DEFAULT_MAX_RESIDUAL = 0.25


@dataclass
class Settings:
    """The knobs, with the defaults a run uses when nobody sets them."""
    enabled: bool = True
    percentile: float = DEFAULT_PERCENTILE
    floor: float = DEFAULT_FLOOR       # absolute minimum confidence
    ceiling: float = CONFIDENCE_CEILING
    jump_mads: float = DEFAULT_JUMP_MADS
    max_gap_s: float = DEFAULT_MAX_GAP_S
    interpolate: bool = True
    reconstruct: bool = True
    min_anchors: int = DEFAULT_MIN_ANCHORS
    max_residual: float = DEFAULT_MAX_RESIDUAL
    smooth: bool = True
    smooth_window: int = 7
    smooth_order: int = 2
    #: Drop a confident detection sitting further than this from the fitted
    #: body, in body lengths. The one error confidence and the jump limit both
    #: pass: a keypoint that is certain and in the wrong place.
    mistrack_residual: float = _repair.DEFAULT_MISTRACK_RESIDUAL
    #: Exchange a left/right pair that fits the body better the other way.
    fix_swaps: bool = True
    swap_margin: float = _repair.DEFAULT_SWAP_MARGIN
    #: ``anatomical`` decides side from the body axis per frame and resolves
    #: the sequence along time; ``shape`` is the older per-frame comparison
    #: against a canonical shape learned from the same trace. Anatomical is the
    #: default: see ``offline_analysis.DEFAULTS`` for the measurement.
    swap_method: str = "auto"
    swap_transition_cost: float = 0.15
    robust_rounds: int = _repair.DEFAULT_ROBUST_ROUNDS
    #: ``rts`` = forward-backward Kalman smoother, ``savgol`` = the centred
    #: polynomial window, ``none``. See ``pose_repair.rts_smooth`` for why the
    #: default changed.
    smoother: str = "rts"
    measure_px: float = _repair.DEFAULT_MEASURE_PX
    process_scale: float = _repair.DEFAULT_PROCESS_SCALE

    @classmethod
    def from_params(cls, params: dict) -> Settings:
        p = params or {}
        return cls(
            mistrack_residual=float(p.get(
                "refine_mistrack_residual", _repair.DEFAULT_MISTRACK_RESIDUAL)),
            fix_swaps=bool(p.get("refine_fix_swaps", True)),
            swap_margin=float(p.get("refine_swap_margin",
                                    _repair.DEFAULT_SWAP_MARGIN)),
            swap_method=str(p.get("refine_swap_method", "auto")
                            or "auto").lower(),
            swap_transition_cost=float(
                p.get("refine_swap_transition_cost", 0.15)),
            robust_rounds=int(p.get("refine_robust_rounds",
                                    _repair.DEFAULT_ROBUST_ROUNDS)),
            smoother=str(p.get("refine_smoother", "rts") or "rts").lower(),
            measure_px=float(p.get("refine_measure_px",
                                   _repair.DEFAULT_MEASURE_PX)),
            process_scale=float(p.get("refine_process_scale",
                                      _repair.DEFAULT_PROCESS_SCALE)),
            enabled=bool(p.get("refine", True)),
            percentile=float(p.get("refine_percentile", DEFAULT_PERCENTILE)),
            floor=float(p.get("refine_min_confidence", DEFAULT_FLOOR)),
            ceiling=float(p.get("refine_confidence_ceiling", CONFIDENCE_CEILING)),
            jump_mads=float(p.get("refine_jump_mads", DEFAULT_JUMP_MADS)),
            max_gap_s=float(p.get("refine_max_gap_s", DEFAULT_MAX_GAP_S)),
            interpolate=bool(p.get("refine_interpolate", True)),
            reconstruct=bool(p.get("refine_reconstruct", True)),
            min_anchors=int(p.get("refine_min_anchors", DEFAULT_MIN_ANCHORS)),
            max_residual=float(p.get("refine_max_residual",
                                     DEFAULT_MAX_RESIDUAL)),
            smooth=bool(p.get("refine_smooth", True)),
            smooth_window=int(p.get("refine_smooth_window", 7)),
            smooth_order=int(p.get("refine_smooth_order", 2)))


# ── the stages ───────────────────────────────────────────────────────────

def confidence_threshold(conf: np.ndarray, s: Settings) -> float:
    """The confidence this keypoint has to clear, from its own distribution."""
    good = np.asarray(conf, float)
    good = good[np.isfinite(good)]
    if good.size == 0:
        return float(s.floor)
    cut = float(np.percentile(good, s.percentile))
    return float(min(max(cut, s.floor), s.ceiling))


def jump_limit(x: np.ndarray, y: np.ndarray, s: Settings) -> float:
    """The furthest this keypoint plausibly moves between frames, in pixels.

    Median + k·spread of the observed steps, where the spread is robust by
    construction: the excursions this is meant to reject are exactly the
    values that would inflate a mean-and-standard-deviation limit until it
    accepted them.
    """
    step = np.hypot(np.diff(np.asarray(x, float)), np.diff(np.asarray(y, float)))
    step = step[np.isfinite(step)]
    if step.size < 8:
        return float("inf")                 # too little to judge; reject none
    median = float(np.median(step))
    # Interquartile range first, refineDLC's own statistic, and the one that
    # ignores the tails entirely. The MAD is the fallback, scaled by 1.4826 so
    # it is comparable to a standard deviation, which is what the multiplier
    # is quoted against.
    q25, q75 = (float(v) for v in np.percentile(step, [25, 75]))
    spread = q75 - q25
    if spread <= 0:
        spread = 1.4826 * float(np.median(np.abs(step - median)))
    if spread <= 0:
        # A trace so regular that both robust measures are exactly zero, a
        # synthetic or heavily filtered one. NOT the mean absolute deviation,
        # which was the first thing tried here and is the one statistic the
        # outliers being rejected can inflate: three 900-px spikes in a
        # 200-frame trace raised the limit from 10 px to 122 and let all three
        # through.
        spread = max(0.25 * median, 1.0)
    return median + s.jump_mads * spread


def fill_gaps(x: np.ndarray, t_s: np.ndarray, max_gap_s: float
              ) -> tuple[np.ndarray, int, int]:
    """Interpolate holes shorter than ``max_gap_s``; leave the rest as NaN.

    Returns ``(filled series, points filled, holes left open)``. A hole longer
    than the limit is data the recording does not have, and inventing a
    straight line across it would put the animal somewhere it may never have
    been, the difference between a gap that contributes nothing to distance
    and one that contributes a confident lie.
    """
    out = np.asarray(x, float).copy()
    t = np.asarray(t_s, float)
    ok = np.isfinite(out)
    if ok.sum() < 2 or ok.all():
        return out, 0, 0
    idx = np.nonzero(ok)[0]
    filled = left = 0
    for a, b in zip(idx, idx[1:]):
        if b - a <= 1:
            continue
        span = float(t[b] - t[a]) if t.size > b else float("inf")
        if span <= max_gap_s:
            out[a + 1:b] = np.interp(t[a + 1:b], [t[a], t[b]], [out[a], out[b]])
            filled += b - a - 1
        else:
            left += 1
    return out, filled, left


# ── putting back what the gates removed ────────────────────────
#
# Gating is subtractive, and on this rig it subtracts a lot: against a model
# whose median confidence is around 0.75, a 0.55 cut alone removes 14-30% of
# every keypoint. Interpolation puts back the short holes but says nothing
# about a keypoint that was missing for a second while the other five were
# visible the whole time, and that is the common case, because the model
# loses ONE part at a time.
#
# That work, and the mistrack and left/right-swap detection that share its
# shape model, live in `pose_repair`. They are one body of reasoning about
# the animal rather than about one keypoint's timeline, and keeping them
# here meant the shape was learned only from frames where EVERY part was
# visible, which is precisely what gating makes rare. `repair_across_parts`
# below is the seam.


def refine_keypoint(x, y, conf, t_s, s: Settings, part: str = ""):
    """Clean one keypoint. Returns ``(x, y, conf, report)``.

    The order is refineDLC's, and it matters: confidence first (a low-score
    detection is often also a wild one, and dropping it first keeps it out of
    the statistics that set the jump limit), then jumps, then filling.
    """
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    conf = np.asarray(conf, float)
    report = RefineReport(part=part, frames=int(x.size))
    if x.size == 0:
        return x, y, conf, report

    thr = confidence_threshold(conf, s)
    report.threshold = thr
    low = np.isfinite(conf) & (conf < thr)
    report.dropped_confidence = int(low.sum())
    x[low] = np.nan
    y[low] = np.nan

    limit = jump_limit(x, y, s)
    report.jump_limit_px = limit if np.isfinite(limit) else 0.0
    if np.isfinite(limit):
        # Judged against the ORIGINAL neighbours, never against an already
        # corrected value: chaining lets one fast dart cascade and freeze the
        # whole trace to a single point.
        ox, oy = x.copy(), y.copy()
        step = np.full(x.size, np.nan)
        step[1:] = np.hypot(np.diff(ox), np.diff(oy))
        bad = np.isfinite(step) & (step > limit)
        report.dropped_jump = int(bad.sum())
        x[bad] = np.nan
        y[bad] = np.nan

    if s.interpolate:
        x, fx, gx = fill_gaps(x, t_s, s.max_gap_s)
        y, _fy, _gy = fill_gaps(y, t_s, s.max_gap_s)
        report.filled = fx
        report.left_open = gx
    return x, y, conf, report


def savgol_coefficients(window: int, order: int) -> np.ndarray:
    """The convolution kernel of a Savitzky-Golay smoother.

    Fitting a low-order polynomial to each window and taking its value at the
    centre is a linear operation, so the whole filter collapses to one set of
    weights that can be convolved. Derived here rather than imported so the
    analyser keeps working on a machine without SciPy, the Jetson build is
    the one that matters, and this is nine lines.
    """
    half = int(window) // 2
    t = np.arange(-half, half + 1, dtype=float)
    # Rows of the design matrix are 1, t, t^2 … ; the centre value of the fit
    # is the first row of its pseudo-inverse.
    A = np.vander(t, int(order) + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def smooth_zero_lag(x: np.ndarray, window: int = 7, order: int = 2
                    ) -> np.ndarray:
    """Take the jitter out without shifting anything in time.

    The live pipeline filters with One-Euro, which is the right choice there
    and the wrong one here: it is *causal*: it only ever knows the past, and it
    pays for smoothing with lag. Offline the whole recording is already on
    disk, so the window is **centred**, the frame after this one is as
    available as the frame before it, and there is no lag to pay.

    Savitzky-Golay rather than a moving average, and for the reason that
    matters to this data: a moving average is a polynomial of order zero, so it
    treats a genuine dart as an outlier and flattens it. Fitting a quadratic
    keeps the peak and removes the tremor around it. DeepLabCut's own
    ``filterpredictions`` offers the same filter for the same reason.

    Holes are preserved. Values are interpolated internally so the filter has
    something continuous to work on, but anything that was missing goes back to
    NaN afterwards: smoothing must not quietly become gap-filling.
    """
    a = np.asarray(x, float)
    k = (int(window) | 1)                     # odd, so the window is centred
    if a.size < k or k < 3 or order >= k:
        return a.copy()
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return a.copy()
    idx = np.arange(a.size)
    filled = np.interp(idx, idx[ok], a[ok])
    # Reflected at the ends, so the first and last frames are not dragged
    # toward whatever the padding would otherwise be.
    pad = k // 2
    ext = np.concatenate([filled[pad:0:-1], filled, filled[-2:-pad - 2:-1]])
    out = np.convolve(ext, savgol_coefficients(k, order)[::-1], mode="valid")
    return np.where(ok, out[:a.size], np.nan)


#: Confidence attached to a position the body shape placed rather than the
#: model detected. Below any real detection the gates let through, so the
#: smoother leans on it only where there was nothing better, and anything
#: downstream that thresholds on confidence still counts it as weak.
RECONSTRUCTED_CONFIDENCE = 0.15


def _smoother_confidence(channels: dict) -> np.ndarray:
    """Per-frame confidence for the smoother, filled positions included.

    A reconstructed keypoint arrives with the confidence of the detection that
    was DROPPED, usually zero, sometimes the high score of a mistrack, and
    neither describes it. It is a position the rest of the animal implies, so
    it is given one weak confidence of its own.
    """
    conf = np.array(channels.get("conf"), float, copy=True)
    filled = channels.get("filled")
    if filled is None or conf.size == 0:
        return conf
    filled = np.asarray(filled, bool)
    if filled.shape != conf.shape:
        return conf
    return np.where(filled, RECONSTRUCTED_CONFIDENCE, conf)


def repair_across_parts(kp: dict, s: Settings):
    """Un-swap, drop mistracks and reconstruct. Returns ``(kp, report|None)``.

    The bridge between this module's ``{part: {"x", "y", …}}`` and
    ``pose_repair``'s ``(frames, parts)`` complex array. ``kp`` gains a
    ``"filled"`` boolean channel per part marking every position that was
    computed rather than detected, so nothing downstream can mistake one for
    the other.
    """
    names = list(kp)
    if not names or not (s.reconstruct or s.fix_swaps
                         or s.mistrack_residual > 0):
        return kp, None
    lengths = {np.asarray(kp[n].get("x")).size for n in names}
    if len(lengths) != 1 or 0 in lengths:
        logger.warning("keypoints have differing frame counts %s, the "
                       "across-keypoint repair needs one timeline", lengths)
        return kp, None

    z = np.stack([np.asarray(kp[n]["x"], float)
                  + 1j * np.asarray(kp[n]["y"], float) for n in names], axis=1)
    z, filled, report = _repair.repair_frames(
        z, names,
        min_anchors=s.min_anchors,
        mistrack_residual=(s.mistrack_residual if s.mistrack_residual > 0
                           else float("inf")),
        swap_margin=s.swap_margin if s.fix_swaps else 0.0,
        swap_method=s.swap_method if s.fix_swaps else "shape",
        swap_transition_cost=s.swap_transition_cost,
        rounds=s.robust_rounds,
        reconstruct=s.reconstruct,
        smooth_window=s.smooth_window)
    out = {}
    for i, name in enumerate(names):
        out[name] = {**kp[name], "x": z[:, i].real.copy(),
                     "y": z[:, i].imag.copy(), "filled": filled[:, i]}
    return out, report


def refine_all(kp: dict, t_s, params: dict) -> tuple[dict, list]:
    """Clean every keypoint of a session. Returns ``(kp, reports)``.

    ``kp`` is the ``{part: {"x", "y", "conf"}}`` the analysis carries. On with
    the defaults, because a trace nobody cleaned is not a neutral starting
    point; it is one with the model's worst frames still in it, and asking
    every user to discover that for themselves is how the numbers came to
    depend on who ran them.
    """
    s = Settings.from_params(params)
    if not s.enabled or not kp:
        return kp, []
    out, reports = {}, []
    for part, channels in kp.items():
        x, y, conf, report = refine_keypoint(
            channels.get("x"), channels.get("y"), channels.get("conf"),
            t_s, s, part=part)
        out[part] = {**channels, "x": x, "y": y, "conf": conf}
        reports.append(report)

    # Then across keypoints, not along one. Everything above works down a
    # single part's timeline and cannot tell "the model lost the snout for a
    # second" from "the animal left", and cannot see a confident detection in
    # the wrong place at all. The other five parts can do both.
    out, repair = repair_across_parts(out, s)
    if repair is not None:
        for r in reports:
            r.reconstructed = repair.reconstructed.get(r.part, 0)
            r.mistracked = repair.mistracked.get(r.part, 0)
        logger.info("across keypoints: %s", repair.summary())

    # Last, so the smoother runs over a trace that is already as complete as it
    # is going to get. Smoothing first would drag every estimate toward the
    # holes it was meant to leave alone.
    if s.smooth and s.smoother != "none":
        t = np.asarray(t_s, float)
        for part in out:
            ch = out[part]
            if s.smoother == "savgol":
                sx = smooth_zero_lag(ch["x"], s.smooth_window, s.smooth_order)
                sy = smooth_zero_lag(ch["y"], s.smooth_window, s.smooth_order)
            else:
                # A reconstructed position is not a measurement, but it is not
                # a hole either: the smoother is told about it at the
                # confidence the fit earned rather than being handed a NaN it
                # would have to bridge blindly.
                sx, sy = _repair.rts_smooth(
                    ch["x"], ch["y"], _smoother_confidence(ch),
                    t, measure_px=s.measure_px,
                    process_scale=s.process_scale)
            out[part] = {**ch, "x": sx, "y": sy}

    if reports:
        # A summary at INFO and the per-keypoint detail at DEBUG: the detail
        # is six clauses long and `build_track` runs several times per
        # recording, so at INFO it buries everything else in a cohort run.
        dropped = sum(r.dropped_confidence + r.dropped_jump for r in reports)
        total = sum(r.frames for r in reports)
        logger.info(
            "cleaned %d keypoints: dropped %d of %d detections (%.1f%%), "
            "filled %d short gaps, left %d longer than %.2f s open",
            len(reports), dropped, total,
            100.0 * dropped / total if total else 0.0,
            sum(r.filled for r in reports),
            sum(r.left_open for r in reports), s.max_gap_s)
        for r in reports:
            logger.debug("  %s", r.line())
    return out, reports


__all__ = ["CONFIDENCE_CEILING", "DEFAULT_JUMP_MADS", "DEFAULT_MAX_GAP_S",
           "DEFAULT_PERCENTILE", "RECONSTRUCTED_CONFIDENCE", "RefineReport",
           "Settings", "confidence_threshold", "fill_gaps", "jump_limit",
           "refine_all", "refine_keypoint", "repair_across_parts",
           "savgol_coefficients", "smooth_zero_lag"]
