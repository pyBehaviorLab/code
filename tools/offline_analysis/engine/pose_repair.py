"""Repairing a keypoint trace with what only an offline pass knows.

A live rig sees each frame once, in order, and has to answer before the next
one arrives. A retrack has the whole recording on disk. Two things follow, and
this module is both of them.

**Across keypoints, the animal has a shape.** When the model loses one part,
the other five still say where it was: a mouse is close to rigid over the parts
a pose model tracks, so the missing one is determined by the rest up to the
noise in them. The same fit answers a harder question the previous code never
asked. It computed how far the visible keypoints sat from the fitted shape and
used that only to *refuse* to fill around a bad frame, so a keypoint that was
confidently in the wrong place stayed there, in the output, at high confidence.
Confidence gating cannot catch that (the detection is confident), and the jump
limit cannot either (a snout mislabelled onto a nearby ear has not moved far).
The shape can: a part sitting a body-length off the fitted body is wrong
whatever its score says.

**Along time, the frame after this one is as available as the frame before.**
So smoothing runs forwards AND backwards. DeepLabCut's ``filterpredictions``
reaches for ARIMA here, whose ``.filter()`` is a state-space Kalman smoother;
this is the same idea reduced to what the data supports, a constant-velocity
model with a fixed-interval RTS backward pass, plus one thing DeepLabCut does
not do: the measurement noise is taken from each detection's own confidence, so
a weak keypoint is pulled toward the trajectory and a strong one holds its
ground. That is the whole reason confidence is worth carrying this far.

Everything reports what it did. A stage that silently moved a third of a
keypoint's positions would be indistinguishable from a model that found them
there.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

#: Anchors needed before a frame's pose can be fitted at all. Three is the
#: minimum a 2-D similarity transform is over-determined by; at two the fit is
#: exact by construction and its residual says nothing about whether it is right.
DEFAULT_MIN_ANCHORS = 3

#: How far a visible keypoint may sit from the fitted body before it is called a
#: mistrack, as a fraction of the animal's own body length. Body-length units
#: rather than pixels so one number covers any arena, any camera and any zoom.
#: 0.35 is wide enough to pass ordinary localisation noise and normal postural
#: non-rigidity, and narrow enough to catch a part landing on a different part.
DEFAULT_MISTRACK_RESIDUAL = 0.35

#: How much better the swapped assignment has to fit before a symmetric pair is
#: exchanged. Well below 1 so a pair that fits about equally either way, which
#: is what an animal seen head-on looks like, is left alone.
DEFAULT_SWAP_MARGIN = 0.6

#: Rounds of fit-then-reject. One is not enough: the mistracked point being
#: looked for is also corrupting the fit that has to find it, so the first
#: round's residuals are biased toward it. Each round removes at most one part
#: per frame, so this is also the most that can ever be dropped from one frame.
DEFAULT_ROBUST_ROUNDS = 3

#: How far clear of the rest of its own frame the worst residual has to stand
#: before it is called a mistrack rather than a badly-fitted frame. Twice the
#: median of the others: a genuine mistrack sits three to five times out on
#: this rig's data, while a frame the model simply fitted poorly has all its
#: parts wrong together.
MISTRACK_SEPARATION = 2.0

#: How far a frame's fitted body pose may sit from its neighbours in time
#: before that frame is no longer trusted to place keypoints, in body-scale
#: units. Catches the case per-frame fitting cannot: a frame corrupted in more
#: of its keypoints than it has good ones, which fits itself perfectly well and
#: is simply in the wrong place.
DEFAULT_POSE_JUMP = 0.25

#: Process noise for the smoother, as a multiple of the trace's own robust
#: acceleration. Estimated from the data rather than fixed, for the same reason
#: the jump limit is: a number in pixels means different things at different
#: frame rates and different zooms.
DEFAULT_PROCESS_SCALE = 1.0

#: Measurement noise at confidence 1.0, in pixels. The published localisation
#: error of a well-trained model on its own test set is 2-5 px; this is the
#: floor the smoother will not pretend to beat.
DEFAULT_MEASURE_PX = 2.0

#: Confidence below which a detection is treated as absent by the smoother
#: rather than merely uncertain.
DEFAULT_SMOOTH_FLOOR = 0.05

#: Name patterns that make two parts a mirror pair. Matched case-insensitively
#: against the whole name, so ``Left_Ear``/``Right_Ear`` and ``ear_l``/``ear_r``
#: both pair and ``Left_Ear``/``Right_Paw`` does not.
_SIDES = (
    (re.compile(r"^left[_\- ]?(.*)$", re.I), re.compile(r"^right[_\- ]?(.*)$", re.I)),
    (re.compile(r"^(.*?)[_\- ]?left$", re.I), re.compile(r"^(.*?)[_\- ]?right$", re.I)),
    # Single-letter side, separator required: ``l_ear``/``r_ear`` pair, while
    # ``lever``/``reward`` cannot, because without the separator every name
    # beginning with l would pair with every name beginning with r.
    (re.compile(r"^l[_\- ](.*)$", re.I), re.compile(r"^r[_\- ](.*)$", re.I)),
    (re.compile(r"^(.*?)[_\- ]l$", re.I), re.compile(r"^(.*?)[_\- ]r$", re.I)),
)


@dataclass
class RepairReport:
    """What the across-keypoints pass did, per part and overall."""

    swapped: dict = field(default_factory=dict)       # part -> frames exchanged
    mistracked: dict = field(default_factory=dict)    # part -> frames dropped
    reconstructed: dict = field(default_factory=dict)  # part -> frames placed
    unfittable: int = 0          # frames with too few anchors to fit at all
    pairs: tuple = ()            # the mirror pairs that were checked
    body_length: float = 0.0     # in the shape's own units
    shape_frames: int = 0        # frames the canonical shape was learned from
    #: Which mirror-pair test actually ran. Worth recording: under ``auto`` it
    #: is chosen from the trace, so a result cannot be reproduced without it.
    method: str = ""

    def summary(self) -> str:
        bits = []
        if self.swapped:
            bits.append(f"un-swapped ({self.method or 'shape'}) " + ", ".join(
                f"{k} {v}" for k, v in sorted(self.swapped.items())))
        if self.mistracked:
            bits.append("dropped as mistracked " + ", ".join(
                f"{k} {v}" for k, v in sorted(self.mistracked.items())))
        if self.reconstructed:
            bits.append("placed from the body " + ", ".join(
                f"{k} {v}" for k, v in sorted(self.reconstructed.items())))
        if not bits:
            return "nothing to repair"
        return "; ".join(bits)


# ── the animal's shape ───────────────────────────────────────────────────

def as_complex(pts: np.ndarray) -> np.ndarray:
    """``(..., 2)`` real → ``(...)`` complex.

    A 2-D similarity transform is multiplication by one complex number, which
    turns the least-squares fit into a closed form instead of an SVD per frame.
    """
    return pts[..., 0] + 1j * pts[..., 1]


def mirror_pairs(names) -> tuple:
    """``((i, j), …)`` index pairs of parts that are each other's mirror.

    From the names, because that is where the information is: sleap-nn's
    ``symmetries`` list is empty in every export seen here, and DeepLabCut has
    no equivalent field at all. A pair is only formed when the two names differ
    ONLY by side, so ``Left_Ear``/``Right_Ear`` pairs and ``Left_Ear``/
    ``Right_Paw`` does not.
    """
    out = []
    lowered = [str(n) for n in names]
    for left_re, right_re in _SIDES:
        for i, a in enumerate(lowered):
            ma = left_re.match(a)
            if not ma:
                continue
            for j, b in enumerate(lowered):
                if j <= i:
                    continue
                mb = right_re.match(b)
                if mb and ma.group(1).lower() == mb.group(1).lower():
                    if (i, j) not in out:
                        out.append((i, j))
    return tuple(out)


def canonical_shape(z: np.ndarray, min_anchors: int = DEFAULT_MIN_ANCHORS,
                    iters: int = 8):
    """The animal's mean shape, learned from partly-observed frames.

    ``z`` is ``(frames, parts)`` complex, NaN where a part is missing. Returns
    ``(shape (parts,) complex, body length, frames used)`` or ``(None, 0, 0)``.

    The previous implementation averaged only the frames where EVERY part was
    visible, and needed twenty of them. That is the one thing gating makes rare:
    the model loses one part at a time, so on a trace where each of six parts is
    missing 15% of the time, fewer than 40% of frames are complete, and on a
    worse one, none are, and reconstruction switched itself off entirely with a
    log line saying the shape could not be learned.

    So the shape is fitted to what is there. Each round fits every frame's pose
    against the current shape using only that frame's visible parts, then
    re-estimates the shape from every frame pulled back into its own frame of
    reference. Missing entries simply do not contribute to the mean. This is
    ordinary EM on a missing-data Procrustes problem and it converges in a
    handful of rounds; eight is past where the reference stops moving.

    Both returned quantities are in the shape's OWN units, not pixels,
    Procrustes fixes a shape only up to scale, and the per-frame scale lives in
    the transform fitted against it. Every use of the length divides by it, so
    it cancels, which is what lets one tolerance cover any arena.
    """
    seen = np.isfinite(z)
    per_frame = seen.sum(axis=1)
    usable = per_frame >= max(2, min_anchors)
    if not usable.any():
        return None, 0.0, 0
    # Start from the most completely observed frames, centred on their own
    # visible parts. A frame with more parts constrains more of the shape.
    order = np.argsort(-per_frame)
    best = order[0]
    ref = np.where(seen[best], z[best], np.nan)
    ref = ref - np.nanmean(ref)
    scale = np.nanmax(np.abs(ref))
    if not np.isfinite(scale) or scale <= 0:
        return None, 0.0, 0
    ref = ref / scale
    n_parts = z.shape[1]

    for _ in range(int(iters)):
        a, b, _res = fit_poses(z, ref, min_anchors)
        good = np.isfinite(a) & (np.abs(a) > 0) & usable
        if good.sum() < 1:
            return None, 0.0, 0
        # Every frame pulled back into the shape's frame of reference. NaNs
        # stay NaN and are excluded from the mean rather than counted as zero.
        pulled = (z[good] - b[good, None]) / a[good, None]
        with np.errstate(invalid="ignore"):
            new = np.nanmean(pulled, axis=0)
        if not np.isfinite(new).all():
            # A part no frame ever saw cannot be part of the shape. Keep the
            # previous estimate for it rather than propagating NaN into every
            # later fit.
            new = np.where(np.isfinite(new), new, ref)
        new = new - np.nanmean(new)
        span = np.nanmax(np.abs(new))
        if not np.isfinite(span) or span <= 0:
            return None, 0.0, 0
        new = new / span
        if np.nanmax(np.abs(new - ref)) < 1e-6:
            ref = new
            break
        ref = new

    if not np.isfinite(ref).all() or n_parts < 2:
        return None, 0.0, 0
    length = float(np.abs(ref[:, None] - ref[None, :]).max())
    if not np.isfinite(length) or length <= 0:
        return None, 0.0, 0
    return ref, length, int(usable.sum())


def fit_poses(z: np.ndarray, shape: np.ndarray, min_anchors: int):
    """Fit every frame's similarity transform onto ``shape``.

    Returns ``(a, b, residual)``, ``a`` carries rotation and scale, ``b`` the
    position, and ``residual`` is the mean distance from the fitted shape in
    the animal's own units. All NaN for a frame with too few anchors.

    Grouped by which parts are present, so every frame with the same anchor set
    is solved in one vectorised least-squares. Six parts is at most 64 groups
    against tens of thousands of frames.
    """
    n_frames, n_parts = z.shape
    seen = np.isfinite(z)
    a = np.full(n_frames, np.nan, complex)
    b = np.full(n_frames, np.nan, complex)
    resid = np.full(n_frames, np.nan)

    bits = (seen * (1 << np.arange(n_parts))).sum(axis=1)
    for pattern in np.unique(bits):
        present = ((pattern >> np.arange(n_parts)) & 1).astype(bool)
        anchors = np.nonzero(present)[0]
        if anchors.size < min_anchors:
            continue
        rows = np.nonzero(bits == pattern)[0]
        src = shape[anchors]
        dst = z[np.ix_(rows, anchors)]
        S = src - src.mean()
        D = dst - dst.mean(axis=1, keepdims=True)
        denom = float((np.abs(S) ** 2).sum())
        if denom <= 0:
            continue
        fit = (D * np.conj(S)).sum(axis=1) / denom
        a[rows] = fit
        b[rows] = dst.mean(axis=1) - fit * src.mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            resid[rows] = np.abs(fit[:, None] * S - D).mean(axis=1) / np.abs(fit)
    return a, b, resid


def part_residuals(z: np.ndarray, shape: np.ndarray, a: np.ndarray,
                   b: np.ndarray, length: float) -> np.ndarray:
    """How far each visible keypoint sits from the fitted body, per frame.

    ``(frames, parts)`` in body-length units; NaN where the part was not seen
    or the frame could not be fitted. This is the quantity the old code
    collapsed to one number per frame and used only as a veto; it is far more
    useful un-collapsed, because it names WHICH keypoint is wrong.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        fitted = a[:, None] * shape[None, :] + b[:, None]
        scale = np.abs(a) * max(length, 1e-9)
        out = np.abs(fitted - z) / scale[:, None]
    return np.where(np.isfinite(z), out, np.nan)


def undo_swaps(z: np.ndarray, shape: np.ndarray, a: np.ndarray, b: np.ndarray,
               pairs, margin: float = DEFAULT_SWAP_MARGIN):
    """Exchange mirror pairs that fit the body better the other way round.

    A left/right swap is the one error the rest of this module cannot see. It
    keeps the pose plausible, the two points are still on the animal, still
    the right distance apart, so the confidence gate passes it, the jump limit
    often passes it (the parts are close together), and even the whole-frame
    residual barely moves, because swapping two nearby points is close to a
    reflection of a small part of the shape.

    Against the canonical shape it is visible: the shape says which of the two
    sits where relative to the snout and the tail, and a swapped frame fits
    that noticeably worse. A swap that PERSISTED for the whole recording would
    have been learned into the canonical shape and would not be detected, but
    a consistent left/right convention is not an error, it is a naming choice,
    and it is the frames that disagree with the animal's own usual arrangement
    that are wrong. That is exactly what this tests, so no separate temporal
    rule is needed.

    Returns ``(z, {pair_index: frames exchanged})``; ``z`` is a copy.
    """
    z = z.copy()
    changed = {}
    if not pairs:
        return z, changed
    with np.errstate(invalid="ignore", divide="ignore"):
        fitted = a[:, None] * shape[None, :] + b[:, None]
    for i, j in pairs:
        both = np.isfinite(z[:, i]) & np.isfinite(z[:, j]) & np.isfinite(a)
        if not both.any():
            continue
        keep = np.abs(fitted[:, i] - z[:, i]) + np.abs(fitted[:, j] - z[:, j])
        swap = np.abs(fitted[:, i] - z[:, j]) + np.abs(fitted[:, j] - z[:, i])
        take = both & np.isfinite(keep) & np.isfinite(swap) & (
            swap < keep * float(margin))
        n = int(take.sum())
        if n:
            z[take, i], z[take, j] = z[take, j].copy(), z[take, i].copy()
            changed[(i, j)] = n
    return z, changed


def repair_frames(z: np.ndarray, names, *, min_anchors: int = DEFAULT_MIN_ANCHORS,
                  mistrack_residual: float = DEFAULT_MISTRACK_RESIDUAL,
                  swap_margin: float = DEFAULT_SWAP_MARGIN,
                  rounds: int = DEFAULT_ROBUST_ROUNDS,
                  reconstruct: bool = True,
                  smooth_window: int = 7,
                  swap_method: str = "auto",
                  swap_transition_cost: float = 0.15):
    """Un-swap, drop mistracks, and place what is missing. Returns
    ``(z, filled mask, RepairReport)``.

    The order is forced by what each stage can see:

    1. **Swaps first.** A swapped pair looks like two mistracked keypoints, so
       dropping mistracks first would throw away two good detections and lose
       the evidence that they were merely exchanged.
    2. **Then mistracks, iteratively.** The bad point is corrupting the fit
       that has to find it, so one pass is biased toward it. Each round refits
       without the points already rejected and asks again.
    3. **Then reconstruction**, which is the only stage that adds anything, and
       which needs the other two to have finished removing what would otherwise
       be reconstructed *from*.
    """
    z = np.asarray(z, complex).copy()
    n_frames, n_parts = z.shape
    report = RepairReport()
    filled = np.zeros(z.shape, bool)
    if n_parts < max(2, min_anchors):
        return z, filled, report

    shape, length, used = canonical_shape(z, min_anchors)
    if shape is None:
        logger.info("the animal's shape could not be learned from this trace, "
                    "so nothing was repaired across keypoints")
        return z, filled, report
    report.body_length, report.shape_frames = float(length), int(used)

    pairs = mirror_pairs(names)
    report.pairs = tuple((str(names[i]), str(names[j])) for i, j in pairs)

    a, b, _res = fit_poses(z, shape, min_anchors)
    if pairs:
        method = str(swap_method).lower()
        if method == "auto":
            # Neither test is right for both regimes, so the trace decides.
            # See swap_resolve.DEFAULT_AUTO_THRESHOLD for the measurement.
            from tools.offline_analysis.engine.swap_resolve import (
                choose_swap_method)
            method, frac = choose_swap_method(z, names, pairs)
            logger.info(
                "swap resolver: about %.0f%% of the judged frames look "
                "mirror-swapped, so the %s test is the one to trust here",
                frac * 100.0, method)
            report.method = method
        if method == "anatomical":
            # Side decided from the body axis in each frame and resolved along
            # time, so a run of swapped frames cannot outvote the reference the
            # way it does when the reference is a shape learned from the same
            # trace. Measured against known truth: at 40 % of the session
            # swapped this recovers 100 % where the shape test recovers 3.9 %,
            # and at 70 % it recovers 100 % where the shape test recovers none
            # and corrupts 270 good frames.
            from tools.offline_analysis.engine.swap_resolve import resolve_swaps
            z, sr = resolve_swaps(z, names, pairs,
                                  transition_cost=swap_transition_cost)
            swapped = {k: v for k, v in sr.pairs.items()}
            report.swapped.update(swapped)
        else:
            z, exchanged = undo_swaps(z, shape, a, b, pairs, swap_margin)
            swapped = {}
            for (i, j), n in exchanged.items():
                swapped[f"{names[i]}<->{names[j]}"] = n
            report.swapped.update(swapped)
        if swapped:
            # The shape was learned from a trace containing the swaps; with
            # them undone it is worth re-learning before anything is judged
            # against it.
            relearned, relength, reused = canonical_shape(z, min_anchors)
            if relearned is not None:
                shape, length = relearned, relength
                report.body_length, report.shape_frames = (float(relength),
                                                           int(reused))
            a, b, _res = fit_poses(z, shape, min_anchors)

    # Mistracks: confident detections in the wrong place.
    dropped = np.zeros(z.shape, bool)
    rows = np.arange(n_frames)
    for _ in range(max(1, int(rounds))):
        r = part_residuals(z, shape, a, b, length)
        # ONE part per frame per round, the worst, and only when it stands
        # clear of the rest of that frame.
        #
        # Neither half of that is optional. A mistracked keypoint drags the
        # least-squares fit for its whole frame, so every OTHER part's residual
        # rises with it: on the synthetic trace a snout displaced by a body
        # length sat at 1.56 while the five good parts, judged against the fit
        # it had corrupted, sat at 0.27-0.69 against a 0.35 threshold. Testing
        # each part against the threshold alone would therefore condemn the
        # innocent five; gating on the frame's median residual instead, which
        # was the first attempt, refuses to condemn anyone, because the
        # outlier inflates the median that is supposed to license accusing it.
        #
        # Comparing the worst against the REST of its own frame escapes that
        # circularity: it asks whether one part disagrees with the fit more
        # than the others do, which is exactly what distinguishes one bad
        # keypoint from a badly-fitted frame. Removing it and re-fitting then
        # collapses the survivors' residuals, so the next round judges against
        # a fit the outlier is no longer bending.
        finite = np.isfinite(r)
        if not finite.any():
            break
        ranked = np.where(finite, r, -np.inf)
        worst = np.argmax(ranked, axis=1)
        worst_val = ranked[rows, worst]
        rest = np.where(finite, r, np.nan).copy()
        rest[rows, worst] = np.nan
        # Frames whose only finite residual was the worst one have no "rest" to
        # compare against. ``nanmedian`` warns on an all-NaN row, so they are
        # excluded before the call rather than after it, a warning per frame
        # would be tens of thousands of lines on a real recording.
        has_rest = np.isfinite(rest).any(axis=1)
        rest_median = np.full(n_frames, np.nan)
        if has_rest.any():
            rest_median[has_rest] = np.nanmedian(rest[has_rest], axis=1)
        clear_of = np.where(np.isfinite(rest_median),
                            MISTRACK_SEPARATION * rest_median, 0.0)
        take = (np.isfinite(worst_val)
                & (worst_val > float(mistrack_residual))
                & (worst_val > clear_of))
        # Two points is the least that can still say where the animal is;
        # below that the frame carries no position at all and dropping more
        # only destroys what little is left.
        take &= (np.isfinite(z).sum(axis=1) - 1) >= 2
        if not take.any():
            break
        bad = np.zeros(z.shape, bool)
        bad[rows[take], worst[take]] = True
        dropped |= bad
        z = np.where(bad, np.nan + 0j, z)
        a, b, _res = fit_poses(z, shape, min_anchors)

    for j in range(n_parts):
        n = int(dropped[:, j].sum())
        if n:
            report.mistracked[str(names[j])] = n
    report.unfittable = int((~np.isfinite(a)).sum())

    if reconstruct:
        z, filled, placed = place_missing(z, shape, a, b, min_anchors,
                                          smooth_window)
        for j in range(n_parts):
            n = int(placed[j])
            if n:
                report.reconstructed[str(names[j])] = n
    return z, filled, report


def place_missing(z: np.ndarray, shape: np.ndarray, a: np.ndarray,
                  b: np.ndarray, min_anchors: int, smooth_window: int):
    """Put a keypoint where the rest of the animal says it was.

    The transform is smoothed along time before it is used. Fitting each frame
    independently is measurably worse than doing nothing: the fit carries its
    own noise, so a keypoint placed from it inherits a tremor the detected
    keypoints do not have, and the ears, which sit close to the body axis
    where a degree of rotation error moves them across it, changed sides five
    times more often. An animal's orientation is a smooth function of time; its
    estimate should be too.

    Bounded, though. Interpolating the pose between two fitted frames is sound;
    continuing it into a stretch where the animal was never fitted is
    extrapolation, and it would confidently draw a mouse that had left.
    """
    n_frames, n_parts = z.shape
    seen = np.isfinite(z)
    out = z.copy()
    filled = np.zeros(z.shape, bool)
    placed = np.zeros(n_parts, int)

    usable = np.isfinite(a) & (np.abs(a) > 0)
    usable &= temporally_consistent(a, b, usable, smooth_window)
    if usable.sum() < 2:
        return out, filled, placed

    a_s = _smooth_complex(a, usable, smooth_window)
    b_s = _smooth_complex(b, usable, smooth_window)
    near = _within(usable, smooth_window)
    ready = near & np.isfinite(a_s) & np.isfinite(b_s)
    for j in range(n_parts):
        gap = ready & ~seen[:, j]
        if not gap.any():
            continue
        out[gap, j] = a_s[gap] * shape[j] + b_s[gap]
        filled[gap, j] = True
        placed[j] = int(gap.sum())
    return out, filled, placed


def temporally_consistent(a: np.ndarray, b: np.ndarray, ok: np.ndarray,
                          window: int = 7,
                          limit: float = DEFAULT_POSE_JUMP) -> np.ndarray:
    """Which frames' fitted poses agree with their neighbours in time.

    The last thing that can go wrong, and the only one the shape alone cannot
    catch. Per-frame robust fitting has a breakdown point: once the corrupted
    keypoints outnumber the good ones, the fit follows the corrupted majority
    and every residual looks reasonable, the survivors fit an exact transform,
    so nothing is flagged. On the test case with three of five anchors thrown
    60-80 units across a 24-unit animal, two were caught and the third hid
    behind the exact fit its removal produced.

    Time settles it. An animal's position and orientation are smooth functions
    of it, so a frame whose fitted pose leaps away from the frames either side
    is wrong however self-consistently wrong it is. This is the same idea as
    the per-keypoint jump limit, applied one level up, to the body's pose
    rather than to a keypoint, and it is available only offline, because it
    needs the frames after this one.

    A rejected frame is not discarded: it loses the right to place keypoints
    from its OWN fit, and gets the interpolated pose of its neighbours instead,
    which is the better answer anyway.
    """
    n = a.size
    good = np.zeros(n, bool)
    idx = np.nonzero(ok)[0]
    if idx.size < 3:
        return ok.copy()
    # A rolling median is the reference: it ignores the excursions being looked
    # for instead of being pulled toward them, which a rolling mean would be.
    half = max(1, int(window) // 2)
    ref_b = _rolling_median(b, ok, half)
    ref_a = _rolling_median(a, ok, half)
    scale = np.abs(ref_a)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Both deviations in body-scale units, so one limit covers any arena.
        moved = np.abs(b - ref_b) / np.where(scale > 0, scale, np.nan)
        turned = np.abs(a - ref_a) / np.where(scale > 0, scale, np.nan)
    good = ok & ((~np.isfinite(moved)) | (moved <= limit)) & (
        (~np.isfinite(turned)) | (turned <= limit))
    lost = int(ok.sum() - good.sum())
    if lost:
        logger.debug("%d frames' fitted pose disagreed with their neighbours "
                     "and were not used to place keypoints", lost)
    # Never reject everything: if the pose is genuinely erratic throughout, the
    # reference is meaningless and this guard has nothing to say.
    return good if good.sum() >= max(2, int(0.2 * ok.sum())) else ok.copy()


def _rolling_median(v: np.ndarray, ok: np.ndarray, half: int) -> np.ndarray:
    """Centred rolling median of a complex series over its valid entries."""
    n = v.size
    out = np.full(n, np.nan, complex)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return out
    # ``idx`` is sorted, so the window is a RANGE inside it and both ends come
    # from a binary search. Testing every valid index against every frame made
    # this quadratic, which is invisible on a test series and the dominant
    # cost of a repair on an hour of video.
    real_ok = v.real[idx].copy()
    imag_ok = v.imag[idx].copy()
    lo_i = np.searchsorted(idx, np.maximum(0, np.arange(n) - half), "left")
    hi_i = np.searchsorted(idx, np.minimum(n, np.arange(n) + half + 1), "left")
    for k in range(n):
        a, b = lo_i[k], hi_i[k]
        if a >= b:
            continue
        # Contiguous slices of the valid entries, so no index array is built
        # per frame either.
        out[k] = np.median(real_ok[a:b]) + 1j * np.median(imag_ok[a:b])
    return out


def _smooth_complex(v: np.ndarray, ok: np.ndarray, window: int) -> np.ndarray:
    """Smooth a complex series over time, interpolating the gaps first.

    Filling is the point here, unlike the keypoint smoother, which deliberately
    puts holes back where it found them, so the interpolation is explicit and
    happens before the smoothing rather than being undone after it.
    """
    from tools.offline_analysis.engine.refine import smooth_zero_lag

    idx = np.arange(v.size)
    good = np.nonzero(ok)[0]
    if good.size < 2:
        return np.full(v.size, np.nan, complex)
    out = np.empty(v.size, complex)
    out.real = smooth_zero_lag(np.interp(idx, good, v.real[good]), window, 2)
    out.imag = smooth_zero_lag(np.interp(idx, good, v.imag[good]), window, 2)
    return out


def _within(ok: np.ndarray, window: int) -> np.ndarray:
    """True where a ``True`` in ``ok`` lies within half a window."""
    reach = max(1, int(window) // 2)
    near = ok.copy()
    for shift in range(1, reach + 1):
        near[shift:] |= ok[:-shift]
        near[:-shift] |= ok[shift:]
    return near


# ── along time: the forward-backward smoother ────────────────────────────

def process_variance(x: np.ndarray, y: np.ndarray, t_s: np.ndarray,
                     scale: float = DEFAULT_PROCESS_SCALE,
                     measure_px: float = DEFAULT_MEASURE_PX) -> float:
    """How hard the animal actually accelerates, in px²/s⁴.

    Estimated from the trace rather than fixed, because a constant chosen on
    one rig is wrong on the next: the same number means different motion at a
    different frame rate, a different zoom or a different animal.

    The subtlety that decides whether the smoother smooths at all: a trace's
    apparent acceleration is mostly its own measurement noise. Differencing
    twice amplifies that noise sixfold (``var(x[k+1] - 2x[k] + x[k-1])`` is
    ``6σ²`` for independent errors), so on this rig's data, roughly 1 px of
    localisation jitter at 30 fps, the naive estimate came out about thirty
    times the animal's real acceleration. Feeding that back as process noise
    tells the filter the animal can go anywhere, so it follows every
    measurement and removes no jitter whatsoever. Measured on a synthetic trace
    with known truth, that cost the whole benefit: 1.10 px median error against
    0.77 for the plain Savitzky-Golay window it was supposed to beat.

    So the known measurement contribution is subtracted before the rest is
    called motion. What remains can be negative when the animal moves less than
    the noise, which is a real answer; it means "as smooth as the model can
    see", and is floored rather than trusted.
    """
    xs = np.asarray(x, float)
    ys = np.asarray(y, float)
    t = np.asarray(t_s, float)
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(t)
    if ok.sum() < 8:
        return float(scale) * 1e3
    xs, ys, t = xs[ok], ys[ok], t[ok]
    dt = np.diff(t)
    step = dt[np.isfinite(dt) & (dt > 0)]
    h = float(np.median(step)) if step.size else 1.0
    if not np.isfinite(h) or h <= 0:
        h = 1.0

    # Second difference of POSITION, not of velocity: it is the quantity whose
    # noise inflation is known exactly, which is what makes the subtraction
    # possible.
    out = []
    sigma2 = max(float(measure_px), 1e-3) ** 2
    for series in (xs, ys):
        d2 = series[2:] - 2.0 * series[1:-1] + series[:-2]
        d2 = d2[np.isfinite(d2)]
        if d2.size < 4:
            continue
        # A robust spread: the excursions this must tolerate are exactly the
        # values that would inflate a variance computed the usual way.
        spread = 1.4826 * float(np.median(np.abs(d2 - np.median(d2))))
        motion = max(spread ** 2 - 6.0 * sigma2, 0.0)
        out.append(motion / h ** 4)
    if not out:
        return float(scale) * 1e3
    # A floor, so a perfectly still animal does not produce a filter that
    # refuses to move at all and lags every real dart.
    return float(scale) * max(float(np.mean(out)), sigma2 / h ** 4 * 0.01)


def rts_smooth(x, y, conf, t_s, *,
               measure_px: float = DEFAULT_MEASURE_PX,
               process_scale: float = DEFAULT_PROCESS_SCALE,
               conf_floor: float = DEFAULT_SMOOTH_FLOOR,
               keep_holes: bool = True):
    """Forward Kalman filter, backward RTS smoother, on a constant-velocity model.

    This is the stage the live pipeline cannot have. One-Euro online is causal,
    it knows only the past and pays for smoothing with lag. A centred
    Savitzky-Golay window, which is what this replaces, removes the lag but
    treats every sample as equally trustworthy and every gap as something to
    interpolate across blindly.

    A smoother does neither. It carries a position AND a velocity with their
    uncertainty, so:

    * a **missing** frame is a prediction with growing uncertainty, not a
      straight line drawn between its neighbours;
    * a **weak** detection is folded in according to how weak it is, the
      measurement variance is ``measure_px² / confidence``, so a 0.2-confidence
      point moves the estimate a fifth as far as a 1.0 one;
    * the **backward** pass then gives every frame the benefit of every later
      frame, which is what makes it a smoother rather than a filter.

    Returns ``(x, y)``. With ``keep_holes`` the frames that had no usable
    detection come back NaN, the smoother's estimate for them exists and is
    good, but returning it would quietly convert "not observed" into "observed
    here", and the distance travelled by an animal that was never seen is not a
    number this pipeline should invent. ``place_missing`` is the stage allowed
    to fill, and it says so in its own report.
    """
    xs = np.asarray(x, float)
    ys = np.asarray(y, float)
    t = np.asarray(t_s, float)
    n = xs.size
    if n < 3:
        return xs.copy(), ys.copy()
    c = (np.asarray(conf, float) if conf is not None
         else np.ones(n))
    if c.size != n:
        c = np.ones(n)
    # Having a position and deserving trust are two different facts, and
    # conflating them threw away work: a hole the interpolator filled, or a
    # keypoint the body shape placed, carries whatever confidence the DROPPED
    # detection had, often near zero, and was then discarded here as
    # unobserved. Presence decides what is returned; confidence only decides
    # how hard the measurement pulls.
    observed = np.isfinite(xs) & np.isfinite(ys)
    if observed.sum() < 2:
        return xs.copy(), ys.copy()
    trust = np.clip(np.nan_to_num(c, nan=1.0), float(conf_floor), 1.0)

    if not np.isfinite(t).all() or t.size != n:
        t = np.arange(n, dtype=float)
    dt = np.diff(t)
    # A non-increasing timestamp would give a negative or zero step and turn
    # the prediction inside out. The median step is the honest stand-in.
    step = float(np.median(dt[np.isfinite(dt) & (dt > 0)])) if np.isfinite(
        dt).any() else 1.0
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, step)

    q = process_variance(xs, ys, t, process_scale, measure_px)
    r_base = max(float(measure_px), 1e-3) ** 2
    r = np.where(observed, r_base / trust, np.inf)

    # Two independent constant-velocity trackers, one per axis, run together as
    # a length-2 leading dimension so the whole thing is one pass.
    meas = np.stack([np.where(observed, xs, 0.0), np.where(observed, ys, 0.0)])
    first = int(np.argmax(observed))
    state = np.stack([meas[:, first], np.zeros(2)], axis=1)       # (2 axes, 2)
    cov = np.array([[[r_base, 0.0], [0.0, q * step ** 2]]] * 2, float)

    xf = np.zeros((n, 2, 2))
    Pf = np.zeros((n, 2, 2, 2))
    xp = np.zeros((n, 2, 2))
    Pp = np.zeros((n, 2, 2, 2))

    # Both matrices are rewritten in place each step rather than rebuilt: at
    # six body parts on a long recording this loop runs a few million times,
    # and two array allocations per step cost more than the arithmetic.
    Fk = np.eye(2)
    Qk = np.empty((2, 2))
    for k in range(n):
        if k == 0:
            pred_x, pred_P = state, cov
        else:
            h = dt[k - 1]
            Fk[0, 1] = h
            pred_x = state @ Fk.T
            pred_P = Fk @ cov @ Fk.T
            # Continuous white-noise acceleration, discretised.
            Qk[0, 0] = q * h ** 3 / 3.0
            Qk[0, 1] = Qk[1, 0] = q * h ** 2 / 2.0
            Qk[1, 1] = q * h
            pred_P = pred_P + Qk
        xp[k], Pp[k] = pred_x, pred_P
        if observed[k]:
            s = pred_P[:, 0, 0] + r[k]
            gain = pred_P[:, :, 0] / s[:, None]                   # (2 axes, 2)
            innov = meas[:, k] - pred_x[:, 0]
            state = pred_x + gain * innov[:, None]
            cov = pred_P - gain[:, :, None] * pred_P[:, None, 0, :]
        else:
            state, cov = pred_x, pred_P
        xf[k], Pf[k] = state, cov

    # Backward pass: every frame gets the benefit of every later frame.
    #
    # The smoothed COVARIANCE the textbook recursion also carries is not
    # computed. It feeds only its own next step, never the estimate, and the
    # gain below is built from the filtered and predicted covariances instead.
    xs_out = xf.copy()
    Fk = np.eye(2)
    for k in range(n - 2, -1, -1):
        Fk[0, 1] = dt[k]
        P = Pp[k + 1]                                   # (2 axes, 2, 2)
        # The inverse of a 2x2 in closed form, both axes in one go.
        # ``np.linalg.inv`` is general-purpose, and on matrices this small its
        # per-call dispatch, not the arithmetic, was the cost of this loop.
        det = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
        usable = np.isfinite(det) & (det != 0.0)
        if not usable.any():
            continue
        adj = np.empty_like(P)
        adj[:, 0, 0] = P[:, 1, 1]
        adj[:, 0, 1] = -P[:, 0, 1]
        adj[:, 1, 0] = -P[:, 1, 0]
        adj[:, 1, 1] = P[:, 0, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = Pf[k] @ Fk.T @ (adj / det[:, None, None])
        step_k = xf[k] + np.einsum("aij,aj->ai", gain,
                                   xs_out[k + 1] - xp[k + 1])
        # A singular predicted covariance leaves that axis on the filtered
        # estimate, which is what the general inverse raising did before.
        xs_out[k] = np.where(usable[:, None], step_k, xf[k])

    out_x, out_y = xs_out[:, 0, 0], xs_out[:, 1, 0]
    if keep_holes:
        out_x = np.where(observed, out_x, np.nan)
        out_y = np.where(observed, out_y, np.nan)
    return out_x, out_y


__all__ = ["DEFAULT_MEASURE_PX", "DEFAULT_MISTRACK_RESIDUAL",
           "DEFAULT_MIN_ANCHORS", "DEFAULT_PROCESS_SCALE",
           "DEFAULT_ROBUST_ROUNDS", "DEFAULT_SWAP_MARGIN", "RepairReport",
           "as_complex", "canonical_shape", "fit_poses", "mirror_pairs",
           "part_residuals", "place_missing", "process_variance",
           "repair_frames", "rts_smooth", "undo_swaps"]
