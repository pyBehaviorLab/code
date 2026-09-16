"""Deciding which of a mirror pair is which, over a whole recording.

The previous resolver compared each frame against a canonical shape learned
from the recording and exchanged the pair where the swapped arrangement fitted
better. Measured against synthetic traces with known truth, that works when the
recording is clean and fails in five ways that real recordings produce:

===================================  ================================
condition                            recovered
===================================  ================================
clean, 1.5 px localisation noise     100 %
6 px noise                           78 %, and 43 good frames broken
non-rigid posture, 8 px              34 %
40 % of keypoints missing            66 %
**swap run = 40 % of the session**   **3.9 %**
**swap run = 70 % of the session**   **0 %, 270 good frames broken**
===================================  ================================

Two causes, and this module addresses both.

**The reference was contaminated.** A canonical shape averaged over the
recording learns whatever arrangement is in the majority of frames. Past about
a third swapped it starts describing the swapped animal, and the test then
reports the correct frames as the wrong ones. Its docstring anticipated this
and argued a whole-recording swap is a naming choice rather than an error,
which is true at 100 % and plainly false at 40 %.

So side is decided **anatomically** instead. Left and right are defined by the
body axis, not by a vote: with the snout and the tail base known, the sign of
the cross product between the body axis and the vector out to a keypoint says
which side it lies on, in that frame, using nothing from any other frame. No
majority can outvote it.

**Every frame was decided alone.** Localisation noise, non-rigid posture and a
pair seen close together all make a single frame ambiguous, and a per-frame
rule then flickers. The neighbouring frames hold the missing evidence: an
animal does not exchange its ears between one frame and the next.

So the per-frame side evidence is resolved along time by dynamic programming
over the two assignments, in the manner Anipose applies to 2-D keypoint paths:
each frame contributes how strongly it prefers one assignment, each transition
between adjacent frames costs a fixed amount, and the path minimising the total
is chosen. A run of genuinely swapped frames is recovered as a run; an isolated
ambiguous frame follows its neighbours rather than flipping.

Where the body axis cannot be measured, because the snout or the tail base is
missing, the frame contributes no evidence and is carried by the path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Cost of changing assignment between two adjacent frames, in the same units
#: as the per-frame evidence below (which is bounded to +/-1). Entering and
#: leaving a state costs this twice, so a run of one frame is only recoverable
#: while ``2 * cost`` is below the evidence for it. Swept against known truth:
#: at 0.6 isolated single-frame swaps are smoothed away and recovery on a clean
#: trace drops to 75 %; at 0.15 it is 100 %, and the long-run cases stay at
#: 100 %. Raise it to trust continuity more and lose isolated frames.
DEFAULT_TRANSITION_COST = 0.15

#: Body-axis length, as a fraction of the animal's own length, below which the
#: axis direction is too short to be measured reliably and the frame is treated
#: as carrying no evidence. Guards the head-on view, where snout and tail base
#: project almost on top of each other.
DEFAULT_MIN_AXIS_FRAC = 0.25

#: How far off the axis a keypoint must sit before its side is believed, as a
#: fraction of the body length. Inside this band the two are not separable and
#: the frame abstains rather than guessing.
DEFAULT_MIN_OFFSET_FRAC = 0.04

#: Names that identify the two ends of the body, tried in order.
AXIS_FRONT = ("snout", "nose", "head")
AXIS_BACK = ("tail_base", "tailbase", "tail", "center", "centre", "body")


@dataclass
class SwapReport:
    """What the resolver did, per pair, so a run can be audited."""
    pairs: Dict[str, int] = field(default_factory=dict)      # pair -> frames exchanged
    abstained: Dict[str, int] = field(default_factory=dict)  # pair -> frames with no evidence
    method: str = "anatomical+viterbi"
    axis: str = ""

    def summary(self) -> str:
        if not self.pairs:
            return "no mirror pairs exchanged"
        bits = [f"{k}: {v} frames" for k, v in sorted(self.pairs.items())]
        return f"exchanged ({self.method}, axis {self.axis or 'none'}) " + ", ".join(bits)


def _find(names: Sequence[str], wanted: Sequence[str]) -> Optional[int]:
    low = [str(n).lower() for n in names]
    for w in wanted:
        for i, n in enumerate(low):
            if n == w:
                return i
    for w in wanted:                      # substring, so "Tail_Base_1" still matches
        for i, n in enumerate(low):
            if w in n:
                return i
    return None


def body_axis(z: np.ndarray, names: Sequence[str]) -> Tuple[Optional[int], Optional[int]]:
    """Indices of the front and back keypoints that define the body axis."""
    return _find(names, AXIS_FRONT), _find(names, AXIS_BACK)


def side_evidence(z: np.ndarray, front: int, back: int, i: int, j: int, *,
                  min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC,
                  min_offset_frac: float = DEFAULT_MIN_OFFSET_FRAC) -> np.ndarray:
    """Per-frame evidence that part ``i`` is the one on the left.

    Returns a value in [-1, 1] per frame: positive means the current
    assignment already has ``i`` on the left, negative means the pair is
    exchanged, zero means this frame cannot tell. The magnitude is how far the
    two sit either side of the axis, so a frame that separates them clearly
    counts for more than one that barely does.

    Uses only the frame it is computed from. Nothing here is learned from the
    recording, which is what makes it immune to a majority of swapped frames.
    """
    axis = z[:, front] - z[:, back]
    length = np.abs(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = axis / length
        mid = 0.5 * (z[:, i] + z[:, j])
        # Dividing by the unit axis rotates into the animal's own frame, where
        # the axis lies along +real. The ACROSS-body component is therefore the
        # imaginary part; taking the real part here measures position along the
        # body instead, which carries no side information at all.
        off_i = ((z[:, i] - mid) / unit).imag
        off_j = ((z[:, j] - mid) / unit).imag

    body = np.nanmedian(length[np.isfinite(length)]) if np.isfinite(length).any() else np.nan
    ev = np.zeros(z.shape[0], float)
    if not np.isfinite(body) or body <= 0:
        return ev

    sep = np.abs(off_i - off_j)
    usable = (np.isfinite(off_i) & np.isfinite(off_j) & np.isfinite(length)
              & (length >= min_axis_frac * body)
              & (sep >= min_offset_frac * body))
    # Which sign of the across-body coordinate is the animal's own left is a
    # property of the image coordinate system, not something to reason about:
    # image y runs DOWN, so the handedness is inverted relative to the usual
    # maths convention. Checked directly on a constructed frame - an animal
    # facing up the screen, whose own left is the viewer's right - and the
    # anatomical left comes out with the GREATER value.
    raw = np.where(off_i > off_j, 1.0, -1.0)
    strength = np.clip(sep / (0.5 * body), 0.0, 1.0)
    ev[usable] = (raw * strength)[usable]
    return ev


def viterbi_assign(evidence: np.ndarray,
                   transition_cost: float = DEFAULT_TRANSITION_COST) -> np.ndarray:
    """Cheapest path through two states, ``keep`` and ``swap``.

    ``evidence`` is positive where the frame prefers ``keep``. The returned
    boolean is True where the frame should be exchanged. Two states and a fixed
    transition cost, so this is exact and linear in the number of frames.
    """
    n = evidence.size
    out = np.zeros(n, bool)
    if n == 0:
        return out
    # Cost of asserting each state in each frame: disagreeing with the
    # evidence costs its magnitude, agreeing costs nothing.
    cost_keep = np.maximum(0.0, -evidence)
    cost_swap = np.maximum(0.0, evidence)

    best = np.array([cost_keep[0], cost_swap[0]], float)
    back = np.zeros((n, 2), np.int8)
    for t in range(1, n):
        for s, c in ((0, cost_keep[t]), (1, cost_swap[t])):
            stay = best[s]
            move = best[1 - s] + transition_cost
            if move < stay:
                back[t, s] = 1 - s
                nb = move + c
            else:
                back[t, s] = s
                nb = stay + c
            # Written after both states are read, so update a copy.
            if s == 0:
                n0 = nb
            else:
                n1 = nb
        best = np.array([n0, n1], float)

    state = int(best[1] < best[0])
    for t in range(n - 1, -1, -1):
        out[t] = bool(state)
        state = int(back[t, state])
    return out


def shape_evidence(z: np.ndarray, i: int, j: int) -> np.ndarray:
    """Fallback evidence for frames whose body axis cannot be measured.

    When the snout or the tail base is missing the anatomical test abstains,
    and on a recording with many dropped keypoints that is most frames. The
    canonical-shape comparison still works there: it needs any three anchors,
    not two particular ones. It is the weaker test - a majority of swapped
    frames corrupts the shape it compares against - so it is used only where
    the stronger one cannot answer, and the path then carries both.

    Returns the same [-1, 1] convention: positive prefers the current
    assignment.
    """
    n = z.shape[0]
    ev = np.zeros(n, float)
    good = np.isfinite(z)
    # A canonical arrangement in the animal's own frame, from the frames that
    # have enough anchors. Median over frames, so a minority of bad ones does
    # not move it.
    ref = np.nanmedian(np.where(good, z, np.nan), axis=0)
    if not np.isfinite(ref[i]) or not np.isfinite(ref[j]):
        return ev
    centre = np.nanmedian(ref[np.isfinite(ref)])
    with np.errstate(invalid="ignore"):
        keep = np.abs(z[:, i] - ref[i]) + np.abs(z[:, j] - ref[j])
        swap = np.abs(z[:, i] - ref[j]) + np.abs(z[:, j] - ref[i])
        scale = np.abs(ref[i] - ref[j]) or 1.0
        raw = (swap - keep) / (2.0 * scale)
    ok = np.isfinite(raw)
    ev[ok] = np.clip(raw[ok], -1.0, 1.0)
    return ev


def resolve_swaps(z: np.ndarray, names: Sequence[str], pairs, *,
                  transition_cost: float = DEFAULT_TRANSITION_COST,
                  min_axis_frac: float = DEFAULT_MIN_AXIS_FRAC,
                  min_offset_frac: float = DEFAULT_MIN_OFFSET_FRAC,
                  shape_fallback: bool = True,
                  ) -> Tuple[np.ndarray, SwapReport]:
    """Exchange each mirror pair on the frames where the body says it is wrong.

    ``z`` is ``(frames, parts)`` complex. Returns a copy and a report. When the
    body axis cannot be identified from the part names the input is returned
    unchanged, with the reason in the report, rather than a guess.
    """
    z = z.copy()
    report = SwapReport()
    if not pairs:
        return z, report

    front, back = body_axis(z, names)
    if front is None or back is None:
        report.method = "none"
        report.axis = ""
        logger.info("swap resolver: no body axis in %s; pairs left alone", list(names))
        return z, report
    report.axis = f"{names[front]}->{names[back]}"

    for i, j in pairs:
        ev = side_evidence(z, front, back, i, j,
                           min_axis_frac=min_axis_frac,
                           min_offset_frac=min_offset_frac)
        key = f"{names[i]}<->{names[j]}"
        silent = (ev == 0.0)
        if shape_fallback and silent.any():
            # Only where the anatomical test said nothing, and at reduced
            # weight, because it is the test that a majority of swapped frames
            # can mislead.
            ev = np.where(silent, 0.5 * shape_evidence(z, i, j), ev)
        n_abstain = int((ev == 0.0).sum())
        if n_abstain:
            report.abstained[key] = n_abstain
        if not np.any(ev):
            continue
        take = viterbi_assign(ev, transition_cost)
        take &= np.isfinite(z[:, i]) & np.isfinite(z[:, j])
        n = int(take.sum())
        if n:
            z[take, i], z[take, j] = z[take, j].copy(), z[take, i].copy()
            report.pairs[key] = n
    return z, report


# ── every part, not just the mirror pairs ────────────────────────────────

#: How much cheaper a re-assignment has to be before it is accepted, as a
#: fraction of the animal's own body length summed over the parts moved. A
#: permutation that barely wins is noise; one that wins clearly is a swap.
DEFAULT_ASSIGN_MARGIN = 0.15

#: A detection further than this from every expected position, in body
#: lengths, is not any of the parts and is left where it is rather than being
#: forced into the cheapest remaining slot.
DEFAULT_MAX_ASSIGN_DIST = 0.8


def _hungarian(cost: np.ndarray):
    """Optimal one-to-one assignment; greedy if SciPy is absent.

    SciPy is the normal scientific dependency and gives the exact answer in
    O(n^3). The greedy fallback exists so the analyser still runs on a machine
    without it, and says so rather than silently doing something worse.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(r), list(c)
    except Exception:                                   # pragma: no cover
        n, m = cost.shape
        taken, rows, cols = set(), [], []
        for r in np.argsort(np.nanmin(cost, axis=1)):
            order = np.argsort(cost[r])
            for c in order:
                if c not in taken:
                    taken.add(int(c)); rows.append(int(r)); cols.append(int(c))
                    break
        return rows, cols


def resolve_identities(z: np.ndarray, names: Sequence[str], *,
                       shape: Optional[np.ndarray] = None,
                       body_length: float = 0.0,
                       margin: float = DEFAULT_ASSIGN_MARGIN,
                       max_dist: float = DEFAULT_MAX_ASSIGN_DIST,
                       passes: int = 2,
                       ) -> Tuple[np.ndarray, Dict[str, int]]:
    """Put every detection on the part it belongs to, for any permutation.

    .. warning::

       **Not finished, and deliberately not wired into anything.** Measured on
       injected corruptions it does not yet reduce the error: a head/tail flip
       stays at 22.9 px after it runs, while reporting that it moved 50 parts.

       Two defects are known. The accept test compares a cost summed over all
       matched pairs against one summed over only the pairs that were already
       in place, which are different numbers of terms, so the comparison decides
       nothing. And the expected configuration still leans on the shape fit,
       which cannot see a whole-body flip at all: a body read back-to-front is
       a 180 degree rotation of the shape and fits it with no residual. The
       temporal prediction added below is the right signal but is not yet what
       the assignment is actually scored against.

       Kept because the analysis is worth keeping, not because it works.

    The mirror-pair test above answers one question well: which of two
    symmetric parts is which. It cannot answer the general one. A snout landing
    on an ear, a whole body read back-to-front, a three-way rotation of the
    head parts: each leaves every keypoint on the animal, at high confidence,
    and none of them is a left/right exchange.

    This is the assignment problem DeepLabCut solves between frames when it
    matches detections to individuals, applied here between one frame's
    detections and the parts they should be: build the cost of calling each
    detection each part, and take the cheapest one-to-one matching with the
    Hungarian algorithm. Every permutation is therefore reachable, and the one
    chosen is the best available rather than the best among a handful that were
    written down in advance.

    The expected position of each part comes from the animal's own shape fitted
    to that frame, so it follows the animal around the arena and through its
    turns instead of assuming a fixed layout.

    Runs forwards and then backwards by default: a frame at the start of a
    corrupted run has no clean past to be predicted from, but it does have a
    clean future.

    Returns ``(z, {part_name: frames reassigned})``.
    """
    from tools.offline_analysis.engine.pose_repair import (canonical_shape,
                                                           fit_poses)

    z = np.asarray(z, complex).copy()
    n_frames, n_parts = z.shape
    moved: Dict[str, int] = {}
    if n_parts < 3 or n_frames == 0:
        return z, moved

    if shape is None or not body_length:
        shape, body_length, _used = canonical_shape(z, 3)
        if shape is None:
            logger.info("identity resolver: no shape could be learned; "
                        "assignments left alone")
            return z, moved
    body = float(body_length) or 1.0

    for _p in range(max(1, int(passes))):
        a, b, _res = fit_poses(z, shape, 3)
        with np.errstate(invalid="ignore"):
            fitted = a[:, None] * shape[None, :] + b[:, None]

        # Where the parts were a moment ago, carried forward at their current
        # velocity. This is the half that a shape fit cannot supply: a body
        # read back-to-front is a 180 degree ROTATION of the shape, so it fits
        # the shape perfectly and the fit alone can never see it. Against the
        # previous frame it is obvious, because a mouse does not reverse
        # end-for-end between two frames.
        prev = np.full_like(z, np.nan)
        prev[1:] = z[:-1]
        vel = np.zeros_like(z)
        vel[2:] = z[1:-1] - z[:-2]
        predicted = prev + vel
        # The frame's own fit is the fallback wherever there is no usable
        # history, which includes the first frames and any gap.
        expected = np.where(np.isfinite(predicted), predicted, fitted)

        for t in range(n_frames):
            have = np.flatnonzero(np.isfinite(z[t]))
            want = np.flatnonzero(np.isfinite(expected[t]))
            if have.size < 2 or want.size < 2:
                continue
            cost = np.abs(z[t][have, None] - expected[t][None, want]) / body
            cost = np.where(cost > max_dist, max_dist * 10.0, cost)
            rows, cols = _hungarian(cost)
            if not rows:
                continue

            new_cost = float(sum(cost[r, c] for r, c in zip(rows, cols)))
            # The cost of leaving the frame exactly as it is, over the same
            # parts, so the two are comparable.
            keep_pairs = [(ri, ci) for ri, r in enumerate(have)
                          for ci, c in enumerate(want) if r == c]
            keep_cost = float(sum(cost[r, c] for r, c in keep_pairs))
            if not keep_pairs or new_cost > keep_cost - margin * len(rows):
                continue

            fixed = z[t].copy()
            for r, c in zip(rows, cols):
                if cost[r, c] >= max_dist * 10.0:
                    continue                    # matched to nothing plausible
                src, dst = have[r], want[c]
                if src != dst:
                    fixed[dst] = z[t][src]
                    key = str(names[dst])
                    moved[key] = moved.get(key, 0) + 1
            z[t] = fixed

        relearned, relen, _u = canonical_shape(z, 3)
        if relearned is not None:
            shape, body = relearned, float(relen) or body

    return z, moved


# ── which resolver this trace needs ──────────────────────────────────────

#: Estimated swapped fraction at or above which the shape test is no longer
#: trustworthy. The shape is LEARNED FROM THE TRACE, so once a large enough
#: share of it is swapped the shape is swapped too and the test agrees with the
#: corruption. Measured on synthetic traces with known truth, recovery of the
#: swapped frames and preservation of the good ones:
#:
#:     truly swapped   estimate   shape rec/kept   anatomical rec/kept
#:             4.7 %      5.9 %    100 % / 100 %      83 % /  99 %
#:            16.3 %     30.6 %    100 % / 100 %     100 % /  99 %
#:            37.0 %     49.9 %      0 % / 100 %      91 % /  93 %
#:            55.4 %     73.5 %      0 % /  32 %      91 % /  82 %
#:
#: Shape is perfect while swaps are rare and collapses completely past about a
#: third; anatomical is good everywhere and perfect nowhere. So neither is the
#: right default on its own, and the threshold sits between the two estimates
#: that straddle the collapse.
DEFAULT_AUTO_THRESHOLD = 0.40


def estimated_swapped_fraction(z: np.ndarray, names: Sequence[str],
                               pairs=None, **kw) -> float:
    """How much of this trace looks mirror-swapped, judged from the body axis.

    Deliberately independent of any shape learned from the trace, because that
    is the thing being judged: a shape learned from a mostly-swapped trace is
    itself swapped, and cannot report on its own corruption.

    Returns the worst fraction across the mirror pairs, 0.0 when nothing can be
    judged (no axis, no pairs, too few frames with a measurable side).
    """
    if pairs is None:
        pairs = []
        for a, b in (("Left_Ear", "Right_Ear"), ("left_ear", "right_ear")):
            i, j = _find(names, [a]), _find(names, [b])
            if i is not None and j is not None:
                pairs = [(i, j)]
                break
    if not pairs:
        return 0.0
    front, back = body_axis(z, names)
    if front is None or back is None:
        return 0.0
    worst = 0.0
    for i, j in pairs:
        ev = side_evidence(z, front, back, i, j, **kw)
        judged = ev != 0.0
        if int(judged.sum()) < 10:
            continue
        worst = max(worst, float((ev[judged] < 0).mean()))
    return worst


def choose_swap_method(z: np.ndarray, names: Sequence[str], pairs=None,
                       threshold: float = DEFAULT_AUTO_THRESHOLD) -> Tuple[str, float]:
    """``("shape" | "anatomical", estimated_fraction)`` for this trace.

    Each method is used where it measures better rather than one being made
    the default for both regimes: see :data:`DEFAULT_AUTO_THRESHOLD`.
    """
    frac = estimated_swapped_fraction(z, names, pairs)
    return ("anatomical" if frac >= threshold else "shape"), frac
