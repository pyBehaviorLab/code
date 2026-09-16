"""Check the numbers, in one command.

    python -m source.analysis.validate                 # the definitions
    python -m source.analysis.validate REC.txt [...]   # + real recordings

Two kinds of check, and they answer different questions.

**Definitions** run each statistic in :mod:`source.analysis.measures` against a
case whose answer can be worked out on paper, a point moving 1 cm per frame at
10 fps has travelled 99 cm after 100 frames, and no amount of refactoring may
change that. The expected value is written here as a literal number; a check
that recomputes the answer with the code it is checking proves nothing.

**Identities** run a real recording through the real pipeline and assert the
things that must be true whatever the data says: the time bins must add up to
the session totals, the per-zone distances plus the distance outside every zone
must equal the total, no zone can be occupied for longer than the recording.
These need no ground truth, which is exactly why they work on YOUR recordings.

**Parity** compares two retracks OF THE SAME CLIP that differ only in how they
were run, fp32 against fp16, the full frame against a letterbox, one engine
against another, and reports the keypoint disagreement as mean / p95 / max
pixels. It answers "did switching to TensorRT change my data?" with a number
instead of an opinion. It reads retracks already stored side by side in the
recording's bundle, so it re-runs nothing.

    python -m source.analysis.validate --parity FOLDER

The rule parity exists to enforce: **judge by keypoint error, never by
detection rate.** The sibling project's BGR/RGB channel bug made the mean
keypoint error 48% WORSE while IMPROVING the fraction of frames with a
detection, so a run that finds more keypoints and places them worse is a
regression, and a coverage number read on its own would have called it an
improvement. Coverage is reported here beside the error and never instead of
it.

Every check prints its name, what was expected, what came out, and PASS or
FAIL. The exit status is 1 if anything failed, so it can sit in a script.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from tools.offline_analysis.engine import measures as m


@dataclass
class Check:
    name: str
    expected: Any
    got: Any
    ok: bool
    note: str = ""
    #: Nothing to check, and that is not a failure. Most recordings in a real
    #: project are latency probes with no poses in them; reporting those as
    #: FAIL would bury the one file that is actually wrong.
    skipped: bool = False


def _near(a: float, b: float, tol: float = 1e-6) -> bool:
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(float(a) - float(b)) <= tol


def _check(name: str, expected, got, tol: float = 1e-6, note: str = "") -> Check:
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        ok = _near(expected, got, tol)
    else:
        ok = expected == got
    return Check(name, expected, got, ok, note)


# ── the definitions, against arithmetic ──────────────────────────────────────

def _at_most(name: str, limit: float, got: float, note: str = "") -> Check:
    """A CEILING, not a nearness.

    `_check` passes when two numbers are within a tolerance of each other,
    which is the wrong question for a threshold: a mean displacement of twice
    the limit is exactly `limit` away from it and would have passed.
    """
    ok = not math.isnan(float(got)) and float(got) <= float(limit)
    return Check(name, f"<= {limit:g}", round(float(got), 4), ok, note)


def _straight_line(n: int = 100, step: float = 1.0, fps: float = 10.0):
    """A point moving `step` cm per frame at `fps`. dt[0] = 0, step[0] = 0.

    Built from whole milliseconds, like a real recording, `t_rel` by
    `cumsum(dt)` drifts, and the frame that should sit exactly on 5.000 s
    lands on 4.999999999999998 and falls inside a `< 5.0` window. Half-open
    windows are only a partition if the boundaries are exact.
    """
    from tools.offline_analysis.engine import offline_analysis as oa

    ts_ms = np.arange(n, dtype=float) * (1000.0 / fps)
    dt = oa.dt_seconds(ts_ms)                      # dt[0] = 0, as in a session
    t_rel = (ts_ms - ts_ms[0]) / 1000.0
    step_cm = np.full(n, step)
    step_cm[0] = 0.0
    speed = np.divide(step_cm, dt, out=np.zeros_like(step_cm), where=dt > 0)
    return dt, step_cm, speed, t_rel


def definition_checks() -> List[Check]:
    out: List[Check] = []
    dt, step_cm, speed, t_rel = _straight_line()

    # 100 frames, the first covering no time: 99 intervals of 0.1 s.
    out.append(_check("duration_s", 9.9, m.duration_s(dt)))
    out.append(_check("distance_cm", 99.0, m.distance_cm(step_cm)))
    out.append(_check("mean_speed_cm_s", 10.0, m.mean_speed_cm_s(step_cm, dt)))
    out.append(_check("peak_speed_cm_s", 10.0, m.peak_speed_cm_s(speed)))

    # Half the frames: half the distance, and the SAME mean speed, the point
    # never changed pace, so a mean that moved would be a denominator bug.
    half = t_rel < 5.0
    out.append(_check("distance_cm(masked)", 49.0, m.distance_cm(step_cm, half)))
    out.append(_check("mean_speed_cm_s(masked)", 10.0,
                      m.mean_speed_cm_s(step_cm, dt, half)))
    out.append(_check("duration_s(masked)", 4.9, m.duration_s(dt, half)))

    out.append(_check("distance_cm(empty)", 0.0,
                      m.distance_cm(step_cm, np.zeros(100, bool))))
    out.append(_check("mean_speed_cm_s(empty)", 0.0,
                      m.mean_speed_cm_s(step_cm, dt, np.zeros(100, bool))))

    # Zones. 40 frames of 0.1 s inside → 4.0 s, whatever the debounce says.
    inz = np.zeros(100, bool)
    inz[10:50] = True
    out.append(_check("time_in_s", 4.0, m.time_in_s(inz, dt)))
    e, ex, lat, tot, visits = m.dwell_entries(inz, dt, 0.2)
    out.append(_check("dwell_entries: entries", 1, e))
    out.append(_check("dwell_entries: exits", 1, ex))
    out.append(_check("dwell_entries: total_s", 4.0, tot))
    # The visit starts at the cumulative time of frame 10 = 10 intervals of
    # 0.1 s minus the zero-length first = 0.9 s.
    out.append(_check("dwell_entries: latency_s", 0.9, lat))

    # A flicker shorter than the debounce is time inside, but not an entry.
    flick = np.zeros(100, bool)
    flick[10] = flick[11] = True                       # 0.2 s, min_dwell 0.5
    e2, _, lat2, tot2, _ = m.dwell_entries(flick, dt, 0.5)
    out.append(_check("dwell_entries: flicker is no entry", 0, e2))
    out.append(_check("dwell_entries: flicker is still time", 0.2, tot2))
    out.append(_check("dwell_entries: no entry, no latency", math.nan, lat2))

    # Bouts. Frames 20-59 are still: 40 frames of 0.1 s = 4.0 s, one bout.
    slow = speed.copy()
    slow[20:60] = 0.0
    t_s, n, lat3 = m.freeze_bouts(slow, dt, thr=1.0, min_s=1.0)
    out.append(_check("freeze_bouts: time_s", 4.0, t_s, tol=1e-9))
    out.append(_check("freeze_bouts: bouts", 1, n))
    out.append(_check("freeze_bouts: latency_s", 1.9, lat3))

    # Transitions: A A B B A is two moves. The unzoned frame between them is
    # not a third one.
    out.append(_check("transitions_total", 2,
                      m.transitions_total(["A", "A", "B", "B", "A"])))
    out.append(_check("transitions_total: gaps are not zones", 1,
                      m.transitions_total(["A", "none", "", "B"])))
    out.append(_check("transitions_total: one zone, no transition", 0,
                      m.transitions_total(["A", "A", "A"])))
    out.append(_check("transitions_total: include filters", 0,
                      m.transitions_total(["A", "B", "A"], include=["A"])))

    # Windows.
    out.append(_check("bin_spans: fixed width", [(0.0, 60.0), (60.0, 120.0)],
                      m.bin_spans(120.0, width=60.0)))
    out.append(_check("bin_spans: count rounds to whole seconds",
                      [(0.0, 75.0), (75.0, 150.0), (150.0, 225.0), (225.0, 300.0)],
                      m.bin_spans(299.8, count=4)))
    out.append(_check("bin_spans: an edge opens the last window",
                      [(0.0, 120.0), (120.0, 300.0)],
                      m.bin_spans(299.8, edges=[120.0])))
    out.append(_check("bin_spans: no ask, no bins", [], m.bin_spans(300.0)))
    out.append(_check("bin_mask is half-open", 1,
                      int(np.sum(m.bin_mask(np.array([0.0, 60.0, 120.0]), 60.0, 120.0)))))

    # Partial windows. One frame short of full is not partial; half empty is.
    out.append(_check("is_partial: 59.8 of 60", False, m.is_partial(59.8, 60.0)))
    out.append(_check("is_partial: 30 of 60", True, m.is_partial(30.0, 60.0)))
    return out


# ── the identities, against a real recording ─────────────────────────────────

def identity_checks(path: str, params: Optional[dict] = None) -> List[Check]:
    """What must hold for any recording, whatever is in it."""
    from tools.offline_analysis.engine import offline_analysis as oa

    out: List[Check] = []
    base = {**oa.DEFAULTS, "plots": False, **(params or {})}
    sess = oa.parse_txt(path)
    tr = oa.build_track(sess, base)
    stem = sess.stem

    if not tr.has_pose:
        return [Check(f"{stem}: nothing to check", "-", "-", True,
                      "no pose data in this recording", skipped=True)]

    out.append(_check(f"{stem}: dt sums to the duration",
                      tr.duration_s, m.duration_s(tr.dt), tol=1e-6))

    for mode, label in (({"bin_size_s": 60.0}, "60 s bins"),
                        ({"bin_count": 3}, "3 equal bins"),
                        ({"bin_edges_s": [120.0]}, "an edge at 120 s")):
        rows = oa.session_rows(sess, {**base, **mode})
        row = rows[0]
        unit = "_Distance_m" if tr.calibrated else "_Distance_px"
        total = row.get(unit.lstrip("_"), row.get("Distance_m", row.get("Distance_px", 0)))
        binned = sum(v for k, v in row.items()
                     if k.endswith(unit) and k != unit.lstrip("_"))
        out.append(_check(f"{stem}: {label} sum to the total distance",
                          round(float(total), 4), round(float(binned), 4),
                          tol=1e-3))

        for z in oa.zone_names(sess):
            key = f"{z}_time_s"
            if key not in row:
                continue
            zbins = sum(v for k, v in row.items()
                        if k.endswith(f"_{z}_time_s") and k != key)
            out.append(_check(f"{stem}: {label} sum to {z} time",
                              round(float(row[key]), 3), round(float(zbins), 3),
                              tol=1e-3))
            break                       # one zone is enough to prove the shape

    # Every frame is in exactly one zone or none, and each step is credited to
    # the frame it arrives at, so the parts must add up to the whole.
    row = oa.session_rows(sess, base)[0]
    if tr.calibrated:
        names = oa.zone_names(sess)
        zsum = sum(float(row.get(f"{z}_distance_m", 0.0)) for z in names)
        # The sheet is rounded to 4 decimals, so a sum of N rounded parts can
        # sit up to N half-units above the rounded whole, 0.1 mm across four
        # arms, which is the rounding and not the analysis. Allowing for it is
        # the difference between a check that means something and one that
        # cries wolf on a real recording.
        slack = 5e-5 * (len(names) + 1)
        out.append(_check(f"{stem}: per-zone distance never exceeds the total",
                          True, zsum <= float(row["Distance_m"]) + slack,
                          note=f"zones {zsum:.4f} of {row['Distance_m']:.4f} m"))
        out.append(_check(f"{stem}: mean speed is distance over duration",
                          round(float(row["Distance_m"]) / tr.duration_s, 4),
                          round(float(row["Mean_Speed_m_s"]), 4), tol=1e-3))

    ztime = sum(float(row.get(f"{z}_time_s", 0.0)) for z in oa.zone_names(sess))
    out.append(_check(f"{stem}: zone time never exceeds the recording",
                      True, ztime <= tr.duration_s + 1e-6,
                      note=f"{ztime:.1f} s of {tr.duration_s:.1f} s"))
    out.extend(live_vs_offline(sess, tr, row, base))
    return out


def live_vs_offline(sess, tr, row, params) -> List[Check]:
    """The same recording, measured the way the rig measured it live.

    The live readout and this workbook are two implementations, one
    streaming, one over arrays, and they filter differently: the offline
    track is SMOOTHED (One-Euro, on by default) while the live computer
    de-jitters by a pixel. On a real Y-maze session that is 18.43 m against
    20.57 m for one animal, ~12%, and nothing on either screen says so.

    So the comparison is against the offline track built the way the live path
    measures, unsmoothed, and the smoothed number the user will actually
    read rides along in the note. Failing means the two disagree by more than
    a quarter about the SAME path, which is a recording to look at: usually
    detector jumps the offline clamp refuses and the live path counts.
    """
    if not tr.calibrated or "Distance_m" not in row:
        return []
    from tools.offline_analysis.engine import offline_analysis as oa

    # In pyBehaveTrack this check re-measured the same path with the LIVE
    # metrics computer and failed when the two disagreed by more than a
    # quarter. That comparison needs the rig's metrics module, and reaching
    # for it is exactly the import that would stop this analyser standing on
    # its own, for a cross-check, not for an answer anyone reads.
    #
    # So the check is not performed here. It is left as a named gap rather
    # than a quiet `return []` inside a `try`, because a validator that
    # silently skips a check is worse than one that does not offer it.
    return []

    # The SAME point the offline track follows, by the same rule: the named
    # body part when the recording has it, otherwise the centroid of every
    # point. Feeding the live computer a different keypoint would produce a
    # disagreement about which animal part moved, dressed up as one about how
    # far it went.
    part = params.get("body_part") or ""
    if not sess.kp:
        return []
    if part and part in sess.kp:
        px = np.asarray(sess.kp[part]["x"], float)
        py = np.asarray(sess.kp[part]["y"], float)
    else:
        px = np.nanmean(np.vstack([np.asarray(v["x"], float)
                                   for v in sess.kp.values()]), axis=0)
        py = np.nanmean(np.vstack([np.asarray(v["y"], float)
                                   for v in sess.kp.values()]), axis=0)
    mc = MetricsComputer(MetricsConfig(px_per_cm=tr.ppc))
    for i in range(len(sess.ts_ms)):
        bp = {k: (v["x"][i], v["y"][i], v["conf"][i]) for k, v in sess.kp.items()}
        mc.update((px[i], py[i]), float(sess.ts_ms[i]), {}, bp)
    live_m = float(mc.get_summary().get("total_distance_cm", 0.0)) / 100.0
    reported_m = float(row["Distance_m"])
    # Compare LIKE WITH LIKE. The workbook's number is smoothed and the live
    # one is not, and there is no fixed ratio between those, a jittery track
    # loses far more to smoothing than a clean one, so a band around that
    # comparison would either miss real errors or cry wolf. So the check is
    # against the offline track built the way the live path measures, and the
    # note carries the smoothed number the user will actually read.
    raw = oa.build_track(sess, {**params, "smooth": False, "rolling_median": False})
    raw_m = m.path_length(raw.step_cm) / 100.0
    if raw_m <= 0:
        return []

    # The live path drops any step under a pixel as jitter. That floor is
    # 1/px_per_cm centimetres per frame, 1.3 mm on a 7.93 px/cm arena, about
    # 3.8 cm/s at 30 fps, and a track that never exceeds it measures ZERO
    # live, which is not a disagreement about the same path but a different
    # question. Judged on the outcome, not on the median step: a mouse whose
    # typical frame moves 0.3 px still spends its distance in the big steps,
    # and those recordings agree to within a few percent.
    if live_m < 0.1 * raw_m:
        floor_cm = 1.0 / (raw.ppc or 1.0)
        fps = 1.0 / float(np.median(raw.dt[raw.dt > 0])) if np.any(raw.dt > 0) else 0.0
        return [Check(f"{sess.stem}: live and offline measure the same path",
                      "-", "-", True, skipped=True,
                      note=f"the live path counted {live_m:.2f} m of {raw_m:.2f} m: "
                           f"this track never clears its 1 px jitter floor "
                           f"({floor_cm:.2f} cm per frame"
                           + (f", ~{floor_cm * fps:.1f} cm/s at {fps:.0f} fps" if fps else "")
                           + ")")]
    ratio = live_m / raw_m
    gap = ((reported_m / raw_m - 1.0) * 100.0)
    hint = ""
    if ratio > 1.25:
        # The live path has no teleport clamp; the offline one drops steps
        # implying more than `max_speed_cm_s`. Live measuring MORE than the
        # unsmoothed offline track means the detector jumped and the analysis
        # refused those jumps, which is the analysis working, and a recording
        # worth looking at before it goes in a figure.
        hint = (", the offline clamp is dropping detector jumps this track "
                "has a lot of; check the tracking")
    elif abs(gap) > 40:
        # Two implementations disagreeing by more than the filters explain is
        # usually not a code difference; it is a track made mostly of jitter,
        # where every filter has something different to remove and none of the
        # numbers means much. Say that, rather than leaving a bare ratio.
        hint = ", jitter-dominated track: check this recording before using it"
    return [Check(f"{sess.stem}: live and offline measure the same path",
                  "within 1.25x unsmoothed", f"{ratio:.3f}x",
                  0.8 <= ratio <= 1.25,
                  f"live {live_m:.2f} m · offline unsmoothed {raw_m:.2f} m · "
                  f"workbook {reported_m:.2f} m ({gap:+.0f}% from smoothing)"
                  + hint)]


# ── reporting ────────────────────────────────────────────────────────────────

# ── parity: two ways of running the same clip ────────────────────────────────
#
# Thresholds are for AGREEMENT, not accuracy: they say the two runs describe
# the same animal in the same place, not that either is right. A parity run
# cannot tell you which of two engines is closer to the truth, only a labelled
# clip can, and saying so plainly is more useful than a number that looks like
# it means more than it does.

#: Mean keypoint displacement, in pixels, at which two runs stop being the same
#: data. A quarter of a pixel is comfortably below the ~1 px de-jitter floor the
#: live path already applies, so a run inside it cannot change a measure.
PARITY_MEAN_PX = 0.25
#: The p95, which is where a systematic shift shows up that the mean hides.
PARITY_P95_PX = 1.0
#: The worst single keypoint. Generous, because one bad frame in a long
#: recording is a detection difference rather than a transform bug.
PARITY_MAX_PX = 8.0
#: How much of the clip has to be comparable before the numbers mean anything.
PARITY_MIN_OVERLAP = 0.5


def keypoint_deltas(rows_a, rows_b) -> dict:
    """How far apart two runs put the same keypoints.

    ``rows_a``/``rows_b`` map frame number → ``{body_part: (x, y, conf)}``.
    Only frames and parts present in BOTH are compared, a keypoint one run
    found and the other did not is a coverage difference, counted separately,
    and averaging it in as a zero would hide exactly the case that matters.

    Returns mean / p95 / max displacement in pixels, the number of compared
    keypoints, and each run's own coverage.
    """
    d: List[float] = []
    only_a = only_b = 0
    shared_frames = 0
    for fn, pa in rows_a.items():
        pb = rows_b.get(fn)
        if pb is None:
            continue
        shared_frames += 1
        for part, va in pa.items():
            vb = pb.get(part)
            if vb is None:
                only_a += 1
                continue
            d.append(math.hypot(float(va[0]) - float(vb[0]),
                                float(va[1]) - float(vb[1])))
        only_b += sum(1 for part in pb if part not in pa)
    arr = np.asarray(d, dtype=float) if d else np.zeros(0)
    total_frames = max(len(rows_a), len(rows_b), 1)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "only_a": int(only_a),
        "only_b": int(only_b),
        "overlap": shared_frames / total_frames,
    }


def parity_checks(path_a: str, path_b: str, label: str = "") -> List[Check]:
    """Two stored runs of one clip, compared keypoint by keypoint.

    Both paths are recordings in this repo's own format, so this works on a
    live run against its own retrack just as well as on two retracks, and it
    re-runs nothing, which is what makes it usable on a six-hour session.
    """
    import os

    name = label or f"{os.path.basename(path_a)} vs {os.path.basename(path_b)}"
    try:
        rows_a = _poses_by_frame(path_a)
        rows_b = _poses_by_frame(path_b)
    except Exception as e:                                # pragma: no cover
        return [Check(f"parity {name}", "readable", str(e), False)]
    if not rows_a or not rows_b:
        return [Check(f"parity {name}", "poses in both", "no poses", True,
                      note="nothing to compare", skipped=True)]

    d = keypoint_deltas(rows_a, rows_b)
    out = [Check(f"parity {name}: frames compared",
                 f">= {PARITY_MIN_OVERLAP:.0%}", f"{d['overlap']:.0%}",
                 d["overlap"] >= PARITY_MIN_OVERLAP,
                 note="two runs of different lengths are not the same clip")]
    if not d["n"]:
        out.append(Check(f"parity {name}", "shared keypoints", 0, False,
                         note="the runs share no body-part names"))
        return out
    out += [
        _at_most(f"parity {name}: mean px", PARITY_MEAN_PX, d["mean"],
                 note=f"{d['n']} keypoints compared"),
        _at_most(f"parity {name}: p95 px", PARITY_P95_PX, d["p95"],
                 note="where a systematic shift shows that the mean hides"),
        _at_most(f"parity {name}: max px", PARITY_MAX_PX, d["max"],
                 note="one bad frame is a detection difference, not a "
                      "transform bug"),
    ]
    # Reported, never judged. A run that finds MORE keypoints and places them
    # worse is a regression; this is the number that made the sibling's BGR bug
    # look like an improvement, so it is shown beside the error and given no
    # vote of its own.
    out.append(Check(f"parity {name}: coverage", "reported, not judged",
                     f"+{d['only_b']} / -{d['only_a']} keypoints", True,
                     note="a run that finds more and places them worse is "
                          "still a regression"))
    return out


def _poses_by_frame(path: str) -> dict:
    """``{frame_number: {body_part: (x, y, conf)}}`` from a recording."""
    from tools.offline_analysis import video_data_schema as vds

    header = vds.VideoDataHeader.parse(path)
    col = vds.pose_column(header)
    out = {}
    for i, row in enumerate(vds.iter_rows(path, header)):
        pose = vds.parse_pose(row.get(col))
        if not pose:
            continue
        fn = vds.parse_int(row.get("frame_number"))
        out[int(fn) if fn is not None else i] = pose
    return out


def parity_in_bundle(recording: str) -> List[Check]:
    """Compare every retrack of one recording against the first.

    The runs are already on disk, side by side in ``<stem>.pbanalysis/poses/``,
    each keyed by the fingerprint of the spec that produced it, which is
    precisely a set of "same clip, run differently" pairs. Nothing is re-run
    and no model is needed, so this answers the fp16 question on a laptop.
    """
    import glob as _glob
    import os

    from tools.offline_analysis.engine.session_bundle import BUNDLE_SUFFIX

    stem = os.path.basename(recording)
    for tail in ("_video_data.txt", ".txt"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    poses = os.path.join(os.path.dirname(recording), stem + BUNDLE_SUFFIX,
                         "poses")
    runs = sorted(_glob.glob(os.path.join(poses, "*_video_data.txt")))
    runs = [r for r in runs if "_corrected_" not in os.path.basename(r)]
    if len(runs) < 2:
        return [Check(f"parity {stem}", "two or more runs", len(runs), True,
                      note="one way of running a clip agrees with itself",
                      skipped=True)]
    out: List[Check] = []
    for other in runs[1:]:
        out.extend(parity_checks(runs[0], other, label=_pair_label(runs[0],
                                                                   other)))
    return out


def _pair_label(a: str, b: str) -> str:
    """Name the pair by what actually DIFFERS between the two runs.

    Two eight-character fingerprints tell an operator nothing; "full vs
    letterbox" tells them what they are looking at.
    """
    import os

    try:
        from tools.offline_analysis import video_data_schema as vds

        ha = vds.VideoDataHeader.parse(a).tracker or {}
        hb = vds.VideoDataHeader.parse(b).tracker or {}
    except Exception:                                     # pragma: no cover
        ha = hb = {}
    fa, fb = _how_it_was_run(ha), _how_it_was_run(hb)
    parts = [f"{k} {fa.get(k)} vs {fb.get(k)}"
             for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    if parts:
        return " / ".join(parts[:3])
    return (os.path.basename(a).split("_")[0] + " vs "
            + os.path.basename(b).split("_")[0])


#: Keys that say what a run IS rather than how it was RUN. A fingerprint names
#: the pair without explaining anything about it, which is the one thing a label
#: must not do.
_NOT_A_SETTING = {"body_parts", "model_sha", "model_path", "identities",
                  "fingerprint", "model_id", "identity_method"}


def _how_it_was_run(block: dict) -> dict:
    """A tracker block flattened to the settings that could differ.

    ``params`` is expanded rather than compared whole: two blob runs differing
    by one threshold would otherwise be labelled with both dictionaries
    printed in full, which is not a label.
    """
    out = {}
    for key, value in (block or {}).items():
        if key in _NOT_A_SETTING:
            continue
        if key == "params" and isinstance(value, dict):
            for k, v in value.items():
                out[k] = v
        else:
            out[key] = value
    return out


def run(paths: Optional[List[str]] = None, *,
        parity: bool = False) -> List[Check]:
    checks = definition_checks()
    for p in paths or []:
        checks.extend(identity_checks(p))
        if parity:
            checks.extend(parity_in_bundle(p))
    return checks


def report(checks: List[Check], out=None) -> int:
    """Print the table and return the exit status.

    `out` is resolved HERE, not in the signature: a default of `sys.stdout`
    is bound once at import, so anything that replaces the stream afterwards,
    a redirect, a test's capture, was printed straight past.
    """
    out = out or sys.stdout
    width = max((len(c.name) for c in checks), default=10)
    failed = 0
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        if not c.ok:
            failed += 1
        line = f"{mark}  {c.name:<{width}}  expected {c.expected!r}  got {c.got!r}"
        if c.note:
            line += f"   ({c.note})"
        print(line, file=out)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed", file=out)
    return 1 if failed else 0


def expand(inputs: List[str]) -> List[str]:
    """Files, globs or folders → recordings. A cohort lives in a folder, and
    "validate my data" should not mean typing forty paths."""
    import glob as _glob
    import os

    out: List[str] = []
    seen = set()
    for item in inputs:
        if os.path.isdir(item):
            out.extend(sorted(_glob.glob(os.path.join(item, "**",
                                                      "*_video_data.txt"),
                                         recursive=True)))
        elif any(c in item for c in "*?["):
            out.extend(sorted(_glob.glob(item, recursive=True)))
        else:
            out.append(item)
    # Deduped: a folder and a glob over it both name the same recording, and
    # checking a file twice reports the same finding twice.
    keep = []
    for path in out:
        real = os.path.normcase(os.path.abspath(path))
        if real not in seen and os.path.exists(path):
            seen.add(real)
            keep.append(path)
    return keep


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    parity = "--parity" in args
    args = [a for a in args if a != "--parity"]
    return report(run(expand(args), parity=parity))


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
