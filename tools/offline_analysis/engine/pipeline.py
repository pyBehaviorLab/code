"""Execute a plan. The one place work actually happens.

:meth:`~source.analysis.session_bundle.Bundle.plan` says what will be done;
this runs exactly that, in the same order, and marks each stage done. The UI
and the CLI both call :func:`run_bundle`, so the sentence the user read in the
Plan column and the work performed cannot drift apart, which is the whole
reason the old tab could claim it had applied zones it had discarded.

Headless. No Qt.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from tools.offline_analysis.engine import offline_analysis as oa
from tools.offline_analysis.engine import retrack2d as rt
from tools.offline_analysis.engine.session_bundle import (Bundle, STAGE_CORRECT,
                                            STAGE_INGEST, STAGE_MEASURE,
                                            STAGE_SPACE, STAGE_TRACK)

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class RunResult:
    """What one recording's run produced, and what it refused to do."""
    stem: str = ""
    rows: List[dict] = field(default_factory=list)
    bins: List[dict] = field(default_factory=list)
    plots: List[str] = field(default_factory=list)
    #: Ordered zone pairs, long form, one row per From/To/Stage.
    transitions: List[dict] = field(default_factory=list)
    #: Whether the space stage ran because the user EDITED the zones, as
    #: opposed to because occupancy had simply never been computed. The stage
    #: clears the bundle's flag before measure runs, so it has to be captured
    #: here or the distinction is lost exactly where it matters.
    space_was_edited: bool = False
    #: The live matplotlib Figure for the Plots tab, when ``live_figures`` was
    #: asked for. Distinct from ``plots``, which are PNGs written to disk.
    #: It is built HERE rather than by the caller because only this stage has
    #: the session the numbers actually came from, with the user's edited
    #: zones and scale applied. Rebuilt outside, the picture and the workbook
    #: disagree about the same recording.
    figure: Any = None
    #: ``{kind: path}`` for the copies written into the run's output folder,
    #: when one was set. Empty means nothing was published, which is the
    #: normal state for a run that only measured.
    published: Dict[str, str] = field(default_factory=dict)
    #: The annotated clip, when one was asked for and could be written.
    video: str = ""
    performed: List[str] = field(default_factory=list)
    skipped: str = ""
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    pose_path: str = ""
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and not self.skipped


def run_bundle(bundle: Bundle, params: Optional[dict] = None, *,
               progress: Optional[Progress] = None,
               cancel: Optional[threading.Event] = None,
               dry_run: bool = False) -> RunResult:
    """Do what the plan says, in order.

    ``dry_run`` exercises every path with a null detector and writes into a
    scratch area, how a user validates a forty-session setup before
    committing six hours to it.
    """
    p = {**oa.DEFAULTS, **(params or {})}
    res = RunResult(stem=bundle.stem)
    t0 = time.monotonic()

    def say(msg: str) -> None:
        if progress is not None:
            progress(f"{bundle.stem}: {msg}")

    steps = bundle.plan(_measure_key(p))
    blocked = [s for s in steps if s.blocked]
    if blocked:
        res.skipped = blocked[0].why
        say(f"skipped, {res.skipped}")
        return res

    try:
        for step in steps:
            if cancel is not None and cancel.is_set():
                res.skipped = "cancelled"
                return res
            if step.stage == STAGE_INGEST:
                say(f"clock: {bundle.clock.label} ({bundle.clock.confidence})")
                res.warnings.extend(bundle.clock.warnings)
                if not dry_run:
                    bundle.mark_done(STAGE_INGEST)
                res.performed.append(STAGE_INGEST)
            elif step.stage == STAGE_TRACK:
                _run_track(bundle, res, say, cancel, dry_run)
            elif step.stage == STAGE_CORRECT:
                _run_correct(bundle, res, p, say, dry_run)
            elif step.stage == STAGE_SPACE:
                # The step already carries WHY. Announcing "edited zones" for
                # a first run, where the reason is simply that occupancy has
                # not been computed yet, tells the user they changed
                # something they did not.
                say(f"re-zoning ({step.why})")
                res.space_was_edited = bool(bundle.space_edited)
                if not dry_run:
                    bundle.space_edited = False
                    bundle.mark_done(STAGE_SPACE)
                res.performed.append(STAGE_SPACE)
            elif step.stage == STAGE_MEASURE:
                _run_measure(bundle, res, p, say, dry_run)
        # Up to date is not the same as "produces nothing". The rows only
        # existed as a SIDE EFFECT of the measure step, so a second Run over
        # an unchanged selection planned no work, returned no rows, and the
        # panel said "No results" about recordings it had just measured.
        # Recomputing costs about a second per file and cannot be stale;
        # dry_run keeps it from rewriting the outputs or re-stamping the
        # bundle, so "up to date" stays true.
        if not res.rows and not res.skipped and STAGE_MEASURE not in res.performed:
            _run_measure(bundle, res, p, say, dry_run=True)
            # Not work: the plan said there was nothing to do, and it was
            # right. `performed` is what the panel reports as having HAPPENED.
            if res.performed and res.performed[-1] == STAGE_MEASURE:
                res.performed.pop()
        if p.get("write_video") and not dry_run and not res.skipped:
            _run_video(bundle, res, p, say, cancel)
        # Last, and only when the run actually produced something: copy the
        # cache's artifacts out under names that name the recording. The
        # cache stays as it is; this is the take-away copy, not a move.
        #
        # Always, now, not only when a folder was chosen: without it the
        # results existed solely inside a hidden `.pbanalysis` directory under
        # a hash for a name. With no folder set they go to `retracked` beside
        # the recording.
        if not dry_run and not res.skipped:
            from tools.offline_analysis.engine import outputs

            res.published = outputs.publish(bundle, str(p.get("output_dir") or ""),
                                            plots=res.plots, video=res.video)
    except rt.Cancelled:
        res.skipped = "cancelled"
    except Exception as e:                                # pragma: no cover
        res.error = f"{type(e).__name__}: {e}"
    res.elapsed_s = time.monotonic() - t0
    return res


def _measure_key(p: dict) -> dict:
    """The subset of the parameters that actually changes the numbers."""
    return {k: p.get(k) for k in (
        "pcut", "body_part", "body_parts_multi", "swap_correct", "smooth",
        "rolling_median", "min_move_cm", "max_speed_cm_s", "min_dwell_s",
        "immobility_cm_s", "freeze_cm_s", "min_freeze_s", "bin_size_s",
        # All three ways of asking for bins change the columns, so all
        # three have to make an existing measurement stale.
        "bin_count", "bin_edges_s",
        "px_per_cm_override", "metrics", "zones_include", "split_stages",
        # Regions add columns, so a recording measured before one was defined
        # is genuinely out of date. Without this the plan says "up to date"
        # and the new columns never appear.
        "regions",
        # Same for the Y-maze roles: each ADDS or CHANGES columns, so a
        # recording measured before one was set is out of date. `heatmap_bins`
        # is deliberately absent, it changes a picture, not a number.
        "novel_arm", "familiar_arm", "start_arm",
        "interaction")}


def _seed_from_recording(bundle: Bundle, spec):
    """Where the animal was, per frame, according to the recording itself.

    Only built for crop-track, and only when the recording has poses: it is
    the offline asymmetry, not a general feature. The window then starts on
    the animal instead of sweeping a grid to find it, so a re-track of an old
    session is better than the live run that produced it.

    Keyed on the row's own ``frame_number``, not on its position in the file,
    ``track_video`` yields the VIDEO's frame index, and after a dropped frame
    the two are different numbers. Read row by row rather than through
    ``parse_txt`` because the parsed session does not carry frame numbers at
    all, and because this needs one pass and no arrays.

    With named animals each one is seeded separately, from ITS columns: the
    numbered suffix (`nose`, `nose#2`) is the recording's way of saying which
    animal a keypoint belongs to, and window one has to start on animal one or
    the identities are scrambled from the first frame.

    Returns ``None`` when there is nothing to seed with, which is also the
    live case, and the window then steers itself, exactly as it does live.
    """
    from tools.offline_analysis import video_data_schema as vds
    from tools.offline_analysis.engine.trackers import (MODE_CROP_TRACK,
                                                        confident_centroid)

    if str((spec.params or {}).get("input_mode") or "") != MODE_CROP_TRACK:
        return None
    names = list(getattr(spec, "identities", []) or [])
    path = bundle.txt_path
    if not path or not os.path.exists(path):
        return None
    centres = {}
    try:
        header = bundle.header or vds.VideoDataHeader.parse(path)
        col = vds.pose_column(header)
        for row in vds.iter_rows(path, header):
            pose = vds.parse_pose(row.get(col))
            if not pose:
                continue
            # Every keypoint counts here: this is a hint about where to look,
            # not a measurement, and the recording's own confidence gate has
            # already been applied to what it stored.
            centre = (_centres_by_identity(pose, names) if names
                      else confident_centroid(pose, conf_min=0.0, good_min=1))
            if not centre:
                continue
            fn = vds.parse_int(row.get("frame_number"))
            if fn is not None:
                centres[int(fn)] = centre
    except Exception:                                     # pragma: no cover
        return None
    return centres.get if centres else None


def _centres_by_identity(pose, names):
    """One centre per named animal, split out of the numbered columns.

    `nose` belongs to the first animal, `nose#2` to the second, the dialect
    this repo already writes. An animal with no keypoints on this frame simply
    gets no seed, and its window carries on steering itself.
    """
    from tools.offline_analysis.engine.trackers import confident_centroid

    groups = {}
    for part, value in (pose or {}).items():
        base, _, suffix = str(part).partition("#")
        try:
            index = int(suffix) - 1 if suffix else 0
        except ValueError:                                # pragma: no cover
            index = 0
        if 0 <= index < len(names):
            groups.setdefault(names[index], {})[base] = value
    out = {}
    for name, parts in groups.items():
        centre = confident_centroid(parts, conf_min=0.0, good_min=1)
        if centre is not None:
            out[name] = centre
    return out


def _run_track(bundle: Bundle, res: RunResult, say: Progress,
               cancel: Optional[threading.Event], dry_run: bool) -> None:
    spec = bundle.track
    say(f"tracking with {spec.backend}"
        + (f" ({os.path.basename(spec.model_path)})" if spec.model_path else ""))

    if dry_run:
        res.performed.append(STAGE_TRACK)
        say("dry run, detector not executed")
        return

    def on_progress(done: int, total: int, fps: float) -> None:
        if total:
            say(f"tracking {done}/{total} frames ({fps:.0f} fps)")
        else:
            say(f"tracking {done} frames ({fps:.0f} fps)")

    stream = rt.track_video(bundle.video_path, spec, bundle.clock,
                            progress=on_progress, cancel=cancel,
                            seed_fn=_seed_from_recording(bundle, spec),
                            n_frames=bundle.n_frames)
    res.warnings.extend(stream.warnings)
    if not len(stream):
        raise RuntimeError("the detector produced no poses")

    out = bundle.pose_path(spec)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stages = _stage_by_frame(bundle)
    rt.write_pose_stream(out, stream, source_header=bundle.header,
                         space=bundle.space, stage_by_frame=stages,
                         mcu_ts_by_frame=_mcu_ts_by_frame(bundle))
    res.pose_path = out
    bundle.mark_done(
        STAGE_TRACK, artifact=out,
        note=json.dumps({"fps": round(stream.fps_achieved, 2),
                         "detected": round(stream.detected_fraction, 4),
                         "partial": stream.partial}))
    if stream.partial:
        res.warnings.append(
            f"tracking stopped early at frame {len(stream)}, the poses that "
            "were produced are kept")
    say(f"tracked {len(stream)} frames, "
        f"{stream.detected_fraction * 100:.0f}% with a detection")
    res.performed.append(STAGE_TRACK)


def _run_correct(bundle: Bundle, res: RunResult, p: dict, say: Progress,
                 dry_run: bool) -> None:
    """Write the detector-error corrections out as their own pose stream.

    The corrections are applied to every analysis either way; this makes them
    durable: a real session file you can open in Verify, hand to another tool,
    or compare against the raw poses. Everything downstream then reads the
    corrected stream, because ``Bundle.active_txt`` prefers it.
    """
    import numpy as np

    from tools.offline_analysis.engine import retrack2d as _rt

    # NOT `active_txt`: that prefers the corrected file, so a re-run would
    # correct its own output a second time.
    path = bundle.uncorrected_txt()
    if not path or not os.path.exists(path):
        raise RuntimeError("nothing to correct")
    say("writing corrected poses")
    if dry_run:
        res.performed.append(STAGE_CORRECT)
        return

    sess = oa.parse_txt(path)
    fixed = oa.corrected_keypoints(sess, p)
    if not fixed:
        raise RuntimeError("this recording has no poses to correct")

    n = len(sess.ts_ms)
    parts = list(fixed)
    stream = _rt.PoseStream(
        frame_numbers=list(range(n)),
        ts_ms=np.asarray(sess.ts_ms, float),
        poses=[{bp: [float(fixed[bp]["x"][i]), float(fixed[bp]["y"][i]),
                     float(fixed[bp]["conf"][i])]
                for bp in parts
                if np.isfinite(fixed[bp]["x"][i])} or None
               for i in range(n)],
        body_parts=parts,
        n_frames_expected=n,
    )
    # The tracker record says what produced these poses: the same detector as before,
    # with a correction pass named on top. A reader must never mistake a
    # corrected stream for raw detector output.
    tracker = dict(getattr(bundle.header, "tracker", {}) or {})
    tracker.update({
        "corrected": True,
        "corrections": {k: p.get(k) for k in (
            "pcut", "swap_correct", "swap_cost_threshold", "head_part",
            "swap_body_part", "max_speed_cm_s", "rolling_median", "smooth",
            "euro_min_cutoff", "euro_beta")},
    })
    stream.tracker = tracker

    out = bundle.corrected_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # The source's own zone column travels with the poses. A correction moves
    # keypoints by a pixel or two; it does not move the animal into another
    # arm, and it is not grounds for re-deciding zone membership against a
    # frame size the poses may not even be in. Recomputing it here turned
    # 4,217 Home_arm frames into 119 on a real recording, and a written
    # correction supersedes the original for everything downstream.
    # Read verbatim from the FILE, not from the parsed session: the parser
    # renames a zone whose name collides with a body part ("Center" becomes
    # "Center_arm" beside a Center keypoint), and writing that back out would
    # leave rows speaking a vocabulary the file's own header does not.
    from tools.offline_analysis import video_data_schema as _vds
    src_locs = [(r.get("location") or "") for r in _vds.iter_rows(path)]
    _rt.write_pose_stream(out, stream, source_header=bundle.header,
                          space=bundle.space,
                          stage_by_frame=_stage_by_frame(bundle),
                          source_locations=src_locs)
    kept = sum(1 for pose in stream.poses if pose)
    say(f"corrected {kept}/{n} frames -> {os.path.basename(out)}")
    bundle.mark_done(STAGE_CORRECT, artifact=out,
                     measure_params=_measure_key(p))
    res.performed.append(STAGE_CORRECT)


def _stage_by_frame(bundle: Bundle) -> Dict[int, str]:
    """The protocol stage each source frame belonged to, so a retrack does not
    lose the phase structure the whole analysis is reported by.

    Both sources, always, whenever both are there: the recording's own
    ``stage`` column wins per frame because it is stamped against the frame and
    needs no alignment, and the MCU ``.tsv`` beside it fills every frame the
    column does not answer for. See ``mcu_states.merge_by_frame``, the merge
    and the cross-check live there, so there is one answer to "what state was
    this frame in" rather than one per caller.
    """
    if not bundle.txt_path or not os.path.exists(bundle.txt_path):
        return {}
    from tools.offline_analysis.engine import mcu_states

    return mcu_states.merge_by_frame(bundle.txt_path,
                                     video_path=bundle.video_path,
                                     stem=bundle.stem, header=bundle.header)


def _run_video(bundle: Bundle, res: RunResult, p: dict, say: Progress,
               cancel) -> None:
    """Write the annotated clip for this recording.

    Opt-in, because it costs roughly as long again as the retrack: every frame
    has to be decoded, drawn on and re-encoded, and most runs do not want one.

    Never fatal. A missing encoder or an unreadable video means no clip, and a
    clip is not what the run was for, the numbers are already computed by the
    time this runs, and losing them to a codec problem would be absurd.
    """
    if not bundle.video_path or not os.path.exists(bundle.video_path):
        res.warnings.append("no video for this recording, so no annotated clip")
        return
    poses = bundle.active_txt()
    if not poses or not os.path.exists(poses):
        res.warnings.append("nothing tracked, so no annotated clip")
        return
    from tools.offline_analysis.engine import annotate

    out = os.path.join(bundle.measure_dir(), f"{bundle.stem}_annotated.mp4")
    say("writing the annotated video")

    def on_progress(done: int, total: int) -> None:
        say(f"annotating {done}/{total} frames" if total
            else f"annotating {done} frames")

    try:
        res.video = annotate.write_annotated_video(
            bundle.video_path, poses, out,
            zones=(bundle.space.zones if bundle.space else None),
            fps=bundle.clock.fps if bundle.clock else 0.0,
            states=_stage_by_frame(bundle),
            progress=on_progress, cancel=cancel)
    except Exception as e:
        res.warnings.append(f"the annotated video could not be written: {e}")
        logger.warning("annotated video for %s: %s", bundle.stem, e)


def _mcu_ts_by_frame(bundle: Bundle) -> Dict[int, str]:
    """The recording's own per-frame MCU clock, to carry into the retrack."""
    if not bundle.txt_path or not os.path.exists(bundle.txt_path):
        return {}
    from tools.offline_analysis.engine import mcu_states

    return mcu_states.ts_by_frame(bundle.txt_path, bundle.header)


def _run_measure(bundle: Bundle, res: RunResult, p: dict, say: Progress,
                 dry_run: bool) -> None:
    path = bundle.active_txt()
    if not path or not os.path.exists(path):
        raise RuntimeError("nothing to analyse")
    say("computing metrics")

    sess = oa.parse_txt(path)
    # The bundle's space wins: it is what the user edited and what the Plan
    # column promised. Falling back to the file's own zones here is how an
    # edit gets silently discarded.
    params = dict(p)
    if bundle.space.zones:
        params["space"] = bundle.space
        if bundle.space.calibrated and not params.get("px_per_cm_override"):
            params["px_per_cm_override"] = bundle.space.px_per_cm
        if _should_rezone(bundle, sess, res):
            sess = _apply_space(sess, bundle.space, params)

    tr = oa.build_track(sess, params)
    res.rows = oa.session_rows(sess, params)
    if oa._want(params, "transitions", True):
        res.transitions = oa.transition_rows(sess, params)
    if oa.wants_bins(params):
        b = oa.time_bins(sess, tr, params)
        if b:
            res.bins = [b]
    if params.get("plots") and not dry_run:
        out_dir = bundle.measure_dir()
        os.makedirs(out_dir, exist_ok=True)
        res.plots = oa.make_plots(sess, tr, out_dir, params)
    # Not gated on `dry_run`: a figure writes nothing, and the commonest way
    # to reach this function is the "already up to date" re-measure below,
    # which passes dry_run=True. Gating it there is how Run over an unchanged
    # selection would show rows and an empty Plots tab.
    if params.get("live_figures"):
        res.figure = oa.build_figure(sess, tr, params)
    if not dry_run:
        bundle.mark_done(STAGE_MEASURE, measure_params=_measure_key(p))
    res.performed.append(STAGE_MEASURE)


def _should_rezone(bundle: Bundle, sess, res: RunResult) -> bool:
    """Whether to recompute `location`, or trust what was recorded.

    Re-zoning with the recording's OWN zones ought to be a no-op. It is not,
    when the zones are normalized and the recording declares the video's
    resolution while the poses are in the camera's, the zones then land in
    the wrong place and every zone number changes. That is silent corruption
    of a recording nobody asked to change, so the recorded assignment wins
    unless the user actually edited the zones.

    An edit still re-zones, because then the user IS asking for new geometry
    and the old column no longer describes it, but the mismatch is reported,
    because those numbers cannot be trusted either until the pose space is
    known.
    """
    from tools.offline_analysis.engine import space as _sp

    part = params_body_part(sess)
    fits = True
    if part is not None:
        fits = _sp.poses_fit_frame((sess.kp[part]["x"], sess.kp[part]["y"]),
                                   bundle.space.frame_size)
    if not fits:
        res.warnings.append(
            f"poses fall outside the declared {bundle.space.frame_size[0]}"
            f"x{bundle.space.frame_size[1]} frame; they are in a different "
            f"coordinate space from the zones, so zone measures from re-drawn "
            f"zones will be wrong until the recording declares its pose "
            f"resolution")
    if bundle.space_edited or res.space_was_edited:
        return True
    has_location = any(
        z and z not in ("na", "none", "None") for z in (sess.location or ()))
    if has_location and not fits:
        # Nothing was edited and the geometry does not line up: keep the
        # assignment the rig made when it could still see the camera frame.
        res.warnings.append(
            "kept the recording's own zone assignment rather than "
            "recomputing it against zones in a different coordinate space")
        return False
    return True


def params_body_part(sess):
    """The keypoint zone assignment would use, or None."""
    if not getattr(sess, "kp", None):
        return None
    return "Center" if "Center" in sess.kp else next(iter(sess.kp))


def _apply_space(sess, space, params):
    """Recompute ``location`` from the bundle's zones before measuring.

    This is re-zoning: the poses are untouched, only which zone each frame
    falls in is recomputed. Costs one polygon lookup per frame.
    """
    from tools.offline_analysis.engine import space as _sp

    part = params.get("body_part") or ""
    ch = sess.kp.get(part) or (next(iter(sess.kp.values())) if sess.kp else None)
    if ch is None:
        return sess
    x = np.asarray(ch["x"], float)
    y = np.asarray(ch["y"], float)
    dt = oa.dt_seconds(sess.ts_ms)
    result = _sp.rezone(space, body_xy=(x, y), dt=dt,
                        min_dwell_s=params.get("min_dwell_s", 0.2))
    sess.location = result.location
    sess.zones = list(space.zones)
    if space.calibrated:
        sess.px_per_cm = space.px_per_cm
    # A zone the user just named "Center" would collide with the body point of
    # the same name and produce two sets of columns that clash. The parser runs
    # this on the FILE's zones; these are new ones, so it has to run again.
    oa._disambiguate_zones(sess)
    return sess


# ── batch ────────────────────────────────────────────────────────────────────

def _in_worker(job):
    """Run one recording in a worker process.

    Top-level, because a pool pickles the function it is given. The live
    figure is dropped rather than returned: a matplotlib Figure does not
    survive a process boundary, and the recordings due one never come here.

    The stage stamps go back with the result. The worker wrote them to disk on
    its own copy of the bundle, and the panel afterwards reads the parent's.
    """
    bundle, params, dry_run = job
    res = run_bundle(bundle, params, dry_run=dry_run)
    res.figure = None
    return bundle.stages, bundle.space_edited, res


def run_all(bundles: Sequence[Bundle], params: Optional[dict] = None, *,
            progress: Optional[Progress] = None,
            cancel: Optional[threading.Event] = None,
            dry_run: bool = False) -> Dict[str, Any]:
    """Run a selection and collect the results.

    One recording's failure is a row in the error list, never an aborted
    batch, a cohort of forty must not be lost to one corrupt file.
    """
    p = {**oa.DEFAULTS, **(params or {})}
    rows: List[dict] = []
    bins: List[dict] = []
    plots: List[str] = []
    transitions: List[dict] = []
    errors: List[str] = []
    skipped: List[dict] = []
    warnings: List[str] = []
    per: List[RunResult] = []

    # Live figures are capped across the whole selection, not per file: forty
    # open matplotlib figures is a slideshow nobody scrolls and a great deal
    # of memory. What the cap dropped is counted and returned, so the panel
    # can say so, a silently short list reads as "that is all there was".
    fig_cap = int(p.get("live_figures") or 0)
    n_figs = n_capped = 0

    # Recordings due a live figure run in this process, because the figure is
    # the one part of a result that cannot come back from a worker. Every
    # recording after them is independent plain data, which is what makes a
    # cohort worth spreading across the machine. A folder of forty used to
    # occupy one core while the rest of the machine sat idle.
    split = min(len(bundles), fig_cap)
    done: List[Optional[RunResult]] = [None] * len(bundles)

    for i in range(split):
        if cancel is not None and cancel.is_set():
            break
        b = bundles[i]
        if progress:
            progress(f"[{i + 1}/{len(bundles)}] {b.stem}")
        r = run_bundle(b, {**p, "live_figures": True}, progress=progress,
                       cancel=cancel, dry_run=dry_run)
        if r.figure is not None:
            n_figs += 1
        done[i] = r

    rest = ([] if (cancel is not None and cancel.is_set())
            else list(range(split, len(bundles))))
    n_jobs = int(p.get("n_jobs") or 0) or max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, len(rest))
    tail_p = {**p, "live_figures": False}

    if rest and n_jobs > 1:
        import multiprocessing as mp

        jobs = [(bundles[i], tail_p, dry_run) for i in rest]
        try:
            with mp.Pool(n_jobs) as pool:
                it = pool.imap(_in_worker, jobs)
                for k, (stages, edited, r) in enumerate(it):
                    # The worker mutated ITS copy; carry the stamps onto the
                    # bundle this process holds, which is the one the panel
                    # reads to decide what is up to date.
                    b = bundles[rest[k]]
                    b.stages, b.space_edited = stages, edited
                    done[rest[k]] = r
                    if progress:
                        progress(f"[{split + k + 1}/{len(bundles)}] {r.stem}")
                    if cancel is not None and cancel.is_set():
                        pool.terminate()
                        break
        except Exception as e:
            # A pool that cannot start is a reason to be slow, never a reason
            # to lose the run.
            logger.warning("parallel run unavailable (%s), running serially", e)
            warnings.append(f"ran on one core: {e}")
            rest = [i for i in rest if done[i] is None]
            n_jobs = 1
        else:
            rest = []

    for i in rest:
        if cancel is not None and cancel.is_set():
            break
        b = bundles[i]
        if progress:
            progress(f"[{i + 1}/{len(bundles)}] {b.stem}")
        done[i] = run_bundle(b, tail_p, progress=progress, cancel=cancel,
                             dry_run=dry_run)

    for i, r in enumerate(done):
        if r is None:
            continue
        if i >= split and r.figure is None and fig_cap and r.ok:
            # Only a recording that RAN counts as dropped by the cap. A
            # blocked one would have had no plot either way, and counting it
            # would report a shortfall the cap did not cause.
            n_capped += 1
        per.append(r)
        rows.extend(r.rows)
        bins.extend(r.bins)
        plots.extend(r.plots)
        transitions.extend(r.transitions)
        if r.error:
            errors.append(f"{r.stem}: {r.error}")
        if r.skipped:
            skipped.append({"File": r.stem, "reason": r.skipped})
        warnings.extend(f"{r.stem}: {w}" for w in r.warnings)

    rows.sort(key=lambda d: (str(d.get("Subject", "")), str(d.get("ExptDate", "")),
                             float(d.get("Stage_start_s", 0) or 0)))
    transitions.sort(key=lambda d: (d.get("File", ""), d.get("Stage", ""),
                                    d.get("From", ""), d.get("To", "")))
    out = {"summary": rows, "bins": bins, "plots": plots, "errors": errors,
           "excluded": skipped, "warnings": warnings, "params": p,
           "results": per, "n_files": len(bundles),
           "transitions": transitions,
           "figures": [r.figure for r in per if r.figure is not None],
           "figures_capped": n_capped,
           "columns": oa.order_row_columns(rows)}
    # The cohort workbook, written where the rest of the run's results went.
    # Every session stacked into one sheet already existed, behind an Export
    # click. A run that has been given a folder should not need one.
    if p.get("output_dir") and rows and not dry_run:
        out["workbook"] = _write_cohort_workbook(out, str(p["output_dir"]))
    return out


def _write_cohort_workbook(res: Dict[str, Any], out_dir: str) -> str:
    """Every recording in this run, in one workbook. Never fatal."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "analysis.xlsx")
        oa.write_workbook(res, path)
        logger.info("cohort workbook: %d rows to %s", len(res["summary"]), path)
        return path
    except Exception as e:
        # The numbers are already computed and already returned; losing them
        # to a spreadsheet library would be absurd.
        res.setdefault("warnings", []).append(
            f"the cohort workbook could not be written: {e}")
        logger.warning("cohort workbook: %s", e)
        return ""
