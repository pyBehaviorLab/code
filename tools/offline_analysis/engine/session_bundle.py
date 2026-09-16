"""One recording's derived state: what is known, what is stale, what Run will do.

A recording is not in a "mode" for the user to pick. It either has a clock,
poses, a space and measures, or it is missing some of them, and the work to be
done is whatever is missing. That is what a bundle models, and
:meth:`Bundle.plan` is the single function the UI renders as its Plan column,
its plan strip, its dry run and its blocked-row reason.

Everything derived lives in ``<stem>.pbanalysis/`` beside the recording.
Nothing is ever written next to the raw video unless the user explicitly
publishes it, the old retrack defaulted to ``overwrite=True`` into the
recording folder.

The stage hashes are what make the cheap operations cheap: editing zones
invalidates ``space`` and ``measure`` and leaves ``track`` alone, so a
two-second re-zone never triggers a six-hour re-inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.offline_analysis.engine.clock import ClockModel, resolve_clock, probe_video
from tools.offline_analysis.engine.retrack2d import TrackSpec
from tools.offline_analysis.engine.space import SpaceModel, space_from_header

BUNDLE_SUFFIX = ".pbanalysis"

STAGE_INGEST = "ingest"
STAGE_TRACK = "track"
#: Write the detector-error corrections out as a pose stream, instead of only
#: applying them in memory while measuring. Makes the correction durable and
#: inspectable; you can look at it in Verify and hand it to another tool.
STAGE_CORRECT = "correct"
STAGE_SPACE = "space"
STAGE_MEASURE = "measure"
STAGE_ORDER = (STAGE_INGEST, STAGE_TRACK, STAGE_CORRECT, STAGE_SPACE,
               STAGE_MEASURE)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


# ── discovery ────────────────────────────────────────────────────────────────

def session_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in ("_video_data", "_tracking", "_data", "_retracked",
                   "_annotated"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def video_name_in_header(txt_path: str, max_lines: int = 200) -> str:
    """The video filename the writer recorded in the session's own header.

    Every format states it, in one of two shapes:

        #video_file  <name>.mp4                 v1 / v2 / v3
        0.000\\tinfo\\tvideo_file\\t<name>.mp4     arena recorder

    Worth reading before guessing from the filename: matching on the stem
    assumes the two names agree, and they do not in every project. The header
    is what the recorder actually wrote, so it is right even when the naming
    convention is one this tool has never seen.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                s = line.rstrip("\n")
                if s.startswith("#video_file"):
                    return s.split(None, 1)[1].strip() if " " in s else ""
                if "\tinfo\tvideo_file\t" in s:
                    return s.rsplit("\t", 1)[1].strip()
                # Data rows start once the header is done; stop rather than
                # scan a 100 MB session for a key that is not there.
                if s.startswith("#columns") or s.lstrip("#").startswith(
                        "frame_number"):
                    break
    except OSError:
        return ""
    return ""


def find_video_for_txt(txt_path: str) -> str:
    """The video paired with a session file.

    Looks beside the file, in a ``video/`` subdirectory, in the parent, and in
    a sibling ``video/``, the layouts this rig actually writes.
    """
    txt_dir = os.path.dirname(os.path.abspath(txt_path))
    stem = session_stem(txt_path)
    parent = os.path.dirname(txt_dir)
    candidates = [txt_dir, os.path.join(txt_dir, "video"),
                  os.path.join(parent, "video"), parent]

    # The header names the video outright; only guess when it does not.
    named = video_name_in_header(txt_path)
    if named:
        named = os.path.basename(named.replace("\\", "/"))
        for d in candidates:
            p = os.path.join(d, named)
            if os.path.isfile(p):
                return p
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for ext in VIDEO_EXTS:
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p):
                return p
    # Loose match, second pass only, and deliberately narrow. A plain
    # substring test pairs a recording called "O" with any file whose name
    # merely contains an o: a stray `_encoder_warmup_14976.mp4` in the temp
    # directory was adopted as a session's video, and its 64x64 size then
    # became the frame the zones were rasterised into, so every zone lookup
    # missed. The candidate must BEGIN with the stem, and a stem too short to
    # identify anything does not get a loose pass at all.
    if len(stem) < 4:
        return ""
    for d in candidates:
        if not os.path.isdir(d):
            continue
        try:
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(VIDEO_EXTS):
                    continue
                if fn.lower().startswith(stem.lower()):
                    return os.path.join(d, fn)
        except OSError:
            continue
    return ""


def find_txt_for_video(video_path: str) -> str:
    stem = session_stem(video_path)
    d = os.path.dirname(os.path.abspath(video_path))
    for cand in (os.path.join(d, stem + "_video_data.txt"),
                 os.path.join(os.path.dirname(d), stem + "_video_data.txt"),
                 os.path.join(os.path.dirname(d), "video_data",
                              stem + "_video_data.txt")):
        if os.path.isfile(cand):
            return cand
    return ""


#: The correction parameters, and only those. A change to a measure threshold
#: must not invalidate a written correction, and vice versa.
CORRECTION_KEYS = ("pcut", "swap_correct", "swap_cost_threshold", "head_part",
                   "swap_body_part", "max_speed_cm_s", "rolling_median",
                   "smooth", "euro_min_cutoff", "euro_beta")


def _correction_key(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: params.get(k) for k in CORRECTION_KEYS}


def _stat_key(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return ""


def _hash(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str,
                      separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# ── steps ────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    """One unit of work Run would perform, with why and roughly how long."""
    stage: str
    why: str = ""
    est_s: float = 0.0
    blocked: bool = False

    @property
    def label(self) -> str:
        if self.blocked:
            return f"BLOCKED · {self.why}"
        return {STAGE_INGEST: "Read", STAGE_TRACK: "Track",
                STAGE_CORRECT: "Correct", STAGE_SPACE: "Re-zone",
                STAGE_MEASURE: "Analyze"}.get(self.stage, self.stage)

    @staticmethod
    def block(why: str) -> "Step":
        return Step(stage="", why=why, blocked=True)


def plan_label(steps: Sequence[Step]) -> str:
    """The Plan cell: what Run will do to this recording, and what it costs."""
    if not steps:
        return "up to date"
    blocked = [s for s in steps if s.blocked]
    if blocked:
        return f"BLOCKED · {blocked[0].why}"
    names = " → ".join(s.label for s in steps)
    total = sum(s.est_s for s in steps)
    return f"{names} · {_duration(total)}"


def _duration(seconds: float) -> str:
    if seconds < 1:
        return "<1 s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"~{seconds / 60:.0f} min"
    return f"~{seconds / 3600:.1f} h"


@dataclass
class StageState:
    hash: str = ""
    done_at: str = ""
    artifact: str = ""
    note: str = ""


# ── the bundle ───────────────────────────────────────────────────────────────

@dataclass
class Bundle:
    """Everything known about one recording, and everything derived from it."""

    txt_path: str = ""
    video_path: str = ""
    stem: str = ""
    clock: ClockModel = field(default_factory=ClockModel)
    space: SpaceModel = field(default_factory=SpaceModel)
    track: Optional[TrackSpec] = None
    stages: Dict[str, StageState] = field(default_factory=dict)
    user_fps: float = 0.0
    n_frames: int = 0
    #: Set when the user edited zones/scale but has not yet run.
    space_edited: bool = False
    #: Write the corrections out as their own pose stream (the "Correct"
    #: intent). Off means they are still applied, just in memory at measure
    #: time, which is what every analysis has always done.
    write_corrected: bool = False
    header: Any = None
    warnings: List[str] = field(default_factory=list)
    #: Cached answers, None until we have actually looked at the rows.
    _poses_in_rows: Optional[bool] = None
    _scanned_parts: Optional[List[str]] = None
    _zoned_in_rows: Optional[bool] = None

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def open(cls, txt_path: str = "", video_path: str = "", *,
             resolve: bool = True) -> "Bundle":
        """Load (or create) the bundle for a recording.

        Either a session file or a bare video is enough; the other is looked
        for beside it.
        """
        txt_path = os.path.abspath(txt_path) if txt_path else ""
        video_path = os.path.abspath(video_path) if video_path else ""
        if txt_path and not video_path:
            video_path = find_video_for_txt(txt_path)
        if video_path and not txt_path:
            txt_path = find_txt_for_video(video_path)
        b = cls(txt_path=txt_path, video_path=video_path,
                stem=session_stem(txt_path or video_path))
        b._load_state()
        if resolve:
            b.resolve()
        return b

    @property
    def dir(self) -> str:
        base = os.path.dirname(self.txt_path or self.video_path or ".")
        return os.path.join(base, self.stem + BUNDLE_SUFFIX)

    @property
    def state_path(self) -> str:
        return os.path.join(self.dir, "session.json")

    @property
    def has_video(self) -> bool:
        return bool(self.video_path) and os.path.exists(self.video_path)

    @property
    def has_source_poses(self) -> bool:
        """Whether poses already exist for this recording.

        The header's ``body_parts`` line is the fast answer, but files written
        before it existed carry poses without declaring them, so the rows
        themselves are the authority, and a recording is only "no poses" once
        we have actually looked.
        """
        if self.stages.get(STAGE_TRACK, StageState()).artifact:
            return True
        if self.header is not None and self.header.body_parts:
            return True
        if self._poses_in_rows is None:
            self._poses_in_rows = self._scan_for_poses()
        return self._poses_in_rows

    def _scan_for_poses(self, limit: int = 200) -> bool:
        return bool(self.scanned_body_parts(limit))

    def scanned_body_parts(self, limit: int = 200) -> List[str]:
        """Body-part names read from the rows themselves.

        Files written before the ``body_parts`` header line carry poses
        without declaring them. Reporting "no pose" for those, while the plan,
        which looks at the rows, correctly schedules an analysis, is the tab
        contradicting itself in front of the user.
        """
        if self._scanned_parts is not None:
            return self._scanned_parts
        parts: List[str] = []
        if self.txt_path and os.path.exists(self.txt_path):
            from tools.offline_analysis import video_data_schema as vds

            # Asked of the schema: an older dialect names this column
            # differently, and reading one name only reported those sessions
            # as having no poses, then demanded a tracker they do not need.
            col = vds.pose_column(self.header) if self.header else "pose_array"
            for i, row in enumerate(vds.iter_rows(self.txt_path, self.header)):
                if i >= limit:
                    break
                pose = vds.parse_pose(row.get(col))
                if pose:
                    parts = list(pose.keys())
                    break
        self._scanned_parts = parts
        return parts

    def has_recorded_zoning(self, limit: int = 400) -> bool:
        """Whether the file already says which zone the animal was in.

        The rig writes a `location` column, computed live from these very
        zones against the camera frame it could see at the time. When it is
        there, re-zoning is not "applying" anything; it is recomputing an
        answer the recording already has, from geometry we can only partly
        reconstruct. So it is not work, and it must not be planned as work.
        """
        if self._zoned_in_rows is not None:
            return self._zoned_in_rows
        found = False
        if self.txt_path and os.path.exists(self.txt_path):
            from tools.offline_analysis import video_data_schema as vds

            try:
                for i, row in enumerate(vds.iter_rows(self.txt_path,
                                                      self.header)):
                    if i >= limit:
                        break
                    z = str(row.get("location") or "").strip()
                    if z and z not in ("na", "none", "None", "-"):
                        found = True
                        break
            except Exception:                      # unreadable → assume not
                found = False
        self._zoned_in_rows = found
        return found

    def body_parts(self) -> List[str]:
        """What this recording's poses are made of, header first, rows as the
        authority when the header is silent."""
        declared = list(self.header.body_parts) if self.header is not None else []
        return declared or self.scanned_body_parts()

    # ── resolution ───────────────────────────────────────────────────

    def resolve(self) -> "Bundle":
        """Read the header, the clock and the space. Cheap; no decoding
        beyond the container probe."""
        from tools.offline_analysis import video_data_schema as vds

        if self.txt_path and os.path.exists(self.txt_path):
            self.header = vds.VideoDataHeader.parse(self.txt_path)
        probe = probe_video(self.video_path) if self.has_video else None
        if probe is not None and probe.ok:
            self.n_frames = probe.n_frames

        self.clock = resolve_clock(
            video_path=self.video_path, txt_path=self.txt_path,
            user_fps=self.user_fps or None, probe=probe,
            allow_pts_walk=False)          # the walk is a Run-time cost, not a listing one
        self.warnings = list(self.clock.warnings)

        if not self.space_edited:
            # The POSE space, when the recording declares one. Zones are
            # scored against pose coordinates, so the frame they are
            # rasterised into has to be the frame the poses live in, which
            # is not the video's size on a rig that encodes smaller than it
            # tracks. Passing the probe's size here overrode the header and
            # undid exactly that.
            declared = (self.header.pose_resolution
                        if self.header is not None else (0, 0))
            if declared and declared[0]:
                fs = declared
            else:
                fs = (probe.frame_size if probe and probe.ok
                      else (self.header.resolution if self.header else (0, 0)))
            if self.header is not None:
                self.space = space_from_header(self.header, fs)
            else:
                self.space = SpaceModel(frame_size=fs, origin="none")
            saved = self._saved_space()
            if saved is not None:
                self.space = saved
        return self

    def _saved_space(self) -> Optional[SpaceModel]:
        p = os.path.join(self.dir, "space.json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return SpaceModel.from_dict(json.load(f))
        except (OSError, ValueError):
            return None

    # ── hashes ───────────────────────────────────────────────────────

    def stage_hash(self, stage: str, measure_params: Optional[dict] = None) -> str:
        ingest = _hash("ingest", _stat_key(self.video_path),
                       _stat_key(self.txt_path), self.clock.source,
                       round(self.clock.fps, 4))
        if stage == STAGE_INGEST:
            return ingest
        track = _hash("track", ingest,
                      self.track.fingerprint() if self.track else "source")
        if stage == STAGE_TRACK:
            return track
        correct = _hash("correct", track, _correction_key(measure_params or {}))
        if stage == STAGE_CORRECT:
            return correct
        space = _hash("space", self.space.to_dict())
        if stage == STAGE_SPACE:
            return space
        return _hash("measure", correct, space, measure_params or {})

    def stale(self, stage: str, measure_params: Optional[dict] = None) -> bool:
        st = self.stages.get(stage)
        if st is None or not st.done_at:
            return True
        return st.hash != self.stage_hash(stage, measure_params)

    def mark_done(self, stage: str, *, artifact: str = "", note: str = "",
                  measure_params: Optional[dict] = None) -> None:
        self.stages[stage] = StageState(
            hash=self.stage_hash(stage, measure_params),
            done_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            artifact=artifact, note=note)
        self.save()

    def invalidate(self, stage: str) -> None:
        self.stages.pop(stage, None)
        self.save()

    # ── the plan ─────────────────────────────────────────────────────

    def plan(self, measure_params: Optional[dict] = None) -> List[Step]:
        """What Run would do, why, and roughly how long. Pure, no side effects.

        This is the whole honesty surface of the redesigned tab: the user sees
        it before pressing anything, so "it said it applied my zones" cannot
        happen again.
        """
        steps: List[Step] = []

        if not self.txt_path and not self.has_video:
            return [Step.block("neither a data file nor a video")]

        if not self.clock.usable:
            if self.has_video:
                return [Step.block("frame rate unknown, set it on this row")]
            return [Step.block("no timestamps and no video")]

        if self.stale(STAGE_INGEST):
            steps.append(Step(STAGE_INGEST, "clock not yet resolved", 1.0))

        needs_track = not self.has_source_poses or (
            self.track is not None and self.stale(STAGE_TRACK))
        if needs_track and self.track is None:
            # No poses and nothing chosen to make them, which is not always a
            # dead end.
            #
            # A session run without tracking still carries the `location` the
            # rig wrote per frame, and zone time, entries, latency and
            # transitions are all computable from it. Distance and speed are
            # not, those need coordinates, but refusing the whole recording
            # because half the measures are unavailable threw away the half
            # that was there.
            if self.has_recorded_zoning():
                needs_track = False
            elif not self.has_video:
                return [Step.block("no poses, and no video to track")]
            else:
                return [Step.block(
                    "no poses yet, choose tracking (the blob tracker needs "
                    "no model)")]
        if needs_track:
            errs = self.track.validate()
            if errs:
                return [Step.block(errs[0])]
            if not self.has_video:
                return [Step.block("tracking needs the video, which is missing")]
            steps.append(Step(STAGE_TRACK,
                              "no poses" if not self.has_source_poses
                              else "tracker changed",
                              self.estimate_track_s()))

        # Writing the corrections out is opt-in; they are applied either way.
        if self.write_corrected and self.stale(STAGE_CORRECT, measure_params):
            steps.append(Step(STAGE_CORRECT,
                              "corrections not yet written", 2.0))

        # A recording that already carries its own `location` column has
        # nothing to apply: the zones in its header are the ones the rig used
        # to write that column. Planning a re-zone for it announced work that
        # was not needed AND silently replaced a correct answer with one
        # recomputed against a frame size the poses are not in.
        if (self.space.zone_names
                and (self.space_edited
                     or (self.stale(STAGE_SPACE)
                         and not self.has_recorded_zoning()))):
            steps.append(Step(STAGE_SPACE,
                              "zones edited" if self.space_edited
                              else "zones not yet applied", 2.0))

        if steps or self.stale(STAGE_MEASURE, measure_params):
            steps.append(Step(STAGE_MEASURE, "metrics out of date", 1.0))
        return steps

    def estimate_track_s(self) -> float:
        """How long tracking would take. Uses the rate the last run actually
        achieved when there is one, an estimate that learns beats a constant."""
        n = max(self.n_frames, 1)
        note = self.stages.get(STAGE_TRACK, StageState()).note
        fps = 0.0
        try:
            fps = float(json.loads(note or "{}").get("fps", 0))
        except (TypeError, ValueError):
            fps = 0.0
        if fps <= 0:
            fps = 45.0 if (self.track and self.track.backend == "blob") else 12.0
        return n / max(fps, 1e-6) / max(1, (self.track.skip if self.track else 1))

    # ── artifacts ────────────────────────────────────────────────────

    def pose_path(self, spec: Optional[TrackSpec] = None) -> str:
        spec = spec or self.track
        key = spec.fingerprint() if spec else "source"
        return os.path.join(self.dir, "poses", f"{key}_video_data.txt")

    def corrected_path(self) -> str:
        key = self.track.fingerprint() if self.track else "source"
        return os.path.join(self.dir, "poses", f"{key}_corrected_video_data.txt")

    def uncorrected_txt(self) -> str:
        """What the CORRECT stage reads: the retracked poses if there are
        any, otherwise the recording's own, never its own output, which
        would correct an already-corrected stream a second time."""
        art = self.stages.get(STAGE_TRACK, StageState()).artifact
        if art and os.path.exists(art):
            return art
        return self.txt_path

    def active_txt(self) -> str:
        """The file the measure stage should read.

        Newest first: a written correction supersedes a retrack, which
        supersedes the recording's own poses.
        """
        for stage in (STAGE_CORRECT, STAGE_TRACK):
            art = self.stages.get(stage, StageState()).artifact
            if art and os.path.exists(art):
                return art
        return self.txt_path

    def measure_dir(self) -> str:
        return os.path.join(self.dir, "measure")

    def publish(self, dest_dir: str = "", suffix: str = "_retracked") -> str:
        """Copy the active poses next to the original, explicitly, never as a
        side effect of running."""
        src = self.active_txt()
        if not src or src == self.txt_path:
            raise ValueError("nothing retracked to publish")
        dest_dir = dest_dir or os.path.dirname(self.txt_path or self.video_path)
        out = os.path.join(dest_dir, f"{self.stem}{suffix}_video_data.txt")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src, out)
        return out

    # ── persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        state = {
            "stem": self.stem,
            "source": {"video": self.video_path, "txt": self.txt_path},
            "clock": {"source": self.clock.source, "fps": self.clock.fps,
                      "vfr": self.clock.vfr, "n_frames": self.clock.n_frames},
            "user_fps": self.user_fps,
            "write_corrected": self.write_corrected,
            "track": (asdict(self.track) if self.track else None),
            "stages": {k: asdict(v) for k, v in self.stages.items()},
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        with open(os.path.join(self.dir, "space.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.space.to_dict(), f, indent=2)

    def _load_state(self) -> None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError):
            return
        self.user_fps = float(state.get("user_fps") or 0.0)
        self.write_corrected = bool(state.get("write_corrected", False))
        tr = state.get("track")
        if tr:
            tr.pop("__dict__", None)
            try:
                self.track = TrackSpec(**tr)
            except TypeError:
                self.track = None
        for name, sd in (state.get("stages") or {}).items():
            self.stages[name] = StageState(**sd)

    def set_space(self, space: SpaceModel) -> None:
        """Adopt an edited space. Invalidates the two cheap stages and, by
        construction, not the expensive one."""
        space.origin = "edited"
        self.space = space
        self.space_edited = True
        self.invalidate(STAGE_SPACE)
        self.invalidate(STAGE_MEASURE)

    def set_interaction(self, cfg: Dict[str, Any],
                        object_zones: Sequence[str]) -> None:
        """Which zones are objects, and how facing them is judged.

        A zone is only scored for investigation when it is MARKED as one,
        `is_interaction_zone` looks for the flag or a name marker, and nothing
        in the app set either, so the whole facing analysis could never run on
        a rig whose zones are called Object1 rather than Object1_IA.

        Only the two cheap stages are invalidated: this changes how occupancy
        is scored, never the poses.
        """
        from tools.offline_analysis.engine import offline_analysis as oa

        want = {str(z) for z in (object_zones or ())}
        # The caller names zones the way the ANALYSIS does, which is not always
        # the way the file does: a zone colliding with a body point is renamed
        # when the file is parsed (`Center` → `Center_arm`). Matching only the
        # recorded name meant marking that zone as an object set no flag at
        # all, and its investigation columns simply never appeared.
        analysed = oa.zone_rename_map(
            [str(z.get("name") or "") for z in self.space.zones],
            self.body_parts())
        for z in self.space.zones:
            name = str(z.get("name") or "")
            if name in want or analysed.get(name) in want:
                z["interaction"] = True
            else:
                z.pop("interaction", None)
        self.space.interaction = dict(cfg or {})
        self.invalidate(STAGE_SPACE)
        self.invalidate(STAGE_MEASURE)
        self.save()

    def set_px_per_cm(self, px_per_cm: float) -> None:
        """Calibrate without redrawing anything.

        A scale line on the frame is one way to say how big the arena is; a
        number the user already has is another. Both land here, and neither
        touches the zones or the poses, only the two cheap stages that turn
        pixels into centimetres.
        """
        self.space.px_per_cm = float(px_per_cm or 0)
        self.space_edited = True
        self.invalidate(STAGE_SPACE)
        self.invalidate(STAGE_MEASURE)
        self.save()

    @property
    def has_poses(self) -> bool:
        """Whether anything has been tracked in this recording yet.

        Reads the parts actually present, not the header's claim: files
        written before the `body_parts` line have poses without declaring
        them.
        """
        return bool(self.body_parts())

    def set_track(self, spec: Optional[TrackSpec]) -> None:
        self.track = spec
        self.invalidate(STAGE_TRACK)
        self.invalidate(STAGE_CORRECT)
        self.invalidate(STAGE_MEASURE)
        self.save()

    def set_write_corrected(self, on: bool) -> None:
        """Turn the Correct stage on or off. Switching it OFF also drops the
        artifact, so `active_txt` goes back to the un-written poses rather
        than silently keeping a corrected file nobody asked for."""
        if bool(on) == self.write_corrected:
            return
        self.write_corrected = bool(on)
        self.invalidate(STAGE_CORRECT)
        self.invalidate(STAGE_MEASURE)
        self.save()

    def set_user_fps(self, fps: float) -> None:
        self.user_fps = float(fps or 0)
        for stage in STAGE_ORDER:
            self.invalidate(stage)
        self.resolve()

    # ── readiness, for the table ─────────────────────────────────────

    def readiness(self) -> Dict[str, str]:
        """One dict of short cells, every one a fact about the file, not a
        setting the user has to guess at."""
        parts = self.body_parts()
        backend = ((self.header.tracker or {}).get("backend", "")
                   if self.header is not None else "")
        if self.stages.get(STAGE_TRACK, StageState()).artifact and self.track:
            pose = f"{self.track.backend} · {len(parts)} bp" if parts else self.track.backend
        elif parts:
            pose = f"{backend or 'pose'} · {len(parts)} bp"
        else:
            pose = ", "
        space = ", "
        if self.space.zone_names:
            space = f"{len(self.space.zone_names)} zones"
            space += (f" · {self.space.px_per_cm:.1f} px/cm"
                      if self.space.calibrated else " · px")
            if self.space_edited:
                space = "edited · " + space
        video = os.path.splitext(self.video_path)[1].lstrip(".") or ", "
        return {
            "recording": self.stem,
            "video": video,
            "clock": self.clock.label,
            "pose": pose,
            "space": space,
            "plan": plan_label(self.plan()),
        }


# ── batch ────────────────────────────────────────────────────────────────────

def summarise(bundles: Sequence[Bundle],
              measure_params: Optional[dict] = None) -> Dict[str, Any]:
    """The plan strip: what a Run over this selection would cost, in total."""
    per_stage: Dict[str, List[float]] = {}
    blocked: List[Tuple[str, str]] = []
    for b in bundles:
        steps = b.plan(measure_params)
        for s in steps:
            if s.blocked:
                blocked.append((b.stem, s.why))
            else:
                per_stage.setdefault(s.stage, []).append(s.est_s)
    lines = []
    for stage in STAGE_ORDER:
        ests = per_stage.get(stage)
        if not ests:
            continue
        lines.append({"stage": stage, "n": len(ests),
                      "est_s": sum(ests), "label": _duration(sum(ests))})
    return {"steps": lines, "blocked": blocked,
            "total_s": sum(sum(v) for v in per_stage.values()),
            "n": len(bundles)}
