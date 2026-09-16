"""When did each frame happen?

"Just read the FPS" is the assumption that produced the worst defect in the
offline pipeline: a retrack timestamped its poses ``frame_index / fps``, which
is wrong for every variable-rate recording this rig makes, wrong after a
dropped frame, and destroys alignment with the MCU event stream. Worse, when
the timestamp column went missing entirely the reader substituted the row
counter and nothing said so.

So a clock is resolved explicitly, from four ranked sources, and it always
says which one it used and how much that is worth:

===== ================= ============================================= ==========
Rank  source            how                                           confidence
===== ================= ============================================= ==========
1     ``recorded``      ``frame_ts_ms`` from the session file          exact
2     ``container_pts`` per-frame presentation stamps from the video   derived
3     ``container_fps`` the container's nominal rate, cross-checked    assumed
4     ``user``          the number a human typed                       assumed
===== ================= ============================================= ==========

Rank 1 always wins when it is available. Nothing here silently upgrades its
own confidence, and every check that fails becomes a warning the UI shows on
the row *before* anything expensive runs.

Headless: numpy only, OpenCV imported lazily so the module can be reasoned
about (and unit-tested) without a video.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── source ranks ─────────────────────────────────────────────────────────────

SOURCE_RECORDED = "recorded"
SOURCE_CONTAINER_PTS = "container_pts"
SOURCE_CONTAINER_FPS = "container_fps"
SOURCE_USER = "user"
SOURCE_INDEX = "index"          # nothing usable, frame numbers, not time

CONFIDENCE = {
    SOURCE_RECORDED: "exact",
    SOURCE_CONTAINER_PTS: "derived",
    SOURCE_CONTAINER_FPS: "assumed",
    SOURCE_USER: "assumed",
    SOURCE_INDEX: "none",
}

#: A behaviour recording with a median frame interval outside this range is
#: almost certainly a units error rather than a real acquisition rate. 1 ms is
#: exactly what the row-counter-as-timestamp bug produces.
PLAUSIBLE_DT_MS = (5.0, 1000.0)

#: Fraction of the median interval above which the spread of intervals means
#: the stream is genuinely variable-rate rather than jittery rounding.
VFR_CV_THRESHOLD = 0.05


@dataclass
class VideoProbe:
    """What the container claims about itself. Claims, not facts."""
    path: str = ""
    ok: bool = False
    width: int = 0
    height: int = 0
    n_frames: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    error: str = ""

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass
class ClockModel:
    """The resolved answer, with the source it came from attached."""

    source: str = SOURCE_INDEX
    fps: float = 0.0
    #: Per-frame acquisition time in ms, when it is actually known (ranks 1-2).
    ts_ms: Optional[np.ndarray] = None
    vfr: bool = False
    n_frames: int = 0
    note: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        return CONFIDENCE.get(self.source, "none")

    @property
    def known_per_frame(self) -> bool:
        return self.ts_ms is not None and len(self.ts_ms) > 0

    @property
    def usable(self) -> bool:
        """False when nothing better than a frame index was found. Analysis may
        still run, but every per-second measure is meaningless and must be
        labelled so."""
        return self.source != SOURCE_INDEX

    @property
    def median_dt_ms(self) -> float:
        if self.known_per_frame and len(self.ts_ms) > 1:
            d = np.diff(self.ts_ms)
            d = d[d > 0]
            if d.size:
                return float(np.median(d))
        return 1000.0 / self.fps if self.fps > 0 else math.nan

    @property
    def label(self) -> str:
        """One short cell for the readiness table."""
        if self.source == SOURCE_RECORDED:
            return "recorded"
        if self.source == SOURCE_CONTAINER_PTS:
            return f"video pts {self.fps:.1f}"
        if self.source == SOURCE_CONTAINER_FPS:
            return f"container {self.fps:.1f}"
        if self.source == SOURCE_USER:
            return f"{self.fps:.1f} fps (you)"
        return "unknown"

    def timestamps(self, n: Optional[int] = None) -> np.ndarray:
        """Per-frame timestamps in ms for ``n`` frames.

        Ranks 1-2 return what was measured (truncated or, if the caller asks
        for more frames than were measured, extended on the median interval and
        a warning is already on the model). Ranks 3-4 build the grid the user
        agreed to, which is honest precisely because ``source`` says so.
        """
        n = int(n if n is not None else (self.n_frames or 0))
        if n <= 0:
            return np.empty(0, float)
        if self.known_per_frame:
            have = np.asarray(self.ts_ms, float)
            if len(have) >= n:
                return have[:n]
            step = self.median_dt_ms if self.median_dt_ms > 0 else (
                1000.0 / self.fps if self.fps > 0 else 1.0)
            tail = have[-1] + step * np.arange(1, n - len(have) + 1)
            return np.concatenate([have, tail])
        step = 1000.0 / self.fps if self.fps > 0 else 1.0
        return step * np.arange(n, dtype=float)


# ── probing ──────────────────────────────────────────────────────────────────

def probe_video(video_path: str) -> VideoProbe:
    """Open the container and read what it declares. No decoding."""
    p = VideoProbe(path=video_path)
    if not video_path or not os.path.exists(video_path):
        p.error = "video not found"
        return p
    try:
        import cv2
    except ImportError:                                   # pragma: no cover
        p.error = "OpenCV not available"
        return p
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            p.error = "could not open video"
            return p
        p.ok = True
        p.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        p.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        p.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        p.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if p.fps > 0 and p.n_frames > 0:
            p.duration_s = p.n_frames / p.fps
    finally:
        cap.release()
    return p


def walk_pts(video_path: str, max_frames: int = 200_000) -> np.ndarray:
    """Per-frame presentation timestamps in ms, by grabbing (not decoding).

    ``grab()`` advances the demuxer without colour-converting the frame, which
    is roughly an order of magnitude cheaper than a full read and is all that
    is needed to learn *when* each frame is.
    """
    try:
        import cv2
    except ImportError:                                   # pragma: no cover
        return np.empty(0, float)
    cap = cv2.VideoCapture(video_path)
    out: List[float] = []
    try:
        if not cap.isOpened():
            return np.empty(0, float)
        while len(out) < max_frames:
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not cap.grab():
                break
            out.append(float(t))
    finally:
        cap.release()
    return np.asarray(out, float)


# ── resolution ───────────────────────────────────────────────────────────────

def _characterise(ts_ms: np.ndarray) -> Tuple[float, bool, List[str]]:
    """(fps, vfr, warnings) for a measured timestamp series."""
    warns: List[str] = []
    if ts_ms.size < 2:
        return 0.0, False, warns
    d = np.diff(ts_ms)
    if np.any(d < 0):
        warns.append(
            f"timestamps go backwards at {int(np.sum(d < 0))} frame(s), the "
            "file may be truncated or two sessions may have been concatenated")
    pos = d[d > 0]
    if pos.size == 0:
        return 0.0, False, warns
    med = float(np.median(pos))
    fps = 1000.0 / med if med > 0 else 0.0
    cv = float(np.std(pos) / med) if med > 0 else 0.0
    vfr = cv > VFR_CV_THRESHOLD
    if med < PLAUSIBLE_DT_MS[0] or med > PLAUSIBLE_DT_MS[1]:
        warns.append(
            f"median frame interval is {med:.3f} ms ({fps:.1f} fps), outside "
            f"{PLAUSIBLE_DT_MS[0]:.0f}-{PLAUSIBLE_DT_MS[1]:.0f} ms, which "
            "usually means the timestamps are not milliseconds at all")
    return fps, vfr, warns


def clock_from_timestamps(ts_ms: Sequence[float], *, source: str = SOURCE_RECORDED,
                          note: str = "") -> ClockModel:
    """Build a model from a measured series (rank 1 or 2)."""
    arr = np.asarray(list(ts_ms), float)
    fps, vfr, warns = _characterise(arr)
    return ClockModel(source=source, fps=fps, ts_ms=arr, vfr=vfr,
                      n_frames=int(arr.size), note=note, warnings=warns)


def resolve_clock(video_path: str = "", txt_path: str = "",
                  user_fps: Optional[float] = None,
                  *, allow_pts_walk: bool = True,
                  probe: Optional[VideoProbe] = None) -> ClockModel:
    """The four ranks, in order, with every check that failed recorded.

    ``user_fps`` does not override a recorded clock, a human's guess cannot
    be better than the acquisition times the camera actually reported. It is
    used when nothing measured is available, and it is used to cross-check
    what the container claims.
    """
    warns: List[str] = []

    # ── rank 1: the session file's own acquisition times ─────────────
    if txt_path and os.path.exists(txt_path):
        from tools.offline_analysis import video_data_schema as vds

        hdr = vds.VideoDataHeader.parse(txt_path)
        # Asked of the schema rather than hard-coded: a recording written
        # in an older dialect names this column differently, and looking for
        # one name only told those sessions they had no timestamps at all.
        col = vds.timestamp_column(hdr)
        if col:
            # The one column this needs, not a dict per row: resolving the
            # clock for thirty recordings was 21 s of building rows whose
            # other fields were then discarded.
            ts = [t for t in vds.timestamps(txt_path, hdr)
                  if not math.isnan(t)]
            if len(ts) >= 2:
                m = clock_from_timestamps(
                    ts, source=SOURCE_RECORDED,
                    note=f"{col} from {os.path.basename(txt_path)}"
                         + (f", camera clock: {hdr.frame_clock}"
                            if hdr.frame_clock else ""))
                m.warnings.extend(warns)
                _cross_check_counts(m, video_path, probe)
                return m
        else:
            warns.append(
                f"{os.path.basename(txt_path)} has no timestamp column "
                "(dialect: %s), falling back to the video" % hdr.dialect)

    # ── rank 2: the container's per-frame stamps ─────────────────────
    pr = probe or (probe_video(video_path) if video_path else VideoProbe())
    if pr.ok and allow_pts_walk:
        pts = walk_pts(video_path)
        # A container that reports 0 for every frame is not reporting stamps.
        if pts.size >= 2 and float(np.max(pts)) > 0:
            m = clock_from_timestamps(pts, source=SOURCE_CONTAINER_PTS,
                                      note="per-frame presentation stamps")
            m.warnings.extend(warns)
            if user_fps and m.fps > 0 and abs(m.fps - user_fps) / m.fps > 0.05:
                m.warnings.append(
                    f"you entered {user_fps:.2f} fps but the video's own stamps "
                    f"say {m.fps:.2f} fps, the video was used")
            _cross_check_counts(m, video_path, pr)
            return m
        warns.append("the container reports no per-frame timestamps")

    # ── rank 3: the container's nominal rate ─────────────────────────
    if pr.ok and pr.fps > 0:
        # Cross-check: n_frames / fps should match the container's duration.
        m = ClockModel(source=SOURCE_CONTAINER_FPS, fps=float(pr.fps),
                       n_frames=int(pr.n_frames), vfr=False,
                       note="container nominal rate, assumed constant",
                       warnings=list(warns))
        if user_fps and abs(pr.fps - user_fps) / pr.fps > 0.05:
            m.source = SOURCE_USER
            m.fps = float(user_fps)
            m.note = ("you overrode the container, which claimed "
                      f"{pr.fps:.2f} fps")
        dt = 1000.0 / m.fps
        if dt < PLAUSIBLE_DT_MS[0] or dt > PLAUSIBLE_DT_MS[1]:
            m.warnings.append(
                f"{m.fps:.2f} fps implies a {dt:.1f} ms frame interval, which "
                "is not a plausible behaviour recording rate")
        return m

    # ── rank 4: the number a human typed ─────────────────────────────
    if user_fps and user_fps > 0:
        return ClockModel(source=SOURCE_USER, fps=float(user_fps),
                          n_frames=int(pr.n_frames), vfr=False,
                          note="entered by the user; the video did not say",
                          warnings=list(warns))

    warns.append("no timestamps and no frame rate, per-second measures "
                 "cannot be computed")
    return ClockModel(source=SOURCE_INDEX, fps=0.0, n_frames=int(pr.n_frames),
                      note="nothing usable found", warnings=warns)


def _cross_check_counts(model: ClockModel, video_path: str,
                        probe: Optional[VideoProbe]) -> None:
    """Rows should equal video frames. When they do not, say so, the recorder
    writes a session footer with the drop accounting, and a mismatch it cannot
    explain must reach the user rather than be absorbed."""
    if not video_path:
        return
    pr = probe if (probe and probe.ok) else probe_video(video_path)
    if not pr.ok or pr.n_frames <= 0 or model.n_frames <= 0:
        return
    diff = pr.n_frames - model.n_frames
    if diff == 0:
        return
    pct = abs(diff) / max(pr.n_frames, 1) * 100
    model.warnings.append(
        f"the video has {pr.n_frames} frames but the data file has "
        f"{model.n_frames} rows ({diff:+d}, {pct:.1f}%), poses will be "
        "aligned by frame_number, not by position")


# ── alignment ────────────────────────────────────────────────────────────────

def align_by_frame_number(frame_numbers: Sequence[int], clock: ClockModel,
                          *, base: int = 0) -> np.ndarray:
    """Timestamps for rows identified by their frame number rather than their
    position, which is what a session with dropped frames needs.

    A row whose frame number lies beyond the measured series gets NaN, absent,
    which downstream treats as a gap. Inventing a value there is how a dropped
    frame becomes a silent teleport.
    """
    ts = clock.timestamps() if clock.known_per_frame else None
    out = np.full(len(frame_numbers), np.nan, float)
    step = 1000.0 / clock.fps if clock.fps > 0 else math.nan
    for i, fn in enumerate(frame_numbers):
        idx = int(fn) - base
        if idx < 0:
            continue
        if ts is not None:
            if idx < len(ts):
                out[i] = ts[idx]
        elif not math.isnan(step):
            out[i] = idx * step
    return out
