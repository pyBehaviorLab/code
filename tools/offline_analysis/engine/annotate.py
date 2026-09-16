"""Drawing poses onto frames, without a GUI attached.

The Verify view could already overlay a recording and save the result, and its
own comment says why that export must not be reimplemented: *"a saved video
that disagreed with the frame the user checked would be worse than no export
at all."* That is right, and it is why this module exists rather than a second
copy of the drawing living in the batch path.

Everything here takes explicit arguments, a frame, a pose, a colour order,
and imports no Qt. The Verify view calls it holding a widget; the pipeline
calls it holding a session file; both get identical pixels.

What the batch path adds is the **state name**. A clip of an operant session
is unreadable without it: the animal sits still for ninety seconds and there
is nothing on screen to say whether that was ``init_trial`` or the
inter-trial interval.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Per-keypoint colours, in the order the model declares its parts. BGR,
#: because that is what OpenCV draws in.
BP_COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0),
             (0, 255, 255), (180, 105, 255), (200, 200, 200),
             (255, 128, 0), (128, 255, 128), (255, 255, 255)]

#: How a mouse's parts join up, in the words models actually use. SLEAP and
#: DLC projects rarely export their skeleton, the 6-point model here declares
#: `n_edges: 0`: so the alternative to naming the joins is a scatter of
#: unconnected dots.
SKELETON = (
    ("tail_base", "center"), ("center", "head"), ("head", "snout"),
    ("head", "left_ear"), ("head", "right_ear"),
    ("tailbase", "centre"), ("centre", "head"),
    ("center", "neck"), ("neck", "head"), ("body", "head"),
    ("nose", "head"),
)


def bp_colour(name: str, order: Sequence[str]) -> tuple:
    """This part's colour, stable for the whole recording."""
    try:
        return BP_COLORS[list(order).index(name) % len(BP_COLORS)]
    except ValueError:
        return BP_COLORS[-1]


def skeleton_pairs(names: Sequence[str]):
    """The joins that can actually be drawn for these parts."""
    lower = {str(n).lower(): n for n in names}
    out = []
    for a, b in SKELETON:
        if a in lower and b in lower and (lower[a], lower[b]) not in out:
            out.append((lower[a], lower[b]))
    return out


def primary_point(pose: dict) -> tuple[float, float] | None:
    """The point a single-animal measure follows, centre if there is one."""
    for want in ("center", "centre", "body", "middle"):
        for bp, v in (pose or {}).items():
            if bp.lower() == want and v and len(v) >= 2 and v[0] is not None:
                return float(v[0]), float(v[1])
    for _bp, v in sorted((pose or {}).items()):
        if v and len(v) >= 2 and v[0] is not None:
            return float(v[0]), float(v[1])
    return None


# ── the drawing ──────────────────────────────────────────────────────────

def draw_pose(frame, pose: dict, *, order: Sequence[str] = (),
              skeleton: Sequence[tuple[str, str]] = (),
              scale: float = 1.0) -> None:
    """Keypoints, joined, in the colours the legend names.

    Part names are deliberately NOT printed beside the points: on a 360x202
    frame six labels overlap each other and the animal, which is most of what
    made a saved video unreadable.
    """
    import cv2

    drawn = {}
    for bp, v in (pose or {}).items():
        if not v or len(v) < 2 or v[0] is None:
            continue
        drawn[bp] = (round(float(v[0]) * scale),
                     round(float(v[1]) * scale))
    if not drawn:
        return

    # Skeleton first, so the joints sit on top of the bones. The recording's
    # declared edges win over a guess from part names.
    pairs = [(a, b) for a, b in skeleton if a in drawn and b in drawn]
    for a, b in (pairs or skeleton_pairs(list(drawn))):
        # Dark casing under a light core, so a bone still reads against an
        # overexposed arena floor.
        cv2.line(frame, drawn[a], drawn[b], (20, 20, 20), 3, cv2.LINE_AA)
        cv2.line(frame, drawn[a], drawn[b], (235, 235, 235), 1, cv2.LINE_AA)

    order = list(order) or list(drawn)
    radius = 3 if min(frame.shape[:2]) < 400 else 4
    for bp, (x, y) in drawn.items():
        cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, bp_colour(bp, order), -1, cv2.LINE_AA)


def draw_trail(frame, points: Sequence[tuple[float, float]],
               scale: float = 1.0) -> None:
    """The path the animal took to get here."""
    import cv2

    pts = [(round(x * scale), round(y * scale))
           for x, y in points if x is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, (90, 190, 255), 1, cv2.LINE_AA)


def draw_zone_label(frame, name: str) -> None:
    """Which zone the animal is in, big, along the bottom."""
    import cv2

    if not name:
        return
    h, w = frame.shape[:2]
    big = min(h, w) >= 400
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9 if big else 0.45
    thick = 2 if big else 1
    (tw, th), _ = cv2.getTextSize(str(name), font, scale, thick)
    org = (max(4, (w - tw) // 2), h - max(6, th // 2))
    cv2.putText(frame, str(name), org, font, scale, (0, 0, 0), thick + 2,
                cv2.LINE_AA)
    cv2.putText(frame, str(name), org, font, scale, (255, 255, 255), thick,
                cv2.LINE_AA)


def draw_state(frame, state: str) -> None:
    """The task state, top-right, the thing an operant clip is unreadable
    without. The animal sitting still for ninety seconds looks identical in
    ``init_trial`` and the inter-trial interval."""
    import cv2

    if not state:
        return
    h, w = frame.shape[:2]
    big = min(h, w) >= 400
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7 if big else 0.4
    thick = 2 if big else 1
    (tw, th), _ = cv2.getTextSize(str(state), font, scale, thick)
    org = (max(4, w - tw - 8), th + 8)
    cv2.putText(frame, str(state), org, font, scale, (0, 0, 0), thick + 2,
                cv2.LINE_AA)
    cv2.putText(frame, str(state), org, font, scale, (120, 255, 160), thick,
                cv2.LINE_AA)


def annotate_frame(frame, *, pose: dict | None = None,
                   zones: list[dict] | None = None,
                   location: str = "", state: str = "",
                   trail: Sequence[tuple[float, float]] = (),
                   order: Sequence[str] = (),
                   skeleton: Sequence[tuple[str, str]] = (),
                   scale: float = 1.0):
    """One frame with everything on it. Never mutates the caller's array."""
    out = frame.copy()
    if zones:
        try:
            from tools.offline_analysis.analyze.zone_renderer import render_zones_opencv
            out = render_zones_opencv(out, zones,
                                      highlight_zone=location or None)
        except Exception as e:                            # pragma: no cover
            logger.debug("zone overlay failed: %s", e)
    if trail:
        draw_trail(out, trail, scale)
    if pose:
        draw_pose(out, pose, order=order, skeleton=skeleton, scale=scale)
    draw_zone_label(out, location)
    draw_state(out, state)
    return out


# ── the whole video ──────────────────────────────────────────────────────

def write_annotated_video(video_path: str, pose_path: str, out_path: str, *,
                          zones: list[dict] | None = None,
                          fps: float = 0.0, trail_frames: int = 60,
                          tracked_span_only: bool = True,
                          states: dict[int, str] | None = None,
                          progress: Callable[[int, int], None] | None = None,
                          cancel: Any = None) -> str:
    """Write ``video_path`` with the poses in ``pose_path`` drawn on it.

    Read straight through rather than seeking per row: a seek per frame turns
    a 9,000-frame session into minutes of thrashing.

    Rows are matched to pictures by **frame number**, never by position. A
    recording that dropped frames has fewer rows than the video has pictures,
    and pairing them in order slides the overlay out of step part way through,
    which looks like a tracking failure and is not one.
    """
    import cv2

    from tools.offline_analysis import video_data_schema as vds
    from tools.offline_analysis.engine.mcu_states import state_of_cell

    header = vds.VideoDataHeader.parse(pose_path)
    order = list(getattr(header, "body_parts", ()) or ())
    rows_by_frame: dict[int, dict] = {}
    for row in vds.iter_rows(pose_path, header):
        fn = vds.parse_int(row.get("frame_number"), -1)
        if fn >= 0:
            rows_by_frame.setdefault(fn, row)
    if not rows_by_frame:
        raise ValueError(f"no pose rows in {os.path.basename(pose_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"could not open {os.path.basename(video_path)}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        rate = float(fps or cap.get(cv2.CAP_PROP_FPS) or 20.0)
        # The pose may be in a different pixel space from the picture (the rig
        # can encode the video smaller than it reads the camera). Scale it,
        # rather than drawing the animal in the wrong place.
        pose_w = _pose_width(header) or width
        scale = float(width) / float(pose_w) if pose_w else 1.0

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 rate, (width, height))
        if not writer.isOpened():
            raise OSError(
                f"no encoder available for {os.path.basename(out_path)}, "
                "try an .avi name")
        # Only the part that was actually tracked, by default. A cancelled or
        # partial retrack would otherwise be written out in full: a few
        # annotated frames buried in an hour of untouched video, which reads
        # as "the overlay is broken" rather than "the retrack stopped early".
        first = min(rows_by_frame) if tracked_span_only else 0
        last = max(rows_by_frame) if tracked_span_only else (total or 1 << 30)
        if tracked_span_only and total and (last - first + 1) < total:
            logger.info(
                "annotating frames %d-%d of %d, the rest of the recording "
                "was never tracked, so it would carry no overlay.",
                first, last, total)

        trail: list[tuple[float, float]] = []
        written = 0
        frame_no = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if cancel is not None and cancel.is_set():
                    break
                if frame_no < first:
                    frame_no += 1
                    continue
                if frame_no > last:
                    break
                row = rows_by_frame.get(frame_no)
                pose = vds.parse_pose(row.get("pose_array")) if row else None
                point = primary_point(pose or {})
                if point is not None:
                    trail.append(point)
                    del trail[:-trail_frames]
                loc = (row or {}).get("location") or ""
                # The pose file's own column first; the joined MCU log where
                # it is blank. A clip of an operant session is unreadable
                # without the state, and the recording that has it in its
                # `.tsv` rather than its rows is the common case.
                state = state_of_cell((row or {}).get("stage"))
                if not state and states:
                    state = states.get(frame_no, "")
                writer.write(annotate_frame(
                    frame, pose=pose, zones=zones,
                    location="" if vds.is_na(loc) else str(loc),
                    state=state,
                    trail=trail, order=order, scale=scale))
                written += 1
                frame_no += 1
                if progress is not None and written % 200 == 0:
                    progress(written, total)
        finally:
            writer.release()
    finally:
        cap.release()
    logger.info("annotated video: %d frames to %s", written, out_path)
    return out_path


def _pose_width(header) -> int:
    """The width the poses are expressed in, from the header, or 0."""
    for key in ("pose_resolution", "resolution"):
        value = str((getattr(header, "info", None) or {}).get(key) or "")
        if "x" in value:
            try:
                return int(value.split("x")[0])
            except ValueError:
                continue
    size = getattr(header, "pose_resolution", None) or getattr(
        header, "resolution", None)
    if isinstance(size, (tuple, list)) and size and size[0]:
        return int(size[0])
    return 0


__all__ = ["BP_COLORS", "SKELETON", "annotate_frame", "bp_colour", "draw_pose",
           "draw_state", "draw_trail", "draw_zone_label", "primary_point",
           "skeleton_pairs", "write_annotated_video"]
