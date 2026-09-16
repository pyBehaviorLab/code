"""FrameLog, per-(box, session) writer for _video_data.txt (v3 format).

One class for one file. The header is written lazily on the first row
write so callers can populate ``write_info`` fields in any order.

Thread-safety: one ``threading.Lock`` covers writer state. The camera
thread (frame writes) and the engine/IPC thread (event writes) can run
concurrently against the same FrameLog.

One row per camera frame, tab-delimited, and nine columns: the frame, three
clocks, the two latencies that produced the pose, the pose, its zone, and the
task's state.

``capture_host_ns`` is the raw host instant the frame was stamped with, kept
so that ``elapsed`` (1 ms) and ``frame_fw_ms`` (1 ms, clamped monotonic) can be
checked and recomputed: both are lossy and neither is reversible.
``frame_fw_ms`` is the board's own time at that frame's capture, written out
raw. ``pose_lag_ms`` is how long after the capture the pose existed, and
``filter_ms`` how much of that the Kalman and optical-flow smoothing took.
State names that landed during a frame's interval are pipe-joined into
``state``.

What is deliberately NOT here: the task's events and their timing, which the
MCU TSV records against the board's own clock and is the account to use; speed,
which follows from the coordinates and the scale calibration; and ``MCU PRINT``
messages, which belong in the per-session ``.log``.

File layout::

    # ---- header block ----
    #version           3
    #session_id        …
    #session_start     2026-04-24T15:02:17.314
    #box_id            2
    #video_file        video.mp4
    #camera            {json}
    #video_codec       {json}
    #pycontrol         {json}
    #zone_config_begin
    { … pretty-printed JSON … }
    #zone_config_end
    #tracker           {json}
    #sync              {json}
    #units             {"speed":"m/s"}      <- the px/m calibration
    #columns           frame  elapsed  capture_host_ns  frame_fw_ms  pose_lag_ms
                       filter_ms  pose  zone  state
    # ============================================================================

    # ---- per-frame rows (tab-separated) ----
    1   00:00.033   35742119033   16   41.7   0.42   [[…]]   Left_poke   init_trial
    2   00:00.066   35742152366   49   41.7   0.42   [[…]]   Left_poke   -
    3   00:00.100   35742185699   82   na     na     na      na          choice_state

    # ---- footer on close ----
    #session_end       2026-04-24T15:02:37.200  total_frames=601 …
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from source.video.recording.drop_log import drop_log as _drop_log
from source.video.tracking.types import PoseSnapshot

logger = logging.getLogger(__name__)

# Sentinel for absent per-frame fields. Interned so the hot path
# reuses one string each row.
_NA = "na"


# ----------------------------------------------------------------------------
# Module-level helpers (used by external callers + ``open_with_headers``).
# ----------------------------------------------------------------------------


def format_elapsed(capture_wall: datetime, session_start: datetime) -> str:
    """Format elapsed time since session_start as MM:SS.mmm
    (HH:MM:SS.mmm if >= 1 h)."""
    delta = capture_wall - session_start
    total_seconds = max(0.0, delta.total_seconds())
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes:02d}:{seconds:06.3f}"


def resolve_px_per_m(zones: Optional[list],
                     resolution: Optional[Tuple[int, int]]) -> Optional[float]:
    """Compute pixels-per-meter from the calibration scale zone.

    Walks ``zones`` for the entry with ``type == "scale"``, runs it
    through ``scale_zone_to_px_per_mm`` against the frame size, and
    converts mm → m. Returns ``None`` when no scale zone is configured,
    resolution is missing/zero, or the calibration is invalid, caller
    falls back to px/s. Shared by operant and maze so the unit decision
    is identical across both modes.
    """
    if not zones or not resolution:
        return None
    try:
        w, h = int(resolution[0]), int(resolution[1])
    except (TypeError, ValueError, IndexError):
        return None
    if w <= 0 or h <= 0:
        return None
    scale_zone = None
    for z in zones:
        if isinstance(z, dict) and z.get("type") == "scale":
            scale_zone = z
            break
    if scale_zone is None:
        return None
    try:
        from source.video.zones.coords import scale_zone_to_px_per_mm
        px_per_mm = scale_zone_to_px_per_mm(scale_zone, w, h)
    except Exception:
        return None
    if not px_per_mm or px_per_mm <= 0:
        return None
    return float(px_per_mm) * 1000.0


def build_filepath(video_dir: str | Path,
                   subject_id: str,
                   setup_id: int,
                   start_dt: datetime) -> Tuple[str, str]:
    """Return ``(video_data_filepath, mp4_basename)`` for one recorded run.

    Filename matches the recorder's stem
    (``<subject>-Box<N>-YYYY-MM-DD-HHMMSS``) so ``.mp4`` and
    ``_video_data.txt`` sit next to each other in the data folder.
    """
    from source.video.recording import build_session_stem
    stem = build_session_stem(subject_id, setup_id, start_dt)
    filepath = str(Path(video_dir) / f"{stem}_video_data.txt")
    return filepath, f"{stem}.mp4"


def open_with_headers(filepath: str | Path,
                      *,
                      subject_id: str,
                      setup_id: int,
                      start_dt: Optional[datetime] = None,
                      task_name: str = "",
                      task_hash: Optional[int] = None,
                      hd_name: str = "",
                      hd_hash: Optional[int] = None,
                      tracking_mode: str = "blob",
                      resolution: Optional[Tuple[int, int]] = None,
                      fps: Optional[float] = None,
                      zones: Optional[list] = None,
                      roi: Optional[List[int]] = None,
                      video_filename: Optional[str] = None,
                      px_per_m: Optional[float] = None,
                      model_type: Optional[str] = None,
                      tracker: Optional[str] = None,
                      n_animals: Optional[int] = None,
                      identities: Optional[list] = None,
                      bodyparts: Optional[list] = None,
                      skeleton: Optional[list] = None,
                      rebind_drop_log: bool = True,
                      ) -> "FrameLog":
    """Construct a FrameLog and push the standard info header fields.

    ``task_hash`` / ``hd_hash`` / ``hd_name``: lineage hashes pulled from
    ``pycboard.sm_info`` so the video file carries the SAME hashes as
    the MCU TSV header (``task_file_hash``, ``hardware_def_hash``).

    ``px_per_m``: pixels per meter computed from the per-box
    calibration scale zone (``scale_zone_to_px_per_mm * 1000``). When
    set, the ``speed`` column is written in m/s and the header carries
    ``#units {"speed":"m/s"}``. When ``None``, speed is px/s and the
    header carries ``#units {"speed":"px/s"}``.

    ``bodyparts``: the keypoint names, IN THE ORDER the pose column
    stores them. The column is positional, so without this the only link
    from a coordinate back to the part it belongs to is the model folder,
    which may have moved, been retrained, or not be on this machine at all.

    ``skeleton``: pairs of those names that are connected. Drawing only;
    it is what lets a reader redraw the figure the operator saw instead of
    guessing an anatomy from a part list.

    Both maze and operant call this, the only structural difference
    between their _video_data.txt files is whether the maze-only
    ``zones`` or ``roi`` info lines are populated.

    Raises if the writer can't be opened; the caller decides recovery.
    """
    dt = start_dt or datetime.now()
    log = FrameLog(filepath, session_start=dt,
                   video_file=video_filename,
                   px_per_m=px_per_m,
                   rebind_drop_log=rebind_drop_log)
    log.write_info("subject_id", subject_id)
    log.write_info("box_id", int(setup_id))
    from source.datetime_formats import format_header_ts_ms
    log.write_info("start_time", format_header_ts_ms(dt))
    if task_name:
        log.write_info("task_name", task_name)
    # Hashes as 8-char zero-padded hex, same format the MCU TSV header
    # and snapshot store use, so one string finds the run across all
    # three artifacts.
    if task_hash is not None:
        log.write_info("task_hash", f"{int(task_hash):08x}")
    if hd_name:
        log.write_info("hd_name", hd_name)
    if hd_hash is not None:
        log.write_info("hd_hash", f"{int(hd_hash):08x}")
    log.write_info("tracking_mode", tracking_mode)
    # Self-describing tracker meta (SLEAP model family / identity method /
    # animal count) so a reader knows how to parse the pose column.
    if model_type:
        log.write_info("model_type", model_type)
    if tracker:
        log.write_info("tracker", tracker)
    if n_animals is not None:
        log.write_info("n_animals", int(n_animals))
    if identities:
        log.write_info("identities", list(identities))
    if bodyparts:
        log.write_info("bodyparts", [str(b) for b in bodyparts])
    if skeleton:
        log.write_info("skeleton", [[str(a), str(b)]
                                    for a, b in skeleton if a and b])
    if resolution is not None:
        log.write_info("resolution", list(resolution))
    if fps is not None:
        log.write_info("fps", float(fps))
    if roi:
        log.write_info("roi", list(roi))
    if zones:
        log.write_info("zones", zones)
    return log


# ----------------------------------------------------------------------------
# FrameLog, one class for the whole _video_data.txt lifecycle.
# ----------------------------------------------------------------------------


class FrameLog:
    """Per-(box, session) writer for ``_video_data.txt`` (v2 format).

    One row per camera frame. MCU state transitions and events that
    arrived between the previous frame and this frame are folded into
    the new row's ``state`` / ``events`` columns (pipe-joined when
    multiple landed in the same interval).

    Live-write algorithm:
      * ``write_frame`` flushes the PREVIOUS frame's accumulated row
        (now finalised because we know nothing else will land in its
        interval), then opens a fresh pending buffer for THIS frame.
      * ``note_state`` / ``note_event`` append into the current
        pending buffer (the frame whose row is about to be written).
      * ``close`` flushes the final pending row.

    A 1-frame write delay is introduced (frame N's row hits disk when
    frame N+1 arrives) so events arriving slightly after their owning
    F can still be folded in. Acceptable for offline analysis; live
    tailing sees frames 1 row behind the camera.
    """

    #: One row per frame, and nothing that can be worked out from elsewhere.
    #:
    #: Cut back to these eight deliberately. What went, and where it lives now:
    #:
    #:   ``speed``        follows from the pose and the scale calibration; the
    #:                    analyser recomputes it from the coordinates;
    #:   ``events``       the task's own events, which the MCU TSV records and
    #:                    is the canonical account of;
    #:   ``pose_fw_ms``   the pose's age, now stated outright as ``pose_lag_ms``
    #:                    instead of left as a subtraction between two columns;
    #:   ``track``        a blob-tracker state that says nothing on a pose rig;
    #:   ``part_zones``   per-part zone membership and which parts were
    #:   ``filled_parts`` predicted; both belong with the analysis rather than
    #:                    in the live record.
    COLUMNS = (
        "frame",     # 1-based per-recording frame number. It is
                     # ``cam_frame_id`` re-based to THIS box's first frame, so
                     # it is NOT comparable across boxes: on a shared camera
                     # the boxes begin recording up to a second or two apart,
                     # and row N of one box was measured as much as 51 frames
                     # (1.7 s) from row N of another on a real 30 fps session.
                     # Join boxes on ``capture_host_ns``, which is the same
                     # instant for all of them because they are cropped from
                     # one CameraFrame.
        "elapsed",   # mm:ss.fff (or hh:mm:ss.fff) since session_start at capture
        "capture_host_ns",  # the RAW host instant this frame was stamped with,
                            # ``host_clock.host_ns()``, before it was reduced to
                            # ``elapsed`` (1 ms) or mapped to ``frame_fw_ms``
                            # (1 ms, and clamped monotonic). Both of those are
                            # lossy and neither is reversible, so without this
                            # column a recording cannot be checked: the frame
                            # interval cannot be recovered below 1 ms, a clamped
                            # frame_fw_ms cannot be told from a real one, and the
                            # host↔MCU mapping cannot be recomputed offline.
                            # It costs one integer per row.
                            #
                            # Read it with ``#timestamp_source`` in the header.
                            # ``host`` means it was taken when read() RETURNED,
                            # so it carries the whole driver and USB delay and is
                            # NOT the exposure instant; ``hw`` means the camera
                            # reported it.
        "frame_fw_ms",  # the board's own time at this frame's capture, written
                        # out raw. The column every later alignment to the task
                        # is built on.
        "pose_lag_ms",  # milliseconds from this pose's frame being CAPTURED to
                        # its pose existing: the whole journey, not the model.
                        # It was called ``infer_ms``, and the name cost a
                        # comparison: read as the model's time it says this rig
                        # is ten times slower than it is. Measured on this
                        # machine, the model is 3.7 ms of a 30.6 ms lag; the
                        # rest is the tick, the queue and the fan-out. ``na``
                        # when the row carries no pose.
        "filter_ms",  # how much of ``pose_lag_ms`` the Kalman and optical-flow
                      # smoothing took, for every keypoint. Sits next to the
                      # number it is part of, so a frame that ran late says
                      # which stage spent the time.
        "pose",      # JSON [[x,y,c],…] snapshot or na
        "zone",      # the zone the CENTROID is in, or na. Read it literally:
                     # with small zones such as pokes the centroid stays
                     # outside while the snout is well inside.
        "state",     # pipe-joined MCU state names in this frame's interval,
                     # "-" if none
    )

    # Event names suppressed because an F-row column already carries the
    # same information.
    SUPPRESS_EVENT_NAMES = frozenset({
        "zone_changed",   # zone column on F rows already shows transitions
    })

    _NONE = "-"

    # 64 KB Python-side buffer + ~4 Hz flush. The interval flush bounds
    # data loss on a hard crash to ``_FLUSH_INTERVAL_S`` of frames.
    #: How far back the MCU clock may step before this stops believing the
    #: value it already has. Measured jitter on a real session was 7 steps,
    #: worst -72 ms, so 250 ms is comfortably above the real thing and far
    #: below a wrong anchor, which is off by seconds or minutes.
    _FW_CLAMP_MAX_MS = 250.0

    _BUFFER_SIZE = 64 * 1024
    _FLUSH_INTERVAL_S = 0.25

    def __init__(self,
                 filepath: str | Path,
                 *,
                 session_start: Optional[datetime] = None,
                 video_file: Optional[str] = None,
                 version: int = 3,
                 px_per_m: Optional[float] = None,
                 rebind_drop_log: bool = True):
        self._path = Path(filepath)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._session_start = session_start or datetime.now()
        self._video_file = video_file or self._path.with_suffix(".mp4").name
        self._version = version

        # Speed unit conversion. None → emit px/s; >0 → emit m/s.
        # Values <= 0 are treated as "no calibration" so a misconfigured
        # scale zone can't produce a divide-by-zero.
        self._px_per_m: Optional[float] = (
            float(px_per_m) if (px_per_m and px_per_m > 0) else None)
        # Pre-cache reciprocal so per-frame speed conversion multiplies
        # instead of dividing on the hot path.
        self._inv_px_per_m: Optional[float] = (
            1.0 / self._px_per_m if self._px_per_m else None)

        self._lock = threading.Lock()
        self._file = self._path.open(
            "w", encoding="utf-8", buffering=self._BUFFER_SIZE)
        self._closed = False
        self._header_written = False
        self._last_flush = time.monotonic()

        # Header field accumulator, populated by ``write_info`` in any
        # order before the first frame row is written.
        self._info: Dict[str, Any] = {}
        # Set when the recorder switched encoders mid-session; written as a
        # closing ``#video_codec_final`` line. See note_encoder_change.
        self._encoder_change: Optional[Dict[str, Any]] = None

        # Per-session counters.
        self._frame_count = 0
        self._pose_count = 0
        #: Rows carrying a board timestamp. With ``_pose_count`` this is what
        #: the closing summary is built from: a file with frames but neither
        #: of these recorded video and nothing else, and used to say so only
        #: in a footer nobody reads.
        self._fw_rows = 0
        # I1 state: the last frame_fw_ms written, and how often the MCU clock
        # had to be clamped forward to keep the column monotonic.
        self._last_frame_fw_ms = None
        self._fw_clamps = 0
        #: Backward steps too large to be anchor jitter, where the previous
        #: value is rejected instead of this one.
        self._fw_reseats = 0
        self._dropped_frames = 0
        self._last_frame_num = -1

        # Pending frame buffer, accumulates state/event names from
        # MCU messages received between this frame's write_frame call
        # and the next one. Flushed on next write_frame (or on close).
        #
        #   {"frame_num": int, "capture_wall": datetime, "frame_fw_ms": float|None,
        #    "pose": PoseSnapshot|None, "zone": str|None, "speed": float|None,
        #    "states": [name, ...], "events": [name, ...]}
        self._pending_frame: Optional[Dict[str, Any]] = None

        # Route drop accounting to this session's directory. The drop log
        # is PROCESS-WIDE, a dry-run writer must pass
        # ``rebind_drop_log=False`` or starting a dry-run on one box
        # re-points every concurrently-recording box's drop rows into
        # ``data/temp/`` while their session _drops.tsv reads clean.
        if rebind_drop_log:
            try:
                _drop_log.set_session_path(self._path.parent / "_drops.tsv")
            except Exception as e:
                logger.debug("drop-log session path not set: %s", e)

    # ---- header (incremental) ----------------------------------------------

    def write_info(self, key: str, value: Any) -> None:
        """Push one header field. Order-free; the header block is
        emitted lazily on the first ``write_frame`` call."""
        self._info[key] = value

    def note_encoder_change(self, encoder: str, replaced: str,
                            lost_frames: int) -> None:
        """The recorder switched encoders part-way through the session.

        It happens only when a Jetson hardware encoder dies at start-up (see
        ``VideoRecorder._recover_dead_nvmpi_writer``). The video file was
        reopened, so everything in it now comes from ``encoder``. A header not
        yet written simply names it; one already written cannot be rewritten
        at the top of an open file, so a closing ``#video_codec_final`` line
        says it either way.
        """
        with self._lock:
            self._info["video_encoder"] = encoder
            self._encoder_change = {"encoder": encoder, "replaced": replaced,
                                    "lost_frames": int(lost_frames)}

    # ---- per-frame -----------------------------------------------------------

    def write_frame(self,
                    frame_number: int,
                    timestamp_ms: int,
                    speed: Optional[float] = None,
                    location: Optional[str] = None,
                    pose_array: Optional[List] = None,
                    frame_fw_ms: Optional[float] = None,
                    pose_fw_ms: Optional[float] = None,
                    pose_snapshot: Optional[PoseSnapshot] = None,
                    track_state: Optional[str] = None,
                    part_zones: Optional[Dict[str, str]] = None,
                    filled_parts: Optional[List[str]] = None,
                    pose_lag_ms: Optional[float] = None,
                    filter_ms: Optional[float] = None,
                    capture_host_ns: Optional[int] = None) -> None:
        """Open a new pending frame row.

        ``pose_snapshot``: a pre-built (possibly multi-instance) snapshot;
        when given it takes precedence over ``pose_array`` and the ``pose``
        column is written per-track. Single-animal callers keep passing the
        flat ``pose_array`` and the output is byte-identical to before.

        Flushes the PREVIOUS pending frame's row to disk first (now
        finalised, any state/event mirrored since that frame was opened
        gets folded into its ``state`` / ``events`` columns).

        ``frame_number``: canonical per-camera frame id
        (``CameraFrame.cam_frame_id``); monotonic and gap-free,
        independent of encoder backpressure.

        ``frame_fw_ms``: raw MCU framework timestamp the PC held when
        this frame was handled, written straight out. ``None`` → ``na``.

        ``pose_fw_ms``: raw MCU framework timestamp the PC held when this
        row's pose RESULT landed. ``None`` (or no pose) → ``na``.

        ``speed``: px/s as computed by the tracker sinks. Converted to
        m/s here when ``px_per_m`` was set at construction time; the
        header's ``#units`` line tells readers which unit they got.
        """
        if self._closed:
            return
        self._ensure_header_written()

        capture_wall = self._session_start + timedelta(milliseconds=int(timestamp_ms))
        # Production single-animal path: keep the raw pose_array on the
        # pending row and serialise it directly at flush, building a
        # PoseSnapshot + per-keypoint dicts just to re-serialise them cost
        # two allocations per recorded frame per box. A pre-built
        # (possibly multi-instance) snapshot still takes precedence.
        pose = pose_snapshot if pose_snapshot is not None else (
            pose_array if pose_array else None)

        with self._lock:
            if pose is not None:
                self._pose_count += 1
            self._last_frame_num = int(frame_number)
            prev = self._pending_frame
            # Pre-frame buffer absorption: if the previous pending has
            # no real frame_num (state/event arrived before any
            # write_frame), fold its accumulated names into THIS frame
            # rather than writing a phantom frame=0 row.
            inherited_states: List[str] = []
            inherited_events: List[str] = []
            write_prev: Optional[Dict[str, Any]] = None
            if prev is not None:
                if prev.get("frame_num") is None:
                    inherited_states = list(prev["states"])
                    inherited_events = list(prev["events"])
                else:
                    write_prev = prev
            # I1 · the MCU clock must never run backwards in this column.
            # Measured on a real session: 7 backward steps, worst -72 ms.
            # The host↔MCU anchor is re-taken on every board message, so a
            # later frame can map through an earlier anchor.
            # A reader aligning video to MCU time cannot recover from that, so
            # the column is clamped forward and the clamps are counted, a
            # storm of them means the anchor is thrashing and the session's
            # sync is suspect, which is worth knowing rather than smoothing.
            if frame_fw_ms is not None:
                last = self._last_frame_fw_ms
                back = (last - frame_fw_ms) if last is not None else 0.0
                if last is not None and back > 0:
                    if back <= self._FW_CLAMP_MAX_MS:
                        # Anchor jitter: the host<->MCU anchor is re-taken on
                        # every board message, so a later frame can map through
                        # an earlier anchor. Small and self-correcting.
                        self._fw_clamps += 1
                        if (self._fw_clamps in (1, 10, 100)
                                or self._fw_clamps % 500 == 0):
                            logger.warning(
                                "frame_fw_ms went backwards (%.0f -> %.0f ms, "
                                "%d time%s this session); clamping forward so "
                                "the column stays monotonic.",
                                last, frame_fw_ms, self._fw_clamps,
                                "" if self._fw_clamps == 1 else "s")
                        frame_fw_ms = last
                    else:
                        # A step this large is not jitter. The PREVIOUS value
                        # was wrong, so re-seat on this one rather than hold
                        # the bad one and clamp every real sample behind it.
                        #
                        # Unbounded, this clamp cost 163 SECONDS of a 600 s
                        # session on all four boxes: the first frame was
                        # stamped 163119 ms (the board's uptime) while every
                        # later frame carried run time, so each real value was
                        # smaller and got clamped away until wall time caught
                        # up. 27% of the session had no usable MCU time and
                        # nothing downstream could align video to it.
                        self._fw_reseats += 1
                        logger.warning(
                            "frame_fw_ms jumped back %.0f ms (%.0f -> %.0f), "
                            "far past the %.0f ms this clamp is for. Treating "
                            "the PREVIOUS value as the wrong one and "
                            "re-seating; %d re-seat%s this session.",
                            back, last, frame_fw_ms, self._FW_CLAMP_MAX_MS,
                            self._fw_reseats,
                            "" if self._fw_reseats == 1 else "s")
                self._last_frame_fw_ms = frame_fw_ms

            self._pending_frame = {
                "frame_num":    int(frame_number),
                "capture_wall": capture_wall,
                "capture_host_ns": capture_host_ns,
                "frame_fw_ms":  frame_fw_ms,
                "pose_fw_ms":   pose_fw_ms if pose is not None else None,
                "pose":         pose,
                "zone":         location if location else None,
                "speed_pxs":    speed,
                "track":        track_state,
                "part_zones":   part_zones or None,
                "filled_parts": list(filled_parts) if filled_parts else None,
                # Only meaningful on a row that carries a pose: they describe
                # how that pose was arrived at.
                "pose_lag_ms":     pose_lag_ms if pose is not None else None,
                "filter_ms":    filter_ms if pose is not None else None,
                "states":       inherited_states,
                "events":       inherited_events,
            }
        if write_prev is not None:
            self._write_frame_row(write_prev)

    def on_dropped_frame(self, count: int = 1, note: str = "dropped") -> None:
        """Count a dropped frame.

        The per-drop detail, with its reason, is in ``_drops.tsv`` next to
        this file, and the total is in the closing summary. It used to be
        written inline in an ``events`` column as well; that column is gone,
        and one account of the drops is better than two that can disagree.
        """
        if self._closed:
            return
        self._ensure_header_written()
        with self._lock:
            self._dropped_frames += count

    # ---- MCU state / event mirrors (engine / IPC thread) ---------------------

    def note_state(self, name: str) -> None:
        """Note a state transition for the current pending frame.

        ``name`` is the resolved state name from
        ``pycboard.sm_info.ID2name[state_id]`` (caller does the lookup),
        appended to the pending frame's ``states`` list and pipe-joined
        into the ``state`` column on flush.
        """
        self._note_mcu("state", name)

    def note_event(self, name: str) -> None:
        """Accepted and dropped: this file no longer has an events column.

        The task's events are the MCU TSV's account, timestamped by the board
        itself rather than bucketed into a frame interval, and that is the one
        a reader should use. The call sites are left in place because they are
        the same mirror that feeds ``note_state``, which this file does keep.
        """
        return

    def _note_mcu(self, kind: str, name: str) -> None:
        """Append a state or event name into the current pending
        frame's buffer.

        Names are stored as bare strings, for authoritative per-event
        MCU timing cross-reference the MCU TSV via ``task_hash``. If no
        frame has been written yet, the entry seeds a pre-frame buffer
        that the first ``write_frame`` picks up.
        """
        if self._closed or not name:
            return
        self._ensure_header_written()
        with self._lock:
            if self._pending_frame is None:
                # No frame seen yet, start a sparse "pre-frame" buffer
                # that the first write_frame will absorb.
                self._pending_frame = {
                    "frame_num":    None,
                    "capture_wall": datetime.now(),
                    "frame_fw_ms":  None,
                    "pose_fw_ms":   None,
                    "pose":         None,
                    "zone":         None,
                    "speed_pxs":    None,
                    "states":       [],
                    "events":       [],
                }
            bucket = ("states" if kind == "state" else "events")
            self._pending_frame[bucket].append(str(name))

    # ---- close --------------------------------------------------------------

    def close(self) -> None:
        # Drain the final pending row OUTSIDE the close-lock, the
        # write call takes the same lock so nesting would deadlock.
        with self._lock:
            final = self._pending_frame
            self._pending_frame = None
        # Skip pre-frame buffers (frame_num=None) that carry no
        # state/event payload, writing a phantom row would be noise.
        # If the pre-frame buffer accumulated names, write it with
        # frame=0 so the events aren't lost.
        if final is not None:
            if final.get("frame_num") is None and not final["states"] and not final["events"]:
                final = None
        if final is not None:
            try:
                self._write_frame_row(final)
            except Exception as e:
                self._warn_io("final frame-row write", e)
        with self._lock:
            if self._closed:
                return
            try:
                # ``_frame_count`` is incremented inside _write_frame_row
                # so it equals the exact number of F rows on disk
                # (cam_frame_id can start large, so don't derive from it).
                self._write_footer_locked(
                    total_frames=self._frame_count,
                    dropped_frames=self._dropped_frames,
                    pose_count=self._pose_count,
                )
                self._file.flush()
                self._file.close()
            finally:
                self._closed = True
        self._say_what_this_session_recorded()

    def _say_what_this_session_recorded(self) -> None:
        """One line per session, and a warning when a column stayed empty.

        The footer already carried these numbers, at the bottom of a file
        nobody opens until the experiment is over. A session that recorded
        video and nothing else - because the board never started, or because
        pose was never enabled - looked exactly like a session that worked,
        until someone read the file days later and found every column ``na``.
        """
        frames = self._frame_count
        if not frames:
            return
        name = self._path.name
        logger.info("%s: %d frames, %d with pose, %d with a board time.",
                    name, frames, self._pose_count, self._fw_rows)
        if self._pose_count == 0 and self._declared_tracker():
            logger.warning(
                "%s: %d frames recorded and NOT ONE carries pose, although the "
                "session declares a %s tracker. The video is fine; the pose "
                "column is empty for the whole run.",
                name, frames, self._declared_tracker())
        if self._fw_rows == 0:
            logger.warning(
                "%s: %d frames recorded with no board timestamp on any row. "
                "The board did not run, so this video cannot be aligned to the "
                "task at all.", name, frames)

    def __enter__(self) -> "FrameLog":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Destructor safety net only. Logging from __del__ during
        # interpreter shutdown / GC is unsafe (handlers may be gone),
        # so stay silent here.
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass

    # ========================================================================
    # Internals
    # ========================================================================

    def _declared_tracker(self) -> str:
        """The tracker this session said it was running, or ``""``.

        Read off the header block rather than a separate flag, so it cannot
        disagree with what the file itself claims.
        """
        mode = str(self._info.get("tracking_mode", "") or "")
        return "" if mode in ("", "none", "unknown") else mode

    def _ensure_header_written(self) -> None:
        """Lazily emit the header block on first frame/event. Idempotent."""
        if self._header_written:
            return
        with self._lock:
            if self._header_written:
                return
            self._write_header_locked()

    def _write_header_locked(self) -> None:
        """Build the header dicts from ``self._info`` and write them.
        Caller MUST hold ``self._lock``."""
        info = self._info
        subject = info.get("subject_id", "unknown")
        box = info.get("box_id", 0)
        res = info.get("resolution", [0, 0])
        fps = info.get("fps", 0)
        roi = info.get("roi")

        session_header = {
            "session_id":
                f"{subject}_box{box}_{self._session_start.strftime('%Y-%m-%d-%H%M%S')}",
            "box_id": box,
        }
        camera_info: Dict[str, Any] = {
            "resolution":
                (f"{int(res[0])}x{int(res[1])}"
                 if res and len(res) >= 2 else "unknown"),
            "fps": fps,
        }
        if roi:
            camera_info["roi"] = list(roi)
        # WHOSE CLOCK the frame times came from. A UVC camera through OpenCV
        # has no per-frame hardware timestamp, so the time in every row below
        # is taken when ``read()`` RETURNED and carries the whole USB and
        # driver delay: measured on this rig, 72 ms on a direct camera and
        # 121 to 125 ms through a CCTV chain. Spinnaker and XIMEA report a
        # real device timestamp instead. An analysis that aligns this file to
        # controller time needs to know which it is holding, and until now the
        # file did not say. ``requested_fps`` is recorded beside the rate the
        # file is actually written at, because a camera that will not hold the
        # requested rate is the normal case rather than the exception.
        for key in ("timestamp_source", "requested_fps", "transport_delay_ms"):
            value = info.get(key)
            if value not in (None, ""):
                camera_info[key] = value
        codec_info = {
            # Named only where the recorder supplies it (a Jetson, see
            # VideoRecorder.header_encoder); everywhere else it stays unknown.
            "encoder": info.get("video_encoder") or "unknown",
            "container": Path(self._video_file).suffix.lstrip("."),
        }
        pycontrol_info = {
            "task": info.get("task_name", ""),
            "subject": subject,
        }
        # Lineage hashes, same values the MCU TSV header carries
        # under ``task_file_hash`` / ``hardware_def_hash``. Skipped when
        # the caller didn't supply them (test fixtures, dry runs).
        if "task_hash" in info:
            pycontrol_info["task_hash"] = info["task_hash"]
        if "hd_name" in info:
            pycontrol_info["hd_name"] = info["hd_name"]
        if "hd_hash" in info:
            pycontrol_info["hd_hash"] = info["hd_hash"]
        zone_config = self._build_zone_config()
        # Self-describing tracker block: a reader can branch on backend /
        # model_type / n_animals without guessing which tracker produced the
        # pose column (DLC keypoints vs SLEAP single/multi vs blob centroid).
        tracker_info: Dict[str, Any] = {
            "backend": info.get("tracking_mode", "unknown"),
        }
        for _k in ("model_type", "tracker", "n_animals", "identities"):
            if info.get(_k) is not None:
                tracker_info[_k] = info[_k]
        # The pose column is positional; these two are what make it readable
        # without the model folder. Written under the tracker block because
        # they describe the tracker that produced the column, not the session.
        if info.get("bodyparts"):
            tracker_info["bodyparts"] = info["bodyparts"]
        if info.get("skeleton"):
            tracker_info["skeleton"] = info["skeleton"]
        sync_meta = {"source": "mcu_event_stream"}

        f = self._file
        f.write(
            "# ============================================================================\n"
            "# pyBehaviorLab - video + tracking session log (one file per box per session)\n"
            "# ============================================================================\n"
        )
        from source.datetime_formats import format_header_ts_ms
        self._kv(f, "version", self._version)
        self._kv(f, "session_id", session_header["session_id"])
        self._kv(f, "session_start", format_header_ts_ms(self._session_start))
        self._kv(f, "box_id", session_header["box_id"])
        self._kv(f, "video_file", self._video_file)
        self._kv_json(f, "camera", camera_info)
        self._kv_json(f, "video_codec", codec_info)
        self._kv_json(f, "pycontrol", pycontrol_info)

        f.write("#zone_config_begin\n")
        f.write(self._format_zone_config(zone_config))
        f.write("\n#zone_config_end\n")

        self._kv_json(f, "tracker", tracker_info)
        self._kv_json(f, "sync", sync_meta)

        # Units declaration, readers parse this to know whether the
        # ``speed`` column is m/s (calibration scale set) or px/s.
        units = {"speed": "m/s" if self._px_per_m is not None else "px/s"}
        self._kv_json(f, "units", units)

        f.write("#columns           " + " ".join(self.COLUMNS) + "\n")
        f.write(
            "# ============================================================================\n"
        )
        f.flush()
        self._header_written = True
        self._last_flush = time.monotonic()

    def _build_zone_config(self) -> Dict[str, Any]:
        """Normalise ``info['zones']`` + ``info['scale']`` + ``info['arena']``
        into the nested zone_config dict the header expects.

        ``info['zones']`` may be either a list of zone dicts OR the
        already-nested ``{"scale":…, "arena":…, "zones":…}`` form.
        """
        zones_raw = self._info.get("zones", [])
        if (isinstance(zones_raw, dict)
                and "zones" in zones_raw
                and isinstance(zones_raw["zones"], dict)):
            return zones_raw

        zones_dict: Dict[str, Any] = {}
        if isinstance(zones_raw, list):
            for z in zones_raw:
                if not isinstance(z, dict):
                    continue
                name = z.get("name")
                if not name:
                    continue
                zones_dict[name] = {k: v for k, v in z.items() if k != "name"}
        elif isinstance(zones_raw, dict):
            zones_dict = {k: v for k, v in zones_raw.items()
                          if k not in ("scale", "arena")}

        return {
            "scale": self._info.get("scale", {}),
            "arena": self._info.get("arena", {}),
            "zones": zones_dict,
        }

    def _write_frame_row(self, pending: Dict[str, Any]) -> None:
        """Write one finalised frame row (the 9-column v2 shape)."""
        with self._lock:
            if self._closed:
                return
            frame_num = pending["frame_num"]
            # Pre-frame buffer that never got a real frame number, mark
            # it with 0.
            if frame_num is None:
                frame_num = 0
            elapsed = format_elapsed(pending["capture_wall"], self._session_start)
            frame_fw = pending.get("frame_fw_ms")
            frame_fw_str = (f"{frame_fw:.0f}"
                            if frame_fw is not None else _NA)
            p = pending["pose"]
            if p is None:
                pose_str = _NA
            elif isinstance(p, PoseSnapshot):
                pose_str = self._pose_to_json(p)
            else:
                pose_str = self._pose_array_to_json(p)
            zone_str = pending["zone"] if pending["zone"] else _NA
            state_str = "|".join(pending["states"]) if pending["states"] else self._NONE
            infer = pending.get("pose_lag_ms")
            infer_str = f"{infer:.1f}" if infer is not None else _NA
            filt = pending.get("filter_ms")
            filter_str = f"{filt:.2f}" if filt is not None else _NA
            if frame_fw is not None:
                self._fw_rows += 1
            try:
                cap_ns = pending.get("capture_host_ns")
                cap_ns_str = _NA if cap_ns is None else str(int(cap_ns))
                self._file.write(
                    f"{frame_num}\t{elapsed}\t{cap_ns_str}\t{frame_fw_str}\t"
                    f"{infer_str}\t{filter_str}\t"
                    f"{pose_str}\t{zone_str}\t{state_str}\n"
                )
            except OSError as e:
                # Disk-full/IO failure on the one path that loses the most
                # data, surface via the rate-limited warning, not a
                # debug-level swallow in the sink.
                self._warn_io("frame row write", e)
                return
            self._frame_count += 1
            # Interval-gated flush (``_FLUSH_INTERVAL_S``) bounds live-tail
            # latency for state/event rows too. An unconditional per-row
            # flush here degraded to one flush PER FRAME during event-dense
            # task phases; close() still drains + flushes everything.
            self._flush_if_due()

    def _format_speed(self, speed_pxs: Optional[float]) -> str:
        """Convert speed (px/s) → output string in the right unit.

        ``_px_per_m`` set → m/s with 3 decimals; None → px/s with 1
        decimal. ``None`` input → ``na``.
        """
        if speed_pxs is None:
            return _NA
        if self._inv_px_per_m is not None:
            return f"{speed_pxs * self._inv_px_per_m:.3f}"
        return f"{speed_pxs:.1f}"

    def _write_footer_locked(self, *, total_frames: int,
                             dropped_frames: int,
                             pose_count: int) -> None:
        if not self._header_written:
            # No frames + no events ever arrived. Don't pollute the
            # file with a header just to write a footer; just leave it.
            return
        if self._encoder_change:
            self._kv_json(self._file, "video_codec_final", self._encoder_change)
        from source.datetime_formats import format_header_ts_ms
        end_wall = format_header_ts_ms(datetime.now())
        parts = [
            f"#session_end     {end_wall}",
            f"total_frames={total_frames}",
            f"dropped_frames={dropped_frames}",
            f"total_pose_inferences={pose_count}",
        ]
        self._file.write("  ".join(parts) + "\n")

    def _warn_io(self, what: str, exc: BaseException) -> None:
        """Surface a tracking-file write/flush failure (hidden data
        loss otherwise). Rate-limited to once / 5 s so the per-frame
        path can't spam the log."""
        now = time.monotonic()
        if now - getattr(self, "_io_warn_at", 0.0) > 5.0:
            self._io_warn_at = now
            logger.warning("TrackingWriter %s failed (data may be lost): %r",
                           what, exc)

    def _flush_if_due(self) -> None:
        """Flush if the interval elapsed. Caller MUST hold ``self._lock``."""
        now = time.monotonic()
        if now - self._last_flush >= self._FLUSH_INTERVAL_S:
            try:
                self._file.flush()
            except Exception as e:
                self._warn_io("flush", e)
            self._last_flush = now

    # ---- formatting helpers (static) ---------------------------------------

    @staticmethod
    def _kv(f, key: str, value: Any) -> None:
        f.write(f"#{key:<17s} {value}\n")

    @staticmethod
    def _kv_json(f, key: str, value: Dict[str, Any]) -> None:
        f.write(f"#{key:<17s} {json.dumps(value, separators=(',', ':'))}\n")

    @staticmethod
    def _format_zone_config(zone_config: Dict[str, Any]) -> str:
        """Pretty-print zone_config with one-zone-per-line compact rows."""
        out: list[str] = ["{"]
        keys = list(zone_config.keys())
        n = len(keys)
        for i, k in enumerate(keys):
            v = zone_config[k]
            if k == "zones" and isinstance(v, dict):
                out.append('  "zones": {')
                zk = list(v.keys())
                for j, zname in enumerate(zk):
                    zv = v[zname]
                    sep = "," if j < len(zk) - 1 else ""
                    out.append(
                        f'    "{zname}": '
                        + json.dumps(zv, separators=(",", ":"))
                        + sep
                    )
                out.append("  }" + ("," if i < n - 1 else ""))
            else:
                out.append(
                    f'  "{k}": '
                    + json.dumps(v, indent=None, separators=(",", ":"))
                    + ("," if i < n - 1 else "")
                )
        out.append("}")
        return "\n".join(out)

    @staticmethod
    def _pose_to_json(pose: PoseSnapshot) -> str:
        """Compact JSON pose for the row's ``pose`` column.

        Single-animal (≤1 instance): ``[[x,y,c], …]``, byte-identical to the
        pre-multi format, so existing readers and files are unaffected.
        Multi-animal (>1 instance): ``{"<track_id>": [[x,y,c], …], …}`` keyed by
        identity, so a reader distinguishes single vs multi by array-vs-object
        (and by the header's ``n_animals``).
        """
        def _arr(inst) -> list:
            return [[round(x, 2), round(y, 2), round(c, 3)]
                    for (x, y, c) in inst.body_parts.values()]

        insts = pose.instances
        if len(insts) <= 1:
            return json.dumps(_arr(pose.primary) if pose.primary else [],
                              separators=(",", ":"))
        out = {(inst.track_id if inst.track_id is not None else str(i)): _arr(inst)
               for i, inst in enumerate(insts)}
        return json.dumps(out, separators=(",", ":"))

    @staticmethod
    def _pose_array_to_json(pose_array: List) -> str:
        """Serialise a raw ``[[x, y, conf], …]`` pose_array (the production
        per-frame path) straight to the row's ``pose`` column,
        byte-identical to the PoseSnapshot single-animal path, without
        the intermediate snapshot/dict builds."""
        arr = []
        for entry in pose_array:
            try:
                arr.append([round(float(entry[0]), 2),
                            round(float(entry[1]), 2),
                            round(float(entry[2]), 3)])
            except (TypeError, IndexError, ValueError):
                continue
        return json.dumps(arr, separators=(",", ":"))


__all__ = [
    "FrameLog",
    "open_with_headers",
    "build_filepath",
    "format_elapsed",
    "resolve_px_per_m",
]
