"""RecorderSink, lossless writes to disk, backpressure surfaces as an alarm.

One sink instance for the whole pipeline. Owns one VideoRecorder + one
TrackingWriter per active box. Bound queue sized for ~3 s of frames at
the camera FPS, large enough to absorb GPU encoder hiccups, small
enough that queue overflow gets surfaced quickly.

Drop policy is LOSSLESS: every recorded frame matters. Two backpressure
sources surface the same way (pipeline_health signal with the box id):

  * Sink queue full      worker can't keep up with camera arrival.
  * recorder.add_frame   encoder thread queue full (NVENC/x264 lag).

Both are user-visible alarms, never silent.

Per-frame TrackingWriter row is written from the same worker pass, using
the most recent pose PoseSink published. If no pose has arrived (skipped
or in-flight), the row's pose_array is empty, the explicit signal that
inference didn't run on this frame.
"""

from __future__ import annotations

import logging
import threading
import time
from source import host_clock
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from source.video.framebus.types import BoxFrame
from source.video.recording.frame_log import FrameLog
from source.video.recording.recorder import VideoRecorder

# Every concrete writer is a FrameLog; aliased ``TrackingWriter`` here.
TrackingWriter = FrameLog
from source.video.framebus.sink_base import (
    DropPolicy, Sink, default_sink_workers,
)

logger = logging.getLogger(__name__)


# Health alarm callback: (box_id, reason: str)
HealthCallback = Callable[[int, str], None]


class RecorderSink(Sink):
    """One sink, multiple per-box recorders. Lossless with surfaced backpressure."""

    name = "recorder"

    def __init__(self, *,
                 health_cb: Optional[HealthCallback] = None) -> None:
        # THREE SECONDS, whatever the rate. 90 frames is three seconds at 30
        # fps and one and a half at 60, so the depth this was designed to have
        # quietly halved on a faster camera. ``set_rate_hint`` re-bounds it
        # once the pipeline knows what the camera delivers; 90 stays the floor
        # so a slow camera cannot make it shallower than it was.
        # Several workers because
        # encoding is per-box independent work that releases the GIL, and one
        # thread cannot feed sixteen encoders; boxes are pinned to a worker so
        # each recorder still sees its own frames strictly in order.
        super().__init__(name="recorder", maxsize=90, seconds=3.0,
                         policy=DropPolicy.LOSSLESS,
                         workers=default_sink_workers())
        self._recorders: Dict[int, VideoRecorder] = {}
        self._tracking_writers: Dict[int, TrackingWriter] = {}
        # Per-box pycboard, frame/pose rows are stamped via
        # ``pycboard.fw_ms_at(capture_host_ns)`` (monotonic extrapolation
        # from the last-message anchor; None until anchored).
        self._pycboards: Dict[int, Any] = {}
        # latest pose snapshot per box: (cam_frame_id, pose_array, location,
        # speed, pose_fw_ms, part_zones). ``pose_fw_ms`` is the raw MCU
        # framework timestamp the PC held when the pose RESULT landed, and
        # ``part_zones`` is which zone each confident keypoint was in - the
        # tracker computes it for the triggers and it was thrown away here.
        self._last_pose: Dict[
            int, Tuple[int, list, Optional[str], float, Optional[int], dict,
                       list, Optional[float], Optional[float]]] = {}
        # box_id -> latest track state (see blob.TRACK_*). Absent until the
        # tracker reports one, which is why the row writer defaults it.
        self._last_track_state: Dict[int, str] = {}
        # box_id -> recording start monotonic_ns (for relative timestamp)
        self._rec_start_host_ns: Dict[int, int] = {}
        # box_id -> cam_frame_id of the first frame accepted after
        # start_recording. Anchors the TSV ``frame_num`` column at 1 per
        # run (the camera counter is already in the thousands from live
        # preview by Record time).
        self._first_cam_frame_id: Dict[int, int] = {}
        self._health_cb = health_cb
        # Boxes whose encoder-death alarm has already fired, so the health
        # banner is raised once per failure rather than every frame while the
        # tracking writer keeps this box's rows flowing.
        self._encoder_failed_alarmed: set[int] = set()
        self._lock = threading.RLock()

    # ── Per-box recorder lifecycle (called by Pipeline) ----------------

    def start_recording(self, setup_id: int, *,
                        recorder: VideoRecorder,
                        tracking_writer: Optional[TrackingWriter] = None,
                        annotate_callback: Optional[Callable] = None,
                        pycboard: Any = None) -> None:
        """Attach a per-box recorder. Caller has already configured/opened it.

        ``pycboard``: the box's board, so the sink can stamp each frame /
        pose row via ``fw_ms_at(capture_host_ns)`` (anchored monotonic
        extrapolation). ``None`` → ``na`` fw columns."""
        with self._lock:
            self._recorders[setup_id] = recorder
            if tracking_writer is not None:
                self._tracking_writers[setup_id] = tracking_writer
            if pycboard is not None:
                self._pycboards[setup_id] = pycboard
            if annotate_callback is not None:
                recorder.annotate_callback = annotate_callback
            self._rec_start_host_ns[setup_id] = host_clock.host_ns()
            self._encoder_failed_alarmed.discard(setup_id)
            self._last_pose.pop(setup_id, None)
            # Reset recording-local frame numbering. Set on first frame
            # whose capture_host_ns >= rec_start_mono (in process()).
            self._first_cam_frame_id.pop(setup_id, None)

    def stop_recording(self, setup_id: int):
        """Detach the per-box recorder and return it so the caller reaps it.

        Closing the VideoRecorder flushes + closes its ffmpeg child
        (~100-300 ms) and must not run under ``self._lock``, so the encoder
        is handed back instead of dropped. The tracking writer is closed
        here (cheap)."""
        # Frames already queued for this box were captured while recording
        # was ON, give the worker a bounded window to write them before
        # the recorder detaches, then account whatever is left (LOSSLESS:
        # every captured frame written or in _drops.tsv).
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if not self._per_box.get(setup_id):
                    break
            time.sleep(0.01)
        with self._lock:
            leftover = len(self._per_box.get(setup_id) or ())
        if leftover:
            from source.video.recording.drop_log import drop_log as _dlog
            _dlog.record(self.name, reason=f"stop_discard_x{leftover}")
            logger.warning(
                "Box %d: %d queued frame(s) not written before recorder "
                "stop (accounted in _drops.tsv)", setup_id, leftover)
        with self._lock:
            recorder = self._recorders.pop(setup_id, None)
            tracking_writer = self._tracking_writers.pop(setup_id, None)
            self._pycboards.pop(setup_id, None)
            self._rec_start_host_ns.pop(setup_id, None)
            self._last_pose.pop(setup_id, None)
            self._first_cam_frame_id.pop(setup_id, None)
            self._encoder_failed_alarmed.discard(setup_id)
        if tracking_writer is not None:
            try:
                tracking_writer.close()
            except Exception as e:
                logger.warning("TrackingWriter close error (box=%d): %s",
                               setup_id, e)
        return recorder

    def is_recording(self, setup_id: int) -> bool:
        with self._lock:
            return setup_id in self._recorders

    # ── Pose result feed (wired by Pipeline) ---------------------------

    def on_pose_result(self, setup_id: int, cam_frame_id: int,
                       pose_array: list,
                       location: Optional[str] = None,
                       speed: float = 0.0,
                       zones_by_body_part: Optional[dict] = None,
                       raw_pose_dict: Optional[dict] = None,
                       capture_host_ns: int = 0,
                       forecast_coords: Optional[dict] = None,
                       infer_done_ns: int = 0,
                       pose_lag_ms: Optional[float] = None,
                       filter_ms: Optional[float] = None) -> None:
        """Cache the latest pose for the write path to stamp onto rows.

        NOT frame-id matching, despite how this once read: ``process`` uses
        whatever pose is cached when a row is written, because PoseSink and
        RecorderSink subscribe to the bus independently and a row is written
        immediately, frame N's pose usually lands after frame N's row, so
        matching on the id would leave most rows blank. ``pose_fw_ms`` below
        carries the inferred frame's own time, which is what lets a reader
        tell a repeated pose from a fresh one.

        ``forecast_coords`` (latency-compensated centroid) is accepted for
        signature compatibility but ignored, _video_data.txt records the
        raw/smoothed position, not the forward-predicted one.

        Signature matches PoseSink's ``PoseResultCallback``: extra
        ``zones_by_body_part`` / ``raw_pose_dict`` are accepted so the
        callback can be wired directly without a wrapper.

        ``pose_lag_ms`` is how long after this frame was CAPTURED its pose
        existed, and ``filter_ms`` how much of that the smoother took. Both
        are measured by the pose sink and written per row, so the pose's age
        is one number the reader can look at rather than a subtraction
        between two timestamp columns.
        """
        with self._lock:
            pose_fw = self._frame_fw_ms(setup_id, capture_host_ns)
            self._last_pose[setup_id] = (
                cam_frame_id, pose_array, location, speed, pose_fw,
                dict(zones_by_body_part or {}),
                # Written by PoseSink's gap filler under the established
                # underscore-prefixed metadata convention, so this needed no
                # change to any callback signature.
                list((raw_pose_dict or {}).get("_filled_parts") or ()),
                pose_lag_ms, filter_ms)

    def _frame_fw_ms(self, setup_id: int, capture_host_ns: int):
        """MCU framework ms at a host capture instant, or ``None``.

        The mapping is affine from the board's last-message anchor
        (``timestamp + 1000 * (host_s - last_message_mono)``), so an unstamped
        frame does NOT map to 0, it maps to roughly minus the host's uptime
        in ms. One such row silently poisons the staleness column, which is
        why a zero instant must be rejected rather than mapped.
        """
        if not capture_host_ns:
            return None
        pyc = self._pycboards.get(setup_id)
        if pyc is None:
            return None
        try:
            return pyc.fw_ms_at(capture_host_ns)
        except Exception:
            return None

    def on_tracker_result(self, setup_id: int, cam_frame_id: int,
                          centroid, location: Optional[str] = None,
                          speed: float = 0.0,
                          zones_by_body_part: Optional[dict] = None,
                          position=None,
                          capture_host_ns: int = 0) -> None:
        """Cache the latest blob-tracker result for the _video_data row.

        Signature matches TrackerSink's ``TrackerResultCallback`` so the
        callback wires directly. Shares ``_last_pose`` with the pose feed,
        a box runs one tracking mode at a time, and the row writer only
        needs the latest location/speed. ``pose_array`` stays None (blob
        has no keypoints), so the pose column reads ``na`` while zone and
        speed are real.
        """
        with self._lock:
            result_fw = self._frame_fw_ms(setup_id, capture_host_ns)
            self._last_pose[setup_id] = (
                cam_frame_id, None, location, speed, result_fw, {}, [],
                None, None)

    def on_track_state(self, setup_id: int, cam_frame_id: int,
                       state: str, capture_host_ns: int = 0) -> None:
        """Cache whether the tracker currently KNOWS where the animal is.

        Written into the row so a reader can separate a measured position
        from a coasted guess, and an occlusion from a detector that simply
        stopped writing. Without it the two are indistinguishable after the
        fact, which is the difference between "the animal was under the
        shelf for 40 s" and a straight line drawn through that period.
        """
        with self._lock:
            self._last_track_state[setup_id] = state

    # ── Sink worker --------------------------------------------------

    def _stamp_clock_source(self, bid, writer) -> None:
        """Record WHOSE CLOCK this file's frame times came from.

        A UVC camera through OpenCV has no per-frame hardware timestamp, so
        every time in this file is taken when ``read()`` returned and carries
        the whole USB and driver delay. Spinnaker and XIMEA report a real
        device timestamp. An analysis aligning this file to controller time
        needs to know which it holds, and the file did not say.
        """
        try:
            recorder = self._recorders.get(bid)
            if recorder is None:
                return
            source = str(getattr(recorder, "timestamp_source", "") or "")
            if source:
                writer.write_info("timestamp_source", source)
            want = float(getattr(recorder, "requested_fps", 0) or 0)
            if want > 0:
                writer.write_info("requested_fps", want)
        except Exception as e:
            logger.debug("box %s: could not stamp the clock source: %s", bid, e)

    def process(self, frame: BoxFrame) -> None:
        # Single lock acquisition covers all per-frame state reads plus the
        # first-frame counter install, avoiding two acquires per frame.
        bid = frame.setup_id
        with self._lock:
            recorder = self._recorders.get(bid)
            tracking_writer = self._tracking_writers.get(bid)
            rec_start = self._rec_start_host_ns.get(bid, frame.capture_host_ns)
            pose_entry = self._last_pose.get(bid)
            track_state = self._last_track_state.get(bid)
            first_cam_id = self._first_cam_frame_id.get(bid)
            if first_cam_id is None and frame.capture_host_ns >= rec_start:
                # Anchor on the first frame captured AT/after record start,
                # a pre-roll frame would anchor the count and then be
                # dropped below, leaving frame_num starting at >1 and every
                # TSV row off-by-N versus the video. Install under the same
                # lock so it's visible to future frames (one worker per
                # box, no concurrent process()).
                self._first_cam_frame_id[bid] = int(frame.cam_frame_id)
                first_cam_id = self._first_cam_frame_id[bid]

        recording = recorder is not None and getattr(recorder, "recording", False)

        # Encoder gave up (fatal write-error streak) while a tracking writer
        # keeps this box's rows flowing: ``recording`` is now False so the
        # ``if recording:`` block below never re-runs and never re-alarms.
        # Raise the health banner once so the operator sees the video is dead
        # instead of a silently-not-growing file. The check + one-shot mark
        # go under the lock (start/stop_recording discard the same set under
        # it); the health_cb fires OUTSIDE the lock to avoid re-entrancy.
        fire_encoder_dead = False
        if recorder is not None and getattr(recorder, "encoder_failed", False):
            with self._lock:
                if bid not in self._encoder_failed_alarmed:
                    self._encoder_failed_alarmed.add(bid)
                    fire_encoder_dead = True
        if fire_encoder_dead and self._health_cb is not None:
            try:
                self._health_cb(bid, "encoder_dead")
            except Exception:
                pass

        if not recording and tracking_writer is None:
            return  # nothing registered for this box, live preview only

        # Drop frames captured before record start. CameraThread's
        # pre-roll deque still holds live-preview frames at Record time;
        # they must not leak into the recording (else frame_num=1 isn't
        # the file start and fw capture ms can go negative).
        if frame.capture_host_ns < rec_start:
            return

        if recording:
            # One owned, contiguous ROI copy at ingest. For a shared camera
            # ``frame.image`` is a non-contiguous view into the whole camera
            # frame; storing that view in the encoder's multi-second queue
            # would pin the entire parent frame (memory blow-up) AND force
            # the ffmpeg writer thread to ``ascontiguousarray`` per write.
            # Copying the ROI once here retains only ROI-sized bytes and
            # hands the writer an already-contiguous buffer. For a
            # non-shared camera the frame is already contiguous, so this is
            # a no-op (numpy returns the same array).
            img = np.ascontiguousarray(frame.image)
            ok = recorder.add_frame(img, frame.capture_host_ns)
            if not ok and self._health_cb is not None:
                try:
                    self._health_cb(frame.setup_id, "encoder_backpressure")
                except Exception:
                    pass

        # Per-frame _video_data row. Written whenever a tracking writer is
        # registered, INDEPENDENT of the video recorder, so the dry-run
        # safety net can capture _video_data.txt with no video file.
        if tracking_writer is not None:
            pose_array: Optional[list] = None
            location: Optional[str] = None
            speed = 0.0
            pose_fw_ms: Optional[int] = None
            part_zones: dict = {}
            filled_parts: list = []
            pose_lag_ms = filter_ms = None
            if pose_entry is not None:
                # Use the latest cached pose for every row. PoseSink and
                # RecorderSink subscribe to the bus independently and the
                # row is written immediately, so pose for frame N usually
                # lands after N's row; exact-frame matching would miss.
                # Repeating a pose between inferences is fine for offline
                # analysis, the snapshot carries the inference timestamp
                # for callers that discount stale samples.
                (_, parr, loc, spd, pose_fw_ms, part_zones,
                 filled_parts, pose_lag_ms, filter_ms) = pose_entry
                pose_array = parr
                location = loc
                speed = spd

            # State / event NAMES land in the row's state/events columns via
            # ``McuRowMirror`` → ``FrameLog.note_state`` / ``note_event``.

            # MCU framework time at this frame's capture instant, map the
            # frame's own ``capture_host_ns`` through the box's host↔fw
            # clock anchor. ``na`` when no board is bound or the run clock
            # isn't anchored yet.
            frame_fw_ms = self._frame_fw_ms(bid, frame.capture_host_ns)

            timestamp_ms = max(0, (frame.capture_host_ns - rec_start) // 1_000_000)
            # Recording-local frame number, 1-based. ``cam_frame_id`` is
            # the camera thread's gap-free monotonic counter; subtract the
            # first accepted frame's id to anchor the column at 1.
            local_frame_num = int(frame.cam_frame_id) - int(first_cam_id) + 1
            # I3 · the header's geometry must be the geometry that was
            # actually written. It came from config, and twice now the two
            # disagreed in a real session (declared 853x830 / 315x176, written
            # 720x700 / 360x202), which silently invalidates every pixel
            # coordinate a reader derives from it. The log emits its header
            # lazily on this first row, so correcting it here still lands.
            if local_frame_num == 1:
                try:
                    fh, fw_px = frame.image.shape[:2]
                    tracking_writer.write_info("resolution", [int(fw_px), int(fh)])
                except Exception as e:
                    logger.debug("box %s: could not stamp real resolution: %s",
                                 bid, e)
                self._stamp_clock_source(bid, tracking_writer)
            try:
                tracking_writer.write_frame(
                    frame_number=local_frame_num,
                    timestamp_ms=int(timestamp_ms),
                    speed=speed,
                    location=location,
                    pose_array=pose_array,
                    frame_fw_ms=frame_fw_ms,
                    pose_fw_ms=pose_fw_ms,
                    track_state=track_state,
                    part_zones=part_zones,
                    filled_parts=filled_parts,
                    pose_lag_ms=pose_lag_ms,
                    filter_ms=filter_ms,
                    # The raw instant everything else on this row is derived
                    # from. elapsed rounds it to 1 ms and frame_fw_ms rounds
                    # AND clamps it, so without this the row cannot be checked.
                    capture_host_ns=frame.capture_host_ns,
                )
            except Exception as e:
                logger.debug("TrackingWriter row error (box=%d): %s",
                             frame.setup_id, e)

    # ── Sink overrides, surface LOSSLESS overflow as health alarm ----

    def _record_drop(self, reason: str, frame: BoxFrame) -> None:
        super()._record_drop(reason, frame)
        if self._health_cb is not None:
            try:
                self._health_cb(frame.setup_id, f"recorder_{reason}")
            except Exception:
                pass
