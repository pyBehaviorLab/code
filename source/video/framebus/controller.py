"""Pipeline, camera + bus + sinks orchestrator.

The single owner of the live video pipeline. Both maze and operant
construct one ``Pipeline``, call its public API, and never reach into
its internals.

Public surface (per box, by topic):

  Box registration
    register_box(box_id, pycboard=...)
    unregister_box(box_id)

  Camera lifecycle
    connect_camera(camera_id, box_id, ...)
    disconnect_camera(box_id)

  Recording (lossless, surfaces backpressure)
    start_recording(box_id, recorder, tracking_writer=..., annotate_callback=...)
    stop_recording(box_id)

  Pose (skip-but-write semantics built in)
    configure_pose_model(...)
    enable_pose(box_id, zone_lookup=...)
    disable_pose(box_id)

  Blob tracking (older non-DLC path)
    enable_blob_tracking(box_id)
    disable_blob_tracking(box_id)
    configure_push_zones(box_id, zones)
    configure_push_tracking(box_id, cfg)

  Health
    on_health(callback)                           # (box_id, reason), record drops, etc

  Shutdown
    shutdown()

The class carries no Qt dependency. Use ``qt_bridge.QtBridge`` to bind
pose / tracker / health events to Qt signals so the GUI sees them on
the main thread.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from source.video.cameras.capture import VideoManager
from source.video.framebus.frame_bus import FrameBus
from source.video.framebus.latency import LatencyBudget
from source.video.framebus.mcu_pusher import MCUPusher
from source.video.framebus.pose_sink import PoseFailedCallback, PoseResultCallback, PoseSink
from source.video.framebus.recorder_sink import RecorderSink
from source.video.framebus.sink_base import notify_subscribers
from source.video.framebus.tracker_sink import TrackerResultCallback, TrackerSink
from source.video.recording.drop_log import drop_log

logger = logging.getLogger(__name__)


def _opencv_thread_count(cores: Optional[int] = None) -> int:
    """OpenCV intra-op thread count that avoids N-camera oversubscription
    without starving the display compositor / encoder opens on a big host.

    The pipeline already parallelises at the box level, so cv2's all-cores
    pool would add N×cores contention. But pinning to 1-2 on a many-core
    16-box rig throttles the 16 serial per-tile ``cv2.resize`` calls (display)
    and the 16 serial ffmpeg-writer opens (multi-start). Scale with the core
    count instead: 1 on small/Jetson hosts (<=4 cores), then ~cores/4 up to a
    ceiling of 6 so a few intra-op threads are available without oversubscribing.
    """
    import os as _os
    c = cores if cores is not None else (_os.cpu_count() or 4)
    if c <= 4:
        return 1
    return max(2, min(6, c // 4))


def _tune_opencv_threads() -> None:
    """Apply ``_opencv_thread_count`` to cv2. Process-global; offline tools run
    in their own process and self-tune. Guarded, a cv2 without setNumThreads
    just keeps its default."""
    try:
        import cv2
        cv2.setNumThreads(_opencv_thread_count())
    except Exception:
        pass


HealthCallback = Callable[[int, str], None]


# Live Pipelines, weakly held. A Pipeline owns worker threads and sits in a
# reference cycle, so a leaked one is freed by the CYCLIC collector at an
# arbitrary later moment, possibly while those threads still touch its
# memory. The registry lets a supervisor (the test suite's teardown, a
# shutdown path) find and stop stragglers without walking every object.
_LIVE_PIPELINES: "weakref.WeakSet[Pipeline]" = weakref.WeakSet()


def live_pipelines() -> list:
    """Snapshot of Pipelines that have not been garbage-collected yet."""
    return list(_LIVE_PIPELINES)


class Pipeline:
    """Central tracking pipeline, single class both operant and maze import.

    Owns: camera lifecycle (`VideoManager`), per-camera `FrameBus`, the
    sink workers (`RecorderSink`, `PoseSink`, `TrackerSink`, `MCUPusher`),
    per-box FW-time sync, and the 100 Hz drain tick. Sinks
    are private implementation details, call sites use the public methods
    below to register a box, connect its camera, configure pose/tracker,
    start/stop recording, etc.

    Each sink owns its thread + bounded queue + drop policy:
      - Recorder: blocking, depth=64 (lossless; surfaces backpressure)
      - Pose:     leaky depth=1 (skip-but-write, empty rows for skipped frames)
      - Tracker:  per-box bounded
      - Push:     event-driven (no frame queue; reads pose/tracker results)

    A slow sink cannot block another sink. Recording stays lossless even
    when DLC stalls.

    Qt-free: use `QtBridge` to relay events into Signals on the main
    thread.

    """

    def __init__(self, *, target_fps: int = 30) -> None:
        # Pin OpenCV's intra-op thread pool low so N camera/sink threads don't
        # oversubscribe the cores. Process-global; set once at startup.
        _tune_opencv_threads()
        # Camera lifecycle (owns the camera threads).
        self.video_manager = VideoManager()
        # Shared-camera ROI segment table ({'boxes': [...]}), owned here;
        # the Camera Connect dialog and project load publish through
        # set_segment_config, connect_camera applies it per camera.
        self._segment_config: Optional[dict] = None
        self.video_manager.set_target_fps(target_fps)

        # Per-camera FrameBus.
        self._buses: Dict[Any, FrameBus] = {}
        # Lens/FOV correction, one machine-level cache shared by every bus,
        # keyed internally by camera_id. Uncalibrated cameras cost nothing.
        from source.video.cameras.lens import LensCorrectionCache
        self.lens = LensCorrectionCache()
        # Last-seen camera liveness per camera id, so the tick watchdog fans
        # a per-box health banner only on a state CHANGE (see the camera
        # liveness contract in capture.py).
        self._cam_liveness: Dict[Any, str] = {}
        # Last shortfall message fanned per camera, so a steady condition
        # banners once instead of every tick.
        self._cam_shortfall: Dict[Any, str] = {}

        # Per-box state.
        # Boxes the pipeline knows about (registered via register_box /
        # connect_camera). Membership only, no clock state.
        self._registered_boxes: set = set()
        self._pycboards: Dict[int, Any] = {}

        # Per-box subscription handles for bus → sinks (so we can
        # unsubscribe cleanly when a box is unregistered).
        self._box_unsubs: Dict[int, List[Callable[[], None]]] = {}

        # ── Central config registries, both modes call these directly,
        # no per-mode duplication of camera/tracking state.
        from source.video.framebus.types import CameraConfig, TrackingConfig
        self._camera_configs:   Dict[str, CameraConfig]   = {}
        self._tracking_configs: Dict[int, TrackingConfig] = {}

        # Observational timing spine, per-stage latency (capture→MCU).
        # Read via latency_snapshot(); recording never alters behaviour.
        self.latency = LatencyBudget()
        _LIVE_PIPELINES.add(self)

        # Sinks.
        self.recorder = RecorderSink(health_cb=self._notify_health)
        self.pose = PoseSink()
        self.tracker = TrackerSink()
        self.push = MCUPusher()
        self.pose.set_latency_budget(self.latency)
        self.tracker.set_latency_budget(self.latency)
        self.push.set_latency_budget(self.latency)

        # Attached further down, once the result-callback lists exist: the
        # probe subscribes to them, and building it here crashed the Pipeline
        # constructor, i.e. the whole app failed to start.
        # Benchmark LED windows, {setup_id: {"a": [x,y,w,h], "b": [...]}} in
        # normalised box coordinates. Set from the project on load; empty in
        # every ordinary session and read only by the latency probe.
        self.bench_leds: Dict[str, Any] = {}

        # Wire pose / tracker results to push + recorder. The recorder gets
        # BOTH feeds so blob-mode sessions land zone/speed in
        # _video_data.txt, not just on the MCU and overlay.
        self.pose.on_result(self.recorder.on_pose_result)
        self.pose.on_result(self.push.on_pose_result)
        self.tracker.on_result(self.recorder.on_tracker_result)
        self.tracker.on_result(self.push.on_tracker_result)
        # Track state (tracked / coasting / lost), the recorder writes it so
        # an occlusion is visible in the session file rather than being a gap
        # the reader has to guess at.
        self.tracker.on_track_state(self.recorder.on_track_state)

        # External result subscribers (qt_bridge relays them to GUI).
        self._on_pose_result: List[PoseResultCallback] = []
        self._on_tracker_result: List[TrackerResultCallback] = []
        self._on_health: List[HealthCallback] = []
        self._on_pose_failed: List[PoseFailedCallback] = []
        # Trigger frames (T2/T3): one TriggerFrame per pose frame, fanned to the
        # GUI (annotation + live plot) via the bridge.
        self._on_trigger_frame: List[Callable[[Any], None]] = []


        # Bridge them into the sinks.
        self.pose.on_result(self._fanout_pose_result)
        self.pose.on_failed(self._fanout_pose_failed)
        self.tracker.on_result(self._fanout_tracker_result)
        # Trigger frames come from the MCU push POLICY (the single evaluator),
        # not a parallel engine, the policy already fires the events and now
        # emits the per-frame state for annotation + the Session-Plot lane.
        self.push.set_trigger_frame_sink(self._fanout_trigger_frame)

        # Tick (drains CameraThreads, publishes to FrameBus).
        # ``drain_recording_buffer()`` returns every frame queued since the
        # last drain, so the tick adds latency without dropping anything (the
        # camera buffer absorbs frames between drains). That latency is half
        # the interval on average, and it was being paid in full: at a 10 ms
        # cadence the measured ``capture_to_poll`` was 5.4 ms across every rig
        # we recorded, 1 to 4 boxes, 30 and 100 fps. Flat at half the interval
        # is the signature of a frame waiting for the next wake-up rather than
        # waiting for work, so the interval was the cost, not the draining.
        #
        # 2 ms trades tick-thread wake-ups for that wait. The drain is a list
        # swap per camera and returns immediately when nothing is queued, so
        # the extra wake-ups cost far less than the 4 ms they buy back, and
        # they matter most at 100 fps where the old interval was a whole frame
        # period. The real fix is for the camera thread to publish on capture
        # and leave this loop as a liveness watchdog; this is the interim.
        self._tick_running = False
        self._tick_thread: Optional[threading.Thread] = None
        self._tick_interval_s = 0.002
        # Monotonic count of frames handed to the buses. Sampled twice by
        # ``tools/diagnose_fps.py`` to show whether the drain keeps up with
        # capture, the stage that had never been measured directly.
        self.n_published = 0

        # Start sink workers.
        self.recorder.start()
        self.pose.start()
        self.tracker.start()
        self.push.start()  # no-op (event-driven sink)

        logger.info("Pipeline initialised (target_fps=%d)", target_fps)

    # ──────────────────────────────────────────────────────────────────
    # Box registration
    # ──────────────────────────────────────────────────────────────────

    def register_box(self, setup_id: int, *, pycboard: Any = None) -> None:
        """Tell the pipeline this box exists; optionally bind its pycboard."""
        self._registered_boxes.add(setup_id)
        if pycboard is not None:
            self._pycboards[setup_id] = pycboard
            self.push.register_box(setup_id, pycboard)

    def update_box_pycboard(self, setup_id: int, pycboard: Any) -> None:
        """Bind or DETACH a box's pycboard reference.

        ``pycboard=None`` is the disconnect path, the camera worker thread
        must not push events to a closed serial port, so we clear the per-box
        entry on MCUPusher and our own ``_pycboards`` map.
        """
        if pycboard is None:
            self._pycboards.pop(setup_id, None)
            self.push.unregister_box(setup_id)
            return
        self.register_box(setup_id, pycboard=pycboard)
        # Re-attach after an MCU disconnect: unregister_box dropped the push
        # policy (coord_mapping / triggers / gates), and register_box created
        # a fresh empty one, rebuild it from the central TrackingConfig, or
        # an MCU port bounce mid-session silently stops every coord/zone/
        # trigger push until the next Record or dialog Apply.
        self.apply_tracking_config(setup_id)

    def unregister_box(self, setup_id: int) -> None:
        # Detach from buses + sinks.
        for unsub in self._box_unsubs.pop(setup_id, []):
            try:
                unsub()
            except Exception:
                pass
        for bus in self._buses.values():
            bus.unregister_box(setup_id)
        self.push.unregister_box(setup_id)
        self.pose.disable_for_box(setup_id)
        self.tracker.disable_for_box(setup_id)
        # Detach + reap the encoder so unregister (e.g. camera disconnect while
        # still recording) doesn't orphan an ffmpeg child. close() can block
        # ~300 ms flushing, so do it off-thread.
        _rec = self.recorder.stop_recording(setup_id)
        if _rec is not None:
            def _reap(r=_rec, b=setup_id):
                try:
                    r.stop_recording()
                    r.close()
                except Exception as e:
                    logger.error("unregister_box reap encoder (box=%s): %s", b, e)
            threading.Thread(target=_reap, daemon=True,
                             name=f"ReapEncoder-{setup_id}").start()
        self._registered_boxes.discard(setup_id)
        self._pycboards.pop(setup_id, None)

    @property
    def tracker_manager(self):
        """The shared :class:`TrackerManager` owned by :class:`TrackerSink`.

        Exposes the per-box blob tracker registry to GUI callers
        (background set, enhancer attach, zone-manager attach). Read-only
        proxy; there is exactly one TrackerManager per pipeline and it
        lives inside the sink.
        """
        return self.tracker.manager

    # ──────────────────────────────────────────────────────────────────
    # Camera lifecycle
    # ──────────────────────────────────────────────────────────────────

    def connect_camera(self, camera_id, setup_id, *,
                       segment_config: Optional[dict] = None,  # default: own table
                       grayscale: Optional[bool] = None,
                       camera_backend: Optional[str] = None,
                       camera_config: Optional[dict] = None) -> bool:
        # Ensure box is registered.
        if setup_id not in self._registered_boxes:
            self.register_box(setup_id)

        if segment_config is None:
            segment_config = self._segment_config

        # Disconnect-first when re-assigning the box to a DIFFERENT camera.
        # Without this the box's old CameraThread + FrameBus are never told
        # (the old bus keeps deriving empty frames and, if this box was its
        # only member, the device is never released). Same-camera reconnect is
        # left to the caller / VideoManager (which reaps a dead same-id thread).
        prev_cam = self.video_manager.box_camera_map.get(setup_id)
        if prev_cam is not None and str(prev_cam) != str(camera_id):
            self.disconnect_camera(setup_id)

        # Source of truth: the central CameraConfig for this camera.
        # Kwargs are an explicit override path for callers that bypass the
        # registry.
        cam_cfg = self.get_camera_config(camera_id)

        effective_backend = camera_backend or cam_cfg.camera_backend or "opencv"
        effective_grayscale = (
            bool(grayscale) if grayscale is not None
            else bool(cam_cfg.grayscale)
        )

        # Push CameraConfig's resolution / fps / strategy onto the video
        # manager BEFORE start_camera so the camera thread's first
        # configure() reads consistent values.
        if cam_cfg.selected_resolution is not None:
            try:
                self.video_manager.set_target_resolution(
                    *cam_cfg.selected_resolution
                )
            except Exception as e:
                logger.debug("set_target_resolution from cam_cfg failed: %s", e)
        if cam_cfg.selected_fps:
            try:
                self.video_manager.set_target_fps(int(cam_cfg.selected_fps))
            except Exception as e:
                logger.debug("set_target_fps from cam_cfg failed: %s", e)
        if cam_cfg.frame_strategy:
            try:
                self.video_manager.set_frame_strategy(cam_cfg.frame_strategy)
            except Exception as e:
                logger.debug("set_frame_strategy from cam_cfg failed: %s", e)

        # Vendor extras live in cam_cfg.extra; explicit camera_config kwarg
        # overrides field-by-field (so callers can patch one parameter
        # without re-supplying the others).
        cfg = dict(cam_cfg.extra or {})
        if camera_config:
            cfg.update(camera_config)
        cfg.setdefault("grayscale", effective_grayscale)
        # Capture-time geometry, so every consumer sees one picture.
        cfg.setdefault("flip_horizontal", bool(cam_cfg.flip_horizontal))
        cfg.setdefault("flip_vertical", bool(cam_cfg.flip_vertical))
        # Carry THIS camera's own resolution + FPS in camera_config so
        # start_camera opens each camera at its own settings instead of the
        # shared VideoManager default (which the last connect overwrites).
        if cam_cfg.selected_resolution is not None:
            cfg.setdefault("width", int(cam_cfg.selected_resolution[0]))
            cfg.setdefault("height", int(cam_cfg.selected_resolution[1]))
        if cam_cfg.selected_fps:
            cfg.setdefault("target_fps", int(cam_cfg.selected_fps))
        if cam_cfg.capture_format:
            cfg.setdefault("capture_format", cam_cfg.capture_format)
        # Which door to open the camera through. An explicit choice is
        # honoured; otherwise, when the capability probe measured more than
        # one backend, the fastest one for THIS resolution and rate is used.
        #
        # This is the same kind of setting as the resolution and the rate and
        # it constrains the same ceiling, measured here, one camera gave
        # 10 fps at 720p through DirectShow and 30 through Media Foundation.
        # Choosing a mode without choosing the door that serves it is choosing
        # half the setting.
        backend_name = str(getattr(cam_cfg, "capture_backend", "") or "")
        if not backend_name and getattr(cam_cfg, "probed_variants", None):
            backend_name = self._fastest_backend_for(cam_cfg)
        if backend_name:
            cfg.setdefault("capture_backend", backend_name)
        # Scientific-camera I/O (Spinnaker/Ximea): exposure/gain + trigger +
        # line output (strobe/sync). Carried per-camera so each opens at its own
        # settings; the backend ignores what it doesn't support.
        if cam_cfg.exposure_us is not None:
            cfg.setdefault("exposure_us", float(cam_cfg.exposure_us))
        if cam_cfg.gain_db is not None:
            cfg.setdefault("gain_db", float(cam_cfg.gain_db))
        cfg.setdefault("trigger", cam_cfg.trigger.to_json())
        cfg.setdefault("line_output", cam_cfg.line_output.to_json())

        ok = self.video_manager.start_camera(
            camera_id, setup_id,
            segment_config=segment_config,
            camera_backend=effective_backend, camera_config=cfg,
        )
        if not ok:
            return False

        # Get-or-create the bus for this camera.
        bus = self._buses.get(camera_id)
        if bus is None:
            bus = FrameBus(camera_id,
                           segment_processor=self.video_manager.segment_processor_for(camera_id),
                           health_cb=self._notify_health,
                           lens=self.lens)
            self._buses[camera_id] = bus
        else:
            # An EXISTING bus keeps the segmenter it was built with, so
            # re-drawing ROIs and pressing Connect on a camera that is already
            # streaming left the old crop in place. ``start_camera`` has
            # already installed the new VideoSegmentProcessor by this point;
            # hand it to the bus so the new regions actually take effect.
            bus.set_segment_processor(
                self.video_manager.segment_processor_for(camera_id))

        # Route the lens-corrected frame back into the camera thread's
        # published display frame: get_last_frame consumers (display tile,
        # ROI helpers, pose probe frame) must share the bus's undistorted
        # coordinate space, or zones drawn on the display don't match
        # where the tracker sees the animal on a calibrated camera.
        try:
            cam_thread = self.video_manager.cameras.get(camera_id)
            if cam_thread is not None:
                bus.set_corrected_frame_cb(cam_thread.publish_corrected_frame)
        except Exception as e:
            logger.debug("corrected-frame routing (cam %s): %s", camera_id, e)

        # Register the box on the bus + subscribe all sinks for this box.
        self._registered_boxes.add(setup_id)
        bus.register_box(setup_id)
        self._subscribe_sinks_for_box(bus, setup_id)

        self._ensure_tick_running()
        return True

    def disconnect_camera(self, setup_id) -> None:
        cam_id = self.video_manager.box_camera_map.get(setup_id)
        # Unsubscribe sinks.
        for unsub in self._box_unsubs.pop(setup_id, []):
            try:
                unsub()
            except Exception:
                pass
        # Disable this box's inference too (symmetric with unregister_box).
        # Without this the pose/tracker sinks keep the box in their enabled
        # sets, so a later camera reconnect would silently resume inference the
        # operator didn't re-arm.
        self.pose.disable_for_box(setup_id)
        self.tracker.disable_for_box(setup_id)
        # Reap a still-open encoder, symmetric with unregister_box. Only
        # operant overrode the disconnect-cleanup hook to stop recording;
        # maze had no override, so disconnecting a camera mid-recording
        # left the ffmpeg child + TrackingWriter alive until process exit.
        _rec = self.recorder.stop_recording(setup_id)
        if _rec is not None:
            logger.warning(
                "Box %s: camera disconnected while recording, closing the "
                "encoder", setup_id)

            def _reap(r=_rec, b=setup_id):
                try:
                    r.stop_recording()
                    r.close()
                except Exception as e:
                    logger.error(
                        "disconnect_camera reap encoder (box=%s): %s", b, e)
            threading.Thread(target=_reap, daemon=True,
                             name=f"ReapEncoder-{setup_id}").start()
        self.video_manager.stop_camera(setup_id)
        if cam_id is None:
            return
        bus = self._buses.get(cam_id)
        if bus is None:
            return
        bus.unregister_box(setup_id)
        if not bus.known_box_ids():
            self._buses.pop(cam_id, None)
            # Forget the closed camera's liveness state, a lingering
            # "dead" entry made a clean reconnect open with a spurious
            # "Camera recovered" banner, and entries grew unbounded over
            # long sessions with box cycling.
            self._cam_liveness.pop(cam_id, None)
            self._cam_shortfall.pop(cam_id, None)

    def bind_box_to_camera(self, setup_id, camera_id) -> bool:
        """Attach a box to an ALREADY-open camera, without reopening it.

        This is the second-and-later box of a shared / CCTV camera: the device
        is streaming for the first box, and the extra boxes only need their
        own slice of it. Everything the attach involves, the box→camera map,
        box registration, the bus registration and the per-box sink
        subscriptions, is pipeline state, so it belongs here rather than
        being assembled by each caller.

        Returns False when that camera is not open, so the caller can fall
        back to a real connect.
        """
        bus = self._buses.get(camera_id)
        if bus is None:
            return False

        # Disconnect-first when re-pointing the box at a DIFFERENT camera,
        # the same guard ``connect_camera`` carries. Overwriting the map
        # entry alone stranded the previous camera: ``stop_camera`` resolves
        # the device through ``box_camera_map[setup_id]``, so once that entry
        # moved, the old CameraThread could never be stopped, its device
        # handle stayed claimed and its bus still listed this box. Nothing
        # released it until the process exited.
        prev_cam = self.video_manager.box_camera_map.get(setup_id)
        if prev_cam is not None and str(prev_cam) != str(camera_id):
            self.disconnect_camera(setup_id)

        self.video_manager.box_camera_map[setup_id] = camera_id
        self.register_box(setup_id)
        bus.register_box(setup_id)
        self._subscribe_sinks_for_box(bus, setup_id)
        return True

    def get_bus(self, camera_id) -> Optional[FrameBus]:
        """Return the FrameBus for ``camera_id`` if a camera is connected.

        Used by external surfaces (zone editor, ROI editor, calibration
        dialogs) that want to subscribe to the live frame stream without
        opening a second VideoCapture on the same device.

        The id may arrive in either form, the index a project saved or the
        identity the picker offers, and a miss here is what sends those
        surfaces off to open a second handle on a busy device.
        """
        bus = self._buses.get(camera_id)
        if bus is not None:
            return bus
        try:
            from source.video.cameras.identity import matching_key
            key = matching_key(camera_id, self._buses.keys())
        except Exception:
            key = None
        return self._buses.get(key) if key is not None else None

    def camera_id_for_box(self, setup_id):
        """The camera this box is bound to, or None."""
        return self.video_manager.box_camera_map.get(setup_id)

    def set_segment_config(self, seg_cfg: Optional[dict]) -> None:
        """Install (or clear) the shared-camera ROI segment table."""
        self._segment_config = dict(seg_cfg) if seg_cfg else None

    @property
    def segment_config(self) -> Optional[dict]:
        return self._segment_config

    def set_pose_enhancer(self, setup_id: int, enhancer) -> None:
        """Attach (or detach with None) a box's tracking enhancer,
        the public route; the GUI must not poke PoseSink internals."""
        self.pose.set_enhancer(setup_id, enhancer)

    def pose_model_info(self):
        """``(body_parts, confidence_threshold)`` of the loaded pose
        model ([], default when none), read-only view for the GUI."""
        return self.pose.model_info()

    def pose_resolved_backend(self) -> str:
        """What the loaded pose model is actually running as, after any
        fallback (e.g. ``sleap_nn:tensorrt/fp16``). Empty when none."""
        return self.pose.resolved_backend()

    def camera_connect_state(self, setup_id) -> str:
        """Public read of a box's camera-connect progress, the GUI polls
        this instead of reaching into CameraThread internals.

        'unbound'    no camera mapped to this box
        'connecting' the camera thread is still opening the device
                     (drivers can retry for seconds on slow opens)
        'connected'  the device reported up (or the box is bound to an
                     already-open shared camera)
        'failed'     the thread gave up or died without connecting
        """
        cam_id = self.video_manager.box_camera_map.get(setup_id)
        if cam_id is None:
            return "unbound"
        thread = self.video_manager.cameras.get(cam_id)
        if thread is None or thread.connected:
            # No thread: the bind-to-open-camera path, the device is the
            # sharing camera's, already up.
            return "connected"
        # Race guard: ``is_alive()`` can be True while the thread is
        # exiting. A failure path always sets ``connection_checked`` via
        # ``_mark_failed_connection``, so require it NOT set to distinguish
        # "alive and retrying" from "alive but about to die".
        if thread.is_alive() and not thread.connection_checked.is_set():
            return "connecting"
        return "failed"

    def unbind_box_camera(self, setup_id) -> None:
        """Drop the box→camera mapping (after a failed connect) so stale
        routing doesn't linger."""
        self.video_manager.box_camera_map.pop(setup_id, None)

    def set_capture_defaults(self, *, target_fps=None, frame_strategy=None,
                             target_resolution=None) -> None:
        """Set the defaults applied to the NEXT camera open.

        These do not retune a running camera, pass a ``camera_id`` to
        ``VideoManager.set_target_fps`` for that. Exposed here so the GUI stops
        driving the manager directly through the ``video_manager`` alias:
        capture configuration is pipeline state, and a caller that sets it
        behind the Pipeline's back can leave the two disagreeing about what
        the next connect will do.
        """
        vm = self.video_manager
        if target_fps is not None:
            vm.set_target_fps(int(target_fps))
        if frame_strategy is not None:
            vm.set_frame_strategy(frame_strategy)
        if target_resolution is not None:
            vm.set_target_resolution(*target_resolution)

    def force_color_for_box(self, setup_id) -> bool:
        """Take a box's camera out of grayscale mid-stream. Returns whether it
        *was* grayscale.

        Pose inference needs 3-channel BGR. The decision is the GUI's;
        performing it belongs here, live capture state is mutated only by
        the pipeline that owns it, never through the ``video_manager``
        alias from outside.
        """
        cam_id = self.camera_id_for_box(setup_id)
        if cam_id is None:
            return False
        cam_thread = self.video_manager.cameras.get(cam_id)
        if cam_thread is None:
            return False
        was_gray = bool(getattr(cam_thread, "grayscale", False))
        if was_gray:
            cam_thread.grayscale = False
        return was_gray

    def set_camera_flip(self, camera_id: Any, *, horizontal: bool = None,
                        vertical: bool = None) -> None:
        """Persist a camera's flip and apply it to the live capture thread.

        The flip is capture-side on purpose (see ``CameraConfig``), so it must
        reach the running thread as well as the stored config, otherwise the
        operator ticks the box, the preview keeps showing the mirrored frame,
        and only a reconnect fixes it.

        The thread reads the flags once per batch, so assigning them takes
        effect on the next batch with no restart and no dropped frames.
        """
        cam_id = str(camera_id)
        updates = {}
        if horizontal is not None:
            updates["flip_horizontal"] = bool(horizontal)
        if vertical is not None:
            updates["flip_vertical"] = bool(vertical)
        if not updates:
            return
        self.update_camera_config(cam_id, **updates)
        cam_thread = self.video_manager.cameras.get(cam_id)
        if cam_thread is None:
            return
        for attr, value in updates.items():
            setattr(cam_thread, attr, value)

    # ──────────────────────────────────────────────────────────────────
    # Lens / FOV correction
    # ──────────────────────────────────────────────────────────────────

    def refresh_lens_correction(self, camera_id: Any = None) -> None:
        """Re-read lens calibration from disk and retry correction on live
        cameras. Call after the calibration wizard saves a profile or a toggle
        changes, so the change takes effect without a reconnect. ``None``
        refreshes every camera.
        """
        self.lens.invalidate(camera_id)
        buses = ([self._buses[camera_id]] if camera_id in self._buses
                 else self._buses.values() if camera_id is None else [])
        for bus in buses:
            bus.reset_lens_failed()

    def set_lens_correction_enabled(self, enabled: bool) -> None:
        """Global on/off for lens correction across all cameras (per-camera
        enable still lives in the store)."""
        self.lens.set_enabled(enabled)

    # ──────────────────────────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────────────────────────

    def start_recording(self, setup_id: int, *,
                        recorder: Any,
                        tracking_writer: Any = None,
                        annotate_callback: Optional[Callable] = None,
                        pycboard: Any = None) -> None:
        # Bind the box's board so RecorderSink can stamp each frame / pose
        # row with the raw ``pycboard.timestamp``. Fall back to the
        # already-registered board when the caller doesn't pass one.
        self.recorder.start_recording(
            setup_id, recorder=recorder, tracking_writer=tracking_writer,
            annotate_callback=annotate_callback,
            pycboard=pycboard if pycboard is not None
            else self._pycboards.get(setup_id),
        )

    def stop_recording(self, setup_id: int):
        """Detach the box's recorder from the sink and return the VideoRecorder
        so the caller reaps its ffmpeg child."""
        return self.recorder.stop_recording(setup_id)

    # ──────────────────────────────────────────────────────────────────
    # Pose
    # ──────────────────────────────────────────────────────────────────

    def configure_pose_model(self, *, tracker_type: str, model_path: str,
                             probe_frame, resize_factor: float = 1.0,
                             body_parts: Optional[list] = None,
                             confidence: float = 0.5,
                             sleap_opts: Optional[dict] = None,
                             dlc_opts: Optional[dict] = None,
                             colour_mode: str = "auto",
                             input_mode: str = "auto",
                             input_wh: Optional[tuple] = None,
                             crop_opts: Optional[dict] = None):
        return self.pose.configure_model(
            tracker_type=tracker_type, model_path=model_path,
            probe_frame=probe_frame, resize_factor=resize_factor,
            body_parts=body_parts, confidence=confidence,
            dlc_opts=dlc_opts, colour_mode=colour_mode,
            sleap_opts=sleap_opts, input_mode=input_mode,
            input_wh=input_wh, crop_opts=crop_opts,
            # The sink cannot see how big the rig is at Init, pose is enabled
            # afterwards, so the engine was built for one box however many
            # there were, and that size is also the ceiling it may never
            # exceed. The Pipeline is where the box count actually lives.
            n_boxes=self.n_pose_boxes(),
        )

    def n_pose_boxes(self) -> int:
        """How many boxes this rig will actually run pose on.

        The engine's batch is fixed when it is built and can only be raised by
        building it again, so this number has to be the rig's, decided once at
        init rather than discovered when a run starts. Counting every
        REGISTERED box instead over-sizes a rig where only some boxes track,
        and for DeepLabCut the count also picks which engine runs, so an
        over-count silently chooses the batched runner for a rig that does not
        need it.

        Falls back to the registered count when no box has committed a config
        yet, which is the first-ever init on a fresh project.
        """
        n = 0
        for tc in self._tracking_configs.values():
            if not getattr(tc, "online_tracking_enabled", True):
                continue
            if tc.has_dlc():
                n += 1
        return n or len(self._registered_boxes)

    def pose_fingerprint(self):
        """Settings the loaded pose model was built from, or ``None``."""
        return self.pose.fingerprint()

    def enable_pose(self, setup_id: int, *, zone_lookup: Any = None) -> None:
        self.pose.enable_for_box(setup_id, zone_lookup=zone_lookup)

    def disable_pose(self, setup_id: int) -> None:
        self.pose.disable_for_box(setup_id)

    def has_pose_model(self) -> bool:
        return self.pose.has_model()

    def set_pose_n_instances(self, n: int) -> None:
        """Configure how many parallel pose-model copies live on the GPU.

        ``n=1`` (default) shares one session; ``n>1`` runs ``n``
        independent copies in parallel via ``MultiInstanceInferenceBackend``.
        Caller (GUI tracking config) typically passes
        ``ceil(N_pose_boxes / 4)`` just before Init DLC.
        """
        self.pose.set_n_instances(n)

    # ──────────────────────────────────────────────────────────────────
    # Blob tracker
    # ──────────────────────────────────────────────────────────────────

    def enable_blob_tracking(self, setup_id: int) -> None:
        self.tracker.enable_for_box(setup_id)

    def disable_blob_tracking(self, setup_id: int) -> None:
        self.tracker.disable_for_box(setup_id)

    # ──────────────────────────────────────────────────────────────────
    # MCU push policy
    # ──────────────────────────────────────────────────────────────────

    def configure_push_zones(self, setup_id: int, zones: list, **kw) -> None:
        self.push.configure_zones(setup_id, zones, **kw)

    def configure_push_tracking(self, setup_id: int, cfg: dict) -> None:
        self.push.configure_tracking(setup_id, cfg)

    def reset_push_policy(self, setup_id: Optional[int] = None) -> None:
        self.push.reset(setup_id)

    def push_stats(self, setup_id: int):
        """Live MCU-push counters for one box (coords / zone events / triggers
        / last-push age), or None when the box has no push policy yet. The
        observable answer to 'is tracking actually reaching the board?'."""
        return self.push.push_stats(setup_id)

    # ──────────────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────────────

    # Display is main-thread polled (see MainWindowBase paint timer), not
    # relayed through a sink; there is no add/remove_display_callback.

    # ──────────────────────────────────────────────────────────────────
    # Health alarms
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _subscribe(subs: List, callback) -> Callable[[], None]:
        """Append ``callback`` to a subscriber list and return its
        unsubscribe closure. All on_* methods share this shape."""
        subs.append(callback)

        def _unsub():
            if callback in subs:
                subs.remove(callback)
        return _unsub

    def on_health(self, callback: HealthCallback) -> Callable[[], None]:
        return self._subscribe(self._on_health, callback)

    def on_pose_result(self, callback: PoseResultCallback) -> Callable[[], None]:
        return self._subscribe(self._on_pose_result, callback)

    def on_tracker_result(self, callback: TrackerResultCallback) -> Callable[[], None]:
        return self._subscribe(self._on_tracker_result, callback)

    def on_pose_failed(self, callback: PoseFailedCallback) -> Callable[[], None]:
        """Subscribe to pose silent-failure events.

        ``callback(box_id, reason, streak)`` runs on the inference worker
        thread. The GUI normally wraps this in ``QtBridge`` so the
        notification reaches the main thread before touching widgets.
        """
        return self._subscribe(self._on_pose_failed, callback)

    def on_trigger_frame(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        """Subscribe to per-frame ``TriggerFrame`` events (annotation + plot).
        Fires on the inference worker; wrap via QtBridge for the GUI."""
        return self._subscribe(self._on_trigger_frame, callback)

    def _fanout_trigger_frame(self, tf) -> None:
        self._fanout(self._on_trigger_frame, "trigger-frame fanout", tf)

    # ──────────────────────────────────────────────────────────────────
    # Pipeline tick
    # ──────────────────────────────────────────────────────────────────

    def _ensure_tick_running(self) -> None:
        if self._tick_running:
            return
        self._tick_running = True
        self._tick_thread = threading.Thread(
            target=self._tick_loop, name="Pipeline-tick", daemon=True
        )
        self._tick_thread.start()

    def _tick_loop(self) -> None:
        while self._tick_running:
            try:
                self._tick()
            except Exception as e:
                logger.error("Pipeline tick error: %s", e)
            time.sleep(self._tick_interval_s)

    def _tick(self) -> None:
        """Drain each connected camera and publish EVERY captured frame.

        ``CameraThread.drain_recording_buffer()`` returns every CameraFrame
        produced since the last drain, in capture order. Each is published
        with its own ``cam_frame_id``, ``capture_host_ns``, and
        ``capture_wall`` (the camera's canonical capture identity). Per-sink
        drop policies handle the cadence:
          - RecorderSink LOSSLESS → keeps all.
          - PoseSink DROP_NEWEST → drops backlog when inference is busy.
          - TrackerSink DROP_OLDEST → keeps recent for blob tracking.
        """
        # One camera snapshot per tick, shared with the liveness watchdog so
        # the drain loop and watchdog don't each rebuild the list.
        cameras = list(self.video_manager.cameras.items())
        # Watchdog: surface camera stall / loss / recovery to the GUI.
        self._observe_camera_liveness(cameras)
        self._observe_camera_shortfall(cameras)
        for cam_id, thread in cameras:
            try:
                if not getattr(thread, "connected", False):
                    continue
                if not thread.isRunning():
                    continue
                # Drain ALL captured frames since last tick.
                cf_list = thread.drain_recording_buffer()
                if not cf_list:
                    continue
                bus = self._buses.get(cam_id)
                if bus is None:
                    # Bus torn down (camera unregistered) while frames were
                    # still in flight, account the discard instead of losing
                    # the whole drained batch silently.
                    drop_log.record("camera_tick",
                                    reason=f"bus_torn_down_x{len(cf_list)}")
                    continue
                for cf in cf_list:
                    img = getattr(cf, "image", None)
                    if not isinstance(img, np.ndarray) or img.size == 0:
                        drop_log.record(
                            "camera_tick",
                            frame_idx=getattr(cf, "cam_frame_id", None),
                            capture_ts=(getattr(cf, "capture_host_ns", 0) or 0)
                            / 1e9,
                            reason="malformed_frame")
                        continue
                    # Timing spine: capture→poll (drain) latency, per frame.
                    self.latency.record_from_ns(
                        "capture_to_poll", cf.capture_host_ns)
                    # Reuse the CameraFrame the camera thread already built,
                    # publish_frame fills in is_shared / box_ids for this bus.
                    bus.publish_frame(cf)
                    self.n_published += 1
            except Exception as e:
                logger.error("Pipeline tick (cam=%s) error: %s", cam_id, e)

    # ──────────────────────────────────────────────────────────────────
    # Timing spine (observational)
    # ──────────────────────────────────────────────────────────────────

    def latency_snapshot(self) -> Dict[str, object]:
        """Per-stage latency stats + total_ms (see LatencyBudget.snapshot).

        Cheap; safe to poll from the GUI thread. All four stages are wired:
        ``capture_to_poll`` and ``poll_to_infer`` (capture → inference done)
        on the camera path, ``infer_to_push`` and ``push_to_wire`` on the MCU
        pusher, so ``total_ms`` is the full capture→wire closed-loop budget
        once a box is tracking and pushing to a board.
        """
        return self.latency.snapshot()

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _subscribe_sinks_for_box(self, bus: FrameBus, setup_id: int) -> None:
        """Wire RecorderSink + PoseSink + TrackerSink for box.

        Idempotent: drop any existing subscriptions for this box first so a
        reconnect doesn't append duplicate callbacks (FrameBus.on_box_frame
        removes by identity, so a later unsub would rip out both)."""
        for unsub in self._box_unsubs.pop(setup_id, []):
            try:
                unsub()
            except Exception:
                pass
        unsubs = self._box_unsubs.setdefault(setup_id, [])
        # Tell the queues what rate they are absorbing. Both are bounded in
        # SECONDS, and a sink is constructed before any camera exists, so
        # until this runs they hold whatever their constructor assumed.
        self._tell_sinks_the_rate(setup_id)
        unsubs.append(bus.on_box_frame(setup_id, self.recorder.submit))
        unsubs.append(bus.on_box_frame(setup_id, self.pose.submit))
        unsubs.append(bus.on_box_frame(setup_id, self.tracker.submit))

    def _tell_sinks_the_rate(self, setup_id: int) -> None:
        """Re-bound the seconds-budgeted sink queues for this box's camera.

        The DELIVERED rate when the camera measured one, because that is what
        the queues actually have to absorb, and the requested rate otherwise.
        Taken as a maximum across boxes: one queue serves them all, so the
        fastest camera sets the depth.
        """
        try:
            cfg = self.video_manager.resolved_settings_for_box(setup_id)
            fps = float(getattr(cfg, "delivered_fps", 0) or 0)
            if fps <= 0:
                fps = float(getattr(cfg, "target_fps", 0) or 0)
        except Exception as e:
            logger.debug("rate hint for box %s: %s", setup_id, e)
            return
        if fps <= 0:
            return
        self._sink_rate_hint = max(getattr(self, "_sink_rate_hint", 0.0), fps)
        for sink in (self.recorder, self.tracker):
            try:
                sink.set_rate_hint(self._sink_rate_hint)
            except Exception as e:
                logger.debug("rate hint for %s: %s", getattr(sink, "name", "?"), e)

    def _observe_camera_liveness(self, cameras=None) -> None:
        """Detect camera frame-flow transitions and banner every box on the
        affected camera. Fans a health message only on a state CHANGE, so a
        steadily-streaming (or steadily-down) camera stays quiet.
        ``cameras`` is the tick's ``(cam_id, thread)`` snapshot; standalone
        callers may omit it."""
        vm = self.video_manager
        if cameras is None:
            cameras = list(vm.cameras.items())
        for cam_id, thread in cameras:
            prev = self._cam_liveness.get(cam_id)
            # A capture thread that DIED mid-stream leaves ``_liveness`` frozen
            # at its last value (usually "streaming"), so a plain state read
            # would never see the failure. Treat "was live, thread now stopped
            # / disconnected" as a real death event so the box isn't silently
            # dark. Only fires on a live→dead transition (not at startup, and
            # not on an intentional disconnect, which removes the box mapping
            # below so no banner is fanned).
            running = thread.isRunning() if hasattr(thread, "isRunning") else True
            connected = getattr(thread, "connected", True)
            if not running or not connected:
                if prev not in ("streaming", "stalled", "reconnecting"):
                    continue
                state = "dead"
            else:
                state = getattr(thread, "_liveness", None)
                if state is None:
                    continue
            if state == prev:
                continue
            self._cam_liveness[cam_id] = state
            if state == "dead":
                reason = "Camera disconnected, capture thread stopped"
            elif state == "stalled":
                reason = "Camera stopped delivering frames"
            elif state == "reconnecting":
                reason = "Camera lost, reconnecting…"
            elif state == "streaming" and prev in ("stalled", "reconnecting", "dead"):
                reason = "Camera recovered"
            else:
                continue   # e.g. initial starting→streaming: not a real event
            for setup_id, mapped in list(vm.box_camera_map.items()):
                if mapped == cam_id:
                    self._notify_health(int(setup_id), reason)

    def _fastest_backend_for(self, cam_cfg) -> str:
        """Backend name that best serves this camera's selected mode, or "".

        Only consulted when the operator has not pinned one. Returns "" when
        the measurements do not actually prefer a door, so a camera whose
        backends tie keeps opening the way it always has.
        """
        from source.video.cameras.probe import Variant, best_for
        variants = [Variant(backend=str(name), modes=list(modes or []))
                    for name, modes in
                    (getattr(cam_cfg, "probed_variants", None) or {}).items()]
        if len(variants) < 2:
            return ""
        wh = cam_cfg.selected_resolution or (0, 0)
        fps = float(cam_cfg.selected_fps or 30)
        pick = best_for(variants, int(wh[0]), int(wh[1]), fps)
        if not pick:
            return ""
        name, mode = pick
        logger.info(
            "camera %s: opening through %s, measured %.1f fps at %dx%d "
            "against %s", cam_cfg.camera_id, name, mode[2], mode[0], mode[1],
            ", ".join(f"{v.backend} {v.best_fps():.1f}" for v in variants))
        return str(name)

    def _observe_camera_shortfall(self, cameras=None) -> None:
        """Banner a camera that is not delivering what it was asked for.

        The stack cannot detect this on its own: every property between the
        request and the sensor answers with the request, so a camera asked for
        720p at 30 fps reports 30 while delivering 10. Only the measured rate
        knows, and it was already being measured, for a display readout, and
        compared against nothing.

        Fires once per camera per change, like the liveness observer above,
        because the condition is steady rather than an event: repeating it
        every tick would bury the banner it belongs in.
        """
        vm = self.video_manager
        if cameras is None:
            cameras = list(vm.cameras.items())
        for cam_id, thread in cameras:
            try:
                short = thread.shortfall(cam_id)
            except Exception as e:
                logger.debug("shortfall(%s) failed: %s", cam_id, e)
                continue
            msg = short.message() if short is not None else ""
            if msg == self._cam_shortfall.get(cam_id, ""):
                continue
            self._cam_shortfall[cam_id] = msg
            if not msg:
                continue
            logger.warning(msg)
            for setup_id, mapped in list(vm.box_camera_map.items()):
                if mapped == cam_id:
                    self._notify_health(int(setup_id), msg)

    def _fanout(self, subs: List, label: str, *args) -> None:
        """Relay a sink event to the external subscribers. Copies the list
        (subscriber lists mutate in place) and log-and-continues per cb."""
        notify_subscribers(list(subs), *args, log=logger,
                           label=f"Pipeline {label}")

    def _notify_health(self, setup_id: int, reason: str) -> None:
        self._fanout(self._on_health, "health", setup_id, reason)

    def _fanout_pose_result(self, setup_id, cam_frame_id, pose_array,
                            location, speed, zones_by_body_part,
                            raw_pose_dict, capture_host_ns: int = 0,
                            forecast_coords=None, infer_done_ns: int = 0,
                            **_) -> None:
        # ``notify_subscribers`` fans the PoseSink's full kwarg set to EVERY
        # subscriber, so we must accept (and ignore) ``infer_done_ns`` etc.,
        # without this the pose fanout raised TypeError every frame and the
        # external GUI-overlay subscribers never ran. The overlay keeps its
        # original raw-position signature.
        self._fanout(self._on_pose_result, "pose-result fanout",
                     setup_id, cam_frame_id, pose_array, location, speed,
                     zones_by_body_part, raw_pose_dict)

    def _fanout_tracker_result(self, setup_id, cam_frame_id, centroid,
                               location, speed, zones_by_body_part,
                               position, capture_host_ns: int = 0) -> None:
        # ``capture_host_ns`` is consumed by the recorder (fw stamping); the
        # external GUI-overlay subscribers keep their original signature.
        self._fanout(self._on_tracker_result, "tracker-result fanout",
                     setup_id, cam_frame_id, centroid, location, speed,
                     zones_by_body_part, position)

    def _fanout_pose_failed(self, setup_id: int, reason: str, streak: int) -> None:
        self._fanout(self._on_pose_failed, "pose-failed fanout",
                     setup_id, reason, streak)

    # ──────────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────────

    def release_all_cameras(self) -> None:
        """Release every camera and drop every bus, keeping the Pipeline alive.

        The sweep after removing all boxes: per-box ``disconnect_camera``
        has already run, so this only catches a device no box still claims.
        Tearing cameras down through the ``video_manager`` alias instead
        would leave the Pipeline's buses and sink subscriptions pointing at
        dead threads, release always goes through here.

        Distinct from ``shutdown``, which additionally stops the sink worker
        threads and the tick, after which the Pipeline cannot be reused.
        """
        try:
            self.video_manager.cleanup()
        except Exception as e:
            logger.debug("VideoManager cleanup error: %s", e)
        self._buses.clear()
        self._box_unsubs.clear()
        # getattr: ownership tests drive this on a partially-constructed
        # Pipeline that never ran __init__.
        getattr(self, "_cam_liveness", {}).clear()

    def shutdown(self) -> None:
        self._tick_running = False
        t = self._tick_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._tick_thread = None

        # Reap any per-box encoder still open, shutdown is the fallback
        # when the GUI's recording_setups bookkeeping and the sink's
        # registry disagree; without this the box's ffmpeg child survived
        # to process exit (unplayable mp4, orphaned process).
        try:
            for bid in list(getattr(self.recorder, "_recorders", {}) or {}):
                try:
                    rec = self.recorder.stop_recording(bid)
                    if rec is not None:
                        rec.stop_recording()
                    logger.warning(
                        "Pipeline.shutdown: reaped still-open recorder for "
                        "box %s", bid)
                except Exception as e:
                    logger.warning("shutdown recorder reap (box %s): %s",
                                   bid, e)
        except Exception:
            pass

        for sink in (self.recorder, self.pose, self.tracker):
            try:
                sink.stop()
            except Exception:
                pass

        try:
            self.pose.shutdown()
        except Exception:
            pass

        try:
            self.video_manager.cleanup()
        except Exception as e:
            logger.debug("VideoManager cleanup error: %s", e)

        self._buses.clear()
        self._box_unsubs.clear()
        logger.info("Pipeline shut down")

    # ══════════════════════════════════════════════════════════════════
    # Central config registries
    # ══════════════════════════════════════════════════════════════════
    #
    # Both operant and maze main_windows call the methods below to
    # read/write the authoritative per-camera and per-box configs. Dialogs
    # are thin views over these, they don't keep parallel state.
    #
    # Persistence: settings.json under
    #   "camera"   -> "camera_configs"   -> {camera_id: {...}}
    #   "tracking" -> "tracking_configs" -> {box_id (str): {...}}
    # JSON requires str keys; we coerce when serialising / deserialising.

    # ── Camera config ────────────────────────────────────────────────

    def get_camera_config(self, camera_id: Any):
        """Return the live ``CameraConfig`` for this camera_id, creating
        an empty one on first access. Callers can mutate the returned
        object directly (e.g. ``cfg.probed_modes = [...]``), those edits
        are visible to subsequent reads via this same registry."""
        from source.video.framebus.types import CameraConfig
        cam_id = str(camera_id)
        cfg = self._camera_configs.get(cam_id)
        if cfg is None:
            cfg = CameraConfig(camera_id=cam_id)
            self._camera_configs[cam_id] = cfg
        return cfg

    @staticmethod
    def _update_cfg_fields(cfg, fields):
        """Setattr each known field on a config object; unknown keys are
        ignored. Shared by the camera + tracking update methods."""
        for k, v in fields.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def update_camera_config(self, camera_id: Any, **fields):
        """Set fields on the camera config and return the updated object.

        Convenient for dialog-side ``controller.update_camera_config(
        cam_id, selected_resolution=(1280, 720), selected_fps=30)`` calls.
        """
        return self._update_cfg_fields(self.get_camera_config(camera_id), fields)

    def describe_camera_features(self, camera_id: Any) -> list:
        """Feature descriptors read from the live camera.

        Empty when the camera is not open or its backend does not introspect,
        which is the signal for the UI to fall back to its static rows.
        """
        vm = getattr(self, "video_manager", None)
        if vm is None or not hasattr(vm, "describe_camera_features"):
            return []
        return vm.describe_camera_features(camera_id)

    def set_camera_feature(self, camera_id: Any, key: str, value) -> bool:
        """Apply a feature now if the camera is streaming; always persist it.

        The value is stored on the camera config either way, so a setting made
        while disconnected is applied at the next connect. Returns True when it
        also reached the hardware immediately.
        """
        cfg = self.get_camera_config(camera_id)
        features = dict(getattr(cfg, "features", None) or {})
        features[str(key)] = value
        self._update_cfg_fields(cfg, {"features": features})

        vm = getattr(self, "video_manager", None)
        if vm is None or not hasattr(vm, "set_camera_feature"):
            return False
        return vm.set_camera_feature(camera_id, key, value)

    def all_camera_configs(self):
        """Snapshot of the registry, for diagnostics / save-all flows."""
        return dict(self._camera_configs)

    def install_camera_configs(self, raw_configs) -> int:
        """Replace every camera's config from a ``{camera_id: to_json()}``
        dict (the shape used by project save and the Save All bundle).
        Returns the count successfully installed.

        Tolerant of malformed entries, bad ones are skipped with a log
        line.
        """
        from source.video.framebus.types import CameraConfig
        n = 0
        for cid, raw in (raw_configs or {}).items():
            try:
                cfg = CameraConfig.from_json(raw, camera_id=str(cid))
                cfg.clamp_fps_to_options()
                self._camera_configs[str(cid)] = cfg
                n += 1
            except Exception as e:
                logger.warning(
                    "install_camera_configs: skipping %r: %s", cid, e
                )
        return n

    def install_tracking_configs(self, raw_configs) -> int:
        """Replace every box's tracking config from a ``{box_id_str: to_json()}``
        dict (project-save shape). Returns count installed.

        Convenience wrapper: builds a full :class:`TrackingConfig` per
        entry via :meth:`TrackingConfig.from_json` so dataclass
        defaults populate fields the saved dict omits. Dialog Apply
        uses :meth:`update_tracking_config` for granular field updates
        on an existing TC, the two paths are distinct on purpose
        (full-replace vs partial-merge).
        """
        from source.video.framebus.types import TrackingConfig
        n = 0
        for bid_str, raw in (raw_configs or {}).items():
            try:
                bid = int(bid_str)
            except (TypeError, ValueError):
                logger.warning(
                    "install_tracking_configs: non-int box_id %r", bid_str
                )
                continue
            try:
                self._tracking_configs[bid] = TrackingConfig.from_json(
                    raw, setup_id=bid
                )
                n += 1
            except Exception as e:
                logger.warning(
                    "install_tracking_configs: skipping box %s: %s", bid, e
                )
        return n

    # ── Tracking config ──────────────────────────────────────────────

    def get_tracking_config(self, setup_id: int):
        """Return the live ``TrackingConfig`` for this box_id, creating
        an empty one on first access."""
        from source.video.framebus.types import TrackingConfig
        bid = int(setup_id)
        cfg = self._tracking_configs.get(bid)
        if cfg is None:
            cfg = TrackingConfig(setup_id=bid)
            self._tracking_configs[bid] = cfg
        return cfg

    def update_tracking_config(self, setup_id: int, **fields):
        return self._update_cfg_fields(self.get_tracking_config(setup_id), fields)

    def all_tracking_configs(self):
        return dict(self._tracking_configs)

    # ── Apply: push central config → sinks (used at connect / config) ─

    def apply_tracking_config(self, setup_id: int) -> None:
        """Push the central ``TrackingConfig`` into PoseSink + MCUPusher.

        Idempotent, safe to call repeatedly (on reconnect or whenever the
        tracking config dialog applies changes). Forwards each field to the
        sink that owns its runtime state.

        Side effects:
          - Per-box rotation toggle on PoseSink
          - MCUPusher coord_mapping + triggers from derived helpers
          - MCUPusher zone-config rebuild (zone_changed event semantics)
        """
        cfg = self.get_tracking_config(setup_id)
        # Rotation, off by default; only forwards keypoint pair when on.
        try:
            self.pose.set_rotation(
                setup_id,
                enabled=cfg.rotation_enabled,
                kp_pair=cfg.rotation_keypoints,
            )
        except Exception as e:
            logger.debug("apply_tracking_config: set_rotation failed (box=%d): %s", setup_id, e)
        # Crop steering, how a result is read, not what the network is fed,
        # so it applies live rather than waiting for the next Init.
        try:
            self.pose.set_crop_opts(
                conf_min=float(getattr(cfg, "pose_crop_conf_min", 0.20)),
                good_min=int(getattr(cfg, "pose_crop_good_min", 3)),
                reacquire=bool(getattr(cfg, "pose_crop_reacquire", True)))
        except Exception as e:
            logger.debug("apply_tracking_config: set_crop_opts failed "
                         "(box=%d): %s", setup_id, e)
        # Pose n-instances, global PoseSink knob. The sink needs this
        # explicit call to spin up additional inference workers.
        try:
            n = int(getattr(cfg, "pose_n_instances", 1))
            if n >= 1:
                self.pose.set_n_instances(n)
        except Exception as e:
            logger.debug("apply_tracking_config: set_n_instances failed (box=%d): %s", setup_id, e)
        # Pose centroid body part, mirrors the zone_change_body_part picker
        # so PoseSink reports that body part's coords/zone as the "centroid"
        # instead of whichever keypoint is first confident.
        try:
            self.pose.set_centroid_body_part(
                setup_id, str(getattr(cfg, "zone_change_body_part", "centroid")
                            or "centroid"))
        except Exception as e:
            logger.debug("apply_tracking_config: set_centroid_body_part failed (box=%d): %s",
                         setup_id, e)
        # Confidence is a post-inference cutoff, not part of the model cache
        # key, so it applies to the running model. Sent here rather than left
        # to configure_model so that changing it does not need the model torn
        # down and rebuilt, see PoseSink.set_confidence.
        try:
            self.pose.set_confidence(float(cfg.confidence_threshold))
        except Exception as e:
            logger.debug("apply_tracking_config: set_confidence failed (box=%d): %s",
                         setup_id, e)
        # Push policy, ONE zone→policy translation (configure_zones,
        # honouring the dialog's push gates + picked body part), then the
        # authored Event-Triggers table merged on top (configure_tracking
        # dedups by event_name, authored wins).
        try:
            self.push.configure_zones(
                setup_id, cfg.zones,
                default_body_part=str(cfg.zone_change_body_part or "centroid"),
                push_coords=bool(cfg.push_coords_to_mcu),
                push_zone_events=bool(cfg.push_zones_to_mcu))
        except Exception as e:
            logger.debug("apply_tracking_config: configure_zones failed (box=%d): %s", setup_id, e)
        try:
            self.push.configure_tracking(setup_id, {
                "triggers":              list(getattr(cfg, "triggers", []) or []),
                # Surface the per-box gate so the intrinsic zone_changed
                # event also honours the dialog toggle (not just the
                # derived coord_mapping/triggers, which are already
                # empty when push_*_to_mcu is False).
                "push_zones_to_mcu":     bool(cfg.push_zones_to_mcu),
                # Coord gate at push time, the dialog coord-mapping merge
                # must honour the "push c.* coordinates" toggle too, not
                # only the zone-derived mapping built by configure_zones.
                "push_coords_to_mcu":    bool(cfg.push_coords_to_mcu),
                # Pose confidence for the MCU push filter. Without this the
                # pusher stays on its hardcoded 0.5 default and disagrees
                # with what PoseSink displays.
                "confidence_threshold":  float(cfg.confidence_threshold),
                # Per-frame pose event gate (OFF by default).
                "push_frame_event":      bool(getattr(cfg, "push_frame_event", False)),
                # Body part whose zone occupancy drives ``zone_changed``.
                # MCUPusher diffs only this body part between ticks so
                # the picker in the tracking dialog controls which
                # keypoint triggers the event.
                "zone_change_body_part": str(cfg.zone_change_body_part),
            })
        except Exception as e:
            logger.debug("apply_tracking_config: configure_tracking failed (box=%d): %s", setup_id, e)
        # Feature context, px/mm (for cm/s, mm thresholds) + body axis (for
        # heading / turning). The scale (px/mm) can't be derived from
        # ``cfg.zones``; those are typed Zone objects that don't carry the
        # calibration length, so it comes from ``set_features_scale`` (the GUI
        # pushes it from the same per-box scale zone it uses for the recorder's
        # px/m). Pass px_per_mm=None here so we never clobber that value; None
        # keeps the extractor on body-length units until a scale arrives.
        try:
            axis = None
            kp = getattr(cfg, "rotation_keypoints", None)
            if kp and len(kp) == 2 and all(kp):
                axis = (str(kp[0]), str(kp[1]))
            self.push.set_features_context(setup_id, axis=axis)
        except Exception as e:
            logger.debug("apply_tracking_config: set_features_context failed (box=%d): %s",
                         setup_id, e)

    def set_features_scale(self, setup_id: int, px_per_mm: Optional[float]) -> None:
        """Set the box's pixels-per-mm for the MCU trigger kinematics (cm/s,
        mm distances). The GUI derives it from the same per-box calibration
        scale zone it uses for the recorder's px/m and pushes it here, the
        typed ``TrackingConfig.zones`` don't carry the calibration length, so
        this is the authoritative source. ``None`` = body-length units."""
        try:
            self.push.set_features_context(setup_id, px_per_mm=px_per_mm)
        except Exception as e:
            logger.debug("set_features_scale failed (box=%d): %s", setup_id, e)


