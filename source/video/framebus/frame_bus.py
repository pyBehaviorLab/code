"""FrameBus, single producer fan-out + per-box segmentation.

One FrameBus per physical camera. Subscribers fall into two channels:

* ``on_camera_frame(callback)``: callback receives the raw CameraFrame
  (full resolution, no crop). Used by the zone editor and any consumer
  that needs the camera-native view. Subscribers are notified inline on
  the producer thread; they MUST be cheap (handoff to a worker queue).

* ``on_box_frame(box_id, callback)``: callback receives a BoxFrame
  derived from each CameraFrame for that box. The bus performs the
  per-box crop (passthrough when the camera is dedicated to one box).

MCU framework timestamps are NOT tagged here. ``RecorderSink`` reads the
raw ``pycboard.timestamp`` per box at frame-handling time, the bus only
fans frames out and crops per box.

The bus carries no Qt dependency. Use ``qt_bridge.QtBridge`` to relay
events to the GUI on the main thread.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from source import host_clock
from typing import Any, Callable, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING

from source.video.framebus.sink_base import notify_subscribers
from source.video.framebus.types import BoxFrame, CameraFrame

# Typing-only import, eager import would create a cycle (capture imports
# CameraFrame from us). Used as `Optional["VideoSegmentProcessor"]`.
if TYPE_CHECKING:
    from source.video.cameras.capture import VideoSegmentProcessor

logger = logging.getLogger(__name__)


CameraFrameCallback = Callable[[CameraFrame], None]
BoxFrameCallback = Callable[[BoxFrame], None]


class FrameBus:
    """Fan-out hub for one physical camera.

    Holds zero state about who's watching beyond the subscriber lists;
    the Pipeline tells the bus about box→camera mappings via
    ``register_box`` / ``unregister_box``.
    """

    def __init__(self, camera_id: Any,
                 segment_processor: Optional["VideoSegmentProcessor"] = None,
                 health_cb: Optional[Callable[[int, str], None]] = None,
                 lens: Optional[Any] = None) -> None:
        self.camera_id = camera_id
        self._segment_processor = segment_processor
        # Lens/FOV correction (LensCorrectionCache, shared across cameras).
        # Applied to the whole frame BEFORE fan-out and crop, so the raw
        # preview/zone editor and every box's ROI inherit one straight
        # coordinate space. None → no correction. A per-camera failure latch
        # disables correction for THIS camera after one remap error rather
        # than throwing on every frame.
        self._lens = lens
        self._lens_failed = False
        # Optional per-box health surfacing (Pipeline._notify_health). Used to
        # raise a banner when a box's ROI can't be extracted so the box isn't
        # silently blacked out of the whole pipeline.
        self._health_cb = health_cb
        # Boxes already alarmed for a bad ROI, so the banner fires once per
        # failure rather than every frame while the geometry stays invalid.
        self._roi_blackout_alarmed: set = set()
        # Subscribers, copy-on-write lists let publish_frame() hold the
        # lock only for the snapshot, never during callbacks.
        self._cam_subs: List[CameraFrameCallback] = []
        self._box_subs: Dict[int, List[BoxFrameCallback]] = {}
        # Boxes this camera serves (populated by Pipeline.register_box).
        self._boxes: set = set()
        # Derived membership snapshot, recomputed only on register/unregister
        # so publish_frame doesn't rebuild a tuple per published frame.
        self._box_ids: Tuple[int, ...] = ()
        self._is_shared: bool = False
        self._lock = threading.RLock()
        # Publishes the lens-corrected frame back to the camera thread's
        # last_frame slot (Pipeline wires CameraThread.publish_corrected_frame)
        # so display/ROI/pose-probe consumers share the corrected space.
        self._corrected_cb = None

    def set_corrected_frame_cb(self, cb) -> None:
        self._corrected_cb = cb

    # ── Subscription -----------------------------------------------------

    def on_camera_frame(self, cb: CameraFrameCallback) -> Callable[[], None]:
        """Register a raw-frame subscriber. Returns an unsubscribe handle."""
        with self._lock:
            self._cam_subs = [*self._cam_subs, cb]

        def _unsub():
            with self._lock:
                self._cam_subs = [c for c in self._cam_subs if c is not cb]
        return _unsub

    def on_box_frame(self, setup_id: int, cb: BoxFrameCallback) -> Callable[[], None]:
        """Register a per-box subscriber. Returns an unsubscribe handle."""
        with self._lock:
            self._box_subs.setdefault(setup_id, [])
            self._box_subs[setup_id] = [*self._box_subs[setup_id], cb]

        def _unsub():
            with self._lock:
                lst = self._box_subs.get(setup_id, [])
                self._box_subs[setup_id] = [c for c in lst if c is not cb]
        return _unsub

    # ── Box registration -------------------------------------------------

    def register_box(self, setup_id: int) -> None:
        """Tell the bus which boxes this camera serves.

        Pipeline calls this when a box widget binds to a camera. Bus uses
        the registry to know whether to passthrough (single box) or crop
        (shared CCTV camera).
        """
        with self._lock:
            self._boxes.add(setup_id)
            self._box_ids = tuple(self._boxes)
            self._is_shared = len(self._boxes) > 1

    def unregister_box(self, setup_id: int) -> None:
        with self._lock:
            self._boxes.discard(setup_id)
            self._box_subs.pop(setup_id, None)
            self._box_ids = tuple(self._boxes)
            self._is_shared = len(self._boxes) > 1

    def known_box_ids(self) -> Tuple[int, ...]:
        with self._lock:
            return self._box_ids

    # ── Publish ---------------------------------------------------------

    def publish_frame(self, cf: "CameraFrame") -> None:
        """Fan out an already-built CameraFrame (the camera thread's own,
        no second dataclass + dict allocation per frame). Sets is_shared /
        box_ids from this bus's registered boxes, fans it out raw, then
        derives one BoxFrame per box (cropped via segment_processor if
        shared).
        """
        with self._lock:
            cam_subs = self._cam_subs
            box_subs = self._box_subs
            boxes = self._box_ids
            is_shared = self._is_shared
            seg = self._segment_processor
            lens = self._lens

        # Lens/FOV correction, straighten the whole frame once, before any
        # subscriber (raw preview, zone editor) or per-box crop sees it, so
        # everything downstream shares one undistorted coordinate space.
        # Uncalibrated cameras resolve to None here and pay nothing.
        if lens is not None and not self._lens_failed:
            h, w = cf.image.shape[:2]
            und = lens.undistorter_for(self.camera_id, (w, h))
            if und is not None:
                try:
                    cf.image = und.apply(cf.image)
                    cf._color_cache.clear()  # rebind invalidates lazy rgb/gray
                    # Push the corrected frame back over the camera
                    # thread's published last_frame, rebinding cf.image
                    # alone left get_last_frame consumers (display, ROI
                    # helpers, pose probe) on the RAW driver frame, i.e.
                    # a second coordinate space.
                    cb = self._corrected_cb
                    if cb is not None:
                        with contextlib.suppress(Exception):
                            cb(cf.image)
                except Exception as e:
                    self._lens_failed = True
                    logger.warning("FrameBus[%s]: lens correction disabled after "
                                   "error: %s", self.camera_id, e)

        cf.is_shared = is_shared
        cf.box_ids = boxes
        frame = cf.image
        cam_frame_id = cf.cam_frame_id
        capture_host_ns = cf.capture_host_ns
        capture_wall = cf.capture_wall

        # Raw fan-out (zone editor, full preview).
        notify_subscribers(cam_subs, cf,
                           log=logger,
                           label=f"FrameBus[{self.camera_id}] camera subscriber")

        # Per-box derive-and-fan-out.
        if not boxes:
            return

        # Cache the source frame dimensions once per publish (avoids
        # frame.shape[:2] per box on a shared CCTV camera).
        fh, fw = frame.shape[:2]
        # Timing spine: stamp the poll (drain+publish) instant once per
        # publish; every derived BoxFrame carries it so sinks can measure
        # the inference stage. Observational, affects no behaviour.
        poll_host_ns = host_clock.host_ns()
        for setup_id in boxes:
            subs = box_subs.get(setup_id) or []
            if not subs:
                continue

            # Crop whenever this box has an entry in the segment_processor,
            # a single box with a declared ROI must crop too, so the
            # recorder / display / pose see the user's region, not the full
            # frame. Passthrough only when the box has NO segment configured.
            crop_origin: Optional[Tuple[int, int]] = None
            crop_size: Optional[Tuple[int, int]] = None
            box_image = frame
            if seg is not None and setup_id in seg.box_index:
                cropped = seg.extract_segment(frame, setup_id)
                if cropped is None:
                    # Invalid/out-of-bounds ROI geometry: this box would get
                    # ZERO frames (no record/pose/track/push). Surface it once
                    # so the box isn't silently dark; keep dropping until the
                    # geometry is valid again.
                    if setup_id not in self._roi_blackout_alarmed:
                        self._roi_blackout_alarmed.add(setup_id)
                        if self._health_cb is not None:
                            try:
                                self._health_cb(int(setup_id), "roi_extract_failed")
                            except Exception:
                                pass
                    continue
                self._roi_blackout_alarmed.discard(setup_id)
                box_image = cropped
                # extract_segment cached the pixel rect (percent OR pixel
                # geometry); reuse it for the writer-header origin/size
                # instead of recomputing 4 rounds per box per frame.
                rect = seg.segment_rect(setup_id, fh, fw)
                if rect is not None:
                    crop_origin = (rect[0], rect[1])
                    crop_size = (rect[2], rect[3])

            bf = BoxFrame(
                image=box_image,
                setup_id=setup_id,
                cam_frame_id=cam_frame_id,
                camera_id=self.camera_id,
                capture_host_ns=capture_host_ns,
                capture_wall=capture_wall,
                is_shared_camera=is_shared,
                poll_host_ns=poll_host_ns,
                crop_origin=crop_origin,
                crop_size=crop_size,
                # Link to parent so image_rgb / image_gray can slice the
                # shared whole-frame conversion buffer.
                _parent_camera_frame=cf,
            )
            notify_subscribers(
                subs, bf, log=logger,
                label=f"FrameBus[{self.camera_id}] box={setup_id} subscriber")

    # ── Configuration changes -------------------------------------------

    def set_segment_processor(self,
                              seg: Optional["VideoSegmentProcessor"]) -> None:
        with self._lock:
            self._segment_processor = seg

    def reset_lens_failed(self) -> None:
        """Clear the per-camera lens failure latch so correction is retried,
        called after a re-calibration or a fresh camera connect."""
        self._lens_failed = False
