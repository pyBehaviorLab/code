"""QtBridge, relays Pipeline events to Qt signals on the main thread.

The Pipeline core is Qt-free: callbacks fire on whatever worker thread
produced them. This bridge wraps a Pipeline and re-emits the same
events as ``Signal``s so GUI code can connect with the usual
slot/signal pattern. ``Qt.ConnectionType.QueuedConnection`` ensures
slots run on the main thread regardless of where the callback fired.

Usage in main_window:

    self.pipeline = Pipeline()
    self.bridge = QtBridge(self.pipeline, parent=self)
    self.bridge.pose_ready.connect(self._on_box_pose)
    self.bridge.tracker_ready.connect(self._on_box_tracker)
    self.bridge.health.connect(self._on_health_warning)

That's the entire pipeline footprint in main_window.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtCore

from source.video.framebus.controller import Pipeline

logger = logging.getLogger(__name__)


class QtBridge(QtCore.QObject):
    """Qt adapter for a Pipeline. Re-emits events as queued-connection signals."""

    # (box_id, cam_frame_id, pose_array, location_or_None, speed,
    #  zones_by_body_part, raw_pose_dict)
    pose_ready = QtCore.Signal(int, int, object, object, float, object, object)

    # (box_id, cam_frame_id, centroid, location_or_None, speed,
    #  zones_by_body_part, position)
    tracker_ready = QtCore.Signal(int, int, object, object, float, object, object)

    # (box_id, reason)
    health = QtCore.Signal(int, str)

    # (box_id, reason, consecutive_empty_streak), fired after N
    # consecutive all-zero-confidence pose results (3, 20, or 100) so the
    # GUI can surface a banner.
    pose_failed = QtCore.Signal(int, str, int)

    # (box_id, TriggerFrame), one per pose frame when the box has trigger
    # rules; drives the video annotation (T2) + real-time plot (T3).
    trigger_ready = QtCore.Signal(int, object)

    def __init__(self, pipeline: Pipeline, *,
                 parent: Optional[QtCore.QObject] = None) -> None:
        """Wrap ``pipeline`` and re-emit pose / tracker / health / pose_failed
        results as queued Qt signals. Display is not relayed off-thread, the
        GUI paints the latest frame from a main-thread polling timer
        (``updateBoxDisplay`` draws zones + keypoints from pose_results /
        tracking_* state).
        """
        super().__init__(parent)
        self._pipeline = pipeline

        # Each Pipeline.on_* returns an unsubscribe handle.
        self._unsub_pose = pipeline.on_pose_result(self._on_pose)
        self._unsub_tracker = pipeline.on_tracker_result(self._on_tracker)
        self._unsub_health = pipeline.on_health(self._on_health)
        self._unsub_pose_failed = pipeline.on_pose_failed(self._on_pose_failed)
        self._unsub_trigger = pipeline.on_trigger_frame(self._on_trigger)

    # ── Sink/result callbacks (worker thread) → Qt signal -----------------

    def _on_pose(self, setup_id, cam_frame_id, pose_array, location,
                 speed, zones_by_body_part, raw_pose_dict):
        self.pose_ready.emit(int(setup_id), int(cam_frame_id), pose_array,
                             location, float(speed), zones_by_body_part,
                             raw_pose_dict)

    def _on_tracker(self, setup_id, cam_frame_id, centroid, location,
                    speed, zones_by_body_part, position):
        self.tracker_ready.emit(int(setup_id), int(cam_frame_id), centroid,
                                location, float(speed), zones_by_body_part,
                                position)

    def _on_health(self, setup_id, reason):
        self.health.emit(int(setup_id), str(reason))

    def _on_pose_failed(self, setup_id, reason, streak):
        self.pose_failed.emit(int(setup_id), str(reason), int(streak))

    def _on_trigger(self, tf):
        self.trigger_ready.emit(int(getattr(tf, "setup_id", 0)), tf)

    def shutdown(self) -> None:
        """Unsubscribe the relays and shut the pipeline down."""
        try:
            self._unsub_pose()
        except Exception:
            pass
        try:
            self._unsub_tracker()
        except Exception:
            pass
        try:
            self._unsub_health()
        except Exception:
            pass
        try:
            self._unsub_pose_failed()
        except Exception:
            pass
        try:
            self._unsub_trigger()
        except Exception:
            pass
        try:
            self._pipeline.shutdown()
        except Exception:
            pass
