"""source.video.framebus, frame distribution and sink orchestration.

Both maze and operant import only from here. The Pipeline is
Qt-free; QtBridge is the thin adapter that exposes events as Signals
so GUI code can use slot/signal wiring on the main thread.

Layout::

    controller.py            orchestrator (Pipeline)
    frame_bus.py             FrameBus, single producer fan-out
    types.py                 CameraFrame, BoxFrame, CameraConfig, TrackingConfig
    latency.py               LatencyBudget, per-stage timing telemetry
    qt_bridge.py             Qt signal adapter
    sink_base.py             Sink ABC + DropPolicy + shared sink helpers
    recorder_sink.py         RecorderSink, lossless writes to disk
    pose_sink.py             PoseSink, DLC/SLEAP inference
    tracker_sink.py          TrackerSink, blob detection
    mcu_pusher.py            MCUPusher + TrackingPushPolicy (pose → MCU)

Architecture (per camera)::

    CameraThread (producer)
            ↓
        FrameBus.publish_frame(camera_frame)
            │   ── on_camera_frame  → raw subscribers (zone editor preview)
            │   ── per-box segmentation (crop if shared, passthrough else)
            ↓
        BoxFrame (image; MCU fw time read per box by RecorderSink)
            │
       ┌────┴──────┬──────────┐
       ↓           ↓          ↓
    Recorder     Pose      Tracker
    (lossless   (skip-but-  (per-box
     + alarm)    write)      bounded)

Each sink runs on its own thread, behind its own bounded queue, with
its own drop policy. A slow sink CANNOT block another sink. Recording
is lossless; the GUI display polls at its own rate and may skip; pose
skips while inference is busy and writes empty pose-array rows for
skipped frames so the user can see exactly which frames DLC didn't
run on.
"""

from .controller import HealthCallback, Pipeline
from .frame_bus import BoxFrameCallback, CameraFrameCallback, FrameBus
from .mcu_pusher import MCUPusher
from .pose_sink import PoseSink
from .recorder_sink import RecorderSink
from .sink_base import DropPolicy, Sink
from .tracker_sink import TrackerSink
from .types import CameraConfig, TrackingConfig

# QtBridge is the ONLY Qt-touching module in this package. Importing it here
# would drag PySide6 into every consumer of the Pipeline and make the
# "controller.py carries no Qt dependency" contract false, which it silently
# was. Same lazy-attribute pattern the sibling packages use.
_LAZY = {"QtBridge": ("qt_bridge", "QtBridge")}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(
            f"module 'source.video.framebus' has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f"source.video.framebus.{spec[0]}")
    return getattr(mod, spec[1])


__all__ = [
    "BoxFrameCallback",
    "CameraConfig",
    "CameraFrameCallback",
    "DropPolicy",
    "FrameBus",
    "HealthCallback",
    "MCUPusher",
    "Pipeline",
    "PoseSink",
    "QtBridge",
    "RecorderSink",
    "Sink",
    "TrackerSink",
    "TrackingConfig",
]
