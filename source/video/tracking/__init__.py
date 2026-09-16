"""Tracking algorithms, blob detection, pose inference (DLC/SLEAP).

These algorithms operate on frames that are already in memory; they do
not own any camera hardware. Camera capture lives in
``source.video.cameras``. Zone geometry + event triggering lives in
``source.video.zones``.

Direct module imports always work::

    from source.video.tracking.blob import BlobTracker
    from source.video.tracking.pose import DLCLiveTracker
    from source.video.zones.triggering import ZoneManager

Shorthand ``from source.video.tracking import X`` is supported via the
lazy attribute lookup below.
"""

# Leaf types, safe to eager-import (no cycles).
from .types import PoseSnapshot  # noqa: F401


_LAZY = {
    "BlobTracker":                  ("blob", "BlobTracker"),
    "TrackerManager":               ("blob", "TrackerManager"),
    "PoseTracker":                  ("pose", "PoseTracker"),
    "DLCLiveTracker":               ("pose", "DLCLiveTracker"),
    "SLEAPTracker":                 ("pose", "SLEAPTracker"),
    "create_pose_tracker":          ("pose", "create_pose_tracker"),
    "TrackingEnhancer":             ("smoothing", "TrackingEnhancer"),
    "InferenceBackend":             ("inference", "InferenceBackend"),
    "ThreadInferenceBackend":       ("inference", "ThreadInferenceBackend"),
    "MultiInstanceInferenceBackend":("inference", "MultiInstanceInferenceBackend"),
    "ModelHandle":                  ("inference", "ModelHandle"),
}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(
            f"module 'source.video.tracking' has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f"source.video.tracking.{spec[0]}")
    return getattr(mod, spec[1])
