"""Camera capture, backends + threads.

Wraps every camera backend (OpenCV / Spinnaker / Ximea) behind a single
``GenericCamera`` interface. Tracking algorithms live in
``source.video.tracking``; video encoding + on-disk writers live in
``source.video.recording``.

Direct module imports always work::

    from source.video.cameras.capture import CameraThread
    from source.video.cameras.factory import CameraFactory

Shorthand ``from source.video.cameras import X`` is supported via the lazy
attribute lookup below.
"""

# Eager leaf-only types, safe (no cycles).
from .base import GenericCamera, ResolvedCameraSettings  # noqa: F401


_LAZY = {
    # camera factory
    "CameraFactory": ("factory", "CameraFactory"),
}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(
            f"module 'source.video.cameras' has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f"source.video.cameras.{spec[0]}")
    return getattr(mod, spec[1])
