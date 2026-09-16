"""
Camera factory - detects available cameras and creates instances.

Backends are no longer wired in here. Each one registers a ``BackendSpec``
(see ``registry.py``), which carries availability, enumeration, construction
and the feature descriptors the setup dialog renders itself from. This module
is the stable public surface over that registry, so existing callers keep
working while a new SDK can be added without touching this file.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from . import registry
from .base import GenericCamera, camera_info

logger = logging.getLogger(__name__)


class CameraFactory:
    """Factory for detecting and creating camera instances."""

    @staticmethod
    def get_available_backends() -> list[str]:
        """Backend ids whose SDK is importable, e.g. ``["opencv", "ximea"]``."""
        return registry.available_backend_ids()

    @staticmethod
    def list_available_cameras(backends: list[str] | None = None) -> list[dict]:
        """Detect all cameras across the given backends (all when None).

        Returns dicts shaped like
        ``{"unique_id": "0-opencv", "backend": "opencv", "model": "USB Camera"}``.
        """
        return registry.list_cameras(backends)

    @staticmethod
    def create_camera(unique_id: str, config: dict | None = None) -> GenericCamera:
        """Create a camera from ``<identifier>-<backend>``.

        Raises ValueError if the backend is unknown or its SDK is missing.
        """
        return registry.create_camera(unique_id, config)


# Enumeration opens every index in turn with a blocking read, 3–20 s on
# Windows, and two threads doing it at once fight over the same devices
# (DirectShow faults outright). One process-wide lock and one cached result:
# the walk happens once and every later caller is free.
_enum_lock = threading.Lock()
_enum_cache: Optional[list[dict]] = None


def invalidate_camera_enumeration() -> None:
    """Drop the cached device lists so the next call re-walks the bus.

    Call after plugging or unplugging a camera.

    There are THREE caches behind an enumeration, and dropping a subset is
    worse than dropping none: the bus walk then finds the new camera while
    the DirectShow list still predates it, so the camera is real but has no
    device path and is issued a fingerprint id no device-path lookup can
    resolve. Every cache that describes what is attached is cleared here.
    """
    global _enum_cache
    with _enum_lock:
        _enum_cache = None
    from . import identity as _ident
    _ident.invalidate_dshow_devices()
    _ident.invalidate_live_cameras()


def _list_opencv_cameras() -> list[dict]:
    """Probe OpenCV camera indices to find connected cameras.

    Tries to open cameras at indices 0..7. Less reliable than
    hardware-specific enumeration but works for USB webcams on all platforms.
    Cached process-wide; see ``invalidate_camera_enumeration``.
    """
    global _enum_cache


    with _enum_lock:
        if _enum_cache is not None:
            return list(_enum_cache)

        from . import identity as _ident

        cameras = []
        os_cams = _ident.list_os_cameras()

        # One scan builds the whole table: every live index with the identity
        # it answers to. Identity must NOT be the index: deriving it that way
        # (``unique_id = f"{idx}-opencv"``) means plugging in a second camera,
        # which can take index 0 and push the existing one to 1, silently
        # repoints a saved "camera 0" at different hardware.
        for entry in _ident.live_cameras(refresh=True):
            idx = int(entry["index"])
            info = camera_info("opencv", entry["id"], _model_name(idx, os_cams))
            # The index is a CURRENT LOCATION, carried for display and as the
            # starting guess. Nothing may store it: it is looked up from the
            # identity on every open, because it changes with what else is
            # plugged in.
            info["index"] = idx
            info["fingerprint"] = entry["fingerprint"]
            cameras.append(info)

        _enum_cache = cameras
        return list(cameras)


def _model_name(idx: int, os_cams: list) -> str:
    """A human label for the picker. Only trustworthy when unambiguous: the OS
    enumeration order is not OpenCV's, so naming by position would mislabel
    (measured: PnP listed the built-in first, OpenCV index 0 was the
    external)."""
    if len(os_cams) == 1 and os_cams[0].get("name"):
        return str(os_cams[0]["name"])
    return f"USB Camera {idx}"
