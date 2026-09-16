"""Camera backend registry, one place a backend declares itself.

The alternative is wiring each backend in by hand: an ``_BACKEND_AVAILABLE``
literal plus an ``if backend == "opencv" / elif "spinnaker" / elif "ximea"``
chain in the factory, which makes adding a vendor a four-site edit plus a
hand-written settings form.

A backend now registers one :class:`BackendSpec`. Everything downstream,
enumeration, construction, and the feature descriptors the setup dialog renders
itself from, goes through it, so a new SDK needs no changes here or in the GUI.
Shipped backends are registered by :func:`ensure_builtin_backends`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .features import CameraFeature

logger = logging.getLogger(__name__)


@dataclass
class BackendSpec:
    """Everything the app needs to know about one camera SDK.

    Only ``id``, ``display_name``, ``list_cameras`` and ``create_camera`` are
    required. A backend without ``describe_features`` still works, its camera
    simply shows the small static set the dialog can infer.
    """

    id: str
    display_name: str
    list_cameras: Callable[[], list]
    create_camera: Callable[..., object]
    is_available: Callable[[], bool] = lambda: True
    describe_features: Callable[[object], list[CameraFeature]] | None = None
    get_feature: Callable[[object, str], object] | None = None
    set_feature: Callable[[object, str, object], bool] | None = None
    module_name: str = ""          # python module that must import, for the UI
    version: str = ""
    origin: str = "builtin"        # builtin | plugin
    source_path: str = ""

    def available(self) -> bool:
        """Whether this SDK can be used on this machine.

        Cached: the check imports the vendor module, which costs well over a
        tenth of a second for a missing SDK, and the answer cannot change
        while the process runs.
        """
        cached = _AVAILABILITY.get(self.id)
        if cached is not None:
            return cached
        try:
            ok = bool(self.is_available())
        except Exception as exc:                       # pragma: no cover
            logger.warning("backend %s availability check failed: %s", self.id, exc)
            ok = False
        _AVAILABILITY[self.id] = ok
        return ok


_REGISTRY: dict[str, BackendSpec] = {}
_BUILTINS_LOADED = False
# backend id -> availability, memoised (see BackendSpec.available)
_AVAILABILITY: dict[str, bool] = {}


def register_backend(spec: BackendSpec, *, replace: bool = True) -> None:
    """Add ``spec`` to the registry. Later registrations win by default."""
    if not spec.id:
        raise ValueError("BackendSpec.id must be a non-empty string")
    if spec.id in _REGISTRY and not replace:
        raise ValueError(f"backend already registered: {spec.id}")
    _REGISTRY[spec.id] = spec
    _AVAILABILITY.pop(spec.id, None)   # a new spec probes for itself
    logger.debug("registered camera backend %r (%s)", spec.id, spec.origin)


def get_backend(backend_id: str) -> BackendSpec | None:
    ensure_builtin_backends()
    return _REGISTRY.get(backend_id)


def all_backends(*, available_only: bool = False) -> list[BackendSpec]:
    ensure_builtin_backends()
    specs = list(_REGISTRY.values())
    if available_only:
        specs = [s for s in specs if s.available()]
    # Stable, readable order: shipped first, then plugins, alphabetically.
    specs.sort(key=lambda s: (s.origin != "builtin", s.display_name.lower()))
    return specs


def available_backend_ids() -> list[str]:
    return [s.id for s in all_backends(available_only=True)]


# --------------------------------------------------------------- built-ins

def ensure_builtin_backends() -> None:
    """Register the shipped backends once, tolerating missing SDKs."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True  # set first so a failure cannot loop

    from . import builtin_backends
    try:
        builtin_backends.register_all()
    except Exception as exc:                           # pragma: no cover
        logger.error("failed registering built-in camera backends: %s", exc)


# ------------------------------------------------- guarded feature access

def describe_features(backend_id: str, camera) -> list[CameraFeature]:
    """Feature descriptors for ``camera``; empty when unsupported or failing."""
    spec = get_backend(backend_id)
    if spec is None or spec.describe_features is None or camera is None:
        return []
    try:
        return list(spec.describe_features(camera) or [])
    except Exception as exc:
        logger.warning("describe_features failed for %s: %s", backend_id, exc)
        return []


def get_feature(backend_id: str, camera, key: str):
    spec = get_backend(backend_id)
    if spec is None or spec.get_feature is None or camera is None:
        return None
    try:
        return spec.get_feature(camera, key)
    except Exception as exc:
        logger.debug("get_feature(%s, %s) failed: %s", backend_id, key, exc)
        return None


def set_feature(backend_id: str, camera, key: str, value) -> bool:
    spec = get_backend(backend_id)
    if spec is None or spec.set_feature is None or camera is None:
        return False
    try:
        return bool(spec.set_feature(camera, key, value))
    except Exception as exc:
        logger.warning("set_feature(%s, %s=%r) failed: %s",
                       backend_id, key, value, exc)
        return False


def list_cameras(backend_ids=None) -> list[dict]:
    """Enumerate cameras across backends, skipping any that misbehave."""
    ensure_builtin_backends()
    wanted = set(backend_ids) if backend_ids is not None else None
    out: list[dict] = []
    for spec in all_backends(available_only=True):
        if wanted is not None and spec.id not in wanted:
            continue
        try:
            out.extend(spec.list_cameras() or [])
        except Exception as exc:
            logger.warning("error scanning %s cameras: %s", spec.id, exc)
    return out


def create_camera(unique_id: str, config: dict | None = None):
    """Build a camera from ``<identifier>-<backend>``."""
    ensure_builtin_backends()
    parts = str(unique_id).rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid unique_id format: {unique_id}. Expected 'ID-backend'.")
    identifier, backend_id = parts
    spec = get_backend(backend_id)
    if spec is None:
        raise ValueError(f"Unknown camera backend: {backend_id}")
    if not spec.available():
        raise ValueError(
            f"{spec.display_name} backend not available "
            f"({spec.module_name or spec.id} not installed)")
    return spec.create_camera(unique_id=unique_id, identifier=identifier,
                              config=config or {})
