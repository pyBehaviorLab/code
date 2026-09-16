"""Zone & event-trigger dataclasses (pure data, no I/O).

Authoritative shape for everything stored in JSON-on-disk and passed between
runtime, analysis, and gui. Optional Shapely caching (`_polygon`, `_prepared`)
gives the live-trigger fast path; without Shapely, contains() falls back to the
pure-Python ray-cast in geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional Shapely accelerators (live triggering only; analysis uses
# the pure-Python fallbacks in geometry.py).
_SHAPELY_AVAILABLE = False
_HAS_CONTAINS_XY = False
_HAS_PREPARED = False
_Polygon = None
_Point = None
_explain_validity = None
_contains_xy = None
_prep = None
try:
    from shapely.geometry import Point as _Point, Polygon as _Polygon  # type: ignore
    from shapely.validation import explain_validity as _explain_validity  # type: ignore
    _SHAPELY_AVAILABLE = True
    try:
        from shapely import contains_xy as _contains_xy  # type: ignore
        _HAS_CONTAINS_XY = True
    except ImportError:
        pass
    try:
        from shapely.prepared import prep as _prep  # type: ignore
        _HAS_PREPARED = True
    except ImportError:
        pass
except ImportError:
    pass


@dataclass
class Zone:
    """A spatial zone with optional event-triggering policy.

    **This is the only Zone class.** ``config.experiment`` imports it rather
    than mirroring it. As two separate dataclasses, the runtime one cannot
    represent eight of the persisted fields, and a project reload then drops
    scale calibration, ellipse geometry and, worst, the per-zone MCU
    transmit policy, leaving every zone on the default.

    Geometry is a list of (x, y) points. ``coord_space`` says whether they are
    normalised [0, 1] or pixels. The Shapely cache (``_polygon``,
    ``_prepared``) is built once in ``__post_init__`` for O(1)
    point-in-polygon; without Shapely, ``contains`` falls back to the
    pure-Python ray-cast in ``geometry``.

    Per-zone transmit policy:
        transmit_mode = "coord"            → push c.<coord_var> = name on entry
                      = "event_enter"      → trigger_event(event_on_enter) on entry
                      = "event_enter_exit" → trigger on entry AND exit

    ``transmit_mode`` and ``coord_var`` are non-optional strings on purpose.
    Consumers read them with ``getattr(z, "transmit_mode", "coord")``, whose
    default only applies when the attribute is *absent*, a present-but-None
    value flows straight through and matches no branch, so the zone is silently
    ignored by the MCU pusher. ``from_dict`` therefore maps a missing OR null
    value onto the working default.
    """
    name: str = ""
    points: List[Tuple[float, float]] = field(default_factory=list)
    zone_type: str = "rectangle"  # rectangle, polygon, circle, line, scale
    enabled: bool = True
    shape_dim: Optional[List[int]] = None  # [width, height] of the frame drawn on
    coord_space: str = "pixel"             # "pixel" | "normalized"
    validated: bool = False
    # scale-type only
    scale_length: Optional[float] = None
    scale_unit: Optional[str] = None
    scale_length_mm: Optional[float] = None
    # ellipse-type only, kept so the shape re-opens as an editable ellipse
    # rather than a frozen polygon of sampled points
    center: Optional[List[float]] = None
    semi_axes: Optional[List[float]] = None
    radius_norm: Optional[float] = None
    # per-zone MCU trigger config
    transmit_mode: str = "coord"
    coord_var: str = "loc_center"
    event_on_enter: str = ""
    event_on_exit: str = ""
    # Shapely accelerators. Excluded from init, repr, equality and
    # serialisation: they are derived, unpicklable, and ``asdict()`` on a live
    # one raises during JSON encoding.
    _polygon: Optional[Any] = field(default=None, init=False, repr=False,
                                    compare=False)
    _prepared: Optional[Any] = field(default=None, init=False, repr=False,
                                     compare=False)

    def __post_init__(self):
        if self.zone_type == "line":
            return  # line zones have no polygon cache
        if not (_SHAPELY_AVAILABLE and self.points and len(self.points) >= 3):
            return
        try:
            self._polygon = _Polygon(self.points)
            if not self._polygon.is_valid:
                self._polygon = self._polygon.buffer(0)
                if not self._polygon.is_valid and _explain_validity is not None:
                    logger.warning(
                        "Zone '%s' has invalid geometry: %s",
                        self.name, _explain_validity(self._polygon),
                    )
            if _HAS_PREPARED and self._polygon is not None:
                self._prepared = _prep(self._polygon)
        except Exception as e:
            logger.error("Failed to create polygon for zone '%s': %s", self.name, e)
            self._polygon = None
            self._prepared = None

    # The Shapely caches are derived state and cannot be pickled, a prepared
    # geometry refuses outright, which surfaced as a PicklingError from
    # anything that deep-copies a config (autosave/reload does). Drop them on
    # the way out and rebuild on the way in, so copies stay cheap and correct.

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_polygon"] = None
        state["_prepared"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()

    def contains(self, x: float, y: float) -> bool:
        """Point-in-zone test; Shapely cache when available, else ray-cast."""
        if _SHAPELY_AVAILABLE and self._polygon is not None:
            try:
                if _HAS_CONTAINS_XY:
                    return bool(_contains_xy(self._polygon, x, y))
                if self._prepared is not None:
                    return self._prepared.contains(_Point(x, y))
                return self._polygon.contains(_Point(x, y))
            except Exception:
                return False
        # Pure-Python fallback (imported lazily to avoid load-time cost).
        from source.video.zones.geometry import point_in_polygon
        return point_in_polygon(x, y, self.points)

    # ── Serialisation ---------------------------------------------------
    # The on-disk key is "type", not "zone_type"; that is what every project
    # file, zone file and tracking config in the wild contains. The field is
    # named zone_type only because `type` shadows the builtin, so the mapping
    # is explicit here rather than via asdict().

    # Identity of a zone: always written, even when empty.
    _ALWAYS = ("name", "type", "points")

    def to_dict(self) -> dict:
        """JSON-shaped dict carrying only what distinguishes this zone.

        Any field still at its class default is omitted, so a plain rectangle
        is not padded with the scale, ellipse and event fields that apply to
        other zone types. Lossless: ``from_dict`` restores exactly those
        defaults for anything absent.
        """
        d = {
            "name": self.name,
            "type": self.zone_type,
            "points": [list(p) for p in self.points],
            "shape_dim": self.shape_dim,
            "coord_space": self.coord_space,
            "enabled": self.enabled,
            "validated": self.validated,
            "scale_length": self.scale_length,
            "scale_unit": self.scale_unit,
            "scale_length_mm": self.scale_length_mm,
            "center": self.center,
            "semi_axes": self.semi_axes,
            "radius_norm": self.radius_norm,
            "transmit_mode": self.transmit_mode,
            "coord_var": self.coord_var,
            "event_on_enter": self.event_on_enter,
            "event_on_exit": self.event_on_exit,
        }
        defaults = self._defaults()
        return {k: v for k, v in d.items()
                if k in self._ALWAYS or (v is not None and v != defaults.get(k))}

    @classmethod
    def _defaults(cls) -> dict:
        """Default value per on-disk key, for the omit-if-default rule."""
        f = cls.__dataclass_fields__
        return {
            "shape_dim": None,
            "coord_space": f["coord_space"].default,
            "enabled": f["enabled"].default,
            "validated": f["validated"].default,
            "scale_length": None,
            "scale_unit": None,
            "scale_length_mm": None,
            "center": None,
            "semi_axes": None,
            "radius_norm": None,
            "transmit_mode": f["transmit_mode"].default,
            "coord_var": f["coord_var"].default,
            "event_on_enter": f["event_on_enter"].default,
            "event_on_exit": f["event_on_exit"].default,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        """Build a Zone from a JSON-shaped dict.

        Tolerates the ``zone_type`` spelling on read: an earlier serialiser in
        ``framebus.types`` wrote that key while everything else wrote ``type``,
        so a dict produced by that writer would otherwise load every zone as a
        default-shaped one.
        """
        if not isinstance(data, dict):
            return cls()

        def _f(key):
            v = data.get(key)
            return None if v is None else float(v)

        def _l(key):
            v = data.get(key)
            return None if v is None else list(v)

        return cls(
            name=str(data.get("name") or ""),
            points=[(p[0], p[1]) for p in (data.get("points") or [])],
            zone_type=str(data.get("type")
                          or data.get("zone_type")
                          or "rectangle"),
            enabled=bool(data.get("enabled", True)),
            shape_dim=_l("shape_dim"),
            coord_space=str(data.get("coord_space", "pixel")),
            validated=bool(data.get("validated", False)),
            scale_length=_f("scale_length"),
            scale_unit=data.get("scale_unit"),
            scale_length_mm=_f("scale_length_mm"),
            center=_l("center"),
            semi_axes=_l("semi_axes"),
            radius_norm=_f("radius_norm"),
            # `or default`, not `.get(k, default)`. An explicit null in the
            # file must resolve to the working default too; see the class
            # docstring for why a None here silently disables the zone.
            transmit_mode=str(data.get("transmit_mode") or "coord"),
            coord_var=str(data.get("coord_var") or "loc_center"),
            event_on_enter=str(data.get("event_on_enter") or ""),
            event_on_exit=str(data.get("event_on_exit") or ""),
        )


__all__ = ["Zone"]
