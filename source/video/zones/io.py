"""JSON I/O for zone configurations.

Three formats supported:
    1. Raw list:    [{"name": ..., "points": [...], "type": ...}, ...]
    2. Wrapped list: {"zones": [<same list>], ...}
    3. Wrapped dict: {"zones": {"<name>": {"values": [...], ...}, ...},
                      "scale": {"length": "<n>", "values": [x1, x2]}}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _has_shape(zone: Dict[str, Any], raw: Dict[str, Any]) -> bool:
    """Whether this entry describes a shape at all.

    Points for a polygon, line or scale; a centre with axes or a radius for an
    ellipse or circle written the compact way.
    """
    if zone.get("points"):
        return True
    return bool(raw.get("center") and (raw.get("semi_axes")
                                       or raw.get("radius_norm")))


def load_zone_config(path: str) -> List[Dict[str, Any]]:
    """Load zone config from a JSON file as raw dicts.

    Normalizes the three formats into a single list of zone-dicts. The
    dict-format `values` are mapped to `points`. Scale info is appended as a
    synthetic zone with type="scale" so downstream calibration code finds it
    on the same iteration.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("zones"), list):
        return data["zones"]

    if isinstance(data, dict) and isinstance(data.get("zones"), dict):
        zones: List[Dict[str, Any]] = []
        for name, zd in data["zones"].items():
            if not isinstance(zd, dict):
                continue
            zone = {
                "name": zd.get("name", name),
                "type": zd.get("type", "polygon"),
                "points": zd.get("points") or zd.get("values") or [],
            }
            if not _has_shape(zone, zd):
                # A zone with no geometry is not a zone. A config can carry
                # entries with empty point lists, an offline analyser wrote
                # some, and read as zones they reach the editor, the picker
                # and the results as shapes that are nowhere.
                continue
            if "coord_space" in zd:
                zone["coord_space"] = zd["coord_space"]
            if zd.get("shape_dim") is not None:
                zone["shape_dim"] = zd["shape_dim"]
            zones.append(zone)

        scale = data.get("scale")
        if isinstance(scale, dict):
            try:
                length = float(scale.get("length"))
                vals = scale.get("values") or []
                if length > 0 and len(vals) >= 2:
                    zones.append({
                        "name": "scale",
                        "type": "scale",
                        "points": [[float(vals[0]), 0.0],
                                    [float(vals[1]), 0.0]],
                        "scale_length": length,
                        "scale_unit": scale.get("unit", "cm"),
                    })
            except (TypeError, ValueError):
                pass

        return zones

    logger.warning("Unexpected zone format in %s", path)
    return []


__all__ = ["load_zone_config"]
