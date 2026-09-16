"""Where the arena is: zones, scale, and recomputing occupancy offline.

Re-zoning did not exist. 2D zone occupancy was string equality against the
``location`` column the recorder wrote at acquisition time, so a zone drawn
slightly wrong could only be fixed by re-recording the experiment. Editing
zones in the Analyze tab wrote them into a dictionary nobody read.

This module makes the cheap operation possible: the poses are already on disk,
so recomputing which zone the animal was in costs one polygon test per frame
and nothing else, seconds, not the hours a re-inference would take.

Three things arrive with it that the live pipeline could not give:

* **Per-zone occupancy series.** ``location`` is one string per frame, so
  overlapping and nested zones were inexpressible; a Y-maze arm inside an
  arena inside a quadrant now simply works.
* **Head as well as body.** Every behavioural assay that scores investigation,
  novel object, social interaction, scores where the *nose* is, not the
  centroid. Both series are computed for every zone.
* **Interaction zones with a facing criterion.** Time in an object's
  interaction area only counts as investigation when the animal is oriented
  toward the object; that is the standard definition and it needs the
  head-body vector, which only pose tracking provides.

Headless: numpy, and OpenCV only for rasterising polygons (with a Shapely
fallback). No Qt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

#: Zone types that are not places the animal can be.
NON_OCCUPANCY_TYPES = frozenset({"scale", "line"})

#: A zone whose name carries this marker, or whose dict says
#: ``"interaction": true``, is scored with the facing criterion.
INTERACTION_MARKER = "_IA"

DEFAULT_INTERACTION = {
    #: Head-to-target angle at or below which the animal counts as facing it.
    "angle_threshold_deg": 45.0,
    #: The head must stay in the zone this long before the visit counts, which
    #: rejects the animal merely passing the object on its way somewhere.
    "confirm_ms": 100.0,
    #: What the animal must be facing. ``outer_edge`` is the behavioural
    #: definition: an interaction area is drawn around an object that sits
    #: against the arena wall, so the object is at the zone's OUTWARD face and
    #: that is what the animal orients to. ``nearest_edge`` and ``centroid``
    #: are available for arenas where that assumption does not hold.
    "target": "outer_edge",
    #: How far past the zone's own ends the outward edge is extended before
    #: the angle is measured. An animal investigating at the corner of the
    #: strip has its nose beyond the drawn edge, and the nearest point on an
    #: unextended segment is then its endpoint, which sits off to one side and
    #: reports the animal as facing away from an object it is nose-to.
    "line_extend_px": 15.0,
    #: How far INSIDE the outward edge the head must reach before the visit
    #: counts at all. An interaction strip is drawn generously so the animal
    #: fits in it; loitering at its inner boundary, a body-length from the
    #: object, is not investigation. Zero disables the gate.
    "inner_offset_px": 50.0,
}


# ── model ────────────────────────────────────────────────────────────────────

@dataclass
class SpaceModel:
    """The arena as this analysis will use it: pixel-space zones plus a scale.

    ``frame_size`` is carried explicitly because zones are only meaningful
    against the frame they were resolved on. Loading a recording whose video
    is a different size is an error with both numbers in it, never a rescale
    behind the user's back.
    """

    frame_size: Tuple[int, int] = (0, 0)
    zones: List[dict] = field(default_factory=list)      # pixel space
    px_per_cm: float = 0.0
    origin: str = "file"          # file | edited | copied:<stem> | none
    interaction: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_INTERACTION))
    warnings: List[str] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        return self.px_per_cm > 0

    @property
    def zone_names(self) -> List[str]:
        return [z["name"] for z in self.zones
                if z.get("name") and z.get("type") not in NON_OCCUPANCY_TYPES]

    @property
    def interaction_zones(self) -> List[str]:
        return [z["name"] for z in self.zones if is_interaction_zone(z)]

    def to_dict(self) -> dict:
        return {"frame_size": list(self.frame_size), "zones": self.zones,
                "px_per_cm": self.px_per_cm, "origin": self.origin,
                "interaction": self.interaction}

    @classmethod
    def from_dict(cls, d: dict) -> "SpaceModel":
        fs = d.get("frame_size") or [0, 0]
        return cls(frame_size=(int(fs[0]), int(fs[1])),
                   zones=list(d.get("zones") or []),
                   px_per_cm=float(d.get("px_per_cm") or 0.0),
                   origin=str(d.get("origin") or "file"),
                   interaction={**DEFAULT_INTERACTION,
                                **(d.get("interaction") or {})})


def is_interaction_zone(zone: dict) -> bool:
    if zone.get("interaction"):
        return True
    name = str(zone.get("name") or "")
    return INTERACTION_MARKER in name


# ── coordinate space ─────────────────────────────────────────────────────────

def looks_normalized(points: Sequence[Sequence[float]],
                     coord_space: Optional[str] = None) -> bool:
    """Whether these points are in 0-1 space.

    An explicit ``coord_space`` always wins; the fallback only fires for files
    written before the field existed. A zone that genuinely lives in the top-
    left 1x1 pixel is not a thing, so the heuristic is safe, but it is a
    fallback, not the rule.
    """
    if coord_space in ("normalized", "pixel"):
        return coord_space == "normalized"
    if not points:
        return False
    return all(0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0
               for p in points if len(p) >= 2)


def zones_to_pixel(zones: Iterable[dict], frame_size: Tuple[int, int]
                   ) -> List[dict]:
    """Every zone in video-pixel coordinates, the canonical offline space.

    Normalised zones are denormalised against ``frame_size``; zones drawn at a
    different resolution (``shape_dim``) are rescaled to it.
    """
    zones = [z for z in (zones or []) if isinstance(z, dict)]
    fw, fh = float(frame_size[0] or 0), float(frame_size[1] or 0)
    if not (fw > 0 and fh > 0):
        # No frame size known, fall back to the frame the zones were drawn on.
        # A normalised zone left un-denormalised would become a 1×1-pixel
        # region in the corner: a silent nothing, not an error.
        for z in zones:
            sd = z.get("shape_dim") or [0, 0]
            if sd and sd[0] and sd[1]:
                fw, fh = float(sd[0]), float(sd[1])
                break
    out: List[dict] = []
    for z in zones:
        zz = dict(z)
        pts = [(float(p[0]), float(p[1])) for p in (z.get("points") or [])
               if len(p) >= 2]
        # ONE decision, applied to every geometry field the zone carries.
        # Converting `points` alone while `center` and `semi_axes` are copied
        # through untouched, and stamping the zone "pixel" regardless, gives
        # an ellipse with its outline in pixels and its centre still
        # normalized, so the editor draws the shape
        # correctly and put its resize handles tens of thousands of pixels
        # off-screen: the zone could be dragged but never resized.
        scale = (1.0, 1.0)
        if pts:
            if looks_normalized(pts, z.get("coord_space")):
                if fw > 0 and fh > 0:
                    scale = (fw, fh)
            else:
                sd = z.get("shape_dim") or [0, 0]
                if fw > 0 and fh > 0 and sd and sd[0] and sd[1]:
                    sx, sy = fw / float(sd[0]), fh / float(sd[1])
                    if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                        scale = (sx, sy)
            pts = [(x * scale[0], y * scale[1]) for x, y in pts]
        zz["points"] = pts
        for key in ("center", "semi_axes"):
            pair = z.get(key)
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                try:
                    zz[key] = [float(pair[0]) * scale[0],
                               float(pair[1]) * scale[1]]
                except (TypeError, ValueError):
                    pass
        zz["coord_space"] = "pixel"
        zz["shape_dim"] = [int(fw), int(fh)]
        zz.setdefault("name", z.get("name") or "zone")
        out.append(zz)
    return out


def px_per_cm_from_zones(zones: Iterable[dict], frame_size: Tuple[int, int]
                         ) -> float:
    """Pixels per centimetre from the scale zone. THE implementation.

    A second copy of this once existed in the Analyze tab without the
    normalisation handling; it was never called, and was deleted rather than
    fixed. Both the live tab and the offline path call this one.

    Returns 0.0 when there is no usable scale, which means every distance is
    in pixels and must be labelled that way, not quietly treated as cm.
    """
    fw, fh = float(frame_size[0] or 0), float(frame_size[1] or 0)
    for z in zones or []:
        if not isinstance(z, dict) or z.get("type") != "scale":
            continue
        pts = z.get("points") or []
        try:
            length = float(z.get("scale_length", 0) or 0)
        except (TypeError, ValueError):
            continue
        if len(pts) < 2 or length <= 0:
            continue
        if looks_normalized(pts, z.get("coord_space")):
            w, h = (fw, fh)
            if not (w > 0 and h > 0):
                sd = z.get("shape_dim") or [0, 0]
                w, h = (float(sd[0]), float(sd[1])) if sd[0] and sd[1] else (1.0, 1.0)
            dx = (float(pts[1][0]) - float(pts[0][0])) * w
            dy = (float(pts[1][1]) - float(pts[0][1])) * h
        else:
            dx = float(pts[1][0]) - float(pts[0][0])
            dy = float(pts[1][1]) - float(pts[0][1])
            sd = z.get("shape_dim") or [0, 0]
            if fw > 0 and fh > 0 and sd and sd[0] and sd[1]:
                dx *= fw / float(sd[0])
                dy *= fh / float(sd[1])
        px = math.hypot(dx, dy)
        if px < 1:
            continue
        unit = str(z.get("scale_unit", "cm")).lower()
        if unit in ("mm",):
            length /= 10.0
        elif unit in ("m", "metre", "meter"):
            length *= 100.0
        return px / length
    return 0.0


def space_from_header(header, frame_size: Optional[Tuple[int, int]] = None,
                      *, origin: str = "file") -> SpaceModel:
    """Build the space a recording was analysed in from its own header."""
    from tools.offline_analysis import video_data_schema as vds

    # The poses' own space when the recording declares it. Zones are stored
    # normalized, so multiplying them by the VIDEO's size, which is all
    # older files carry, puts them where the poses never go.
    #
    # Tested on the width, not on truthiness: `(0, 0)` is a non-empty tuple,
    # so `pose_resolution or resolution` silently selects the absent one.
    pose_fs = tuple(getattr(header, "pose_resolution", (0, 0)) or (0, 0))
    fs = frame_size or (pose_fs if pose_fs and pose_fs[0]
                        else header.resolution)
    zones = zones_to_pixel(vds.zones_as_list(header.zones), fs)
    ppc = header.px_per_cm or px_per_cm_from_zones(zones, fs)
    m = SpaceModel(frame_size=(int(fs[0]), int(fs[1])), zones=zones,
                   px_per_cm=float(ppc), origin=origin)
    if not m.zone_names:
        m.warnings.append("this recording has no zones, locomotion only")
    if not m.calibrated:
        m.warnings.append(
            "no scale: distances will be in PIXELS, not centimetres")
    return m


# ── fast point-in-zone ───────────────────────────────────────────────────────

class ZoneRaster:
    """Point-in-zone by lookup instead of by geometry.

    Twenty thousand frames times a dozen zones is a quarter of a million
    polygon tests; rasterising each zone once into a frame-sized mask turns
    each of them into an array index. Falls back to Shapely when OpenCV is
    unavailable, which is slower but identical in result.
    """

    def __init__(self, zones: Sequence[dict], frame_size: Tuple[int, int]):
        self.frame_size = (int(frame_size[0] or 0), int(frame_size[1] or 0))
        self.names: List[str] = []
        self._masks: Dict[str, np.ndarray] = {}
        self._polys: Dict[str, Any] = {}
        w, h = self.frame_size
        for z in zones:
            name = z.get("name")
            if not name or z.get("type") in NON_OCCUPANCY_TYPES:
                continue
            if z.get("enabled") is False:
                continue
            pts = z.get("points") or []
            if len(pts) < 3:
                continue
            self.names.append(name)
            if w > 0 and h > 0 and _cv2() is not None:
                self._masks[name] = _rasterise(pts, w, h)
            else:
                self._polys[name] = _shapely_polygon(pts)

    def contains(self, name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean series for one zone over arrays of x, y (NaN → False)."""
        n = len(x)
        out = np.zeros(n, bool)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            return out
        mask = self._masks.get(name)
        if mask is not None:
            h, w = mask.shape
            xi = np.clip(np.rint(x[valid]).astype(int), 0, w - 1)
            yi = np.clip(np.rint(y[valid]).astype(int), 0, h - 1)
            # A point outside the frame is outside every zone, clipping would
            # otherwise smear it onto the border pixel.
            inside = ((x[valid] >= -0.5) & (x[valid] <= w - 0.5) &
                      (y[valid] >= -0.5) & (y[valid] <= h - 0.5))
            hit = mask[yi, xi].astype(bool) & inside
            out[valid] = hit
            return out
        poly = self._polys.get(name)
        if poly is None:
            return out
        idx = np.nonzero(valid)[0]
        for i in idx:
            out[i] = _shapely_contains(poly, float(x[i]), float(y[i]))
        return out

    def all_series(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: self.contains(name, x, y) for name in self.names}


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:                                   # pragma: no cover
        return None


def _rasterise(points: Sequence[Sequence[float]], w: int, h: int) -> np.ndarray:
    cv2 = _cv2()
    mask = np.zeros((h, w), np.uint8)
    poly = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in points],
                      dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 1)
    return mask


def _shapely_polygon(points):                             # pragma: no cover
    try:
        from shapely.geometry import Polygon
        p = Polygon([(float(a[0]), float(a[1])) for a in points])
        return p if p.is_valid else p.buffer(0)
    except Exception:
        return None


def _shapely_contains(poly, x: float, y: float) -> bool:  # pragma: no cover
    try:
        from shapely.geometry import Point
        return bool(poly.contains(Point(x, y)))
    except Exception:
        return False


# ── geometry for the facing criterion ────────────────────────────────────────

def polygon_centroid(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    pts = np.asarray([[float(p[0]), float(p[1])] for p in points], float)
    if pts.size == 0:
        return (math.nan, math.nan)
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def nearest_point_on_polygon(points: Sequence[Sequence[float]],
                             x: float, y: float) -> Tuple[float, float]:
    """Closest point on the polygon's boundary, the object's near face as
    seen from where the animal's head is."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 2:
        return (math.nan, math.nan)
    best = (math.inf, pts[0])
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        t = 0.0 if den == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / den))
        px, py = ax + t * dx, ay + t * dy
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best[0]:
            best = (d, (px, py))
    return best[1]


def outer_edge(points: Sequence[Sequence[float]],
               arena_centre: Tuple[float, float]
               ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The zone's outward-facing edge, where the object is.

    An interaction area is a strip drawn in front of an object that stands
    against the arena wall. The animal investigating it faces *outward*, so
    measuring the angle to the nearest edge is wrong the moment the nose is
    inside the strip: the nearest edge is then whichever side it happens to be
    closest to, which is often behind it.

    The outward edge is the one whose midpoint is farthest from the arena
    centre *along its own outward normal*, distance alone ties on symmetric
    shapes, the projection does not.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3:
        return None
    # Winding decides which way "outward" points.
    area2 = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] -
                pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    sign = 1.0 if area2 > 0 else -1.0
    cx, cy = arena_centre
    best = (-math.inf, None)
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        ex, ey = bx - ax, by - ay
        n = math.hypot(ex, ey)
        if n < 1e-9:
            continue
        nx, ny = sign * (ey / n), sign * (-ex / n)      # outward normal
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        score = (mx - cx) * nx + (my - cy) * ny
        if score > best[0]:
            best = (score, ((ax, ay), (bx, by)))
    return best[1]


def extend_segment(a: Tuple[float, float], b: Tuple[float, float],
                   by_px: float) -> Tuple[Tuple[float, float],
                                          Tuple[float, float]]:
    """``a``–``b`` lengthened by ``by_px`` at both ends.

    The object is not confined to the width of the strip somebody drew in
    front of it. Without this, a nose past the end of the segment measures its
    angle to the segment's ENDPOINT, off to one side, and an animal pressed
    against the object at the corner is scored as facing away from it.
    """
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy)
    if n < 1e-9 or by_px <= 0:
        return a, b
    ux, uy = dx / n, dy / n
    return ((ax - ux * by_px, ay - uy * by_px),
            (bx + ux * by_px, by + uy * by_px))


def inward_normal(a: Tuple[float, float], b: Tuple[float, float],
                  arena_centre: Tuple[float, float]
                  ) -> Optional[Tuple[float, float]]:
    """Unit normal of ``a``–``b`` pointing toward the arena centre."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    nx, ny = -dy / n, dx / n
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    if nx * (arena_centre[0] - mx) + ny * (arena_centre[1] - my) < 0:
        nx, ny = -nx, -ny
    return (nx, ny)


def past_line(point: Tuple[float, float], on_line: Tuple[float, float],
              normal: Tuple[float, float]) -> bool:
    """Whether ``point`` is on the normal's side of the line through ``on_line``."""
    return ((point[0] - on_line[0]) * normal[0]
            + (point[1] - on_line[1]) * normal[1]) >= 0.0


def nearest_point_on_segment(a: Tuple[float, float], b: Tuple[float, float],
                             x: float, y: float) -> Tuple[float, float]:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    t = 0.0 if den == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / den))
    return (ax + t * dx, ay + t * dy)


def facing_angle_deg(head: Tuple[float, float], body: Tuple[float, float],
                     target: Tuple[float, float]) -> float:
    """Angle between where the animal is pointing and where the target is.

    0 = looking straight at it, 180 = directly away. The heading is the
    body→head vector, which is why this measure needs a pose and cannot be
    computed from a centroid alone.
    """
    hx, hy = head
    bx, by = body
    tx, ty = target
    vx, vy = hx - bx, hy - by                    # heading
    wx, wy = tx - hx, ty - hy                    # head → target
    nv, nw = math.hypot(vx, vy), math.hypot(wx, wy)
    if nv < 1e-9 or nw < 1e-9:
        return math.nan
    cos = max(-1.0, min(1.0, (vx * wx + vy * wy) / (nv * nw)))
    return math.degrees(math.acos(cos))


# ── the results ──────────────────────────────────────────────────────────────

@dataclass
class InteractionResult:
    """Investigation of one interaction zone, by the standard definition."""
    zone: str
    in_zone: np.ndarray                       # head inside, dwell-confirmed
    facing: np.ndarray                        # and oriented toward the target
    angle_deg: np.ndarray
    time_s: float = 0.0                       # confirmed head-in-zone time
    facing_time_s: float = 0.0                # the investigation measure
    entries: int = 0
    facing_entries: int = 0


@dataclass
class ZoneResult:
    """Occupancy recomputed from poses that already exist."""
    names: List[str] = field(default_factory=list)
    body: Dict[str, np.ndarray] = field(default_factory=dict)
    head: Dict[str, np.ndarray] = field(default_factory=dict)
    location: List[str] = field(default_factory=list)
    interaction: Dict[str, InteractionResult] = field(default_factory=dict)
    n_frames: int = 0

    def series(self, zone: str, point: str = "body") -> np.ndarray:
        src = self.head if point == "head" else self.body
        return src.get(zone, np.zeros(self.n_frames, bool))


def _confirm(mask: np.ndarray, dt: np.ndarray, min_s: float) -> np.ndarray:
    """Keep only the part of each run that follows ``min_s`` continuously
    inside. A visit shorter than that never becomes True at all."""
    if min_s <= 0:
        return mask.copy()
    out = np.zeros(len(mask), bool)
    run = 0.0
    for i, m in enumerate(mask):
        if m:
            run += float(dt[i]) if i < len(dt) else 0.0
            out[i] = run >= min_s
        else:
            run = 0.0
    return out


def count_entries(mask: np.ndarray, dt: np.ndarray, min_dwell_s: float) -> int:
    """Entries that lasted at least ``min_dwell_s``, a nose crossing the
    boundary for one frame is not a visit."""
    entries = 0
    inside = False
    dwell = 0.0
    counted = False
    for i, m in enumerate(mask):
        if m:
            step = float(dt[i]) if i < len(dt) else 0.0
            if not inside:
                inside, dwell, counted = True, step, False
            else:
                dwell += step
            if not counted and dwell >= min_dwell_s:
                entries += 1
                counted = True
        else:
            inside, dwell, counted = False, 0.0, False
    return entries


def _raster_key(space: SpaceModel):
    """What the cached raster was built FROM, names, shapes and frame.

NOT ``id(space.zones)``, the identity of the list. Zones are edited in
    place: renaming a zone whose name collides with a body point changes the
    dicts and not the list, so a raster keyed on the list keeps answering with
    the old names, the Y-maze centre measures as ``Center`` after everything
    else had agreed to call it ``Center_arm``, and the zone filter dropped it.
    Hashing the geometry costs a few dozen tuples against a frame-sized mask
    per zone, which is what the cache exists to avoid.

    ``None`` when the zones cannot be hashed, which disables caching rather
    than risking a stale answer.
    """
    try:
        return (space.frame_size,
                tuple((z.get("name"), z.get("type"),
                       tuple((float(p[0]), float(p[1]))
                             for p in (z.get("points") or ())))
                      for z in space.zones))
    except Exception:                                     # pragma: no cover
        return None


def _raster_for(space: SpaceModel) -> ZoneRaster:
    """One raster per space, reused across calls.

    A fourteen-stage protocol re-zones fourteen times; rebuilding a
    frame-sized mask per zone each time is pure waste, and at 1080p with a
    dozen zones it is tens of megabytes of it.
    """
    key = _raster_key(space)
    cached = getattr(space, "_raster_cache", None)
    if cached is not None and key is not None and cached[0] == key:
        return cached[1]
    raster = ZoneRaster(space.zones, space.frame_size)
    try:
        space._raster_cache = (key, raster)
    except Exception:                                     # frozen model
        pass
    return raster


def rezone(space: SpaceModel, *, body_xy: Tuple[np.ndarray, np.ndarray],
           head_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
           dt: Optional[np.ndarray] = None,
           min_dwell_s: float = 0.2) -> ZoneResult:
    """Recompute occupancy from existing poses. No video, no inference.

    Returns a boolean series per zone for the body point and, when a head
    point is supplied, for the head as well, plus the single ``location``
    string per frame that keeps older readers working. Overlapping zones are
    simply all True at once, which the string never allowed.
    """
    bx, by = np.asarray(body_xy[0], float), np.asarray(body_xy[1], float)
    n = len(bx)
    dt = np.zeros(n) if dt is None else np.asarray(dt, float)
    raster = _raster_for(space)
    res = ZoneResult(names=list(raster.names), n_frames=n)
    res.body = raster.all_series(bx, by)

    if head_xy is not None:
        hx, hy = np.asarray(head_xy[0], float), np.asarray(head_xy[1], float)
        res.head = raster.all_series(hx, hy)
    else:
        hx = hy = None
        res.head = {name: np.zeros(n, bool) for name in raster.names}

    # `location` keeps the single-zone model: the smallest zone containing the
    # body point, so a nested arm beats the arena that contains it.
    areas = {z["name"]: _polygon_area(z.get("points") or [])
             for z in space.zones if z.get("name")}
    order = sorted(res.names, key=lambda nm: areas.get(nm, math.inf))
    loc = [""] * n
    for name in reversed(order):                 # largest first, smallest wins
        m = res.body.get(name)
        if m is None:
            continue
        for i in np.nonzero(m)[0]:
            loc[i] = name
    res.location = loc

    if hx is not None:
        cfg = {**DEFAULT_INTERACTION, **(space.interaction or {})}
        centre = arena_centre(space)
        for z in space.zones:
            if not is_interaction_zone(z) or z.get("name") not in res.names:
                continue
            res.interaction[z["name"]] = _interaction(
                z, res.head[z["name"]], (hx, hy), (bx, by), dt, cfg,
                min_dwell_s, centre)
    return res


def arena_centre(space: SpaceModel) -> Tuple[float, float]:
    """The middle of the arena, which is what "outward" is measured from.

    The frame centre when the frame size is known, a recording is framed on
    its arena, otherwise the centroid of every zone drawn in it.
    """
    w, h = space.frame_size
    if w and h:
        return (w / 2.0, h / 2.0)
    pts = [p for z in space.zones for p in (z.get("points") or [])]
    if not pts:
        return (0.0, 0.0)
    arr = np.asarray([[float(p[0]), float(p[1])] for p in pts], float)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray([[float(p[0]), float(p[1])] for p in points], float)
    if len(pts) < 3:
        return math.inf
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _zone_num(zone: dict, key: str, default) -> float:
    """A per-zone numeric override, or the run's own setting."""
    try:
        value = zone.get(key)
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _interaction(zone: dict, head_in: np.ndarray,
                 head_xy, body_xy, dt: np.ndarray, cfg: dict,
                 min_dwell_s: float,
                 centre: Tuple[float, float] = (0.0, 0.0)) -> InteractionResult:
    hx, hy = head_xy
    bx, by = body_xy
    n = len(hx)
    confirm_s = float(cfg.get("confirm_ms", 100.0)) / 1000.0
    thresh = float(cfg.get("angle_threshold_deg", 45.0))
    mode = str(cfg.get("target", "outer_edge"))
    # Per zone, falling back to the run's setting: one object may stand
    # further into the arena than another, and the strip drawn for it is a
    # different depth. The zone dict is where that belongs, beside the shape.
    extend_px = _zone_num(zone, "ia_line_extend_px",
                          cfg.get("line_extend_px", 15.0))
    offset_px = _zone_num(zone, "ia_inner_offset_px",
                          cfg.get("inner_offset_px", 50.0))
    pts = zone.get("points") or []
    centroid = polygon_centroid(pts)
    edge = outer_edge(pts, centre) if mode == "outer_edge" else None
    if edge is not None and extend_px > 0:
        edge = extend_segment(edge[0], edge[1], extend_px)

    # The inner gate: the head has to be at least `offset_px` inside the
    # outward face before the visit counts. Being in the strip is not being at
    # the object, the strip is drawn big enough for the animal to stand in.
    reached = head_in
    if edge is not None and offset_px > 0:
        normal = inward_normal(edge[0], edge[1], centre)
        if normal is not None:
            inner = (edge[0][0] + normal[0] * offset_px,
                     edge[0][1] + normal[1] * offset_px)
            near = np.zeros(n, bool)
            for i in np.nonzero(head_in)[0]:
                if np.isfinite(hx[i]) and np.isfinite(hy[i]):
                    # Past the inner line means BEYOND it toward the object,
                    # i.e. NOT on the arena-centre side.
                    near[i] = not past_line((float(hx[i]), float(hy[i])),
                                            inner, normal)
            reached = near

    confirmed = _confirm(reached, dt, confirm_s)
    angles = np.full(n, math.nan)
    facing = np.zeros(n, bool)
    for i in np.nonzero(confirmed)[0]:
        if not (np.isfinite(hx[i]) and np.isfinite(bx[i])):
            continue
        if edge is not None:
            target = nearest_point_on_segment(edge[0], edge[1],
                                              float(hx[i]), float(hy[i]))
        elif mode == "centroid":
            target = centroid
        else:
            target = nearest_point_on_polygon(pts, float(hx[i]), float(hy[i]))
        a = facing_angle_deg((float(hx[i]), float(hy[i])),
                             (float(bx[i]), float(by[i])), target)
        angles[i] = a
        if not math.isnan(a) and a <= thresh:
            facing[i] = True

    return InteractionResult(
        zone=str(zone.get("name")), in_zone=confirmed, facing=facing,
        angle_deg=angles,
        time_s=float(np.sum(dt[confirmed])) if n else 0.0,
        facing_time_s=float(np.sum(dt[facing])) if n else 0.0,
        entries=count_entries(confirmed, dt, min_dwell_s),
        facing_entries=count_entries(facing, dt, min_dwell_s),
    )


# ── region aggregation ───────────────────────────────────────────────────────

def aggregate_regions(result: ZoneResult, regions: Dict[str, Sequence[str]]
                      ) -> ZoneResult:
    """Combine sub-zones into named regions (``OpenArm`` from four segments).

    Papers report regions, not the mesh of sub-zones an experimenter drew to
    make the mesh tile the arena. The combined series is the OR of the parts,
    so time in a region is never double-counted.
    """
    out = ZoneResult(names=[], body={}, head={}, location=list(result.location),
                     interaction=dict(result.interaction),
                     n_frames=result.n_frames)
    for region, parts in regions.items():
        present = [p for p in parts if p in result.names]
        if not present:
            continue
        out.names.append(region)
        out.body[region] = np.logical_or.reduce(
            [result.body[p] for p in present])
        out.head[region] = np.logical_or.reduce(
            [result.head[p] for p in present])
    return out


def poses_fit_frame(xy, frame_size, tol: float = 0.02) -> bool:
    """Whether these pose coordinates live in this frame at all.

    Normalized zones are turned into pixels by multiplying by the frame the
    recording DECLARES. On a rig that encodes the video smaller than the
    camera reads it, the poses are in camera pixels and the declared
    resolution is the video's, so the zones land somewhere else entirely and
    every zone number is quietly wrong.

    A few stray keypoints outside the frame are normal (a nose at the edge, a
    bad detection). A steady fraction of them is a different coordinate space.
    """
    x, y = xy
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any() or not frame_size or not all(frame_size):
        return True
    w, h = float(frame_size[0]), float(frame_size[1])
    outside = ((x[ok] < -1) | (x[ok] > w + 1)
               | (y[ok] < -1) | (y[ok] > h + 1))
    return float(np.mean(outside)) <= tol


def suggest_regions(names: Sequence[str],
                    min_parts: int = 2) -> Dict[str, List[str]]:
    """Guess which zones are parts of one region, from their names.

    An experimenter who needs an elevated-plus maze to tile cleanly draws
    ``Open_1``/``Open_2`` or ``arm_left_a``/``arm_left_b``; the paper reports
    one open-arm number. The shared prefix is the region, and this only ever
    proposes, nothing is grouped until the user says so.

    Splits on the separator actually used, so ``Left_arm_1`` and ``Left_arm_2``
    suggest ``Left_arm`` rather than ``Left``.
    """
    import re

    groups: Dict[str, List[str]] = {}
    for n in names:
        # A trailing NUMBER may hug the name (Open1) or be separated
        # (Open_1). A trailing LETTER must be separated (arm_a), or "Center"
        # would suggest a region called "Cente".
        stem = re.sub(r"[\s_\-]*[0-9]+$", "", str(n))
        if stem == str(n):
            stem = re.sub(r"[\s_\-][A-Za-z]$", "", str(n))
        stem = stem.rstrip(" _-")
        if not stem or stem == n:
            continue
        groups.setdefault(stem, []).append(n)
    return {k: sorted(v) for k, v in sorted(groups.items())
            if len(v) >= min_parts}


def region_masks(result: "ZoneResult", regions: Dict[str, Sequence[str]]
                 ) -> Dict[str, Dict[str, np.ndarray]]:
    """``{region: {"body": mask, "head": mask, "parts": [...]}}``.

    Thin wrapper over :func:`aggregate_regions` that keeps the part list, so a
    caller can say WHICH zones a region turned out to cover, a region whose
    parts are all missing must not silently become a column of zeros.
    """
    agg = aggregate_regions(result, regions)
    out: Dict[str, Dict[str, Any]] = {}
    for region in agg.names:
        out[region] = {
            "body": agg.body[region],
            "head": agg.head[region],
            "parts": [p for p in regions[region] if p in result.names],
        }
    return out


def infer_pose_frame(header, xy, locations=None, zones_norm=None):
    """The frame the POSES are in, when the recording does not declare it.

    A rig that reads the camera at one size and encodes the video smaller
    writes poses in camera pixels and `resolution` as the video's. Nothing in
    the file says so. Drawn straight onto the video every keypoint lands at
    `camera/video` times its true position -- markers beside the animal
    rather than on it -- and normalized zones multiplied by the video size
    land somewhere the animal never goes.

    `pose_resolution` records this for anything written since; these are the
    recordings made before it existed. The scale is recovered rather than
    guessed from a list of common resolutions, and it is recovered from the
    file's OWN ground truth: `location` was decided at record time, in the
    pose space, against these same normalized zones. The scale that puts the
    poses back in the zones the recording named is the scale the rig used.

    Returns ``(w, h)`` -- always a multiple of the declared frame, since the
    video is a scaled copy of it -- or ``None`` when there is no evidence.
    Ambiguity returns None: a wrong correction is worse than none, because
    none is at least visible as an offset.
    """
    declared = tuple(getattr(header, "pose_resolution", (0, 0)) or (0, 0))
    if declared and declared[0] and declared[1]:
        return (int(declared[0]), int(declared[1]))

    base = tuple(getattr(header, "resolution", (0, 0)) or (0, 0))
    if not (base and base[0] and base[1]):
        return None
    W, H = float(base[0]), float(base[1])

    x = np.asarray(xy[0], float)
    y = np.asarray(xy[1], float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return None
    x, y = x[ok], y[ok]

    # Already inside the declared frame: nothing to recover.
    if poses_fit_frame((x, y), base):
        return (int(W), int(H))

    # The scale must at least contain the poses. Below that the animal would
    # be outside its own arena, which no scale can make true.
    lo = max(1.0, float(max(x.max() / W, y.max() / H)))
    cands = [round(lo + i * 0.01, 2) for i in range(0, 201)]

    if locations is not None and zones_norm:
        loc = np.asarray(locations, object)[ok]
        polys = {z.get("name"): np.asarray(z.get("points") or [], float)
                 for z in zones_norm if z.get("points")}
        named = np.array([str(v).strip() for v in loc])
        scored = named != ""
        scored &= ~np.isin(np.char.lower(named.astype(str)),
                           ["none", "na", "nan", "-"])
        if scored.sum() >= 20:
            best, best_hit = None, 0.0
            for k in cands:
                hit = _agreement(x[scored], y[scored], named[scored],
                                 polys, W * k, H * k)
                if hit > best_hit:
                    best, best_hit = k, hit
            # A real recovery agrees with the recording almost everywhere.
            # Anything less is a coincidence, and a coincidence must not be
            # allowed to move the animal.
            if best is not None and best_hit >= 0.80:
                return (int(round(W * best)), int(round(H * best)))
            return None

    # No usable `location` column: the poses themselves are the only
    # evidence, and they can only say how far out the space reaches.
    return (int(round(W * lo)), int(round(H * lo)))


def _agreement(x, y, named, polys, w, h) -> float:
    """Fraction of frames whose pose falls in the zone the file recorded."""
    try:
        from matplotlib.path import Path as _Path
    except Exception:                                    # pragma: no cover
        _Path = None
    n = np.column_stack((x / float(w), y / float(h)))
    hit = np.zeros(len(n), bool)
    for name, poly in polys.items():
        if len(poly) < 3:
            continue
        inside = (_Path(poly).contains_points(n) if _Path is not None
                  else _inside_polygon(n, poly))
        hit |= inside & (named == name)
    return float(hit.mean()) if len(hit) else 0.0


def _inside_polygon(pts, poly) -> np.ndarray:
    """Even-odd point-in-polygon, so this works without matplotlib."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        crosses = ((y0 > y) != (y1 > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x1 - x0) * (y - y0) / (y1 - y0) + x0
        inside ^= crosses & (x < xint)
    return inside
