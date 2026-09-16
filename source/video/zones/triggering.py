"""
Zone Manager for pyOperant

Owns zone geometry + topology for tracking-based zone occupancy queries.
Live trigger evaluation runs in the sinks (``TrackingPushPolicy`` /
``PoseSink``), which consume the precomputed occupancy lookups below,
this class only stores the zones, precomputes their nesting topology, and
answers "which zones contain this point?".

Features:
- Precomputed zone topology (nesting, overlap, depth)
- Fast innermost-first point-in-zone queries via the Zone Shapely cache
"""

import logging
from typing import Dict, List, Optional

# Zone is the canonical dataclass in source/video/zones/schema.py.
from source.video.zones.schema import Zone

logger = logging.getLogger(__name__)

# Availability probe only: the Zone class owns its own Shapely cache
# (source.video.zones.schema); topology building gates on this bool.
try:
    import shapely.geometry  # noqa: F401
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not installed. Zone checking will be disabled. Install with: pip install shapely")


class ZoneManager:
    """
    Manages zones for tracking-based zone occupancy.

    Precomputes zone topology (nesting tree, overlap groups, depth ordering)
    at zone mutation time so per-frame occupancy checks use the optimal
    iteration order (innermost-first).

    Usage:
        manager = ZoneManager()
        manager.add_zone(zone)
        zones = manager.get_zones_at_point(x, y)   # innermost-first
        innermost = manager.deepest_zone_at_point(x, y)
    """

    def __init__(self):
        self.zones: Dict[str, Zone] = {}  # name -> Zone

        # Topology data (precomputed at zone mutation time)
        self._zone_depth: Dict[str, int] = {}              # zone_name -> nesting depth (0=outermost)
        self._zone_parent: Dict[str, Optional[str]] = {}   # zone_name -> parent zone name
        self._zones_sorted: List[str] = []                 # all zone names, innermost-first

    # ============ Zone Management ============

    def add_zone(self, zone: Zone) -> None:
        """Add or update a zone."""
        self.zones[zone.name] = zone
        logger.debug(f"Added zone '{zone.name}' with {len(zone.points)} points")
        self._build_topology()

    def clear_zones(self) -> None:
        """Remove all zones."""
        self.zones.clear()
        self._zone_depth.clear()
        self._zone_parent.clear()
        self._zones_sorted.clear()
        logger.debug("Cleared all zones")

    # ============ Topology Precomputation ============

    def _build_topology(self) -> None:
        """Precompute zone nesting depths, parents, and innermost-first order.

        Called at zone add/remove time, NOT per-frame.
        Uses Shapely polygon-polygon containment (not point-in-polygon).
        """
        names = list(self.zones.keys())
        n = len(names)

        # Reset topology
        self._zone_depth = {}
        self._zone_parent = dict.fromkeys(names)
        self._zones_sorted = []

        if n == 0:
            return

        if not SHAPELY_AVAILABLE:
            # Without Shapely, flat topology (all depth 0)
            self._zone_depth = dict.fromkeys(names, 0)
            self._zones_sorted = names[:]
            return

        # Step 1: Containment matrix, A contains B (polygon-polygon)
        # contains_map[a][b] = True means zone 'a' fully contains zone 'b'
        contains_map: Dict[str, Dict[str, bool]] = {a: {} for a in names}
        area_map: Dict[str, float] = {}

        for name in names:
            zone = self.zones[name]
            if zone._polygon is not None:
                area_map[name] = zone._polygon.area
            else:
                area_map[name] = 0.0

        for i, a_name in enumerate(names):
            a_zone = self.zones[a_name]
            if a_zone._polygon is None:
                continue
            for j, b_name in enumerate(names):
                if i == j:
                    continue
                b_zone = self.zones[b_name]
                if b_zone._polygon is None:
                    continue
                try:
                    contains_map[a_name][b_name] = a_zone._polygon.contains(b_zone._polygon)
                except Exception:
                    contains_map[a_name][b_name] = False

        # Step 2: Parent assignment, parent is the smallest-area container
        for child in names:
            best_parent = None
            best_area = float('inf')
            for candidate in names:
                if candidate == child:
                    continue
                if contains_map.get(candidate, {}).get(child, False):
                    cand_area = area_map.get(candidate, 0.0)
                    if cand_area < best_area:
                        best_area = cand_area
                        best_parent = candidate
            self._zone_parent[child] = best_parent

        # Step 3: Depth, walk parent chains
        for name in names:
            depth = 0
            current = name
            visited = set()
            while self._zone_parent.get(current) is not None:
                current = self._zone_parent[current]
                depth += 1
                if current in visited:
                    logger.warning(f"Cycle detected in zone topology at '{current}'")
                    break
                visited.add(current)
            self._zone_depth[name] = depth

        # Step 4: Sorted order, innermost first (descending depth), then alphabetical
        self._zones_sorted = sorted(
            names,
            key=lambda name: (-self._zone_depth.get(name, 0), name)
        )
        logger.debug("Topology built: %d zones", n)

    # ============ Topology Query API ============

    def get_zones_at_point(self, x: float, y: float) -> List[str]:
        """Get zones containing point, ordered by specificity (innermost first).

        When mouse is at Reward_port (depth 2), returns:
        ["Reward_port", "Quadrant_NW", "Arena"]
        """
        containing = []
        for name in self._zones_sorted:
            zone = self.zones.get(name)
            if zone and zone.enabled and zone.contains(x, y):
                containing.append(name)
        return containing

    def deepest_zone_at_point(self, x: float, y: float):
        """Return only the innermost zone containing (x,y), or None.

        ``_zones_sorted`` is ordered innermost-first (depth desc), so the
        first hit is the deepest zone. Early termination saves work in the
        common case where the animal is in a specific small zone, no need
        to walk the rest of the list. Activates spatially-aware nested-zone
        semantics in PoseSink (per-body-part trigger events fire on the
        innermost transition, not on every containing zone).
        """
        for name in self._zones_sorted:
            zone = self.zones.get(name)
            if zone and zone.enabled and zone.contains(x, y):
                return name
        return None
