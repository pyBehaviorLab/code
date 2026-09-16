"""MCU output bridge, pose/tracker results → pycboard.

* :class:`TrackingPushPolicy` (the rules)
      Per-box policy deciding what to send to the MCU each tick:
          coord_mapping: {coord_name -> body_part} → queue_set_coordinates
          triggers: [{condition, body_part, zones, event_name}] → queue_trigger_event
      Dirty-flagged: coords pushed only on change; trigger events fire
      only on the false→true transition (one event per zone entry, not
      one per frame). Shared by maze and operant; only the config differs.

* :class:`MCUPusher` (the wiring)
      Not a frame Sink, subscribes to :meth:`PoseSink.on_result` and
      :meth:`TrackerSink.on_result` and dispatches each result through
      the box's :class:`TrackingPushPolicy`. Per-box state:
      ``{box_id -> pycboard}`` + ``{box_id -> policy}``. Pipeline calls
      :meth:`register_box` first; :meth:`configure_zones` /
      :meth:`configure_tracking` seed the policy from the dialog config.

Threading: result callbacks fire on the source sink's worker thread.
The queue_* wrappers enqueue onto pycboard's write queue (drained
serially with the reader, no port race). Locking here is per-box dict
access only.
"""

from __future__ import annotations

import logging
import math
import threading
from source import host_clock
from typing import Any, Dict, List, Optional

# Leaf modules (import nothing from framebus), hoisted so the per-rule
# trigger loop doesn't pay a sys.modules lookup per pose frame.
from source.video.features import FeatureExtractor
from source.video.trigger_engine import RuleState, TriggerFrame

logger = logging.getLogger(__name__)


# Sentinel so a value of ``None`` from _resolve_coord still counts as
# "changed" on first sight.
_SENTINEL = object()

# Fallback pose-confidence threshold for boxes that haven't been configured
# yet; matches PoseSink's default. Configured boxes use their own
# TrackingConfig.confidence_threshold (see MCUPusher._pose_conf_thresh).
_POSE_PUSH_CONF_THRESH = 0.5

# Value pushed for an X/Y component when the body part was NOT detected this
# frame (below confidence / occluded). Pixel coordinates are always >= 0, so
# a negative sentinel unambiguously means "lost", a task reads
# ``if c.head_x < 0: ...`` to tell "not detected" from "really at the origin".
# It survives repr()/eval() serialization to the board (NaN would not).
COORD_LOST = -1.0


# =============================================================================
#  TrackingPushPolicy, the rules
# =============================================================================


class TrackingPushPolicy:
    """Per-box policy that maps tracker output → MCU set_coordinates / trigger_event."""

    def __init__(self,
                 coord_mapping: Optional[Dict[str, str]] = None,
                 triggers: Optional[List[Dict[str, Any]]] = None,
                 push_zone_changed: bool = True,
                 push_frame_event: bool = False,
                 zone_change_body_part: str = "centroid"):
        # {coord_name: body_part}, e.g. {"loc_center": "centroid", "speed": "centroid"}
        self.coord_mapping: Dict[str, str] = dict(coord_mapping or {})
        # [{condition, body_part, zones, event_name}, ...]
        self.triggers: List[Dict[str, Any]] = list(triggers or [])
        # Gate for the intrinsic ``zone_changed`` event, mirroring the
        # per-box ``TrackingConfig.push_zones_to_mcu`` flag: off means the
        # MCU sees no zone activity. Set by MCUPusher.configure_tracking.
        self.push_zone_changed: bool = bool(push_zone_changed)
        # Push-time gate for c.* coordinate writes, mirroring
        # ``TrackingConfig.push_coords_to_mcu``. Gating only at
        # coord_mapping BUILD time (configure_from_zones) left the
        # dialog-merged mappings pushing with the toggle off.
        self.push_coords: bool = True
        # Gate for the per-frame intrinsic ``frame_event`` (fires every push
        # when ON). Mirrors ``TrackingConfig.push_frame_event``; default OFF.
        self.push_frame_event: bool = bool(push_frame_event)
        # Body part defining the intrinsic ``zone_changed`` event. Default
        # "centroid" (PoseSink synthesises it for every tracker mode); the
        # user can pick any keypoint via the tracking dialog.
        self.zone_change_body_part: str = str(
            zone_change_body_part or "centroid")
        # Dirty caches
        self._coord_cache: Dict[str, Any] = {}
        self._trigger_state: Dict[str, bool] = {}
        # Per-event in-zone history for exit_edge detection (None=unseen).
        self._in_zone_history: Dict[str, Optional[bool]] = {}
        # Previous-tick zone-occupancy snapshot for zone_change_body_part.
        # frozenset() = in no zone; None = body part not yet observed
        # (establish baseline, don't fire on first observation).
        self._prev_zone_set_for_change: Optional[frozenset] = None
        # One-shot warning when the MCU's sm.events lacks "zone_changed"
        # (stale framework missing the auto-injection); tells the operator
        # to re-upload pyControl.
        self._warned_no_zone_changed: bool = False
        # One-shot info log confirming the first ``zone_changed`` fired.
        self._logged_first_zone_changed: bool = False
        # Same one-shot guards for the per-frame ``frame_event``.
        self._warned_no_frame_event: bool = False
        self._logged_first_frame_event: bool = False
        # Kinematic features (speed/turning/angle/elongation/…) for the
        # advanced conditions. One extractor per box; updated each push.
        self._features = FeatureExtractor()
        # Per-event "held ≥ T" counter for duration-gated boolean conditions
        # (freezing / rearing / facing), keyed by event_name → first-true t_s.
        self._cond_since: Dict[str, Optional[float]] = {}
        # Observer for the per-frame trigger state (annotation + Session-Plot
        # lane). Set by MCUPusher; receives a TriggerFrame-like object.
        self._trigger_frame_cb = None
        self._setup_id: int = 0
        # Live push counters, the observable answer to "is anything actually
        # reaching the MCU?" (coord writes / zone_changed events / trigger
        # events since the last reset, plus the wall-clock of the last push).
        # zone_changed and coord writes produce no TSV row by design, so
        # without these the whole chain can be working and look dead (M0).
        self._stats = {"coords": 0, "zone_events": 0,
                       "triggers": 0, "last_push_ns": 0}

    def set_trigger_frame_cb(self, cb, setup_id: int = 0) -> None:
        self._trigger_frame_cb = cb
        self._setup_id = int(setup_id)

    def set_features_context(self, px_per_mm=None, axis=None) -> None:
        """Real-unit scale (px per mm) + body-axis keypoints for the extractor."""
        if px_per_mm is not None:
            self._features.px_per_mm = float(px_per_mm) if px_per_mm else None
        if axis:
            self._features.axis = tuple(axis)

    # ── Configuration ───────────────────────────────────────────────

    def update(self,
               coord_mapping: Optional[Dict[str, str]] = None,
               triggers: Optional[List[Dict[str, Any]]] = None):
        """Replace coord_mapping / triggers (e.g. after Zone Config save).

        Keeps the push counters: a mid-session dialog Apply reconfigures
        the policy but is not a new run, and zeroing here meant the
        counters could never span a session.
        """
        if coord_mapping is not None:
            self.coord_mapping = dict(coord_mapping)
        if triggers is not None:
            self.triggers = list(triggers)
        self.reset(clear_stats=False)

    def configure_from_zones(self, zones, default_body_part: str = "centroid",
                             *, push_coords: bool = True,
                             push_zone_events: bool = True):
        """Translate per-zone transmit fields into coord_mapping + triggers,
        the ONE zone→policy translator.

        Each Zone object in `zones` carries:
          transmit_mode: "coord" | "event_enter" | "event_enter_exit"
          coord_var, event_on_enter, event_on_exit

        Coord-mode zones with the same coord_var contribute to a single
        coord_mapping entry (the body_part is the shared default);
        gated off entirely by ``push_coords`` (the dialog's
        push-coords-to-MCU toggle).

        Event-mode zones become trigger entries, gated by
        ``push_zone_events``:
          event_enter      → one trigger with condition="in_zone"
          event_enter_exit → two triggers, "in_zone" + "exit_edge"
        """
        self.push_coords = bool(push_coords)
        coord_mapping: Dict[str, str] = {}
        triggers: List[Dict[str, Any]] = []
        for z in zones:
            mode = getattr(z, "transmit_mode", "coord")
            if mode == "coord" and push_coords:
                cv = getattr(z, "coord_var", "loc_center") or "loc_center"
                # The resolver's naming convention claims x/y-suffixed
                # names as pixel components BEFORE zone-name resolution,
                # a coord-mode zone named e.g. "zone_x" would silently
                # push a pixel float instead of the zone-name string.
                cvl = cv.lower()
                if cvl in ("x", "y") or cvl.endswith(("_x", "_y")):
                    logger.warning(
                        "Zone %r: coord_var %r ends in a pixel-component "
                        "suffix; it will carry the body part's pixel "
                        "coordinate, NOT the zone name. Rename it if you "
                        "wanted zone-name transmission.",
                        getattr(z, "name", "?"), cv)
                # Last-writer-wins if multiple zones target the same coord;
                # push-time resolution picks the zone the body_part is in.
                coord_mapping[cv] = default_body_part
            elif mode in ("event_enter", "event_enter_exit") and push_zone_events:
                if z.event_on_enter:
                    triggers.append({
                        "condition": "in_zone",
                        "body_part": default_body_part,
                        "zones":     [z.name],
                        "event_name": z.event_on_enter,
                    })
                if mode == "event_enter_exit" and z.event_on_exit:
                    triggers.append({
                        "condition": "exit_edge",
                        "body_part": default_body_part,
                        "zones":     [z.name],
                        "event_name": z.event_on_exit,
                    })
            elif mode not in ("coord", "event_enter", "event_enter_exit"):
                # A typo'd or null transmit_mode otherwise makes the zone
                # transmit nothing, with no diagnostic at all.
                logger.warning(
                    "Zone %r: unknown transmit_mode %r, zone will not "
                    "push anything (use coord / event_enter / "
                    "event_enter_exit).", getattr(z, "name", "?"), mode)
        self.coord_mapping = coord_mapping
        self.triggers = triggers
        # Run-start freshness comes from policy re-creation (register_box /
        # unregister_box), not from here, a dialog Apply routes through
        # this translator and must not zero the session's push counters.
        self.reset(clear_stats=False)

    def reset(self, *, clear_stats: bool = True):
        """Clear caches, call on framework start to avoid stale state.

        ``clear_stats=False`` (mid-session reconfigure) keeps the push
        counters running across the config change.
        """
        self._coord_cache.clear()
        self._trigger_state.clear()
        self._cond_since.clear()
        # Per-event in-zone history for exit_edge detection (None=unseen).
        self._in_zone_history: Dict[str, Optional[bool]] = {}
        # zone_changed previous-occupancy snapshot (single body part).
        self._prev_zone_set_for_change = None
        if clear_stats:
            # Fresh counters for the new run.
            self._stats = {"coords": 0, "zone_events": 0,
                           "triggers": 0, "last_push_ns": 0}

    def stats(self) -> Dict[str, Any]:
        """Snapshot of live push counters + policy shape for the health panel.

        ``last_push_age_s`` is the wall-clock since the last successful queue
        to the board (None if nothing has been pushed), the direct answer to
        "is it still sending?".
        """
        last = self._stats.get("last_push_ns", 0)
        age = None if not last else (host_clock.host_ns() - last) / 1e9
        return {
            "coords_pushed":   int(self._stats.get("coords", 0)),
            "zone_events":     int(self._stats.get("zone_events", 0)),
            "triggers_fired":  int(self._stats.get("triggers", 0)),
            "last_push_age_s": age,
            "n_coord_maps":    len(self.coord_mapping),
            "n_triggers":      len(self.triggers),
        }

    # ── Per-tick push ───────────────────────────────────────────────

    def push(self, pycboard,
             zones_by_body_part: Dict[str, Any],
             speed: float = 0.0,
             body_part_coords: Optional[Dict[str, Any]] = None):
        """Apply policy: push changed coords, fire newly-true triggers.

        Args:
            pycboard: object exposing set_coordinates(name, value) and
                trigger_event(name). Can be a real Pycboard or any proxy
                with the same interface.
            zones_by_body_part: {body_part: {zone_name: bool}} or
                {body_part: zone_name_str}.  Empty dict if no occupancy.
            speed: float scalar, only used when a coord is mapped to "speed".
            body_part_coords: ``{body_part: (x, y)}`` of accepted (above-
                conf) keypoints in pose-pipeline pixel coords. Used by
                ``_resolve_coord`` when a coord_name resolves to a
                pixel component (``x``/``y`` or ``<bp>_x``/``<bp>_y``).
        """
        if pycboard is None:
            return
        # Non-Pycboard proxies don't track this; default True keeps them active.
        if not getattr(pycboard, "framework_running", True):
            return

        # Order is the contract. The c.* coords MUST reach the MCU before any
        # event: a task's zone_changed / per-zone handler reads c.loc_center
        # the instant the event arrives, and writes drain FIFO, so an event
        # queued first would be handled against the PREVIOUS zone, and an
        # "enter and stay" would never re-fire.
        self._push_coords(pycboard, zones_by_body_part, speed,
                          body_part_coords)
        self._push_frame_event(pycboard)
        self._push_zone_changed(pycboard, zones_by_body_part)

        # Kinematic features feed the advanced trigger conditions
        # (speed / turning / angle / elongation), so they read one
        # consistent smoothed state. The update copies the keypoint dict,
        # skip it entirely when nothing consumes features (no trigger
        # rules AND no trigger-frame subscriber).
        if self.triggers or self._trigger_frame_cb is not None:
            # The shared clock, because this is a RATE, not a timeout: the
            # feature tracker differences successive `now_s` values to get
            # speed, turning and angular velocity, and the trigger rules fire
            # on those. On a 15.6 ms clock at 20 fps the dt between two frames
            # 50 ms apart reads as 46.9 or 62.5, so every speed derived from
            # it was wrong by up to a third, and the thresholds are set in
            # cm/s by the operator.
            now_s = host_clock.host_s()
            coords = body_part_coords or {}
            self._features.update(coords, coords.get("centroid"),
                                  float(speed), now_s)
            states = self._push_triggers(pycboard, zones_by_body_part, now_s)
            self._emit_trigger_frame(states)

    # ── push steps ──────────────────────────────────────────────────

    def _push_coords(self, pycboard, zones_by_body_part, speed,
                     body_part_coords) -> None:
        """Write each mapped coord, but only when its value actually changed.

        Dirty-flagging matters: these run at pose rate, and re-sending an
        unchanged zone name every frame would saturate the serial link.
        """
        if not self.push_coords:
            return
        for coord_name, body_part in self.coord_mapping.items():
            new_val = self._resolve_coord(coord_name, body_part,
                                          zones_by_body_part, speed,
                                          body_part_coords)
            if self._coord_cache.get(coord_name, _SENTINEL) != new_val:
                self._coord_cache[coord_name] = new_val
                pycboard.queue_set_coordinates(coord_name, new_val)
                self._stats["coords"] += 1
                self._stats["last_push_ns"] = host_clock.host_ns()

    def _mcu_has_event(self, pycboard, name: str, warn_flag: str,
                       message: str) -> bool:
        """Whether the board's state machine knows this intrinsic event.

        A board running an older pyControl never had the event auto-injected,
        so the push would be dropped MCU-side with no sign here. Warn once per
        event kind; this is called every frame.
        """
        sm_info = getattr(pycboard, "sm_info", None)
        if name in (getattr(sm_info, "events", None) or {}):
            return True
        if not getattr(self, warn_flag, False):
            logger.warning(message)
            setattr(self, warn_flag, True)
        return False

    def _push_frame_event(self, pycboard) -> None:
        """Fire the per-frame intrinsic event, when the operator enabled it.

        Unlike zone_changed this fires on EVERY push, so a task can poll c.*
        each frame instead of only on zone-change edges. Silent (wire byte
        b'Z'), so it never reaches the TSV. OFF by default.
        """
        if not self.push_frame_event:
            return
        if not self._mcu_has_event(
                pycboard, "frame_event", "_warned_no_frame_event",
                "TrackingPushPolicy: push_frame_event=ON but the MCU's "
                "sm.events has no 'frame_event', re-upload the pyControl "
                "framework (state_machine.py) so the intrinsic event "
                "auto-injects. Frame events dropped until then."):
            return
        try:
            pycboard.queue_trigger_intrinsic_event("frame_event")
        except Exception:
            return
        # Counted like every other push: uncounted, last_push_age_s reads
        # None on a rig with only frame_event enabled, while the board is in
        # fact being written every frame.
        self._stats["frame_events"] = self._stats.get("frame_events", 0) + 1
        self._stats["last_push_ns"] = host_clock.host_ns()
        if not self._logged_first_frame_event:
            logger.info("TrackingPushPolicy: 'frame_event' firing to MCU per "
                        "pose frame (silent, no TSV row).")
            self._logged_first_frame_event = True

    def _occupied_zones(self, zones_by_body_part) -> frozenset:
        """Zones the zone-change body part is in right now.

        Read with .get(), not iteration: PoseSink drops the key entirely when
        the part is in no zone, so "key absent" must map to "in no zone" or
        the falling edge is missed.
        """
        bp_zones = (zones_by_body_part or {}).get(self.zone_change_body_part)
        if isinstance(bp_zones, dict):
            return frozenset(z for z, inside in bp_zones.items() if inside)
        if isinstance(bp_zones, str):
            return frozenset((bp_zones,)) if bp_zones else frozenset()
        return frozenset()          # body part absent OR in no zone

    def _push_zone_changed(self, pycboard, zones_by_body_part) -> None:
        """Fire the intrinsic zone_changed event on entry OR exit.

        Other body parts use per-zone triggers; this is the configured
        zone-change part only. The first observation just establishes the
        baseline, firing on it would report a change that never happened.
        """
        in_set = self._occupied_zones(zones_by_body_part)
        prev = self._prev_zone_set_for_change
        if prev is None:
            self._prev_zone_set_for_change = in_set
            return
        if prev == in_set:
            return
        self._prev_zone_set_for_change = in_set
        if not self.push_zone_changed:
            return
        # Warn, but push anyway, unlike frame_event this is NOT a gate. The
        # task's own sm.events need not list zone_changed for the framework to
        # accept it, so refusing here would break boards that work.
        self._mcu_has_event(
            pycboard, "zone_changed", "_warned_no_zone_changed",
            "TrackingPushPolicy: zone occupancy changed AND "
            "push_zones_to_mcu=ON, but the MCU's sm.events does NOT "
            "contain 'zone_changed', re-upload the pyControl framework "
            "files (state_machine.py + framework.py) so the "
            "auto-injection of the intrinsic 'zone_changed' event takes "
            "effect. Zone events will be silently dropped on the MCU "
            "side until then.")
        try:
            # Silent dispatch, like entry/exit, zone_changed is processed by
            # the MCU state machine but not logged to the TSV (add a print()
            # in the state handler for visibility). Wire byte b'Z'.
            pycboard.queue_trigger_intrinsic_event("zone_changed")
            self._stats["zone_events"] += 1
            self._stats["last_push_ns"] = host_clock.host_ns()
        except Exception:
            return
        if not self._logged_first_zone_changed:
            logger.info("TrackingPushPolicy: 'zone_changed' fired to MCU "
                        "silently (body_part=%s, push_zones_to_mcu=ON), no "
                        "TSV row, like entry/exit",
                        self.zone_change_body_part)
            self._logged_first_zone_changed = True

    def _push_triggers(self, pycboard, zones_by_body_part,
                       now_s: float) -> list:
        """Evaluate every trigger, fire the newly-true ones, and return the
        per-rule states the annotation / Session-Plot lane reads.

        One evaluation feeds both consumers, so the overlay can never disagree
        with what was actually sent to the board.
        """
        states = []
        want_states = self._trigger_frame_cb is not None
        for i, trig in enumerate(self.triggers):
            event_name = trig.get("event_name")
            # Index-qualified: an event_enter_exit zone whose enter and
            # exit share one event name is TWO rules, a bare event_name
            # key made their edge detectors overwrite each other's
            # history every frame.
            key = f"{event_name or '_t'}#{i}"
            in_zone_now = self._eval_in_zone(trig, zones_by_body_part)
            active, value, geom = self._evaluate_condition(
                trig, zones_by_body_part, now_s, rule_key=key)

            if trig.get("condition", "in_zone") == "exit_edge":
                # Inverted edge: fire on the frame the part LEAVES the zone.
                fire = self._in_zone_history.get(key) is True and not in_zone_now
            else:
                fire = active and not self._trigger_state.get(key, False)
                self._trigger_state[key] = active

            if fire and event_name:
                pycboard.queue_trigger_event(event_name)
                self._stats["triggers"] += 1
                self._stats["last_push_ns"] = host_clock.host_ns()
            self._in_zone_history[key] = in_zone_now

            if want_states:
                states.append(RuleState(
                    id=event_name or f"rule{i}",
                    name=trig.get("name") or event_name or f"rule{i}",
                    active=active, fired=fire, value=value, geom=geom,
                    color=trig.get("color", "#39c5cf"),
                    show_on_video=bool(trig.get("show_on_video", True)),
                    plot=bool(trig.get("plot", True)),
                    threshold=trig.get("threshold")))
        return states

    def _emit_trigger_frame(self, states: list) -> None:
        """Hand this frame's rule states to the plot / annotation lane."""
        if self._trigger_frame_cb is None:
            return
        try:
            self._trigger_frame_cb(TriggerFrame(
                setup_id=self._setup_id, cam_frame_id=0, states=states))
        except Exception as e:
            logger.debug("trigger frame cb: %s", e)

    # ── Internal ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_coord(coord_name: str, body_part: str,
                       zones_by_body_part: Dict[str, Any],
                       speed: float,
                       body_part_coords: Optional[Dict[str, Any]] = None):
        """Resolve a coord_name → MCU value.

        Naming convention:
          - ``speed`` → scalar speed (px/s).
          - ``x`` / ``y`` (case-insensitive) → the picked body_part's
            X / Y pixel coord, in pose-pipeline space (same coord
            system as zones).
          - ``<anything>_x`` / ``<anything>_y`` → same as ``x`` / ``y``
            (suffix lets the user name the variable freely, e.g.
            ``head_x``, ``nose_x``).
          - anything else → name of the zone the body_part is currently
            in, or ``""`` if not in any zone.

        Returns ``COORD_LOST`` (-1.0) when an X/Y component is requested but
        the body_part wasn't detected this frame (e.g. below confidence
        threshold) so a task can distinguish "lost" from "at the origin".
        """
        if coord_name == "speed":
            return float(speed)
        # X/Y component of a body_part, independent of zone occupancy.
        name_l = coord_name.lower()
        if name_l == "x" or name_l.endswith("_x"):
            if body_part_coords:
                xy = body_part_coords.get(body_part)
                if xy is not None and len(xy) >= 1:
                    return float(xy[0])
            return COORD_LOST
        if name_l == "y" or name_l.endswith("_y"):
            if body_part_coords:
                xy = body_part_coords.get(body_part)
                if xy is not None and len(xy) >= 2:
                    return float(xy[1])
            return COORD_LOST
        # Zone-name resolution.
        bp_zones = zones_by_body_part.get(body_part, {})
        if isinstance(bp_zones, dict):
            for zname, inside in bp_zones.items():
                if inside:
                    return zname
            return ""
        if isinstance(bp_zones, str):
            return bp_zones
        return ""

    @staticmethod
    def _eval_in_zone(trig: Dict[str, Any],
                      zones_by_body_part: Dict[str, Any]) -> bool:
        """Pure 'is body_part currently in any of trig.zones' check, no edge logic."""
        body_part = trig.get("body_part", "centroid")
        zones = trig.get("zones", []) or []
        bp_zones = zones_by_body_part.get(body_part, {})
        if isinstance(bp_zones, dict):
            return any(bp_zones.get(zname, False) for zname in zones)
        if isinstance(bp_zones, str):
            return bp_zones in zones
        return False

    def _evaluate_condition(self, trig: Dict[str, Any],
                            zones_by_body_part: Dict[str, Any],
                            now_s: float = 0.0,
                            rule_key: str = ""):
        """Evaluate a trigger against the current frame → ``(active, value,
        geom)``. ``value`` is the underlying scalar for the plot; ``geom`` an
        optional overlay hint. Zone conditions read ``zones_by_body_part``;
        kinematic/posture conditions read ``self._features``. Duration-gated
        conditions (freezing/rearing/facing) require the raw condition to hold
        for ``trig['duration_ms']`` before going active."""
        cond = trig.get("condition", "in_zone")
        thr = trig.get("threshold")
        unit = trig.get("unit", "px")
        fx = self._features

        raw = False          # instantaneous condition (pre-duration)
        value = None
        geom = None

        if cond in ("in_zone", "enter_zone", "not_in_zone",
                    "exit_zone", "exit_edge"):
            inz = self._eval_in_zone(trig, zones_by_body_part)
            # ``exit_edge`` is authored by configure_from_zones for the exit
            # half of an event_enter_exit zone. Its actual firing is edge-based
            # (the true→false transition tracked in _in_zone_history), but its
            # rule-state display lane must not be stuck False, represent it as
            # a level "outside the zone", like exit_zone.
            raw = inz if cond in ("in_zone", "enter_zone") else (not inz)
            zs = trig.get("zones") or []
            geom = {"type": "zone", "zone": zs[0] if zs else ""}
        elif cond in ("speed_gt", "speed_lt"):
            value = fx.speed_in(unit if unit in ("mm", "cm", "bodylen") else "px")
            # Guard BOTH branches: without the outer guard, speed_gt with an
            # unset threshold evaluated ``value > None`` → TypeError (the old
            # ``… if thr is not None`` bound only to the speed_lt branch).
            raw = (False if thr is None
                   else (value > thr) if cond == "speed_gt" else (value < thr))
        elif cond in ("rotation_gt", "rotation_lt"):
            tv = fx.turning_deg_s()
            value = abs(tv) if tv is not None else None
            if value is not None and thr is not None:
                raw = (value > thr) if cond == "rotation_gt" else (value < thr)
        elif cond in ("head_angle_gt", "head_angle_lt"):
            neck = trig.get("neck", trig.get("body_part", "neck"))
            hv = fx.head_body_angle_deg(neck)
            value = abs(hv) if hv is not None else None
            if value is not None and thr is not None:
                raw = (value > thr) if cond == "head_angle_gt" else (value < thr)
        elif cond == "facing_line":
            target = trig.get("target")
            if not target:
                # No editor authors "target" today, without this guard the
                # rule silently measured the angle to the frame ORIGIN
                # (0,0) and looked like it worked. Warn once, never fire.
                if not trig.get("_warned_no_target"):
                    trig["_warned_no_target"] = True
                    logger.warning(
                        "facing_line trigger %r has no 'target' point, "
                        "rule disabled (author it in the trigger JSON).",
                        trig.get("event_name") or trig.get("name") or "?")
                value, raw = None, False
            else:
                tx, ty = target
                value = fx.facing_deg((tx, ty))
                raw = (value is not None
                       and abs(value) <= (thr if thr is not None else 30.0))
        elif cond == "elongation_gt":
            value = fx.elongation()
            raw = value is not None and thr is not None and value > thr
        elif cond == "rearing":
            e = fx.elongation()
            value = e
            rho = thr if thr is not None else 0.6
            speed_max = trig.get("speed_max", 2.0)
            raw = (e is not None and e < rho
                   and fx.speed_in("bodylen") < speed_max)   # proxy: foreshortened + slow
        elif cond == "freezing":
            value = fx.speed_in(unit if unit in ("mm", "cm", "bodylen") else "px")
            raw = value < (thr if thr is not None else 1.0)
        elif cond in ("distance_gt", "distance_lt"):
            value = fx.distance_in(trig.get("part_a", ""), trig.get("part_b", ""), unit)
            if value is not None and thr is not None:
                raw = (value > thr) if cond == "distance_gt" else (value < thr)

        # Duration gate: the raw condition must hold ≥ duration_ms to go active.
        # Stable per-rule key: id(trig) is reused after a reconfigure GCs
        # the old dict, so a new rule could inherit another's duration-gate
        # start time. Also: the no-duration branch must not grow a key per
        # frame that only reset() ever pruned.
        dur_ms = float(trig.get("duration_ms", 0.0) or 0.0)
        key = f"{cond}|{rule_key}"
        if dur_ms <= 0:
            active = raw
            self._cond_since.pop(key, None)
        else:
            since = self._cond_since.get(key)
            if raw:
                if since is None:
                    since = now_s
                    self._cond_since[key] = since
                active = (now_s - since) * 1000.0 >= dur_ms
            else:
                self._cond_since[key] = None
                active = False
        return active, value, geom


# =============================================================================
#  MCUPusher, the wiring
# =============================================================================


class MCUPusher:
    """Per-box edge-detected MCU push. Result-driven (no frame queue).

    Not a :class:`~source.video.framebus.sink_base.Sink`, does not subscribe
    to FrameBus; consumed via the result callbacks of PoseSink and
    TrackerSink instead.
    """

    name = "mcu_pusher"

    def __init__(self) -> None:
        self._pycboards: Dict[int, Any] = {}
        self._policies: Dict[int, TrackingPushPolicy] = {}
        # Per-box pose confidence threshold for the MCU push, mirroring the
        # box's configured PoseSink threshold (set via configure_tracking).
        self._pose_conf_thresh: Dict[int, float] = {}
        self._lock = threading.RLock()
        # Observational timing spine (shared LatencyBudget). Records the two
        # MCU-side stages the capture→inference path can't see: infer→push
        # (inference-done → coords about to hit the port) and push→wire (the
        # serial write itself). None until the Pipeline wires it.
        self._latency: Optional[Any] = None

    def set_latency_budget(self, budget: Optional[Any]) -> None:
        """Attach the pipeline's LatencyBudget so the push path closes the
        capture→wire timing spine. ``None`` disables recording.

        The final stage is recorded by the board itself: the serial write
        happens on the GUI thread, one process_data tick after the push
        queues it, so only the drain knows when it truly landed.
        """
        self._latency = budget
        with self._lock:
            boards = list(self._pycboards.values())
        for board in boards:
            self._attach_write_budget(board)

    def _attach_write_budget(self, pycboard) -> None:
        """Give a board the budget so its drain records ``push_to_wire``."""
        setter = getattr(pycboard, "set_write_latency_budget", None)
        if callable(setter):
            try:
                setter(self._latency)
            except Exception as e:
                logger.debug("attach write latency budget: %s", e)

    # ── Lifecycle (Pipeline-managed) ----------------------------------

    def start(self) -> None:
        """No-op, event-driven, no worker thread. Pipeline calls it at
        startup for sink-lifecycle parity; the shutdown loop skips us."""
        return

    # ── Box registration ----------------------------------------------

    def _policy(self, setup_id: int) -> TrackingPushPolicy:
        """Get-or-create the box's policy WITH its trigger-frame callback
        wired. Every policy-creation site must come through here: a policy
        created bare (the old ``setdefault`` sites) had no callback, so
        after a ``reset()`` the annotation overlay + Session-Plot trigger
        lane went permanently dead until the next ``register_box``.
        Caller must hold ``self._lock``."""
        pol = self._policies.get(setup_id)
        if pol is None:
            pol = TrackingPushPolicy()
            self._policies[setup_id] = pol
        if pol._trigger_frame_cb is None:
            # Route the policy's per-frame trigger state to the GUI relay
            # (annotation + Session-Plot lane) via the pipeline sink.
            pol.set_trigger_frame_cb(self._dispatch_trigger_frame, setup_id)
        return pol

    def register_box(self, setup_id: int, pycboard: Any) -> None:
        self._attach_write_budget(pycboard)
        with self._lock:
            self._pycboards[setup_id] = pycboard
            self._policy(setup_id)

    def set_trigger_frame_sink(self, cb) -> None:
        """Pipeline sets this to fan policy trigger-frames to the GUI."""
        self._trigger_frame_sink = cb

    def _dispatch_trigger_frame(self, tf) -> None:
        sink = getattr(self, "_trigger_frame_sink", None)
        if sink is not None:
            sink(tf)

    def unregister_box(self, setup_id: int) -> None:
        with self._lock:
            self._pycboards.pop(setup_id, None)
            self._policies.pop(setup_id, None)
            self._pose_conf_thresh.pop(setup_id, None)

    # ── Policy configuration -------------------------------------------

    def configure_zones(self, setup_id: int, zones: list,
                        default_body_part: str = "centroid", *,
                        push_coords: bool = True,
                        push_zone_events: bool = True) -> None:
        """Build the policy from a list of Zone objects (or dicts).

        Wraps TrackingPushPolicy.configure_from_zones, the single
        zone→coord_mapping/triggers translator.
        """
        with self._lock:
            policy = self._policy(setup_id)
            try:
                from source.video.zones.triggering import Zone
                zone_objs = [Zone.from_dict(z) if isinstance(z, dict) else z
                             for z in (zones or [])]
                policy.configure_from_zones(
                    zone_objs, default_body_part,
                    push_coords=push_coords,
                    push_zone_events=push_zone_events)
            except Exception as e:
                logger.error("MCUPusher.configure_zones (box=%d): %s", setup_id, e)

    def set_features_context(self, setup_id: int, px_per_mm=None, axis=None) -> None:
        """Route real-unit scale + body-axis to one box's policy extractor."""
        with self._lock:
            policy = self._policy(setup_id)
            policy.set_features_context(px_per_mm=px_per_mm, axis=axis)

    def push_stats(self, setup_id: int) -> Optional[Dict[str, Any]]:
        """Live push counters for one box (coords / zone events / triggers /
        last-push age), or None if the box has no policy yet."""
        with self._lock:
            policy = self._policies.get(setup_id)
            return policy.stats() if policy is not None else None

    def configure_tracking(self, setup_id: int, cfg: dict) -> None:
        """Apply tracking-configure dialog ``coord_mapping`` + ``triggers``.

        MERGES on top of whatever ``configure_zones`` already populated
        from per-zone fields, so the user can use either or both:
          - per-zone ``transmit_mode``/``coord_var``/``event_on_*`` for
            zones that should always push that way,
          - dialog Coord Mapping + Triggers tables for explicit overrides.
        Dialog values win on ``coord_name`` / ``event_name`` collision.

        Also reads ``push_zones_to_mcu`` from cfg and writes it onto the
        policy's ``push_zone_changed`` gate so the intrinsic
        ``zone_changed`` event honours the dialog toggle. ``cfg`` here
        may be a raw TrackingConfig.to_json() dict OR a wrapper with a
        ``"tracking"`` sub-key, both shapes are tolerated.
        """
        with self._lock:
            policy = self._policy(setup_id)
            tracking = cfg.get("tracking", cfg) if isinstance(cfg, dict) else {}
            if not isinstance(tracking, dict):
                return
            cm = tracking.get("coord_mapping")
            tg = tracking.get("triggers")
            new_cm = dict(policy.coord_mapping)
            if isinstance(cm, dict):
                new_cm.update(cm)
            new_tg = list(policy.triggers)
            if isinstance(tg, list):
                dialog_events = {t.get("event_name") for t in tg
                                 if isinstance(t, dict)}
                new_tg = [t for t in new_tg
                          if t.get("event_name") not in dialog_events] + list(tg)
            policy.update(coord_mapping=new_cm, triggers=new_tg)
            # Mirror the box's configured pose confidence threshold so the
            # push filter matches what PoseSink displays, a hardcoded 0.5
            # here let below-display-threshold keypoints reach the MCU.
            if "confidence_threshold" in tracking:
                try:
                    self._pose_conf_thresh[setup_id] = float(
                        tracking["confidence_threshold"])
                except (TypeError, ValueError):
                    pass
            # Honour the dialog push-zones gate (default True when absent).
            if "push_zones_to_mcu" in tracking:
                policy.push_zone_changed = bool(
                    tracking.get("push_zones_to_mcu", True))
            # Coord-write gate, applied at push time so dialog-merged
            # mappings honour the toggle exactly like zone-derived ones.
            if "push_coords_to_mcu" in tracking:
                policy.push_coords = bool(
                    tracking.get("push_coords_to_mcu", True))
            # Per-frame pose event gate (default False when absent).
            if "push_frame_event" in tracking:
                policy.push_frame_event = bool(
                    tracking.get("push_frame_event", False))
            # Body part the intrinsic zone_changed event diffs against.
            # Live re-bind so a mid-session edit takes effect next push;
            # reset the prev snapshot so the new part doesn't fire on its
            # first observation.
            if "zone_change_body_part" in tracking:
                bp = str(tracking.get("zone_change_body_part", "")
                         or "centroid")
                if bp != policy.zone_change_body_part:
                    policy.zone_change_body_part = bp
                    policy._prev_zone_set_for_change = None

    def reset(self, setup_id: Optional[int] = None) -> None:
        """Drop cached push policies so the next push rebuilds clean."""
        with self._lock:
            if setup_id is None:
                self._policies.clear()
            else:
                self._policies.pop(setup_id, None)

    # ── Result-event consumers (Pipeline wires these into Pose/TrackerSink)

    def on_pose_result(self, setup_id: int, cam_frame_id: int,
                       pose_array: list, location: Optional[str],
                       speed: float, zones_by_body_part: dict,
                       raw_pose_dict: dict, capture_host_ns: int = 0,
                       forecast_coords: Optional[Dict[str, Any]] = None,
                       infer_done_ns: int = 0,
                       pose_lag_ms: Optional[float] = None,
                       filter_ms: Optional[float] = None) -> None:
        # ``pose_lag_ms`` / ``filter_ms`` are for the session file; the push path
        # has its own stage in the latency budget and ignores them.
        # ``capture_host_ns`` unused here (no fw stamping on the push path).
        # Build {body_part: (x, y)} for the policy's pixel-component
        # resolver, filtered on the same confidence threshold PoseSink
        # uses so below-threshold detections don't bleed into MCU vars.
        body_part_coords: Dict[str, Any] = {}
        conf_thresh = self._pose_conf_thresh.get(setup_id, _POSE_PUSH_CONF_THRESH)
        for name, kp in (raw_pose_dict or {}).items():
            try:
                if not isinstance(name, str) or name.startswith("_"):
                    continue  # skip metadata keys e.g. "_rotation_rad"
                if not hasattr(kp, "__len__") or len(kp) < 2:
                    continue
                conf = float(kp[2]) if len(kp) > 2 else 1.0
                if conf < conf_thresh:
                    continue
                x, y = float(kp[0]), float(kp[1])
                # Wire hygiene: a NaN survives repr() but the MCU-side
                # eval() raises NameError and silently drops the write;
                # negative pixels collide with the -1.0 COORD_LOST
                # sentinel tasks test with ``if c.head_x < 0``.
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                body_part_coords[name] = (max(0.0, x), max(0.0, y))
            except (TypeError, ValueError, IndexError):
                continue
        # Latency-compensated positions (centroid / picked body part) override
        # the raw coords for the MCU push so closed-loop tasks reading c.* act
        # on where the animal WILL be when the command lands. Only present for
        # parts the enhancer forecasts; every other keypoint stays raw.
        if forecast_coords:
            for name, xy in forecast_coords.items():
                if xy is not None:
                    body_part_coords[name] = (float(xy[0]), float(xy[1]))
        self._push(setup_id, zones_by_body_part, speed, body_part_coords,
                   infer_done_ns=infer_done_ns)

    def on_tracker_result(self, setup_id: int, cam_frame_id: int,
                          centroid, location: Optional[str],
                          speed: float, zones_by_body_part: dict,
                          position, capture_host_ns: int = 0,
                          infer_done_ns: int = 0) -> None:
        body_part_coords: Optional[Dict[str, Any]] = None
        try:
            if centroid is not None:
                cx, cy = float(centroid[0]), float(centroid[1])
                # Same wire hygiene as the pose path: no NaN (MCU eval
                # drops it silently), no negatives (COORD_LOST collision).
                if math.isfinite(cx) and math.isfinite(cy):
                    body_part_coords = {
                        "centroid": (max(0.0, cx), max(0.0, cy))}
        except (TypeError, ValueError, IndexError):
            body_part_coords = None
        self._push(setup_id, zones_by_body_part, speed, body_part_coords,
                   infer_done_ns=infer_done_ns)

    # ── Internal -------------------------------------------------------

    def _push(self, setup_id: int, zones_by_body_part: dict, speed: float,
              body_part_coords: Optional[Dict[str, Any]] = None,
              infer_done_ns: int = 0) -> None:
        with self._lock:
            pyc = self._pycboards.get(setup_id)
            policy = self._policies.get(setup_id)
        if pyc is None or policy is None:
            return
        try:
            if not getattr(pyc, "framework_running", False):
                return
            # Timing spine: infer→push is inference-done → coords queued for
            # the port. push→wire is NOT measured here, policy.push only
            # enqueues; the serial write happens later on the GUI thread, so
            # Pycboard._drain_pending_writes records that stage when the write
            # actually lands. Timing policy.push would report the queue append
            # (microseconds) and hide a whole process_data tick.
            if self._latency is not None:
                self._latency.record_from_ns("infer_to_push", infer_done_ns)
            policy.push(pyc, zones_by_body_part, speed, body_part_coords)
        except Exception as e:
            logger.error("MCUPusher push error (box=%d): %s", setup_id, e)
