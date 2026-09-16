"""Tests for the DLC → MCU push chain.

A. ``_rebuild_zone_manager`` lazy-creates the per-box ZoneManager on
   ``tracker_manager.zone_managers`` so ``get_zone_manager(box_id)``
   returns it and zone events / zone-derived coords fire in DLC mode.

B. MCUPusher threads body-part XY coords from PoseSink / TrackerSink
   into ``TrackingPushPolicy.push``, and ``_resolve_coord`` honours the
   ``x``/``y`` (and ``<bp>_x``/``<bp>_y``) naming convention to push
   pixel coordinates instead of zone-name strings.

C. (Smoke) The intrinsic ``zone_changed`` event.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest

from source.video.framebus.mcu_pusher import TrackingPushPolicy
from source.video.framebus.mcu_pusher import MCUPusher


class _FakePycboard:
    """Minimal Pycboard surface for the policy: queue_set_coordinates,
    queue_trigger_event, framework_running, sm_info.events."""

    class _SmInfo:
        def __init__(self, events):
            self.events = events

    def __init__(self, sm_events=None, framework_running=True):
        self.framework_running = framework_running
        self.sm_info = self._SmInfo(sm_events or {})
        self.coords: list = []   # [(name, value), ...]
        self.events: list = []   # [event_name, ...]  (silent path now)

    def queue_set_coordinates(self, name, value):
        self.coords.append((name, value))

    def queue_trigger_event(self, name):
        # User-visible (TSV-logged) trigger path.
        self.events.append(name)

    def queue_trigger_intrinsic_event(self, name):
        # Silent dispatch path, MCU state machine processes the event
        # but no TSV row appears, like entry/exit.
        self.events.append(name)


# Zone-manager lazy create


def test_rebuild_zone_manager_lazy_creates_on_empty_registry(monkeypatch, tmp_path):
    """``_rebuild_zone_manager`` must create a ZoneManager when the
    tracker_manager registry is empty for that box."""
    from source.video.zones.triggering import ZoneManager
    from source.gui import base as base_mod

    # Tiny stand-in for tracker_manager
    class _TM:
        def __init__(self):
            self.zone_managers = {}
        def get_zone_manager(self, setup_id):
            return self.zone_managers.get(setup_id)
        def set_zone_manager(self, setup_id, zm):
            self.zone_managers[setup_id] = zm

    # Stand-in for the main window, only the bits _rebuild_zone_manager touches
    class _MW:
        tracker_manager = _TM()
        _ZONE_MIN_POINTS = 2
        def _box_frame_wh(self, _bid):
            return (640, 480)

    mw = _MW()
    zones = [
        {"name": "zoneA", "type": "polygon",
         "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
         "coord_space": "normalized"},
    ]
    # Bind base method to instance so ``self`` resolves to our mock.
    base_mod.MainWindowBase._rebuild_zone_manager(mw, 1, zones)
    zm = mw.tracker_manager.get_zone_manager(1)
    assert zm is not None, "ZoneManager was not lazy-created"
    assert isinstance(zm, ZoneManager)
    assert len(zm.zones) == 1
    assert "zoneA" in zm.zones
    assert zm.zones["zoneA"].name == "zoneA"


# Body-part coord components (c.x / c.y / c.<bp>_x ...)


def test_resolve_coord_x_returns_body_part_pixel_x():
    body_part_coords = {"centroid": (123.4, 56.7)}
    assert TrackingPushPolicy._resolve_coord(
        "x", "centroid", {}, 0.0, body_part_coords) == pytest.approx(123.4)


def test_resolve_coord_y_returns_body_part_pixel_y():
    body_part_coords = {"centroid": (123.4, 56.7)}
    assert TrackingPushPolicy._resolve_coord(
        "y", "centroid", {}, 0.0, body_part_coords) == pytest.approx(56.7)


def test_resolve_coord_bp_x_suffix_for_per_part_x():
    body_part_coords = {"head": (10.0, 20.0), "tail": (30.0, 40.0)}
    assert TrackingPushPolicy._resolve_coord(
        "head_x", "head", {}, 0.0, body_part_coords) == pytest.approx(10.0)
    assert TrackingPushPolicy._resolve_coord(
        "tail_y", "tail", {}, 0.0, body_part_coords) == pytest.approx(40.0)


def test_resolve_coord_x_returns_lost_sentinel_when_part_missing():
    """Body part below confidence → not in body_part_coords → COORD_LOST.

    A negative sentinel (not 0.0, a valid origin pixel) lets a task tell
    "not detected" from "really at the top-left corner"."""
    from source.video.framebus.mcu_pusher import COORD_LOST
    body_part_coords = {"head": (10.0, 20.0)}
    assert TrackingPushPolicy._resolve_coord(
        "x", "tail", {}, 0.0, body_part_coords) == COORD_LOST
    assert COORD_LOST < 0


def test_resolve_coord_speed_unchanged_by_new_arg():
    assert TrackingPushPolicy._resolve_coord(
        "speed", "centroid", {}, 4.5, {"centroid": (1, 2)}) == pytest.approx(4.5)


def test_resolve_coord_zone_name_legacy_path_intact():
    """Names that aren't speed / x / y / *_x / *_y resolve to zone
    occupancy."""
    zones = {"centroid": {"reward_zone": True}}
    val = TrackingPushPolicy._resolve_coord(
        "loc_center", "centroid", zones, 0.0, {"centroid": (1, 2)})
    assert val == "reward_zone"


def test_policy_push_sends_x_y_on_change():
    """End-to-end: with coord_mapping={'x': 'centroid', 'y': 'centroid'},
    a non-zero body_part_coords push should produce two
    queue_set_coordinates calls."""
    policy = TrackingPushPolicy(
        coord_mapping={"x": "centroid", "y": "centroid"})
    pyc = _FakePycboard()
    policy.push(pyc, {}, speed=0.0,
                body_part_coords={"centroid": (100.0, 200.0)})
    names = [n for n, _ in pyc.coords]
    assert "x" in names and "y" in names, f"got {pyc.coords}"
    by_name = dict(pyc.coords)
    assert by_name["x"] == pytest.approx(100.0)
    assert by_name["y"] == pytest.approx(200.0)


def test_policy_push_dirty_flag_skips_unchanged_xy():
    """Same XY twice in a row → only one push per coord (dirty cache)."""
    policy = TrackingPushPolicy(coord_mapping={"x": "centroid"})
    pyc = _FakePycboard()
    policy.push(pyc, {}, 0.0, {"centroid": (50.0, 50.0)})
    policy.push(pyc, {}, 0.0, {"centroid": (50.0, 50.0)})
    assert pyc.coords == [("x", 50.0)]


# MCUPusher → policy plumbing, verify body_part_coords actually flows


def test_pushsink_on_pose_result_threads_coords_to_policy():
    """``MCUPusher.on_pose_result`` must extract body_part_coords from
    raw_pose_dict (filtered by confidence) and forward them so the
    policy can resolve x/y."""
    sink = MCUPusher()
    pyc = _FakePycboard()
    sink.register_box(1, pyc)
    sink._policies[1].update(coord_mapping={"x": "head", "y": "head"})
    raw_pose = {
        "head": (12.0, 34.0, 0.9),         # accepted
        "tail": (90.0, 90.0, 0.1),         # below conf → dropped
        "_rotation_rad": 0.123,            # metadata key → skipped
    }
    sink.on_pose_result(
        setup_id=1, cam_frame_id=0,
        pose_array=[(12.0, 34.0, 0.9), (90.0, 90.0, 0.1)],
        location=None, speed=0.0,
        zones_by_body_part={},
        raw_pose_dict=raw_pose,
    )
    by_name = dict(pyc.coords)
    assert by_name == {"x": 12.0, "y": 34.0}


def test_pushsink_on_tracker_result_passes_centroid_xy():
    sink = MCUPusher()
    pyc = _FakePycboard()
    sink.register_box(1, pyc)
    sink._policies[1].update(coord_mapping={"x": "centroid", "y": "centroid"})
    sink.on_tracker_result(
        setup_id=1, cam_frame_id=0,
        centroid=(77.7, 88.8), location=None, speed=0.0,
        zones_by_body_part={}, position=(70, 80, 20, 20),
    )
    by_name = dict(pyc.coords)
    assert by_name == {"x": 77.7, "y": 88.8}


def test_pushsink_silent_when_framework_not_running():
    """Pre-existing contract: nothing pushed when framework_running
    is False. Body-part-coord plumbing must not break this."""
    sink = MCUPusher()
    pyc = _FakePycboard(framework_running=False)
    sink.register_box(1, pyc)
    sink._policies[1].update(coord_mapping={"x": "centroid"})
    sink.on_tracker_result(
        setup_id=1, cam_frame_id=0,
        centroid=(1.0, 2.0), location=None, speed=0.0,
        zones_by_body_part={}, position=None,
    )
    assert pyc.coords == []
    assert pyc.events == []


# zone_changed intrinsic event, smoke test


def test_zone_changed_fires_regardless_of_task_sm_events():
    """``zone_changed`` is a FRAMEWORK-INTRINSIC event auto-registered
    by ``state_machine.setup_state_machine`` alongside ``entry``/``exit``.
    The host does NOT gate on whether the task declared it, the
    operator checkbox (``push_zone_changed``) is the only gate. The MCU
    framework guarantees the event ID exists, so the host can always
    fire it."""
    policy = TrackingPushPolicy()
    pyc = _FakePycboard(sm_events={})  # task didn't declare anything
    # First tick establishes baseline (no fire on first observation).
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    assert pyc.events == []
    # Different zone → zone_changed fires regardless of sm_events.
    policy.push(pyc, {"centroid": {"zoneB": True}}, 0.0)
    assert pyc.events == ["zone_changed"]


def test_zone_changed_suppressed_when_operator_unchecks_push():
    """Operator can opt out via ``push_zone_changed=False``, then no
    zone_changed event fires regardless of how many transitions
    happen. This is the only gate the host applies."""
    policy = TrackingPushPolicy(push_zone_changed=False)
    pyc = _FakePycboard(sm_events={"zone_changed": 1})
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    policy.push(pyc, {"centroid": {"zoneB": True}}, 0.0)
    assert pyc.events == []


def test_zone_changed_uses_silent_intrinsic_dispatch_not_logged_event():
    """``zone_changed`` MUST route through
    ``queue_trigger_intrinsic_event`` (silent, no TSV row), NOT
    ``queue_trigger_event`` (which would log a row per transition and
    flood the TSV at every confidence flicker). Like entry/exit,
    zone_changed is a framework-intrinsic event that stays silent unless
    the user explicitly prints. The policy must hit the intrinsic path."""

    class _SeparatePycboard:
        class _SmInfo:
            def __init__(self, events): self.events = events
        def __init__(self):
            self.framework_running = True
            self.sm_info = self._SmInfo({"zone_changed": 1})
            self.logged_events = []      # via queue_trigger_event (TSV-visible)
            self.silent_events = []      # via queue_trigger_intrinsic_event
        def queue_set_coordinates(self, *_): pass
        def queue_trigger_event(self, name):
            self.logged_events.append(name)
        def queue_trigger_intrinsic_event(self, name):
            self.silent_events.append(name)

    policy = TrackingPushPolicy()
    pyc = _SeparatePycboard()
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    policy.push(pyc, {"centroid": {"zoneB": True}}, 0.0)
    assert pyc.logged_events == [], (
        "zone_changed must NOT take the TSV-logging path")
    assert pyc.silent_events == ["zone_changed"], (
        "zone_changed must take the silent intrinsic dispatch path")


def test_zone_changed_fires_on_exit_when_body_part_leaves_all_zones():
    """PoseSink drops the body_part's key from zones_by_body_part the
    instant it leaves every zone. The diff must catch this as 'in_set
    went from {zone} → empty' even though the key vanished from the
    dict."""
    policy = TrackingPushPolicy()
    pyc = _FakePycboard(sm_events={"zone_changed": 1})
    # Baseline: centroid in zoneA.
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    assert pyc.events == []
    # EXIT: centroid no longer in zones_by_body_part at all.
    # ``zones_by_body_part = {}`` simulates PoseSink dropping the key
    # because the body part left every zone.
    policy.push(pyc, {}, 0.0)
    assert pyc.events == ["zone_changed"]
    # Re-ENTER: zoneB now → another fire.
    policy.push(pyc, {"centroid": {"zoneB": True}}, 0.0)
    assert pyc.events == ["zone_changed", "zone_changed"]


def test_zone_changed_diffs_only_configured_body_part():
    """Only the configured body part's zone occupancy fires
    zone_changed. Other body parts moving around must NOT trigger it,
    they belong to the per-zone trigger system instead."""
    policy = TrackingPushPolicy(zone_change_body_part="head")
    pyc = _FakePycboard(sm_events={"zone_changed": 1})
    # Baseline: head in zoneA, tail elsewhere.
    policy.push(pyc, {"head": {"zoneA": True},
                      "tail": {"zoneB": True}}, 0.0)
    assert pyc.events == []
    # Tail moves but head stays put → NO fire.
    policy.push(pyc, {"head": {"zoneA": True},
                      "tail": {"zoneC": True}}, 0.0)
    assert pyc.events == []
    # Head moves → fire.
    policy.push(pyc, {"head": {"zoneB": True},
                      "tail": {"zoneC": True}}, 0.0)
    assert pyc.events == ["zone_changed"]


def test_zone_changed_body_part_can_be_rebound_mid_session():
    """``configure_tracking`` can change the diff target at runtime;
    next push uses the new body part and the prev snapshot is reset
    so the change of subject doesn't itself fire a spurious event."""
    from source.video.framebus.mcu_pusher import MCUPusher
    pusher = MCUPusher()
    pyc = _FakePycboard(sm_events={"zone_changed": 1})
    pusher.register_box(1, pyc)
    pusher.configure_tracking(1, {"zone_change_body_part": "head"})
    # Drive a tick to set the policy's prev snapshot via the head body part.
    pusher.on_pose_result(
        setup_id=1, cam_frame_id=0,
        pose_array=[[0, 0, 1.0]], location=None, speed=0.0,
        zones_by_body_part={"head": {"zoneA": True}},
        raw_pose_dict={"head": (0, 0, 1.0)},
    )
    # Re-bind to a different body part. The reset should prevent the
    # very next push from firing solely because we switched.
    pusher.configure_tracking(1, {"zone_change_body_part": "tail"})
    pusher.on_pose_result(
        setup_id=1, cam_frame_id=1,
        pose_array=[[0, 0, 1.0]], location=None, speed=0.0,
        zones_by_body_part={"head": {"zoneA": True},
                            "tail": {"zoneB": True}},
        raw_pose_dict={"head": (0, 0, 1.0), "tail": (10, 10, 1.0)},
    )
    assert pyc.events == []
    # Now move the tail → fires.
    pusher.on_pose_result(
        setup_id=1, cam_frame_id=2,
        pose_array=[[0, 0, 1.0]], location=None, speed=0.0,
        zones_by_body_part={"head": {"zoneA": True},
                            "tail": {"zoneC": True}},
        raw_pose_dict={"head": (0, 0, 1.0), "tail": (20, 20, 1.0)},
    )
    assert pyc.events == ["zone_changed"]


# ── Per-frame ``frame_event`` (opt-in poll) ───────────────────────────────

def test_frame_event_off_by_default():
    """A default policy must NOT emit ``frame_event``; it is opt-in so the
    repo behaves exactly as before until the operator enables it."""
    policy = TrackingPushPolicy()
    pyc = _FakePycboard(sm_events={"frame_event": 1})
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    assert "frame_event" not in pyc.events


def test_frame_event_fires_every_push_when_enabled():
    """With ``push_frame_event=True`` the event fires on EVERY push (pose
    rate), including pushes where the zone did not change; that is the
    whole point (a steady poll, not an edge)."""
    policy = TrackingPushPolicy(push_frame_event=True)
    pyc = _FakePycboard(sm_events={"frame_event": 1})
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)   # no zone change
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)   # still no change
    assert pyc.events == ["frame_event", "frame_event"]


def test_frame_event_coord_is_queued_before_event():
    """The coord write must reach the MCU BEFORE the frame_event, so the
    task reads a CURRENT ``c.loc_center`` when the event runs."""
    order = []

    class _OrderedPycboard:
        class _SmInfo:
            def __init__(self, events): self.events = events
        def __init__(self):
            self.framework_running = True
            self.sm_info = self._SmInfo({"frame_event": 1})
        def queue_set_coordinates(self, name, value):
            order.append(("coord", name, value))
        def queue_trigger_event(self, name):
            order.append(("logged", name))
        def queue_trigger_intrinsic_event(self, name):
            order.append(("silent", name))

    policy = TrackingPushPolicy(
        coord_mapping={"loc_center": "centroid"}, push_frame_event=True)
    pyc = _OrderedPycboard()
    policy.push(pyc, {"centroid": {"RightArm": True}}, 0.0)
    assert order == [("coord", "loc_center", "RightArm"),
                     ("silent", "frame_event")]


def test_frame_event_uses_silent_dispatch_not_logged():
    """``frame_event`` must take the silent intrinsic path, at pose rate
    the TSV would otherwise gain ~20-25 rows/sec of noise."""

    class _SeparatePycboard:
        class _SmInfo:
            def __init__(self, events): self.events = events
        def __init__(self):
            self.framework_running = True
            self.sm_info = self._SmInfo({"frame_event": 1})
            self.logged_events = []
            self.silent_events = []
        def queue_set_coordinates(self, *_): pass
        def queue_trigger_event(self, name): self.logged_events.append(name)
        def queue_trigger_intrinsic_event(self, name):
            self.silent_events.append(name)

    policy = TrackingPushPolicy(push_frame_event=True)
    pyc = _SeparatePycboard()
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    assert pyc.logged_events == []
    assert pyc.silent_events == ["frame_event"]


def test_push_frame_event_round_trips_through_configs():
    """The flag survives both config layers: host TrackingConfig
    (project save/load) and the framebus TrackingConfig (settings.json).
    Default is False in both."""
    from source.config.experiment import TrackingConfig as HostTC
    from source.video.framebus.types import TrackingConfig as BusTC

    # Host side, default off, explicit on round-trips.
    assert HostTC().push_frame_event is False
    assert HostTC.from_dict({"push_frame_event": True}).push_frame_event is True
    assert HostTC.from_dict(
        {"push_frame_event": True}).to_compact()["push_frame_event"] is True

    # Framebus side, default off, explicit on round-trips.
    assert BusTC(setup_id=1).push_frame_event is False
    rt = BusTC.from_json(BusTC(setup_id=1, push_frame_event=True).to_json())
    assert rt.push_frame_event is True


def test_configure_tracking_honours_push_frame_event_gate():
    """MCUPusher.configure_tracking plumbs the cfg flag onto the policy."""
    pusher = MCUPusher()
    pusher.configure_tracking(1, {"tracking": {"push_frame_event": True}})
    pyc = _FakePycboard(sm_events={"frame_event": 1})
    with pusher._lock:
        policy = pusher._policies[1]
    assert policy.push_frame_event is True
    policy.push(pyc, {"centroid": {"zoneA": True}}, 0.0)
    assert pyc.events == ["frame_event"]
