"""Characterisation of ``TrackingPushPolicy.push``.

Pins the behaviour of a 180-line per-frame method BEFORE splitting it. This is
the closed-loop path, what the MCU sees, and in what order, so the ordering
guarantees matter as much as the values:

  * coords are written BEFORE any event is queued, because a task's handler
    reads ``c.*`` the instant the event lands and writes drain FIFO
  * coords are dirty-flagged, so an unchanged value costs no serial traffic
  * ``zone_changed`` fires on entry AND exit, with the first frame only
    establishing a baseline
  * a missing intrinsic event on the MCU warns once, not every frame
  * trigger conditions are edge-detected, and ``exit_edge`` is inverted
"""
from __future__ import annotations

from types import SimpleNamespace

from source.video.framebus.mcu_pusher import TrackingPushPolicy


class _Pyc:
    """Records the ORDER of everything the policy sends to the board."""

    framework_running = True

    def __init__(self, events=("zone_changed", "frame_event")):
        self.calls = []                       # ordered log of every send
        self.sm_info = SimpleNamespace(events=dict.fromkeys(events, 1))

    def queue_set_coordinates(self, name, val):
        self.calls.append(("coord", name, val))

    def queue_trigger_event(self, name):
        self.calls.append(("event", name))

    def queue_trigger_intrinsic_event(self, name):
        self.calls.append(("intrinsic", name))

    # convenience views
    @property
    def coords(self):
        return {n: v for k, n, v in
                (c for c in self.calls if c[0] == "coord")}

    def named(self, kind):
        return [c[1] for c in self.calls if c[0] == kind]


def _zones(**bp):
    """{'nose': {'left': True}} from _zones(nose='left')."""
    return {k: {v: True} for k, v in bp.items()}


# ── guards ───────────────────────────────────────────────────────────────

def test_no_board_is_a_no_op():
    TrackingPushPolicy(coord_mapping={"loc": "centroid"}).push(None, {})


def test_a_stopped_framework_is_not_pushed_to():
    p = TrackingPushPolicy(coord_mapping={"loc": "centroid"})
    pyc = _Pyc()
    pyc.framework_running = False
    p.push(pyc, _zones(centroid="left"))
    assert pyc.calls == []


# ── coords ───────────────────────────────────────────────────────────────

def test_coord_is_written_once_then_only_on_change():
    """Dirty-flagged: re-sending an unchanged coord every frame would flood
    the serial link at pose rate."""
    p = TrackingPushPolicy(coord_mapping={"loc_center": "centroid"})
    pyc = _Pyc()
    p.push(pyc, _zones(centroid="left"))
    p.push(pyc, _zones(centroid="left"))          # unchanged → silent
    p.push(pyc, _zones(centroid="right"))         # changed → written
    assert pyc.named("coord") == ["loc_center", "loc_center"]
    assert pyc.calls[0] == ("coord", "loc_center", "left")


def test_speed_can_be_mapped_to_a_coord():
    """The COORD name is what selects the scalar, not the body part."""
    p = TrackingPushPolicy(coord_mapping={"speed": "centroid"})
    pyc = _Pyc()
    p.push(pyc, {}, speed=12.5)
    assert ("coord", "speed", 12.5) in pyc.calls


# ── ordering: the whole point of the method ─────────────────────────────

def test_coords_are_written_before_any_event_is_queued():
    """A task reads ``c.loc_center`` the instant ``zone_changed`` arrives.
    Writes drain FIFO, so an event queued first would be handled against the
    PREVIOUS zone, and an 'enter and stay' would never re-fire."""
    p = TrackingPushPolicy(coord_mapping={"loc_center": "centroid"},
                           push_zone_changed=True,
                           zone_change_body_part="centroid")
    pyc = _Pyc()
    p.push(pyc, _zones(centroid="left"))          # baseline frame
    pyc.calls.clear()
    p.push(pyc, _zones(centroid="right"))         # the change
    kinds = [c[0] for c in pyc.calls]
    assert kinds.index("coord") < kinds.index("intrinsic")


# ── zone_changed ─────────────────────────────────────────────────────────

def test_first_frame_only_establishes_a_baseline():
    p = TrackingPushPolicy(push_zone_changed=True,
                           zone_change_body_part="centroid")
    pyc = _Pyc()
    p.push(pyc, _zones(centroid="left"))
    assert pyc.named("intrinsic") == []


def test_zone_changed_fires_on_entry_and_on_exit():
    p = TrackingPushPolicy(push_zone_changed=True,
                           zone_change_body_part="centroid")
    pyc = _Pyc()
    p.push(pyc, {})                               # baseline: in no zone
    p.push(pyc, _zones(centroid="left"))          # entry
    p.push(pyc, _zones(centroid="left"))          # stay → silent
    p.push(pyc, {})                               # exit  → fires
    assert pyc.named("intrinsic") == ["zone_changed", "zone_changed"]


def test_a_body_part_that_left_every_zone_is_treated_as_in_no_zone():
    """PoseSink drops the key entirely when the part is in no zone, so the
    falling edge is 'key absent', not 'empty dict'."""
    p = TrackingPushPolicy(push_zone_changed=True,
                           zone_change_body_part="nose")
    pyc = _Pyc()
    p.push(pyc, _zones(nose="left"))
    p.push(pyc, {"head": {"left": True}})         # nose key gone
    assert pyc.named("intrinsic") == ["zone_changed"]


def test_zone_changed_respects_the_operator_toggle():
    p = TrackingPushPolicy(push_zone_changed=False,
                           zone_change_body_part="centroid")
    pyc = _Pyc()
    p.push(pyc, {})
    p.push(pyc, _zones(centroid="left"))
    assert pyc.named("intrinsic") == []


def test_a_string_zone_value_is_accepted():
    p = TrackingPushPolicy(push_zone_changed=True,
                           zone_change_body_part="centroid")
    pyc = _Pyc()
    p.push(pyc, {"centroid": ""})
    p.push(pyc, {"centroid": "left"})
    assert pyc.named("intrinsic") == ["zone_changed"]


def test_missing_zone_changed_on_the_mcu_warns_once(caplog):
    """Old framework on the board: the event was never auto-injected, so the
    push is dropped MCU-side. Warn once, not at pose rate."""
    p = TrackingPushPolicy(push_zone_changed=True,
                           zone_change_body_part="centroid")
    pyc = _Pyc(events=())                         # sm.events has nothing
    with caplog.at_level("WARNING"):
        p.push(pyc, {})
        p.push(pyc, _zones(centroid="left"))
        p.push(pyc, {})
    assert sum("does NOT contain 'zone_changed'" in r.message
               for r in caplog.records) == 1
    # ...but the push still happens. Unlike frame_event this is NOT a gate:
    # the task's own sm.events need not list zone_changed for the framework
    # to accept it, so refusing here would break boards that work.
    assert pyc.named("intrinsic") == ["zone_changed", "zone_changed"]


def test_frame_event_is_gated_but_zone_changed_is_not(caplog):
    """The two intrinsic events treat a missing sm.events entry differently,
    and that asymmetry is deliberate, frame_event is dropped, zone_changed
    is warned about and sent anyway."""
    pyc = _Pyc(events=())
    frame_only = TrackingPushPolicy(push_frame_event=True)
    frame_only.push(pyc, {})
    assert pyc.named("intrinsic") == []            # gated

    pyc2 = _Pyc(events=())
    zone_only = TrackingPushPolicy(push_zone_changed=True,
                                   zone_change_body_part="centroid")
    zone_only.push(pyc2, {})
    zone_only.push(pyc2, _zones(centroid="left"))
    assert pyc2.named("intrinsic") == ["zone_changed"]   # sent regardless


# ── frame_event ──────────────────────────────────────────────────────────

def test_frame_event_is_off_by_default():
    p = TrackingPushPolicy()
    pyc = _Pyc()
    p.push(pyc, {})
    assert "frame_event" not in pyc.named("intrinsic")


def test_frame_event_fires_every_frame_when_enabled():
    p = TrackingPushPolicy(push_frame_event=True)
    pyc = _Pyc()
    p.push(pyc, {})
    p.push(pyc, {})
    assert pyc.named("intrinsic") == ["frame_event", "frame_event"]


def test_missing_frame_event_on_the_mcu_warns_once(caplog):
    p = TrackingPushPolicy(push_frame_event=True)
    pyc = _Pyc(events=())
    with caplog.at_level("WARNING"):
        p.push(pyc, {})
        p.push(pyc, {})
    assert sum("no 'frame_event'" in r.message for r in caplog.records) == 1
    assert pyc.named("intrinsic") == []


# ── triggers ─────────────────────────────────────────────────────────────

def test_in_zone_trigger_fires_on_the_rising_edge_only():
    p = TrackingPushPolicy(triggers=[{"event_name": "reward",
                                      "condition": "in_zone",
                                      "zones": ["left"],
                                      "body_part": "centroid"}])
    p.reset()
    pyc = _Pyc()
    p.push(pyc, _zones(centroid="left"))
    p.push(pyc, _zones(centroid="left"))          # still inside → silent
    assert pyc.named("event") == ["reward"]


def test_exit_edge_trigger_fires_on_leaving():
    p = TrackingPushPolicy(triggers=[{"event_name": "left_zone",
                                      "condition": "exit_edge",
                                      "zones": ["left"],
                                      "body_part": "centroid"}])
    p.reset()
    pyc = _Pyc()
    p.push(pyc, _zones(centroid="left"))          # enter → silent
    assert pyc.named("event") == []
    p.push(pyc, {})                               # leave → fires
    assert pyc.named("event") == ["left_zone"]


def test_a_trigger_frame_is_emitted_for_the_plot_lane():
    seen = []
    p = TrackingPushPolicy(triggers=[{"event_name": "reward",
                                      "condition": "in_zone",
                                      "zones": ["left"],
                                      "body_part": "centroid",
                                      "name": "Reward"}])
    p.reset()
    p.set_trigger_frame_cb(seen.append)
    p.push(_Pyc(), _zones(centroid="left"))
    assert len(seen) == 1
    state = seen[0].states[0]
    assert state.id == "reward" and state.active and state.fired


def test_push_without_triggers_still_emits_an_empty_frame():
    seen = []
    p = TrackingPushPolicy()
    p.set_trigger_frame_cb(seen.append)
    p.push(_Pyc(), {})
    assert len(seen) == 1 and seen[0].states == []
