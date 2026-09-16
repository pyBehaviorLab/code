"""Wave 4: MCU trigger-chain correctness + observability (M1, M2, M4, M0).

  M1  a trigger event not in the task's events list → a NAMED, actionable
      warning, not "port may be down", and no transport-error path.
  M2  authored triggers restored on TrackingConfig survive a tracking start,
      _apply_push_policy merges them, not only session dialog globals.
  M4  exit_edge is a real evaluator case (rule-state not stuck False).
  M0  push counters advance so "is it sending?" is observable.
"""
from types import SimpleNamespace

from source.video.framebus.mcu_pusher import TrackingPushPolicy
from source.gui.base import MainWindowBase


# ── M0: push counters ──────────────────────────────────────────────────────

class _Board:
    framework_running = True

    def __init__(self):
        self.coords = []
        self.events = []
        self.intrinsic = []

    def queue_set_coordinates(self, name, val):
        self.coords.append((name, val))

    def queue_trigger_event(self, name):
        self.events.append(name)

    def queue_trigger_intrinsic_event(self, name):
        self.intrinsic.append(name)


def test_push_stats_count_coords_and_zone_events():
    p = TrackingPushPolicy(coord_mapping={"loc_center": "centroid"})
    board = _Board()
    # Two pushes with the body part in a zone → coord write then zone_changed.
    p.push(board, {"centroid": {"A": True}})
    p.push(board, {"centroid": {}})   # occupancy changed → zone_changed fires
    s = p.stats()
    assert s["coords_pushed"] >= 1
    assert s["zone_events"] >= 1
    assert s["last_push_age_s"] is not None


def test_push_stats_reset_zeroes_counters():
    p = TrackingPushPolicy(coord_mapping={"loc_center": "centroid"})
    p.push(_Board(), {"centroid": {"A": True}})
    p.reset()
    s = p.stats()
    assert s["coords_pushed"] == 0 and s["zone_events"] == 0
    assert s["last_push_age_s"] is None


# ── M4: exit_edge evaluator case ────────────────────────────────────────────

def test_exit_edge_rule_state_tracks_outside_zone():
    trig = {"condition": "exit_edge", "body_part": "centroid",
            "zones": ["A"], "event_name": "left_A"}
    p = TrackingPushPolicy(triggers=[trig])
    active_inside, _, _ = p._evaluate_condition(
        trig, {"centroid": {"A": True}}, 0.0)
    active_outside, _, _ = p._evaluate_condition(
        trig, {"centroid": {"A": False}}, 0.0)
    # Inside the zone → not "exited"; outside → active. Previously stuck False.
    assert active_inside is False
    assert active_outside is True


# ── M1: named unknown-event warning ─────────────────────────────────────────

def test_unknown_event_name_warns_named_not_port_down(caplog):
    from source.communication import pycboard as pcb

    board = pcb.Pycboard.__new__(pcb.Pycboard)
    board.framework_running = True
    board.sm_info = SimpleNamespace(events={"poke": 1})   # "left_A" is NOT here
    import queue as _q
    board._pending_writes = _q.Queue()
    board._pending_writes.put(("event", 0, "left_A", "u"))

    with caplog.at_level("WARNING"):
        board._drain_pending_writes()
    text = caplog.text.lower()
    assert "not in the running task" in text or "events list" in text
    assert "port may be down" not in text


# ── M2: authored triggers survive tracking start ────────────────────────────

class _Pipe:
    def __init__(self, tc_triggers):
        self._tc = SimpleNamespace(
            triggers=tc_triggers, zone_change_body_part="centroid",
            push_coords_to_mcu=True, push_zones_to_mcu=True)
        self.calls = []

    def get_tracking_config(self, sid):
        return self._tc

    def configure_push_zones(self, sid, zones, **kw):
        self.calls.append(("zones", sid))

    def configure_push_tracking(self, sid, cfg):
        self.calls.append(("tracking", sid, cfg))

    def update_tracking_config(self, sid, **f):
        pass


def test_apply_push_policy_merges_tc_triggers():
    # Triggers restored from the project onto the TC, dialog never opened this
    # session (_tracking_dialog_globals empty). They MUST still be applied.
    authored = [{"condition": "in_zone", "zones": ["A"], "event_name": "enter_A"}]
    pipe = _Pipe(authored)
    host = SimpleNamespace(
        pipeline=pipe,
        tracking_zones={1: [{"name": "A", "points": [[0.1, 0.1]]}]},
        _tracking_dialog_globals={},
    )
    MainWindowBase._apply_push_policy(host, 1)
    tracking_calls = [c for c in pipe.calls if c[0] == "tracking"]
    assert tracking_calls, "configure_push_tracking never called"
    sent = tracking_calls[-1][2]["triggers"]
    assert any(t.get("event_name") == "enter_A" for t in sent)
