"""TrackingPushPolicy evaluates the advanced conditions and emits a per-frame
trigger frame. A condition the UI offers and the policy never evaluates is
worse than one that is not offered at all."""
from __future__ import annotations

from source.video.framebus.mcu_pusher import TrackingPushPolicy


class _Pyc:
    framework_running = True

    def __init__(self):
        self.events = []
        self.coords = {}

    def queue_trigger_event(self, name):
        self.events.append(name)

    def queue_set_coordinates(self, name, val):
        self.coords[name] = val

    def queue_trigger_intrinsic_event(self, name):
        pass


def _policy(trig):
    p = TrackingPushPolicy(triggers=[trig])
    p.reset()
    return p


def test_speed_gt_fires():
    p = _policy({"condition": "speed_gt", "threshold": 50, "event_name": "fast"})
    pyc = _Pyc()
    p.push(pyc, {}, speed=100.0, body_part_coords={"centroid": (0, 0)})
    assert pyc.events == ["fast"]


def test_speed_gt_below_threshold_silent():
    p = _policy({"condition": "speed_gt", "threshold": 50, "event_name": "fast"})
    pyc = _Pyc()
    p.push(pyc, {}, speed=10.0, body_part_coords={"centroid": (0, 0)})
    assert pyc.events == []


def test_speed_gt_with_no_threshold_does_not_raise():
    # An unset threshold must no-op quietly: ``value > None`` raises
    # TypeError, and that aborts the rest of the per-frame trigger evaluation
    #, every later rule stops firing because one was left half-configured.
    p = _policy({"condition": "speed_gt", "event_name": "fast"})   # no threshold
    pyc = _Pyc()
    p.push(pyc, {}, speed=999.0, body_part_coords={"centroid": (0, 0)})
    assert pyc.events == []       # didn't fire, and (crucially) didn't raise


def test_speed_lt_freezing():
    p = _policy({"condition": "freezing", "threshold": 10, "event_name": "frozen"})
    pyc = _Pyc()
    p.push(pyc, {}, speed=3.0, body_part_coords={"centroid": (0, 0)})
    assert pyc.events == ["frozen"]


def test_distance_condition_fires():
    p = _policy({"condition": "distance_lt", "part_a": "nose", "part_b": "tail",
                 "threshold": 40, "event_name": "near"})
    pyc = _Pyc()
    p.push(pyc, {}, speed=0.0,
           body_part_coords={"nose": (0, 0), "tail": (30, 0), "centroid": (15, 0)})
    assert pyc.events == ["near"]


def test_distance_gt_and_head_angle_conditions():
    # distance_gt: nose↔tail = 100 px > 40 → fires
    p = _policy({"condition": "distance_gt", "part_a": "nose", "part_b": "tail",
                 "threshold": 40, "event_name": "far"})
    pyc = _Pyc()
    p.push(pyc, {}, speed=0.0,
           body_part_coords={"nose": (0, 0), "tail": (100, 0), "centroid": (50, 0)})
    assert pyc.events == ["far"]


def test_edge_only_no_refire():
    p = _policy({"condition": "speed_gt", "threshold": 50, "event_name": "fast"})
    pyc = _Pyc()
    p.push(pyc, {}, speed=100.0, body_part_coords={"centroid": (0, 0)})
    p.push(pyc, {}, speed=120.0, body_part_coords={"centroid": (0, 0)})  # still active
    assert pyc.events == ["fast"]                                        # fired once


def test_zone_condition_still_works():
    p = _policy({"condition": "in_zone", "body_part": "centroid",
                 "zones": ["reward"], "event_name": "in_reward"})
    pyc = _Pyc()
    p.push(pyc, {"centroid": {"reward": True}}, speed=0.0,
           body_part_coords={"centroid": (0, 0)})
    assert pyc.events == ["in_reward"]


def test_emits_trigger_frame():
    p = _policy({"condition": "speed_gt", "threshold": 50, "event_name": "fast",
                 "name": "running", "color": "#e3b341"})
    frames = []
    p.set_trigger_frame_cb(frames.append, setup_id=3)
    pyc = _Pyc()
    p.push(pyc, {}, speed=100.0, body_part_coords={"centroid": (0, 0)})
    assert frames and frames[0].setup_id == 3
    s = frames[0].states[0]
    assert s.name == "running" and s.active and s.fired
    assert s.value == 100.0 and s.color == "#e3b341"


def test_scale_unit_speed():
    # speed threshold in cm/s with a 10 px/mm scale (→ 100 px/cm)
    p = _policy({"condition": "speed_gt", "threshold": 2.0, "unit": "cm",
                 "event_name": "fast"})
    p.set_features_context(px_per_mm=10.0)
    pyc = _Pyc()
    p.push(pyc, {}, speed=300.0, body_part_coords={"centroid": (0, 0)})  # 3 cm/s > 2
    assert pyc.events == ["fast"]
