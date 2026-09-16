"""Characterisation of ``PoseSink._on_pose_done``.

Pins the behaviour of a 218-line per-frame method BEFORE splitting it, so the
split is provably behaviour-preserving. Each test names a decision the method
makes:

  * shape the pose dict into a fixed-order array, rounded
  * notice the silent-failure signature (every keypoint at confidence 0)
  * pick a centroid: pinned body part, then a supplied "centroid", then the
    first confident keypoint
  * forecast ahead for the MCU while reporting the raw position for display
  * resolve zone occupancy per body part, innermost wins
  * hand subscribers exactly the fields the recorder and pusher read

The method runs on the backend worker thread, so it is called directly here,
that is how the real backend invokes it.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from source.video.framebus.pose_sink import PoseSink

BODY_PARTS = ["nose", "head", "tail"]


class _Zones:
    """Minimal ZoneManager-shaped lookup: a point is 'in' a zone when its x
    falls in the zone's span."""

    def __init__(self, spans=None):
        self.spans = spans or {"left": (0, 50), "right": (50, 100)}

    def get_zones_at_point(self, x, y):
        return [n for n, (lo, hi) in self.spans.items() if lo <= x < hi]


def _sink(confidence=0.5, body_parts=None, zones=None):
    """A PoseSink with only the state ``_on_pose_done`` reads."""
    s = PoseSink.__new__(PoseSink)
    s._lock = threading.RLock()
    s._confidence = float(confidence)
    s._body_parts = list(BODY_PARTS if body_parts is None else body_parts)
    s._centroid_body_part = {}
    s._zone_lookup = {1: zones} if zones is not None else {}
    s._enhancers = {}
    s._rotation_enabled = {}
    s._rotation_pair = {}
    s._rotation_state = {}
    s._empty_result_streak = {}
    s._empty_warned_ns = {}
    s._prev_capture_ns = {}
    s._prev_centroid = {}
    # No input transform: these cases speak the box's own pixels, which is
    # what "full" mode, or a box already at the canonical shape, produces.
    s._letterboxes = {}
    s._input_transform = {}
    s._crop_states = {}
    s._input_mode = "letterbox"
    s._latency = None
    s._on_result = []
    s._on_failed = []
    return s


def _capture(sink):
    """Collect the single notify_subscribers call ``_on_pose_done`` makes."""
    got = {}

    def _sub(setup_id, cam_frame_id, pose_array, location, speed,
             zones_by_body_part, raw_pose, **kw):
        got.update(setup_id=setup_id, cam_frame_id=cam_frame_id,
                   pose_array=pose_array, location=location, speed=speed,
                   zones=zones_by_body_part, raw=raw_pose, **kw)
    sink._on_result.append(_sub)
    return got


def _pose(nose=(10.0, 20.0, 0.9), head=(12.0, 22.0, 0.8),
          tail=(14.0, 24.0, 0.7)):
    return {"nose": list(nose), "head": list(head), "tail": list(tail)}


# ── pose array shaping ───────────────────────────────────────────────────

def test_pose_array_follows_body_part_order_and_rounds():
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 7, {"tail": [1.234567, 2.345678, 0.987654],
                           "nose": [3.0, 4.0, 0.5],
                           "head": [5.0, 6.0, 0.6]})
    arr = got["pose_array"]
    assert len(arr) == 3
    assert arr[0] == [3.0, 4.0, 0.5]                 # nose first, per order
    assert arr[2] == [1.23, 2.35, 0.988]             # xy 2dp, conf 3dp


def test_missing_body_part_becomes_a_zero_row():
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 1, {"nose": [1.0, 2.0, 0.9]})
    assert got["pose_array"][1] == [0.0, 0.0, 0.0]   # head absent
    assert got["pose_array"][2] == [0.0, 0.0, 0.0]   # tail absent


def test_keypoint_without_confidence_defaults_to_full_confidence():
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 1, {"nose": [1.0, 2.0]})
    assert got["pose_array"][0] == [1.0, 2.0, 1.0]


def test_body_parts_fall_back_to_the_pose_keys_when_unset():
    s = _sink(body_parts=[])
    got = _capture(s)
    s._on_pose_done(1, 1, {"only": [1.0, 2.0, 0.9]})
    assert got["pose_array"] == [[1.0, 2.0, 0.9]]


# ── silent-failure detection ─────────────────────────────────────────────

def test_all_zero_confidence_warns_only_after_a_real_stall():
    """The signature of a model that never loaded, but measured in TIME.

    Three consecutive empties is 0.1 s at 30 fps, which is an empty arena and
    not a stall: with a following crop window on a scene with no animal, most
    results come back empty by design because the window is searching. That
    threshold fired one second after a SLEAP model had loaded perfectly and
    blamed it for failing to initialise.
    """
    s = _sink()
    fails = []
    s._on_failed.append(lambda sid, reason, streak, **kw:
                        fails.append((sid, streak)))
    dead = {p: [0.0, 0.0, 0.0] for p in BODY_PARTS}
    n = s._EMPTY_STALL_FRAMES
    for _ in range(n - 1):
        s._on_pose_done(1, 1, dict(dead))
    assert fails == [], "warned before the stall threshold"
    s._on_pose_done(1, 1, dict(dead))
    assert fails == [(1, n)]


def test_an_empty_arena_never_trips_it():
    """The case that made the old warning noise: no animal in view, a crop
    window searching, a handful of empty results between detections."""
    s = _sink()
    fails = []
    s._on_failed.append(lambda sid, reason, streak, **kw:
                        fails.append(sid))
    dead = {p: [0.0, 0.0, 0.0] for p in BODY_PARTS}
    for _ in range(40):                 # the animal wanders in and out
        for _ in range(5):
            s._on_pose_done(1, 1, dict(dead))
        s._on_pose_done(1, 1, _pose())
    assert fails == [], f"warned {len(fails)} time(s) for an empty arena"


def test_the_warning_names_the_backend_that_is_running():
    """It said "DLC" and "check the DLCLiveTracker traceback" whatever the
    backend was, so a SLEAP failure was reported as a DLC one."""
    s = _sink()
    s._tracker_type = "sleap"
    said = []
    s._on_failed.append(lambda sid, reason, streak, **kw: said.append(reason))
    dead = {p: [0.0, 0.0, 0.0] for p in BODY_PARTS}
    for _ in range(s._EMPTY_STALL_FRAMES):
        s._on_pose_done(1, 1, dict(dead))
    assert said, "no warning raised"
    assert "sleap" in said[0].lower()
    assert "DLCLiveTracker" not in said[0]


def test_a_good_result_resets_the_streak():
    s = _sink()
    dead = {p: [0.0, 0.0, 0.0] for p in BODY_PARTS}
    s._on_pose_done(1, 1, dict(dead))
    s._on_pose_done(1, 2, _pose())
    assert s._empty_result_streak[1] == 0


def test_a_flapping_model_does_not_re_warn_every_time_it_recovers():
    """One frame with any confidence resets the count, so without a time floor
    a model that finds something occasionally climbs back to the threshold and
    warns again, forever. Seen on a camera with no signal: hundreds of
    identical lines in one run, which is how a log stops being read."""
    s = _sink()
    fails = []
    s._on_failed.append(lambda sid, reason, streak, **kw:
                        fails.append((sid, streak)))
    dead = {p: [0.0, 0.0, 0.0] for p in BODY_PARTS}
    for cycle in range(4):
        for _ in range(s._EMPTY_STALL_FRAMES):
            s._on_pose_done(1, cycle, dict(dead))
        s._on_pose_done(1, cycle, _pose())
    assert len(fails) == 1, f"warned {len(fails)} times for one flapping model"


def test_each_box_is_throttled_on_its_own():
    """One box's warning must not silence another's, they fail separately."""
    s = _sink()
    fails = []
    s._on_failed.append(lambda sid, reason, streak, **kw:
                        fails.append(sid))
    dead = {p: [0.0, 0.0, 0.0] for p in BODY_PARTS}
    for box in (1, 2, 3):
        s._zone_lookup.setdefault(box, None)
        for _ in range(s._EMPTY_STALL_FRAMES):
            s._on_pose_done(box, 1, dict(dead))
    assert sorted(fails) == [1, 2, 3]


# ── centroid selection ───────────────────────────────────────────────────

def test_pinned_body_part_defines_the_centroid():
    s = _sink(zones=_Zones())
    s.set_centroid_body_part(1, "tail")
    got = _capture(s)
    s._on_pose_done(1, 1, _pose(tail=(80.0, 5.0, 0.9)))
    assert got["zones"].get("centroid") == {"right": True}


def test_pinned_body_part_below_confidence_yields_no_centroid():
    s = _sink(zones=_Zones())
    s.set_centroid_body_part(1, "tail")
    got = _capture(s)
    s._on_pose_done(1, 1, _pose(tail=(80.0, 5.0, 0.1)))
    assert "centroid" not in got["zones"]
    assert got["location"] is None


def test_a_supplied_centroid_key_is_used_when_nothing_is_pinned():
    s = _sink(zones=_Zones())
    got = _capture(s)
    pose = _pose()
    pose["centroid"] = [70.0, 5.0, 0.9]
    s._on_pose_done(1, 1, pose)
    assert got["zones"].get("centroid") == {"right": True}


def test_otherwise_the_first_confident_keypoint_wins():
    s = _sink(zones=_Zones())
    got = _capture(s)
    # nose below threshold, head above → head defines the centroid
    s._on_pose_done(1, 1, _pose(nose=(80.0, 1.0, 0.1),
                                head=(10.0, 1.0, 0.9)))
    assert got["zones"].get("centroid") == {"left": True}


# ── zone occupancy ───────────────────────────────────────────────────────

def test_each_confident_body_part_gets_its_own_zone():
    s = _sink(zones=_Zones())
    got = _capture(s)
    s._on_pose_done(1, 1, _pose(nose=(10.0, 1.0, 0.9),
                                head=(80.0, 1.0, 0.9),
                                tail=(20.0, 1.0, 0.9)))
    assert got["zones"]["nose"] == {"left": True}
    assert got["zones"]["head"] == {"right": True}
    assert got["zones"]["tail"] == {"left": True}


def test_a_low_confidence_body_part_is_not_placed_in_a_zone():
    s = _sink(zones=_Zones())
    got = _capture(s)
    s._on_pose_done(1, 1, _pose(head=(80.0, 1.0, 0.1)))
    assert "head" not in got["zones"]


def test_location_comes_from_the_raw_centroid():
    """_video_data.txt and the overlay must describe the captured frame, so
    ``location`` is never the forward-predicted position."""
    s = _sink(zones=_Zones())
    got = _capture(s)
    s._on_pose_done(1, 1, _pose(nose=(10.0, 1.0, 0.9)))
    assert got["location"] == "left"


def test_no_zone_lookup_means_no_zones():
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 1, _pose())
    assert got["zones"] == {}
    assert got["location"] is None


# ── what subscribers receive ─────────────────────────────────────────────

def test_subscribers_get_the_capture_instant_not_the_inference_instant():
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 42, _pose(), capture_host_ns=123456789)
    assert got["capture_host_ns"] == 123456789
    assert got["cam_frame_id"] == 42
    assert got["infer_done_ns"] > 0


def test_raw_pose_dict_is_passed_through():
    s = _sink()
    got = _capture(s)
    pose = _pose()
    s._on_pose_done(1, 1, pose)
    assert got["raw"] is pose


def test_speed_is_zero_on_the_first_frame_then_measured():
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 1, _pose(nose=(0.0, 0.0, 0.9)))
    assert got["speed"] == 0.0
    s._on_pose_done(1, 2, _pose(nose=(30.0, 40.0, 0.9)))
    assert got["speed"] >= 0.0


def test_without_a_forecast_no_forecast_coords_are_sent():
    """No enhancer → the MCU push carries no latency-compensated override."""
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 1, _pose())
    assert got["forecast_coords"] is None


# ── rotation ─────────────────────────────────────────────────────────────

def test_rotation_is_absent_unless_enabled():
    s = _sink()
    pose = _pose()
    s._on_pose_done(1, 1, pose)
    assert "_rotation_rad" not in pose


def test_rotation_writes_a_value_when_enabled():
    s = _sink()
    s._rotation_enabled[1] = True
    s._rotation_pair[1] = ("nose", "tail")
    s._rotation_state[1] = {}
    pose = _pose()
    s._on_pose_done(1, 1, pose)
    assert "_rotation_rad" in pose


# ── robustness ───────────────────────────────────────────────────────────

def test_a_broken_zone_lookup_does_not_kill_the_frame():
    class _Boom:
        def get_zones_at_point(self, x, y):
            raise RuntimeError("zone backend down")
    s = _sink(zones=_Boom())
    got = _capture(s)
    s._on_pose_done(1, 1, _pose())
    assert got["pose_array"], "subscribers must still be notified"


def test_a_malformed_pose_entry_does_not_kill_the_frame():
    s = _sink()
    got = _capture(s)
    s._on_pose_done(1, 1, {"nose": "not-a-point", "head": [1.0, 2.0, 0.9]})
    assert "pose_array" in got
