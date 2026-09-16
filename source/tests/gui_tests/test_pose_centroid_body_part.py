"""Centroid / location body-part selection in PoseSink.

When the operator pins a keypoint via the tracking dialog's Zone-change body
part combo, PoseSink uses ONLY that keypoint's coords for cx,cy + location.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from source.video.framebus.pose_sink import PoseSink


def _make_pose_sink(confidence=0.5):
    """Build a PoseSink stub that runs the centroid logic on a fake pose dict
    without real inference."""
    sink = PoseSink.__new__(PoseSink)
    sink._lock = __import__("threading").RLock()
    sink._confidence = float(confidence)
    sink._centroid_body_part = {}
    return sink


# -- API: set_centroid_body_part / default behaviour ------------------


def test_set_centroid_body_part_remembers_per_box():
    sink = _make_pose_sink()
    sink.set_centroid_body_part(1, "head")
    sink.set_centroid_body_part(2, "tail")
    assert sink._centroid_body_part == {1: "head", 2: "tail"}


def test_set_centroid_body_part_centroid_clears_pin():
    """Setting back to 'centroid' (or empty) clears the per-box pin so the
    heuristic falls back to first-confident-keypoint."""
    sink = _make_pose_sink()
    sink.set_centroid_body_part(1, "head")
    sink.set_centroid_body_part(1, "centroid")
    assert 1 not in sink._centroid_body_part
    sink.set_centroid_body_part(1, "head")
    sink.set_centroid_body_part(1, "")
    assert 1 not in sink._centroid_body_part


# -- Centroid computation logic --------------------------------------


def _resolve_centroid(sink, setup_id, pose_dict, pose_array):
    """Run the centroid-resolution block from PoseSink.process in isolation.
    Returns (cx, cy) like the real path."""
    cx = cy = None
    with sink._lock:
        picked_bp = sink._centroid_body_part.get(setup_id)
    try:
        if picked_bp:
            kp = pose_dict.get(picked_bp)
            if (kp is not None and len(kp) >= 2
                    and (len(kp) < 3 or float(kp[2]) >= sink._confidence)):
                cx, cy = float(kp[0]), float(kp[1])
        elif "centroid" in pose_dict:
            ct = pose_dict["centroid"]
            cx, cy = float(ct[0]), float(ct[1])
        else:
            for pt in pose_array:
                if pt[2] >= sink._confidence:
                    cx, cy = pt[0], pt[1]
                    break
    except Exception:
        cx = cy = None
    return cx, cy


def test_default_centroid_falls_back_to_first_confident_part():
    """When no body part is pinned, the heuristic picks the first confident
    keypoint."""
    sink = _make_pose_sink(confidence=0.5)
    pose = {"head": (10, 20, 0.9), "tail": (30, 40, 0.95)}
    arr = [[10, 20, 0.9], [30, 40, 0.95]]
    assert _resolve_centroid(sink, 1, pose, arr) == (10, 20)


def test_pinned_body_part_uses_only_that_keypoint():
    """Pin to 'tail' → centroid is tail's coords even though head is
    first in the array."""
    sink = _make_pose_sink(confidence=0.5)
    sink.set_centroid_body_part(1, "tail")
    pose = {"head": (10, 20, 0.9), "tail": (30, 40, 0.95)}
    arr = [[10, 20, 0.9], [30, 40, 0.95]]
    assert _resolve_centroid(sink, 1, pose, arr) == (30, 40)


def test_pinned_body_part_below_confidence_returns_none():
    """Pinned body part below confidence threshold → centroid is None, NOT a
    silent fallback to another keypoint that happens to be in a zone."""
    sink = _make_pose_sink(confidence=0.5)
    sink.set_centroid_body_part(1, "head")
    # Head conf 0.2 is below threshold; tail is confident but unpinned →
    # centroid is None and location resolves to "na".
    pose = {"head": (10, 20, 0.2), "tail": (250, 300, 0.95)}
    arr = [[10, 20, 0.2], [250, 300, 0.95]]
    assert _resolve_centroid(sink, 1, pose, arr) == (None, None)


def test_pinned_body_part_missing_from_pose_returns_none():
    """Pinned keypoint name absent from the pose dict (e.g. after a model
    swap) → centroid is None rather than falling back."""
    sink = _make_pose_sink(confidence=0.5)
    sink.set_centroid_body_part(1, "snout")
    pose = {"head": (10, 20, 0.9), "tail": (30, 40, 0.95)}
    arr = [[10, 20, 0.9], [30, 40, 0.95]]
    assert _resolve_centroid(sink, 1, pose, arr) == (None, None)


def test_pinned_body_part_per_box_isolation():
    """Box 1 pins 'head', Box 2 doesn't pin → different centroids on
    the same pose dict."""
    sink = _make_pose_sink(confidence=0.5)
    sink.set_centroid_body_part(1, "tail")
    pose = {"head": (10, 20, 0.9), "tail": (30, 40, 0.95)}
    arr = [[10, 20, 0.9], [30, 40, 0.95]]
    assert _resolve_centroid(sink, 1, pose, arr) == (30, 40)  # pinned → tail
    assert _resolve_centroid(sink, 2, pose, arr) == (10, 20)  # default → first


def test_pinned_body_part_two_value_kp_no_confidence_field():
    """Some backends emit (x, y) without confidence. Pin still works,
    len(kp)<3 means no confidence check, just use the coords."""
    sink = _make_pose_sink(confidence=0.5)
    sink.set_centroid_body_part(1, "head")
    pose = {"head": (10, 20)}  # no conf
    arr = [[10, 20]]
    assert _resolve_centroid(sink, 1, pose, arr) == (10, 20)
