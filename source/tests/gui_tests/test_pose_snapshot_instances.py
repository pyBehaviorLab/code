"""M1, instance/track-ID data model (behaviour-preserving for single-animal).

``PoseSnapshot`` is now a list of ``InstanceSnapshot``; single-animal is the
1-instance case and its accessors delegate to ``.primary`` so existing callers
are untouched. Multi-animal fills more instances, each with a ``track_id``.
"""
from __future__ import annotations

from source.video.tracking.types import PoseSnapshot, InstanceSnapshot


# ── single-animal: unchanged behaviour via .primary ────────────────────

def test_single_delegates_to_primary():
    ps = PoseSnapshot.single(
        capture_fw_ms=16.0,
        body_parts={"nose": (10.0, 20.0, 0.9), "tail": (30.0, 40.0, 0.8)},
        confidence_threshold=0.5, inference_fw_ms=20.0)
    assert ps.n_instances == 1
    # back-compat accessors read the primary instance:
    assert ps.body_parts == {"nose": (10.0, 20.0, 0.9), "tail": (30.0, 40.0, 0.8)}
    assert ps.centroid == (20.0, 30.0)
    assert ps.get_part("nose") == (10.0, 20.0)
    assert ps.get_part("centroid") == (20.0, 30.0)
    assert set(ps.part_names) == {"nose", "tail"}
    assert ps.confident_parts == {"nose": (10.0, 20.0), "tail": (30.0, 40.0)}
    assert ps.inference_latency_ms == 4.0


def test_confidence_gate_still_applies():
    ps = PoseSnapshot.single(
        capture_fw_ms=0.0,
        body_parts={"a": (1.0, 1.0, 0.9), "b": (5.0, 5.0, 0.1)},
        confidence_threshold=0.5)
    # low-confidence part excluded from centroid and get_part.
    assert ps.centroid == (1.0, 1.0)
    assert ps.get_part("b") is None


def test_empty_snapshot_is_safe():
    ps = PoseSnapshot(capture_fw_ms=0.0)
    assert ps.n_instances == 0 and ps.primary is None
    assert ps.body_parts == {} and ps.centroid is None
    assert ps.get_part("nose") is None and ps.part_names == []


# ── multi-animal: instances + track IDs ────────────────────────────────

def test_multi_instance_primary_and_ids():
    male = InstanceSnapshot({"nose": (1.0, 1.0, 0.99)}, track_id="male", score=0.95)
    female = InstanceSnapshot({"nose": (9.0, 9.0, 0.98)}, track_id="female")
    ps = PoseSnapshot(capture_fw_ms=5.0, instances=(male, female))
    assert ps.n_instances == 2
    assert ps.primary is male                 # focal = instance 0
    assert ps.body_parts == {"nose": (1.0, 1.0, 0.99)}   # delegates to primary
    assert [i.track_id for i in ps.instances] == ["male", "female"]
    assert ps.instances[0].score == 0.95


# ── serialization round-trip (+ back-compat) ───────────────────────────

def test_to_from_dict_preserves_instances_and_ids():
    ps = PoseSnapshot(
        capture_fw_ms=7.0,
        instances=(InstanceSnapshot({"a": (1.0, 2.0, 0.9)}, track_id="m", score=0.8),
                   InstanceSnapshot({"a": (3.0, 4.0, 0.7)}, track_id="f")),
        inference_fw_ms=9.0)
    back = PoseSnapshot.from_dict(ps.to_dict())
    assert back.n_instances == 2
    assert back.instances[0].track_id == "m" and back.instances[0].score == 0.8
    assert back.instances[1].track_id == "f"
    assert back.instances[0].body_parts == {"a": (1.0, 2.0, 0.9)}
    assert back.inference_fw_ms == 9.0


def test_from_dict_back_compat_old_body_parts_shape():
    # A pre-M1 dict (single body_parts, no "instances") loads as 1 instance.
    old = {"capture_fw_ms": 3.0, "body_parts": {"nose": [1.0, 2.0, 0.9]},
           "confidence_threshold": 0.5}
    ps = PoseSnapshot.from_dict(old)
    assert ps.n_instances == 1
    assert ps.body_parts == {"nose": (1.0, 2.0, 0.9)}
