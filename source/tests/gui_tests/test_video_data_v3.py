"""M2, the self-describing, per-track ``_video_data.txt`` v3 format.

Single-animal output stays byte-compatible (flat ``[[x,y,c],…]`` pose column);
multi-animal writes a per-track ``{track_id: [[x,y,c],…]}`` map plus a
self-describing ``#tracker`` header (backend / model_type / n_animals /
identities) so a reader never has to guess which tracker produced the row.
"""
from __future__ import annotations

from pathlib import Path

from source.video.recording.frame_log import open_with_headers
from source.video.tracking.types import PoseSnapshot, InstanceSnapshot
from source.video.framebus.types import TrackingConfig


def _read(p) -> str:
    return Path(p).read_text(encoding="utf-8")


def test_single_animal_pose_column_is_flat_array(tmp_path):
    log = open_with_headers(str(tmp_path / "v.txt"), setup_id=1,
                            subject_id="s", tracking_mode="dlc")
    log.write_frame(0, 0, pose_array=[[10, 20, 0.9], [30, 40, 0.8]])
    log.close()
    txt = _read(tmp_path / "v.txt")
    # Flat JSON array, byte-compatible with pre-multi files (not an object).
    assert "[[10.0,20.0,0.9],[30.0,40.0,0.8]]" in txt
    assert '"n_animals"' not in txt          # single-animal: no multi header noise


def test_multi_animal_pose_column_per_track_and_header(tmp_path):
    log = open_with_headers(str(tmp_path / "v.txt"), setup_id=1, subject_id="s",
                            tracking_mode="sleap", model_type="topdown",
                            tracker="oks/hungarian", n_animals=2,
                            identities=["male", "female"])
    ps = PoseSnapshot(capture_fw_ms=0.0, instances=(
        InstanceSnapshot({"nose": (1.0, 2.0, 0.9)}, track_id="male"),
        InstanceSnapshot({"nose": (8.0, 9.0, 0.8)}, track_id="female")))
    log.write_frame(0, 0, pose_snapshot=ps)
    log.close()
    txt = _read(tmp_path / "v.txt")
    # Self-describing header: a reader branches on these without guessing.
    assert '"backend":"sleap"' in txt or '"backend": "sleap"' in txt
    assert "n_animals" in txt and "topdown" in txt
    assert "male" in txt and "female" in txt
    # Per-track pose column keyed by identity.
    assert '{"male":[[1.0,2.0,0.9]],"female":[[8.0,9.0,0.8]]}' in txt


def test_config_multianimal_roundtrip():
    tc = TrackingConfig(setup_id=1, tracker_type="sleap", n_animals=2,
                        identity_method="tracker", identities=("m", "f"))
    back = TrackingConfig.from_json(tc.to_json(), setup_id=1)
    assert back.n_animals == 2
    assert back.identity_method == "tracker"
    assert back.identities == ("m", "f")
    # Defaults for a single-animal / non-SLEAP config.
    d = TrackingConfig(setup_id=1, tracker_type="blob")
    assert d.n_animals == 1 and d.identity_method == "none"
