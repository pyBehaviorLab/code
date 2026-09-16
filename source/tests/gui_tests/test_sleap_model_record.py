"""S4, SLEAP model recorded in the run snapshot.

The DLC/SLEAP model manifest now (a) hashes sleap-nn files (``.ckpt`` +
``training_config.*``) that the old patterns missed, and (b) folds a top-down
**centroid** model into the same manifest so a two-model pipeline is fully
recorded.
"""
from __future__ import annotations

from pathlib import Path

from source.config.snapshot_store import _build_dlc_manifest


def _mk_model(root: Path, name: str, weights: bytes) -> Path:
    d = root / name
    d.mkdir()
    (d / "best.ckpt").write_bytes(weights)
    (d / "training_config.json").write_text("{}", encoding="utf-8")
    return d


def test_sleap_nn_files_are_hashed(tmp_path):
    # Pre-S4 patterns missed .ckpt / training_config → sleap-nn weights weren't
    # recorded at all. They must be captured now.
    m = _mk_model(tmp_path, "single", b"weights-v1")
    _, man = _build_dlc_manifest(m)
    rels = {f["rel_path"] for f in man["files"]}
    assert "best.ckpt" in rels and "training_config.json" in rels
    ckpt = next(f for f in man["files"] if f["rel_path"] == "best.ckpt")
    assert ckpt["djb2"]                         # non-empty hash


def test_centroid_model_folded_into_manifest(tmp_path):
    centered = _mk_model(tmp_path, "centered", b"centered-weights")
    centroid = _mk_model(tmp_path, "centroid", b"centroid-weights")

    _, man_solo = _build_dlc_manifest(centered)
    _, man_pair = _build_dlc_manifest(centered, extra_paths=[centroid])
    rels = {f["rel_path"] for f in man_pair["files"]}

    assert "best.ckpt" in rels                                   # primary stage
    assert any(r.startswith("__extra0__/") for r in rels)        # centroid folded in
    # The paired manifest hashes strictly more files than the solo one.
    assert len(man_pair["files"]) == len(man_solo["files"]) + 2


def test_changing_centroid_changes_the_hashes(tmp_path):
    centered = _mk_model(tmp_path, "centered", b"same")
    c1 = _mk_model(tmp_path, "c1", b"centroid-A")
    c2 = _mk_model(tmp_path, "c2", b"centroid-B")
    _, m1 = _build_dlc_manifest(centered, extra_paths=[c1])
    _, m2 = _build_dlc_manifest(centered, extra_paths=[c2])
    h1 = {f["rel_path"]: f["djb2"] for f in m1["files"]}
    h2 = {f["rel_path"]: f["djb2"] for f in m2["files"]}
    # different centroid weights → different centroid-file hash.
    assert h1["__extra0__/best.ckpt"] != h2["__extra0__/best.ckpt"]
