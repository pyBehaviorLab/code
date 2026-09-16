"""A model file must be able to describe itself.

These pin the claims the tracking dialog will make on the operator's behalf.
The two DeepLabCut cases are read from the real configs in ``models/`` when
they are present, because a schema invented in a test proves nothing about the
files this rig actually loads.
"""
import json
from pathlib import Path

import pytest

from source.video.tracking.model_config import GRAYSCALE, RGB, ModelInfo

REPO = Path(__file__).resolve().parents[3]


# ── nothing to read ──────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["", "does/not/exist", "."])
def test_missing_config_is_not_an_error(path):
    """A model with no config must degrade, never raise; it is still loadable."""
    info = ModelInfo.read(path)
    assert info.ok is False
    assert info.family == "unknown"
    assert info.body_parts == ()


def test_unparseable_config_is_reported_not_raised(tmp_path):
    (tmp_path / "pose_cfg.yaml").write_text("{{{ not yaml", encoding="utf-8")
    info = ModelInfo.read(str(tmp_path))
    assert info.ok is False or info.warnings


# ── SLEAP ────────────────────────────────────────────────────────────────

def _sleap(tmp_path, head, preprocessing=None, backbone=None):
    cfg = {
        "data_config": {"preprocessing": preprocessing or {}},
        "model_config": {"head_configs": head,
                         "backbone_config": backbone or {}},
    }
    (tmp_path / "training_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return ModelInfo.read(str(tmp_path))


def test_sleap_single_instance(tmp_path):
    info = _sleap(tmp_path, {"single_instance": {"confmaps": {"sigma": 5.0}}})
    assert info.backend == "sleap"
    assert info.family == "single"
    assert info.multi_animal is False
    assert info.needs_centroid is False


@pytest.mark.parametrize("head,family,needs_centroid", [
    ("centroid", "centroid", False),
    ("centered_instance", "centered_instance", True),
    ("bottomup", "bottomup", False),
    ("multi_class_topdown", "multi_class_topdown", True),
    ("multi_class_bottomup", "multi_class_bottomup", False),
])
def test_sleap_families(tmp_path, head, family, needs_centroid):
    info = _sleap(tmp_path, {head: {"confmaps": {}}})
    assert info.family == family
    assert info.needs_centroid is needs_centroid


class TestIdentities:
    """Identity names come from the model, never from the operator."""

    def test_top_down_id_reads_class_vectors(self, tmp_path):
        info = _sleap(tmp_path, {"multi_class_topdown": {
            "class_vectors": {"classes": ["mouse_A", "mouse_B"]}}})
        assert info.identities == ("mouse_A", "mouse_B")
        assert info.n_identities == 2, "the count IS the length of the class list"

    def test_bottom_up_id_reads_class_maps(self, tmp_path):
        info = _sleap(tmp_path, {"multi_class_bottomup": {
            "class_maps": {"classes": ["a", "b", "c"]}}})
        assert info.identities == ("a", "b", "c")
        assert info.n_identities == 3

    def test_id_model_without_names_warns(self, tmp_path):
        """SLEAP allows null classes, inferred from the labels at training
        time. The operator must be told, not silently given zero animals."""
        info = _sleap(tmp_path, {"multi_class_bottomup": {"class_maps": {}}})
        assert info.identities == ()
        assert any("class names" in w for w in info.warnings)

    def test_non_identity_model_has_no_identities(self, tmp_path):
        assert _sleap(tmp_path, {"bottomup": {}}).identities == ()


class TestPreprocessing:
    def test_scale_and_crop_are_read(self, tmp_path):
        info = _sleap(tmp_path, {"centered_instance": {}},
                      preprocessing={"scale": 0.5, "crop_size": 192})
        assert info.native_scale == 0.5
        assert info.crop_size == 192

    def test_min_crop_size_is_NOT_the_crop_size(self, tmp_path):
        """It is a floor for dynamic cropping, not the input the model takes.

        A real v4 config states ``min_crop_size: 100`` alongside a genuine
        224x176 input; reading the 100 as the window configured the crop to a
        size the model was never trained at, and one not divisible by its
        stride of 16, so the encoder arithmetic would break too.
        """
        info = _sleap(tmp_path, {"centered_instance": {}},
                      preprocessing={"min_crop_size": 96})
        assert info.crop_size is None

    def test_the_input_window_comes_from_max_width_and_height(self, tmp_path):
        """How sleap-nn actually states the size it was trained to take."""
        info = _sleap(tmp_path, {"single_instance": {}},
                      preprocessing={"max_width": 224, "max_height": 176})
        assert (info.input_w, info.input_h) == (224, 176)

    def test_an_unstated_window_is_None_not_a_guess(self, tmp_path):
        info = _sleap(tmp_path, {"single_instance": {}}, preprocessing={})
        assert info.input_w is None and info.input_h is None

    @pytest.mark.parametrize("pre,expect", [
        ({"ensure_grayscale": True}, GRAYSCALE),
        ({"ensure_rgb": True}, RGB),
        ({}, None),
    ])
    def test_channels(self, tmp_path, pre, expect):
        """None means the file did not say, a different claim from a default."""
        assert _sleap(tmp_path, {"bottomup": {}}, preprocessing=pre).channels == expect

    def test_backbone_and_stride(self, tmp_path):
        info = _sleap(tmp_path, {"bottomup": {}},
                      backbone={"unet": {"max_stride": 16, "output_stride": 2}})
        assert info.backbone == "unet"
        assert info.max_stride == 16


def test_sleap_skeleton_nodes_become_body_parts(tmp_path):
    cfg = {"data_config": {"preprocessing": {},
                           "skeletons": [{"nodes": [{"name": "snout"},
                                                    {"name": "tail"}]}]},
           "model_config": {"head_configs": {"single_instance": {}}}}
    (tmp_path / "training_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert ModelInfo.read(str(tmp_path)).body_parts == ("snout", "tail")


# ── DeepLabCut, against the real files ───────────────────────────────────

def _skip_unless(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not present on this machine")
    return str(path)


def test_dlc_single_animal_model_reads_as_single():
    p = _skip_unless(REPO / "models" / "Multimaze_mobilenet_v2_1.0")
    info = ModelInfo.read(p)
    assert info.backend == "dlc"
    assert info.family == "single"
    assert info.multi_animal is False
    assert info.body_parts == ("Head", "Center", "Tailbase")
    assert info.backbone == "mobilenet_v2_1.0"
    assert info.channels == RGB


def test_dlc_multi_animal_model_is_detected_and_warned_about():
    """The defect this closes: a multi-animal DLC model loaded silently as
    single-animal, because we read only ``all_joints_names``."""
    p = _skip_unless(REPO / "models" / "dlc" / "SuperAnimal-TopViewMouse")
    info = ModelInfo.read(p)
    assert info.multi_animal is True, "dataset_type/PAF flag mark this multi-animal"
    assert info.family == "multi-animal"
    assert len(info.body_parts) == 27
    assert any("single_animal=False" in w for w in info.warnings)


@pytest.mark.parametrize("field,flag", [
    ("dataset_type", "multi-animal-imgaug"),
    ("partaffinityfield_predict", True),
])
def test_either_dlc_multi_animal_marker_is_enough(tmp_path, field, flag):
    import yaml
    cfg = {"all_joints_names": ["a"], field: flag}
    (tmp_path / "pose_cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert ModelInfo.read(str(tmp_path)).multi_animal is True


def test_dlc_global_scale_is_not_reported_as_an_inference_scale(tmp_path):
    """``global_scale`` is a training field. DLCLive never reads it, so
    reporting it as the native inference scale would be a lie the dialog then
    shows to the operator."""
    import yaml
    cfg = {"all_joints_names": ["a"], "global_scale": 0.8}
    (tmp_path / "pose_cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert ModelInfo.read(str(tmp_path)).native_scale is None


# ── shape of the answer ──────────────────────────────────────────────────

def test_summary_is_one_line_and_mentions_the_essentials(tmp_path):
    info = _sleap(tmp_path, {"multi_class_topdown": {
                      "class_vectors": {"classes": ["x", "y"]}}},
                  preprocessing={"scale": 0.5, "ensure_grayscale": True})
    s = info.summary()
    assert "\n" not in s
    assert "multi_class_topdown" in s and "2 identities" in s and GRAYSCALE in s


def test_config_file_may_be_given_directly(tmp_path):
    cfg = {"model_config": {"head_configs": {"bottomup": {}}}}
    p = tmp_path / "training_config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert ModelInfo.read(str(p)).family == "bottomup"
