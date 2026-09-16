"""What a model gets fed, decided from the model rather than assumed.

The rule these pin cost real data before it existed. A SLEAP model trained on
native-scale 224x176 crops was letterboxed from a 513x315 arena, shrinking the
animal to 44% of the size the network was trained on; measured against that
model's own shipped self-check, the same weights give 0.3 px error on their
native crop and 9-11 px through the letterbox, with most keypoints landing in
the padding and being dropped.

The end-to-end proof lives in
``tools/offline_analysis/tests/test_crop_trained_input.py``, which needs the
model. These are the decision rules alone, so they run anywhere.
"""
from pathlib import Path

import pytest

from source.video.tracking.input_policy import CROP_MARGIN, crop_trained, plan
from source.video.tracking.model_config import ModelInfo

_ROOT = Path(__file__).resolve().parents[3]
_SLEAP = _ROOT / "models" / "sleap"
_DLC = _ROOT / "models" / "dlc_6point_v2"

needs_sleap = pytest.mark.skipif(
    not (_SLEAP / "training_config.yaml").is_file(),
    reason="needs models/sleap")
needs_dlc = pytest.mark.skipif(
    not (_DLC / "export_metadata.json").is_file(),
    reason="needs models/dlc_6point_v2")


class TestReadingCropTraining:
    """``max_width``/``max_height`` alone cannot tell the two apart."""

    def test_a_declared_input_far_smaller_than_the_frame_is_a_crop(self):
        info = ModelInfo(input_w=224, input_h=176, native_scale=1.0)
        assert crop_trained(info, 513, 315)

    def test_a_frame_smaller_than_the_window_is_still_native_scale(self):
        """Padding a small frame preserves the pixel size exactly as cropping
        a large one does. Requiring the frame to be BIGGER sent a 16-box rig's
        160x120 ROI to a letterbox that upscaled it 1.4x."""
        info = ModelInfo(input_w=224, input_h=176, native_scale=1.0)
        assert crop_trained(info, 160, 120)

    def test_a_downscaled_model_is_not_read_as_native_scale(self):
        """``scale < 1`` means it was trained on shrunken frames, so shrinking
        is exactly right for it."""
        info = ModelInfo(input_w=224, input_h=176, native_scale=0.5)
        assert not crop_trained(info, 513, 315)

    def test_an_unstated_scale_is_not_read_as_native(self):
        """DeepLabCut states no scale, and its declared input comes from an
        export's baked-in graph shape rather than a training crop. Reading
        silence as 1.0 made every DLC export look crop-trained."""
        info = ModelInfo(input_w=360, input_h=202, native_scale=None)
        assert not crop_trained(info, 513, 315)

    def test_the_margin_decides_following_not_cutting(self):
        """``CROP_MARGIN`` no longer gates crop-track; it only distinguishes
        a window that has somewhere to move from one that covers the frame."""
        model = str(_SLEAP)
        if not (_SLEAP / "training_config.yaml").is_file():
            pytest.skip("needs models/sleap")
        roomy = plan("sleap", model, 513, 315, asked="auto")
        snug = plan("sleap", model, 160, 120, asked="auto")
        assert roomy.mode == snug.mode == "crop_track"
        assert "follows" in roomy.reason
        assert "padded" in snug.reason
        assert CROP_MARGIN > 1.0


class TestRefusals:
    """A mode the model cannot use is refused OUT LOUD, never downgraded
    silently: an operator who chose a window and got a letterbox would be
    measuring something other than what they asked for."""

    def test_a_top_down_model_may_not_be_crop_tracked(self, tmp_path):
        got = plan("sleap", str(tmp_path), 513, 315, asked="crop_track")
        assert got.mode == "letterbox"

    @needs_dlc
    def test_full_frame_is_refused_for_a_fixed_shape_export(self):
        """It does not run slower, it throws on every frame, and the batch
        path turns that into an empty pose, so the run looks healthy while
        recording nothing."""
        got = plan("dlc", str(_DLC), 640, 480, engine="export", asked="full")
        assert got.mode == "letterbox"
        assert got.size == (360, 202)
        assert "fixed at export" in got.reason

    @needs_sleap
    def test_a_window_the_stride_cannot_divide_is_refused(self):
        got = plan("sleap", str(_SLEAP), 513, 315, asked="crop_track",
                   asked_size=(300, 250))
        assert got.mode == "letterbox"
        assert "stride" in got.reason

    @needs_sleap
    def test_the_models_own_size_is_never_stride_checked(self):
        """A size the model declared is by definition one it accepts. DLC's
        export is built at a height its own backbone stride does not divide,
        because the runner pads internally."""
        got = plan("sleap", str(_SLEAP), 513, 315, asked="crop_track")
        assert got.mode == "crop_track"


class TestTheDefault:
    @needs_sleap
    def test_auto_cuts_a_window_for_a_crop_trained_model(self):
        got = plan("sleap", str(_SLEAP), 513, 315, asked="auto")
        assert got.mode == "crop_track"
        assert got.size == (224, 176)
        assert got.from_model

    @needs_sleap
    def test_a_frame_smaller_than_the_window_uses_the_whole_frame(self):
        """A 16-box rig gives each box a 160x120 ROI against a 224x176 window.

        There is nothing to crop and nowhere to follow, so the whole frame is
        used at native scale. Padding it out to the model's window instead
        would run inference over a large black border, twice the pixels for
        the same picture. The only padding kept is what the backbone stride
        requires: a UNet that halves four times cannot take a height of 120.
        """
        got = plan("sleap", str(_SLEAP), 160, 120, asked="auto")
        assert got.mode == "crop_track"
        assert got.size == (160, 128)          # 120 -> 128, the stride of 16
        assert got.size[0] == 160, "the width was already stride-safe"
        assert "smaller than the model's window" in got.reason

    @needs_sleap
    def test_a_window_sized_frame_is_padded_not_resized(self):
        """240x190 onto a 224x176 window is only a 0.93 resize, which looks
        harmless and is not: the model works at native scale, so the honest
        answer is to cut the 16 px and pad the rest, at scale exactly 1."""
        got = plan("sleap", str(_SLEAP), 240, 190, asked="auto")
        assert got.mode == "crop_track"
        assert got.size == (224, 176)

    @needs_dlc
    def test_auto_hands_the_whole_frame_to_a_convolutional_runner(self):
        """DLC's PyTorch runner pads to the backbone stride itself and its
        config asks for no resize at inference, so nothing needs scaling,
        and it measured the most accurate of the three DLC paths."""
        got = plan("dlc", str(_DLC), 513, 315, engine="pytorch", asked="auto")
        assert got.mode == "full"
        assert got.size == (513, 315)

    @needs_dlc
    def test_auto_letterboxes_a_fixed_shape_export(self):
        got = plan("dlc", str(_DLC), 513, 315, engine="export", asked="auto")
        assert got.mode == "letterbox"
        assert got.size == (360, 202)

    def test_an_unrecognised_engine_letterboxes_rather_than_guessing(self):
        """``full`` is only chosen for an engine positively known to take any
        size. One nobody recognised has unknown input rules."""
        got = plan("dlc", "", 513, 315, engine="dlclive", asked="auto")
        assert got.mode == "letterbox"


class TestExplicitChoicesAreHonoured:
    @needs_sleap
    def test_letterbox_stays_letterbox_on_a_crop_model(self):
        got = plan("sleap", str(_SLEAP), 513, 315, asked="letterbox")
        assert got.mode == "letterbox"

    @needs_sleap
    def test_a_typed_window_overrides_the_model(self):
        got = plan("sleap", str(_SLEAP), 513, 315, asked="crop_track",
                   asked_size=(320, 256))
        assert got.mode == "crop_track"
        assert got.size == (320, 256)
        assert not got.from_model

    @needs_dlc
    def test_dlc_dynamic_reaches_the_pytorch_runner(self):
        """DeepLabCut-Live cuts and restores its own window inside the runner,
        so the mode name only has to survive to the constructor."""
        got = plan("dlc", str(_DLC), 513, 315, engine="pytorch",
                   asked="dlc_dynamic")
        assert got.mode == "dlc_dynamic"

    @needs_dlc
    @pytest.mark.parametrize("engine", ["export", "tensorflow", ""])
    def test_dlc_dynamic_is_refused_where_it_would_do_nothing(self, engine):
        """It exists in exactly one place: dlclive's PyTorch runner. The
        TensorFlow runners never receive the keyword, and an exported graph is
        not driven through dlclive at all, so anywhere else it is a setting
        that would be silently ignored."""
        got = plan("dlc", str(_DLC), 513, 315, engine=engine,
                   asked="dlc_dynamic")
        assert got.mode == "letterbox"
        assert "PyTorch runner" in got.reason

    @needs_sleap
    def test_sleap_is_never_given_dlcs_cropping(self):
        got = plan("sleap", str(_SLEAP), 513, 315, asked="dlc_dynamic")
        assert got.mode == "letterbox"
        assert "SLEAP" in got.reason
