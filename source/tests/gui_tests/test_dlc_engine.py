"""Which engine runs a DeepLabCut model, and letting the PyTorch one run at all.

Three separate things stopped ``model_type="pytorch"`` from ever working, and
each hid the next:

1. the tracker tried the ONNX export first and returned, so the setting was
   dead for any folder holding one;
2. the default was ``base``, which is DeepLabCut-Live's **TensorFlow** runner,
   so on a rig without TensorFlow a PyTorch model failed with
   ``ModuleNotFoundError: tensorflow``, an error naming the wrong thing;
3. the PyTorch runner takes the ``.pt`` file, not the folder, and then
   ``torch.load(weights_only=True)`` refuses the checkpoint because DeepLabCut
   pickles its own enums and a ``pathlib.WindowsPath`` into the config.
"""
from pathlib import Path

import pytest

from source.video.tracking import dlc_engine

_ROOT = Path(__file__).resolve().parents[3]
_DLC = _ROOT / "models" / "dlc_6point_v2"
_TF = _ROOT / "models" / "Multimaze_mobilenet_v2_1.0"

needs_dlc = pytest.mark.skipif(not (_DLC / "export_metadata.json").is_file(),
                               reason="needs models/dlc_6point_v2")
needs_tf_model = pytest.mark.skipif(not (_TF / "pose_cfg.yaml").is_file(),
                                    reason="needs a DLC TensorFlow model")


class TestWhatIsOnDisk:
    @needs_dlc
    def test_a_folder_with_both_offers_both(self):
        assert dlc_engine.available(str(_DLC)) == ["export", "pytorch"]

    @needs_dlc
    def test_the_snapshot_is_found_inside_the_folder(self):
        """dlclive torch.loads this path, so the folder is not an answer."""
        got = dlc_engine.snapshot_path(str(_DLC))
        assert got is not None and got.endswith(".pt")

    @needs_tf_model
    def test_a_tensorflow_folder_is_recognised_as_one(self):
        assert dlc_engine.has_tensorflow_snapshot(str(_TF))
        assert dlc_engine.available(str(_TF)) == ["tensorflow"]

    def test_an_empty_folder_offers_nothing(self, tmp_path):
        assert dlc_engine.available(str(tmp_path)) == []


class TestResolving:
    @needs_dlc
    def test_pytorch_is_honoured_and_gets_the_snapshot_file(self):
        engine, path, note = dlc_engine.resolve(str(_DLC), "pytorch")
        assert engine == "pytorch"
        assert path.endswith(".pt")
        assert note == ""

    @needs_dlc
    def test_auto_prefers_the_export(self):
        """Roughly 3x faster on the same weights; the accuracy trade is the
        operator's to make, not ours to make silently."""
        engine, _path, note = dlc_engine.resolve(str(_DLC), "auto")
        assert engine == "export"
        assert note == ""

    @needs_dlc
    def test_a_downgrade_is_reported_rather_than_applied_quietly(self):
        """``base`` needs TensorFlow. Where it is absent the run still happens,
        on a different engine, and says so."""
        engine, _path, note = dlc_engine.resolve(str(_DLC), "base")
        if engine == "tensorflow":
            pytest.skip("TensorFlow is installed here, so nothing is downgraded")
        assert engine in ("export", "pytorch")
        assert "not available" in note

    def test_an_unrecognised_folder_is_handed_to_dlclive(self, tmp_path):
        """Our checks cover the layouts this repo knows; DeepLabCut-Live has
        its own detection and the model-zoo conventions. Refusing here would
        turn a folder it could open into a failure blaming our checks."""
        engine, path, _note = dlc_engine.resolve(str(tmp_path), "auto")
        assert engine == dlc_engine.PASSTHROUGH
        assert path == str(tmp_path)

    def test_a_nonsense_engine_name_is_refused(self):
        engine, _path, note = dlc_engine.resolve("anywhere", "banana")
        assert engine == "none"
        assert "banana" in note


class TestTheVocabularyTranslation:
    """Ours is wider than dlclive's: it has no name for this repo's export, and
    its ``tensorrt`` is TensorFlow-TensorRT, not our ONNX/TRT graph. Conflating
    the two sent a "tensorrt" choice to a runner needing TensorFlow."""

    def test_pytorch_maps_straight_through(self):
        assert dlc_engine.dlclive_model_type("pytorch", "pytorch") == "pytorch"

    def test_a_tensorflow_variant_is_preserved(self):
        assert dlc_engine.dlclive_model_type("tensorflow", "lite") == "lite"

    def test_anything_else_falls_back_to_base(self):
        assert dlc_engine.dlclive_model_type("tensorflow", "onnx") == "base"


@needs_dlc
class TestOpeningTheCheckpoint:
    def test_torch_load_can_open_a_deeplabcut_snapshot(self):
        """The allow-list is driven by the error, not a fixed list: torch names
        one refused global per attempt, so each round stands that one in. A
        hard-coded list would be another place that has to learn about the next
        DeepLabCut release.
        """
        torch = pytest.importorskip("torch")
        snapshot = dlc_engine.snapshot_path(str(_DLC))
        dlc_engine.allow_checkpoint_globals(snapshot)
        raw = torch.load(snapshot, map_location="cpu", weights_only=True)
        assert "config" in raw and "pose" in raw
        assert raw["config"]["metadata"]["bodyparts"]

    def test_the_stand_ins_still_compare_as_strings(self):
        """``cfg["method"] == "td"`` has to keep working, so the stand-in is a
        ``str`` subclass rather than an opaque object."""
        torch = pytest.importorskip("torch")
        snapshot = dlc_engine.snapshot_path(str(_DLC))
        dlc_engine.allow_checkpoint_globals(snapshot)
        cfg = torch.load(snapshot, map_location="cpu",
                         weights_only=True)["config"]
        assert cfg["method"] in ("bu", "td")
        assert isinstance(cfg["net_type"], str)
