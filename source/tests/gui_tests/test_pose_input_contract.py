"""The input contract has to survive the dialog, the file, and the reload.

A control that saves but does not restore is worse than one that does not
exist: the operator sets it, sees it take effect, and finds it reverted next
session with nothing said. So this drives the real panel rather than asserting
on a settings dict in isolation.
"""
from pathlib import Path

import pytest

from source.tests.qt_dispose import WidgetBin
from source.video.framebus.types import TrackingConfig

_BIN = WidgetBin()

#: A real crop-trained SLEAP export, not a fixture. What the dialog shows about
#: a model has to be read from a model, a hand-written stand-in would agree
#: with whatever the test author believed the config said.
_SLEAP_MODEL = Path(__file__).resolve().parents[3] / "models" / "sleap"

pytestmark = pytest.mark.skipif(
    not (_SLEAP_MODEL / "training_config.yaml").is_file(),
    reason="needs the SLEAP model in models/sleap")


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()


@pytest.fixture
def qapp():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp):
    from source.gui.widgets.tracking_panel import TrackingSettingsPanel
    p = _BIN.add(TrackingSettingsPanel(box_ids=[1]))
    p.build_event_trigger_widget().setParent(p)
    return p


_SAVED = {"pose_input_mode": "crop_track", "pose_input_w": 224,
          "pose_input_h": 176, "pose_crop_conf_min": 0.35,
          "pose_crop_good_min": 4, "pose_crop_reacquire": False}


class TestTheDialogRoundTrip:
    def test_every_field_saves(self, panel, qapp):
        panel.pose_input_mode_combo.setCurrentText("crop-track")
        panel.pose_input_w_spin.setValue(224)
        panel.pose_input_h_spin.setValue(176)
        panel.pose_crop_conf_spin.setValue(0.35)
        panel.pose_crop_good_spin.setValue(4)
        panel.pose_crop_reacquire_cb.setChecked(False)
        qapp.processEvents()
        got = panel.get_settings()
        for key, want in _SAVED.items():
            assert got[key] == want, key

    def test_every_field_restores(self, panel, qapp):
        from source.gui.widgets.tracking_panel import UnifiedTrackingDialog
        # The CLASS as self: `_set_combo_text` is a staticmethod, so it
        # resolves. Passing None works only while the settings dict happens to
        # miss every combo, which is a test that stops testing the moment the
        # dict grows.
        UnifiedTrackingDialog._apply_sleap_params(
            UnifiedTrackingDialog, panel, dict(_SAVED, sleap_runtime="onnx"))
        qapp.processEvents()
        got = panel.get_settings()
        for key, want in _SAVED.items():
            assert got[key] == want, key

    def test_the_confidence_field_is_not_an_integer(self, panel, qapp):
        """Without decimals a typed 0.35 stores as 0, which reads as 'trust
        every keypoint', the opposite of what was asked for."""
        panel.pose_crop_conf_spin.setValue(0.35)
        assert panel.get_settings()["pose_crop_conf_min"] == pytest.approx(0.35)

    def test_a_project_that_never_touched_it_asks_the_model(self, panel):
        """The default is ``auto``, not ``letterbox``.

        ``letterbox`` was the default and it is wrong for a crop-trained model:
        it scales the arena onto the training window, shrinking the animal past
        anything the network saw. Measured on the SLEAP model's own shipped
        self-check, that costs 9-11 px against 1.7-2.1 px for the window, with
        most keypoints landing in the padding and being dropped. ``auto`` reads
        the answer off the model config instead of assuming one.
        """
        assert panel.get_settings()["pose_input_mode"] == "auto"


class TestWhatTheDialogShows:
    def test_the_window_is_only_editable_when_a_window_is_used(self, panel, qapp):
        panel.pose_input_mode_combo.setCurrentText("letterbox")
        qapp.processEvents()
        assert not panel.pose_input_w_spin.isEnabled()
        panel.pose_input_mode_combo.setCurrentText("crop-track")
        qapp.processEvents()
        assert panel.pose_input_w_spin.isEnabled()

    def test_each_backends_steering_belongs_to_it_alone(self, panel, qapp):
        """This replaces a test asserting our follow-confidence was editable
        under BOTH following modes. It is not a DeepLabCut setting.

        The two mechanisms are unrelated: ours cuts a fixed window at the
        model's training size and steers it by the mean of the confident
        keypoints; DeepLabCut-Live's crops to the bounding box of whatever it
        last found, expanded by a margin, and lives only in its PyTorch runner.
        A follow-confidence set under dlc-dynamic was read by nothing.
        """
        # isHidden rather than isVisible: every child of a panel that was never
        # shown reports isVisible() False regardless of its own flag, so the
        # assertion would pass without testing anything.
        panel.mode_sleap.setChecked(True)
        panel.pose_input_mode_combo.setCurrentText("crop-track")
        qapp.processEvents()
        panel._sync_pose_input_row()
        assert panel.pose_crop_conf_spin.isEnabled()
        assert panel.dlc_dynamic_widget.isHidden()

        panel.mode_dlc.setChecked(True)
        if panel.pose_input_mode_combo.findText("dlc-dynamic") < 0:
            panel.pose_input_mode_combo.addItem("dlc-dynamic")
        panel.pose_input_mode_combo.setCurrentText("dlc-dynamic")
        qapp.processEvents()
        panel._sync_pose_input_row()
        assert not panel.dlc_dynamic_widget.isHidden()
        assert panel.pose_crop_widget.isHidden()

    def test_an_unset_window_is_filled_from_the_model(self, panel, qapp):
        """The model states its training window; the operator should not have
        to retype it.

        This replaces a test that asserted the dialog says "will fall back to
        letterbox" for an unset window. It did say that, while the size sat in
        the model config unread, which is precisely how a crop-trained model
        ended up letterboxed.
        """
        panel.mode_sleap.setChecked(True)
        panel.model_path.setText(str(_SLEAP_MODEL))
        panel._on_model_path_changed(str(_SLEAP_MODEL))
        qapp.processEvents()
        assert (int(panel.pose_input_w_spin.value()),
                int(panel.pose_input_h_spin.value())) == (224, 176)

    def test_a_window_the_stride_cannot_divide_says_it_will_fall_back(
            self, panel, qapp):
        """A side the backbone stride does not divide breaks the encoder shape
        arithmetic, so the pipeline letterboxes instead, said here, at typing
        time, rather than discovered at Init."""
        panel.model_path.setText(str(_SLEAP_MODEL))
        panel.mode_sleap.setChecked(True)
        panel.pose_input_mode_combo.setCurrentText("crop-track")
        panel.pose_input_w_spin.setValue(300)
        panel.pose_input_h_spin.setValue(250)
        panel._validate_pose_window()
        assert "stride" in panel.pose_input_note.text()

    def test_a_valid_window_says_so(self, panel, qapp):
        panel.model_path.setText(str(_SLEAP_MODEL))
        panel.mode_sleap.setChecked(True)
        panel.pose_input_mode_combo.setCurrentText("crop-track")
        panel.pose_input_w_spin.setValue(224)
        panel.pose_input_h_spin.setValue(176)
        panel._validate_pose_window()
        assert "224x176" in panel.pose_input_note.text()

    def test_auto_resolves_a_crop_trained_model_to_the_window(self, panel, qapp):
        """The whole point of the default: this model was trained on
        native-scale 224x176 crops, so ``auto`` must cut a window, not scale
        the arena onto it."""
        panel.model_path.setText(str(_SLEAP_MODEL))
        panel.mode_sleap.setChecked(True)
        panel.pose_input_mode_combo.setCurrentText("auto")
        panel.pose_input_w_spin.setValue(0)
        panel.pose_input_h_spin.setValue(0)
        plan = panel._resolved_input_plan()
        assert plan is not None
        assert plan.mode == "crop_track"
        assert plan.size == (224, 176)

    def test_changing_the_window_asks_for_a_re_init(self, panel, qapp):
        """It changes what the network is fed, so it cannot apply live."""
        panel._dlc_initialized = True
        panel.pose_input_w_spin.setValue(224)
        panel.pose_input_w_spin.valueChanged.emit(224)
        qapp.processEvents()
        assert panel._dlc_initialized is False


class TestTheStoredContract:
    def test_config_defaults_to_asking_the_model(self):
        """See test_a_project_that_never_touched_it_asks_the_model."""
        assert TrackingConfig(setup_id=1).pose_input_mode == "auto"

    def test_a_deliberately_zeroed_window_is_not_read_as_absent(self):
        """`d.get(k, 224) or 224` would put 224 back; 0 means 'ask the model'."""
        c = TrackingConfig.from_json({"setup_id": 1, "pose_input_w": 0})
        assert c.pose_input_w == 0

    def test_the_json_round_trip_keeps_every_field(self):
        c = TrackingConfig.from_json({"setup_id": 1, **_SAVED})
        back = TrackingConfig.from_json(c.to_json())
        assert back.pose_input_mode == "crop_track"
        assert (back.pose_input_w, back.pose_input_h) == (224, 176)
        assert back.pose_crop_conf_min == pytest.approx(0.35)
        assert back.pose_crop_good_min == 4
        assert back.pose_crop_reacquire is False

    def test_dlc_dynamic_reaches_the_dlc_constructor_options(self):
        c = TrackingConfig.from_json({"setup_id": 1, "tracker_type": "dlc",
                                      "pose_input_mode": "dlc_dynamic",
                                      "dlc_dynamic_margin": 20})
        opts = c.derived_dlc_opts()
        assert opts["dynamic_crop"] is True
        assert opts["dynamic_margin"] == 20

    def test_sleap_never_receives_dlc_options(self):
        c = TrackingConfig.from_json({"setup_id": 1, "tracker_type": "sleap",
                                      "pose_input_mode": "crop_track"})
        assert c.derived_dlc_opts() == {}
