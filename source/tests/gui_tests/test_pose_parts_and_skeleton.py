"""Body parts and the skeleton, for BOTH toolkits.

The rig read DeepLabCut's layout and only DeepLabCut's. A sleap-nn model
directory holds ``training_config.yaml``; the dialog scanned for
``pose_cfg.yaml`` / ``pose.yaml`` / ``config.yaml`` and the keys ``bodyparts``
/ ``all_joints_names`` / ``multianimalbodyparts`` - none of which a SLEAP model
has. So every SLEAP model silently loaded the hardcoded fallback ``["center"]``,
and that one wrong name was pushed down into the tracker, where it relabelled a
six-keypoint network as a single part pointing at keypoint 0 (the snout).

The skeleton was worse: nothing read one at all, for either toolkit. The
overlay joined keypoints in LIST ORDER, which for this project's model means
snout-to-head-to-ear-to-ear - an anatomy the model never declared.

sleap-nn's own export guide is explicit that the runtime "reads the full
training skeleton from training_config.yaml"; the export metadata beside the
weights is not a substitute, because ``edge_inds`` there is empty for a
single-instance model.
"""
from __future__ import annotations

import os

import pytest

from source.video.tracking.model_config import ModelInfo

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SLEAP = os.path.join(REPO, "models", "sleap")
DLC = os.path.join(REPO, "models", "Multimaze_mobilenet_v2_1.0")
BIG_DLC = os.path.join(REPO, "models", "dlc", "SuperAnimal-TopViewMouse")

needs_sleap = pytest.mark.skipif(not os.path.isdir(SLEAP),
                                 reason="no SLEAP model in the repo")
needs_dlc = pytest.mark.skipif(not os.path.isdir(DLC),
                               reason="no DLC model in the repo")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6 import QtWidgets
    return (QtWidgets.QApplication.instance()
            or QtWidgets.QApplication(["test"]))


@pytest.fixture
def panel(qt_app):
    """A panel that is really gone by the end of the test.

    ``deleteLater`` only schedules the destruction; the suite asserts no
    top-level widget outlives its test, so the loop has to be drained here.
    """
    from source.gui.widgets.tracking_panel import TrackingSettingsPanel
    made = []

    def build(model_path, sleap):
        p = TrackingSettingsPanel(box_ids=[1])
        (p.mode_sleap if sleap else p.mode_dlc).setChecked(True)
        p.model_path.setText(model_path)
        p._load_body_parts_from_model()
        made.append(p)
        return p

    yield build
    # Exactly what conftest prescribes, and deliberately NOT ``close()``:
    # a close runs the widget's own closeEvent, which for the tracking
    # dialogs is arbitrary application code.
    from PySide6 import QtCore
    for p in made:
        p.deleteLater()
    qt_app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


class TestTheReaderKnowsBothToolkits:
    @needs_sleap
    def test_a_sleap_model_reports_its_nodes(self):
        info = ModelInfo.read(SLEAP)
        assert info.backend == "sleap"
        assert info.body_parts == ("Snout", "Head", "Left_Ear", "Right_Ear",
                                   "center", "Tail_Base")

    @needs_sleap
    def test_a_sleap_model_reports_its_skeleton(self):
        """The five edges are in training_config.yaml, and were discarded."""
        assert ModelInfo.read(SLEAP).skeleton == (
            ("Snout", "Head"), ("Left_Ear", "Head"), ("Right_Ear", "Head"),
            ("Head", "center"), ("center", "Tail_Base"))

    @needs_dlc
    def test_a_dlc_model_still_reports_its_parts(self):
        assert ModelInfo.read(DLC).body_parts == ("Head", "Center", "Tailbase")

    @needs_dlc
    def test_an_exported_dlc_folder_declares_no_skeleton(self):
        """DeepLabCut keeps the skeleton in the PROJECT config.yaml, which an
        exported model folder does not ship. Empty is the honest answer."""
        assert ModelInfo.read(DLC).skeleton == ()

    @pytest.mark.skipif(not os.path.isdir(BIG_DLC), reason="no SuperAnimal")
    def test_the_paf_graph_is_never_mistaken_for_a_skeleton(self):
        """``partaffinityfield_graph`` is the COMPLETE graph here - 351 edges
        for 27 parts. Drawing it would connect every part to every other."""
        info = ModelInfo.read(BIG_DLC)
        assert len(info.body_parts) == 27
        assert info.skeleton == ()


class TestEdgesAreNamesNotIndices:
    def test_index_pairs_are_resolved_against_the_nodes(self, tmp_path):
        (tmp_path / "training_config.yaml").write_text(
            "data_config:\n"
            "  skeletons:\n"
            "  - nodes:\n"
            "    - name: a\n"
            "    - name: b\n"
            "    edges:\n"
            "    - [0, 1]\n", encoding="utf-8")
        assert ModelInfo.read(str(tmp_path)).skeleton == (("a", "b"),)

    def test_inline_names_work_too(self, tmp_path):
        (tmp_path / "training_config.yaml").write_text(
            "data_config:\n"
            "  skeletons:\n"
            "  - nodes: [{name: a}, {name: b}]\n"
            "    edges:\n"
            "    - {source: a, destination: b}\n", encoding="utf-8")
        assert ModelInfo.read(str(tmp_path)).skeleton == (("a", "b"),)

    def test_an_edge_naming_an_unknown_part_is_dropped_and_said(self, tmp_path):
        """Silently keeping it produces a line that never draws."""
        (tmp_path / "training_config.yaml").write_text(
            "data_config:\n"
            "  skeletons:\n"
            "  - nodes: [{name: a}, {name: b}]\n"
            "    edges:\n"
            "    - {source: {name: a}, destination: {name: ghost}}\n",
            encoding="utf-8")
        info = ModelInfo.read(str(tmp_path))
        assert info.skeleton == ()
        assert any("ghost" in w for w in info.warnings)


class TestTheDialogUsesThatReader:
    @needs_sleap
    def test_a_sleap_model_fills_the_pickers(self, panel):
        panel = panel(SLEAP, True)
        assert panel.body_parts_list == ["Snout", "Head", "Left_Ear",
                                         "Right_Ear", "center", "Tail_Base"]
        assert len(panel.skeleton_edges) == 5
        assert panel.skeleton_table.rowCount() == 5
        combo = panel.zone_body_part_combo
        listed = [combo.itemText(i) for i in range(combo.count())]
        assert "Tail_Base" in listed and "center" in listed
        assert set(panel.annotate_checkboxes) == set(panel.body_parts_list)

    @needs_dlc
    def test_a_dlc_model_gets_an_empty_editable_skeleton(self, panel):
        panel = panel(DLC, False)
        assert panel.body_parts_list == ["Head", "Center", "Tailbase"]
        assert panel.skeleton_edges == []
        panel._add_skeleton_edge()
        assert panel.skeleton_edges == [("Head", "Center")]
        assert panel.skeleton_table.rowCount() == 1
        panel._drop_skeleton_edge(0)
        assert panel.skeleton_edges == []

    @needs_sleap
    def test_reset_goes_back_to_what_the_model_says(self, panel):
        panel = panel(SLEAP, True)
        panel.skeleton_edges = []
        panel._reset_skeleton()
        assert len(panel.skeleton_edges) == 5


class TestTheModelOutranksTheDialog:
    def test_the_reported_names_win_over_a_stale_list(self):
        """``["center"]`` against a six-node model is how a snout came to be
        called a centroid."""
        from source.video.framebus.pose_sink import PoseSink
        assert PoseSink._reconcile_parts(
            ["center"], ["Snout", "Head", "Left_Ear", "Right_Ear",
                         "center", "Tail_Base"]) == [
            "Snout", "Head", "Left_Ear", "Right_Ear", "center", "Tail_Base"]

    def test_the_dialog_list_is_used_when_the_model_reports_nothing(self):
        """The blob tracker cannot introspect itself."""
        from source.video.framebus.pose_sink import PoseSink
        assert PoseSink._reconcile_parts(["centroid"], []) == ["centroid"]

    def test_neither_is_not_an_error(self):
        from source.video.framebus.pose_sink import PoseSink
        assert PoseSink._reconcile_parts([], []) == []


class TestItSurvivesTheRoundTrip:
    def test_the_skeleton_reaches_the_config_and_back(self):
        from source.video.framebus.types import TrackingConfig
        cfg = TrackingConfig(setup_id=1, keypoint_names=("a", "b"),
                             skeleton=(("a", "b"),))
        assert TrackingConfig.from_json(cfg.to_json()).skeleton == (("a", "b"),)

    def test_a_config_without_one_is_not_an_error(self):
        from source.video.framebus.types import TrackingConfig
        assert TrackingConfig.from_json({"box_id": 1}).skeleton == ()
