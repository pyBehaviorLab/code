"""The Pipeline owns the video path; the GUI asks it rather than reaching past it.

`base.py` exposes `self.video_manager = self.pipeline.video_manager` as a
convenience alias, and code reached through it to mutate state the Pipeline
owns. Read-only queries through the alias are tolerable; these pin the ones
that were not.
"""
import ast
from pathlib import Path

import pytest

from source.video.framebus.controller import Pipeline

SOURCE = Path(__file__).resolve().parents[2]


class _FakeThread:
    def __init__(self, grayscale=False):
        self.grayscale = grayscale


def _pipeline_with(box_camera_map=None, cameras=None):
    """A Pipeline shell, no cameras opened, no sinks started."""
    p = Pipeline.__new__(Pipeline)
    p.video_manager = type("VM", (), {})()
    p.video_manager.box_camera_map = dict(box_camera_map or {})
    p.video_manager.cameras = dict(cameras or {})
    p._buses = {}
    p._box_unsubs = {}
    return p


# ---- the API that replaces the reach-throughs ----------------------------

def test_camera_id_for_box():
    p = _pipeline_with({1: 0, 2: "cam-b"})
    assert p.camera_id_for_box(1) == 0
    assert p.camera_id_for_box(2) == "cam-b"
    assert p.camera_id_for_box(99) is None


def test_force_color_reports_and_clears_grayscale():
    """Pose needs 3-channel BGR; a grayscale stream makes it return empty
    results with no error at all."""
    thread = _FakeThread(grayscale=True)
    p = _pipeline_with({1: 0}, {0: thread})
    assert p.force_color_for_box(1) is True
    assert thread.grayscale is False
    # Idempotent: a second call reports it was already colour.
    assert p.force_color_for_box(1) is False


@pytest.mark.parametrize("bcm,cams", [({}, {}), ({1: 0}, {})])
def test_force_color_is_safe_when_there_is_no_camera(bcm, cams):
    assert _pipeline_with(bcm, cams).force_color_for_box(1) is False


def test_release_all_cameras_keeps_the_pipeline_usable():
    """Distinct from shutdown(), which also stops the sink workers and the
    tick, after which the Pipeline cannot be reused. The remove-all-boxes
    sweep needs the cameras gone and the Pipeline alive."""
    cleaned = []
    p = _pipeline_with({1: 0}, {0: _FakeThread()})
    p.video_manager.cleanup = lambda: cleaned.append(True)
    p._buses = {0: object()}
    p._box_unsubs = {1: []}

    p.release_all_cameras()

    assert cleaned == [True]
    assert p._buses == {}
    assert p._box_unsubs == {}
    # The sinks were never touched, no attribute for them even exists on this
    # shell, so any attempt to stop them would have raised.


# ---- no new reach-throughs that MUTATE ----------------------------------

MUTATING = {"cleanup", "start_camera", "stop_camera", "set_target_fps",
            "set_target_resolution", "set_frame_strategy"}


def _video_manager_mutations(path):
    """(lineno, attr) for `<x>.video_manager.<mutating call>` outside the
    pipeline itself."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATING):
            continue
        inner = node.func.value
        if (isinstance(inner, ast.Attribute)
                and inner.attr == "video_manager"):
            hits.append((node.lineno, node.func.attr))
    return hits


def test_gui_does_not_mutate_the_video_manager_directly():
    """Reading through the alias is tolerable; driving camera lifecycle
    through it is not, it tears state down behind the Pipeline's back and
    leaves its buses and sink subscriptions pointing at dead threads."""
    offenders = []
    for path in (SOURCE / "gui").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for lineno, attr in _video_manager_mutations(path):
            offenders.append(f"{path.relative_to(SOURCE)}:{lineno} .{attr}()")
    assert not offenders, (
        "call the equivalent Pipeline method instead:\n  " + "\n  ".join(offenders))


def test_pose_subsystem_goes_through_the_pipeline():
    src = (SOURCE / "gui" / "pose_subsystem.py").read_text(encoding="utf-8")
    assert "pipeline.force_color_for_box" in src
    assert "cam_thread.grayscale = False" not in src, (
        "pose still mutates live capture state from outside the pipeline")


def test_recorder_geometry_single_cropper():
    """Exactly one thing crops a recorded frame.

    Either the FrameBus already sliced this box out of a shared camera
    (``segment_size``), or the recorder crops it itself (``roi``), never
    both, which would crop twice, and never neither on a box that needs it.
    The recorder's roi cannot be deleted "because the bus already crops": on
    the operant no-segment path the bus does not.
    """
    import inspect

    from source.gui.base import MainWindowBase
    src = inspect.getsource(MainWindowBase._video_recorder_geometry)
    assert "assert not (segment_size and roi)" in src, (
        "the one-cropper invariant is no longer enforced")
