"""Each mode's tile-aspect choice must survive being constructed.

Maze declared ``_tile_aspect_mode = "preserve"`` at class level. The base
assigned ``self._tile_aspect_mode = "stretch"`` in ``__init__``, and an
INSTANCE attribute shadows a subclass's class attribute, so every maze window
was built in stretch mode and every arena was drawn distorted: a circular field
reads as elliptical, and the distortion changes with the window size, which is
indistinguishable from a lens that needs calibrating.

The base's default is a class attribute now, so a subclass that declares its
own value actually gets it. These read the resolved value off the class, which
is what an instance inherits; no Qt window is built.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from source.gui.base import MainWindowBase


def _resolved(cls) -> str:
    """What an instance of ``cls`` would see, absent an __init__ assignment."""
    return cls._tile_aspect_mode


def test_the_base_default_is_a_class_attribute():
    """If it goes back to being assigned in __init__, it silently overrides
    every subclass again."""
    assert "_tile_aspect_mode" in vars(MainWindowBase)
    assert _resolved(MainWindowBase) == "stretch"


def test_the_base_never_assigns_it_on_an_instance():
    """It was set in ``_setup_pipeline``, which every window calls, so the
    subclass's class attribute was overwritten on construction. Any instance
    assignment in the base brings that back, wherever it is written."""
    import inspect

    src = inspect.getsource(MainWindowBase)
    assert "self._tile_aspect_mode =" not in src, (
        "assigning it on the base's instance shadows a subclass's choice")


def test_maze_preserves_the_arena_shape():
    from source.gui.maze import MainWindow as MazeWindow

    assert _resolved(MazeWindow) == "preserve"


def test_operant_fills_the_cell():
    """A grid of sixteen chambers is read by glancing across it, so the height
    goes to the animals rather than to letterbox bars."""
    from source.gui.operant import MainWindow as OperantWindow

    assert _resolved(OperantWindow) in ("stretch", "preserve")
    import inspect
    src = inspect.getsource(OperantWindow.__init__)
    assert 'self._tile_aspect_mode = "stretch"' in src


def test_the_maze_tile_widget_letterboxes_too():
    """Two settings decide this: the window's mode picks how the frame is
    resized, and the tile's mode picks how it is finally painted. A tile left
    on stretch squashes the picture whatever the window decided."""
    from source.gui.widgets.video_tile import VideoTile
    from source.gui.widgets.setup_widget import SetupWidget  # noqa: F401
    import inspect

    src = inspect.getsource(SetupWidget)
    assert "VideoTile(lock_widget_aspect=True)" in src, (
        "maze's tile must take the letterbox default AND hold the camera's "
        "shape as the widget grows")
    assert "AspectBox(" in src, (
        "the tile goes in the layout through the box that sizes it")
    assert VideoTile.ASPECT_LETTERBOX == "letterbox"
    # ...and the default must BE letterbox.
    import inspect as _i
    sig = _i.signature(VideoTile.__init__)
    assert sig.parameters["aspect_mode"].default == VideoTile.ASPECT_LETTERBOX
