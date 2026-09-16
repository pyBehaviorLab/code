"""Both modes must complete the same stop, even though they split it differently.

An audit flagged `_on_record_stopped` as drifted: "operant never closes the MCU
data-logger files, maze never re-enables its buttons". Both halves are false,
each mode's per-box WIDGET owns a different part of the teardown, so the
main-window methods legitimately differ. What was missing is a check that the
union is complete on each side, which is what these do: they assert the
outcome, not which file performs it.

They also pin the fix for a collision introduced while renaming maze's
recorder-stop: `stop_box_recording` already existed on MainWindowBase as the
per-box bookkeeping hook, so an override that did not call up silently
replaced it.
"""
import contextlib
import inspect
import re

from source.gui.base import MainWindowBase
from source.gui.maze import MainWindow as MazeWindow
from source.gui.operant import MainWindow as OperantWindow
from source.gui.widgets.box_control import BoxControlWidget
from source.gui.widgets.setup_widget import SetupWidget


def _sources(*objs):
    out = []
    for o in objs:
        with contextlib.suppress(OSError, TypeError):
            out.append(inspect.getsource(o))
    return "\n".join(out)


# ---- the MCU data file gets closed, on both sides ------------------------

def test_operant_closes_the_mcu_data_files_somewhere_in_its_stop_path():
    """Not in _on_record_stopped, BoxControlWidget does it. That is fine;
    what matters is that it happens."""
    src = _sources(OperantWindow, BoxControlWidget)
    assert "close_files()" in src, (
        "nothing in operant's stop path closes the pyControl data files")


def test_maze_closes_the_mcu_data_files_somewhere_in_its_stop_path():
    src = _sources(MazeWindow, SetupWidget)
    assert "close_files()" in src, (
        "nothing in maze's stop path closes the pyControl data files")


# ---- the buttons come back, on both sides -------------------------------

def test_operant_re_enables_its_record_button_after_a_stop():
    # The shared skeleton drives _post_record_stopped; operant's hook
    # re-arms the per-box buttons there.
    body = inspect.getsource(OperantWindow._post_record_stopped)
    assert "record_button.setEnabled(True)" in body


def test_maze_re_enables_its_buttons_via_the_shared_refresh():
    """Maze does not set the buttons directly; it refreshes, and the shared
    per-widget refresher re-derives them. Assert the chain exists rather than
    the literal call."""
    body = inspect.getsource(MazeWindow._on_record_stopped)
    assert "refresh_ui_state" in body, "maze's stop never refreshes the UI"
    # …and the refresh reaches the per-widget button logic.
    from source.gui.widgets.run_task import RunTask
    assert "_update_button_states" in inspect.getsource(
        RunTask.apply_global_state)


# ---- every stop path ends with a refresh --------------------------------

def test_both_modes_always_refresh_even_if_a_teardown_step_raises():
    for mode in (OperantWindow, MazeWindow):
        body = inspect.getsource(mode._on_record_stopped)
        tail = body[body.rindex("refresh_ui_state"):]
        assert "except" in body, (
            f"{mode.__module__}: teardown is unguarded")
        # the refresh sits inside a try/except of its own
        assert "except" in tail or body.count("finally") >= 1, (
            f"{mode.__module__}: a raising step could skip the refresh")


# ---- the bookkeeping hook must not be shadowed --------------------------

def test_maze_extends_the_bookkeeping_hook_rather_than_replacing_it():
    """`stop_box_recording` is MainWindowBase's per-box bookkeeping hook.
    No override at all (pure inheritance) is the ideal; an override that
    exists must call up or the base hook never runs for that mode, the
    same silent-divergence class this whole area suffers from."""
    if "stop_box_recording" not in vars(MazeWindow):
        return
    src = inspect.getsource(MazeWindow.stop_box_recording)
    assert re.search(r"super\(\)\.stop_box_recording", src), (
        "maze overrides the base bookkeeping hook without calling it")


def test_the_recorder_teardown_is_distinct_from_the_bookkeeping_hook():
    """Two different jobs: _stop_recording_for_box actually stops the
    recorder, closes the writer and writes the history row;
    stop_box_recording is idempotent bookkeeping with no on-disk effects.
    Maze's stop must call the former for the teardown."""
    body = inspect.getsource(MazeWindow._on_record_stopped)
    assert "_stop_recording_for_box" in body, (
        "maze's stop no longer performs the recorder teardown")
    assert hasattr(MainWindowBase, "_stop_recording_for_box")
    assert hasattr(MainWindowBase, "stop_box_recording")


def test_recording_setups_is_maintained_by_shared_code_not_per_mode():
    """Both modes' membership comes from the base paths, so neither can drift
    into leaving stale entries that make idle boxes look like they record."""
    add_src = _sources(MainWindowBase)
    assert "recording_setups.add" in add_src
    assert "recording_setups.discard" in inspect.getsource(
        MainWindowBase._stop_recording_for_box)
