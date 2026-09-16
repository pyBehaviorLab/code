"""The shared _on_record_stopped skeleton's ordering is load-bearing.

Maze mirrors a synthetic ``framework_stopped`` event into the open
_video_data.txt writer; the shared teardown that follows is the sole
closer of that writer. If the mirror ever runs after the close, the
footer line silently vanishes from every maze session log.
"""
from source.gui.base import MainWindowBase


class _Host:
    """Records the order the skeleton drives its steps in."""

    _on_record_stopped = MainWindowBase._on_record_stopped

    def __init__(self, fail_step=None):
        self.calls = []
        self._fail_step = fail_step

    def _step(self, name):
        self.calls.append(name)
        if name == self._fail_step:
            raise RuntimeError(f"{name} exploded")

    def _pre_record_stopped(self, sid, data):
        self._step("pre")

    def stop_tracking_for_box(self, sid):
        self._step("stop_tracking")

    def _stop_recording_for_box(self, sid):
        self._step("stop_recording")

    def _post_record_stopped(self, sid, data):
        self._step("post")

    def stop_box_recording(self, sid):
        self._step("bookkeeping")

    def refresh_ui_state(self):
        self._step("refresh")


def test_the_writer_mirror_precedes_the_writer_close():
    h = _Host()
    h._on_record_stopped(1)
    assert h.calls.index("pre") < h.calls.index("stop_recording"), (
        "the framework_stopped mirror must run while the writer is open")


def test_full_order():
    h = _Host()
    h._on_record_stopped(1, {"fw_ms": 123})
    assert h.calls == ["pre", "stop_tracking", "stop_recording",
                       "post", "bookkeeping", "refresh"]


def test_a_failing_teardown_step_still_refreshes_the_ui():
    for step in ("stop_tracking", "stop_recording", "bookkeeping", "post"):
        h = _Host(fail_step=step)
        h._on_record_stopped(1)
        assert h.calls[-1] == "refresh", f"refresh skipped when {step} failed"


def test_a_failing_step_does_not_abort_the_guarded_tail():
    """The three guarded steps swallow their own errors, a dead tracker
    must not stop the recorder close or the bookkeeping."""
    h = _Host(fail_step="stop_tracking")
    h._on_record_stopped(1)
    assert "stop_recording" in h.calls and "bookkeeping" in h.calls
