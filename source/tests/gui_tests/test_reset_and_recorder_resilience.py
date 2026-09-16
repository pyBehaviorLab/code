"""Reset-aware status messages + recorder write-error resilience.

Reset (the button that reads "Reset" once a task is on the board) must show
the task name and the word "reset" in the STATUS LABEL, not just the log,
on start / success / failure, and failures must carry the cause. Previously
the status showed "Resetting task.." (no name) → "Uploaded : {task}" (wrong
verb) → "Upload failed" (wrong verb, no cause).

The recorder encode loop must survive a transient write error instead of
dying silently with recording stuck True.
"""
import numpy as np

from source.gui.widgets.run_task import RunTask, _short_cause

# ── _short_cause helper ─────────────────────────────────────────────────

def test_short_cause_collapses_and_truncates():
    assert _short_cause(ValueError("bad  \n  thing")) == "bad thing"
    long = _short_cause(RuntimeError("x" * 100))
    assert len(long) <= 48 and long.endswith("…")
    assert _short_cause(RuntimeError("")) == "RuntimeError"   # falls back to type


# ── on_upload_clicked reset-aware strings (drive the real handler) ──────

class _Board:
    status = {"framework": True}


class _Stub:
    """Minimal object carrying just what on_upload_clicked touches."""
    on_upload_clicked = RunTask.on_upload_clicked
    _post_upload_success_ui = RunTask._post_upload_success_ui
    _TASK_PLACEHOLDERS = RunTask._TASK_PLACEHOLDERS

    def __init__(self, uploaded, upload_ok=True, raise_exc=None):
        self.pycboard = _Board()
        self.task_uploaded = uploaded
        self.setup_id = 1
        self._upload_ok = upload_ok
        self._raise = raise_exc
        self.statuses = []       # (text, kind)
        self.briefs = []         # (text, error)

    # hooks the handler calls
    def _upload_log(self, msg): pass
    def _upload_brief_status(self, text, error=False): self.briefs.append((text, error))
    def _upload_task_text(self): return "ReversalLearning.py"
    def mcu_upload_task(self, rel, log=None, hw_def_path=None):
        if self._raise:
            raise self._raise
        return self._upload_ok
    def set_status(self, text, kind): self.statuses.append((text, kind))
    def _update_button_states(self): pass
    def _after_upload_clicked(self, *a): pass


def test_reset_success_names_task_and_says_reset():
    s = _Stub(uploaded=True, upload_ok=True)
    s.on_upload_clicked()
    # start brief
    assert any("Resetting 'ReversalLearning'" in t for t, _e in s.briefs)
    # success status label, reset verb + name, NOT "Uploaded"
    assert any(t.startswith("Reset 'ReversalLearning'") for t, _k in s.statuses)
    assert not any("Uploaded" in t for t, _k in s.statuses)


def test_upload_success_says_uploaded():
    s = _Stub(uploaded=False, upload_ok=True)
    s.on_upload_clicked()
    assert any(t == "Uploaded 'ReversalLearning'" for t, _k in s.statuses)


def test_reset_failure_carries_verb_and_cause():
    s = _Stub(uploaded=True, raise_exc=RuntimeError("port busy"))
    s.on_upload_clicked()
    # error brief uses the RESET verb + the cause, not a bare "Upload failed"
    errs = [t for t, e in s.briefs if e]
    assert errs and errs[-1] == "Reset failed: port busy"


def test_upload_soft_failure_uses_upload_verb():
    s = _Stub(uploaded=False, upload_ok=False)   # mcu_upload_task returns False
    s.on_upload_clicked()
    errs = [t for t, e in s.briefs if e]
    assert errs and errs[-1] == "Upload failed"


# ── recorder write-error resilience ─────────────────────────────────────

def test_record_loop_survives_transient_write_error():
    from source.video.recording.recorder import VideoRecorder

    class _FlakyWriter:
        def __init__(self): self.calls = 0
        def write(self, frame):
            self.calls += 1
            if self.calls == 1:
                raise OSError("broken pipe (transient)")   # first write fails
        def release(self): pass

    rec = VideoRecorder(camera_id=0, fps=20.0, resolution=(64, 48))
    rec.writer = _FlakyWriter()
    rec._target_size = (64, 48)
    rec.recording = True
    # push two frames: first triggers the error, second must still be written
    f = np.zeros((48, 64, 3), np.uint8)
    rec.frame_queue.append((f, 1))
    rec.frame_queue.append((f, 2))

    import threading
    import time
    th = threading.Thread(target=rec._external_record_loop, daemon=True)
    th.start()
    time.sleep(0.3)
    rec.recording = False
    th.join(timeout=2.0)

    assert not th.is_alive()                 # loop did NOT die on the error
    assert rec.writer.calls >= 2             # kept going, wrote the 2nd frame
    assert rec.frame_count >= 1              # at least the good frame counted


def test_record_loop_gives_up_after_sustained_failure():
    from source.video.recording.recorder import VideoRecorder

    class _DeadWriter:
        def write(self, frame): raise OSError("encoder gone")
        def release(self): pass

    fired = []
    rec = VideoRecorder(camera_id=0, fps=20.0, resolution=(64, 48))
    rec.writer = _DeadWriter()
    rec._target_size = (64, 48)
    rec._WRITE_ERROR_FATAL = 5               # small threshold for the test
    rec.drop_callback = lambda n, reason: fired.append(reason)
    rec.recording = True
    f = np.zeros((48, 64, 3), np.uint8)
    for i in range(20):
        rec.frame_queue.append((f, i))

    import threading
    th = threading.Thread(target=rec._external_record_loop, daemon=True)
    th.start()
    th.join(timeout=2.0)

    assert not th.is_alive()
    assert rec.recording is False            # gave up
    assert "encoder_write_failed" in fired   # signalled health


def test_record_loop_gives_up_when_writer_is_dead():
    """A writer returning False AND reporting is_healthy()==False (dead ffmpeg
    pipe) must trip the fatal-alarm path instead of silently losing video."""
    from source.video.recording.recorder import VideoRecorder

    class _DeadPipeWriter:
        def write(self, frame): return False       # rejects
        def is_healthy(self): return False         # ...because it's dead
        def release(self): pass

    fired = []
    rec = VideoRecorder(camera_id=0, fps=20.0, resolution=(64, 48))
    rec.writer = _DeadPipeWriter()
    rec._target_size = (64, 48)
    rec._WRITE_ERROR_FATAL = 5
    rec.drop_callback = lambda n, reason: fired.append(reason)
    rec.recording = True
    f = np.zeros((48, 64, 3), np.uint8)
    for i in range(20):
        rec.frame_queue.append((f, i))

    import threading
    th = threading.Thread(target=rec._external_record_loop, daemon=True)
    th.start()
    th.join(timeout=2.0)

    assert not th.is_alive()
    assert rec.recording is False
    assert rec.encoder_failed is True        # RecorderSink can now alarm
    assert "encoder_write_failed" in fired


def test_record_loop_survives_transient_backpressure():
    """A writer returning False but still HEALTHY (transient full queue) must
    drop the frame and keep recording, a slow disk must not end the session."""
    from source.video.recording.recorder import VideoRecorder

    class _BackpressureWriter:
        def __init__(self): self.calls = 0
        def write(self, frame):
            self.calls += 1
            return False                          # always "queue full"
        def is_healthy(self): return True         # ...but writer is alive
        def release(self): pass

    fired = []
    rec = VideoRecorder(camera_id=0, fps=20.0, resolution=(64, 48))
    rec.writer = _BackpressureWriter()
    rec._target_size = (64, 48)
    rec._WRITE_ERROR_FATAL = 5
    rec.drop_callback = lambda n, reason: fired.append(reason)
    rec.recording = True
    f = np.zeros((48, 64, 3), np.uint8)
    for i in range(10):
        rec.frame_queue.append((f, i))

    import threading
    import time
    th = threading.Thread(target=rec._external_record_loop, daemon=True)
    th.start()
    time.sleep(0.3)
    rec.recording = False
    th.join(timeout=2.0)

    assert rec.encoder_failed is False           # NOT escalated to fatal
    assert rec.recording is False                # only because we stopped it
    assert "queue_full" in fired                 # drops were accounted
    assert "encoder_write_failed" not in fired


def test_record_loop_accepts_writer_returning_none():
    """A writer whose write() returns None on success (cv2.VideoWriter) must
    NOT be mistaken for a rejection."""
    from source.video.recording.recorder import VideoRecorder

    class _NoneWriter:
        def __init__(self): self.calls = 0
        def write(self, frame): self.calls += 1  # returns None
        def release(self): pass

    rec = VideoRecorder(camera_id=0, fps=20.0, resolution=(64, 48))
    rec.writer = _NoneWriter()
    rec._target_size = (64, 48)
    rec.recording = True
    f = np.zeros((48, 64, 3), np.uint8)
    rec.frame_queue.append((f, 1))
    rec.frame_queue.append((f, 2))

    import threading
    import time
    th = threading.Thread(target=rec._external_record_loop, daemon=True)
    th.start()
    time.sleep(0.3)
    rec.recording = False
    th.join(timeout=2.0)

    assert rec.frame_count >= 2               # both frames counted, no false reject
    assert rec.encoder_failed is False
