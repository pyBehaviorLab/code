"""Recording/video robustness.

RecorderSink.stop_recording returns the detached VideoRecorder so the caller
always reaps its ffmpeg child. A dead/disconnected port (OSError) is
classified as a fatal board error -> mcu_stop("error:…").
"""
from __future__ import annotations

from types import SimpleNamespace

from source.video.framebus.recorder_sink import RecorderSink
from source.gui.widgets.run_task import RunTask


# ---- the sink hands back the encoder -------------------------------------
def test_stop_recording_returns_the_encoder():
    sink = RecorderSink()
    fake_rec = SimpleNamespace(name="encoder")
    sink.start_recording(1, recorder=fake_rec)
    assert sink.is_recording(1) is True
    out = sink.stop_recording(1)
    assert out is fake_rec, "sink must return the detached encoder for reaping"
    assert sink.is_recording(1) is False
    # Second stop on an already-detached box returns None (idempotent).
    assert sink.stop_recording(1) is None


# ---- OSError counts as a fatal board error -------------------------------
def _drive_tick(exc):
    """Run RunTask.tick_active against a fake board that raises ``exc`` in
    process_data; return the list of mcu_stop reasons it triggered."""
    reasons = []

    class Board:
        framework_running = True
        def process_data(self):
            raise exc

    fake = SimpleNamespace(
        pycboard=Board(),
        setup_id=1,
        _log_func=lambda *a, **k: None,
        mcu_stop=lambda reason: reasons.append(reason),
    )
    RunTask.tick_active(fake)
    return reasons


def test_oserror_is_a_fatal_board_error():
    reasons = _drive_tick(OSError("device disconnected"))
    assert reasons and reasons[0].startswith("error:"), (
        "a port OSError must stop the box, not leave it ticking on a dead port"
    )


def test_serial_message_still_classified():
    reasons = _drive_tick(ValueError("serial read failed"))
    assert reasons and reasons[0].startswith("error:")


def test_transient_parse_error_does_not_stop():
    # A non-port error with no 'serial' marker is logged + skipped, not fatal.
    reasons = _drive_tick(ValueError("bad checksum on a frame"))
    assert reasons == [], "transient parse errors must not stop the box"


# ---- frame numbering + tracker feed into _video_data rows -----------------
def _make_frame(cam_frame_id, capture_host_ns):
    import numpy as np
    from datetime import datetime
    from source.video.framebus.types import BoxFrame

    return BoxFrame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        setup_id=1,
        cam_frame_id=cam_frame_id,
        camera_id=0,
        capture_host_ns=capture_host_ns,
        capture_wall=datetime.now(),
        is_shared_camera=False,
    )


class _RowCapture:
    """Fake TrackingWriter capturing write_frame kwargs."""

    def __init__(self):
        self.rows = []

    def write_frame(self, **kw):
        self.rows.append(kw)

    def close(self):
        pass


def _sink_with_writer():
    sink = RecorderSink()
    tw = _RowCapture()
    sink.start_recording(1, recorder=SimpleNamespace(recording=False),
                         tracking_writer=tw)
    rec_start = sink._rec_start_host_ns[1]
    return sink, tw, rec_start


def test_preroll_frame_does_not_anchor_frame_numbers():
    # A frame captured before record start must be dropped WITHOUT anchoring
    # the frame counter, so the first written row is frame_number == 1 and
    # TSV rows map 1:1 onto video frames.
    sink, tw, rec_start = _sink_with_writer()
    sink.process(_make_frame(100, rec_start - 1_000_000))  # pre-roll
    assert tw.rows == [], "pre-roll frame must not produce a row"
    sink.process(_make_frame(101, rec_start + 1_000_000))
    assert tw.rows and tw.rows[0]["frame_number"] == 1
    sink.process(_make_frame(102, rec_start + 2_000_000))
    assert tw.rows[1]["frame_number"] == 2


def test_tracker_result_lands_in_video_data_rows():
    # Blob-mode sessions: TrackerSink results must reach the _video_data row
    # (zone + speed), with pose_array staying None (no keypoints).
    sink, tw, rec_start = _sink_with_writer()
    sink.on_tracker_result(1, 50, (10.0, 20.0), "arm_A", 3.5,
                           {"centroid": {"arm_A": True}}, (0, 0, 5, 5),
                           capture_host_ns=rec_start)
    sink.process(_make_frame(51, rec_start + 1_000_000))
    assert tw.rows, "row must be written"
    row = tw.rows[0]
    assert row["location"] == "arm_A"
    assert row["speed"] == 3.5
    assert row["pose_array"] is None


def test_pipeline_wires_tracker_results_to_recorder():
    # Wiring guard: the Pipeline must subscribe the recorder to BOTH feeds.
    import inspect
    from source.video.framebus import controller

    src = inspect.getsource(controller)
    assert "self.tracker.on_result(self.recorder.on_tracker_result)" in src
    assert "self.pose.on_result(self.recorder.on_pose_result)" in src
