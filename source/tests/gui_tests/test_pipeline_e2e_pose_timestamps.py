"""The pose path end to end, through a real window, against a known clock.

Driven from the GUI, because that is where the decisions are made: a real
``MainWindow``, real boxes added through ``add_setup``, a real ``Pipeline``
underneath it, the real ``PoseSink`` and ``RecorderSink``, and a real
``FrameLog`` writing a real file that is read back off disk.

Three stand-ins, each a thing this file is not about:

  * the camera, so frames can be published at instants chosen by the test;
  * the inference backend, returning a keypoint that carries the box id, so a
    pose in a file can be traced back to the box it was inferred for;
  * the board, whose firmware clock is an exact line through host time, which
    is what turns the timestamp assertions into arithmetic.

What is under test, in the order the operator meets it:

  init      a configured box loads its model when its camera comes up, and
            loading does not start inference;
  run start what Record costs on a ready box: one enable call and no model
            work, for DeepLabCut and for SLEAP;
  time      that ``frame_fw_ms`` is the board's time at that frame's capture,
            that ``pose_lag_ms`` is the measured time from that capture to the
            pose existing, and that ``filter_ms`` is the smoother's share of
            it;
  scale     one setup, then four on one camera, with attribution checked from
            the files: box 2's animal must never appear in box 3's recording.

The timestamps are why this exists. Everything downstream, the alignment to
the task and the analysis, is derived from them, and an error there is
invisible in the GUI and permanent in the data.
"""
from __future__ import annotations

import json
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import threading
import time

import numpy as np
import pytest
from PySide6 import QtWidgets

from source import host_clock
from source.tests.qt_dispose import WidgetBin
from source.video.cameras import capture
from source.video.framebus.types import CameraFrame, TrackingConfig

W, H = 320, 240
PARTS = ["snout", "centre", "tail"]

#: The board's firmware clock as a line through host time. Far from zero on
#: purpose: a mapping that accidentally returned host milliseconds would pass
#: a test whose board started at the same place.
FW_EPOCH_MS = 400_000.0
PERIOD_NS = 33_000_000

_BIN = WidgetBin()


def _frame(v=128):
    return np.full((H, W, 3), int(v), np.uint8)


def _quadrant(box_number):
    """One cell of a 2x2 split, in the percent geometry the bus wants."""
    i = (int(box_number) - 1) % 4
    return {"boxes": [{"box_number": int(box_number),
                       "geometry": {"percent": {"x": 0.5 * (i % 2),
                                                "y": 0.5 * (i // 2),
                                                "width": 0.5,
                                                "height": 0.5}}}]}


# ── stand-ins ─────────────────────────────────────────────────────────


class _FakeCam:
    def __init__(self, camera_id=None, width=None, height=None,
                 target_fps=None, camera_config=None):
        self.camera_id = camera_id
        self.width, self.height = int(width or W), int(height or H)
        self.target_fps = target_fps
        self.connected = True
        self.stopped = False
        self.grayscale = False
        self.connection_checked = threading.Event()
        self.connection_checked.set()

    def start(self):
        pass

    def isRunning(self):
        return not self.stopped

    is_alive = isRunning

    def stop(self):
        self.stopped = True
        self.connected = False

    def wait(self, timeout=None):
        return True

    def set_target_fps(self, f):
        self.target_fps = f

    def get_last_frame(self):
        return _frame()

    def get_latest_frame_versioned(self, last_version):
        return None

    def drain_recording_buffer(self):
        return []


class _Board:
    """Firmware time as an exact function of host time.

    The real board re-anchors on every message, so its mapping drifts by
    design. This one does not, because the point is to know what each row
    SHOULD say and compare against it.
    """

    def __init__(self, host0_ns: int):
        self.host0_ns = int(host0_ns)
        self.framework_running = False
        self.status = {"framework": True, "serial": True}
        self.data_consumers = []

    def fw_ms_at(self, host_ns):
        return FW_EPOCH_MS + (int(host_ns) - self.host0_ns) / 1e6

    @property
    def timestamp(self):
        return self.fw_ms_at(host_clock.host_ns())


class _Model:
    batched = True

    def get_body_parts(self):
        return list(PARTS)


class _Handle:
    def __init__(self, key):
        self.key = key
        self.model = _Model()

    @property
    def is_initialized(self):
        return True


#: How a pose says which box it was inferred for. NOT the coordinates: the
#: pipeline maps model coordinates back through the box's own input transform
#: (a letterbox, or a CCTV crop at half scale), so an x that went in as "2"
#: comes out as 4. Confidence passes through untouched, so it is the one field
#: that can carry an identity from the backend to the file.
def _tag(box_id):
    return round(0.50 + int(box_id) / 100.0, 4)


class _Backend:
    """Returns a pose tagged with the box it came from.

    ``submit_batch`` calls back inline: this file is about what reaches the
    file, and a scheduler in between only adds a race to wait on. Batch sizes
    are recorded, because "did the boxes ride in one pass" is one of the
    questions asked here.
    """

    def __init__(self):
        self.builds = []          # models actually CONSTRUCTED
        self.asks = []            # every get_or_create, hit or miss
        self.batches = []
        self.delay_s = 0.0
        self._cache = {}
        #: Bumped per batch and written into the coordinates, so two results
        #: are distinguishable. With a constant pose every result reads as
        #: the same one and "was this refreshed" cannot be asked.
        self._serial = 0

    def get_or_create(self, key, **kw):
        """Caches by key, exactly as the real backend does.

        Without the cache this double reports a build for every ask, and
        "was the model rebuilt at Record" becomes unanswerable: the run start
        legitimately re-configures the box, and the whole question is whether
        that costs a model.
        """
        self.asks.append(tuple(key))
        handle = self._cache.get(tuple(key))
        if handle is None:
            self.builds.append(tuple(key))
            handle = self._cache[tuple(key)] = _Handle(key)
        return handle

    def submit_batch(self, model, items, on_done):
        self.batches.append(len(items))
        if self.delay_s:
            time.sleep(self.delay_s)
        self._serial += 1
        for box_id, _frame_in in items:
            on_done(box_id, {p: [10.0 * i + self._serial, 20.0 * i,
                                 _tag(box_id)]
                             for i, p in enumerate(PARTS, start=1)})
        return len(items)

    def shutdown(self, wait=False):
        pass


class _Rec:
    def __init__(self):
        self.frames = 0
        self.recording = True
        self.encoder_failed = False

    def add_frame(self, frame, capture_host_ns=0, *a, **kw):
        self.frames += 1
        return True

    def stop_recording(self, *a, **kw):
        return None

    def stop(self, *a, **kw):
        pass

    def release(self, *a, **kw):
        pass


# ── harness ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _quiet_modals(monkeypatch):
    """Offscreen, a modal waits for a click that never comes."""
    yes = QtWidgets.QMessageBox.StandardButton.Yes
    ok = QtWidgets.QMessageBox.StandardButton.Ok
    for name, answer in (("question", yes), ("warning", ok),
                         ("information", ok), ("critical", ok)):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda *a, **k: answer))
    yield
    _BIN.drain()


class Rig:
    """A real operant window with N boxes, a camera, and a board."""

    def __init__(self, tmp_path, boxes, tracker_type="dlc"):
        from source.gui.operant import MainWindow

        self.tmp_path = tmp_path
        self.boxes = list(boxes)
        # Readiness refuses a model that is not on disk, and says so, so the
        # folder has to exist for the flow to be exercised at all.
        self.model_dir = tmp_path / "models" / tracker_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.window = _BIN.add(MainWindow())
        self.pipeline = self.window.pipeline
        # The window has no Camera Connect dialog here, so the resolution the
        # dialog would have chosen is set directly. Without it the camera
        # refuses to start and every assertion below fails for the wrong
        # reason.
        self.pipeline.update_camera_config(
            0, selected_resolution=(W, H), selected_fps=30,
            camera_backend="opencv")
        self.backend = _Backend()
        self.pipeline.pose._backend = self.backend
        self.host0_ns = host_clock.host_ns()
        self.board = _Board(self.host0_ns)
        self.logs, self.recs = {}, {}

        for bid in self.boxes:
            self.window.add_setup()
            self.pipeline.update_tracking_config(
                bid, tracker_type=tracker_type,
                dlc_model_path=str(self.model_dir),
                keypoint_names=tuple(PARTS), online_tracking_enabled=True,
                confidence_threshold=0.5, user_applied=True)

    # -- lifecycle ----------------------------------------------------

    def connect_cameras(self):
        shared = len(self.boxes) > 1
        for bid in self.boxes:
            ok = self.pipeline.connect_camera(
                0, bid, segment_config=_quadrant(bid) if shared else None)
            assert ok, f"box {bid} did not attach to the camera"
            self.window.video_manager.box_camera_map[bid] = 0

    def make_ready(self):
        """What the window does when a camera comes up."""
        return self.window.ensure_pose_ready(self.boxes)

    def start_recording(self):
        from source.video.recording.frame_log import FrameLog

        for bid in self.boxes:
            self.recs[bid] = _Rec()
            log = FrameLog(self.tmp_path / f"box{bid}_video_data.txt")
            log.write_info("tracking_mode", "dlc")
            self.logs[bid] = log
            self.pipeline.start_recording(bid, recorder=self.recs[bid],
                                          tracking_writer=log,
                                          pycboard=self.board)

    def start_tracking(self):
        """The run start, through the window's own entry point."""
        return [self.window.start_tracking_for_box(bid, force=True)
                for bid in self.boxes]

    def publish(self, n, pace_s=0.002, real_stamps=False):
        """``pace_s`` is WALL time between publishes; the stamps are always a
        frame period apart. The two are separate on purpose: most checks here
        only need the stamps and run fastest when the frames are pushed as
        quickly as the loop allows, but anything about REFRESH needs the real
        cadence, or the whole run is over before one inference finishes and
        the file carries a single pose."""
        bus = self.pipeline.get_bus(0)
        assert bus is not None
        stamps = []
        for i in range(n):
            ts = (host_clock.host_ns() if real_stamps
                  else self.host0_ns + i * PERIOD_NS)
            stamps.append(ts)
            bus.publish_frame(CameraFrame(
                image=_frame(100 + i % 40), cam_frame_id=i, camera_id=0,
                capture_host_ns=ts, capture_wall=time.time(),
                is_shared=len(self.boxes) > 1, box_ids=tuple(self.boxes)))
            time.sleep(pace_s)
        self.stamps = stamps
        return stamps

    def drain(self, timeout=12.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            pending = 0
            for sink in (self.pipeline.recorder, self.pipeline.tracker,
                         self.pipeline.pose):
                with sink._lock:
                    pending += sum(len(d) for d in sink._per_box.values())
            if pending == 0:
                time.sleep(0.4)
                return
            time.sleep(0.05)
        raise AssertionError("the sinks did not drain")

    def finish(self):
        """Close the session files so they can be read.

        A ``FrameLog`` buffers 64 KB and holds the current row until the next
        frame arrives, so a file read while recording is still open is short
        by at least one row and usually by all of them. Reading before closing
        is how a test can report "the recorder wrote 17 frames and the file
        has none" when both halves are working.
        """
        for bid in self.boxes:
            try:
                self.pipeline.stop_recording(bid)
            except Exception:
                pass
            log = self.logs.pop(bid, None)
            if log is not None:
                log.close()

    def stop(self):
        self.finish()
        for bid in self.boxes:
            try:
                self.pipeline.stop_recording(bid)
            except Exception:
                pass
            log = self.logs.get(bid)
            if log is not None:
                log.close()
        try:
            self.window.close()
        except Exception:
            pass

    # -- reading back -------------------------------------------------

    def rows(self, bid):
        path = self.tmp_path / f"box{bid}_video_data.txt"
        text = path.read_text(encoding="utf-8")
        cols = next(ln for ln in text.split("\n")
                    if ln.startswith("#columns")).split(None, 1)[1].split()
        out = []
        for ln in text.split("\n"):
            if not ln or ln.startswith("#") or not ln[0].isdigit():
                continue
            out.append(dict(zip(cols, ln.split("\t"))))
        return out

    def expected_fw(self, host_ns):
        return FW_EPOCH_MS + (host_ns - self.host0_ns) / 1e6


@pytest.fixture
def rig(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "CameraThread", _FakeCam)
    made = []

    def _build(boxes=(1,), tracker_type="dlc"):
        r = Rig(tmp_path, boxes, tracker_type)
        made.append(r)
        return r

    yield _build
    for r in made:
        r.stop()


def _run(rig_factory, boxes=(1,), tracker_type="dlc", frames=24, delay_s=0.0,
         pace_s=0.002, real_stamps=False):
    """Load, ready, record, track, publish, drain. The whole flow."""
    r = rig_factory(boxes, tracker_type)
    r.connect_cameras()
    assert r.make_ready() == len(boxes), "not every configured box got ready"
    r.backend.delay_s = delay_s
    r.start_recording()
    r.start_tracking()
    r.publish(frames, pace_s=pace_s, real_stamps=real_stamps)
    r.drain()
    r.finish()
    return r


# ══ init: when the model is loaded, and what that does ════════════════


class TestInit:
    def test_a_configured_box_is_made_ready_when_its_camera_comes_up(self, rig):
        r = rig()
        r.connect_cameras()
        assert not r.window.pose_box_is_ready(1), "ready before anything loaded"
        assert r.make_ready() == 1
        assert r.pipeline.has_pose_model()
        assert r.window.pose_box_is_ready(1), (
            "the box loaded a model and still reports that it needs an init")

    def test_a_box_with_no_tracking_configured_is_left_alone(self, rig):
        """It is not misconfigured, it is a box that records video."""
        r = rig()
        r.pipeline.update_tracking_config(1, dlc_model_path=None,
                                          tracker_type="blob")
        r.connect_cameras()
        assert r.window.pose_configured_boxes() == []
        assert r.make_ready() == 0
        assert not r.pipeline.has_pose_model()

    def test_readiness_does_not_start_inference(self, rig, tmp_path):
        """Opening a project must not begin tracking every animal in it."""
        r = rig()
        r.connect_cameras()
        r.make_ready()
        r.start_recording()
        r.publish(10)
        r.drain()
        r.finish()
        assert r.backend.batches == [], "inference ran before the run started"
        assert all(row["pose"] == "na" for row in r.rows(1))

    def test_a_run_start_costs_one_enable_and_no_model_work(self, rig):
        """A model built here is a model built with the animal in the box."""
        r = rig()
        r.connect_cameras()
        r.make_ready()
        builds = len(r.backend.builds)
        r.start_recording()
        assert all(r.start_tracking()), "the run did not start tracking"
        r.publish(16)
        r.drain()
        r.finish()
        assert len(r.backend.builds) == builds, (
            "the model was BUILT again at Record. The run start does "
            "re-configure the box, which is cheap and correct; paying for a "
            "model there is neither.")
        assert len(r.backend.asks) > builds, (
            "the run start did not go through the loader at all, so this "
            "test is not exercising the path it claims to")
        assert r.backend.batches, "no inference ran after the run started"
        assert any(row["pose"] != "na" for row in r.rows(1))

    def test_being_ready_twice_does_not_load_twice(self, rig):
        r = rig()
        r.connect_cameras()
        r.make_ready()
        builds = len(r.backend.builds)
        assert r.make_ready() == 1
        assert len(r.backend.builds) == builds

    @pytest.mark.parametrize("tracker_type", ["dlc", "sleap"])
    def test_both_backends_reach_a_ready_box(self, rig, tracker_type):
        r = rig(tracker_type=tracker_type)
        r.connect_cameras()
        assert r.make_ready() == 1
        assert r.window.pose_box_is_ready(1)
        assert r.pipeline.pose_fingerprint()[0] == tracker_type

    def test_switching_backend_is_a_different_model(self, rig):
        """Or a SLEAP box quietly keeps running the DeepLabCut one."""
        r = rig(tracker_type="dlc")
        r.connect_cameras()
        r.make_ready()
        dlc = r.pipeline.pose_fingerprint()
        other = r.tmp_path / "models" / "sleap"
        other.mkdir(parents=True, exist_ok=True)
        r.pipeline.update_tracking_config(
            1, tracker_type="sleap", dlc_model_path=str(other))
        assert not r.window.pose_box_is_ready(1), (
            "the box reports ready with the other backend's model loaded")
        assert r.make_ready() == 1
        assert r.pipeline.pose_fingerprint() != dlc

    def test_a_settings_change_makes_the_box_not_ready(self, rig):
        r = rig()
        r.connect_cameras()
        r.make_ready()
        r.pipeline.update_tracking_config(1, confidence_threshold=0.9)
        assert not r.window.pose_box_is_ready(1)
        r.make_ready()
        assert r.window.pose_box_is_ready(1)

    def test_a_runtime_knob_does_not(self, rig):
        """Asking for an init after a push-gate change would be the same fault
        in the other direction."""
        r = rig()
        r.connect_cameras()
        r.make_ready()
        r.pipeline.update_tracking_config(1, push_coords_to_mcu=False,
                                          pose_n_instances=3)
        assert r.window.pose_box_is_ready(1)

    def test_the_engine_is_sized_for_the_boxes_that_will_track(self, rig):
        r = rig(boxes=(1, 2, 3, 4))
        r.pipeline.update_tracking_config(4, online_tracking_enabled=False)
        assert r.pipeline.n_pose_boxes() == 3


# ══ timestamps in the session file ════════════════════════════════════


class TestTimestamps:
    def test_frame_fw_ms_is_the_board_time_of_that_frames_capture(self, rig):
        """Row by row, against the mapping. This column is what every later
        alignment to the task is built on: a constant error shifts every
        event, a per-row error scatters them, and neither shows in the GUI."""
        r = _run(rig, frames=24)
        rows = [row for row in r.rows(1) if row["frame_fw_ms"] != "na"]
        assert len(rows) >= 15, f"only {len(rows)} rows carry a board time"

        # Which published frame the run started on is not fixed: recording
        # begins between publishes. Anchor on the first row, then every later
        # row must match ITS OWN frame, one period apart.
        first = float(rows[0]["frame_fw_ms"])
        base = int(rows[0]["frame"])
        for row in rows:
            steps = int(row["frame"]) - base
            want = first + steps * (PERIOD_NS / 1e6)
            assert abs(float(row["frame_fw_ms"]) - want) <= 1.0, (
                f"frame {row['frame']}: board time {row['frame_fw_ms']} "
                f"against {want:.1f}")

    def test_the_board_time_is_the_boards_and_not_the_hosts(self, rig):
        """The whole column would still be self-consistent if it carried host
        milliseconds, so check it against the epoch only the board has."""
        r = _run(rig, frames=16)
        fw = [float(row["frame_fw_ms"]) for row in r.rows(1)
              if row["frame_fw_ms"] != "na"]
        assert fw, "no row carries a board time"
        assert FW_EPOCH_MS <= min(fw) <= FW_EPOCH_MS + 2000, (
            f"the first row says {min(fw)}, the board says {FW_EPOCH_MS}")

    def test_the_board_time_never_steps_backwards(self, rig):
        r = _run(rig, frames=24)
        fw = [float(row["frame_fw_ms"]) for row in r.rows(1)
              if row["frame_fw_ms"] != "na"]
        assert fw == sorted(fw)

    def test_a_frame_period_of_host_time_is_a_frame_period_of_board_time(
            self, rig):
        """If the two clocks were being mixed anywhere, the spacing is where
        it shows."""
        r = _run(rig, frames=24)
        fw = [float(row["frame_fw_ms"]) for row in r.rows(1)
              if row["frame_fw_ms"] != "na"]
        gaps = [b - a for a, b in zip(fw, fw[1:])]
        assert gaps
        assert max(abs(g - PERIOD_NS / 1e6) for g in gaps) <= 1.0, (
            f"gaps: {[round(g, 2) for g in gaps[:8]]}")

    def test_the_pose_age_is_the_measured_inference_time(self, rig):
        """``pose_lag_ms`` is capture to pose-exists, measured by the sink.

        A model made to take 20 ms has to show up as at least that: the number
        is the operator's only account of how far behind the animal the
        keypoints are, and one that reported near zero on a slow model would
        be worse than none.
        """
        r = _run(rig, frames=20, delay_s=0.02, pace_s=0.012,
                 real_stamps=True)
        ages = [float(row["pose_lag_ms"]) for row in r.rows(1)
                if row["pose_lag_ms"] != "na"]
        assert ages, "no row carries an inference latency"
        assert min(ages) >= 15.0, (
            f"a 20 ms model reported as little as {min(ages):.1f} ms")
        assert max(ages) < 2000.0, f"{max(ages):.0f} ms is not plausible"

    def test_the_smoother_is_a_small_share_of_it(self, rig):
        """``filter_ms`` is the Kalman and optical-flow pass over every part.
        It has to be measured, present, and a fraction of the inference it
        sits inside; a smoother that cost more than the model would be the
        thing to fix."""
        r = _run(rig, frames=20, delay_s=0.02, pace_s=0.012,
                 real_stamps=True)
        rows = [row for row in r.rows(1)
                if row["pose_lag_ms"] != "na" and row["filter_ms"] != "na"]
        assert rows, "no row carries a smoother latency"
        for row in rows:
            filt, infer = float(row["filter_ms"]), float(row["pose_lag_ms"])
            assert filt >= 0.0
            assert filt <= infer, (
                f"the smoother took {filt:.2f} ms of a {infer:.1f} ms "
                "inference, which cannot be a part of it")

    def test_a_repeated_pose_keeps_the_age_it_was_measured_with(self, rig):
        """A slow model means one pose is written onto several rows, and every
        one of them carries that pose's own latency. Rows whose latency
        changed while the pose did not would mean the number was being
        recomputed per row rather than carried with the result."""
        r = _run(rig, frames=24, delay_s=0.02, pace_s=0.012,
                 real_stamps=True)
        rows = [row for row in r.rows(1) if row["pose"] != "na"]
        assert len(rows) >= 8
        by_pose = {}
        for row in rows:
            by_pose.setdefault(row["pose"], set()).add(row["pose_lag_ms"])
        for pose, latencies in by_pose.items():
            assert len(latencies) == 1, (
                f"one pose carries {len(latencies)} different latencies: "
                f"{sorted(latencies)}")
        assert len(by_pose) > 1, (
            "one pose was written for the whole run, so nothing was refreshed")

    def test_the_two_time_columns_agree(self, rig):
        """``elapsed`` comes from the host, ``frame_fw_ms`` from the board.
        Different code writes each, and they have to tell the same story."""
        r = _run(rig, frames=24)
        rows = [row for row in r.rows(1) if row["frame_fw_ms"] != "na"]
        first_fw = float(rows[0]["frame_fw_ms"])

        def _seconds(text):
            parts = [float(p) for p in text.split(":")]
            out = parts[-1]
            if len(parts) > 1:
                out += parts[-2] * 60
            if len(parts) > 2:
                out += parts[-3] * 3600
            return out

        base = _seconds(rows[0]["elapsed"])
        for row in rows:
            host_s = _seconds(row["elapsed"]) - base
            board_s = (float(row["frame_fw_ms"]) - first_fw) / 1000.0
            assert abs(host_s - board_s) <= 0.05, (
                f"host says {host_s:.3f}s, board says {board_s:.3f}s")

    def test_the_pose_written_is_the_pose_inferred(self, rig):
        r = _run(rig, frames=16)
        posed = [row for row in r.rows(1) if row["pose"] != "na"]
        assert posed
        points = json.loads(posed[-1]["pose"])
        assert len(points) == len(PARTS)
        assert all(len(p) == 3 for p in points), "x, y and confidence"
        assert {round(p[2], 4) for p in points} == {_tag(1)}, (
            "box 1's row carries another box's pose")


# ══ inference latency ═════════════════════════════════════════════════


class TestLatency:
    def test_the_time_the_model_takes_is_the_time_reported(self, rig):
        """The number an operator reads has to be the number the pipeline
        spends. A 30 ms model has to show up as roughly 30 ms."""
        r = _run(rig, frames=24, delay_s=0.03)
        samples = list(r.pipeline.latency._rings.get("poll_to_infer", ()))
        assert samples, "poll_to_infer was never sampled"
        median = sorted(samples)[len(samples) // 2]
        assert 20.0 <= median <= 300.0, (
            f"a 30 ms model measured as {median:.1f} ms")

    def test_a_faster_model_reports_a_smaller_number(self, rig):
        """Absolute values depend on the host; the ORDER must not."""
        slow = _run(rig, frames=20, delay_s=0.03)
        slow_med = sorted(slow.pipeline.latency._rings["poll_to_infer"])
        fast = _run(rig, frames=20, delay_s=0.0)
        fast_med = sorted(fast.pipeline.latency._rings["poll_to_infer"])
        assert slow_med and fast_med
        assert (slow_med[len(slow_med) // 2]
                > fast_med[len(fast_med) // 2]), "the stage does not track the model"


# ══ one setup, then four ══════════════════════════════════════════════


class TestScale:
    def test_one_box_is_posed_from_the_first_result_onward(self, rig):
        """There is a warm-up: rows are written as frames arrive, and the
        first inference has not answered yet. After it has, every row carries
        a pose."""
        r = _run(rig, frames=30)
        rows = r.rows(1)
        assert r.recs[1].frames == len(rows), "the recorder is lossless"
        posed = [i for i, row in enumerate(rows) if row["pose"] != "na"]
        assert posed, "no row ever got a pose"
        after = rows[posed[0]:]
        missing = [row["frame"] for row in after if row["pose"] == "na"]
        assert not missing, f"gaps after the first pose: {missing[:8]}"

    def test_four_boxes_each_get_their_own_animal(self, rig):
        """The model is shared and the sinks run pools, so a mix-up here would
        put one box's animal in another's session with nothing to say so."""
        boxes = (1, 2, 3, 4)
        r = _run(rig, boxes=boxes, frames=24)
        for bid in boxes:
            rows = r.rows(bid)
            posed = [json.loads(row["pose"]) for row in rows
                     if row["pose"] != "na"]
            assert posed, f"box {bid} never got a pose"
            tags = {round(p[2], 4) for pose in posed for p in pose}
            assert tags == {_tag(bid)}, (
                f"box {bid}'s file carries poses tagged "
                f"{sorted(tags)}, its own tag is {_tag(bid)}")

    def test_four_boxes_all_record_losslessly(self, rig):
        boxes = (1, 2, 3, 4)
        r = _run(rig, boxes=boxes, frames=24)
        for bid in boxes:
            assert r.recs[bid].frames == len(r.rows(bid)), (
                f"box {bid}: recorded frames and written rows disagree")

    def test_the_boxes_ride_in_one_forward_pass(self, rig):
        """Four boxes on one camera should batch, not queue: one call pays the
        launch overhead instead of four."""
        r = _run(rig, boxes=(1, 2, 3, 4), frames=24)
        assert r.backend.batches, "nothing was ever submitted"
        assert max(r.backend.batches) > 1, (
            f"every batch held one box: {r.backend.batches[:12]}")

    def test_every_box_agrees_about_when_a_frame_was_captured(self, rig):
        """Same camera, same instants, four files written by different
        workers. They have to agree row for row."""
        boxes = (1, 2, 3, 4)
        r = _run(rig, boxes=boxes, frames=24)
        # By TIME, not by frame number: the number is local to each box's own
        # recording, which starts between publishes, so box 1's frame 1 and
        # box 4's frame 1 are not the same capture.
        per_box = {bid: {round(float(row["frame_fw_ms"]), 1)
                         for row in r.rows(bid) if row["frame_fw_ms"] != "na"}
                   for bid in boxes}
        shared = set.intersection(*per_box.values())
        assert len(shared) >= 10, (
            "the boxes hardly share a capture instant: "
            + ", ".join(f"box {b}: {len(v)}" for b, v in per_box.items()))
        for bid, times in per_box.items():
            extra = times - shared
            assert len(extra) <= 4, (
                f"box {bid} has {len(extra)} capture instants no other box "
                f"saw, from one camera: {sorted(extra)[:5]}")

    def test_the_latency_columns_are_per_box_too(self, rig):
        """Not just the coordinates: each box's latencies have to be its own
        pose's, or the number describes another box's inference."""
        boxes = (1, 2, 3, 4)
        r = _run(rig, boxes=boxes, frames=24, delay_s=0.01,
                 real_stamps=True)
        for bid in boxes:
            rows = [row for row in r.rows(bid) if row["pose_lag_ms"] != "na"]
            assert rows, f"box {bid} has no inference latency"
            for row in rows:
                infer = float(row["pose_lag_ms"])
                assert 0.0 <= infer <= 5000.0, (
                    f"box {bid}: {infer:.0f} ms is not a latency this box saw")
                assert row["filter_ms"] != "na", (
                    f"box {bid} has an inference latency and no smoother one")
