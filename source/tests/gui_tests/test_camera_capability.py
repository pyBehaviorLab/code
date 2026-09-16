"""A camera that under-delivers must say so.

Built from a real measurement rather than an invented one: asked for 1280x720
at 30 fps, this rig's webcam reported ``CAP_PROP_FPS == 30.0`` and delivered
9.8; at 1920x1080 it reported 30.0 and delivered 5.0. Nothing in the stack
objected, because every layer between the request and the sensor answers with
the request.

So what is pinned here is the one comparison that cannot be fooled, frames
that arrived against frames that were asked for, and the rule that it stays
quiet until it has something true to say.
"""
from __future__ import annotations

import pytest

from source.video.cameras.base import SHORTFALL_RATIO, Shortfall


def _short(thread):
    """``thread.shortfall()`` with the None case the old free function had."""
    return None if thread is None else thread.shortfall()


class _Cam:
    """The camera object surface ``assess`` reads."""

    def __init__(self, *, req_wh=(1280, 720), act_wh=(1280, 720),
                 fourcc="MJPG", backend="DSHOW"):
        self._requested_width, self._requested_height = req_wh
        self._act = act_wh
        self._fourcc = fourcc
        self._open_backend = backend

    def get_width(self):
        return self._act[0]

    def get_height(self):
        return self._act[1]

    def _read_fourcc_str(self):
        return self._fourcc


class _Thread:
    """Stands in for CameraThread, which now owns ``shortfall()`` because it
    already holds the requested rate, the measured rate and the camera."""

    from source.video.cameras.capture import CameraThread as _CT
    shortfall = _CT.shortfall

    def __init__(self, target, measured, cam=None, camera_id="cam0"):
        self._t, self._m = target, measured
        self._camera = cam if cam is not None else _Cam()
        self.camera_id = camera_id

    def get_target_fps(self):
        return self._t

    def get_measured_fps(self):
        return self._m


def test_the_measured_case_that_prompted_this():
    """720p asked at 30, delivering 9.8, uncompressed, the real reading."""
    s = _short(_Thread(30.0, 9.8, _Cam(fourcc="YUY2")))
    assert s is not None
    assert s.rate_differs
    assert s.uncompressed_at_high_res
    msg = s.message()
    assert "30 fps" in msg and "9.8" in msg
    assert "YUY2" in msg and "uncompressed" in msg


def test_a_camera_meeting_its_request_says_nothing():
    assert _short(_Thread(30.0, 30.0)) is None
    assert _short(_Thread(30.0, 29.97)) is None, (
        "a camera negotiating 29.97 against 30 is fine, and warning about it "
        "would train the operator to ignore this")


def test_it_stays_quiet_until_the_rate_has_been_measured():
    """Before the background calibration settles there is no measurement, and
    an unmeasured camera is not a slow one."""
    assert _short(_Thread(30.0, 0.0)) is None


def test_the_threshold_is_where_it_claims_to_be():
    just_over = 30.0 * SHORTFALL_RATIO + 0.2
    just_under = 30.0 * SHORTFALL_RATIO - 0.2
    assert _short(_Thread(30.0, just_over)) is None
    assert _short(_Thread(30.0, just_under)) is not None


def test_a_silently_upgraded_resolution_is_reported():
    """320x240 was asked for and 640x360 arrived, measured on this rig. The
    rate was fine, so ONLY the size check can catch it."""
    s = _short(_Thread(30.0, 30.0,
                       _Cam(req_wh=(320, 240), act_wh=(640, 360))))
    assert s is not None
    assert s.size_differs and not s.rate_differs
    assert "320x240" in s.message() and "640x360" in s.message()


def test_uncompressed_is_only_blamed_when_it_could_be_the_cause():
    """At VGA the bus is not the limit, so naming the format would send the
    operator after the wrong thing."""
    small = Shortfall("c", 30.0, 5.0, (640, 480), (640, 480), "YUY2")
    assert not small.uncompressed_at_high_res
    assert "uncompressed" not in small.message()
    big = Shortfall("c", 30.0, 5.0, (1920, 1080), (1920, 1080), "YUY2")
    assert big.uncompressed_at_high_res
    assert "uncompressed" in big.message()


def test_a_compressed_stream_is_not_blamed():
    s = Shortfall("c", 30.0, 5.0, (1280, 720), (1280, 720), "MJPG")
    assert not s.uncompressed_at_high_res


@pytest.mark.parametrize("thread", [None, _Thread(0.0, 0.0)])
def test_it_never_raises_on_a_camera_that_cannot_answer(thread):
    """Called from the pipeline tick, so a raise here would take the tick
    down for every box."""
    assert _short(thread) is None


def test_it_survives_a_camera_object_that_answers_nothing():
    class _Mute:
        pass

    t = _Thread(30.0, 5.0, cam=_Mute())
    s = _short(t)
    assert s is not None, "the rate shortfall is knowable without the extras"
    assert s.message()


def test_the_drivers_own_complaint_is_reported_before_any_measurement():
    """The early half of the truth.

    The background rate calibration treats a low first reading as "still
    settling" and retries twice at ~12 s, so a CONFIRMED shortfall arrives
    roughly fifty seconds after Connect, measured on this rig. The driver,
    though, said at open that it could not honour the format. Waiting for the
    slow half means the operator starts the session uninformed.
    """
    cam = _Cam(fourcc="YUY2")
    cam.format_warning = ("Camera 0: requested MJPG but the driver is "
                          "capturing in YUY2 at 1280x720")
    s = _short(_Thread(30.0, 0.0, cam))       # 0.0 = not measured yet
    assert s is not None
    assert "YUY2" in s.message()


def test_the_measured_shortfall_supersedes_the_format_complaint():
    """Once the rate is known it is the better answer, it carries the actual
    numbers rather than a prediction."""
    cam = _Cam(fourcc="YUY2")
    cam.format_warning = "Camera 0: requested MJPG but ..."
    s = _short(_Thread(30.0, 9.8, cam))
    msg = s.message()
    assert "9.8" in msg and "30 fps" in msg


def test_a_clean_camera_with_no_complaint_stays_silent():
    cam = _Cam(fourcc="MJPG")
    cam.format_warning = ""
    assert _short(_Thread(30.0, 0.0, cam)) is None


# ── Capability discovery ────────────────────────────────────────────────

from source.video.cameras.probe import Variant, best_for  # noqa: E402

#: The real reading from this rig: the same camera through two doors. Only
#: one of them honours MJPG, and the difference is 3x at 720p and 6x at 1080p.
DSHOW = Variant("dshow", [(640, 480, 30.0), (1280, 720, 10.0),
                          (1920, 1080, 5.0)])
MSMF = Variant("msmf", [(640, 480, 30.0), (1280, 720, 30.0),
                        (1920, 1080, 29.2)])


def test_the_requested_size_is_served_by_whichever_backend_is_fastest():
    """The whole point: 720p exists on both doors, and one is 3x the other."""
    backend, mode = best_for([DSHOW, MSMF], 1280, 720, 30)
    assert backend == "msmf"
    assert mode == (1280, 720, 30.0)


def test_a_tie_stays_on_the_platform_default():
    """At VGA both doors deliver 30. Moving the rig off its default backend
    for no gain is a change with only risk in it."""
    backend, _mode = best_for([DSHOW, MSMF], 640, 480, 30)
    assert backend == "dshow", "an even contest must not move the rig"


def test_an_unavailable_size_falls_back_to_the_largest_that_holds_the_rate():
    """A session that needs 30 fps needs it more than it needs pixels."""
    backend, mode = best_for([DSHOW, MSMF], 2560, 1440, 30)
    assert mode[2] >= 24.0
    assert (mode[0], mode[1]) == (1920, 1080)
    assert backend == "msmf"


def test_when_nothing_meets_the_rate_the_fastest_mode_wins():
    slow = Variant("dshow", [(1920, 1080, 5.0), (1280, 720, 10.0)])
    _backend, mode = best_for([slow], 3840, 2160, 30)
    assert mode == (1280, 720, 10.0)


def test_no_measurable_mode_returns_nothing_rather_than_guessing():
    assert best_for([], 640, 480, 30) is None
    assert best_for([Variant("dshow", []), Variant("msmf", None)],
                    640, 480, 30) is None


def test_a_machine_vision_camera_is_asked_once_not_probed_per_backend():
    """There is one SDK per scientific camera and its answer already accounts
    for exposure, ROI and link speed. Probing 'backends' for it would measure
    the same door twice."""
    import source.video.cameras.probe as cap

    calls = []

    def _fake_probe(camera_id, backend, target_fps, report=None,
                    pixel_format="mjpeg"):
        calls.append(backend)

        class _R:
            modes = [(2048, 1536, 60.0)]
            error = None
        return _R()

    orig = cap.probe_camera
    cap.probe_camera = _fake_probe
    try:
        variants = cap.probe_all("cam", family="spinnaker", target_fps=60)
    finally:
        cap.probe_camera = orig
    assert calls == ["spinnaker"], f"probed {calls}"
    assert len(variants) == 1 and variants[0].backend == "spinnaker"


def test_a_closing_camera_is_not_a_size_mismatch():
    """A camera mid-open or mid-close reports 0x0. Treating that as a
    mismatch banners every shutdown as a fault, which is how a warning
    channel becomes noise."""
    s = Shortfall("c", 30.0, 30.0, (640, 480), (0, 0))
    assert not s.size_differs
    assert s.message() == ""


def test_a_backend_that_cannot_report_its_format_is_not_accused():
    """Media Foundation never reports FOURCC. Unknown is not uncompressed,
    and guessing produces a false warning on a backend that is in fact
    delivering 2.6x what the other one manages."""
    s = Shortfall("c", 30.0, 26.2, (1280, 720), (1280, 720), pixel_format="")
    assert not s.uncompressed_at_high_res
    assert "uncompressed" not in s.message()


# ── Machine-vision cameras ──────────────────────────────────────────────
#
# A Spinnaker or Ximea camera does not need measuring: the SDK already
# reports the achievable rate under the current exposure, ROI and link
# budget. That reading was being taken and then used only to print a line in
# a session header, never compared against what was asked for.


class _SciCam:
    """A machine-vision camera: no FOURCC, no requested-size attributes, but
    an authoritative ``capable_fps`` from the SDK."""

    def __init__(self, capable, width=2048, height=1536):
        self.capable_fps = capable
        self._w, self._h = width, height

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h


def test_a_scientific_camera_reports_a_shortfall_without_being_measured():
    """FLIR: if the resulting rate is below the requested one, the exposure
    time is longer than the frame time. The camera knows that immediately,
    waiting ~50 s to measure what the SDK will simply tell us is a choice to
    let the operator start a session uninformed."""
    s = _short(_Thread(60.0, 0.0, _SciCam(capable=18.5)))
    assert s is not None
    assert s.capability_short
    msg = s.message()
    assert "60 fps" in msg and "18.5" in msg
    assert "exposure" in msg


def test_a_scientific_camera_that_can_meet_the_rate_says_nothing():
    assert _short(_Thread(60.0, 0.0, _SciCam(capable=60.0))) is None
    assert _short(_Thread(60.0, 0.0, _SciCam(capable=59.9))) is None


def test_the_measured_rate_still_wins_once_it_exists():
    """The SDK's number is a prediction; delivered frames are the fact."""
    s = _short(_Thread(60.0, 20.0, _SciCam(capable=60.0)))
    assert s is not None and s.rate_differs
    assert "20.0" in s.message()


def test_a_uvc_camera_has_no_capability_reading_and_is_unaffected():
    """``capable_fps`` is 0.0 on every UVC device, and 0.0 must never be read
    as "capable of nothing"."""
    s = _short(_Thread(30.0, 30.0, _Cam(fourcc="MJPG")))
    assert s is None
