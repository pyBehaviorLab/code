"""A camera's offered modes are ASKED for, not guessed by trial.

The parser is tested against real output captured from this rig's cameras, so
the fixtures are evidence rather than illustration. Nothing here opens a
device.

Why this exists at all: measuring cannot answer the question. The same camera
at 1920x1080 measured 30.13 fps and then 24.88 fps minutes apart, because
auto-exposure lengthens integration in dimmer light, and the rig stored 1.0 fps
for a mode the camera runs at 30 because the pixel format had silently fallen
back to uncompressed. See docs/dev/camera-modes-and-rates.md.
"""
from __future__ import annotations

import pytest

from source.video.cameras.enumerate_modes import (
    OfferedMode,
    _parse_dshow,
    _parse_v4l2,
    _rates_in_range,
    enumerate_modes,
    rates_for,
    sizes_for,
)

#: Verbatim from ``ffmpeg -f dshow -list_options true`` for camc1aefd1e, the
#: 4K camera whose MJPEG rows the rig had stored as 1.0 fps.
DSHOW_4K = """
[in#0 @ 0] DirectShow video device options (from video devices)
[in#0 @ 0]  Pin "Capture" (alternative pin name "0")
[in#0 @ 0]   vcodec=mjpeg  min s=3840x2160 fps=5 max s=3840x2160 fps=30
[in#0 @ 0]   vcodec=mjpeg  min s=3840x2160 fps=5 max s=3840x2160 fps=30 (pc, center)
[in#0 @ 0]   vcodec=mjpeg  min s=1920x1080 fps=5 max s=1920x1080 fps=30
[in#0 @ 0]   vcodec=mjpeg  min s=2592x1944 fps=5 max s=2592x1944 fps=30
[in#0 @ 0]   vcodec=mjpeg  min s=640x480 fps=5 max s=640x480 fps=30
[in#0 @ 0]   pixel_format=yuyv422  min s=3840x2160 fps=1 max s=3840x2160 fps=1
[in#0 @ 0]   pixel_format=yuyv422  min s=1920x1080 fps=3 max s=1920x1080 fps=3
"""

#: ``v4l2-ctl --list-formats-ext``, which is the only source on any platform
#: that reports the camera's real interval LIST rather than a range.
V4L2_CTL = """
ioctl: VIDIOC_ENUM_FMT
	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.040s (25.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.200s (5.000 fps)
"""


# ── the fault the enumeration exists to fix ──────────────────────────────

def test_the_compressed_and_uncompressed_rows_are_kept_apart():
    """The rig stored 1.0 fps for 4K. That is the UNCOMPRESSED row.

    A UVC driver renegotiates the pixel format whenever the size changes and
    lands back on uncompressed, so a probe that believes it is measuring MJPEG
    can be measuring YUYV. Enumeration reports both, separately, so the mistake
    is not available to make.
    """
    modes = _parse_dshow(DSHOW_4K)
    mjpeg = [m for m in modes if m.size == (3840, 2160) and m.pixel_format == "mjpeg"]
    yuy2 = [m for m in modes if m.size == (3840, 2160) and m.pixel_format == "yuy2"]
    assert mjpeg and mjpeg[0].max_rate == 30.0
    assert yuy2 and yuy2[0].max_rate == 1.0, (
        "1.0 fps at 4K is the uncompressed row, and it is what was stored as "
        "the camera's capability")


def test_sizes_the_fixed_ladder_never_tries_are_found():
    """A trial probe can only find sizes it thinks to try."""
    sizes = sizes_for(_parse_dshow(DSHOW_4K), "mjpeg")
    assert (2592, 1944) in sizes, (
        "this camera offers it and RESOLUTION_LADDER does not contain it")


def test_formats_are_named_the_way_the_rest_of_the_rig_names_them():
    """The Format cell offers mjpeg / yuy2 / h264, so the enumeration must
    speak the same names or nothing downstream can match them up."""
    formats = {m.pixel_format for m in _parse_dshow(DSHOW_4K)}
    assert formats == {"mjpeg", "yuy2"}, formats


# ── rates: exact on Linux, bounded on Windows ────────────────────────────

def test_v4l2_gives_the_cameras_actual_interval_list():
    modes = _parse_v4l2("") or []
    assert modes == []          # ffmpeg's v4l2 lister prints no intervals
    from source.video.cameras.enumerate_modes import _merge
    assert _merge([]) == []


def test_v4l2_ctl_output_is_parsed_as_an_exact_list(monkeypatch):
    import subprocess

    from source.video.cameras import enumerate_modes as em

    class _Proc:
        stdout = V4L2_CTL

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    modes = em._v4l2_ctl_rates("/dev/video0")
    at_640 = rates_for(modes, (640, 480), "mjpeg")
    assert at_640 == (25.0, 30.0), at_640
    at_1080 = rates_for(modes, (1920, 1080), "mjpeg")
    assert at_1080 == (15.0, 30.0), (
        f"{at_1080}: the camera lists exactly these, and 20 is not among them")


def test_a_windows_range_is_filled_from_the_standard_intervals():
    """DirectShow reports bounds, not a list, so the list is reconstructed
    from the standard UVC intervals inside those bounds. It is a narrowing,
    not an answer, and the exact list is confirmed when the camera opens."""
    assert _rates_in_range(15, 30) == (15.0, 20.0, 24.0, 25.0, 30.0)
    assert _rates_in_range(5, 5) == (5.0,), "a single value is not a range"
    assert _rates_in_range(1, 1) == (1.0,)


def test_a_range_keeps_its_own_endpoints_even_when_not_standard():
    """The bounds came from the device, so they are real whatever they are."""
    rates = _rates_in_range(7.0, 22.0)
    assert 7.0 in rates and 22.0 in rates


# ── a non-answer must not be mistaken for an answer ──────────────────────

def test_no_device_gives_nothing():
    assert enumerate_modes("") == []
    assert enumerate_modes(None) == []


def test_a_failure_to_ask_returns_empty_rather_than_raising(monkeypatch):
    """Empty means the question could not be ASKED, so the caller falls back
    to the trial probe. It must never look like 'this camera has no modes'."""
    from source.video.cameras import enumerate_modes as em
    monkeypatch.setattr(em, "_run_ffmpeg",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert em.enumerate_modes("whatever") == []


def test_rates_for_tells_an_unoffered_size_from_a_rateless_one():
    modes = [OfferedMode(640, 480, "mjpeg", (30.0,))]
    assert rates_for(modes, (640, 480)) == (30.0,)
    assert rates_for(modes, (1920, 1080)) == ()
    assert rates_for(modes, (640, 480), "yuy2") == ()


def test_sizes_come_back_largest_first():
    sizes = sizes_for(_parse_dshow(DSHOW_4K), "mjpeg")
    assert sizes == sorted(sizes, key=lambda s: -(s[0] * s[1])), sizes


@pytest.mark.parametrize("junk", ["", "not ffmpeg output at all",
                                  "vcodec=mjpeg min s=AxB fps=x max s=CxD fps=y"])
def test_unparseable_output_yields_nothing_and_does_not_raise(junk):
    assert _parse_dshow(junk) == []


# ── offered and delivered are different facts and must not mix ───────────

import json  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402

from source.video.cameras import calibration_store as store  # noqa: E402

_KEY = "unit-test-offered"


@pytest.fixture
def clean_entry():
    yield _KEY
    p = (pathlib.Path(os.environ.get("LOCALAPPDATA") or pathlib.Path.home())
         / "pybehaviorlab" / "camera_calibrations.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.pop(_KEY, None) is not None:
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")


_OFFERED = [OfferedMode(1920, 1080, "mjpeg", (15.0, 30.0)),
            OfferedMode(640, 480, "mjpeg", (15.0, 25.0, 30.0)),
            OfferedMode(1920, 1080, "yuy2", (3.0,))]


def test_offered_round_trips(clean_entry):
    store.put_offered(_KEY, _OFFERED)
    back = store.get_offered(_KEY)
    assert len(back) == 3
    assert rates_for(back, (640, 480), "mjpeg") == (15.0, 25.0, 30.0)
    assert rates_for(back, (1920, 1080), "yuy2") == (3.0,)


def test_offered_does_not_disturb_the_measurements(clean_entry):
    """One is a property of the camera, the other of the room."""
    store.put(_KEY, "opencv", "t", [[640, 480, 24.9]], {},
              variants={"dshow": [[640, 480, 24.9]]})
    store.put_offered(_KEY, _OFFERED)
    assert store.get_variants(_KEY)["dshow"] == [(640, 480, 24.9)], (
        "enumerating overwrote what the camera was measured delivering")
    assert store.get_offered(_KEY), "and the enumeration is there too"


def test_measuring_does_not_disturb_the_enumeration(clean_entry):
    store.put_offered(_KEY, _OFFERED)
    store.put(_KEY, "opencv", "t", [[640, 480, 24.9]], {},
              variants={"dshow": [[640, 480, 24.9]]})
    assert len(store.get_offered(_KEY)) == 3, (
        "a measurement overwrote the camera's own list of modes")


def test_an_empty_enumeration_is_never_written(clean_entry):
    """Empty means the question could not be ASKED.

    Writing it over a real answer would turn a temporary failure, ffmpeg
    missing or a busy device, into a permanent loss of the camera's modes.
    """
    store.put_offered(_KEY, _OFFERED)
    assert store.put_offered(_KEY, []) is False
    assert len(store.get_offered(_KEY)) == 3


def test_an_unknown_camera_offers_nothing_and_does_not_raise():
    assert store.get_offered("no-such-camera-at-all") == []


def test_a_re_probe_keeps_the_rates_a_camera_was_seen_to_round_away(clean_entry):
    """``put`` used to rebuild the entry and discard every other writer.

    The enumeration and the learned rejections are written by different code
    for different reasons, and both were thrown away by the next Detect.
    """
    store.put(_KEY, "opencv", "t", [[640, 480, 29.9]], {},
              variants={"dshow": [[640, 480, 29.9]]})
    store.note_rate_rejected(_KEY, "dshow", (640, 480), 25, 30.0)
    store.put(_KEY, "opencv", "t", [[640, 480, 29.9]], {},
              variants={"dshow": [[640, 480, 29.9]]})
    assert store.get_rejected_rates(_KEY, "dshow", (640, 480)) == {25}


# ── min == max is a DISCRETE interval, and must not be padded out ────────
#
# DirectShow reports one capability per (format, size) with a min and a max
# frame interval, and the two cases mean different things. Raw
# ``ffmpeg -list_options`` on this rig:
#
#     cam5b4028a9  mjpeg 640x480   min=10       max=60.0002   a RANGE
#     camc1aefd1e  mjpeg 640x480   min=5        max=30        a RANGE
#     cam519bc077  mjpeg 640x480   min=120.101  max=120.101   a SINGLE
#
# cam519bc077 reports min == max on EVERY line: it is a discrete-interval
# device, and its 640x480 MJPEG really is 120 fps and nothing else. Measured
# by opening it: asking for 30 delivered 95. Offering the ladder below such a
# value invents rates the camera does not have.

def test_a_single_reported_interval_is_the_whole_list():
    from source.video.cameras.enumerate_modes import _rates_in_range
    assert _rates_in_range(120.101, 120.101) == (120.101,), (
        "padded a discrete interval out into rates the camera does not have")


def test_a_range_offers_the_standard_rates_inside_it():
    from source.video.cameras.enumerate_modes import _rates_in_range
    assert _rates_in_range(10.0, 60.0002) == (10.0, 15.0, 20.0, 24.0, 25.0,
                                              30.0, 50.0, 60.0)


def test_a_near_miss_endpoint_is_not_listed_twice():
    """Every picker labels a rate as a whole number, so a reported 60.0002
    beside the standard 60.0 rendered as two items both reading "60"."""
    from source.video.cameras.enumerate_modes import _rates_in_range
    rounded = [round(r) for r in _rates_in_range(10.0, 60.0002)]
    assert len(rounded) == len(set(rounded)), "duplicate rates after rounding"


def test_fractional_standard_rates_are_not_offered():
    """7.5 is a real UVC interval, but it reached the operator as "8" and
    would then have been requested as 8."""
    from source.video.cameras.enumerate_modes import _rates_in_range
    assert 7.5 not in _rates_in_range(5.0, 30.0)
    assert 8 not in [round(r) for r in _rates_in_range(5.0, 30.0)]
