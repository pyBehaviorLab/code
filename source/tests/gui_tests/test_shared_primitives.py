"""One timing policy for FPS probes; the three containment tests stay distinct.

Two different outcomes from the same audit finding, because "looks the same"
and "means the same" are different questions.
"""
import ast
import re
from pathlib import Path

import numpy as np

from source.video.cameras.base import (
    FPS_MIN_FRAMES,
    SDK_FPS_DRAIN_S,
    SDK_FPS_MEASURE_S,
    UVC_FPS_DRAIN_S,
    UVC_FPS_MEASURE_S,
    measure_delivered_fps,
)
from source.video.zones.geometry import point_in_polygon, point_inside_px
from source.video.zones.schema import Zone

SOURCE = Path(__file__).resolve().parents[2]
BACKENDS = ("opencv", "spinnaker", "ximea")


# ---- FPS: one settle-then-count loop -------------------------------------

def test_counts_frames_over_the_window():
    counted = {"drain": 0}

    def drain():
        counted["drain"] += 1

    fps = measure_delivered_fps(drain, lambda: 1,
                                drain_s=0.02, measure_s=0.10)
    assert counted["drain"] > 0, "settle phase never ran"
    assert fps > 0


def test_too_few_frames_is_zero_not_a_wild_rate():
    """One frame in a window says nothing; dividing by a tiny elapsed would
    invent a huge number and the picker would offer a mode that cannot run."""
    calls = {"n": 0}

    def once():
        calls["n"] += 1
        return 1 if calls["n"] <= FPS_MIN_FRAMES - 1 else 0

    assert measure_delivered_fps(lambda: None, once,
                                 drain_s=0.0, measure_s=0.05) == 0.0


def test_a_camera_that_stops_mid_window_still_reports_what_it_delivered():
    """Aborting is information, not an error, the frames counted before the
    device went away are real, and must not be thrown away or re-raised."""
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] > 5:
            raise OSError("device went away")
        return 1

    fps = measure_delivered_fps(lambda: None, flaky,
                                drain_s=0.0, measure_s=5.0)
    assert state["n"] == 6, "the loop did not stop at the failure"
    assert fps > 0, "5 delivered frames were discarded"
    # And it returned immediately rather than sitting out the 5 s window.


def test_a_drain_that_raises_does_not_abort_the_measurement():
    fps = measure_delivered_fps(
        lambda: (_ for _ in ()).throw(OSError("boom")), lambda: 1,
        drain_s=0.05, measure_s=0.10)
    assert fps > 0


def _hardcoded_windows(path):
    """Literal seconds compared against a monotonic clock, the pattern the
    per-backend copies used before there was a shared policy."""
    src = path.read_text(encoding="utf-8")
    return re.findall(r"monotonic\(\)\s*[-+][^<\n]*<\s*([0-9]+\.[0-9]+)", src)


def test_no_backend_hardcodes_its_own_measurement_window():
    """They each had one, and they DIFFERED, OpenCV settled 2.0s and counted
    over 3.0s, the SDK backends 0.5s and 2.0s, so the rates the three
    produced were not comparable."""
    offenders = {}
    for name in BACKENDS:
        found = _hardcoded_windows(SOURCE / "video" / "cameras" / f"{name}.py")
        if found:
            offenders[name] = found
    assert not offenders, (
        f"hardcoded FPS windows are back: {offenders}. Use the named "
        "constants in cameras/base.py")


def test_every_backend_measures_through_the_shared_helper():
    for name in BACKENDS:
        src = (SOURCE / "video" / "cameras" / f"{name}.py").read_text(
            encoding="utf-8")
        if "def measure_fps_at" not in src:
            continue
        assert "measure_delivered_fps" in src, (
            f"{name} grew its own measurement loop again")


def test_the_two_window_profiles_are_named_and_distinct():
    """The difference is deliberate, a UVC camera renegotiates its mode and
    re-converges auto-exposure on every open; an SDK camera does not."""
    assert UVC_FPS_DRAIN_S > SDK_FPS_DRAIN_S
    assert UVC_FPS_MEASURE_S > SDK_FPS_MEASURE_S


# ---- containment: three tests, on purpose --------------------------------

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_interior_and_exterior_agree_across_all_three():
    z = Zone.from_dict({"name": "S", "points": SQUARE, "type": "polygon"})
    pts_int = np.array([[int(x), int(y)] for x, y in SQUARE], dtype=np.int32)
    for x, y in [(5.0, 5.0), (1.0, 9.0)]:
        assert z.contains(x, y)
        assert point_in_polygon(x, y, SQUARE)
        assert point_inside_px(pts_int, x, y)
    for x, y in [(-1.0, 5.0), (20.0, 20.0)]:
        assert not z.contains(x, y)
        assert not point_in_polygon(x, y, SQUARE)
        assert not point_inside_px(pts_int, x, y)


def test_the_overlay_test_is_boundary_inclusive():
    """Documented divergence: the overlay highlights a point exactly on an
    edge, Shapely's contains does not. Sub-pixel, but real, and written down
    rather than left to be rediscovered."""
    pts_int = np.array([[int(x), int(y)] for x, y in SQUARE], dtype=np.int32)
    assert point_inside_px(pts_int, 0.0, 5.0) is True


def test_overlay_test_survives_a_degenerate_polygon():
    assert point_inside_px(None, 1.0, 1.0) is False
    assert point_inside_px(np.array([[0, 0], [1, 1]], dtype=np.int32),
                           0.5, 0.5) is False


def _cv2_containment_sites():
    hits = []
    for path in SOURCE.rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        if path.name == "geometry.py":
            continue          # the one wrapper is allowed to call cv2
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pointPolygonTest"):
                hits.append(f"{path.relative_to(SOURCE)}:{node.lineno}")
    return hits


def test_display_containment_goes_through_the_one_wrapper():
    """Two call sites had their own copy, each re-deciding the boundary rule."""
    assert not _cv2_containment_sites(), (
        "call point_inside_px instead:\n  "
        + "\n  ".join(_cv2_containment_sites()))
