"""The window has to find the animal, keep it, and give back honest pixels.

Three separable jobs, tested separately: the cut and its inverse (pure
geometry), the steering (which is where a plain mean instead of a confident
mean walks the window off the animal), and recovery (which must cost a handful
of inferences, not a sweep, and must not let one lost box stall the others).

As with the letterbox, the inverse is the part that reaches zones, triggers and
the MCU, so it is tested over many geometries rather than a couple of examples.
"""
import random

import numpy as np

from source.video.tracking.crop_tracker import (
    BoxCropState,
    CropWindow,
    confident_centroid,
    count_confident,
    crop_at,
    scan_points,
)

_FRAMES = [(64, 48), (160, 120), (240, 178), (360, 202), (640, 480), (1920, 1080)]
_WINDOWS = [(32, 32), (224, 176), (128, 96), (256, 256)]


def _frame(w, h, mark=None):
    img = np.zeros((h, w, 3), np.uint8)
    if mark is not None:
        img[mark[1], mark[0]] = 255
    return img


class TestTheCutAndItsInverse:
    def test_the_cut_is_always_the_requested_size(self):
        for fw, fh in _FRAMES:
            for w, h in _WINDOWS:
                cut, _ = crop_at(_frame(fw, fh), fw / 2, fh / 2, w, h)
                assert cut.shape[:2] == (h, w), (fw, fh, w, h, cut.shape)

    def test_the_window_stays_inside_the_frame_when_it_fits(self):
        rng = random.Random(11)
        for fw, fh in _FRAMES:
            for w, h in _WINDOWS:
                if w > fw or h > fh:
                    continue
                for _ in range(5):
                    cx, cy = rng.uniform(-50, fw + 50), rng.uniform(-50, fh + 50)
                    _, win = crop_at(_frame(fw, fh), cx, cy, w, h)
                    assert 0 <= win.x0 <= fw - w, (fw, w, cx, win.x0)
                    assert 0 <= win.y0 <= fh - h, (fh, h, cy, win.y0)

    def test_a_pixel_survives_the_round_trip_exactly(self):
        """The inverse is an offset, so it is exact, no rounding excuse."""
        rng = random.Random(5)
        for fw, fh in _FRAMES:
            for w, h in _WINDOWS:
                if w > fw or h > fh:
                    continue
                mx, my = rng.randrange(fw), rng.randrange(fh)
                cut, win = crop_at(_frame(fw, fh, (mx, my)), mx, my, w, h)
                found = np.argwhere(cut[:, :, 0] == 255)
                assert len(found) == 1, "the marker left the window"
                cy_in, cx_in = found[0]
                assert win.to_source(float(cx_in), float(cy_in)) == (mx, my)

    def test_a_frame_smaller_than_the_window_is_padded_not_refused(self):
        cut, win = crop_at(_frame(64, 48), 32, 24, 224, 176)
        assert cut.shape[:2] == (176, 224)
        assert (win.x0, win.y0) == (0, 0), "padding must not move the offset"

    def test_pose_mapping_keeps_confidence_and_undetected_parts(self):
        win = CropWindow(x0=100, y0=40, w=224, h=176)
        out = win.pose_to_source({"nose": [10.0, 20.0, 0.9], "tail": None})
        assert out["nose"] == [110.0, 60.0, 0.9]
        assert out["tail"] is None


class TestSteering:
    def test_only_confident_points_steer(self):
        """A below-threshold keypoint is not a weak opinion; it is none."""
        pose = {"a": [0.0, 0.0, 0.9], "b": [10.0, 10.0, 0.9],
                "junk": [1000.0, 1000.0, 0.01]}
        assert confident_centroid(pose, 0.2, 2) == (5.0, 5.0)

    def test_a_plain_mean_would_have_walked_off(self):
        """Pins the bug this exists to avoid, not just the happy path."""
        pose = {"a": [0.0, 0.0, 0.9], "b": [10.0, 10.0, 0.9],
                "junk": [1000.0, 1000.0, 0.01]}
        naive = np.mean([[v[0], v[1]] for v in pose.values()], axis=0)
        assert naive[0] > 300, "fixture no longer demonstrates the failure"
        assert confident_centroid(pose, 0.2, 2)[0] == 5.0

    def test_nan_coordinates_are_excluded(self):
        pose = {"a": [np.nan, np.nan, 0.99], "b": [4.0, 6.0, 0.9],
                "c": [6.0, 10.0, 0.9]}
        assert confident_centroid(pose, 0.2, 2) == (5.0, 8.0)

    def test_too_few_confident_points_is_no_answer_at_all(self):
        pose = {"a": [1.0, 1.0, 0.9], "b": [2.0, 2.0, 0.01]}
        assert confident_centroid(pose, 0.2, good_min=3) is None

    def test_a_lost_frame_keeps_the_last_position(self):
        """Resetting to the frame centre on one bad frame would throw away a
        fix that is probably still nearly right."""
        st = BoxCropState(224, 176)
        st.update({"a": [10.0, 10.0, 0.9], "b": [20.0, 20.0, 0.9],
                   "c": [30.0, 30.0, 0.9]})
        assert (st.cx, st.cy) == (20.0, 20.0)
        assert st.update({"a": None, "b": None, "c": None}) is False
        assert (st.cx, st.cy) == (20.0, 20.0)
        assert st.lost_frames == 1

    def test_a_motion_hint_wins_over_the_last_centroid(self):
        st = BoxCropState(224, 176)
        st.cx, st.cy = 10.0, 10.0
        assert st.centre_for(640, 480, hint=(300.0, 200.0)) == (300.0, 200.0)

    def test_with_no_fix_and_no_hint_it_starts_in_the_middle(self):
        assert BoxCropState(224, 176).centre_for(640, 480) == (320.0, 240.0)


class TestRecovery:
    """Lost windows look for the animal one window per frame.

    The published loop sweeps the grid synchronously, which suits a
    single-animal script. Here the frames keep arriving, so trying one more
    window per frame recovers in a handful of frames, costs no extra inference,
    and cannot stall the boxes that are still tracking.
    """

    def test_the_grid_covers_the_frame(self):
        for fw, fh in _FRAMES:
            pts = scan_points(fw, fh, 224, 176)
            assert pts, (fw, fh)
            assert all(np.isfinite(p).all() for p in pts)

    def test_a_frame_barely_larger_than_the_window_is_a_couple_of_looks(self):
        assert len(scan_points(360, 202, 224, 176)) <= 3

    def test_one_bad_frame_does_not_start_a_search(self):
        """Noise is not a lost window."""
        st = BoxCropState(64, 64, good_min=2, lost_before_scan=3)
        st.cx, st.cy = 100.0, 100.0
        st.update({"a": None})
        assert st.searching is False
        assert st.centre_for(640, 480) == (100.0, 100.0)

    def test_a_persistently_lost_window_starts_looking(self):
        st = BoxCropState(64, 64, good_min=2, lost_before_scan=3)
        st.cx, st.cy = 100.0, 100.0
        for _ in range(3):
            st.update({"a": None})
        assert st.searching is True
        assert st.centre_for(640, 480) != (100.0, 100.0)

    def test_the_search_advances_every_frame_and_wraps(self):
        st = BoxCropState(64, 64, good_min=2, lost_before_scan=1)
        st.update({"a": None})
        grid = scan_points(320, 240, 64, 64)
        seen = [st.centre_for(320, 240) for _ in range(len(grid))]
        assert len(set(seen)) == len(grid), "the search skipped or repeated"
        assert st.centre_for(320, 240) == seen[0], "it did not wrap around"

    def test_a_small_frame_recovers_in_a_handful_of_frames(self):
        """On the published clip the grid is 2-3 windows, so a lost box is
        back within ~150 ms at 20 fps rather than stalling anything."""
        assert len(scan_points(360, 202, 224, 176)) <= 3

    def test_finding_the_animal_again_stops_the_search(self):
        st = BoxCropState(64, 64, good_min=2, lost_before_scan=1)
        st.update({"a": None})
        assert st.searching
        assert st.update({"a": [5.0, 5.0, 0.9], "b": [7.0, 7.0, 0.9]}) is True
        assert st.searching is False
        assert st.centre_for(640, 480) == (6.0, 6.0)

    def test_recovery_can_be_switched_off(self):
        """A rig that would rather see empty poses than a wandering window."""
        st = BoxCropState(64, 64, good_min=2, lost_before_scan=1, reacquire=False)
        st.cx, st.cy = 50.0, 50.0
        for _ in range(10):
            st.update({"a": None})
        assert st.searching is False
        assert st.centre_for(640, 480) == (50.0, 50.0)

    def test_the_search_is_counted_for_the_health_panel(self):
        st = BoxCropState(64, 64, good_min=2, lost_before_scan=1)
        st.update({"a": None})
        for _ in range(4):
            st.centre_for(320, 240)
        assert st.scans == 4

    def test_counting_confidence(self):
        pose = {"a": [0, 0, 0.9], "b": [0, 0, 0.1], "c": None}
        assert count_confident(pose, 0.2) == 1
