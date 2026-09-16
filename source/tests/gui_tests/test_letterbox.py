"""The letterbox inverse has to be exact before anything is wired to it.

Every keypoint the model returns travels back through ``to_source`` before a
zone decides membership, a trigger fires, or a coordinate is pushed to the
MCU. A quiet error here does not crash anything, it moves the animal, and the
session records a trigger that fired in the wrong place. So the mapping is
property-tested over many random geometries rather than a couple of examples.

No hypothesis in this environment, so the randomness is seeded: the same cases
run every time, and a failure names a geometry that can be reproduced.
"""
import random

import numpy as np
import pytest

from source.video.tracking.letterbox import Letterbox, canonical_shape

# Sizes a rig plausibly produces: ROI crops off 640x480 up to 4K sensors,
# including the awkward odd numbers a hand-drawn rectangle gives.
_SIZES = [17, 32, 63, 64, 100, 189, 229, 256, 320, 480, 512, 640, 721, 1080,
          1280, 1920, 2160]


def _cases(n=400, seed=20260819):
    rng = random.Random(seed)
    for _ in range(n):
        yield (rng.choice(_SIZES), rng.choice(_SIZES),
               rng.choice(_SIZES), rng.choice(_SIZES))


class TestTheInverseIsExact:
    def test_round_trip_returns_the_point_it_started_from(self):
        rng = random.Random(4242)
        for sw, sh, dw, dh in _cases():
            lb = Letterbox.fit(sw, sh, dw, dh)
            for _ in range(3):
                x, y = rng.uniform(0, sw), rng.uniform(0, sh)
                mx, my = lb.to_model(x, y)
                bx, by = lb.to_source(mx, my)
                assert bx == pytest.approx(x, abs=1e-6), (sw, sh, dw, dh)
                assert by == pytest.approx(y, abs=1e-6), (sw, sh, dw, dh)

    def test_the_corners_land_inside_the_canonical_frame(self):
        """A source corner must not map outside the input the model sees."""
        for sw, sh, dw, dh in _cases():
            lb = Letterbox.fit(sw, sh, dw, dh)
            for x, y in ((0, 0), (sw, 0), (0, sh), (sw, sh)):
                mx, my = lb.to_model(x, y)
                assert -0.5 <= mx <= dw + 0.5, (sw, sh, dw, dh, mx)
                assert -0.5 <= my <= dh + 0.5, (sw, sh, dw, dh, my)

    def test_the_scale_is_uniform_so_the_animal_keeps_its_proportions(self):
        """A stretched animal is one the model was never trained on."""
        for sw, sh, dw, dh in _cases():
            lb = Letterbox.fit(sw, sh, dw, dh)
            far_x, _ = lb.to_model(100.0, 0.0)
            _, far_y = lb.to_model(0.0, 100.0)
            assert (far_x - lb.pad_x) == pytest.approx(far_y - lb.pad_y,
                                                       abs=1e-9)


class TestTheImageMatchesTheMapping:
    def test_padded_output_is_exactly_the_canonical_shape(self):
        for sw, sh, dw, dh in _cases(120, seed=7):
            lb = Letterbox.fit(sw, sh, dw, dh)
            out = lb.apply(np.zeros((sh, sw, 3), np.uint8))
            assert out.shape[:2] == (dh, dw), (sw, sh, dw, dh, out.shape)

    def test_a_grayscale_frame_stays_grayscale(self):
        lb = Letterbox.fit(100, 50, 256, 256)
        out = lb.apply(np.zeros((50, 100), np.uint8))
        assert out.shape == (256, 256)

    def test_the_padding_is_padding_not_image(self):
        """Content must sit inside the pad, or the mapping lies about where."""
        lb = Letterbox.fit(100, 50, 200, 200)
        out = lb.apply(np.full((50, 100), 255, np.uint8))
        assert out[0, 0] == 0 and out[-1, -1] == 0
        assert out[lb.dst_h // 2, lb.dst_w // 2] == 255

    def test_a_landmark_lands_where_to_model_says_it_will(self):
        """The image transform and the coordinate transform must agree."""
        src = np.zeros((120, 400), np.uint8)
        src[60, 300] = 255
        lb = Letterbox.fit(400, 120, 256, 256)
        out = lb.apply(src)
        mx, my = lb.to_model(300, 60)
        window = out[int(my) - 2:int(my) + 3, int(mx) - 2:int(mx) + 3]
        assert window.max() > 0, "the bright pixel is not where the map says"


class TestTheCommonRigPaysNothing:
    def test_same_shape_is_identity(self):
        lb = Letterbox.fit(640, 480, 640, 480)
        assert lb.is_identity
        assert lb.to_source(*lb.to_model(11.0, 7.0)) == (11.0, 7.0)

    def test_identity_returns_the_very_same_array(self):
        lb = Letterbox.fit(64, 64, 64, 64)
        frame = np.zeros((64, 64, 3), np.uint8)
        assert lb.apply(frame) is frame

    def test_identity_returns_the_very_same_pose(self):
        lb = Letterbox.fit(64, 64, 64, 64)
        pose = {"nose": [1.0, 2.0, 0.9]}
        assert lb.pose_to_source(pose) is pose


class TestPoseMapping:
    def test_confidence_survives_the_trip(self):
        lb = Letterbox.fit(100, 100, 200, 200)
        out = lb.pose_to_source({"nose": [10.0, 20.0, 0.75]})
        assert out["nose"][2] == 0.75

    def test_an_undetected_part_stays_undetected(self):
        """None is 'not found', not a coordinate to be mapped to (0, 0)."""
        lb = Letterbox.fit(100, 100, 200, 200)
        assert lb.pose_to_source({"nose": None})["nose"] is None

    def test_a_keypoint_found_in_the_padding_is_not_the_animal(self):
        """The pad is black filler; a detection there holds no picture behind
        it, and reporting it would send a coordinate outside the box's frame
        to the zone lookup and the MCU."""
        lb = Letterbox.fit(200, 200, 400, 120)      # pads left and right
        assert lb.pad_x > 0
        in_pad_x = lb.pad_x / 2.0                   # squarely inside the fill
        assert lb.pose_to_source({"nose": [in_pad_x, 60.0, 0.99]})["nose"] is None

    def test_a_keypoint_on_the_frame_edge_is_still_believed(self):
        """Sub-pixel localisation at the edge must not be mistaken for pad."""
        lb = Letterbox.fit(200, 200, 400, 120)
        mx, my = lb.to_model(199.9, 0.1)
        assert lb.pose_to_source({"nose": [mx, my, 0.9]})["nose"] is not None

    def test_a_keypoint_maps_back_to_where_it_was(self):
        lb = Letterbox.fit(100, 40, 256, 256)
        mx, my = lb.to_model(30.0, 12.0)
        out = lb.pose_to_source({"nose": [mx, my, 0.5]})
        assert out["nose"][0] == pytest.approx(30.0, abs=1e-6)
        assert out["nose"][1] == pytest.approx(12.0, abs=1e-6)


class TestCanonicalShape:
    def test_nothing_is_upscaled_past_its_own_resolution(self):
        assert canonical_shape([(640, 480), (320, 240), (500, 500)]) == (640, 500)

    def test_stride_rounds_up_because_the_network_downsamples(self):
        assert canonical_shape([(100, 100)], stride=32) == (128, 128)

    def test_an_exact_multiple_is_left_alone(self):
        assert canonical_shape([(128, 64)], stride=32) == (128, 64)

    def test_no_boxes_means_no_canonical_shape(self):
        assert canonical_shape([]) is None


class TestThroughTheSink:
    """The whole chain, on a mix of box shapes.

    Boxes must key the model on the LETTERBOXED size, not their own crop: keyed
    on the crop, ``get_or_create`` evicts the previous model on every key miss,
    so N boxes mean N builds, one surviving handle, and every box inferring on
    a model built for some other box's shape. On the DLC TensorFlow engines the frame
    could not even be fed, because ``init_inference`` compiles the input
    placeholder at the probe's size.
    """

    @staticmethod
    def _sink():
        import datetime

        from source.video.framebus.pose_sink import PoseSink
        from source.video.tracking.inference import ModelHandle

        class FindBrightest:
            """Stands in for a model: reports the brightest pixel shown."""
            is_initialized = True

            def __init__(self):
                self.batches = []
                self.body_parts = ["nose"]

            def get_body_parts(self):
                return ["nose"]

            def predict(self, f):
                g = f[:, :, 0] if f.ndim == 3 else f
                y, x = np.unravel_index(int(np.argmax(g)), g.shape)
                return {"nose": [float(x), float(y), 0.99]}

            def predict_batch(self, frames):
                self.batches.append([f.shape[:2] for f in frames])
                return [self.predict(f) for f in frames]

        class FakeBackend:
            def __init__(self):
                self.keys = []
                self.model = FindBrightest()

            def get_or_create(self, key, **kw):
                self.keys.append(key)
                return ModelHandle(key=key, model=self.model)

            def submit_batch(self, handle, items, on_done):
                poses = handle.model.predict_batch([f for _, f in items])
                for bid, pose in zip([b for b, _ in items], poses):
                    on_done(bid, pose)
                return len(items)

            def shutdown(self, *a, **k):
                pass

        sink = PoseSink(backend=FakeBackend())
        return sink, datetime

    def _run(self, truth):
        sink, datetime = self._sink()
        from source.video.framebus.types import BoxFrame
        got = {}
        sink._on_result.append(
            lambda sid, cfid, arr, *a, **k: got.__setitem__(sid, arr))
        for sid in truth:
            sink.enable_for_box(sid)
        sink.configure_model(tracker_type="dlc", model_path="models/none",
                             probe_frame=np.zeros((240, 320, 3), np.uint8),
                             resize_factor=1.0, body_parts=["nose"],
                             confidence=0.5)
        frames = []
        for sid, (w, h, mx, my) in truth.items():
            img = np.zeros((h, w, 3), np.uint8)
            img[my, mx] = 255
            frames.append((sid, BoxFrame(
                image=img, setup_id=sid, cam_frame_id=1, camera_id=0,
                capture_host_ns=1, capture_wall=datetime.datetime.now(),
                is_shared_camera=True, poll_host_ns=1)))
        sink._dispatch_batch(frames)
        return sink, got

    def test_differently_cropped_boxes_share_one_model(self):
        truth = {1: (320, 240, 200, 100), 2: (160, 200, 40, 150)}
        sink, _ = self._run(truth)
        assert len(sink._backend.keys) == 1, (
            "a second key means the first box's model was evicted")

    def test_they_ride_in_one_batch_not_one_per_shape(self):
        truth = {1: (320, 240, 200, 100), 2: (160, 200, 40, 150),
                 3: (300, 90, 250, 45)}
        sink, _ = self._run(truth)
        batches = sink._backend.model.batches
        assert len(batches[-1]) == 3, batches
        assert len(set(batches[-1])) == 1, "the batch is not one shape"

    def test_every_box_gets_coordinates_in_its_own_pixels(self):
        """The point of the inverse: box 2's answer is in box 2's frame."""
        truth = {1: (320, 240, 200, 100), 2: (160, 200, 40, 150),
                 3: (300, 90, 250, 45)}
        _, got = self._run(truth)
        for sid, (_w, _h, mx, my) in truth.items():
            x, y, _c = got[sid][0]
            # Half a pixel of grid rounding is the documented cost of fitting
            # an integer number of pixels; anything more is a mapping error.
            assert x == pytest.approx(mx, abs=0.5), (sid, x, mx)
            assert y == pytest.approx(my, abs=0.5), (sid, y, my)


def test_a_degenerate_size_is_refused_not_guessed():
    """A zero-width crop is a bug upstream; silently 'fixing' it hides it."""
    with pytest.raises(ValueError):
        Letterbox.fit(0, 100, 256, 256)
