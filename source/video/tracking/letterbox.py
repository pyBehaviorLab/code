"""Fit a box's frame to one canonical model input, reversibly.

Boxes on a shared camera are cropped to whatever rectangle the operator drew,
so their frames differ in size. The pose model is keyed on ``(path, w, h,
resize, type, sig)`` and the backend holds one model at a time, so two boxes of
different sizes are two keys: initialising the second **evicts** the first, and
every box then infers on a model built for some other box's shape. On the
TensorFlow DLC engines that is not merely wasteful, the session's input
placeholder is built as ``[1, H, W, 3]`` at ``init_inference``, so a frame of
another size cannot be fed to it at all.

The fix is to make every box present the same shape: scale by one factor and
pad the remainder. Padding, never stretching, a stretched frame changes the
animal's proportions, and a pose model asked to find a shape it was never
trained on fails quietly, with confident keypoints in the wrong places.

Everything downstream, zone membership, triggers, the coordinate push to the
MCU, the recorded ``_video_data.txt``, consumes keypoints in the box's own
frame pixels. So the transform is only half the job: its inverse has to be
exact, and it is the part written and tested first.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Letterbox:
    """Maps one box's frame onto a canonical model input, and back.

    ``scale`` is uniform by construction, so a point's round trip through
    :meth:`to_model` and :meth:`to_source` is exact. What rounding costs is
    the pixel grid, not the mapping: the resized image is at most half a pixel
    from where the nominal scale says it should be, because a frame can only
    be an integer number of pixels wide.
    """

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    pad_x: int
    pad_y: int

    @classmethod
    def fit(cls, src_w: int, src_h: int, dst_w: int, dst_h: int) -> "Letterbox":
        """Largest uniform scale that fits the source inside the target."""
        if min(src_w, src_h, dst_w, dst_h) <= 0:
            raise ValueError(
                f"letterbox needs positive sizes, got src={src_w}x{src_h} "
                f"dst={dst_w}x{dst_h}")
        scale = min(dst_w / float(src_w), dst_h / float(src_h))
        inner_w = max(1, min(dst_w, round(src_w * scale)))
        inner_h = max(1, min(dst_h, round(src_h * scale)))
        # Centred, so a model with a central bias is not handed an animal that
        # sits systematically to one side of every frame.
        return cls(src_w=int(src_w), src_h=int(src_h),
                   dst_w=int(dst_w), dst_h=int(dst_h), scale=float(scale),
                   pad_x=(dst_w - inner_w) // 2, pad_y=(dst_h - inner_h) // 2)

    @property
    def is_identity(self) -> bool:
        """True when the frame already is the canonical shape.

        The single-camera-per-box rig is the common case and pays nothing:
        callers skip the resize and the copy entirely.
        """
        return (self.src_w == self.dst_w and self.src_h == self.dst_h
                and self.pad_x == 0 and self.pad_y == 0)

    @property
    def inner_size(self) -> Tuple[int, int]:
        """``(w, h)`` of the real image inside the padding."""
        return (self.dst_w - 2 * self.pad_x, self.dst_h - 2 * self.pad_y)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` scaled and padded to the canonical shape.

        Shrinking uses ``INTER_AREA``, which averages the pixels it discards
        rather than sampling one of them, on a small animal against a plain
        arena floor, point sampling is what drops the animal out of the frame.
        """
        if self.is_identity:
            return frame
        inner_w, inner_h = self.inner_size
        interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (inner_w, inner_h), interpolation=interp)
        right = self.dst_w - inner_w - self.pad_x
        bottom = self.dst_h - inner_h - self.pad_y
        if self.pad_x or self.pad_y or right or bottom:
            resized = cv2.copyMakeBorder(
                resized, self.pad_y, bottom, self.pad_x, right,
                cv2.BORDER_CONSTANT, value=0)
        return resized

    def to_model(self, x: float, y: float) -> Tuple[float, float]:
        """Source-frame pixels → canonical-input pixels."""
        return (x * self.scale + self.pad_x, y * self.scale + self.pad_y)

    def to_source(self, x: float, y: float) -> Tuple[float, float]:
        """Canonical-input pixels → source-frame pixels.

        This is the direction that matters in production: it is what every
        keypoint the model returns has to travel through before a zone, a
        trigger or the MCU sees it.
        """
        return ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)

    def pose_to_source(self, pose: dict) -> dict:
        """Map a whole ``{part: [x, y, conf]}`` result back to source pixels.

        Identity transforms return the input untouched, so the common rig adds
        no allocation. A part reported as ``None`` (not detected) stays
        ``None``: there is no coordinate to map.
        """
        if self.is_identity or not pose:
            return pose
        out = {}
        for part, value in pose.items():
            if value is None or len(value) < 2:
                out[part] = value
                continue
            x, y = self.to_source(float(value[0]), float(value[1]))
            out[part] = (None if self._in_the_padding(x, y)
                         else [x, y, *value[2:]])
        return out

    #: How far outside the frame a keypoint may still be believed, in source
    #: pixels. An animal against the frame edge can be localised a fraction of
    #: a pixel past it; the padding starts much further out than this.
    _EDGE_SLACK_PX = 1.0

    def _in_the_padding(self, x: float, y: float) -> bool:
        """True if a source-space point came from the fill, not the image.

        The padding is black filler, so a keypoint there is not the animal,
        it is the model finding something in a region that holds no picture.
        Reporting it would push a coordinate outside the box's own frame to
        the zone lookup and on to the MCU. Dropping it says the honest thing:
        this part was not found. Clamping to the edge is the tempting
        alternative and the wrong one, it invents a plausible position.
        """
        s = self._EDGE_SLACK_PX
        return not (-s <= x <= self.src_w + s and -s <= y <= self.src_h + s)


def canonical_shape(shapes: Iterable[Tuple[int, int]],
                    stride: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Pick one input shape for a set of box frame shapes ``(w, h)``.

    The widest width and the tallest height, so no box is ever upscaled past
    its own resolution, inventing pixels costs inference time and adds no
    information. When the model states a ``max_stride`` the result is rounded
    up to a multiple of it, because a network that downsamples by that factor
    needs an input it divides evenly.

    Returns ``None`` for an empty set: there is nothing to be canonical about.
    """
    sizes = [(int(w), int(h)) for w, h in shapes if w > 0 and h > 0]
    if not sizes:
        return None
    w = max(s[0] for s in sizes)
    h = max(s[1] for s in sizes)
    if stride and stride > 1:
        w = int(np.ceil(w / stride) * stride)
        h = int(np.ceil(h / stride) * stride)
    return (w, h)
