"""Follow the animal with a fixed window cut at native resolution.

A model trained on crops is trained at one pixel scale, and scaling a frame to
fit it throws that away, which is why the letterbox, right for a model trained
on whole frames, is wrong here. Cutting a window of exactly the training size
keeps every pixel the size the network learned, and as a side effect gives
every box the same input shape, so one model still serves them all and the
batch stays homogeneous.

The window has to move, because the animal does. It follows the mean of the
keypoints the model was confident about, which is far steadier than any single
landmark, the snout is the fastest-moving part of a mouse and would make the
window jitter. When confidence collapses the window is lost, and the only way
back is to look: a coarse grid of candidate windows, scored by how many
keypoints each yields. That recovery depends on nothing but the model, no
background, no motion history, so it works from a bare camera stream.

Everything downstream reads box-frame pixels, so :meth:`CropWindow.to_source`
is the half that must be exact: it is an offset, never a scale.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Peak confidence at which a keypoint is worth steering by.
DEFAULT_CONF_MIN = 0.20
#: Confident keypoints needed before the frame counts as tracked.
DEFAULT_GOOD_MIN = 3
#: Lost frames tolerated before the window stops following and starts looking.
#: One bad frame is noise; three in a row is a window on the wrong place.
DEFAULT_LOST_BEFORE_SCAN = 3


@dataclass(frozen=True)
class CropWindow:
    """One cut of a frame, and the offset that undoes it."""

    x0: int
    y0: int
    w: int
    h: int

    def to_source(self, x: float, y: float) -> Tuple[float, float]:
        """Crop pixels → box-frame pixels. An offset; there is no scale."""
        return (x + self.x0, y + self.y0)

    def pose_to_source(self, pose: dict) -> dict:
        """Map a whole ``{part: [x, y, conf]}`` result back to the box frame.

        A part reported as ``None`` stays ``None``; there is no coordinate to
        move. Unlike the letterbox there is no padding to reject: a keypoint
        inside the window is inside the image, because the window is a cut of
        it rather than a scaled copy with fill.
        """
        if not pose:
            return pose
        out = {}
        for part, value in pose.items():
            if value is None or len(value) < 2:
                out[part] = value
                continue
            x, y = self.to_source(float(value[0]), float(value[1]))
            out[part] = [x, y, *value[2:]]
        return out


def crop_at(frame: np.ndarray, cx: float, cy: float,
            w: int, h: int) -> Tuple[np.ndarray, CropWindow]:
    """Cut a ``w`` x ``h`` window centred as near ``(cx, cy)`` as it fits.

    The clamp keeps the window inside the frame, so an animal at the edge sits
    off-centre in its crop rather than half outside it, which is what the
    training-time translation augmentation exists to cover.

    A frame smaller than the window is padded rather than refused: a rig whose
    ROI is smaller than the model's crop should still track, just with a
    border. The pad goes bottom-right so the offset stays the honest
    ``(x0, y0)``.
    """
    fh, fw = frame.shape[:2]
    x0 = max(0, min(round(cx - w / 2.0), max(0, fw - w)))
    y0 = max(0, min(round(cy - h / 2.0), max(0, fh - h)))
    cut = frame[y0:y0 + h, x0:x0 + w]
    if cut.shape[0] != h or cut.shape[1] != w:
        cut = cv2.copyMakeBorder(cut, 0, h - cut.shape[0], 0, w - cut.shape[1],
                                 cv2.BORDER_CONSTANT, value=0)
    return cut, CropWindow(x0=x0, y0=y0, w=w, h=h)


def confident_centroid(pose: dict, conf_min: float = DEFAULT_CONF_MIN,
                       good_min: int = DEFAULT_GOOD_MIN
                       ) -> Optional[Tuple[float, float]]:
    """Where to put the window next, or ``None`` if the pose does not say.

    The mean of the confident points only. A below-threshold keypoint is not a
    weak opinion about position; it is no opinion, and averaging it in walks
    the window off the animal within a few frames. Points that came back as
    NaN are excluded for the same reason.
    """
    xs: List[float] = []
    ys: List[float] = []
    for value in (pose or {}).values():
        if value is None or len(value) < 3:
            continue
        x, y, conf = float(value[0]), float(value[1]), float(value[2])
        if conf < conf_min or not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < max(1, good_min):
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))


def scan_points(frame_w: int, frame_h: int, w: int, h: int
                ) -> List[Tuple[float, float]]:
    """Window centres covering the frame, stepping by half a window.

    Half-window steps so an animal straddling two windows is whole in a third.
    On a frame not much larger than the window this is one or two points, which
    is why recovery costs a handful of inferences rather than a sweep.
    """
    step_x = max(1, w // 2)
    step_y = max(1, h // 2)
    xs = list(range(w // 2, max(w // 2 + 1, frame_w - w // 2 + 1), step_x))
    ys = list(range(h // 2, max(h // 2 + 1, frame_h - h // 2 + 1), step_y))
    return [(float(x), float(y)) for y in ys for x in xs]


def count_confident(pose: dict, conf_min: float = DEFAULT_CONF_MIN) -> int:
    """How many keypoints this pose is confident about."""
    n = 0
    for value in (pose or {}).values():
        if value is None or len(value) < 3:
            continue
        if float(value[2]) >= conf_min:
            n += 1
    return n


class BoxCropState:
    """Where one box's window is, and whether it still knows.

    One per box: the window size belongs to the model, the position belongs to
    the animal.
    """

    def __init__(self, w: int, h: int, conf_min: float = DEFAULT_CONF_MIN,
                 good_min: int = DEFAULT_GOOD_MIN, reacquire: bool = True,
                 lost_before_scan: int = DEFAULT_LOST_BEFORE_SCAN):
        self.w = int(w)
        self.h = int(h)
        self.conf_min = float(conf_min)
        self.good_min = int(good_min)
        self.reacquire = bool(reacquire)
        self.lost_before_scan = int(lost_before_scan)
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        #: Consecutive frames with too few confident keypoints.
        self.lost_frames = 0
        #: Windows tried while lost, for the health panel, a crop that
        #: searches often is wrong even when it never crashes.
        self.scans = 0
        self._scan_i = -1

    @property
    def has_fix(self) -> bool:
        return self.cx is not None and self.cy is not None

    @property
    def searching(self) -> bool:
        """True when the window has given up following and started looking."""
        return (self.reacquire and self.lost_frames >= self.lost_before_scan)

    def centre_for(self, frame_w: int, frame_h: int,
                   hint: Optional[Tuple[float, float]] = None
                   ) -> Tuple[float, float]:
        """Where to cut this frame.

        While tracking: the motion filter's prediction if there is one, since
        it beats the last centroid exactly when the animal is fast and the
        window is most likely to be left behind; otherwise the last centroid.

        While lost: the next point on a coarse grid. The published loop sweeps
        that grid synchronously, which suits a single-animal script; here the
        frames keep coming, so trying ONE more window per frame recovers in a
        handful of frames, costs no extra inference, and cannot stall the boxes
        that are still tracking.
        """
        if self.searching:
            return self._next_scan_centre(frame_w, frame_h)
        if hint is not None and all(np.isfinite(v) for v in hint):
            return (float(hint[0]), float(hint[1]))
        if self.has_fix:
            return (self.cx, self.cy)
        return (frame_w / 2.0, frame_h / 2.0)

    def _next_scan_centre(self, frame_w: int, frame_h: int
                          ) -> Tuple[float, float]:
        points = scan_points(frame_w, frame_h, self.w, self.h)
        self._scan_i = (self._scan_i + 1) % len(points)
        self.scans += 1
        return points[self._scan_i]

    def update(self, pose: dict) -> bool:
        """Take a pose in BOX coordinates; return whether the box is tracked.

        The window is re-centred only on a good pose, so a momentary loss keeps
        the last known position rather than resetting to the frame centre.
        """
        centre = confident_centroid(pose, self.conf_min, self.good_min)
        if centre is None:
            self.lost_frames += 1
            return False
        found_again = self.searching
        self.cx, self.cy = centre
        self.lost_frames = 0
        self._scan_i = -1
        if found_again:
            logger.info("crop window re-acquired the animal at (%.0f, %.0f)",
                        self.cx, self.cy)
        return True


__all__ = ["DEFAULT_CONF_MIN", "DEFAULT_GOOD_MIN", "DEFAULT_LOST_BEFORE_SCAN",
           "BoxCropState", "CropWindow", "confident_centroid",
           "count_confident", "crop_at", "scan_points"]
