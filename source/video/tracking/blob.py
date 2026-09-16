"""Real-time blob (contour) object tracking for behavioural rigs.

Background subtraction with optional CLAHE, illumination normalization,
and MOG2 adaptive background, plus asymmetric morphology and
contour detection. Optical flow + Kalman smoothing via TrackingEnhancer.
Thread-safe for multi-box operation; integrates with the zone manager
for event-triggered protocols.

Detection pipeline:
    Grayscale -> CLAHE -> (optional illumination norm) -> Blur
    -> Background subtract (static or adaptive) -> Threshold
    -> Directional mask -> Morphology (open small, close large)
    -> Contour -> blob select -> Optical Flow + Kalman smoothing

Pose (DLC/SLEAP) tracking lives in ``source.video.tracking.pose``.

Usage:
    tracker = BlobTracker(box_id=1)
    tracker.set_background(background_frame)
    tracker.initialize(frame)
    success, position = tracker.update(frame)  # position = (x, y, w, h)
"""

import cv2
import math
import numpy as np
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from source.log import get_logger

logger = get_logger()

# self_norm ("Simple") defaults, straight from mousefinder's PCGTop, which
# uses one `minsize` for both the texture blur and the speck erosion. THE
# OWNER of these numbers: BlobConfig and the framebus TrackingConfig carry
# copies as dataclass defaults (they cannot import this module without
# dragging cv2 into the config layer), and a test asserts all three agree.
#
# Sigma 0 means "derive from the frame", mousefinder's own rule, height/20,
# rather than "off", because there is no sensible fixed pixel width for it.
SELF_NORM_SIGMA = 0.0
# -1 = derive from the frame; 0 = off; anything else is the operator's word.
# Both are absolute pixel widths, so one fixed number cannot suit every
# camera: mousefinder's 10 was tuned on its own view, and on a 570x290 ROI
# the same 10 erodes a mouse from 594 px down to 242 px and costs 8 points of
# tracking accuracy (measured, 2700 frames). Scaling by frame height keeps
# mousefinder's numbers at mousefinder's scale, height/58 is 10 at the 580 px
# view they came from, while a small ROI gets the ~5 that measured best on it.
SELF_NORM_AUTO = -1
SELF_NORM_SMOOTH_SIGMA = SELF_NORM_AUTO
SELF_NORM_MINSIZE = SELF_NORM_AUTO
_SELF_NORM_AUTO_DIVISOR = 58.0

# What the tracker believes about the animal right now. Detection answers
# "where is it in this frame"; these answer "do we still know where it is",
# which is a question about time and cannot be read off one frame.
#
# The distinction that matters downstream is COASTING vs TRACKED: a coasted
# position is the motion model's guess, not a measurement, and analysis that
# cannot tell them apart will read a confident straight line through an
# occlusion that never happened.
TRACK_TRACKED = "tracked"        # measured this frame
TRACK_REACQUIRED = "reacquired"  # measured, but the track had been lost
TRACK_COASTING = "coasting"      # predicted; within the coast budget
TRACK_LOST = "lost"              # coast exhausted, position is unknown

# How ``estimate_area_bracket`` tells the animal from the furniture.
# A blob counts as moving when at least this fraction of it is absent from
# the pixels every sample frame agreed on; half allows for a slow animal
# that still overlaps its own previous position.
_AREA_MOVED_FRACTION = 0.5
# Fallback shape test for a single frame: the clutter that outweighs an
# animal in a background-free mask is edge shadow, long thin strips
# (measured 141x26, 72x21), while an animal's bounding box is roughly
# square (27x30). Anything longer than this ratio is not the subject.
_AREA_MAX_ELONGATION = 3.0

# Acquiring a track in self_norm needs the same "which blob moved?" evidence,
# built from the live mask stream: samples this many frames apart, and what
# all of them agree is foreground is the arena's own furniture.
_SN_STATIC_SAMPLES = 3
_SN_STATIC_STRIDE = 8
# Centroid travel between samples below which the animal counts as settled
# and the reference stops learning, so a resting animal is never mistaken
# for furniture.
_SN_STATIC_MIN_STEP_PX = 3.0
# Frames the tracker will wait for that reference before acquiring anyway.
# Waiting costs a second of no detection at session start; not waiting costs
# the whole session, because a shadow picked at frame 0 sits inside its own
# gate forever after and the track never leaves it.
_SN_ACQUIRE_MAX_WAIT_FRAMES = 60

# Optional Cython acceleration for illumination normalization.
_HAS_CYTHON_IMAGE_OPS = False
try:
    from source.cython._image_ops import fused_illumination_normalize as _cy_illum_norm
    _HAS_CYTHON_IMAGE_OPS = True
except ImportError:
    pass


def _wide_gaussian(src: np.ndarray, sigma: float, dst=None) -> np.ndarray:
    """Gaussian blur that stays cheap as sigma grows.

    ``cv2.GaussianBlur`` with a large sigma is a genuinely large separable
    convolution, sigma 24 means a 145-tap kernel, and on a build without an
    SIMD/IPP path that is ~100 ms for one 640x480 float32 frame. self_norm
    needs two such blurs per frame, which put the whole mode at ~3 fps: far
    below any camera it would run behind.

    A Gaussian that wide is by definition band-limited well below the pixel
    grid, so computing it at reduced resolution and scaling back up is not an
    approximation of the intent; it IS the low-frequency field being asked
    for. The downsample factor keeps ~6 sigma of support in the small image,
    which holds the error to a few grey levels out of 255 (0.2 % on the Li
    ratio) for a 15x speedup.
    """
    factor = max(1, min(8, round(sigma / 6.0)))
    if factor == 1:
        return cv2.GaussianBlur(src, (0, 0), sigma, dst=dst)
    h, w = src.shape[:2]
    small = cv2.resize(src, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    cv2.GaussianBlur(small, (0, 0), sigma / factor, dst=small)
    return cv2.resize(small, (w, h), dst=dst,
                      interpolation=cv2.INTER_LINEAR)


def threshold_li(image: np.ndarray) -> float:
    """Li's iterative minimum-cross-entropy threshold (numpy only).

    A faithful port of the classic Li & Lee method, the same algorithm
    ``skimage.filters.threshold_li`` implements and the one mousefinder uses
    to pick a threshold automatically instead of asking the operator to
    hand-tune a number. Kept dependency-free (no scikit-image) so the Jetson
    build stays lean.

    Returns a scalar threshold in the image's own value range. For the
    self-normalised image (frame ÷ its own blur) that range is ratios around
    1.0; for a plain difference image it is 0-255.
    """
    img = np.asarray(image, dtype=np.float64).ravel()
    img = img[np.isfinite(img)]
    if img.size == 0:
        return 0.0
    val_range = float(img.max() - img.min())
    if val_range <= 0:
        return float(img.flat[0])
    tolerance = val_range / 1024.0
    # Li's cross-entropy uses log(mean), which needs strictly positive means;
    # shift the data so its minimum sits at 1.0.
    offset = float(img.min()) - 1.0
    shifted = img - offset
    t_next = float(shifted.mean())
    t_curr = -2.0 * tolerance
    # Bounded iteration, converges in a handful of steps; the cap only guards
    # against a pathological non-converging input.
    for _ in range(100):
        if abs(t_next - t_curr) <= tolerance:
            break
        t_curr = t_next
        fg = shifted > t_curr
        mean_fg = float(shifted[fg].mean()) if fg.any() else 1e-6
        mean_bg = float(shifted[~fg].mean()) if (~fg).any() else 1e-6
        mean_fg = max(mean_fg, 1e-6)
        mean_bg = max(mean_bg, 1e-6)
        denom = np.log(mean_fg) - np.log(mean_bg)
        if denom == 0:
            break
        t_next = (mean_fg - mean_bg) / denom
    return float(t_next + offset)


class BlobTracker:
    """Background subtraction tracker for behavioral experiments.

    Uses background subtraction and contour analysis to find the single
    largest foreground region (the animal) in each frame.

    Attributes:
        setup_id: Box identifier for multi-box setups
        is_initialized: Whether background has been set
        background: Background image for subtraction
        last_position: Last known position (x, y, w, h)
        tracking_active: Whether tracking is currently active
    """

    # Below this a contour is sensor noise on any rig, at any scale, the
    # one area bound that stays hard. Everything above it competes, with the
    # operator's Min/Max Area acting as a strong preference rather than a
    # veto (see _select_candidate).
    _ABSOLUTE_NOISE_AREA = 12.0

    # How much an out-of-range blob is marked down. Well under 1 so an
    # in-range candidate of comparable quality always wins, but far enough
    # above 0 that the animal still beats a speck when the bounds are wrong.
    _OUT_OF_RANGE_PENALTY = 0.55

    def __init__(self, setup_id, callback=None):
        """Initialize tracker.

        Args:
            setup_id: Box identifier
            callback: Optional callback(box_id, x, y, w, h, timestamp)
        """
        self.setup_id = setup_id
        self.is_initialized = False
        self.background = None
        self.background_gray = None
        self.last_position = None
        self.tracking_active = False
        self.callback = callback
        self.lock = threading.Lock()
        self.enhancer = None  # TrackingEnhancer (OF + KF), set externally

        # Tracking statistics
        self.frame_count = 0
        self.success_count = 0
        self.start_time = None

        # Core detection parameters
        self.threshold = 25           # Binary threshold for bg subtraction
        self.min_area = 100           # Min contour area in pixels
        self.max_area = 50000         # Max contour area
        self.detect_dark = True       # True = dark animal on light background

        # Mouse-sized noise filter parameters
        self.subject_min_area = 150     # Typical mouse is 200-800 px
        self.aspect_ratio_max = 8.0   # Reject line-like noise (cables, shadows)
        self.solidity_min = 0.3       # Tolerate motion blur
        self.use_median_prefilter = True  # 5x5 median blur kills salt-pepper noise

        # Preprocessing parameters.
        # CLAHE is off by default: applying it independently to bg and
        # live frames before absdiff creates phantom motion (per-frame
        # tile histograms shift background intensities when the animal
        # enters a tile). Enable only for very uneven lighting, and
        # expect reduced stability.
        self.use_clahe = False
        self.clahe_clip_limit = 3.0
        self.clahe_tile_size = 8
        self.use_adaptive_threshold = False
        self.use_illumination_norm = False
        self.blur_kernel_size = 5
        self.blur_mode = "gaussian"       # "gaussian" or "median"

        # Background model:
        # "static" (default): the captured reference frame, unchanged.
        # "running_avg": exponential running average masked by the detected
        #                animal so it never leaks into the reference;
        #                absorbs slow lighting drift without fading a still animal.
        # "mog2": OpenCV MOG2 adaptive subtractor, auto-frozen when the animal
        #         is stationary.
        # "self_norm": NO background reference at all (mousefinder-style). Each
        #         frame is divided by its own Gaussian blur to cancel uneven
        #         illumination, then thresholded at a Li-estimated ratio. Immune
        #         to lighting drift and to a stale/mis-registered background;
        #         needs a consistently dark (or light) animal against the arena.
        #
        # Static is the default because it is what capturing a background
        # implies: the operator's reference frame stays the subtraction
        # reference. The config layer leaves bg_mode None unless the user
        # picks one, so this default is what most sessions actually run.
        self.bg_mode = "static"
        # self_norm parameters. sigma is the illumination-correction blur width
        # (auto-set to crop_height/20 on first frame when 0); threshold is the
        # ratio below/above which a pixel is foreground, estimated by
        # ``estimate_threshold`` (Li's method) or hand-set. minsize erodes
        # sub-animal specks after thresholding (mousefinder's minimum_filter).
        self._self_norm_sigma = SELF_NORM_SIGMA
        # None = "not estimated yet" (auto-Li on the first frame). A numeric 0
        # would be a valid-looking ratio, so a degenerate all-dark first frame
        # returning 0 must not be mistaken for "already calibrated".
        self._self_norm_threshold: Optional[float] = None
        # Second (post-correction) blur and the speck erosion, both from
        # mousefinder's PCGTop where one `minsize` drives each. Split here so
        # texture can be softened without also eroding the animal.
        #
        # Default -1 = derived from frame height (see SELF_NORM_AUTO), which
        # reproduces mousefinder's own values on a mousefinder-sized view.
        # Note the coupling: erosion shrinks the contour, so raising minsize
        # lowers measured blob area, re-run Auto Threshold (which derives the
        # area bracket from the eroded mask) after changing it, or a
        # previously-fine min_area starts rejecting the animal. Set either to
        # 0 for the plain divide-and-threshold.
        self._self_norm_smooth_sigma = SELF_NORM_SMOOTH_SIGMA
        self._self_norm_minsize = SELF_NORM_MINSIZE
        # True once a ratio arrives from the operator (Auto button or a saved
        # project), a re-estimate on the next frame must not overwrite it.
        self._self_norm_ratio_pinned = False
        self._sn_erode_kernel = None
        self._sn_erode_size = 0
        # Acquisition reference: pixels foreground in every recent sample of
        # the live mask stream, i.e. what never moves. Only self_norm needs
        # it, a subtraction mode's mask holds nothing that stayed put.
        self._sn_static_samples = deque(maxlen=_SN_STATIC_SAMPLES)
        self._sn_static = None
        self._sn_static_countdown = 0
        self._sn_moving = None            # per-frame, only while acquiring
        self._sn_frames_seen = 0
        self._sn_acquire_waited = False   # log the wait once, not per frame
        self._sn_have_track = False       # gates learning; see _note_static_sample
        self._sn_static_last_centroid = None
        # Pooled float32 scratch for the self_norm path (allocated on first
        # use / shape change) so the hot path doesn't allocate 3 full-frame
        # arrays per frame.
        self._sn_x = None
        self._sn_blur = None
        self._sn_corrected = None
        self._sn_smooth = None
        self._sn_buf_shape = None
        self._bg_running_f32 = None       # float32 accumulator for running_avg
        self._bg_running_alpha = 0.002    # Bonsai default range 0.001-0.01
        self._mog2 = None                 # MOG2 subtractor instance
        self._mog2_history = 500
        self._mog2_threshold = 25

        # Morphological kernels -- asymmetric: small open, large close
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Pre-create CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )

        # Pre-allocated buffer for Cython illumination normalization
        self._illum_norm_dst = None

        # Per-frame scratch buffers for the static background-subtraction
        # path (absdiff, threshold, polarity-compare, morph-open/close),
        # reused via OpenCV's ``dst=`` to avoid per-frame allocation.
        # Re-allocated only when the frame shape changes (first call or
        # ROI resize).
        #
        # ``detect_with_mask`` (calibration preview) copies the returned
        # mask before handing it to the UI so the next live frame can
        # overwrite ``_mask_buf`` safely.
        self._diff_buf: Optional[np.ndarray] = None
        self._mask_buf: Optional[np.ndarray] = None
        self._polarity_buf: Optional[np.ndarray] = None
        self._morph_tmp_buf: Optional[np.ndarray] = None
        self._bg_inv_buf: Optional[np.ndarray] = None
        self._buf_shape: Optional[tuple] = None

        # Ping-pong scratch buffers for the grayscale pre-processing pipeline
        # (median / CLAHE / final blur), reused via ``dst=`` so each live
        # frame doesn't allocate 1-3 full-frame arrays. Re-allocated on shape
        # change, same as the detection pool above.
        self._prep_buf_a: Optional[np.ndarray] = None
        self._prep_buf_b: Optional[np.ndarray] = None
        self._prep_buf_shape: Optional[tuple] = None

        # Adaptive-bg (MOG2) learning rate control. Auto-frozen when the
        # animal stays roughly still for `_stationary_threshold_frames`
        # consecutive frames so the still subject doesn't fade into bg.
        self._adaptive_learning_rate = 0.005   # normal rate
        self._adaptive_frozen = False
        self._stationary_px_threshold = 2.0    # centroid jitter under this px = "still"
        self._stationary_threshold_frames = 30 # ~1 s at 30 fps before freezing
        self._stationary_streak = 0

        # Rolling tracking-quality metrics (calibration UI reads these).
        self._recent_results = deque(maxlen=120)   # 1 (success) / 0 (miss)
        self._recent_areas = deque(maxlen=120)     # contour area px
        self._recent_centroids = deque(maxlen=30)  # last 30 centroids for jitter
        self._jitter_px = 0.0                      # centroid std-dev over last 30 frames

        self._last_contour = None   # numpy array of contour points (for filled overlay)
        # cv2.contourArea computed during detection; stashed so the
        # rolling-quality metrics path doesn't recompute it.
        self._last_area: float = 0.0

        # Moments-based centroid. Stored separately from last_position
        # (bbox) because the bbox center drifts when the animal rears,
        # grooms, or the tail sticks out.
        self.last_centroid = None   # (cx, cy) in pixels, float

        # ---- Spatial gating + temporal debounce (robust contour selection) --
        # Prefer the largest contour NEAR the predicted position, and
        # require an out-of-gate "challenger" blob to persist a few
        # frames before jumping to it, stops a bedding clump, dropping,
        # reflection, or tail-split from teleporting the centroid. A
        # constant-velocity motion model (EMA of centroid deltas) gives
        # the prediction, so it works with or without the OF+KF enhancer.
        # use_spatial_gating=False restores pure-largest.
        self.use_spatial_gating = True
        self._predicted_centroid = None   # gate center for the next frame
        self._velocity = (0.0, 0.0)       # px/frame, EMA
        self._vel_alpha = 0.5             # EMA weight on the newest delta
        self._gate_base_px = 50.0         # min gate radius
        self._gate_speed_factor = 4.0     # gate grows with predicted speed
        self._gate_max_px = 400.0         # cap so the gate never spans the arena
        self._coast_streak = 0            # consecutive frames with no in-gate hit
        self._max_coast = 30              # after this, drop the gate + re-acquire
        # Temporal debounce: an out-of-gate challenger must reappear within
        # _debounce_radius for _debounce_frames in a row, and be size-plausible
        # vs the recent track, before we commit the track to it.
        self._debounce_frames = 2
        self._debounce_radius_px = 50.0
        self._challenger_pos = None
        self._challenger_streak = 0
        # Compared against the MEDIAN recent area, not the last one, and wide
        # enough to survive a rear or a partial occlusion halving the silhouette.
        self._area_ratio_lo = 0.25
        self._area_ratio_hi = 4.0
        # Selection weights: proximity leads, size plausibility keeps a nearby
        # shadow or bedding clump from winning on distance alone.
        self._w_proximity = 0.6
        self._w_plausibility = 0.4
        # This frame's grayscale and its mean brightness. Acquisition ranks
        # blobs by how animal-coloured they are; see _contrast_score.
        self._sn_gray = None
        self._sn_frame_level = 0.0
        # Escape hatch: frames the committed blob has been implausibly sized
        # for. Past _max_implausible the track is assumed wrong and the motion
        # model is dropped so the next frame re-acquires on size alone.
        self._implausible_streak = 0
        self._max_implausible = 15
        self._bg_shape_warned = False
        # Diagnostics for a mis-set Max Area (see _detect_contours).
        self._rejected_too_large = 0
        self._largest_rejected_area = 0.0
        # Contours this frame that fell outside the configured area range.
        # Kept as demoted candidates so wrong bounds degrade instead of
        # blinding the tracker, see _select_candidate.
        self._area_fallback = []
        # True while the blob being tracked is one of those, and its area,
        # the number the operator has to bracket with Min/Max Area.
        self._tracked_out_of_range = False
        self._out_of_range_area = 0.0

        logger.info(f"BlobTracker initialized for box {setup_id}")

    def _prepare_gray(self, gray):
        """Preprocess a grayscale frame (background or live).

        Pipeline: median pre-filter -> CLAHE -> (optional illumination norm) ->
        (optional final blur, skipped when the median pre-filter already ran and
        no contrast stage did).

        Called identically on the background AND every live frame so both
        go through the same pipeline; any param change recomputes
        background_gray from the stored raw background.

        Args:
            gray: Grayscale frame (uint8)

        Returns:
            Preprocessed grayscale frame (uint8). This may be one of the
            pooled ping-pong buffers, which the next call overwrites,
            callers that retain it across frames must copy (see
            :meth:`_recompute_background_gray`).
        """
        h, w = gray.shape[:2]
        if self._prep_buf_shape != (h, w):
            self._prep_buf_a = np.empty((h, w), dtype=np.uint8)
            self._prep_buf_b = np.empty((h, w), dtype=np.uint8)
            self._prep_buf_shape = (h, w)

        result = gray

        # 0. Median pre-filter, kills salt-and-pepper sensor noise
        # before contrast enhancement amplifies it. Write into the pooled
        # buffer not currently holding ``result`` so src != dst.
        if self.use_median_prefilter:
            dst = self._prep_buf_b if result is self._prep_buf_a else self._prep_buf_a
            cv2.medianBlur(result, 5, dst=dst)
            result = dst

        # 1. CLAHE contrast enhancement
        if self.use_clahe:
            dst = self._prep_buf_b if result is self._prep_buf_a else self._prep_buf_a
            self._clahe.apply(result, dst=dst)
            result = dst

        # 2. Illumination normalization (divide-by-blurred)
        if self.use_illumination_norm:
            blurred_bg = cv2.GaussianBlur(result, (51, 51), 0).astype(np.float32)
            if _HAS_CYTHON_IMAGE_OPS:
                if (self._illum_norm_dst is None or
                        self._illum_norm_dst.shape != (h, w)):
                    self._illum_norm_dst = np.empty((h, w), dtype=np.uint8)
                result = _cy_illum_norm(result, blurred_bg, self._illum_norm_dst)
            else:
                blurred_bg[blurred_bg < 1] = 1
                normalized = (result.astype(np.float32) / blurred_bg) * 128
                result = np.clip(normalized, 0, 255).astype(np.uint8)

        # 3. Final blur (Gaussian or median). Skip it when the median
        # pre-filter (step 0) already ran AND no contrast stage (CLAHE /
        # illumination-norm) sat between them: a second denoise pass over an
        # already-median-filtered frame just softens the animal's edges,
        # hurting contour area + moments-centroid accuracy (AnyMaze-style
        # tracking wants minimal, single-pass smoothing), and costs a
        # full-frame op every frame for nothing. When a contrast stage ran,
        # keep the post-blur: CLAHE / illum-norm amplify noise that needs it.
        _contrast_ran = self.use_clahe or self.use_illumination_norm
        if not (self.use_median_prefilter and not _contrast_ran):
            k = self.blur_kernel_size
            dst = self._prep_buf_b if result is self._prep_buf_a else self._prep_buf_a
            if self.blur_mode == "median":
                cv2.medianBlur(result, k, dst=dst)
            else:
                cv2.GaussianBlur(result, (k, k), 0, dst=dst)
            result = dst

        return result

    def _ensure_bufs(self, shape: tuple) -> None:
        """Allocate per-frame scratch buffers sized to ``shape`` once.

        Re-allocated only when the frame shape changes; live detection
        reuses these buffers via OpenCV's ``dst=`` parameter.
        """
        if self._buf_shape == shape:
            return
        h, w = shape
        self._diff_buf = np.empty((h, w), dtype=np.uint8)
        self._mask_buf = np.empty((h, w), dtype=np.uint8)
        self._polarity_buf = np.empty((h, w), dtype=np.uint8)
        self._morph_tmp_buf = np.empty((h, w), dtype=np.uint8)
        self._bg_inv_buf = np.empty((h, w), dtype=np.uint8)
        # Pooled uint8 target for the running-avg background so the
        # per-frame convertScaleAbs writes in place instead of allocating a
        # fresh HxW image every successful frame.
        self._bg_gray_buf = np.empty((h, w), dtype=np.uint8)
        self._buf_shape = shape

    def _self_norm_mask(self, gray):
        """Background-free foreground mask (mousefinder-style).

        Divide the frame by its own Gaussian blur to cancel uneven
        illumination, then threshold at the Li-estimated ratio. No background
        reference is consulted, so a stale/mis-registered background and slow
        lighting drift cannot corrupt it. Writes into the pooled ``_mask_buf``.
        """
        h, w = gray.shape[:2]
        # This method WRITES _mask_buf, so it allocates it rather than
        # assuming a caller already did. estimate_area_bracket calls straight
        # in without going through the detect path, and cv2.erode asserts on
        # an empty dst, so Auto Threshold died the moment speck removal was
        # on by default.
        self._ensure_bufs((h, w))
        if self._sn_buf_shape != (h, w):
            self._sn_x = np.empty((h, w), dtype=np.float32)
            self._sn_blur = np.empty((h, w), dtype=np.float32)
            self._sn_corrected = np.empty((h, w), dtype=np.float32)
            self._sn_smooth = np.empty((h, w), dtype=np.float32)
            self._sn_buf_shape = (h, w)
        sigma = self._self_norm_sigma
        if sigma <= 0:
            sigma = max(1.0, h / 20.0)   # mousefinder default: 1/20 the height
        self._sn_x[:] = gray             # uint8 → float32 into the pooled buffer
        _wide_gaussian(self._sn_x, sigma, dst=self._sn_blur)
        np.maximum(self._sn_blur, 1.0, out=self._sn_blur)
        np.divide(self._sn_x, self._sn_blur, out=self._sn_corrected)  # ratios ≈ 1
        thr = self._self_norm_threshold
        if thr is None:
            # Auto-estimate once from the first frame; the Auto button (or
            # estimate_threshold) can recompute it deliberately. Li runs on the
            # UNSMOOTHED ratio image, matching mousefinder, where the smoothing
            # below only shapes what the threshold is then applied to.
            thr = float(threshold_li(self._sn_corrected))
            self._self_norm_threshold = thr
        # Second blur: collapses arena texture (bedding, gravel) and thin
        # structures toward 1.0 so they fall on the background side of the
        # ratio, leaving the animal's solid core. 0 = skip.
        smooth_sigma = self._auto_scaled(self._self_norm_smooth_sigma, h)
        if smooth_sigma > 0:
            _wide_gaussian(self._sn_corrected, float(smooth_sigma),
                           dst=self._sn_smooth)
            src = self._sn_smooth
        else:
            src = self._sn_corrected
        # Dark animal → below the ratio is foreground; light animal → above.
        # cv2.compare writes 255/0 straight into the pooled uint8 mask (no bool
        # temporary).
        cmp_op = cv2.CMP_LT if self.detect_dark else cv2.CMP_GT
        cv2.compare(src, float(thr), cmp_op, dst=self._mask_buf)
        # Speck removal, erosion by an NxN box IS a minimum filter on a binary
        # image, which is what mousefinder applies at this point.
        n = int(round(self._auto_scaled(self._self_norm_minsize, h)))
        if n > 1:
            if self._sn_erode_size != n or self._sn_erode_kernel is None:
                self._sn_erode_kernel = np.ones((n, n), np.uint8)
                self._sn_erode_size = n
            cv2.erode(self._mask_buf, self._sn_erode_kernel,
                      dst=self._mask_buf)
        return self._mask_buf

    @staticmethod
    def _auto_scaled(value, frame_height: int) -> float:
        """Resolve a self_norm width: negative means "derive from the frame".

        Kept separate from the stored setting so the operator's own number is
        never quietly rescaled, only the sentinel is resolved, and only here,
        where the frame being processed is finally known.
        """
        if value >= 0:
            return float(value)
        return max(1.0, frame_height / _SELF_NORM_AUTO_DIVISOR)

    def estimate_threshold(self, frame) -> Optional[float]:
        """Estimate a detection threshold from a sample frame via Li's method.

        For ``self_norm`` this is the ratio on the illumination-normalised
        image; for the background-subtraction modes it is a level on the
        difference image (so the Auto button proposes a fixed threshold). Sets
        the corresponding attribute and returns the value (None if it can't be
        computed, e.g. no background for a subtraction mode).
        """
        with self.lock:
            if frame is None:
                return None
            gray = self._resolve_gray(frame, None)
            gray_processed = self._prepare_gray(gray)
            if self.bg_mode == "self_norm":
                sigma = self._self_norm_sigma or max(1.0, gray.shape[0] / 20.0)
                x = gray_processed.astype(np.float32)
                # Same blur the live path uses, or the ratio pinned here
                # would not be the ratio the tracker then applies.
                blur = _wide_gaussian(x, sigma)
                np.maximum(blur, 1.0, out=blur)
                thr = float(threshold_li(x / blur))
                self._self_norm_threshold = thr
                # Deliberate estimate: hold it against later sigma edits and
                # persist it, rather than silently re-deriving next frame.
                self._self_norm_ratio_pinned = True
                return thr
            if self.background_gray is None:
                return None
            bg = self.background_gray
            if gray_processed.shape != bg.shape:
                bg = cv2.resize(bg, (gray_processed.shape[1], gray_processed.shape[0]))
            diff = cv2.absdiff(gray_processed, bg)
            thr = float(threshold_li(diff))
            self.threshold = int(max(1, min(255, round(thr))))
            return float(self.threshold)

    def _sample_mask(self, frame):
        """Foreground mask for one calibration sample, in the current mode.

        Caller holds ``self.lock``. Returns None when the mode has nothing to
        compare against yet (a subtraction mode with no background).
        """
        gray_processed = self._prepare_gray(self._resolve_gray(frame, None))
        if self.bg_mode == "self_norm":
            return self._self_norm_mask(gray_processed).copy()
        if self.background_gray is None:
            return None
        bg = self.background_gray
        if gray_processed.shape != bg.shape:
            bg = cv2.resize(
                bg, (gray_processed.shape[1], gray_processed.shape[0]))
        diff = cv2.absdiff(gray_processed, bg)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return mask

    def estimate_area_bracket(self, frames) -> Optional[tuple[int, int]]:
        """Propose (min_area, max_area) for the animal from sample frames.

        Brackets the subject's blob generously, half to four times its area,
        so the operator starts inside the range instead of hand-guessing pixel
        counts after the threshold moved under them.

        Finding which blob *is* the subject is the hard half. A background-free
        mask marks every dark static thing as foreground too, and on a real
        arena those outweigh the animal: measured on a 640x480 operant
        recording, cropped to the arena, the largest blob was a 1321 px edge
        shadow while the mouse was 326 px, so "biggest contour" would have set
        min_area at 660 and filtered the animal out of its own session.

        Pass **several frames spaced in time** and the animal is the blob that
        moved: pixels foreground in every sample are furniture and drop out.
        A single frame (or a stationary animal) falls back to the shape test in
        ``_AREA_MAX_ELONGATION``. Accepts one frame or a sequence; returns None
        when nothing is detectable.
        """
        with self.lock:
            samples = frames if isinstance(frames, (list, tuple)) else [frames]
            masks = []
            for frame in samples:
                if frame is None:
                    continue
                mask = self._sample_mask(frame)
                if mask is None:
                    return None
                masks.append(mask)
            if not masks:
                return None
            # The newest sample is what the operator is looking at, so it is
            # the one whose blobs get bracketed; the earlier masks only say
            # which of those blobs stayed put. Older frames of another size
            # (the camera changed resolution mid-sample) can say nothing about
            # it and drop out.
            ref = masks[-1]
            masks = [m for m in masks if m.shape == ref.shape]
            cnts, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            moving = None
            if len(masks) > 1:
                static = ref.copy()
                for mask in masks[:-1]:
                    cv2.bitwise_and(static, mask, dst=static)
                moving = cv2.subtract(ref, static)
            # Narrow the field in stages, each skipped if it would empty it:
            # blobs that moved, then blobs shaped like an animal, then all.
            if moving is not None:
                shortlist = [c for c in cnts
                             if self._moved_fraction(c, moving)
                             >= _AREA_MOVED_FRACTION]
                cnts = shortlist or cnts
            compact = [c for c in cnts if self._elongation(c)
                       <= _AREA_MAX_ELONGATION]
            cnts = compact or cnts
            area = max(cv2.contourArea(c) for c in cnts)
            if area <= 0:
                return None
            return max(10, round(area * 0.5)), round(area * 4.0)

    @staticmethod
    def _moved_fraction(contour, moving) -> float:
        """Share of ``contour``'s pixels that are not static across samples."""
        x, y, w, h = cv2.boundingRect(contour)
        blob = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED,
                         offset=(-x, -y))
        filled = cv2.countNonZero(blob)
        if filled == 0:
            return 0.0
        cv2.bitwise_and(blob, moving[y:y + h, x:x + w], dst=blob)
        return cv2.countNonZero(blob) / filled

    @staticmethod
    def _elongation(contour) -> float:
        """Long side over short side of the contour's bounding box."""
        _, _, w, h = cv2.boundingRect(contour)
        return max(w, h) / max(1, min(w, h))

    def _detect_contours(self, gray_processed, bg_gray):
        """Core detection pipeline: subtract -> threshold -> morph -> contours.

        Returns
        -------
        tuple: (mask, candidates)
            mask: binary mask after morphology (a view into ``self._mask_buf``;
                callers that retain it across frames must copy, see
                :meth:`detect_with_mask`)
            candidates: list of ``(area, (x,y,w,h), (cx,cy), contour)`` for
                every contour passing the area / aspect / solidity filters.
                The caller selects which one is the animal.
        """
        self._ensure_bufs(gray_processed.shape)

        # 1. Background subtraction (or, for self_norm, no background at all)
        if self.bg_mode == "self_norm":
            mask = self._self_norm_mask(gray_processed)
        elif self.bg_mode == "mog2" and self._mog2 is not None:
            lr = 0.0 if self._adaptive_frozen else self._adaptive_learning_rate
            mog_out = self._mog2.apply(gray_processed, learningRate=lr)
            # MOG2 marks shadows as 127, foreground as 255
            cv2.threshold(mog_out, 200, 255, cv2.THRESH_BINARY,
                          dst=self._mask_buf)
            mask = self._mask_buf
        else:
            # Static background subtraction
            cv2.absdiff(gray_processed, bg_gray, dst=self._diff_buf)
            diff = self._diff_buf

            # 2. Threshold (adaptive Otsu or fixed)
            if self.use_adaptive_threshold:
                cv2.threshold(diff, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                              dst=self._mask_buf)
            else:
                cv2.threshold(diff, self.threshold, 255,
                              cv2.THRESH_BINARY,
                              dst=self._mask_buf)
            mask = self._mask_buf

            # 3. Directional mask -- restrict to correct polarity
            if not self.detect_dark:
                cv2.compare(gray_processed, bg_gray, cv2.CMP_GT,
                            dst=self._polarity_buf)
            else:
                cv2.compare(bg_gray, gray_processed, cv2.CMP_GT,
                            dst=self._polarity_buf)
            cv2.bitwise_and(mask, self._polarity_buf, dst=mask)

        # 4. Morphological cleanup -- asymmetric kernels. morphologyEx
        # ping-pongs between mask and _morph_tmp_buf so neither call
        # relies on in-place behaviour (varies across OpenCV builds).
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel,
                         dst=self._morph_tmp_buf)
        cv2.morphologyEx(self._morph_tmp_buf, cv2.MORPH_CLOSE,
                         self._close_kernel, dst=mask)

        # 5. Find contours and collect EVERY candidate that passes the
        # mouse-sized noise filters (area / aspect / solidity). Choosing
        # among candidates (largest vs spatially gated) is the caller's job,
        # see _select_candidate() / update().
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        candidates = []
        # Clear BEFORE the early return, or a frame with no contours at all
        # leaves the previous frame's demoted candidates in place and the
        # tracker keeps reporting a blob that is no longer there.
        self._area_fallback = []
        if not contours:
            return mask, candidates

        # The operator's Min Area is authoritative. Raising it silently to
        # subject_min_area (the code-level noise floor) made any dialog
        # value below 150 px² a no-op, a small subject (pup, insect,
        # marked patch) could never pass the filter and nothing said why.
        # The floor survives as a one-time hint only.
        eff_min_area = self.min_area
        if (self.min_area < self.subject_min_area
                and not getattr(self, "_warned_small_min_area", False)):
            self._warned_small_min_area = True
            logger.info(
                "Blob min_area %d is below the %d px² noise floor, "
                "honouring the configured value (more noise candidates "
                "may pass the area filter).",
                self.min_area, self.subject_min_area)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ok = eff_min_area <= area <= self.max_area
            if area > self.max_area:
                # A Max Area set below the animal is the single most common
                # way to mis-calibrate this tracker, and dropping the animal
                # here would be fatal: the only thing left to track would be
                # whatever speck happened to fall inside the range.
                # Counted for the quality panel, and kept as a fallback
                # candidate below rather than discarded.
                self._rejected_too_large += 1
                self._largest_rejected_area = max(
                    self._largest_rejected_area, float(area))
            elif area < self._ABSOLUTE_NOISE_AREA:
                # Genuinely a few pixels of sensor noise. Nothing can make
                # this the animal, and keeping it would flood the pool.
                continue

            # Aspect ratio filter, reject line-like noise (cables, shadows)
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / float(min(w, h))
            if aspect > self.aspect_ratio_max:
                continue

            # Solidity filter, area / convex hull area. A mouse is reasonably
            # solid; broken-up noise has low solidity.
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and (area / hull_area) < self.solidity_min:
                continue

            # Moments centroid, more accurate than the bbox center for
            # asymmetric shapes (rearing, grooming, tail-out). m00 is
            # area, (m10/m00, m01/m00) is centroid.
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            cand = (float(area), (x, y, w, h), (cx, cy), cnt)
            if area_ok:
                candidates.append(cand)
            else:
                self._area_fallback.append(cand)

        return mask, candidates

    # ── Candidate selection: spatial gating + temporal debounce ──────────
    # A candidate is (area, (x,y,w,h), (cx,cy), contour).

    def _compute_gate_px(self) -> float:
        """Gate radius around the predicted position. Grows with predicted
        speed (so a fast animal isn't gated out) and with animal size, capped
        so the gate never spans the whole arena."""
        vx, vy = self._velocity
        speed = (vx * vx + vy * vy) ** 0.5
        base = self._gate_base_px
        if self._last_area > 0:
            base = max(base, 2.0 * (self._last_area ** 0.5))
        return min(self._gate_max_px, base + self._gate_speed_factor * speed)

    def _median_recent_area(self) -> float:
        """Median contour area over the recent track, 0 before it warms up.

        The median, not ``_last_area``: the moment the track latches onto the
        wrong blob, ``_last_area`` becomes the wrong blob's area and every
        size comparison is then anchored to the mistake.
        """
        if not self._recent_areas:
            return 0.0
        areas = sorted(self._recent_areas)
        return areas[len(areas) // 2]

    def _select_candidate(self, candidates, predicted, gate_px, may_wait=False):
        """Pick the tracked blob. Returns (candidate, in_gate).

        Every candidate is scored on proximity to the prediction AND size
        plausibility against the recent track, and the best total wins.

        Proximity is deliberately a scoring term rather than a filter. As a
        hard filter it is an absorbing state: once the track sits on a
        stationary distractor, that distractor is inside its own gate on
        every subsequent frame, so the "nothing in gate" recovery branch can
        never run and the animal, however obviously correct, is never
        even considered. Scoring lets a strongly size-plausible candidate
        outside the gate outrank a poor one inside it, while a near, correct-
        sized blob still wins comfortably in the normal case.

        ``in_gate`` reports whether the winner was actually within
        ``gate_px``; the caller still debounces a winner that was not.
        """
        # Contours outside the configured area range are demoted, not
        # discarded. The bounds are an operator setting and a wrong one used
        # to be unrecoverable: with Max Area below the animal the only
        # candidates left were specks, so the box sat on a speck while the
        # mask plainly showed the animal. Scoring them at a penalty means a
        # correct-sized blob still wins whenever one exists, and the animal is
        # still tracked (loudly, via ``tracked_out_of_range``) when the bounds
        # are simply wrong.
        fallback = getattr(self, "_area_fallback", None) or []
        out_of_range = {id(c[3]) for c in fallback}
        if self.bg_mode == "self_norm":
            # Furniture is not a candidate for anything. Dropping it here
            # rather than only at acquisition is what stops a merge-and-part
            # from stranding the track: when the animal brushes past a shadow
            # and leaves, the shadow is no longer something to strand ON.
            candidates = self._drop_static(candidates)
            fallback = self._drop_static(fallback)
        pool = candidates + fallback
        if not pool:
            return None, False
        if predicted is None:
            # No track yet: acquire from the in-range blobs, or from all of
            # them when the range excluded everything.
            pick = self._acquire(candidates or pool, may_wait)
            if pick is None:
                return None, False
            self._tracked_out_of_range = not candidates
            self._out_of_range_area = pick[0] if not candidates else 0.0
            return pick, True
        px, py = predicted
        med = self._median_recent_area()

        def score(cand):
            area, _bbox, (cx, cy), cnt = cand
            dist = math.hypot(cx - px, cy - py)
            proximity = 1.0 / (1.0 + dist / max(1.0, gate_px))
            if med <= 0.0 or area <= 0.0:
                plausibility = 1.0
            else:
                # Symmetric in log-space: half the expected size is penalised
                # exactly as much as twice it.
                plausibility = math.exp(-abs(math.log(area / med)))
            total = (self._w_proximity * proximity
                     + self._w_plausibility * plausibility)
            if id(cnt) in out_of_range:
                total *= self._OUT_OF_RANGE_PENALTY
            return total

        best = max(pool, key=score)
        self._tracked_out_of_range = id(best[3]) in out_of_range
        self._out_of_range_area = best[0] if self._tracked_out_of_range else 0.0
        bx, by = best[2]
        in_gate = ((bx - px) ** 2 + (by - py) ** 2) <= gate_px * gate_px
        return best, in_gate

    def _contrast_score(self, cand) -> float:
        """How animal-coloured this blob is, 0..1. Neutral (0.5) if unknown.

        Used to pick between blobs at acquisition, where there is no track to
        reason from. A mouse is a near-black object on a pale floor, whereas
        the shadows and edge bands sharing its mask in ``self_norm`` are
        mid-grey, so this ranks better than size, which is what picked
        furniture in the first place.

        Measured as a *tiebreak only*: adding it as a third term to the
        tracking score changed nothing on a 2700-frame session (90.0 % vs
        89.6 % on-animal), because once a track exists proximity and size
        already settle the question. Not worth a per-frame cost.
        """
        gray = self._sn_gray
        if gray is None or self.bg_mode != "self_norm":
            return 0.5
        x, y, w, h = cand[1]
        patch = gray[y:y + h, x:x + w]
        if patch.size == 0:
            return 0.5
        level = float(patch.mean())
        frame_level = self._sn_frame_level
        if frame_level <= 0:
            return 0.5
        # Distance from the frame's own brightness, in the polarity the
        # operator configured, normalised by it so the score is exposure- and
        # camera-independent.
        delta = (frame_level - level) if self.detect_dark else (level - frame_level)
        return max(0.0, min(1.0, delta / frame_level))

    def _acquire(self, pool, may_wait):
        """Pick a blob with no track yet to go on. Returns None to keep waiting.

        Plain largest is right for the subtraction modes: their mask only
        holds what changed, so the arena's own furniture was never in it. In
        ``self_norm`` there is no background to subtract, so the mask contains
        every dark static thing in view, and on a real arena those outweigh
        the animal, measured on a 640x480 operant session, a 1373 px edge
        shadow against a ~330 px mouse. Picking it is not a one-frame error:
        being static it then sits at distance 0 from every later prediction,
        so proximity re-selects it for the rest of the session.

        So for ``self_norm`` the same evidence the area estimate uses narrows
        the field first, each stage skipped if it would leave nothing: blobs
        that moved against the static reference, then blobs shaped like an
        animal rather than like edge shadow. Of what remains the winner is
        the most animal-coloured, not the biggest, size is what misled the
        old rule, while a mouse is reliably the darkest thing in the arena.

        ``may_wait`` belongs to the live path alone. The stateless preview
        never feeds the reference, so a preview that waited for one would wait
        for ever and show the operator a permanently empty overlay.
        """
        if not pool:
            return None
        if self.bg_mode != "self_norm":
            return max(pool, key=lambda c: c[0])
        # No reference yet? Waiting beats guessing, but only up to a bound,
        # a camera whose view never settles must still track.
        if (self._sn_moving is None and may_wait
                and self._sn_frames_seen < _SN_ACQUIRE_MAX_WAIT_FRAMES):
            if not self._sn_acquire_waited:
                self._sn_acquire_waited = True
                logger.info(
                    "Box %s: holding acquisition until the furniture "
                    "reference is built (a background-free mask cannot "
                    "tell an animal from a shadow on one frame)",
                    self.setup_id)
            return None
        compact = [c for c in pool
                   if self._elongation(c[3]) <= _AREA_MAX_ELONGATION]
        return max(compact or pool, key=self._contrast_score)

    def _note_static_sample(self, mask) -> None:
        """Refresh the self_norm furniture reference from the mask stream.

        Samples are spaced ``_SN_STATIC_STRIDE`` frames apart so the animal
        stands somewhere different in each; what every sample still calls
        foreground is furniture. Cheap, one AND per sample, nothing per frame.

        "Hasn't moved lately" describes a sleeping mouse just as well as a
        shelf, and a furniture verdict on the animal would erase it. What
        separates them is history: furniture is static from the first frame,
        whereas a resting animal was moving beforehand. So learning only runs
        while the tracked animal is *travelling*, then successive samples
        catch it in different places and the intersection drops it on its own,
        with no need to special-case it. The moment it settles, learning
        freezes, and the reference keeps whatever it already knew.

        Exempting the tracked blob explicitly would be the obvious
        alternative and is a trap: it protects whatever the track happens to
        hold, so a track that has slipped onto a shadow keeps that shadow out
        of the reference, the one thing that could have freed it.
        """
        if mask is None:
            return
        self._sn_frames_seen += 1
        self._sn_static_countdown -= 1
        if self._sn_static_countdown > 0:
            return
        centroid = self.last_centroid
        # The gate guards an established reference; it must never stop one
        # from being built, or a track that acquired badly and then sat still
        # would freeze learning at two samples and wait forever to be freed.
        if (self._sn_static is not None and self._sn_have_track
                and centroid is not None):
            last = self._sn_static_last_centroid
            self._sn_static_last_centroid = centroid
            if last is not None and math.hypot(centroid[0] - last[0],
                                               centroid[1] - last[1]) \
                    < _SN_STATIC_MIN_STEP_PX:
                return                    # settled: hold, don't learn
        self._sn_static_countdown = _SN_STATIC_STRIDE
        samples = self._sn_static_samples
        if samples and samples[0].shape != mask.shape:
            samples.clear()               # resolution changed; start over
            self._sn_static = None
        samples.append(mask.copy())
        if len(samples) < samples.maxlen:
            return
        static = samples[0].copy()
        for older in list(samples)[1:]:
            cv2.bitwise_and(static, older, dst=static)
        self._sn_static = static

    def _drop_static(self, seq):
        """Blobs that moved against the furniture reference, or ``seq`` intact.

        Whole contours are judged, never pixels: subtracting the reference
        from the mask itself would carve a hole in an animal standing on a
        shadow and split it into fragments. A blob only loses if MOST of it
        is furniture, so an animal overlapping a shadow still passes whole.
        """
        moving = self._sn_moving
        if moving is None or not seq:
            return seq
        live = [c for c in seq
                if self._moved_fraction(c[3], moving) >= _AREA_MOVED_FRACTION]
        return live or seq

    def _challenger_confirmed(self, candidate) -> bool:
        """Temporal debounce for an out-of-gate candidate.

        Returns True only once the challenger has reappeared within
        ``_debounce_radius_px`` for ``_debounce_frames`` consecutive frames
        AND is a plausible size vs the recent track, so a wrong-size
        clump/merge or 1-frame flicker can't make the track jump.
        """
        area, _bbox, (cx, cy), _cnt = candidate
        # Median of the recent track, not the last frame's area: if the track
        # has drifted onto a distractor, _last_area IS the distractor and the
        # real animal gets rejected for being the wrong size.
        reference = self._median_recent_area()
        if reference > 0:
            ratio = area / reference
            if not (self._area_ratio_lo <= ratio <= self._area_ratio_hi):
                self._challenger_pos = None
                self._challenger_streak = 0
                return False
        if (self._challenger_pos is not None
                and (cx - self._challenger_pos[0]) ** 2
                + (cy - self._challenger_pos[1]) ** 2
                <= self._debounce_radius_px ** 2):
            self._challenger_streak += 1
        else:
            self._challenger_streak = 1
        self._challenger_pos = (cx, cy)
        if self._challenger_streak >= self._debounce_frames:
            self._challenger_pos = None
            self._challenger_streak = 0
            return True
        return False

    def _update_motion_model(self, success, centroid) -> None:
        """Maintain the constant-velocity gate prediction for the next frame.

        Uses an EMA of frame-to-frame centroid deltas (works with or without
        the OF+KF enhancer). On a miss the gate coasts forward and the
        velocity decays; after ``_max_coast`` consecutive misses the gate is
        dropped so the next real blob re-acquires from scratch.
        """
        if not self.use_spatial_gating:
            self._predicted_centroid = None
            return
        if success and centroid is not None:
            prev = self.last_centroid
            if prev is not None:
                a = self._vel_alpha
                self._velocity = (
                    a * (centroid[0] - prev[0]) + (1 - a) * self._velocity[0],
                    a * (centroid[1] - prev[1]) + (1 - a) * self._velocity[1],
                )
            self._predicted_centroid = (centroid[0] + self._velocity[0],
                                        centroid[1] + self._velocity[1])
            # A hit right after a loss is a re-acquisition, worth telling
            # apart from an uninterrupted track: the animal may have moved
            # while unseen, so the first position after it is not continuous
            # with the last one before it.
            self.track_state = (TRACK_REACQUIRED if self._coast_streak
                                else TRACK_TRACKED)
            self._coast_streak = 0
        else:
            self._coast_streak += 1
            if (self._coast_streak > self._max_coast
                    or self._predicted_centroid is None):
                self.track_state = TRACK_LOST
                self._predicted_centroid = None
                self._velocity = (0.0, 0.0)
                self._challenger_pos = None
                self._challenger_streak = 0
            else:
                self.track_state = TRACK_COASTING
                self._velocity = (self._velocity[0] * 0.7,
                                  self._velocity[1] * 0.7)
                self._predicted_centroid = (
                    self._predicted_centroid[0] + self._velocity[0],
                    self._predicted_centroid[1] + self._velocity[1])

    def _reset_motion_model(self) -> None:
        """Clear the gate prediction + debounce state (acquisition restart)."""
        self._predicted_centroid = None
        self._velocity = (0.0, 0.0)
        self._coast_streak = 0
        self._challenger_pos = None
        self._challenger_streak = 0

    def update_params(self, threshold=None, min_area=None, max_area=None,
                      detect_dark=None, use_clahe=None, clahe_clip_limit=None,
                      clahe_tile_size=None, use_adaptive_threshold=None,
                      blur_kernel_size=None,
                      blur_mode=None, bg_mode=None, open_kernel_size=None,
                      close_kernel_size=None, self_norm_ratio=None,
                      self_norm_sigma=None, self_norm_smooth_sigma=None,
                      self_norm_minsize=None):
        """Update the DIALOG-configurable detection parameters.

        The quality heuristics with no TrackingConfig field / dialog
        widget (subject_min_area, aspect_ratio_max, solidity_min,
        use_median_prefilter, use_spatial_gating, use_illumination_norm)
        are deliberate code-level defaults, set the attribute directly
        (tests do) rather than pretending they're configurable here.

        Args:
            threshold: Binary threshold (1-255)
            min_area: Minimum contour area in pixels
            max_area: Maximum contour area in pixels
            detect_dark: True for dark animal on light bg
            use_clahe: Enable/disable CLAHE
            clahe_clip_limit: CLAHE clip limit (1.0-10.0)
            clahe_tile_size: CLAHE tile grid size
            use_adaptive_threshold: Use Otsu instead of fixed
            blur_kernel_size: Blur kernel size (odd integer)
            blur_mode: "gaussian" or "median"
            bg_mode: "static" (default), "running_avg", "mog2", or "self_norm"
                (background-free, mousefinder-style)
            open_kernel_size: Morphological open kernel (odd, default 3)
            close_kernel_size: Morphological close kernel (odd, default 7)
            self_norm_ratio: Calibrated self_norm ratio. Restoring a saved
                value here is what makes "calibrate once, reload forever"
                work, it pins the ratio so no auto-estimate overwrites it.
                0 / None clears the pin and re-estimates on the next frame.
            self_norm_sigma: Illumination-correction blur width (0 = auto,
                crop_height/20)
            self_norm_smooth_sigma: Post-correction blur that suppresses arena
                texture (0 = skip)
            self_norm_minsize: Speck-erosion kernel size (0/1 = skip)
        """
        recreate_clahe = False
        reprocess_bg = False

        if threshold is not None:
            self.threshold = max(1, min(255, int(threshold)))
        if min_area is not None:
            self.min_area = max(1, int(min_area))
        if max_area is not None:
            self.max_area = max(1, int(max_area))
        # Validate min ≤ max (swap if inverted)
        if self.min_area > self.max_area:
            logger.warning(f"Box {self.setup_id}: min_area ({self.min_area}) > max_area ({self.max_area}), swapping")
            self.min_area, self.max_area = self.max_area, self.min_area
        if detect_dark is not None:
            self.detect_dark = detect_dark
        if blur_mode is not None:
            self.blur_mode = blur_mode
            reprocess_bg = True

        if use_clahe is not None and use_clahe != self.use_clahe:
            self.use_clahe = use_clahe
            reprocess_bg = True
        if clahe_clip_limit is not None and clahe_clip_limit != self.clahe_clip_limit:
            self.clahe_clip_limit = clahe_clip_limit
            recreate_clahe = True
            reprocess_bg = True
        if clahe_tile_size is not None and clahe_tile_size != self.clahe_tile_size:
            self.clahe_tile_size = clahe_tile_size
            recreate_clahe = True
            reprocess_bg = True
        if use_adaptive_threshold is not None:
            self.use_adaptive_threshold = use_adaptive_threshold

        # ── self_norm parameters ─────────────────────────────────────────
        # Geometry changes invalidate an AUTO-derived ratio (the corrected
        # image itself changes), but must never discard one the operator
        # calibrated or a project supplied, hence the pin.
        sn_geometry_changed = False
        if self_norm_sigma is not None:
            v = max(0.0, float(self_norm_sigma))
            if v != self._self_norm_sigma:
                self._self_norm_sigma = v
                sn_geometry_changed = True
        # Clamped at the AUTO sentinel, not at 0: -1 is a meaningful value
        # here ("derive from the frame"), and flooring it to 0 would silently
        # turn the stage off instead.
        if self_norm_smooth_sigma is not None:
            v = max(float(SELF_NORM_AUTO), float(self_norm_smooth_sigma))
            if v != self._self_norm_smooth_sigma:
                self._self_norm_smooth_sigma = v
                sn_geometry_changed = True
        if self_norm_minsize is not None:
            self._self_norm_minsize = max(SELF_NORM_AUTO,
                                          int(self_norm_minsize))
        if self_norm_ratio is not None:
            try:
                r = float(self_norm_ratio)
            except (TypeError, ValueError):
                r = 0.0
            if r > 0:
                self._self_norm_threshold = r
                self._self_norm_ratio_pinned = True
            else:
                # Explicit "no saved ratio", go back to auto-estimating.
                self._self_norm_threshold = None
                self._self_norm_ratio_pinned = False
        elif sn_geometry_changed and not self._self_norm_ratio_pinned:
            self._self_norm_threshold = None
        if blur_kernel_size is not None and blur_kernel_size != self.blur_kernel_size:
            blur_kernel_size = max(1, int(blur_kernel_size))
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1  # Force odd for OpenCV
            self.blur_kernel_size = blur_kernel_size
            reprocess_bg = True

        # Morphological kernel sizes (must be odd for symmetry)
        if open_kernel_size is not None:
            s = max(1, int(open_kernel_size))
            if s % 2 == 0:
                s += 1
            self._open_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (s, s))
        if close_kernel_size is not None:
            s = max(1, int(close_kernel_size))
            if s % 2 == 0:
                s += 1
            self._close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (s, s))

        # Background model switch
        if bg_mode is not None and bg_mode != self.bg_mode:
            self.bg_mode = bg_mode
            self._mog2 = None
            self._adaptive_frozen = False
            self._stationary_streak = 0
            if bg_mode == "mog2":
                self._init_mog2()
            elif bg_mode == "running_avg":
                if self.background_gray is not None:
                    self._bg_running_f32 = self.background_gray.astype(np.float32)

        if recreate_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
            )
        if reprocess_bg and self.background is not None:
            self._recompute_background_gray()

    def _init_mog2(self):
        """Initialize MOG2 background subtractor for adaptive mode."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self._mog2_history,
            varThreshold=self._mog2_threshold,
            detectShadows=True)
        # Seed with background if available
        if self.background_gray is not None:
            for _ in range(10):
                self._mog2.apply(self.background_gray, learningRate=0.5)
        logger.info(f"Box {self.setup_id}: MOG2 background subtractor initialized")

    def get_quality_metrics(self) -> dict:
        """Snapshot of rolling tracking quality. Read by the calibration UI.

        Returns:
            dict with keys:
              success_rate    -- fraction of recent frames that produced a centroid
              jitter_px       -- centroid std-dev over the last 30 frames
              area_median     -- median contour area in the recent window
              area_p10/p90    -- 10th / 90th percentile of recent areas
              frames_in_window -- N frames the metrics are computed over
              adaptive_frozen -- True if MOG2 learning is currently frozen
              rejected_too_large -- contours dropped for exceeding max_area
              largest_rejected_area -- biggest such contour, in px²

        The last two exist so a Max Area set below the animal is visible.
        Without them the failure is silent and deeply confusing: the mask
        still shows the animal, but the box sits on whatever speck survived
        the filter.
        """
        oversize = {
            "rejected_too_large": int(self._rejected_too_large),
            "largest_rejected_area": float(self._largest_rejected_area),
            # The blob currently being tracked is outside the configured area
            # range; it is only tracked at all because the bounds are soft.
            "tracked_out_of_range": bool(self._tracked_out_of_range),
            # Area of that blob, what Min/Max Area must be set around.
            "out_of_range_area": float(self._out_of_range_area),
        }
        n = len(self._recent_results)
        if n == 0:
            return {"success_rate": 0.0, "jitter_px": 0.0,
                    "area_median": 0.0, "area_p10": 0.0, "area_p90": 0.0,
                    "frames_in_window": 0,
                    "adaptive_frozen": self._adaptive_frozen, **oversize}
        success_rate = sum(self._recent_results) / n
        areas = sorted(self._recent_areas)
        if areas:
            mid = areas[len(areas) // 2]
            p10 = areas[max(0, int(0.10 * len(areas)) - 1)]
            p90 = areas[min(len(areas) - 1, int(0.90 * len(areas)))]
        else:
            mid = p10 = p90 = 0.0
        return {
            "success_rate": float(success_rate),
            "jitter_px": float(self._jitter_px),
            "area_median": float(mid),
            "area_p10": float(p10),
            "area_p90": float(p90),
            "frames_in_window": int(n),
            "adaptive_frozen": bool(self._adaptive_frozen),
            **oversize,
        }

    def reset_quality_metrics(self) -> None:
        """Wipe the rolling-quality buffers. Called when calibration params
        change so the UI shows fresh feedback for the new settings."""
        self._recent_results.clear()
        self._recent_areas.clear()
        self._recent_centroids.clear()
        self._jitter_px = 0.0
        self._stationary_streak = 0
        self._rejected_too_large = 0
        self._largest_rejected_area = 0.0

    def _recompute_background_gray(self):
        """Derive ``background_gray`` from ``self.background``, BGR→gray (or a
        copy when already single-channel), then the same CLAHE/blur prep as
        live frames. Assumes ``self.background`` is set (caller guards)."""
        if len(self.background.shape) == 3:
            raw_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        else:
            raw_gray = self.background.copy()
        # Copy: _prepare_gray returns a pooled buffer that live frames reuse,
        # but background_gray is the persistent subtraction reference.
        self.background_gray = self._prepare_gray(raw_gray).copy()

    def set_background(self, frame):
        """Set background image for subtraction.

        Args:
            frame: Background frame (BGR or grayscale).
        """
        with self.lock:
            if frame is not None:
                self.background = frame.copy()
                self._recompute_background_gray()
                # Seed the running-average reference so we don't start cold
                self._bg_running_f32 = self.background_gray.astype(np.float32)
                logger.info(f"Box {self.setup_id}: Background set ({frame.shape})")
            else:
                self.background = None
                self.background_gray = None
                self._bg_running_f32 = None

    def initialize(self, frame, roi=None):
        """Initialize tracker. ROI is ignored -- uses background subtraction.

        Args:
            frame: Initial frame (BGR or grayscale)
            roi: Ignored (kept for interface compatibility)

        Returns:
            bool: True if initialization successful
        """
        with self.lock:
            if frame is None:
                logger.error(f"Box {self.setup_id}: Cannot initialize with None frame")
                return False

            # self_norm needs no captured reference; that is its whole point
            # (no "capture background" ceremony). Every other mode seeds one
            # from the first frame when none was set.
            if self.bg_mode != "self_norm" and self.background is None:
                self.background = frame.copy()
                self._recompute_background_gray()

            if self.bg_mode == "self_norm" and self._self_norm_threshold is None:
                # Auto-pick the Li threshold from this first frame so tracking
                # starts calibrated without an explicit Auto click.
                try:
                    gray = self._resolve_gray(frame, None)
                    gp = self._prepare_gray(gray)
                    sigma = self._self_norm_sigma or max(1.0, gray.shape[0] / 20.0)
                    xf = gp.astype(np.float32)
                    bl = _wide_gaussian(xf, sigma)   # as the live path blurs
                    np.maximum(bl, 1.0, out=bl)
                    self._self_norm_threshold = float(threshold_li(xf / bl))
                    logger.info("Box %s: self_norm auto threshold = %.3f",
                                self.setup_id, self._self_norm_threshold)
                except Exception as e:
                    logger.debug("self_norm auto-threshold failed: %s", e)

            if self.bg_mode == "mog2" and self._mog2 is None:
                self._init_mog2()

            self.is_initialized = True
            self.tracking_active = True
            self.frame_count = 0
            self.success_count = 0
            self.start_time = time.time()
            self._reset_motion_model()
            logger.info(f"Box {self.setup_id}: Tracker initialized")
            return True

    def auto_initialize(self, frame):
        """Auto-initialize tracker (same as initialize for this tracker)."""
        return self.initialize(frame)

    def update(self, frame, timestamp=None, frame_gray=None):
        """Detect animal in frame, smoothed by OF+KF if enhancer set.

        Args:
            frame: New frame (BGR or grayscale)
            timestamp: Optional timestamp for callback

        Returns:
            tuple: (success, position) where position is (x, y, w, h) or None
        """
        with self.lock:
            if not self.is_initialized or not self.tracking_active:
                return False, None

            # self_norm needs NO background reference; every other mode does.
            if self.bg_mode != "self_norm" and self.background_gray is None:
                return False, None

            try:
                self.frame_count += 1
                gray = self._resolve_gray(frame, frame_gray)
                bg_gray = (None if self.bg_mode == "self_norm"
                           else self._aligned_background(gray))
                gray_processed = self._prepare_gray(gray)
                mask, chosen = self._detect_and_select(gray_processed, bg_gray)

                success, position, centroid = self._unpack_choice(chosen)
                success, position, centroid = self._apply_enhancer(
                    success, position, centroid, gray, timestamp)

                self._update_adaptive_freeze(success, centroid)
                self._record_quality(success, centroid)
                # Runs on hit AND miss (coast), and BEFORE last_centroid is
                # overwritten below, so the velocity EMA sees the true delta.
                self._update_motion_model(success, centroid)

                if not success:
                    return False, self.last_position

                self.success_count += 1
                self._update_running_background(mask, gray_processed)
                self.last_position = position
                self.last_centroid = centroid
                self._notify(position, timestamp)
                return True, position

            except Exception as e:
                logger.error(f"Box {self.setup_id}: Tracking error: {e}")
                return False, None

    # ── update steps ─────────────────────────────────────────────────
    # All are called from inside update's lock and its try, so none of them
    # locks or guards again.

    def _resolve_gray(self, frame, frame_gray):
        """The single-channel view to work from.

        Prefers the caller's pre-converted view: on a shared CCTV camera
        ``BoxFrame.image_gray`` is computed once per CameraFrame and handed to
        every box, so converting again here would repeat that work per box.
        """
        if frame_gray is not None:
            return frame_gray
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _aligned_background(self, gray):
        """The reference background at the frame's size.

        A resolution or ROI change makes the stored background a different
        view of the arena, not merely a different size, stretching it onto
        the new frame mis-registers every edge, so ``absdiff`` lights up along
        the arena walls and the largest blob is wherever the mismatch is
        worst. The resize keeps the box running (a per-frame raise would drop
        it), but it is a broken state and has to say so, once, loudly enough
        for the operator to re-capture.
        """
        bg_gray = self.background_gray
        if gray.shape != bg_gray.shape:
            if not self._bg_shape_warned:
                self._bg_shape_warned = True
                logger.warning(
                    "Box %s: background is %s but frames are %s, re-capture "
                    "the background for this box. Tracking until then is "
                    "unreliable: the stretched reference mis-registers and "
                    "detection can lock onto arena edges.",
                    self.setup_id, bg_gray.shape[:2], gray.shape[:2])
            return cv2.resize(bg_gray, (gray.shape[1], gray.shape[0]))
        self._bg_shape_warned = False
        return bg_gray

    def _detect_and_select(self, gray_processed, bg_gray):
        """``(mask, chosen_blob_or_None)``, spatial gate + temporal debounce.

        A blob outside the predicted gate is only committed once it has
        persisted near the same spot for a few frames AND is a plausible size
        against the recent track; otherwise the tracker coasts. Without that,
        one larger distractor teleports the centroid for a single frame.
        """
        mask, candidates = self._detect_contours(gray_processed, bg_gray)
        predicted = self._predicted_centroid if self.use_spatial_gating else None
        if self.bg_mode == "self_norm":
            self._sn_gray = gray_processed
            self._sn_frame_level = float(gray_processed.mean())
            self._note_static_sample(mask)
            # Recomputed against THIS frame's mask, not cached alongside the
            # reference: the animal has moved on since the last sample, and a
            # stale moving-region would no longer cover where it now is.
            self._sn_moving = (
                cv2.subtract(mask, self._sn_static)
                if (self._sn_static is not None
                    and self._sn_static.shape == mask.shape)
                else None)
        gate_px = self._compute_gate_px()
        chosen, in_gate = self._select_candidate(candidates, predicted, gate_px,
                                                 may_wait=True)
        if (chosen is not None and not in_gate and predicted is not None
                and not self._challenger_confirmed(chosen)):
            chosen = None
        self._check_track_plausibility(chosen)
        self._sn_have_track = chosen is not None
        return mask, chosen

    def _check_track_plausibility(self, chosen) -> None:
        """Drop the motion model when the committed blob stays the wrong size.

        Selection is anchored on the prediction, so a track that has latched
        onto a distractor keeps re-selecting it. Sustained disagreement with
        the recent median area is the signal that the anchor itself is wrong;
        clearing it forces the next frame to re-acquire on size alone.
        """
        med = self._median_recent_area()
        if chosen is None or med <= 0.0:
            return
        ratio = chosen[0] / med
        if self._area_ratio_lo <= ratio <= self._area_ratio_hi:
            self._implausible_streak = 0
            return
        self._implausible_streak += 1
        if self._implausible_streak >= self._max_implausible:
            logger.info(
                "Box %s: tracked blob has been implausibly sized "
                "(%.0f px vs median %.0f) for %d frames, re-acquiring",
                self.setup_id, chosen[0], med, self._implausible_streak)
            self._reset_motion_model()
            self._implausible_streak = 0

    def _unpack_choice(self, chosen):
        """``(success, position, centroid)``, caching the chosen blob's area
        and contour for the quality metrics."""
        if chosen is None:
            return False, None, None
        area, position, centroid, contour = chosen
        self._last_area = float(area)
        self._last_contour = contour
        self._challenger_pos = None
        self._challenger_streak = 0
        return True, position, centroid

    def _apply_enhancer(self, success, position, centroid, gray, timestamp):
        """Smooth the hit through optical flow + Kalman, or coast on a miss.

        The enhancer is fed the moments centroid, not the bounding-box centre,
        the box can jitter around a stable body. On a miss it can still predict
        a position, which is what keeps a briefly-occluded animal tracked.
        """
        if self.enhancer is None:
            return success, position, centroid
        ts_ns = int((timestamp or 0) * 1_000_000) if timestamp else 0
        if success and position and centroid is not None:
            _x, _y, w, h = position
            cx, cy = centroid
            result = self.enhancer.update(cx, cy, 1.0, gray, ts_ns,
                                          detected=True)
            if result is not None:
                scx, scy, _, _ = result
                centroid = (float(scx), float(scy))
                position = (int(scx - w / 2), int(scy - h / 2), w, h)
            return success, position, centroid
        if self.enhancer._initialized:
            result = self.enhancer.update(0, 0, 0.0, gray, ts_ns,
                                          detected=False)
            if result is not None and not self.enhancer.is_tracking_lost():
                scx, scy, _, _ = result
                lp = self.last_position
                w = lp[2] if lp else 20
                h = lp[3] if lp else 20
                centroid = (float(scx), float(scy))
                position = (int(scx - w / 2), int(scy - h / 2), w, h)
                success = True
        return success, position, centroid

    def _update_adaptive_freeze(self, success, centroid):
        """Stop an adaptive model learning a motionless animal into the
        background, the classic way a still subject silently disappears.
        Only MOG2 adapts; the other modes have nothing to freeze.
        """
        if self.bg_mode != "mog2":
            return
        if not success:
            # No detection: freeze rather than learn the void.
            self._adaptive_frozen = True
            return
        if centroid is None or self.last_centroid is None:
            return
        dx = centroid[0] - self.last_centroid[0]
        dy = centroid[1] - self.last_centroid[1]
        moved = (dx * dx + dy * dy) ** 0.5
        if moved < self._stationary_px_threshold:
            self._stationary_streak += 1
        else:
            self._stationary_streak = 0
        frozen = self._stationary_streak >= self._stationary_threshold_frames
        if frozen != self._adaptive_frozen:
            self._adaptive_frozen = frozen

    def _record_quality(self, success, centroid):
        """Rolling hit-rate, area and jitter for the calibration UI."""
        self._recent_results.append(1 if success else 0)
        if not (success and centroid is not None):
            return
        self._recent_centroids.append((float(centroid[0]), float(centroid[1])))
        if self._last_contour is not None:
            self._recent_areas.append(self._last_area)   # cached by detect
        if len(self._recent_centroids) >= 2:
            # deque -> 2-col float array; mean + variance computed in C.
            arr = np.asarray(self._recent_centroids, dtype=np.float64)
            d = arr - arr.mean(axis=0)
            self._jitter_px = float(np.sqrt((d * d).sum(axis=1).mean()))

    def _update_running_background(self, mask, gray_processed):
        """Accumulate the running-average reference, masking out the animal.

        The inverse-mask accumulate (Bonsai / EthoVision style) is what stops
        the subject's own pixels leaking into the reference it is detected
        against. Safe to overwrite here: this frame already read
        ``background_gray`` as its subtraction reference.
        """
        if self.bg_mode != "running_avg" or mask is None:
            return
        if (self._bg_running_f32 is None
                or self._bg_running_f32.shape != gray_processed.shape):
            self._bg_running_f32 = gray_processed.astype(np.float32)
        # Pooled buffer, avoids a per-frame HxW allocation.
        cv2.bitwise_not(mask, dst=self._bg_inv_buf)
        cv2.accumulateWeighted(gray_processed, self._bg_running_f32,
                               self._bg_running_alpha, mask=self._bg_inv_buf)
        if (self._bg_gray_buf is not None
                and self._bg_gray_buf.shape == gray_processed.shape):
            cv2.convertScaleAbs(self._bg_running_f32, dst=self._bg_gray_buf)
            self.background_gray = self._bg_gray_buf
        else:
            self.background_gray = cv2.convertScaleAbs(self._bg_running_f32)

    def _notify(self, position, timestamp):
        """Hand the box to the subscriber. A subscriber that raises must not
        cost the tracker its frame."""
        if not self.callback:
            return
        try:
            x, y, w, h = position
            self.callback(self.setup_id, x, y, w, h, timestamp)
        except Exception as e:
            logger.warning(f"Box {self.setup_id}: Callback error: {e}")

    def detect_with_mask(self, frame):
        """Run detection and return position + binary mask.

        Used by calibration dialog for live preview. Does NOT update
        statistics or last_position (read-only preview).

        Args:
            frame: Input frame (BGR or grayscale)

        Returns:
            tuple: (success, position, mask)
        """
        # self_norm needs NO background, refusing here would blank the
        # calibration preview for the exact no-background workflow the mode
        # exists for. Every other mode still requires a captured reference.
        if self.background_gray is None and self.bg_mode != "self_norm":
            h = frame.shape[0] if frame is not None else 100
            w = frame.shape[1] if frame is not None else 100
            return False, None, np.zeros((h, w), dtype=np.uint8)

        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            bg_gray = self.background_gray
            if bg_gray is not None and gray.shape != bg_gray.shape:
                bg_gray = cv2.resize(bg_gray, (gray.shape[1], gray.shape[0]))

            gray_processed = self._prepare_gray(gray)
            if self.bg_mode == "self_norm":
                # Brightness scoring reads these; a stale frame from the live
                # path would rank this frame's blobs by another frame's light.
                self._sn_gray = gray_processed
                self._sn_frame_level = float(gray_processed.mean())
            mask, candidates = self._detect_contours(gray_processed, bg_gray)
            # Preview is stateless, predicted=None so no spatial gate or
            # debounce, and live tracking state is untouched.
            chosen, _ = self._select_candidate(candidates, None, 0.0)
            success = chosen is not None
            position = chosen[1] if success else None
            # _detect_contours returns a view into the pooled ``_mask_buf``;
            # the UI displays it asynchronously, so hand it an owned copy.
            if mask is not None:
                mask = mask.copy()
            return success, position, mask

        except Exception as e:
            logger.error(f"Box {self.setup_id}: detect_with_mask error: {e}")
            h = frame.shape[0] if frame is not None else 100
            w = frame.shape[1] if frame is not None else 100
            return False, None, np.zeros((h, w), dtype=np.uint8)

    def stop(self):
        """Stop tracking."""
        with self.lock:
            self.tracking_active = False
            if self.start_time:
                elapsed = time.time() - self.start_time
                rate = (self.success_count / self.frame_count * 100
                        ) if self.frame_count > 0 else 0
                logger.info(f"Box {self.setup_id}: Stopped. "
                            f"Frames: {self.frame_count}, "
                            f"Success: {rate:.1f}%, "
                            f"Duration: {elapsed:.1f}s")

    def reset(self):
        """Reset tracker state."""
        with self.lock:
            self.is_initialized = False
            self.tracking_active = False
            self.last_position = None
            self.last_centroid = None
            self.frame_count = 0
            self.success_count = 0
            self.start_time = None
            self._mog2 = None
            self._bg_running_f32 = None
            self._reset_motion_model()
            logger.info(f"Box {self.setup_id}: Tracker reset")


class TrackerManager:
    """Manages the per-box blob trackers (pose lives in PoseSink).

    Provides centralized management of trackers across multiple boxes,
    including initialization, updates, zone integration, and statistics.

    Builds contour-based background-subtraction (blob) trackers; pose
    inference lives in ``PoseSink``.
    """

    def __init__(self):
        """Initialize tracker manager."""
        self.trackers = {}        # box_id -> BlobTracker
        self.backgrounds = {}     # box_id -> background frame
        self.zone_managers = {}   # box_id -> ZoneManager
        self._enhancers = {}      # box_id -> TrackingEnhancer (for pose mode)
        self.lock = threading.Lock()
        self.global_callback = None
        logger.info("TrackerManager initialized")

    def register_callback(self, callback):
        """Register global tracking callback.

        Args:
            callback: Function called with (box_id, x, y, w, h, timestamp)
        """
        self.global_callback = callback
        logger.info("Global tracking callback registered")

    def create_tracker(self, setup_id):
        """Return the box's BlobTracker, creating it on first use."""
        with self.lock:
            if setup_id in self.trackers:
                return self.trackers[setup_id]
            self.trackers[setup_id] = BlobTracker(setup_id, callback=self.global_callback)
            return self.trackers[setup_id]

    def get_tracker(self, setup_id):
        """Get blob tracker for a box.

        Args:
            setup_id: Box identifier

        Returns:
            BlobTracker instance, or None
        """
        return self.trackers.get(setup_id)

    def set_background(self, setup_id, frame):
        """Set background for a box and activate blob tracking.

        Auto-creates tracker if needed and activates it immediately -- no
        ROI initialization required.

        Args:
            setup_id: Box identifier
            frame: Background frame
        """
        with self.lock:
            self.backgrounds[setup_id] = frame.copy() if frame is not None else None
            tracker = self.trackers.get(setup_id)
            if tracker is None:
                # Auto-create blob tracker when background is set
                tracker = BlobTracker(setup_id, callback=self.global_callback)
                self.trackers[setup_id] = tracker
            tracker.set_background(frame)
            if frame is not None:
                # Blob tracker is ready as soon as it has a background
                tracker.is_initialized = True
                tracker.tracking_active = True
                if tracker.start_time is None:
                    tracker.start_time = time.time()
                logger.info(f"Box {setup_id}: Blob tracker activated with background")
            logger.info(f"Background set for box {setup_id}")

    def load_background(self, setup_id, filepath):
        """Load background image from file.

        Args:
            setup_id: Box identifier
            filepath: Path to load image from

        Returns:
            bool: True if successful
        """
        try:
            filepath = Path(filepath)
            if filepath.exists():
                bg = cv2.imread(str(filepath))
                if bg is not None:
                    self.set_background(setup_id, bg)
                    logger.info(f"Background loaded for box {setup_id}: {filepath}")
                    return True
            logger.warning(f"Failed to load background from {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading background for box {setup_id}: {str(e)}")
            return False

    def auto_initialize_tracker(self, setup_id, frame):
        """Auto-initialize tracker using background subtraction.

        BlobTracker just needs a background (auto-activated by set_background).

        Args:
            setup_id: Box identifier
            frame: Current frame

        Returns:
            bool: True if successful
        """
        # Reuse existing tracker, only create if missing
        tracker = self.trackers.get(setup_id)
        if tracker is None:
            tracker = self.create_tracker(setup_id)

        # Set background if available
        if setup_id in self.backgrounds:
            tracker.set_background(self.backgrounds[setup_id])

        return tracker.auto_initialize(frame)

    def update_tracker(self, setup_id, frame, timestamp=None, frame_gray=None):
        """Update tracker for a box.

        Args:
            setup_id: Box identifier
            frame: New frame (BGR)
            timestamp: Optional timestamp
            frame_gray: Optional pre-converted grayscale view of ``frame``,
                passed by TrackerSink as ``BoxFrame.image_gray`` so
                CCTV-shared trackers don't each ``cvtColor`` the same pixels.

        Returns:
            tuple: (success, position)
        """
        tracker = self.trackers.get(setup_id)
        if tracker:
            return tracker.update(frame, timestamp, frame_gray=frame_gray)
        return False, None

    def stop_tracker(self, setup_id):
        """Stop tracker for a box."""
        tracker = self.trackers.get(setup_id)
        if tracker:
            tracker.stop()

    # =====================================================================
    # Tracking Enhancer (Optical Flow + Kalman Filter)
    # =====================================================================

    def set_enhancer(self, setup_id, enhancer):
        """Set or clear the TrackingEnhancer for a box's tracker.

        The enhancer smooths centroid jitter via Kalman filter, validates
        detection jumps via optical flow, and predicts through brief
        occlusions. Also stored in ``_enhancers`` so pose mode can access
        it without going through the blob tracker.

        Args:
            setup_id: Box identifier
            enhancer: TrackingEnhancer instance, or None to disable
        """
        # Always store centrally (needed for pose mode access)
        if enhancer is not None:
            self._enhancers[setup_id] = enhancer
        else:
            self._enhancers.pop(setup_id, None)
        # Also attach to blob tracker if one exists
        tracker = self.trackers.get(setup_id)
        if tracker:
            tracker.enhancer = enhancer
        logger.info(f"Box {setup_id}: Enhancer {'set' if enhancer else 'cleared'}")

    def get_enhancer(self, setup_id):
        """Return the TrackingEnhancer for ``box_id``, or None if none
        was attached.  Used by TrackerSink/PoseSink for the
        latency-compensated zone-lookup forecast."""
        return self._enhancers.get(setup_id)

    # =====================================================================
    # Zone Manager Integration
    # =====================================================================

    def set_zone_manager(self, setup_id, zone_manager):
        """Set zone manager for a box.

        Args:
            setup_id: Box identifier
            zone_manager: ZoneManager instance for zone-based event processing
        """
        self.zone_managers[setup_id] = zone_manager
        logger.info(f"Zone manager set for box {setup_id}")

    def get_zone_manager(self, setup_id):
        """Get zone manager for a box.

        Args:
            setup_id: Box identifier

        Returns:
            ZoneManager instance or None
        """
        return self.zone_managers.get(setup_id)

