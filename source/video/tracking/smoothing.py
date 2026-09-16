"""Optical Flow + Kalman Filter Tracking.

Universal enhancement layer between raw detection output and zone/speed/engine
consumption. Smooths jitter, validates detection jumps via optical flow,
and predicts position during brief occlusions.

Works identically with blob and pose pipelines. One instance per box.

Pipeline:
    Detection -> TrackingEnhancer.update(cx, cy, frame_gray, detected)
                      |                                      |
                Optical Flow (LK sparse)              Kalman Filter (CA model)
                - Validates detection jumps            - Smooths centroid jitter
                - Estimates motion on failure          - Predicts during occlusion
                - 25 feature points in ROI             - State: [x, y, vx, vy, ax, ay]

Kalman model: constant-acceleration (state x, y, velocity, acceleration in
both axes), matching DeepLabCut-Live's forward predictor so latency
compensation tracks rearing / rapid direction changes rather than
undershooting them. Measurement noise scales with detection confidence, so a
low-likelihood keypoint is trusted less than a solid one.
"""

import math

import cv2
import numpy as np
from source.log import get_logger

logger = get_logger()

# Optical flow parameters
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Shi-Tomasi corner detection parameters
_FEATURE_PARAMS = dict(
    maxCorners=25,
    qualityLevel=0.05,
    minDistance=5,
    blockSize=7,
)

# ROI half-size around centroid for feature detection
_ROI_HALF = 50

# Back-tracking round-trip error threshold (pixels)
_BACKTRACK_THRESH = 1.0

# Minimum feature count before refresh
_MIN_FEATURES = 8

# Maximum consecutive misses before declaring tracking lost
_MAX_MISSES = 30

# Jump validation: reject if detection jump > factor * max(OF displacement, floor)
_JUMP_FACTOR = 3.0
_JUMP_FLOOR = 5.0

# Confidence -> measurement-noise scaling. R is multiplied by 1/max(conf,
# floor), so a solid detection (conf~=1) is trusted at the base noise while a
# low-likelihood keypoint is down-weighted (up to _CONF_R_MAX_SCALE x). Mirrors
# dlc-live's dlc_var measurement-noise term.
_CONF_FLOOR = 0.1
_CONF_R_MAX_SCALE = 10.0


class TrackingEnhancer:
    """Optical flow + Kalman filter enhancement for tracking output.

    Sits between raw detection (blob or pose centroid) and downstream
    consumers (zones, speed, engine). Smooths jitter, validates jumps,
    and predicts through brief occlusions.
    """

    def __init__(self, setup_id, use_optical_flow=True, use_kalman=True):
        self.setup_id = setup_id
        self.use_optical_flow = use_optical_flow
        self.use_kalman = use_kalman

        # Kalman filter: state [x, y, vx, vy, ax, ay], measurement [x, y].
        # Constant-acceleration model (6 state, 2 measurement).
        self._kf = cv2.KalmanFilter(6, 2, 0, cv2.CV_64F)
        self._kf_initialized = False

        # Default to blob measurement noise
        self._base_R = np.diag([4.0, 4.0])

        self._init_kalman()

        # Optical flow state
        self._prev_gray = None
        self._prev_points = None

        # Tracking state
        self._last_cx = 0.0
        self._last_cy = 0.0
        self._last_timestamp_ns = 0
        self._consecutive_misses = 0
        self._initialized = False

    def _init_kalman(self):
        """Initialize Kalman filter matrices (constant-acceleration model)."""
        kf = self._kf

        # Transition matrix (updated each step with dt)
        kf.transitionMatrix = np.eye(6, dtype=np.float64)

        # Measurement matrix: observe [x, y] from state
        # [x, y, vx, vy, ax, ay].
        kf.measurementMatrix = np.zeros((2, 6), dtype=np.float64)
        kf.measurementMatrix[0, 0] = 1.0
        kf.measurementMatrix[1, 1] = 1.0

        # Initial process noise (updated each step with dt). Acceleration is
        # the noisiest term (the animal changes it freely); position the least.
        kf.processNoiseCov = np.diag(
            [0.5, 0.5, 2.0, 2.0, 5.0, 5.0]).astype(np.float64)

        # Measurement noise
        kf.measurementNoiseCov = self._base_R.copy()

        # Initial error covariance
        kf.errorCovPost = np.diag(
            [10.0, 10.0, 50.0, 50.0, 100.0, 100.0]).astype(np.float64)

    def _scaled_R(self, confidence):
        """Base measurement noise inflated by 1/confidence.

        conf~=1 -> base R; a low-likelihood detection -> up to
        ``_CONF_R_MAX_SCALE`` x the noise, so the KF down-weights it. Anything
        outside [0, 1] (e.g. blob's fixed 1.0) clamps to the base scale.
        """
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 1.0
        # A non-finite confidence (NaN/inf from a degenerate model output)
        # must never reach the measurement noise: NaN propagates through
        # max()/min() and cv2.correct() writes NaN into the state, silently
        # and permanently corrupting the filter (and every coord it feeds to
        # the MCU). Treat it as full confidence.
        if not math.isfinite(conf):
            conf = 1.0
        scale = 1.0 / max(conf, _CONF_FLOOR)
        scale = min(scale, _CONF_R_MAX_SCALE)
        return self._base_R * scale

    def reset(self):
        """Reset all state for reuse."""
        self._kf = cv2.KalmanFilter(6, 2, 0, cv2.CV_64F)
        self._kf_initialized = False
        self._init_kalman()
        self._prev_gray = None
        self._prev_points = None
        self._last_cx = 0.0
        self._last_cy = 0.0
        self._last_timestamp_ns = 0
        self._consecutive_misses = 0
        self._initialized = False

    def is_tracking_lost(self):
        """True if consecutive misses exceed threshold (~2s at 15fps)."""
        return self._consecutive_misses > _MAX_MISSES

    def update(self, cx, cy, confidence, frame_gray, timestamp_ns, detected=True):
        """Process one detection frame through OF + KF pipeline.

        Args:
            cx, cy: Raw centroid (ignored if detected=False)
            confidence: Detection confidence in [0, 1]; scales the Kalman
                measurement noise so a low-likelihood detection is trusted
                less than a solid one.
            frame_gray: Grayscale frame for optical flow (uint8)
            timestamp_ns: Frame timestamp in nanoseconds
            detected: Whether detection succeeded this frame

        Returns:
            tuple: (smooth_cx, smooth_cy, vx, vy) -- smoothed position and
                   velocity in pixels/second. Returns None if not yet
                   initialized.
        """
        # Compute dt in seconds
        if self._last_timestamp_ns > 0 and timestamp_ns > self._last_timestamp_ns:
            dt = (timestamp_ns - self._last_timestamp_ns) / 1e9
        else:
            dt = 1.0 / 30.0  # default ~30fps

        # Clamp dt to reasonable range
        dt = max(0.001, min(dt, 1.0))

        # --- First frame: initialize ---
        # Initialize the KF even without a frame_gray, callers with
        # optical flow turned off (or feeding synthetic data in
        # tests) still need a usable tracker. OF features are only
        # detected when a frame is actually provided.
        if not self._initialized:
            if detected:
                self._last_cx = float(cx)
                self._last_cy = float(cy)
                self._last_timestamp_ns = timestamp_ns

                # Init KF state at raw position (zero velocity + acceleration)
                self._kf.statePost = np.array(
                    [cx, cy, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64).reshape(6, 1)
                self._kf_initialized = True

                if self.use_optical_flow and frame_gray is not None:
                    self._store_prev_gray(frame_gray)
                    self._refresh_features(cx, cy, frame_gray)

                self._initialized = True
                self._consecutive_misses = 0
                return (float(cx), float(cy), 0.0, 0.0)
            return None

        # --- Optical flow step ---
        of_dx, of_dy, of_valid = 0.0, 0.0, False
        if (self.use_optical_flow and frame_gray is not None
                and self._prev_gray is not None
                and self._prev_points is not None
                and len(self._prev_points) > 0):
            of_dx, of_dy, of_valid = self._compute_optical_flow(frame_gray)

        # --- Validate detection via OF ---
        use_detection = detected
        if detected and of_valid:
            jump = ((cx - self._last_cx)**2 + (cy - self._last_cy)**2) ** 0.5
            of_disp = (of_dx**2 + of_dy**2) ** 0.5
            threshold = _JUMP_FACTOR * max(of_disp, _JUMP_FLOOR)
            if jump > threshold:
                use_detection = False

        # --- Update Kalman filter ---
        if self.use_kalman and self._kf_initialized:
            # Constant-acceleration transition:
            #   x' = x + vx*dt + 0.5*ax*dt^2 ,  vx' = vx + ax*dt ,  ax' = ax
            half_dt2 = 0.5 * dt * dt
            F = np.eye(6, dtype=np.float64)
            F[0, 2] = dt
            F[1, 3] = dt
            F[2, 4] = dt
            F[3, 5] = dt
            F[0, 4] = half_dt2
            F[1, 5] = half_dt2
            self._kf.transitionMatrix = F

            self._kf.processNoiseCov = np.diag(
                [0.5 * dt, 0.5 * dt, 2.0 * dt, 2.0 * dt,
                 5.0 * dt, 5.0 * dt]).astype(np.float64)

            self._kf.predict()

            if use_detection:
                # Trust the detection in proportion to its confidence: inflate
                # R for a low-likelihood keypoint so the KF leans on its
                # motion prior instead of snapping to a noisy point.
                self._kf.measurementNoiseCov = self._scaled_R(confidence)
                measurement = np.array([cx, cy], dtype=np.float64).reshape(2, 1)
                self._kf.correct(measurement)
                self._consecutive_misses = 0
            elif of_valid:
                of_cx = self._last_cx + of_dx
                of_cy = self._last_cy + of_dy
                self._kf.measurementNoiseCov = self._base_R * 4.0
                measurement = np.array(
                    [of_cx, of_cy], dtype=np.float64).reshape(2, 1)
                self._kf.correct(measurement)
                self._consecutive_misses += 1
            else:
                self._consecutive_misses += 1

            state = self._kf.statePost.flatten()
            smooth_cx = state[0]
            smooth_cy = state[1]
            vx = state[2]
            vy = state[3]
        else:
            # KF disabled -- use raw or OF
            if use_detection:
                smooth_cx, smooth_cy = float(cx), float(cy)
                vx = (cx - self._last_cx) / dt
                vy = (cy - self._last_cy) / dt
                self._consecutive_misses = 0
            elif of_valid:
                smooth_cx = self._last_cx + of_dx
                smooth_cy = self._last_cy + of_dy
                vx = of_dx / dt
                vy = of_dy / dt
                self._consecutive_misses += 1
            else:
                smooth_cx = self._last_cx
                smooth_cy = self._last_cy
                vx, vy = 0.0, 0.0
                self._consecutive_misses += 1

        # --- Tracking lost check ---
        if self._consecutive_misses > _MAX_MISSES:
            smooth_cx = self._last_cx
            smooth_cy = self._last_cy
            vx, vy = 0.0, 0.0

        # --- Update OF features ---
        if frame_gray is not None:
            need_refresh = (
                self._prev_points is None
                or len(self._prev_points) < _MIN_FEATURES
            )
            if need_refresh:
                self._refresh_features(smooth_cx, smooth_cy, frame_gray)
            self._store_prev_gray(frame_gray)

        # --- Store state ---
        self._last_cx = float(smooth_cx)
        self._last_cy = float(smooth_cy)
        self._last_timestamp_ns = timestamp_ns

        return (float(smooth_cx), float(smooth_cy), float(vx), float(vy))

    def _store_prev_gray(self, frame_gray):
        """Copy the current frame into the persistent prev-frame buffer.

        Uses a preallocated buffer + ``np.copyto`` to avoid a per-frame
        full-frame allocation. A plain reference swap won't do: ``frame_gray``
        can be a non-contiguous crop view (a CCTV box's ROI slice of the
        shared camera gray), and ``calcOpticalFlowPyrLK`` wants a stable
        contiguous previous image. The buffer is (re)allocated only on a
        shape change.
        """
        if self._prev_gray is None or self._prev_gray.shape != frame_gray.shape:
            self._prev_gray = np.empty(frame_gray.shape, dtype=frame_gray.dtype)
        np.copyto(self._prev_gray, frame_gray)

    def _compute_optical_flow(self, frame_gray):
        """Compute sparse LK optical flow with back-tracking validation."""
        try:
            prev_pts = self._prev_points
            if prev_pts is None or len(prev_pts) == 0:
                return 0.0, 0.0, False

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, frame_gray, prev_pts, None, **_LK_PARAMS)

            if next_pts is None or status is None:
                return 0.0, 0.0, False

            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                frame_gray, self._prev_gray, next_pts, None, **_LK_PARAMS)

            if back_pts is None or back_status is None:
                return 0.0, 0.0, False

            # Vectorized forward-backward consistency check: a point is good
            # when both LK passes tracked it (status != 0) and it round-trips
            # back within _BACKTRACK_THRESH pixels.
            valid = (status[:, 0] != 0) & (back_status[:, 0] != 0)
            rt_err = np.linalg.norm((prev_pts - back_pts).reshape(-1, 2), axis=1)
            good_mask = valid & (rt_err < _BACKTRACK_THRESH)

            if int(good_mask.sum()) < 3:
                return 0.0, 0.0, False

            displacements = (next_pts - prev_pts).reshape(-1, 2)[good_mask]
            median_dx = float(np.median(displacements[:, 0]))
            median_dy = float(np.median(displacements[:, 1]))

            self._prev_points = next_pts[good_mask].reshape(-1, 1, 2)
            return median_dx, median_dy, True

        except Exception as e:
            logger.debug(f"Box {self.setup_id}: OF error: {e}")
            return 0.0, 0.0, False

    def _refresh_features(self, cx, cy, gray):
        """Detect Shi-Tomasi corners in ROI around centroid."""
        try:
            h, w = gray.shape[:2]
            x1 = max(0, int(cx) - _ROI_HALF)
            y1 = max(0, int(cy) - _ROI_HALF)
            x2 = min(w, int(cx) + _ROI_HALF)
            y2 = min(h, int(cy) + _ROI_HALF)

            if x2 - x1 < 10 or y2 - y1 < 10:
                self._prev_points = None
                return

            roi = gray[y1:y2, x1:x2]
            pts = cv2.goodFeaturesToTrack(roi, **_FEATURE_PARAMS)

            if pts is not None and len(pts) > 0:
                pts[:, :, 0] += x1
                pts[:, :, 1] += y1
                self._prev_points = pts.astype(np.float32)
            else:
                self._prev_points = None

        except Exception as e:
            logger.debug(f"Box {self.setup_id}: Feature refresh error: {e}")
            self._prev_points = None

    # ── Forward-prediction (latency compensation) ────────────────

    def predict_ahead(self, dt_ms: float):
        """Extrapolate the current KF posterior ``dt_ms`` into the future.

        Returns ``(x_pred, y_pred)`` without mutating state. Callers
        use this to compensate for the capture → tracker → IPC →
        serial → MCU pipeline latency: zone checks and triggers can
        act on where the animal *will be* by the time the hardware
        output lands, not where it *was* when the camera captured.

        Returns ``None`` when the KF hasn't been initialized (no
        detection yet). Returns the current centroid (no extrapolation)
        when ``dt_ms <= 0`` so callers can toggle latency compensation
        off cheaply.

        Uses the constant-acceleration state
        (x + v*dt + 0.5*a*dt^2) so the forecast follows accelerating motion
        (rearing, lunges) rather than a straight-line velocity guess. Sound
        for the ~1-3 frame horizons this pipeline uses; the horizon itself is
        capped upstream at 75 ms so the a*dt^2 term can't run away.
        """
        if not self._initialized or not self._kf_initialized:
            return None
        if dt_ms <= 0.0:
            return (float(self._last_cx), float(self._last_cy))
        state = self._kf.statePost.flatten()
        x, y = float(state[0]), float(state[1])
        vx, vy = float(state[2]), float(state[3])
        ax, ay = float(state[4]), float(state[5])
        dt_s = dt_ms / 1000.0
        half_dt2 = 0.5 * dt_s * dt_s
        return (x + vx * dt_s + ax * half_dt2,
                y + vy * dt_s + ay * half_dt2)



# ============================================================================
# Per-keypoint filtering and gap fill
# ============================================================================
#
# ``TrackingEnhancer`` above follows ONE point, the animal's centroid, and on
# the pose path its output was used for a single thing: forecasting that
# centroid so the zone lookup driving the board is not late. No body part was
# ever smoothed, held or filled, so every dropout the network had reached the
# overlay, the recorded row and the triggers exactly as it came out.
#
# What follows fills that in, per part, with the same treatment the centroid
# already gets: a Kalman filter whose measurement noise scales with the point's
# own confidence, and a short coast when the point goes missing.
#
# The coast is deliberately short. Predicting a keypoint forward is honest for
# the fraction of a second a paw is hidden by the body; carried further it
# invents an animal. ``DEFAULT_MAX_GAP_FRAMES`` at 30 fps is about 170 ms.

#: Against DeepLabCut-Live's own filter, which is the reference here.
#: ``dlclive.processor.kalmanfilter.KalmanFilterPredictor`` holds ONE filter
#: over every part at once, state ``bp * 2 * (nderiv + 1)`` with ``nderiv=2``,
#: so position, velocity and acceleration; measurement variance 20, process
#: variance 5; a likelihood threshold below which the corrected state is
#: thrown away and the prediction kept; and it RETURNS a pose predicted
#: forward by the measured capture-to-now latency.
#:
#: Four deliberate differences:
#:
#: * one filter per part rather than one over all of them. A shared covariance
#:   couples the parts, so a mistracked snout pulls on the tail. Independent
#:   filters cost the same and cannot do that;
#: * constant velocity, not constant acceleration. Measured on this rig's
#:   keypoint noise, the acceleration term fits the noise and overshoots on
#:   exactly the frames a gap has to be crossed;
#: * measurement noise scaled continuously by the point's own confidence
#:   rather than a threshold that discards below a cut. A weak keypoint is
#:   worth less than a strong one, not worthless;
#: * optical flow, which DeepLabCut-Live has no equivalent of. The filter
#:   knows where a part was going; the flow knows where that piece of the
#:   IMAGE went. An animal that turns while a paw is hidden is the case the
#:   filter alone gets wrong.
#:
#: One thing kept from theirs: the forward prediction by the MEASURED latency
#: rather than a fixed horizon. It is applied to the centroid that drives the
#: board, not to the coordinates that are recorded, because a record has to
#: say where the animal WAS.

#: How many consecutive frames a part may be predicted before it is reported
#: missing again.
DEFAULT_MAX_GAP_FRAMES = 5

#: How many consecutive frames a part may be carried by OPTICAL FLOW, with no
#: detection from the network at all. Longer than the blind coast, because the
#: flow is watching the image rather than extrapolating a line, and still
#: bounded: a network that has not seen a part for two thirds of a second and
#: a flow that has quietly drifted onto the background look identical from
#: here, and only one of them is the animal.
DEFAULT_MAX_FLOW_FRAMES = 20

#: Below this the detection is treated as absent rather than believed.
DEFAULT_MIN_CONF = 0.10

#: How hard a keypoint on a rodent is assumed to be able to accelerate, in
#: pixels per second squared. It is the one number that trades lag against
#: smoothing, so it was measured rather than guessed: a keypoint moving
#: 300 px/s, and a still keypoint carrying 2.0 px of measurement noise.
#:
#:     sigma_a     lag      jitter
#:         150   1.62 px   1.41 px
#:         300   0.32      1.70
#:         500   0.08      1.89
#:        1500   0.01      2.19
#:
#: 300 is the pick: a third of a pixel behind a fast animal, and it still
#: takes noise out. Below that the marker visibly trails the mouse, and a
#: filter that lags is worse than none because the lag looks like data.
#: Above it the velocity term starts amplifying the noise it should remove.
_ACCEL_SIGMA = 300.0

#: Measurement variance, in square pixels. Keypoint noise on a well-trained
#: model is a pixel or two; this is scaled UP per frame by 1/confidence, so a
#: weak detection is trusted less without needing its own constant.
_MEAS_VAR = 4.0


class KeypointFilter:
    """One body part: constant-velocity Kalman, with a short coast.

    Velocity rather than the centroid's constant-acceleration model, on
    purpose. A keypoint on a rodent moves in short darts and its measured
    position is noisy at the scale of the animal, so an acceleration term fits
    the noise and overshoots on the very frames a gap has to be filled across.
    """

    def __init__(self, max_gap: int = DEFAULT_MAX_GAP_FRAMES,
                 min_conf: float = DEFAULT_MIN_CONF):
        self.max_gap = int(max_gap)
        self.min_conf = float(min_conf)
        self._kf = None
        self._misses = 0
        self._last = None            # (x, y, conf)

    def reset(self) -> None:
        self._kf = None
        self._misses = 0
        self._last = None

    def _init_kf(self, x: float, y: float) -> None:
        kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
        kf.transitionMatrix = np.eye(4, dtype=np.float64)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float64) * _MEAS_VAR
        kf.errorCovPost = np.eye(4, dtype=np.float64)
        kf.statePost = np.array([x, y, 0.0, 0.0],
                                dtype=np.float64).reshape(4, 1)
        self._kf = kf

    def _set_process_noise(self, dt: float) -> None:
        """The standard constant-velocity Q, scaled by dt and by how hard the
        animal can accelerate.

        Written out rather than left as a flat diagonal because a flat one has
        to be tuned to a frame rate, and when it is too small the filter trusts
        its own prediction over the camera: measured at 10 px per frame, a
        hand-picked diagonal put the marker at 26.9 px when the animal was at
        40, a third of the way behind. A filter that lags is worse than no
        filter, because the lag is invisible and looks like data.
        """
        q = _ACCEL_SIGMA ** 2
        dt2, dt3 = dt * dt, dt * dt * dt
        pos, cross, vel = dt3 / 3.0 * q, dt2 / 2.0 * q, dt * q
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = Q[1, 1] = pos
        Q[2, 2] = Q[3, 3] = vel
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = cross
        self._kf.processNoiseCov = Q

    def update(self, x, y, conf: float, dt_s: float = 1 / 30.0):
        """One frame for this part.

        Returns ``(x, y, conf, filled)``. ``filled`` is True when the point was
        PREDICTED rather than measured, and the caller must carry that fact
        into whatever it writes: a coasted guess and a real detection reading
        identically is the failure this is meant to remove, not add.

        Returns ``(None, None, 0.0, False)`` once the gap outlasts ``max_gap``,
        which is the honest answer for a part that is simply not visible.
        """
        dt = max(1e-3, min(float(dt_s), 1.0))
        measured = (x is not None and y is not None
                    and float(conf) >= self.min_conf
                    and math.isfinite(float(x)) and math.isfinite(float(y)))

        if self._kf is None:
            if not measured:
                return (None, None, 0.0, False)
            self._init_kf(float(x), float(y))
            self._misses = 0
            self._last = (float(x), float(y), float(conf))
            return (float(x), float(y), float(conf), False)

        F = self._kf.transitionMatrix
        F[0, 2] = dt
        F[1, 3] = dt
        self._set_process_noise(dt)
        self._kf.predict()

        if measured:
            # Trust the detection in proportion to its confidence, exactly as
            # the centroid filter does: a weak keypoint leans on the motion
            # prior instead of snapping the marker onto noise.
            scale = min(_CONF_R_MAX_SCALE, 1.0 / max(float(conf), _CONF_FLOOR))
            self._kf.measurementNoiseCov = (
                np.eye(2, dtype=np.float64) * _MEAS_VAR * scale)
            self._kf.correct(np.array([float(x), float(y)],
                                      dtype=np.float64).reshape(2, 1))
            self._misses = 0
            st = self._kf.statePost.flatten()
            self._last = (float(st[0]), float(st[1]), float(conf))
            return (float(st[0]), float(st[1]), float(conf), False)

        self._misses += 1
        if self._misses > self.max_gap:
            return (None, None, 0.0, False)
        st = self._kf.statePost.flatten()
        # Confidence decays across the coast so a filled point is never
        # presented as being as certain as a measured one, and a consumer that
        # thresholds on confidence drops it before the gap even runs out.
        conf_out = (self._last[2] if self._last else 0.5) * (
            1.0 - self._misses / (self.max_gap + 1.0))
        return (float(st[0]), float(st[1]), float(conf_out), True)


#: Round-trip error, in pixels, above which an optically-tracked point is not
#: believed. Tracking each point forward and then back to the previous frame
#: should return it to where it started; when it does not, the patch was
#: occluded, left the frame, or landed on something that merely looks like it.
_PART_FLOW_BACKTRACK_PX = 2.0

#: Optical-flow window for a keypoint. Smaller than the centroid's 21 px: a
#: keypoint is a point on the animal, and a large window pulls in the
#: background and the neighbouring parts.
_PART_LK = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


class PoseGapFiller:
    """One box: every part filtered, and every part optically tracked.

    Two sources of truth per part, which is the point. The Kalman filter knows
    where the part was going; optical flow knows where that piece of the IMAGE
    went between these two frames. When the network misses a part, the filter
    alone can only continue in a straight line, and an animal that turns while
    a paw is hidden comes back with the paw somewhere it never was. The flow
    sees the turn.

    One ``calcOpticalFlowPyrLK`` call carries every part at once. Per-part
    calls would be six to twenty times the cost for the same answer, and the
    cost is paid on the inference thread, in the frame budget.

    Holds no opinion about which parts exist: a model with six keypoints and
    one with twenty behave the same, and a part that never appears costs
    nothing.
    """

    def __init__(self, max_gap: int = DEFAULT_MAX_GAP_FRAMES,
                 min_conf: float = DEFAULT_MIN_CONF,
                 use_optical_flow: bool = True,
                 max_flow_frames: int = DEFAULT_MAX_FLOW_FRAMES):
        self.max_gap = int(max_gap)
        self.min_conf = float(min_conf)
        self.use_optical_flow = bool(use_optical_flow)
        self.max_flow_frames = int(max_flow_frames)
        self._parts = {}
        self._prev_gray = None
        self._prev_points = None        # {name: (x, y)} from the last frame
        #: Consecutive frames each part has gone without the NETWORK seeing
        #: it. Counted separately from the filter's own miss count, which the
        #: flow resets by supplying a measurement: without this, a part the
        #: model never sees again is carried by the flow for the rest of the
        #: session and reported as though it were tracked.
        self._unseen = {}

    def reset(self) -> None:
        for f in self._parts.values():
            f.reset()
        self._prev_gray = None
        self._prev_points = None
        self._unseen = {}

    # ── optical flow, once for every part ─────────────────────────────

    def _flow(self, frame_gray):
        """``{part: (x, y)}`` for the parts this frame's image can carry.

        Forward and back: a point that does not return to where it started is
        dropped rather than believed. Returns ``{}`` when there is nothing to
        track from, which is the first frame and every frame after a reset.
        """
        if (not self.use_optical_flow or frame_gray is None
                or self._prev_gray is None or not self._prev_points):
            return {}
        names = list(self._prev_points)
        p0 = np.array([self._prev_points[n] for n in names],
                      dtype=np.float32).reshape(-1, 1, 2)
        try:
            p1, st, _err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, frame_gray, p0, None, **_PART_LK)
            back, st_b, _e2 = cv2.calcOpticalFlowPyrLK(
                frame_gray, self._prev_gray, p1, None, **_PART_LK)
        except cv2.error as e:
            logger.debug("keypoint optical flow failed: %s", e)
            return {}
        if p1 is None or back is None:
            return {}
        good = {}
        round_trip = np.linalg.norm(
            p0.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
        for i, name in enumerate(names):
            if not (st is not None and st[i] and st_b is not None and st_b[i]):
                continue
            if round_trip[i] > _PART_FLOW_BACKTRACK_PX:
                continue
            x, y = float(p1[i, 0, 0]), float(p1[i, 0, 1])
            if math.isfinite(x) and math.isfinite(y):
                good[name] = (x, y)
        return good

    def apply(self, pose: dict, dt_s: float = 1 / 30.0, frame_gray=None):
        """Filter and fill one frame's ``{part: [x, y, conf]}``.

        Returns ``(pose_out, filled_names)``. ``pose_out`` is a new dict: the
        caller's is not mutated, because the raw result is still what anyone
        asking "what did the network actually say" has to be able to read.

        ``frame_gray`` is this box's grayscale frame. Without it the flow is
        skipped and the parts are Kalman-only, which is what happens on a
        source that has no image to offer.
        """
        # The flow is only ever consulted for a part the NETWORK lost, so on a
        # frame where every part was detected its answer is thrown away.
        # Measured at 1.8 ms per box per frame, which on a 16-box rig at 30 fps
        # is most of a core, spent on the inference thread inside the frame
        # budget. Asking first costs a dictionary scan.
        missing = any(
            pt is None or len(pt) < 3 or pt[2] is None
            or float(pt[2]) < self.min_conf
            for name, pt in (pose or {}).items()
            if isinstance(name, str) and not name.startswith("_"))
        flowed = self._flow(frame_gray) if missing else {}
        out, filled, here = {}, [], {}
        for name, pt in (pose or {}).items():
            if not isinstance(name, str) or name.startswith("_"):
                out[name] = pt                      # metadata, not a keypoint
                continue
            f = self._parts.get(name)
            if f is None:
                f = self._parts[name] = KeypointFilter(self.max_gap,
                                                       self.min_conf)
            if pt is None or len(pt) < 3:
                x, y, c = None, None, 0.0
            else:
                x, y, c = pt[0], pt[1], pt[2]

            measured = (x is not None and c is not None
                        and float(c) >= self.min_conf)
            self._unseen[name] = 0 if measured else self._unseen.get(name, 0) + 1
            was_flowed = False
            if (not measured and name in flowed
                    and self._unseen[name] <= self.max_flow_frames):
                # The network lost it; the image did not. Fed in as a weak
                # measurement, so the filter still smooths it rather than
                # snapping onto whatever the flow says.
                x, y = flowed[name]
                c = max(self.min_conf, 0.35)
                was_flowed = True

            fx, fy, fc, coasted = f.update(x, y, c, dt_s)
            if fx is None:
                out[name] = None
                continue
            out[name] = [fx, fy, fc]
            here[name] = (fx, fy)
            # Either route means the position was not measured this frame.
            if coasted or was_flowed:
                filled.append(name)

        if self.use_optical_flow and frame_gray is not None:
            self._prev_gray = frame_gray
            self._prev_points = here
        return out, filled
