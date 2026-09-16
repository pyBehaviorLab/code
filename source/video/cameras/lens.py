"""Lens / field-of-view correction, one module, three responsibilities.

A wide-angle or cheap USB lens bows straight arena walls into curves, so a
mouse near a corner is reported several pixels from where it truly is. That
bend propagates into every pose coordinate, zone crossing and distance number
the pipeline computes. This module removes it once, upstream of everything,
so tracking, recording, the live preview and zone tests all share one straight
coordinate space.

Three parts, kept together because they are one feature:

* **Estimation (off the hot path).** :func:`find_board` locates a checkerboard
  in a frame; :func:`solve` feeds a set of detections to ``cv2.calibrateCamera``
  and returns a :class:`CalibrationProfile` (camera matrix ``K`` + distortion
  coefficients + a quality number). Runs for a few seconds when the user
  calibrates a camera.

* **Correction (every frame).** :class:`Undistorter` bakes a profile into two
  fixed-point (``CV_16SC2``) remap maps via ``initUndistortRectifyMap``, the
  polynomial maths runs once at build time; each frame is then a single
  memory-bound ``cv2.remap`` (~0.3-0.9 ms at 720p). Maps rebuild only if the
  frame size changes; a profile calibrated at one resolution is reused at
  another by scaling ``K``.

* **Persistence + wiring.** :class:`LensCalibrationStore` keeps profiles in a
  machine-level JSON keyed by stable USB identity (distortion is a property of
  the lens, not the experiment), and :class:`LensCorrectionCache` turns a
  ``camera_id`` into "the undistorter for this camera, or ``None``" with all
  the expensive work done lazily and once.

The estimation + correction half is headless (NumPy + OpenCV only) and
unit-tested with synthetic boards; the cache injects its dependencies so it is
testable without a camera.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from source.log import get_logger

try:  # OpenCV is a hard runtime dep everywhere the pipeline runs.
    import cv2
except Exception:  # pragma: no cover - only in a stripped test env
    cv2 = None  # type: ignore

logger = get_logger()

STORE_FILENAME = "camera_lens.json"
STORE_VERSION = 1


# ── board specification ─────────────────────────────────────────────────

@dataclass(frozen=True)
class BoardSpec:
    """A checkerboard's geometry.

    ``cols`` / ``rows`` are the number of *inner* corners (where four squares
    meet), NOT the number of squares, a board printed as a 10x7 grid of
    squares has 9x6 inner corners. ``square_mm`` is the side of one square in
    millimetres; it only scales the (discarded) extrinsics, so it does not
    affect the distortion correction, but it is stored to identify the run and to
    let a future metric-scale feature reuse the same capture.
    """

    cols: int = 9
    rows: int = 6
    square_mm: float = 25.0

    @property
    def inner_corners(self) -> int:
        return self.cols * self.rows

    def to_json(self) -> dict:
        return {"cols": self.cols, "rows": self.rows, "square_mm": self.square_mm}

    @classmethod
    def from_json(cls, d: dict) -> "BoardSpec":
        d = d or {}
        return cls(
            cols=int(d.get("cols", 9)),
            rows=int(d.get("rows", 6)),
            square_mm=float(d.get("square_mm", 25.0)),
        )


#: Printable paper sizes, in millimetres.
PAPER_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0),
            "Letter": (215.9, 279.4)}


def render_board(board: BoardSpec, *, paper: str = "A4", dpi: int = 300,
                 landscape: Optional[bool] = None,
                 caption: bool = True) -> np.ndarray:
    """A printable image of ``board``, at true scale for the given paper.

    The squares come out ``board.square_mm`` across when the file is printed at
    100 %, which is the whole point: the calibration solves for lens distortion
    from the corner geometry, and a board scaled to "fit page" by the print
    dialog has square sizes that no longer match what was asked for.

    ``cols`` / ``rows`` are inner corners, so the drawn grid is one square
    larger in each direction. The board is surrounded by a white margin because
    ``findChessboardCorners`` needs a quiet border to locate the outer squares.

    Returns a BGR image. Raises ``ValueError`` if the board cannot fit the page
    at true scale, rather than silently shrinking it, which would produce a
    board whose printed squares are not the size written on it.
    """
    if paper not in PAPER_MM:
        raise ValueError(f"unknown paper {paper!r}; known: {sorted(PAPER_MM)}")
    page_w_mm, page_h_mm = PAPER_MM[paper]
    if landscape is None:
        # The standard 9x6 board at 25 mm is 250 mm wide, which does not fit A4
        # portrait and does fit it turned. Choosing the orientation that works
        # is more useful than refusing and making the operator work out why.
        need_mm = (board.cols + 1) * board.square_mm + 20.0
        landscape = need_mm > page_w_mm and need_mm <= page_h_mm
    if landscape:
        page_w_mm, page_h_mm = page_h_mm, page_w_mm

    px_per_mm = float(dpi) / 25.4
    sq_px = int(round(board.square_mm * px_per_mm))
    n_x, n_y = board.cols + 1, board.rows + 1          # squares, not corners
    grid_w, grid_h = n_x * sq_px, n_y * sq_px
    page_w = int(round(page_w_mm * px_per_mm))
    page_h = int(round(page_h_mm * px_per_mm))

    # A quiet border for the detector, and room for the caption. The border is
    # preferred generous and allowed to shrink to the minimum that still lets
    # findChessboardCorners see the outer squares: refusing a board that misses
    # the page by two millimetres of whitespace helps nobody.
    caption_px = int(round(10 * px_per_mm)) if caption else 0
    margin = 0
    for want_mm in (sq_px / px_per_mm / 2.0, 10.0, 8.0, 6.0, 5.0):
        cand = int(round(want_mm * px_per_mm))
        if (grid_w + 2 * cand <= page_w
                and grid_h + 2 * cand + caption_px <= page_h):
            margin = cand
            break
    if margin == 0:
        margin = int(round(5 * px_per_mm))
    if grid_w + 2 * margin > page_w or grid_h + 2 * margin + caption_px > page_h:
        need_w = (grid_w + 2 * margin) / px_per_mm
        need_h = (grid_h + 2 * margin + caption_px) / px_per_mm
        raise ValueError(
            f"a {board.cols}x{board.rows} board of {board.square_mm:g} mm "
            f"squares needs {need_w:.0f}x{need_h:.0f} mm and {paper}"
            f"{' landscape' if landscape else ''} is "
            f"{page_w_mm:.0f}x{page_h_mm:.0f} mm. Use larger paper, a smaller "
            f"square, or fewer corners.")

    page = np.full((page_h, page_w, 3), 255, np.uint8)
    x0 = (page_w - grid_w) // 2
    y0 = (page_h - caption_px - grid_h) // 2
    for iy in range(n_y):
        for ix in range(n_x):
            if (ix + iy) % 2 == 0:
                continue
            y, x = y0 + iy * sq_px, x0 + ix * sq_px
            page[y:y + sq_px, x:x + sq_px] = 0

    if caption:
        text = (f"{board.cols}x{board.rows} inner corners  |  "
                f"{board.square_mm:g} mm squares  |  {paper} at {dpi} dpi  |  "
                f"PRINT AT 100% (no scaling), mount flat")
        scale = max(0.4, page_w / 2400.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(page, text, ((page_w - tw) // 2,
                                 y0 + grid_h + int(th) + margin // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return page


def object_points(board: BoardSpec) -> np.ndarray:
    """Ideal 3-D coordinates of the board's inner corners (Z=0 plane), ordered
    to match ``find_board``'s corner ordering. Units are millimetres when
    ``square_mm`` is set in mm."""
    grid = np.zeros((board.inner_corners, 3), np.float32)
    grid[:, :2] = np.mgrid[0:board.cols, 0:board.rows].T.reshape(-1, 2)
    grid *= float(board.square_mm)
    return grid


# ── detection ───────────────────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def find_board(frame: np.ndarray, board: BoardSpec) -> Optional[np.ndarray]:
    """Locate the checkerboard inner corners in ``frame``.

    Returns an ``(N, 1, 2)`` float32 array of sub-pixel corner locations, or
    ``None`` if no complete board is visible. Prefers
    ``findChessboardCornersSB`` (robust to blur and lighting, already
    sub-pixel); falls back to the classic detector + ``cornerSubPix`` on
    OpenCV builds without SB.
    """
    if cv2 is None:
        return None
    gray = _to_gray(frame)
    size = (board.cols, board.rows)

    sb = getattr(cv2, "findChessboardCornersSB", None)
    if sb is not None:
        try:
            ok, corners = sb(gray, size)
            if ok and corners is not None:
                return corners.astype(np.float32)
        except cv2.error:
            pass  # fall through to classic detector

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, size, flags)
    if not ok or corners is None:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return corners.astype(np.float32)


def board_signature(corners: np.ndarray, image_size: Tuple[int, int],
                    grid: int = 3) -> Tuple[int, int, int]:
    """A coarse pose fingerprint used by the wizard to spread captures out.

    Returns ``(cell_x, cell_y, size_bucket)``: which cell of a ``grid x grid``
    partition the board centre falls in, plus how much of the frame the board
    fills (0 = small/far, 2 = large/near). Two captures with the same
    signature carry almost the same information, so the wizard keeps only
    diverse ones, good calibration needs the board seen across the *whole*
    frame, especially the corners where distortion is worst.
    """
    w, h = image_size
    pts = corners.reshape(-1, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    cell_x = min(grid - 1, max(0, int(cx / max(1, w) * grid)))
    cell_y = min(grid - 1, max(0, int(cy / max(1, h) * grid)))
    span = (pts.max(axis=0) - pts.min(axis=0))
    frac = (span[0] * span[1]) / max(1.0, float(w * h))
    size_bucket = 0 if frac < 0.06 else (1 if frac < 0.18 else 2)
    return (cell_x, cell_y, size_bucket)


def sharpness(frame: np.ndarray) -> float:
    """Variance-of-Laplacian focus measure, higher is sharper. The wizard
    rejects blurred board views (motion / defocus) before they poison the
    solve."""
    if cv2 is None:
        return 0.0
    gray = _to_gray(frame)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── profile ─────────────────────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """The estimated lens model for one camera.

    ``camera_matrix`` is the 3x3 intrinsic matrix ``K``; ``dist_coeffs`` are
    the ``[k1, k2, p1, p2, k3]`` distortion coefficients. ``image_size`` is
    the ``(width, height)`` the calibration was performed at, needed to scale
    ``K`` when the camera later runs at a different resolution. ``rms_error``
    is the mean reprojection error in pixels (under ~0.5 is excellent, over
    ~1.0 warrants a redo).
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    rms_error: float
    board: BoardSpec = field(default_factory=BoardSpec)
    n_views: int = 0
    created: str = ""

    def __post_init__(self):
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        if not self.created:
            self.created = datetime.now().isoformat(timespec="seconds")

    @property
    def quality(self) -> str:
        """A plain verdict for the UI, not raw matrices."""
        if self.rms_error <= 0.5:
            return "excellent"
        if self.rms_error <= 1.0:
            return "good"
        return "poor"

    def scaled_matrix(self, target_size: Tuple[int, int]) -> np.ndarray:
        """``K`` rescaled from ``image_size`` to ``target_size``.

        Focal lengths and the principal point scale with resolution; the
        distortion coefficients are in normalized units and need no change as
        long as the aspect ratio is preserved.
        """
        tw, th = int(target_size[0]), int(target_size[1])
        sw, sh = self.image_size
        if (tw, th) == (sw, sh):
            return self.camera_matrix.copy()
        sx = tw / float(sw)
        sy = th / float(sh)
        K = self.camera_matrix.copy()
        K[0, 0] *= sx  # fx
        K[0, 2] *= sx  # cx
        K[1, 1] *= sy  # fy
        K[1, 2] *= sy  # cy
        return K

    def to_json(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "rms_error": float(self.rms_error),
            "board": self.board.to_json(),
            "n_views": int(self.n_views),
            "created": self.created,
        }

    @classmethod
    def from_json(cls, d: dict) -> "CalibrationProfile":
        return cls(
            camera_matrix=np.asarray(d["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(d["dist_coeffs"], dtype=np.float64),
            image_size=tuple(d["image_size"]),
            rms_error=float(d.get("rms_error", 0.0)),
            board=BoardSpec.from_json(d.get("board")),
            n_views=int(d.get("n_views", 0)),
            created=str(d.get("created") or ""),
        )


def solve(corner_sets: Sequence[np.ndarray],
          image_size: Tuple[int, int],
          board: BoardSpec) -> CalibrationProfile:
    """Estimate the lens model from a set of board detections.

    ``corner_sets`` is a list of ``find_board`` results (each ``(N,1,2)``),
    ``image_size`` is ``(width, height)``. Raises ``ValueError`` if there are
    too few views to solve reliably (need at least 3; ~15+ is recommended).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for calibration")
    if len(corner_sets) < 3:
        raise ValueError(
            f"need at least 3 board views to calibrate, got {len(corner_sets)}")

    objp = object_points(board)
    obj_points = [objp for _ in corner_sets]
    img_points = [c.astype(np.float32) for c in corner_sets]
    w, h = int(image_size[0]), int(image_size[1])

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)
    return CalibrationProfile(
        camera_matrix=K,
        dist_coeffs=dist,
        image_size=(w, h),
        rms_error=float(rms),
        board=board,
        n_views=len(corner_sets),
    )


# ── correction (hot path) ───────────────────────────────────────────────

class Undistorter:
    """Applies a :class:`CalibrationProfile` to frames at (near) zero cost.

    Construction precomputes two fixed-point remap maps for a given frame
    size; :meth:`apply` is then a single ``cv2.remap``. ``alpha=0`` (the
    default) zooms slightly so every output pixel is valid, no black border
    that would confuse tracking; ``alpha=1`` keeps the full field of view but
    introduces invalid edges (used only by the wizard's review pane).

    :meth:`apply` transparently rebuilds the maps if handed a frame of a
    different size than it was built for, so a resolution change mid-session is
    handled without the caller thinking about it.
    """

    def __init__(self, profile: CalibrationProfile,
                 frame_size: Tuple[int, int], alpha: float = 0.0):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for undistortion")
        self.profile = profile
        self.alpha = float(alpha)
        self._size: Tuple[int, int] = (0, 0)
        self._map1 = None
        self._map2 = None
        self._new_K: Optional[np.ndarray] = None
        self._build(int(frame_size[0]), int(frame_size[1]))

    def _build(self, w: int, h: int) -> None:
        K = self.profile.scaled_matrix((w, h))
        dist = self.profile.dist_coeffs
        # A camera matrix cropped to keep only valid pixels (alpha=0) or the
        # whole FOV (alpha=1). The maths below runs ONCE per size.
        new_K, _roi = cv2.getOptimalNewCameraMatrix(
            K, dist, (w, h), self.alpha, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (w, h), cv2.CV_16SC2)
        self._map1, self._map2 = map1, map2
        self._new_K = new_K
        self._size = (w, h)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return an undistorted copy of ``frame`` (same size and dtype)."""
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            self._build(w, h)
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    @property
    def size(self) -> Tuple[int, int]:
        return self._size


# ── persistence ─────────────────────────────────────────────────────────

def _store_dir() -> Path:
    """The machine-level config dir, reusing the capability cache's location
    (``<user config>/pybehaviorlab``) so both live together."""
    from source.video.cameras.calibration_store import store_path
    return store_path().parent


@dataclass(frozen=True)
class CalibrationEntry:
    """One camera's stored calibration: its identity, a human label, the lens
    :class:`CalibrationProfile`, whether the correction is active, and when it
    was calibrated."""

    identity_key: str
    friendly: str
    profile: CalibrationProfile
    enabled: bool
    calibrated_at: str


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class LensCalibrationStore:
    """Read/write store over ``<user config>/pybehaviorlab/camera_lens.json``.

    Keyed by the stable USB identity ``unique_id`` (see ``usb_identity.py``),
    exactly like the capability cache, calibrate a lens once and every project
    on this machine reuses the correction. Mutations persist immediately and
    atomically (temp file + ``os.replace``); a corrupt file is treated as empty
    rather than taking the app down. Construct with an explicit ``path`` in
    tests.
    """

    def __init__(self, path: Optional[os.PathLike] = None):
        self._path = Path(path) if path else (_store_dir() / STORE_FILENAME)
        self._cameras: Dict[str, dict] = {}
        self._load()

    # ── persistence ─────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            cams = raw.get("cameras", {})
            if isinstance(cams, dict):
                self._cameras = {str(k): dict(v) for k, v in cams.items()
                                 if isinstance(v, dict)}
        except Exception as e:
            logger.warning("lens store unreadable (%s); starting empty: %s",
                           self._path, e)
            self._cameras = {}

    def _save(self) -> None:
        data = {"version": STORE_VERSION, "cameras": self._cameras}
        path = os.fspath(self._path)
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ── reads ───────────────────────────────────────────────────────────

    def get(self, identity_key: str) -> Optional[CalibrationEntry]:
        """Return the stored calibration, or ``None`` if unknown/corrupt."""
        entry = self._cameras.get(identity_key)
        if not entry:
            return None
        try:
            profile = CalibrationProfile.from_json(entry["profile"])
        except Exception as e:
            logger.warning("lens entry for %s is unreadable: %s", identity_key, e)
            return None
        return CalibrationEntry(
            identity_key=identity_key,
            friendly=str(entry.get("friendly") or ""),
            profile=profile,
            enabled=bool(entry.get("enabled", True)),
            calibrated_at=str(entry.get("calibrated_at") or ""),
        )

    def get_enabled_profile(self, identity_key: str) -> Optional[CalibrationProfile]:
        """The profile to apply for this camera, or ``None`` when the camera is
        uncalibrated or its correction is toggled off. The single gate the
        runtime pipeline consults."""
        entry = self.get(identity_key)
        if entry is None or not entry.enabled:
            return None
        return entry.profile

    def keys(self) -> List[str]:
        return list(self._cameras.keys())

    # ── writes ──────────────────────────────────────────────────────────

    def put(self, identity_key: str, profile: CalibrationProfile,
            friendly: str = "", enabled: bool = True,
            calibrated_at: Optional[str] = None) -> Optional[CalibrationEntry]:
        """Store (or overwrite) one camera's calibration and persist."""
        self._cameras[identity_key] = {
            "friendly": friendly,
            "profile": profile.to_json(),
            "enabled": bool(enabled),
            "calibrated_at": calibrated_at or _now_iso(),
        }
        self._save()
        return self.get(identity_key)

    def set_enabled(self, identity_key: str, enabled: bool) -> bool:
        """Toggle the correction on/off without discarding the profile.
        Returns True if the camera existed."""
        entry = self._cameras.get(identity_key)
        if not entry:
            return False
        entry["enabled"] = bool(enabled)
        self._save()
        return True

    def forget(self, identity_key: str) -> bool:
        """Remove a camera's calibration; return True if it existed."""
        if identity_key in self._cameras:
            del self._cameras[identity_key]
            self._save()
            return True
        return False


# ── runtime cache (bridge to the frame pipeline) ────────────────────────

class LensCorrectionCache:
    """Per-camera cache of :class:`Undistorter` objects (or ``None``).

    The pipeline knows a camera only by ``camera_id`` and hands it raw frames.
    This turns that into "give me the undistorter for this camera, or ``None``"
    with all the expensive work done lazily and once: resolve ``camera_id`` to
    a stable USB identity, look it up in the machine store, build a fixed-point
    :class:`Undistorter` for the incoming frame size. The result is cached per
    ``camera_id``, so steady-state cost is a dict lookup and the remap is the
    only per-frame work, and only for calibrated cameras.

    Dependencies are injected so this is unit-testable without a camera.
    """

    def __init__(self,
                 store: Optional[Any] = None,
                 resolve: Optional[Callable[[Any], str]] = None,
                 alpha: float = 0.0):
        self._store = store          # LensCalibrationStore; built lazily if None
        self._store_injected = store is not None  # tests pass a fixed store
        self._resolve = resolve      # camera_id -> identity key; lazy default
        self._alpha = float(alpha)
        # camera_id -> Undistorter | None (None = resolved, no correction)
        self._by_cam: Dict[Any, Any] = {}
        self._key_by_cam: Dict[Any, str] = {}
        self._enabled = True

    # ── configuration ───────────────────────────────────────────────────

    def set_enabled(self, enabled: bool) -> None:
        """Global on/off. When off, :meth:`undistorter_for` always returns
        ``None`` (no per-frame work at all)."""
        self._enabled = bool(enabled)

    def invalidate(self, camera_id: Any = None) -> None:
        """Drop cached undistorters so the next frame re-resolves from the
        store. Call after the wizard saves a new profile or a toggle changes.
        ``None`` clears everything (and reloads the store from disk, so a
        just-saved calibration is picked up without a restart)."""
        if camera_id is None:
            self._by_cam.clear()
            self._key_by_cam.clear()
            if not self._store_injected:
                self._store = None  # force a fresh read of the JSON next time
        else:
            self._by_cam.pop(camera_id, None)
            self._key_by_cam.pop(camera_id, None)

    # ── lazy dependency access ──────────────────────────────────────────

    def _get_store(self) -> LensCalibrationStore:
        if self._store is None:
            self._store = LensCalibrationStore()
        return self._store

    def _identity_key(self, camera_id: Any) -> Optional[str]:
        if self._resolve is not None:
            try:
                return self._resolve(camera_id)
            except Exception as e:
                logger.debug("lens: injected resolve failed for %s: %s",
                             camera_id, e)
                return None
        from source.video.cameras.usb_identity import resolve_identity
        try:
            info = resolve_identity(camera_id)
        except Exception as e:
            logger.debug("lens: identity resolve failed for %s: %s", camera_id, e)
            return None
        # A weak (index-only) identity can't be matched to a stored profile
        # reliably, so it gets no correction, same rule as the capability cache.
        if info.get("weak"):
            return None
        return info.get("unique_id")

    # ── the hot query ───────────────────────────────────────────────────

    def undistorter_for(self, camera_id: Any,
                        frame_size: Tuple[int, int]) -> Optional[Undistorter]:
        """The undistorter for this camera, or ``None`` if it is uncalibrated,
        its correction is disabled, or the global switch is off.

        ``frame_size`` is ``(width, height)``: used to build the remap maps
        the first time a camera is seen. Cheap on every call after the first.
        """
        if not self._enabled:
            return None
        if camera_id in self._by_cam:
            return self._by_cam[camera_id]

        result = self._resolve_undistorter(camera_id, frame_size)
        self._by_cam[camera_id] = result
        return result

    def _resolve_undistorter(self, camera_id, frame_size) -> Optional[Undistorter]:
        # Open the store FIRST and bail out if there is no calibration at all.
        # This is the common case (nobody has calibrated) and must be a true
        # no-op: resolving a camera's identity can spawn an ``ffmpeg
        # -list_devices`` subprocess (Windows DirectShow), and doing that on the
        # frame path while the camera is being opened/closed contends for the
        # device and can wedge it. Only touch identity when a profile exists.
        try:
            store = self._get_store()
            has_any = bool(store.keys())
        except Exception as e:
            logger.warning("lens: store open failed: %s", e)
            return None
        if not has_any:
            return None

        key = self._identity_key(camera_id)
        if not key:
            return None
        self._key_by_cam[camera_id] = key
        try:
            profile = store.get_enabled_profile(key)
        except Exception as e:
            logger.warning("lens: store lookup failed for %s: %s", key, e)
            return None
        if profile is None:
            return None
        try:
            und = Undistorter(profile, frame_size, alpha=self._alpha)
            logger.info("lens: correction active for camera %s (%s, rms=%.3f)",
                        camera_id, key, profile.rms_error)
            return und
        except Exception as e:
            logger.warning("lens: failed to build undistorter for %s: %s", key, e)
            return None
