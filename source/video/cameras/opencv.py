"""
OpenCV camera backend - wraps cv2.VideoCapture to match GenericCamera interface.

Default backend for USB webcams, with retry logic and DirectShow backend
selection on Windows.

OpenCV cameras have small hardware buffers (typically 1-4 frames), so
get_available_images() does a tight drain loop expecting only a few frames
per call.

``cap.get(CAP_PROP_FPS)`` reflects the request, not what the driver delivers,
so configure() falls back to a timed measurement of delivered frames.
"""

from __future__ import annotations

import time
from source import host_clock
import logging
import os
from contextlib import contextmanager
from typing import Optional

import cv2

from .base import (
    UVC_FPS_DRAIN_S,
    UVC_FPS_MEASURE_S,
    ResolvedCameraSettings,
    GenericCamera,
    measure_delivered_fps,
)

logger = logging.getLogger(__name__)

#: What each capture_format the GUI can offer means as a FOURCC. ``"auto"`` is
#: deliberately absent: it means "leave the driver's negotiated format alone",
#: which is the escape hatch for a camera whose driver refuses a forced format.
#: A name that is in neither is not silently substituted, it is reported. That
#: is what "h264" used to do, quietly opening in uncompressed YUYV.
_FOURCC_FOR_FORMAT = {
    "mjpeg": "MJPG", "mjpg": "MJPG",
    "yuy2": "YUYV", "yuyv": "YUYV", "yuv": "YUYV", "raw": "YUYV",
    "h264": "H264",
}

#: The reverse, for reading back what the driver actually settled on. Only the
#: rig's own names appear here, so a rejection is filed under the same spelling
#: the enumeration and the Format cell use.
_FORMAT_FOR_FOURCC = {"MJPG": "mjpeg", "YUYV": "yuy2", "YUY2": "yuy2",
                      "H264": "h264", "NV12": "nv12", "BGR3": "bgr24"}


#: When set, the only backend anything in this module will use.
#:
#: Capability is not a property of the camera alone; it is a property of the
#: camera THROUGH a backend. The same UVC webcam here measured 10 fps at 720p
#: on DirectShow and 30 on Media Foundation, because only one of them honoured
#: the MJPG request. So a probe that fixes the backend first is not measuring
#: the camera, it is measuring one door into it. Pinning the choice lets the
#: capability probe measure each door in turn.
#:
#: A module-level pin rather than a parameter on the three probe entry points:
#: ``measure_fps_at`` runs through ``begin_capturing``, which walks the order
#: itself, so a parameter would have to be threaded through the open path as
#: well and every future caller would have to remember to pass it.
_forced_backend: Optional[int] = None


@contextmanager
def use_backend(cv_backend: Optional[int]):
    """Pin every open in this module to ``cv_backend`` for the duration.

    Restores the previous pin on exit, including on an exception, a probe
    that raises must not leave the whole application talking to one backend.
    """
    global _forced_backend
    prev = _forced_backend
    _forced_backend = cv_backend
    try:
        yield
    finally:
        _forced_backend = prev


def _opencv_backend_order() -> list[int]:
    """Return cv2.CAP_* backends to try, in priority order.

    Linux: some Jetson kernels expose ``/dev/video0`` as a CSI /
    libcamera-managed node where ``CAP_V4L2`` returns isOpened=False but
    CAP_ANY (GStreamer/FFMPEG) opens it fine; other USB UVC cameras need
    CAP_V4L2 because CAP_ANY's GStreamer auto-pipeline fails on them.
    ``begin_capturing`` walks this list and uses the first that works.
    """
    if _forced_backend is not None:
        return [_forced_backend]
    import sys as _sys
    if os.name == 'nt':
        # DSHOW first. MSMF was tried in front of it and had to be reverted:
        # on the same UVC camera, one run delivered 30 fps and the next took
        # 37 s to reach a first frame and then produced nothing at all
        # (MSMF "can't grab frame", MF_E_HW_MFT_FAILED_START_STREAMING).
        # DSHOW opens in ~0.6 s and delivers every time.
        #
        # Reliable-but-slow beats fast-but-intermittent for a rig that has to
        # start a session unattended. MSMF stays as a fallback for devices
        # DSHOW cannot open at all. Note that DSHOW may negotiate YUY2, whose
        # bandwidth caps the real frame rate far below the requested one,
        # that is what the FOURCC warning after opening is there to surface.
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if _sys.platform == 'darwin':
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    # Linux/Jetson: V4L2 first, it is the clean UVC path and the one that
    # honours FOURCC and frame size the way this module expects.
    #
    # Then GStreamer, NAMED rather than left to CAP_ANY. CAP_ANY does try it,
    # but it also tries FFMPEG first on some builds and reports the same
    # failure for "no GStreamer in this build" as for "this device refused",
    # so an explicit entry is the difference between a diagnosable failure and
    # a silent fallback. On a Jetson the CSI nodes are libcamera or nvarguscam
    # managed and GStreamer is the only path that opens them at all.
    #
    # cv2.CAP_GSTREAMER is a plain constant and exists whether or not the
    # build has GStreamer support, so it is looked up defensively: an OpenCV
    # wheel without it would otherwise raise AttributeError at import time on
    # a machine that never needed it.
    order = [cv2.CAP_V4L2]
    gst = getattr(cv2, "CAP_GSTREAMER", None)
    if gst is not None:
        order.append(gst)
    order.append(cv2.CAP_ANY)
    return order


def _uvc_backends() -> dict:
    """Named OS capture backends worth probing on this platform.

    Only the ones that can differ in what they negotiate. ``CAP_ANY`` is left
    out: it resolves to one of the others, so probing it measures a door
    already measured under its real name and doubles the cost for nothing.
    """
    import sys as _sys
    if os.name == "nt":
        return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    if _sys.platform == "darwin":
        return {"avfoundation": cv2.CAP_AVFOUNDATION}
    # Linux/Jetson: V4L2 is the clean UVC path, but some Jetson kernels expose
    # the camera as a CSI / libcamera-managed node that V4L2 refuses and
    # CAP_ANY (GStreamer) opens. There it is not a duplicate of another door,
    # it is the only one, so it is measured too.
    return {"v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}


def cv_backend_for(name: Optional[str]) -> Optional[int]:
    """The cv2 constant for a stored backend name, or ``None`` for "auto".

    Unknown names resolve to ``None`` rather than raising: a project written
    on Linux names ``v4l2``, and opening it on Windows must fall back to the
    platform order instead of refusing to connect the camera.
    """
    if not name:
        return None
    try:
        return _uvc_backends().get(str(name).strip().lower())
    except Exception:
        return None


def _pick_opencv_backend() -> int:
    """Single-backend convenience for sites that don't iterate.

    Returns the first (preferred) backend from ``_opencv_backend_order()``.
    Sites needing fallback (like begin_capturing) iterate that list directly.
    """
    return _opencv_backend_order()[0]


def _same_format(a: str, b: str) -> bool:
    """Do two FOURCCs name the same pixel format?

    One format can have several codes. OpenCV is set with ``YUYV`` and the
    driver reports ``YUY2``: the same thing, and comparing the raw codes
    reported a mismatch on every single YUY2 open, warning the operator that
    the camera had ignored a request it had in fact honoured. A warning that
    fires when nothing is wrong is worse than none, because it is the one that
    teaches people to ignore the real ones.
    """
    if not a or not b:
        return False
    a, b = str(a).upper(), str(b).upper()
    if a == b:
        return True
    return (_FORMAT_FOR_FOURCC.get(a, a.lower())
            == _FORMAT_FOR_FOURCC.get(b, b.lower()))


def _fourcc_str(cap) -> str:
    """The pixel format a live handle is actually delivering ('MJPG'/'YUYV')."""
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
    except Exception:
        return "?"
    if code <= 0:
        return "?"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"


def _coerce_cam_id(camera_id):
    """Turn the identity the app carries into the address OpenCV opens.

    What is stored and passed around is a camera's IDENTITY (``fp3557d2de``);
    what ``cv2.VideoCapture`` takes is an index, and that index changes with
    whatever else is plugged in. So the identity is resolved against the
    cameras currently connected, on every open, rather than a stale index
    being remembered. Numeric ids (older project files, a hand-typed index)
    are used as-is, and a device path, ``/dev/video0``, passes through
    untouched.
    """
    if isinstance(camera_id, int):
        return camera_id
    s = str(camera_id)
    if s.isdigit():
        return int(s)
    if os.sep in s or s.startswith("/dev/"):
        return camera_id                      # a real device path
    try:
        from source.video.cameras.identity import index_for_id, live_cameras
        # allow_scan: this is the open path. An unresolved identity here is a
        # failed connect, which is worse than the wait.
        idx = index_for_id(s, allow_scan=True)
        if idx is None:
            # Say WHICH cameras are here. "Could not open camera fp755b78f6"
            # after twelve retries describes the symptom; the cause is that
            # this camera is unplugged, or another process (a second copy of
            # the app) is holding it.
            here = [e["id"] for e in live_cameras(cached_only=True)]
            logger.error(
                "camera %s is not connected. Cameras found now: %s. Either it "
                "is unplugged, or another program (a second copy of this app?) "
                "has it open.", s, here or "none")
    except Exception as e:
        logger.debug("camera id %s could not be resolved: %s", s, e)
        idx = None
    return camera_id if idx is None else idx


class OpenCVCamera(GenericCamera):
    """OpenCV VideoCapture wrapper implementing GenericCamera interface.

    Args:
        camera_id: Integer camera index (0, 1, 2, ...) or device path string.
        width: Requested capture width (must be honoured by the camera).
        height: Requested capture height (must be honoured by the camera).
        fps: Requested capture FPS.

    width/height are required (no default fallback) so a driver that quietly
    delivers a smaller frame can't misplace ROIs. Use
    ``probe_supported_resolutions()`` to discover valid modes.
    """

    # Largest-first ladder to probe what a camera actually supports. Many
    # UVC webcams accept the request but deliver a different size, so the
    # probe also grabs a frame and checks ``frame.shape``.
    RESOLUTION_LADDER: tuple[tuple[int, int], ...] = (
        (3840, 2160),
        (2560, 1440),
        (1920, 1200),
        (1920, 1080),
        (1600, 1200),
        (1280, 1024),
        (1280, 960),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
        (320, 240),
    )

    def __init__(self, camera_id, width: int, height: int, fps: float = 20,
                 capture_format: Optional[str] = None,
                 cv_backend: Optional[int] = None):
        super().__init__()
        if not (isinstance(width, int) and isinstance(height, int)
                and width > 0 and height > 0):
            raise ValueError(
                f"OpenCVCamera requires positive integer width/height, "
                f"got {width!r}x{height!r}"
            )
        # Resolved once here, so the capture loop below opens an address and
        # every later log line still reports the identity it was asked for.
        self._camera_id = _coerce_cam_id(camera_id)
        #: The IDENTITY the app carries (``cam3cbb52d3``), before it was
        #: resolved to the index ``cv2.VideoCapture`` opens. Anything that
        #: keys per-machine storage must use this: the index changes with
        #: whatever else is plugged in, which is the whole reason
        #: ``_coerce_cam_id`` resolves it fresh on every open.
        self._camera_identity = camera_id
        self._requested_width = int(width)
        self._requested_height = int(height)
        self._width = int(width)
        self._height = int(height)
        self._cap: cv2.VideoCapture | None = None
        self._capturing = False
        self._pending_fps: float = float(fps)  # applied on begin_capturing()
        #: The rate begin_capturing last put on the device. A live change is
        #: compared with this, not with the request, to tell a NEW rate from
        #: the same one re-applied. See _needs_reopen_for_rate.
        self._rate_on_device: Optional[float] = None
        # UVC capture pixel format: "mjpeg" (compressed, keeps full FPS over
        # USB, the default) or "yuv"/"raw" (uncompressed, may cap FPS at
        # higher resolutions). Set on begin_capturing() and read back to
        # verify the driver honoured it.
        # The OS capture backend to open through, chosen by the
        # capability probe because it decides whether MJPG is honoured
        # and therefore what rate this camera can reach. None = walk the
        # platform's preference order as before.
        self._cv_backend = cv_backend
        self._capture_format = (str(capture_format).lower()
                                if capture_format else "mjpeg")
        # Populated by begin_capturing() when the driver refuses the requested
        # format (e.g. stays on uncompressed YUY2, which throttles USB FPS).
        self.format_warning: Optional[str] = None
        #: Set when the backend refused a one-frame capture queue, so frames
        #: arrive later than they were captured. See _seal_buffersize.
        self.buffer_warning: Optional[str] = None
        #: Said once, not once per rate change.
        self._warned_unknown_format = False
        #: What the device was seen delivering after it opened, and the
        #: complaint if that is not what was asked for. A UVC driver accepts
        #: a rate it cannot hold and reports it back unchanged, so the request
        #: is not evidence of anything.
        self.delivered_fps: Optional[float] = None
        self.fps_warning: Optional[str] = None
        #: The requested rate the measurement above was taken at. A rate
        #: change makes it stale, and a stale delivered rate is worse than
        #: none: the recorder would stamp the file with a number measured
        #: for a mode the camera is no longer running.
        self._delivered_for_fps: Optional[float] = None

        # Build unique_id
        self.serial_number = str(camera_id)
        self.device_model = "USB Camera"
        self.unique_id = f"{camera_id}-opencv"

    # --- Resolution probe (used by GUI to populate the dropdown) -------------

    @classmethod
    def native_mode(cls, camera_id) -> Optional[tuple[int, int]]:
        """Open ``camera_id`` briefly and return the ``(w, h)`` it delivers.

        Used as a degrade fallback when the resolution ladder finds nothing
        but the camera is still reachable. Returns ``None`` on any failure.
        """
        cam_id_typed = _coerce_cam_id(camera_id)
        cap = None
        try:
            cap = cv2.VideoCapture(cam_id_typed, _pick_opencv_backend())
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            fh, fw = frame.shape[:2]
            return (int(fw), int(fh))
        except Exception:
            return None
        finally:
            if cap is not None:
                cap.release()

    #: The formats Detect knows how to ask for, in the order they are offered.
    #: The rig's own spellings, the ones the Format cell and the enumeration
    #: both use.
    PROBE_FORMATS = ("mjpeg", "yuy2")

    @classmethod
    def supports_format(cls, camera_id, pixel_format, size=(640, 480)):
        """Can this camera actually deliver ``pixel_format``?

        Returns ``(ok, negotiated)`` where ``negotiated`` is the FOURCC the
        driver settled on. This exists because asking is not the same as
        getting: a UVC driver accepts any FOURCC you set, reports it back
        unchanged, and then streams whatever it pleases. Every camera on this
        rig was running YUY2 while the application believed it had MJPEG, and
        nothing anywhere checked. Detect calls this FIRST so the operator is
        told, rather than being handed rates measured in a format they did not
        choose.
        """
        want = _FOURCC_FOR_FORMAT.get(str(pixel_format or "").lower())
        if not want:
            return (False, "")
        cap = None
        try:
            cap = cv2.VideoCapture(_coerce_cam_id(camera_id),
                                   _pick_opencv_backend())
            if cap is None or not cap.isOpened():
                return (False, "")
            # Size BEFORE format; see ``_apply_fourcc``.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*want))
            for _ in range(5):
                cap.read()
            got = _fourcc_str(cap)
            # Compare the rig's NAME, not the raw code. YUYV and YUY2 are two
            # spellings of one format: OpenCV is set with "YUYV" and the
            # driver reports "YUY2", so a raw comparison calls a camera that
            # did exactly what was asked unsupported.
            want_name = _FORMAT_FOR_FOURCC.get(want, str(pixel_format).lower())
            got_name = _FORMAT_FOR_FOURCC.get(got, got.lower())
            return (got_name == want_name, got)
        except Exception as e:
            logger.warning("supports_format(%s, %s) failed: %s",
                           camera_id, pixel_format, e)
            return (False, "")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    @classmethod
    def probe_supported_resolutions(
        cls,
        camera_id,
        candidates: Optional[list[tuple[int, int]]] = None,
        report=None,
        pixel_format: str = "mjpeg",
    ) -> tuple[list[tuple[int, int]], Optional[tuple[int, int]]]:
        """Probe ``camera_id`` for resolutions it actually delivers.

        Opens the camera, sets FOURCC=MJPG, then for each ``(w, h)`` in
        ``candidates`` (default: ``RESOLUTION_LADDER``, largest-first):
          1. ``set(WIDTH/HEIGHT)`` and check ``get()`` returns the same values
          2. grab a frame and confirm ``frame.shape`` matches.
        A mode only counts as supported if both checks pass, some drivers
        report the requested size from ``get()`` even when they actually
        deliver a different one.

        Returns ``(confirmed, max_mode)`` where ``confirmed`` is the list of
        accepted ``(w, h)`` pairs in ladder order (largest first) and
        ``max_mode`` is the first entry, or ``None`` when nothing worked.
        """
        if candidates is None:
            candidates = list(cls.RESOLUTION_LADDER)

        cam_id_typed = _coerce_cam_id(camera_id)

        backend = _pick_opencv_backend()
        # The format is the operator's choice and it decides which sizes exist
        # at all, so it is probed FOR rather than assumed: this used to force
        # MJPG regardless of what the Format cell said.
        _want = _FOURCC_FOR_FORMAT.get(str(pixel_format or "").lower(), "MJPG")
        confirmed: list[tuple[int, int]] = []
        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = cv2.VideoCapture(cam_id_typed, backend)
            if not cap.isOpened():
                # Name the backend and the address that failed. "could not
                # open camera" discarded exactly the fact that mattered when
                # Media Foundation reported zero modes on a rig where
                # DirectShow had just measured three: with only that sentence
                # in the log there is no way to tell a backend that cannot
                # serve this camera from one that was handed a busy device,
                # and the two lead to opposite fixes.
                _bname = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF",
                          cv2.CAP_V4L2: "V4L2", cv2.CAP_ANY: "ANY",
                          cv2.CAP_AVFOUNDATION: "AVFOUNDATION"}.get(
                              backend, str(backend))
                logger.warning(
                    f"probe_supported_resolutions({camera_id}): could not open "
                    f"camera via {_bname} at index {cam_id_typed!r} "
                    f"(VideoCapture returned a closed handle; the device is "
                    f"busy, or this backend cannot serve it)"
                )
                return ([], None)
            for w, h in candidates:
                # Size first, then the format, and BOTH inside the loop: a
                # size change renegotiates the format, so a FOURCC set once
                # before the loop is undone by the first resize. See
                # ``_apply_fourcc``.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*_want))
                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if got_w != w or got_h != h:
                    continue
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                fh, fw = frame.shape[:2]
                if fw == w and fh == h:
                    confirmed.append((w, h))
                    if callable(report):
                        try:
                            report("resolution_found", w=w, h=h)
                        except Exception:
                            pass
        finally:
            if cap is not None:
                cap.release()

        max_mode = confirmed[0] if confirmed else None
        logger.info(
            f"probe_supported_resolutions({camera_id}): {len(confirmed)} mode(s); "
            f"max={max_mode}"
        )
        return (confirmed, max_mode)

    @classmethod
    def measure_fps_at(
        cls,
        camera_id,
        width: int,
        height: int,
        target_fps: float = 30.0,
        drain_s: float = UVC_FPS_DRAIN_S,
        measure_s: float = UVC_FPS_MEASURE_S,
        report=None,
        pixel_format: str = "mjpeg",
    ) -> float:
        """Open ``camera_id`` at ``(width, height)``, measure realistic FPS.

        Uses ``begin_capturing`` + ``get_available_images`` (the same path
        the runtime camera thread uses), so the returned rate matches what
        the pipeline will actually receive.  Returns ``0.0`` on failure.

        ``drain_s`` lets the driver settle / FIFO empty; ``measure_s`` is
        the count window.  Defaults sized for a snappy GUI probe (~5 s
        per mode). Camera is closed before returning.
        """
        cam = None
        try:
            _fmt = str(pixel_format or "mjpeg").lower()
            _want_name = _FORMAT_FOR_FOURCC.get(
                _FOURCC_FOR_FORMAT.get(_fmt, ""), _fmt)
            cam = cls(camera_id, int(width), int(height), float(target_fps),
                      capture_format=_fmt)
            cam.begin_capturing()
            fourcc = cam._read_fourcc_str()
            if _FORMAT_FOR_FOURCC.get(fourcc, fourcc.lower()) != _want_name:
                # The driver renegotiated. One clean reopen usually lands on
                # the requested format; measuring the first attempt is how the
                # same mode came out 18 fps once and 8 fps the next time.
                logger.info("measure_fps_at(%s, %dx%d): asked %s, got %s, "
                            "reopening once", camera_id, width, height,
                            _fmt, fourcc)
                cam.stop_capturing()
                cam = cls(camera_id, int(width), int(height), float(target_fps),
                          capture_format=_fmt)
                cam.begin_capturing()
                fourcc = cam._read_fourcc_str()
            if _FORMAT_FOR_FOURCC.get(fourcc, fourcc.lower()) != _want_name:
                # Say so rather than returning a rate measured in a format
                # nobody chose, which is what made every stored number wrong.
                logger.warning(
                    "measure_fps_at(%s, %dx%d): asked for %s but the camera "
                    "delivered %s; the measurement is NOT for the requested "
                    "format.", camera_id, width, height, _fmt, fourcc)
                return 0.0

            def _count():
                batch = cam.get_available_images()
                return len(batch["images"]) if batch and batch.get("images") else 0

            measured = measure_delivered_fps(
                cam.get_available_images, _count,
                drain_s=drain_s, measure_s=measure_s)
            if measured <= 0.0:
                return 0.0
            if callable(report):
                try:
                    report("fps_measured", w=int(width), h=int(height),
                           fps=measured, fourcc=fourcc)
                except Exception:
                    pass
            return measured
        except Exception as e:
            logger.warning(
                f"OpenCVCamera.measure_fps_at({camera_id}, {width}x{height}): {e}"
            )
            return 0.0
        finally:
            if cam is not None:
                try:
                    cam.stop_capturing()
                except Exception:
                    pass

    @classmethod
    def measure_fps_for_modes(
        cls,
        camera_id,
        modes,
        target_fps: float = 30.0,
        drain_s: float = 0.5,
        measure_s: float = 1.2,
        report=None,
        pixel_format: str = "mjpeg",
    ) -> list[tuple[int, int, float]]:
        """Measure realistic FPS at every ``(w, h)`` in ``modes`` on ONE open
        handle, instead of re-opening the camera per mode.

        The per-mode re-open + 2 s drain + 3 s measure in ``measure_fps_at``
        dominated calibration time (~5 s × modes, so ~40 s for a webcam). Here
        the camera is opened once; each mode just re-sets W/H, drains briefly
        for the driver to renegotiate, then counts frames. Emits the same
        ``measuring`` / ``mode_done`` progress events per mode so the caller's
        live log still updates.

        Returns ``[(w, h, fps), ...]`` (fps 0.0 for a mode that yielded no
        frames). Returns ``[]`` if the camera could not be opened, so the
        caller can fall back to the per-mode path.
        """
        mode_list = [(int(w), int(h)) for (w, h) in modes]
        if not mode_list:
            return []
        cam_id_typed = _coerce_cam_id(camera_id)

        backend = _pick_opencv_backend()
        out: list[tuple[int, int, float]] = []
        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = cv2.VideoCapture(cam_id_typed, backend)
            if not cap.isOpened():
                return []
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            try:
                cap.set(cv2.CAP_PROP_FPS, float(target_fps))
            except Exception:
                pass
            total = len(mode_list)
            for i, (w, h) in enumerate(mode_list, start=1):
                if callable(report):
                    try:
                        report("measuring", w=w, h=h, index=i, total=total)
                    except Exception:
                        pass
                # FOURCC is re-applied for EVERY mode, before the size: a UVC
                # driver renegotiates the format when the resolution changes
                # and commonly lands back on uncompressed YUY2, whose USB
                # bandwidth ceiling then halves the rate. Setting MJPG once
                # before the loop measured the first mode compressed and the
                # rest raw, which is why one mode read 18 fps in one run and
                # 8 fps in the next.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*_FOURCC_FOR_FORMAT.get(
                            str(pixel_format or "").lower(), "MJPG")))
                # Let the driver renegotiate the new mode / FIFO settle.
                t_end = time.monotonic() + float(drain_s)
                while time.monotonic() < t_end:
                    cap.read()
                fourcc = _fourcc_str(cap)
                # Verify the switch actually took. Many UVC/DSHOW drivers
                # ignore a mid-stream resolution change on a live handle and
                # keep delivering the previous size, measuring then would
                # report the WRONG mode's FPS and silently cap capture. If any
                # mode can't be confirmed on this shared handle, bail so the
                # caller falls back to the reliable per-mode re-open path.
                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if got_w != w or got_h != h:
                    logger.info(
                        "measure_fps_for_modes(%s): handle stuck at %dx%d for "
                        "requested %dx%d, falling back to per-mode probe.",
                        camera_id, got_w, got_h, w, h)
                    return []
                n = 0
                t0 = time.monotonic()
                while time.monotonic() - t0 < float(measure_s):
                    ret, frame = cap.read()
                    # Confirm the delivered frame is the requested size too.
                    if ret and frame is not None:
                        fh, fw = frame.shape[:2]
                        if fw != w or fh != h:
                            logger.info(
                                "measure_fps_for_modes(%s): delivered %dx%d "
                                "for requested %dx%d, per-mode fallback.",
                                camera_id, fw, fh, w, h)
                            return []
                        n += 1
                elapsed = time.monotonic() - t0
                fps = (n / elapsed) if (elapsed > 0 and n >= 2) else 0.0
                out.append((w, h, float(fps)))
                if callable(report):
                    try:
                        # The format is part of the measurement, not a detail:
                        # the same mode delivers ~2x the rate compressed, so a
                        # bare number is not comparable between modes.
                        report("mode_done", w=w, h=h, fps=fps, fourcc=fourcc)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(
                f"OpenCVCamera.measure_fps_for_modes({camera_id}): {e}")
            # Return [] (not the partial ``out``) so probe.py falls back to
            # the reliable per-mode path instead of dropping unmeasured modes.
            return []
        finally:
            if cap is not None:
                cap.release()
        return out

    # --- Configuration --------------------------------------------------------

    def configure(
        self,
        fps: Optional[float] = None,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> ResolvedCameraSettings:
        """Apply settings; store target_fps verbatim.

        OpenCV's ``cap.get(CAP_PROP_FPS)`` is unreliable across drivers, so we
        push the requested fps and trust the user's choice. Background
        calibration in ``CameraThread`` populates ``self.measured_fps`` for
        GUI display, but never modifies ``self.config.target_fps``.

        ``target_fps`` is therefore still the request, verbatim. What the
        device is OBSERVED delivering travels separately as
        ``delivered_fps``, and only when it was measured at THIS rate: a
        measurement from a previous rate is dropped rather than carried,
        because the recorder stamps the file with it.
        """
        prev = self.config
        target_fps = fps if fps is not None else (prev.target_fps if prev else self._pending_fps)
        self._pending_fps = float(target_fps)

        if self._cap and self._cap.isOpened():
            accepted = self._cap.set(cv2.CAP_PROP_FPS, float(target_fps))
            if self._needs_reopen_for_rate(accepted, target_fps):
                self._reopen_at_rate(float(target_fps))
                # What the camera now runs at: the new rate, or the previous
                # one when the new rate would not open.
                target_fps = self._pending_fps
            else:
                # Setting the rate renegotiates the pixel format, and this
                # method runs at the end of begin_capturing and again whenever
                # the rate changes. Without the re-seal, begin_capturing
                # established MJPG and then this line put the camera back to
                # YUY2 four lines later, which is why the format warning never
                # fired: by the time anyone looked, the format had been
                # correct and was not any more.
                # Only a format we actually asked for is re-sealed. "auto" was
                # never sealed in the first place, so re-sealing it would
                # impose the very choice it exists to avoid.
                resealed = self._wanted_fourcc()
                if resealed:
                    self._seal_fourcc(resealed)

        cfg = ResolvedCameraSettings(
            target_fps=float(target_fps),
            delivered_fps=self._delivered_at(target_fps),
            # OpenCV exposes no per-frame hardware timestamp on any backend,
            # so every frame time from this camera is taken at handover.
            timestamp_source="host",
            exposure_us=exposure_us,
            gain_db=gain_db,
            width=self.get_width(),
            height=self.get_height(),
            roi=roi if roi is not None else (prev.roi if prev else None),
        )
        self.config = cfg
        return cfg

    def _needs_reopen_for_rate(self, accepted, target_fps) -> bool:
        """True when a streaming V4L2 camera refused a NEW rate.

        uvcvideo does not change the frame interval of a running stream:
        ``cap.set(CAP_PROP_FPS)`` returns False and the camera keeps its old
        rate. Measured on the Jetson rig: streaming at 30, set 15 -> refused,
        still delivering 30; reopened at 15 -> 15. A rate picked for a camera
        that was already live therefore never took effect. Other backends keep
        their behaviour, and the same rate re-applied at the end of
        begin_capturing is not a change.
        """
        if accepted or getattr(self, "_open_backend", None) != "V4L2":
            return False
        on_device = self._rate_on_device
        return (on_device is not None
                and abs(float(target_fps) - on_device) > 1e-6)

    def _reopen_at_rate(self, fps: float) -> None:
        """Close and reopen the camera so ``fps`` reaches the driver.

        Runs on the capture thread, which applies queued rate changes in its
        own loop, so nothing is reading from the handle meanwhile. When the
        camera will not reopen at the new rate it is reopened at the rate it
        had, so a refused change never leaves the box without a camera.
        """
        previous = self._rate_on_device
        logger.info(
            "Camera %s: V4L2 does not change the rate of a running stream; "
            "reopening at %.0f fps", self._camera_id, fps)
        self.stop_capturing()
        self._pending_fps = float(fps)
        try:
            self.begin_capturing()
        except Exception as e:
            if previous is None:
                raise
            logger.error(
                "Camera %s: reopening at %.0f fps failed (%s); reopening at "
                "the previous %.0f fps", self._camera_id, fps, e, previous)
            self._pending_fps = float(previous)
            self.begin_capturing()

    # --- Frame dimensions ----------------------------------------------------

    def get_width(self) -> int:
        if self._cap and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    def get_height(self) -> int:
        if self._cap and self._cap.isOpened():
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    # --- Acquisition ---------------------------------------------------------

    def _apply_fourcc(self, fmt: str) -> None:
        """Set width/height FIRST, then the pixel format.

        The order is the whole of it, and it used to be the other way round on
        the belief that "the resolution must follow the FOURCC or the driver
        clamps it". Measured on this rig, DirectShow does the opposite: a
        resolution set AFTER the FOURCC renegotiates the format and lands back
        on uncompressed.

            fourcc, then size   ->  YUY2  10.7 fps at 1920x1080
            size, then fourcc   ->  MJPG  56.7 fps at 1920x1080

        Every camera on this rig was therefore streaming YUY2 while the
        application believed it had asked for MJPEG, and every rate it
        measured was an uncompressed rate: that is where "10 fps at 1080p"
        came from on a camera that does nearly 60.

        Still only half of it. Every property set afterwards renegotiates the
        format again, so ``_seal_fourcc`` has to run LAST. See its docstring.
        """
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))

    #: Settle then count. Short, because this runs on every camera open and
    #: the rig already waits out a camera-settle period before recording.
    _FPS_DRAIN_S = 0.4
    _FPS_MEASURE_S = 1.2
    #: How far a delivered rate may sit from the request before it is a
    #: complaint rather than jitter. Whichever is larger.
    _FPS_TOLERANCE_FRAC = 0.10
    _FPS_TOLERANCE_MIN = 1.0

    def _delivered_at(self, target_fps) -> Optional[float]:
        """The delivered-rate measurement for THIS rate, or ``None``.

        Only a measurement taken at the rate being configured travels with it.
        ``configure`` runs again on every rate change, and carrying the
        previous rate's number would hand the recorder a measurement for a
        mode the camera is no longer running: the same class of error the
        field exists to remove. Stale is dropped rather than kept, because
        no measurement makes the recorder fall back to the request, while a
        wrong one makes it stamp the file confidently and incorrectly.
        """
        if (self._delivered_for_fps is None
                or abs(float(target_fps) - self._delivered_for_fps) > 1e-6):
            self.delivered_fps = None
            self.fps_warning = None
        return self.delivered_fps

    def _measure_delivered_rate(self) -> None:
        """Observe the real delivery rate and complain if it is not the ask."""
        want = float(self._pending_fps or 0.0)

        def _count():
            batch = self.get_available_images()
            return len(batch["images"]) if batch and batch.get("images") else 0

        try:
            got = measure_delivered_fps(
                self.get_available_images, _count,
                drain_s=self._FPS_DRAIN_S, measure_s=self._FPS_MEASURE_S)
        except Exception as e:
            logger.debug("camera %s: could not measure the delivered rate: %s",
                         self._camera_id, e)
            return
        if got <= 0.0:
            return
        self.delivered_fps = got
        self._delivered_for_fps = want
        if want <= 0:
            return
        slack = max(self._FPS_TOLERANCE_MIN, want * self._FPS_TOLERANCE_FRAC)
        if abs(got - want) <= slack:
            self.fps_warning = None
            return
        # What the driver says it is running at, not the request echoed back:
        # a UVC mode with one interval runs at that interval whatever was asked
        # (640x480 MJPEG on this rig is 120 fps only).
        try:
            reported = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            reported = 0.0
        said = (f"the driver reports {reported:.0f}" if reported > 0
                else "the driver reports no rate")
        self.fps_warning = (
            f"Camera {self._camera_id}: asked for {want:.0f} fps and "
            f"{said}, but it is delivering "
            f"{got:.1f} fps. The recording is written at the delivered rate, "
            f"so it plays back correctly; the rate you picked is not one this "
            f"camera holds at {self._requested_width}x{self._requested_height}."
        )
        logger.warning(self.fps_warning)
        # ONLY when the camera delivered MORE than it was asked for.
        #
        # A UVC camera exposes discrete rates and rounds UP to the nearest one
        # it has, so delivering more than asked is proof the asked-for rate
        # does not exist: asked 25 -> 30.03, asked 10 -> 14.98.
        #
        # Delivering LESS is a condition, not a capability. Measured on this
        # rig at 1920x1080, minutes apart: 30.13 fps and then 24.88 fps, the
        # same camera and the same mode, because auto-exposure lengthens
        # integration in dimmer light. Banning 30 fps because the room was
        # dark once would strip a rate the camera really has, permanently,
        # and the operator would have no way to get it back.
        if got > want:
            self._remember_rate_rejected(want, got)
        else:
            logger.info(
                "camera %s: %.0f fps was not reached (%.1f delivered), but "
                "the camera was not asked for a rate it lacks; a shortfall is "
                "a condition (light, bandwidth, load), so the rate stays on "
                "offer.", self._camera_id, want, got)

    def _remember_rate_rejected(self, asked, got) -> None:
        """Record a rate this camera does not offer, against door and size."""
        try:
            from source.video.cameras import calibration_store as _store
            from source.video.cameras.usb_identity import resolve_identity
            ident = resolve_identity(
                getattr(self, "_camera_identity", self._camera_id), "opencv")
            door = {cv2.CAP_MSMF: "msmf", cv2.CAP_DSHOW: "dshow",
                    cv2.CAP_V4L2: "v4l2", cv2.CAP_ANY: "any",
                    cv2.CAP_AVFOUNDATION: "avfoundation"}.get(
                        getattr(self, "_cv_backend", None), "")
            if not door:
                door = str(getattr(self, "_open_backend", "") or "").lower()
            # The FORMAT the camera is actually in, not the one that was
            # asked for: a UVC driver renegotiates the pixel format whenever
            # size or rate changes, and the rate a size can reach depends on
            # which format it landed in. Filed under the wrong one, this
            # rejection deletes a rate from a format that really has it.
            fmt = None
            try:
                fourcc = self._read_fourcc_str()
                fmt = _FORMAT_FOR_FOURCC.get(str(fourcc or "").upper())
            except Exception:
                fmt = None
            if not fmt:
                cfgd = str(getattr(self, "_capture_format", "") or "").lower()
                fmt = cfgd if cfgd and cfgd != "auto" else None
            _store.note_rate_rejected(
                ident.get("unique_id"), door or "opencv",
                (self._requested_width, self._requested_height), asked, got,
                pixel_format=fmt)
        except Exception as e:
            logger.debug("camera %s: could not record the rejected rate: %s",
                         self._camera_id, e)

    def _wanted_fourcc(self) -> Optional[str]:
        """The FOURCC this camera was asked to capture in, or ``None``.

        ``None`` means the driver's own choice stands: either "auto" was
        chosen deliberately, or the configured name is one this backend has
        no rule for, which is said out loud rather than turned into YUYV.
        """
        if self._capture_format == "auto":
            return None
        fourcc = _FOURCC_FOR_FORMAT.get(self._capture_format)
        if fourcc is None and not self._warned_unknown_format:
            self._warned_unknown_format = True
            logger.warning(
                "camera %s: capture format %r is not one this backend knows "
                "(%s); leaving the driver's own choice rather than silently "
                "substituting one.", self.serial_number, self._capture_format,
                ", ".join(sorted(set(_FOURCC_FOR_FORMAT))))
        return fourcc

    def _seal_fourcc(self, fmt: str) -> None:
        """Assert the pixel format as the LAST property set on the capture.

        Measured on a UVC camera through DirectShow, 800x600, 2026-09-10:

            FOURCC, W, H                       -> YUY2 at 30
            FOURCC, W, H, FPS                  -> YUY2 at 30
            FOURCC, W, H, FPS, FOURCC          -> MJPG at 30
            FOURCC, W, H, FOURCC               -> MJPG at 60

        Setting the size or the rate makes the driver renegotiate, and it
        answers from its uncompressed list, so whichever of them runs last
        decides the format. Sealing the format afterwards costs one property
        set and is the difference between 30 fps uncompressed and 60 fps MJPG
        on the same camera.

        The camera advertised MJPG at 800x600 the whole time. The request was
        being made and then undone by the next line.
        """
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))

    def _read_fourcc_str(self) -> str:
        """Read CAP_PROP_FOURCC back as a 4-char code ('MJPG', 'YUYV', …),
        or '' if the driver reports nothing usable."""
        try:
            code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        except (TypeError, ValueError):
            return ""
        if code <= 0:
            return ""
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip()

    def begin_capturing(self) -> None:
        """Open the camera and start capturing."""
        if self._cap and self._cap.isOpened():
            return  # Already open

        # Try backends in priority order (see _opencv_backend_order).
        # Each backend gets 4 retries × 0.5 s.
        #
        # DO NOT call cap.read() inside this loop before the FOURCC /
        # width / height settings below, a read in default pixel format
        # locks some UVC drivers into YUY2 and caps the camera at 640×480
        # even when MJPG would unlock 1080p.
        # A pinned backend is tried FIRST, not exclusively. The pin comes
        # from a capability measurement, and a backend that measured
        # fastest can still fail to open later, a driver update, a
        # different port, another process holding the device. Falling
        # back to the platform order costs a few seconds; refusing to
        # open costs the session.
        backends = _opencv_backend_order()
        if self._cv_backend is not None:
            backends = ([self._cv_backend]
                        + [b for b in backends if b != self._cv_backend])
        max_retries_per_backend = 4
        retry_delay = 0.5
        last_error = None

        for backend in backends:
            for attempt in range(max_retries_per_backend):
                cap = None
                try:
                    cap = cv2.VideoCapture(self._camera_id, backend)
                    if cap.isOpened():
                        self._cap = cap
                        self._open_backend = {
                            cv2.CAP_MSMF: "MSMF", cv2.CAP_DSHOW: "DSHOW",
                            cv2.CAP_V4L2: "V4L2", cv2.CAP_ANY: "ANY",
                            cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
                        }.get(backend, str(backend))
                        logger.info(
                            f"OpenCVCamera {self._camera_id} opened with "
                            f"backend={backend} on attempt {attempt + 1}"
                        )
                        break
                    cap.release()
                except Exception as e:
                    last_error = e
                    # Release the half-constructed handle, an unreleased
                    # VideoCapture keeps the OS device claimed, blocking
                    # every retry (and any other process) from opening it.
                    if cap is not None and cap is not self._cap:
                        try:
                            cap.release()
                        except Exception as rel_err:
                            logger.debug(
                                f"OpenCVCamera {self._camera_id} release "
                                f"after failed open: {rel_err}")
                    logger.warning(
                        f"OpenCVCamera {self._camera_id} exception on open "
                        f"(backend={backend}, attempt {attempt + 1}/"
                        f"{max_retries_per_backend}): {e}"
                    )
                if self._cap is None and attempt < max_retries_per_backend - 1:
                    logger.warning(
                        f"OpenCVCamera {self._camera_id} open failed "
                        f"(backend={backend}, attempt {attempt + 1}/"
                        f"{max_retries_per_backend}), retrying in "
                        f"{retry_delay}s..."
                    )
                    time.sleep(retry_delay)
            if self._cap is not None:
                break  # Backend succeeded, stop trying others.
            logger.warning(
                f"OpenCVCamera {self._camera_id} backend={backend} "
                f"exhausted {max_retries_per_backend} retries; "
                f"trying next backend (if any)..."
            )
        # For the post-loop error message below.
        max_retries = max_retries_per_backend * len(backends)

        if self._cap is None or not self._cap.isOpened():
            msg = f"Could not open camera {self._camera_id} after {max_retries} attempts"
            if last_error:
                msg += f" (last error: {last_error})"
            raise RuntimeError(msg)

        # Pixel format must be set BEFORE width/height: many UVC webcams expose
        # 1080p / 4K only over MJPG; uncompressed YUY2 is bandwidth-limited to
        # ~640x480 over USB 2.0, so without MJPG the camera silently clamps to
        # its YUY2 max and the real FPS collapses. "yuv"/"raw" requests the
        # uncompressed format on purpose (lower latency at low resolution).
        # "auto" means leave the driver's negotiated format alone.
        wanted = self._wanted_fourcc()
        if wanted:
            self._apply_fourcc(wanted)
        # Buffersize 1 = low-latency live mode: the driver keeps only the
        # freshest frame, so ``read()`` blocks one frame-period and returns
        # one frame per call. The capture loop publishes EVERY frame (display
        # at the full capture rate) and buffers each for recording. A larger
        # buffer lets the driver queue frames, which would batch several per
        # call and drop the display to capture_rate / batch_size.
        #
        # Set here so the format negotiation below sees it, and re-applied
        # and CHECKED after that negotiation by ``_seal_buffersize``: the
        # format seal renegotiates the stream and can hand the property back
        # to the driver's default, and nothing downstream would notice.
        # V4L2 is the exception to "one": see _queue_depth.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._queue_depth())
        # Request the target FPS AFTER the format: an out-of-range value can
        # make some drivers renegotiate back to YUY2, so the FOURCC is verified
        # last (below), after the FPS is applied.
        self._cap.set(cv2.CAP_PROP_FPS, self._pending_fps)

        # Read the FOURCC back, the driver may ignore the request with no
        # error. Retry once, then warn loudly if MJPG was requested but the
        # driver stayed uncompressed (USB bandwidth then caps the real FPS).
        # The format is sealed AFTER the buffer size and the rate, because
        # both renegotiate it. Nothing else may touch the capture below this
        # line before the readback.
        # Nothing to seal, and nothing to check, when the format was never
        # requested: re-sealing under Auto would impose the very choice it
        # exists to avoid.
        if not wanted:
            got = None
        else:
            self._seal_fourcc(wanted)
            got = self._read_fourcc_str()
            if not _same_format(got, wanted):
                self._apply_fourcc(wanted)
                self._cap.set(cv2.CAP_PROP_FPS, self._pending_fps)
                self._seal_fourcc(wanted)
                got = self._read_fourcc_str()
        if wanted and got and not _same_format(got, wanted):
            extra = (" USB bandwidth will cap the real FPS. Pick an "
                     "MJPG-capable mode." if wanted == "MJPG" else "")
            self.format_warning = (
                f"Camera {self._camera_id}: requested {wanted} but the driver "
                f"is capturing in {got} at "
                f"{self._requested_width}x{self._requested_height}.{extra}"
            )
            logger.warning(self.format_warning)
        elif wanted and not got:
            # Some backends (Media Foundation) never report FOURCC. Unknown is
            # not the same as wrong, and the measured delivery rate is the
            # check that does not need the driver's cooperation.
            logger.debug(
                "Camera %s: the backend does not report a pixel format; "
                "relying on the measured rate instead.", self._camera_id)

        # The format seal above renegotiates the stream, so the queue depth is
        # re-applied and verified only now, after everything that could undo
        # it. See _seal_buffersize for why this is not a formality.
        self._seal_buffersize()

        # Validate: cap.get() reports what the driver chose, which may differ
        # from the request without raising any error. Warn loudly so the GUI
        # can surface the mismatch instead of acting on stale ROI coordinates.
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._width = actual_w
        self._height = actual_h
        if actual_w != self._requested_width or actual_h != self._requested_height:
            logger.warning(
                f"Camera {self._camera_id}: requested "
                f"{self._requested_width}x{self._requested_height}, "
                f"driver delivered {actual_w}x{actual_h}. "
                f"Use 'Detect Max' to pick a supported mode."
            )

        self._capturing = True
        self.device_model = f"USB Camera {self._camera_id}"

        # WHAT IT ACTUALLY DELIVERS, before anything downstream believes the
        # request. ``cap.set(CAP_PROP_FPS, 10)`` returns success and
        # ``cap.get`` reads back 10 on a camera that then delivers 15, so the
        # request is not evidence of anything. Measured on this rig:
        #
        #     asked 30 -> reported 30.0 -> delivered 30.01   honoured
        #     asked 20 -> reported 20.0 -> delivered 19.92   honoured
        #     asked 10 -> reported 10.0 -> delivered 15.00   NOT honoured
        #
        # The recorder stamps the file with this rather than the request, so a
        # session asked for 10 fps no longer writes 15 fps of frames into a
        # file declared as 10, which played back 1.5x slow with every derived
        # timestamp wrong.
        self._rate_on_device = float(self._pending_fps)
        self._measure_delivered_rate()

        # Initial ResolvedCameraSettings carries both the request and the
        # measurement; CameraThread later writes self.measured_fps for GUI
        # display only.
        self.configure(fps=self._pending_fps)

        # Name the backend and the pixel format, not just the size. A camera
        # that negotiates an uncompressed format cannot carry its requested
        # rate over USB and starves, silently, because every property still
        # reads back as asked. Logging what was actually agreed is what turns
        # "the display is slow" into "the camera is delivering YUY2".
        fourcc = self._read_fourcc_str()
        logger.info(
            f"OpenCVCamera {self._camera_id} opened: "
            f"{self._width}x{self._height} target_fps={self._pending_fps} "
            f"backend={getattr(self, '_open_backend', '?')} fourcc={fourcc}"
        )
        # No warning on the format alone. An uncompressed mode was measured
        # sustaining 29 fps at 640x480 here, so predicting starvation from
        # YUY2 would cry wolf on a perfectly healthy rig, and a warning that
        # is usually wrong is worse than none. What the format is good for is
        # EXPLAINING a low rate once one is observed, which is why it is in
        # the line above; the FPS readout and tools/diagnose_fps.py are what
        # decide whether delivery is actually short.

    # apply_measured_fps inherited from GenericCamera: writes self.measured_fps

    #: Capture queue depth on V4L2 (Linux, Jetson). With ONE buffer the driver
    #: has nowhere to put a frame that arrives while the host still holds the
    #: previous one, so it drops it: a camera faster than the host's per-frame
    #: work loses every other frame. Measured on the Jetson rig, 640x480 MJPEG
    #: (a 120 fps mode): one buffer 49.5 fps, two 98.8 fps, and frame age the
    #: same for both (median 14 ms, p95 18 ms), so the second buffer costs no
    #: freshness. Every other backend keeps a queue of one.
    _V4L2_QUEUE_DEPTH = 2

    def _queue_depth(self) -> int:
        """Capture queue depth to request for the backend this camera opened on."""
        if getattr(self, "_open_backend", None) == "V4L2":
            return self._V4L2_QUEUE_DEPTH
        return 1

    def _seal_buffersize(self) -> None:
        """Re-apply the one-frame queue after the format seal, and say what
        the driver actually did with it.

        This matters more than a normal property check. ``get_available_images``
        documents that ``read()`` returns "one fresh frame" per call, and the
        whole live-latency story rests on that being true. It is only true if
        the driver honours a queue depth of 1. When it does not, the driver
        keeps handing back the OLDEST queued frame: every frame is still
        delivered and still recorded, so nothing looks broken and no drop is
        counted, but each one reaches the host N frame-periods after the light
        that made it. Host-side stamping happens at retrieval, so that delay is
        invisible to every measurement the pipeline takes of itself, and it
        lands in full on anything that reacts to what a frame shows.

        Cost is frame period times queue depth, which is why a 30 fps rig
        suffers where a 100 fps one looks fine on the same hardware.

        ``CAP_PROP_BUFFERSIZE`` is not supported on every backend. DirectShow
        in particular accepts the call and ignores it, so the request is not
        evidence, and neither is the readback on backends that do not report
        it. Both are logged for exactly that reason.
        """
        want = self._queue_depth()
        try:
            accepted = bool(self._cap.set(cv2.CAP_PROP_BUFFERSIZE, want))
            got = self._cap.get(cv2.CAP_PROP_BUFFERSIZE)
        except Exception as e:                    # backend without the property
            logger.warning(
                "Camera %s: could not apply a %d-frame capture queue (%s). "
                "Frames may reach the pipeline several periods after capture.",
                self._camera_id, want, e)
            return
        backend = getattr(self, "_open_backend", "?")
        readable = got is not None and got > 0
        if accepted and (not readable or int(got) == want):
            logger.info("Camera %s: capture queue depth %s (backend=%s)",
                        self._camera_id,
                        int(got) if readable else f"{want} (not reported back)",
                        backend)
            return
        # Either the set was refused, or the driver reports a deeper queue.
        self.buffer_warning = (
            f"Camera {self._camera_id}: the {backend} backend did not accept a "
            f"{want}-frame capture queue (set={'ok' if accepted else 'refused'}, "
            f"reads back {int(got) if readable else 'unreported'}). Frames are "
            f"delivered from the driver's own queue, so each one reaches the "
            f"pipeline later than it was captured by roughly the queue depth "
            f"times the frame period. Recording is unaffected; anything "
            f"reacting to frame content is not."
        )
        logger.warning(self.buffer_warning)

    def stop_capturing(self) -> None:
        self._capturing = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
            logger.debug(f"OpenCVCamera {self._camera_id} released")

    def get_available_images(self) -> dict | None:
        """Return the freshest frame from the camera (one per call).

        ``read()`` blocks until the next frame; with ``CAP_PROP_BUFFERSIZE=1``
        the driver holds only the latest, so each call yields exactly one
        fresh frame and the capture loop publishes EVERY frame, display
        runs at the full capture rate. The loop spins with no sleep while
        frames are flowing, so consecutive calls drain the camera at
        delivery rate, one fresh frame per call.

        OpenCV exposes no per-frame hardware timestamp, so the frame is
        host-stamped at retrieval (``timestamp_source = "host_backstamped"``).
        """
        if not self._cap or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        return {
            "images": [frame],
            "timestamps": [host_clock.host_ns()],
            "dropped_frames": 0,
            "timestamp_source": "host_backstamped",
        }

    def reconnect(self) -> bool:
        """Attempt to reconnect the camera (for error recovery)."""
        self.stop_capturing()
        time.sleep(0.5)
        try:
            self.begin_capturing()
            return True
        except RuntimeError:
            return False
