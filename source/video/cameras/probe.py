"""Backend-agnostic camera calibration probe.

One entry point, :func:`probe_camera`, drives any backend's
``probe_supported_resolutions`` + ``measure_fps_at`` primitives and returns
a :class:`ProbeResult`. It emits progress through a plain ``report`` callable
(``report(kind, **data)``) so callers can surface live updates without this
module ever touching Qt.

The report protocol (kinds and payloads):
    report("resolutions", modes=[(w, h), ...])          once, after probing
    report("measuring",   w=, h=, index=i, total=n)     per mode, before measure
    report("mode_done",   w=, h=, fps=)                  per mode, after measure

Backends also emit their own granular sub-events (``resolution_found``,
``fps_measured``); unknown kinds are safe to ignore.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from source.video.cameras.base import SHORTFALL_RATIO as _SHORTFALL_RATIO
from source.log import get_logger

logger = get_logger()

ReportFn = Optional[Callable[..., None]]


@dataclass
class ProbeResult:
    """Outcome of probing one camera.

    ``modes`` is the list of ``(width, height, realistic_fps)`` tuples, the
    same shape ``CameraConfig.probed_modes`` stores. ``error`` is set (and
    ``modes`` left empty) when the camera could not be probed. ``degraded``
    marks a best-effort single-mode result when the ladder probe found
    nothing but the camera was still reachable.
    """

    camera_id: str
    backend: str
    modes: list[tuple[int, int, float]] = field(default_factory=list)
    error: str | None = None
    degraded: bool = False


def _emit(report: ReportFn, kind: str, **data: Any) -> None:
    """Call ``report`` defensively, a caller's handler must never break a probe."""
    if report is None:
        return
    try:
        report(kind, **data)
    except Exception as e:  # pragma: no cover - handler bugs are the caller's
        logger.debug("probe report(%s) handler error: %s", kind, e)


def _backend_class(backend: str):
    """Return the camera class for a backend id, or raise ValueError."""
    backend = (backend or "opencv").lower()
    if backend == "opencv":
        from source.video.cameras.opencv import OpenCVCamera
        return OpenCVCamera
    if backend == "spinnaker":
        from source.video.cameras.spinnaker import SPINNAKER_AVAILABLE, SpinnakerCamera
        if not SPINNAKER_AVAILABLE:
            raise ValueError("backend not available (SDK missing)")
        return SpinnakerCamera
    if backend == "ximea":
        from source.video.cameras.ximea import XIMEA_AVAILABLE, XimeaCamera
        if not XIMEA_AVAILABLE:
            raise ValueError("backend not available (SDK missing)")
        return XimeaCamera
    raise ValueError(f"Unknown camera backend: {backend}")


def _try_degrade(cls, camera_id: Any, backend: str, target_fps: float,
                 report: ReportFn) -> list[tuple[int, int, float]] | None:
    """When the ladder probe found nothing, try a single native mode.

    Returns a one-entry ``[(w, h, fps)]`` list if the camera is reachable
    (native resolution readable), else ``None``.
    """
    native_fn = getattr(cls, "native_mode", None)
    if not callable(native_fn):
        return None
    try:
        native = native_fn(camera_id)
    except Exception as e:
        logger.debug("native_mode(%s) failed: %s", camera_id, e)
        return None
    if not native:
        return None
    w, h = int(native[0]), int(native[1])
    fps = 0.0
    try:
        fps = float(cls.measure_fps_at(camera_id, w, h, target_fps=target_fps,
                                       report=report) or 0.0)
    except Exception as e:
        logger.debug("degrade measure_fps_at(%s) failed: %s", camera_id, e)
    return [(w, h, fps)]


def _rate_targets(camera_id, backend: str, pixel_format: str) -> dict:
    """``{(w, h): rate}`` the camera ADVERTISES for each size in this format.

    Every mode used to be measured while asking for a fixed 30 fps, and the
    number that came back was filed as that size's ceiling. It is not one.
    Measured on this rig, one camera in MJPEG:

        320x240   offers 120, asked 30 -> recorded 32.8
        800x600   offers  60, asked 30 -> recorded 33.2
        640x480   offers 120, asked 30 -> recorded 99.8

    The first two honoured the 30; the third has only a 120 interval and
    ignored it. So the column was a mixture of "what 30 delivered here" and
    "what this mode does when it refuses 30", presented as a capability. A
    mode is measured at ITS OWN advertised rate, so the number means the one
    thing worth knowing: what this mode delivers when it is set to the rate
    the camera says it has.

    Empty when the camera has not been enumerated, and the caller falls back
    to its default target.
    """
    if (backend or "opencv").lower() != "opencv":
        return {}
    try:
        from source.video.cameras import calibration_store as _store
        from source.video.cameras.enumerate_modes import rates_for
        from source.video.cameras.usb_identity import resolve_identity
        uid = resolve_identity(camera_id, "opencv").get("unique_id")
        offered = _store.get_offered(uid)
        if not offered:
            return {}
        out = {}
        for m in offered:
            if pixel_format and m.pixel_format != str(pixel_format).lower():
                continue
            rates = rates_for(offered, (m.width, m.height), m.pixel_format)
            if rates:
                out[(int(m.width), int(m.height))] = float(max(rates))
        return out
    except Exception as e:
        logger.debug("rate targets for %s: %s", camera_id, e)
        return {}


def _fmt_kw(cls, pixel_format) -> dict:
    """``{"pixel_format": ...}`` for backends that take one, else ``{}``.

    Only the UVC backend has a format to choose. A machine-vision SDK
    delivers one pixel layout and its probe signature has no such argument,
    so passing it would be a TypeError rather than a refinement.
    """
    if not pixel_format:
        return {}
    import inspect
    try:
        params = inspect.signature(cls.probe_supported_resolutions).parameters
    except (TypeError, ValueError):
        return {}
    return {"pixel_format": pixel_format} if "pixel_format" in params else {}


def probe_camera(camera_id: Any, backend: str, target_fps: float,
                 report: ReportFn = None,
                 pixel_format: str = "mjpeg") -> ProbeResult:
    """Probe one camera for supported ``(w, h, fps)`` modes.

    Dispatches on ``backend`` ("opencv"/"spinnaker"/"ximea"), runs that
    backend's resolution probe, then measures a realistic FPS at each mode.
    All exceptions are caught and folded into ``ProbeResult.error`` so a
    single bad camera never aborts a multi-camera batch.
    """
    cam = str(camera_id)
    backend = (backend or "opencv").lower()
    try:
        target_fps = float(target_fps or 30.0)
    except (TypeError, ValueError):
        target_fps = 30.0

    try:
        cls = _backend_class(backend)
    except ValueError as e:
        return ProbeResult(cam, backend, error=str(e))

    try:
        modes, _max_mode = cls.probe_supported_resolutions(
            camera_id, report=report, **_fmt_kw(cls, pixel_format))
    except Exception as e:
        logger.warning("probe_supported_resolutions(%s, %s) failed: %s",
                       cam, backend, e)
        return ProbeResult(cam, backend, error=f"resolution probe failed: {e}")

    degraded = False
    if not modes:
        # Reachable-but-empty: fall back to the native mode so the camera
        # still gets a usable single-mode calibration.
        fallback = _try_degrade(cls, camera_id, backend, target_fps, report)
        if fallback:
            _emit(report, "resolutions", modes=[(w, h) for (w, h, _f) in fallback])
            for i, (w, h, fps) in enumerate(fallback, start=1):
                _emit(report, "measuring", w=w, h=h, index=i, total=len(fallback))
                _emit(report, "mode_done", w=w, h=h, fps=fps)
            return ProbeResult(cam, backend, modes=list(fallback), degraded=True)
        return ProbeResult(cam, backend, error="no supported modes detected")

    _emit(report, "resolutions", modes=[(int(w), int(h)) for (w, h) in modes])

    targets = _rate_targets(camera_id, backend, pixel_format)

    # Fast path: measure every mode on ONE open handle when the backend
    # supports it (~3x faster than re-opening the camera per mode). Emits its
    # own per-mode measuring/mode_done events. Falls back to the per-mode loop
    # if it's unavailable or returns nothing.
    #
    # Skipped when the modes want DIFFERENT rates from each other, which is
    # the normal case: it takes one target for the whole batch, so using it
    # would measure every size at one rate, which is the fault this is fixing.
    batched = getattr(cls, "measure_fps_for_modes", None)
    if len(set(targets.values())) > 1:
        batched = None
    if callable(batched):
        try:
            measured = list(batched(
                camera_id, [(int(w), int(h)) for (w, h) in modes],
                target_fps=target_fps, report=report,
                **_fmt_kw(cls, pixel_format)) or [])
        except Exception as e:
            logger.debug("measure_fps_for_modes(%s) failed: %s", cam, e)
            measured = []
        # Accept the fast path ONLY when it measured every mode, a short
        # result means the shared-handle probe bailed (resolution switch not
        # honoured), so fall through to the reliable per-mode loop rather than
        # dropping the unmeasured modes.
        if measured and len(measured) == len(modes):
            return ProbeResult(cam, backend, modes=measured, degraded=degraded)

    result_modes: list[tuple[int, int, float]] = []
    total = len(modes)
    for i, (w, h) in enumerate(modes, start=1):
        _emit(report, "measuring", w=int(w), h=int(h), index=i, total=total)
        want = targets.get((int(w), int(h)), target_fps)
        try:
            fps = float(cls.measure_fps_at(
                camera_id, int(w), int(h), target_fps=want,
                report=report, **_fmt_kw(cls, pixel_format)) or 0.0)
        except Exception as e:
            logger.debug("measure_fps_at(%s, %dx%d) failed: %s", cam, w, h, e)
            fps = 0.0
        _emit(report, "mode_done", w=int(w), h=int(h), fps=fps)
        result_modes.append((int(w), int(h), fps))

    return ProbeResult(cam, backend, modes=result_modes, degraded=degraded)


@dataclass(frozen=True)
class Variant:
    """What a camera can do THROUGH one particular door.

    ``backend`` is the OS capture backend for UVC ("dshow", "msmf", "v4l2",
    "avfoundation", "any") or the SDK family name for a machine-vision camera
    ("spinnaker", "ximea"), where there is only one door and it is
    authoritative.
    """

    backend: str
    modes: list = None          # [(w, h, measured_fps), ...]
    error: str = ""

    def best_fps(self) -> float:
        return max((float(m[2]) for m in (self.modes or [])), default=0.0)

    def max_pixels(self) -> int:
        return max((int(m[0]) * int(m[1]) for m in (self.modes or [])),
                   default=0)


#: Seconds to wait for the OS to finish releasing a camera between backend
#: probes. 1.5 s was enough here for Media Foundation to open a device
#: DirectShow had just closed; the retry below covers a slower host.
BACKEND_RELEASE_S = 1.5


def _probe_one_backend(camera_id, name, cv_backend, target_fps, report,
                       pixel_format="mjpeg"):
    """Probe through one backend, retrying once if it found nothing.

    An empty result is ambiguous, a backend that genuinely cannot serve this
    camera looks exactly like one that arrived before the device was free.
    Retrying once separates them, and only costs time on the failing path.
    """
    from source.video.cameras import opencv as _ocv
    for attempt in (1, 2):
        with _ocv.use_backend(cv_backend):
            res = probe_camera(camera_id, "opencv", target_fps,
                               report=report, pixel_format=pixel_format)
        if res.modes or attempt == 2:
            return res
        logger.info("probe: %s via %s found nothing; the device may still be "
                    "held by the previous backend, retrying once after %.1fs",
                    camera_id, name, BACKEND_RELEASE_S * 2)
        time.sleep(BACKEND_RELEASE_S * 2)
    return res


def probe_all(camera_id: Any, family: str = "opencv",
              target_fps: float = 30.0, report=None,
              pixel_format: str = "mjpeg", doors=None) -> list:
    """Every door into this camera, measured.

    For a UVC camera each OS backend is probed in turn, because the backend
    decides whether the MJPG request is honoured and therefore what the camera
    can deliver. For a machine-vision camera there is one SDK and its answer
    already accounts for exposure, ROI and link speed, so it is asked once.

    Slow by nature; this is what the machine-level calibration cache exists
    to avoid repeating.
    """
    family = (family or "opencv").lower()

    if family != "opencv":
        res = probe_camera(camera_id, family, target_fps, report=report,
                           pixel_format=pixel_format)
        return [Variant(backend=family, modes=list(res.modes or []),
                        error=res.error or "")]

    from source.video.cameras import opencv as _ocv
    out = []
    # FRAGILE DOORS FIRST, while the device has not been opened yet.
    #
    # Media Foundation will not take a camera DirectShow has just closed: on
    # this rig it reported "device is busy" 1.5 s and again 3.0 s after the
    # DirectShow probe finished, so it measured zero modes and was recorded
    # as a door that cannot serve the camera. Probed FIRST, on an untouched
    # device, the same camera gave three modes at 25.7 fps. DirectShow opens
    # a device Media Foundation has released without complaint, so the cost
    # of this order is nothing.
    # ONLY the door the operator chose, when they chose one. Walking every
    # door on every Detect measured backends nobody was going to open and,
    # on this rig, cost 50-73 s per camera in Media Foundation opens alone.
    # The door is a choice in the dialog now, so Detect answers for it.
    available = _ocv._uvc_backends()
    wanted = [d for d in (doors or []) if d in available]
    if wanted:
        door_items = [(d, available[d]) for d in wanted]
    else:
        door_items = sorted(available.items(),
                            key=lambda kv: 0 if kv[0] == "msmf" else 1)
    for i, (name, cv_backend) in enumerate(door_items):
        try:
            # A USB camera is not free the instant the previous handle is
            # dropped: the OS tears the device down asynchronously, and the
            # next backend meets a device that is still claimed. Measured in
            # the GUI, DirectShow probed fine and Media Foundation reported
            # "could not open camera" and zero modes, which then looks like a
            # backend that cannot serve the camera rather than one that was
            # never given a chance, and the rig keeps the slower door.
            if i:
                time.sleep(BACKEND_RELEASE_S)
            res = _probe_one_backend(camera_id, name, cv_backend, target_fps,
                                     report, pixel_format=pixel_format)
            out.append(Variant(backend=name, modes=list(res.modes or []),
                               error=res.error or ""))
            logger.info("probe: %s via %s -> %d mode(s), best %.1f fps",
                        camera_id, name, len(res.modes or []),
                        max((float(m[2]) for m in (res.modes or [])),
                            default=0.0))
        except Exception as e:
            logger.warning("probe: probing %s via %s failed: %s",
                           camera_id, name, e)
            out.append(Variant(backend=name, modes=[], error=str(e)))
    return out


def merged_modes(variants: list) -> list:
    """One ``[(w, h, fps)]`` list holding the BEST rate any door achieved.

    What the operator should be offered. Showing one backend's numbers
    presents its limits as the camera's: this rig's webcam listed
    "1280x720 @ 10.0" because that is what DirectShow manages, while the same
    camera does 30 through Media Foundation. Picking the mode and picking the
    door that serves it are one decision, so the list is the union and the
    door is resolved from it.
    """
    best: dict = {}
    for v in variants or []:
        for m in (v.modes or []):
            try:
                key = (int(m[0]), int(m[1]))
                fps = float(m[2])
            except (TypeError, ValueError, IndexError):
                continue
            if fps > best.get(key, 0.0):
                best[key] = fps
    return [(w, h, fps) for (w, h), fps in
            sorted(best.items(), key=lambda kv: -kv[0][0] * kv[0][1])]


def best_for(variants: list, want_w: int, want_h: int,
             want_fps: float) -> Optional[tuple]:
    """``(backend, (w, h, fps))`` that best serves the request, or ``None``.

    Preference order, in the order an operator would apply it:

    1. the requested size, at the highest measured rate any door achieves,
       this is the choice that turned 10 fps into 30 on this rig;
    2. failing that, the largest size that still meets the requested rate,
       because a session that needs 30 fps needs it more than it needs pixels;
    3. failing that, whatever delivers the highest rate at all.

    A tie goes to the FIRST variant, which is the platform's preferred
    backend, so an even contest never moves the rig off its default.
    """
    want_wh = (int(want_w), int(want_h))
    exact = []
    for v in variants or []:
        for m in (v.modes or []):
            if (int(m[0]), int(m[1])) == want_wh:
                exact.append((v.backend, (int(m[0]), int(m[1]), float(m[2]))))
    if exact:
        return max(exact, key=lambda bm: bm[1][2])

    meets = []
    for v in variants or []:
        for m in (v.modes or []):
            if float(m[2]) >= float(want_fps) * _SHORTFALL_RATIO:
                meets.append((v.backend,
                              (int(m[0]), int(m[1]), float(m[2]))))
    if meets:
        return max(meets, key=lambda bm: bm[1][0] * bm[1][1])

    everything = [(v.backend, (int(m[0]), int(m[1]), float(m[2])))
                  for v in (variants or []) for m in (v.modes or [])]
    if not everything:
        return None
    return max(everything, key=lambda bm: bm[1][2])
