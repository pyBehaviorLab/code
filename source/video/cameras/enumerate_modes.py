"""What a camera OFFERS, asked rather than guessed.

A USB Video Class camera lists, in its descriptors, every frame size it
supports and for each size the exact frame intervals it can be set to. That
list is the ground truth: a camera listing 30, 20 and 15 does not have 25, and
asking for 25 makes the driver pick the nearest it does have, report success,
and report 25 back when asked.

``cv2.VideoCapture`` exposes none of this. Its whole interface is ``set`` then
``get``, and ``get`` returns the request on most UVC drivers, so an
OpenCV-only application can only learn a camera's modes by setting one and
counting frames. That is what this rig used to do, and measuring cannot answer
the question:

  * a measured rate is the capability multiplied by the CONDITIONS. The same
    camera at 1920x1080 measured 30.13 fps and then 24.88 fps minutes apart,
    because auto-exposure lengthens integration in dimmer light. Calibrate in
    a dim room and a rate the camera really has is never offered again.
  * a UVC driver renegotiates the pixel format whenever size or rate changes
    and lands back on uncompressed. This rig stored 1.0 fps at 3840x2160,
    which is exactly the camera's uncompressed row; its MJPEG row is 30.
  * a fixed ladder of candidate sizes can only find sizes it thinks to try.
    The same camera offers 2592x1944 and 2048x1536, which were never probed.

Every platform backend can answer properly: DirectShow through
``IAMStreamConfig::GetStreamCaps``, Media Foundation through ``IMFMediaType``,
V4L2 through ``VIDIOC_ENUM_FRAMEINTERVALS``. None is reachable from OpenCV, but
FFmpeg reads all of them and is already a hard requirement of this application
because it encodes every recording. So the enumeration is asked of FFmpeg, and
measuring is left to answer the different question it is good at: what the
camera DELIVERS right now, under this light and this load.

See ``docs/dev/camera-modes-and-rates.md`` for the full reasoning and the
measurements behind it.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from source.log import get_logger

logger = get_logger()

#: Long enough for a slow hub, short enough that a wedged call cannot hold up
#: a dialog. The call itself is about a second on this rig.
_TIMEOUT_S = 25

#: ``vcodec=mjpeg  min s=1920x1080 fps=5 max s=1920x1080 fps=30``
#: ``pixel_format=yuyv422  min s=640x480 fps=1 max s=640x480 fps=30``
_DSHOW_OPTION = re.compile(
    r"(?:vcodec|pixel_format)=(?P<fmt>\S+)\s+"
    r"min\s+s=(?P<minw>\d+)x(?P<minh>\d+)\s+fps=(?P<minfps>[\d.]+)\s+"
    r"max\s+s=(?P<maxw>\d+)x(?P<maxh>\d+)\s+fps=(?P<maxfps>[\d.]+)")

#: V4L2 through FFmpeg: ``Raw       :     yuyv422 :  640x480 320x240``
_V4L2_FORMAT = re.compile(
    r"^\s*(?:Raw|Compressed)\s*:\s*(?P<fmt>\S+)\s*:\s*(?P<sizes>.+)$")
_SIZE = re.compile(r"(\d+)x(\d+)")

#: What FFmpeg calls a format against what the rig calls it. The rig's names
#: are the ones the Format cell offers and the capture backend understands.
_FORMAT_ALIASES = {
    "mjpeg": "mjpeg", "mjpg": "mjpeg",
    "yuyv422": "yuy2", "yuv422p": "yuy2", "yuyv": "yuy2",
    "h264": "h264", "nv12": "nv12", "bgr24": "bgr24",
}

#: Standard UVC frame intervals, used only to fill a min..max RANGE. A
#: DirectShow device reports a range rather than a list, and the true list is
#: a subset of these. Anything outside the reported range is excluded, and a
#: rate the camera turns out not to have is caught when it is opened.
_UVC_STANDARD_RATES = (5.0, 7.5, 10.0, 15.0, 20.0, 24.0, 25.0, 30.0,
                       50.0, 60.0, 100.0, 120.0)


@dataclass(frozen=True)
class OfferedMode:
    """One (size, pixel format) the camera lists, and the rates it lists for it."""

    width: int
    height: int
    pixel_format: str          # the rig's name: mjpeg, yuy2, h264, ...
    rates: tuple = field(default_factory=tuple)

    @property
    def size(self) -> tuple:
        return (self.width, self.height)

    @property
    def max_rate(self) -> float:
        return max(self.rates) if self.rates else 0.0


def _rates_in_range(lo: float, hi: float) -> tuple:
    """The rates a reported ``lo..hi`` capability actually offers.

    DirectShow reports one capability per (format, size) with a minimum and a
    maximum frame interval, and the two cases mean different things. Measured
    on this rig with ``ffmpeg -list_options``:

        cam5b4028a9  mjpeg 640x480   min=10       max=60.0002    a RANGE
        camc1aefd1e  mjpeg 640x480   min=5        max=30         a RANGE
        cam519bc077  mjpeg 640x480   min=120.101  max=120.101    a SINGLE

    ``min == max`` is the device reporting a DISCRETE interval: UVC
    ``bFrameIntervalType`` >= 1, one frame descriptor per supported interval.
    cam519bc077 reports that way for every mode it has, so its 640x480 MJPEG
    really is 120 fps and nothing else, while its 640x480 YUY2 really is 30
    and nothing else. Offering the ladder below such a value invents rates the
    camera does not have, which is worse than a short list: the operator picks
    30, the driver rounds to the one interval it owns, and the recording is
    written at a rate nobody chose.

    ``min < max`` is a CONTINUOUS range (``bFrameIntervalType`` 0), where any
    interval between the two is valid, so the standard UVC rates inside it are
    offered along with the endpoints.

    Fractional standard rates are excluded. 7.5 is a real UVC interval, but
    every picker in this application labels a rate as a whole number, so it
    reached the operator as "8" and would have been requested as 8.
    """
    lo, hi = min(lo, hi), max(lo, hi)
    if hi <= 0:
        return ()
    if abs(hi - lo) < 1e-6:
        # Discrete: this interval, and no other.
        return (round(hi, 3),)
    inside = [r for r in _UVC_STANDARD_RATES
              if lo - 1e-6 <= r <= hi + 1e-6 and float(r).is_integer()]
    out = set(inside)
    # The endpoints are exact values from the device, so they belong in the
    # list, but not when a standard rate already stands in for them: a
    # reported 60.0002 beside the standard 60.0 became two items both
    # labelled "60".
    for edge in (lo, hi):
        if edge > 0 and not any(abs(r - edge) < 0.5 for r in out):
            out.add(round(edge, 3))
    return tuple(sorted(out))


def _merge(modes: List[OfferedMode]) -> List[OfferedMode]:
    """One entry per (size, format), rates unioned, largest size first."""
    by_key: dict = {}
    for m in modes:
        key = (m.width, m.height, m.pixel_format)
        by_key.setdefault(key, set()).update(m.rates)
    out = [OfferedMode(w, h, f, tuple(sorted(rates)))
           for (w, h, f), rates in by_key.items()]
    out.sort(key=lambda m: (-(m.width * m.height), m.pixel_format))
    return out


def _run_ffmpeg(args: List[str]) -> str:
    """FFmpeg's stderr for a listing call.

    It exits non-zero by design when listing: it prints what was asked for and
    then fails to open the dummy input, so the return code says nothing.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", *args],
        capture_output=True, text=True, timeout=_TIMEOUT_S,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return proc.stderr or ""


def _parse_dshow(text: str) -> List[OfferedMode]:
    modes: List[OfferedMode] = []
    for line in text.splitlines():
        m = _DSHOW_OPTION.search(line)
        if not m:
            continue
        fmt = _FORMAT_ALIASES.get(m.group("fmt").lower(), m.group("fmt").lower())
        try:
            w, h = int(m.group("maxw")), int(m.group("maxh"))
            rates = _rates_in_range(float(m.group("minfps")),
                                    float(m.group("maxfps")))
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0 and rates:
            modes.append(OfferedMode(w, h, fmt, rates))
    return _merge(modes)


def _parse_v4l2(text: str) -> List[OfferedMode]:
    modes: List[OfferedMode] = []
    for line in text.splitlines():
        m = _V4L2_FORMAT.match(line)
        if not m:
            continue
        fmt = _FORMAT_ALIASES.get(m.group("fmt").lower(), m.group("fmt").lower())
        for sw, sh in _SIZE.findall(m.group("sizes")):
            # FFmpeg's V4L2 lister prints sizes without their intervals. The
            # rates are filled in by ``v4l2-ctl`` below when it is present,
            # and left empty otherwise so a caller can tell "not known" from
            # "none", rather than a guess being recorded as an enumeration.
            modes.append(OfferedMode(int(sw), int(sh), fmt, ()))
    return _merge(modes)


def _v4l2_ctl_rates(device: str) -> List[OfferedMode]:
    """Exact sizes AND intervals from ``v4l2-ctl``, when it is installed.

    This is the only source on any platform that reports the camera's real
    interval LIST rather than a range, so it is preferred where available.
    """
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "-d", str(device), "--list-formats-ext"],
            capture_output=True, text=True, timeout=_TIMEOUT_S)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []
    modes: List[OfferedMode] = []
    fmt = ""
    size: Optional[tuple] = None
    rates: List[float] = []
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        m = re.match(r"\[\d+\]:\s*'(\w+)'", line)
        if m:
            if size and rates:
                modes.append(OfferedMode(size[0], size[1], fmt, tuple(sorted(rates))))
            fmt = _FORMAT_ALIASES.get(m.group(1).lower(), m.group(1).lower())
            size, rates = None, []
            continue
        m = re.match(r"Size:\s*\w+\s*(\d+)x(\d+)", line)
        if m:
            if size and rates:
                modes.append(OfferedMode(size[0], size[1], fmt, tuple(sorted(rates))))
            size, rates = (int(m.group(1)), int(m.group(2))), []
            continue
        m = re.search(r"Interval:.*\(([\d.]+)\s*fps\)", line)
        if m and size:
            try:
                rates.append(round(float(m.group(1)), 3))
            except ValueError:
                pass
    if size and rates:
        modes.append(OfferedMode(size[0], size[1], fmt, tuple(sorted(rates))))
    return _merge(modes)


def enumerate_modes(device: str) -> List[OfferedMode]:
    """Every (size, format, rates) ``device`` reports, or ``[]`` if unknown.

    ``device`` is a DirectShow moniker or name on Windows (the
    ``@device_pnp_`` path that ``identity.list_dshow_devices`` already
    records), and a ``/dev/videoN`` path on Linux.

    An empty list means the question could not be ASKED, not that the camera
    has no modes. Callers must fall back to the trial probe rather than treat
    it as an answer.
    """
    if not device:
        return []
    try:
        if sys.platform == "win32":
            text = _run_ffmpeg(["-f", "dshow", "-list_options", "true",
                                "-i", f"video={device}"])
            modes = _parse_dshow(text)
        else:
            exact = _v4l2_ctl_rates(device)
            if exact:
                modes = exact
            else:
                text = _run_ffmpeg(["-f", "v4l2", "-list_formats", "all",
                                    "-i", str(device)])
                modes = _parse_v4l2(text)
    except FileNotFoundError:
        logger.warning(
            "camera modes: ffmpeg not found, so a camera's offered modes "
            "cannot be read and have to be probed by trial, which measures "
            "the room's lighting as much as the camera.")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("camera modes: listing %s timed out after %d s.",
                       device, _TIMEOUT_S)
        return []
    except Exception as e:
        logger.warning("camera modes: listing %s failed (%s).", device, e)
        return []
    if modes:
        logger.info("camera modes: %s offers %d (size, format) combination(s), "
                    "largest %dx%d, top rate %.0f fps.",
                    device, len(modes), modes[0].width, modes[0].height,
                    max((m.max_rate for m in modes), default=0.0))
    return modes


def rates_for(modes: List[OfferedMode], size, pixel_format=None) -> tuple:
    """The rates ``size`` is offered at, across formats or in one of them.

    Returns ``()`` when the size is not offered, which is different from a
    size offered at no rate and must not be smoothed into it.
    """
    want = tuple(size) if size else None
    if not want:
        return ()
    rates: set = set()
    for m in modes or ():
        if m.size != want:
            continue
        if pixel_format and m.pixel_format != pixel_format:
            continue
        rates.update(m.rates)
    return tuple(sorted(rates))


def sizes_for(modes: List[OfferedMode], pixel_format=None) -> List[tuple]:
    """Every size offered, largest first, optionally in one pixel format."""
    seen, out = set(), []
    for m in modes or ():
        if pixel_format and m.pixel_format != pixel_format:
            continue
        if m.size not in seen:
            seen.add(m.size)
            out.append(m.size)
    return out
