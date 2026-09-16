"""Shipped camera backends, expressed as registry specs.

Each backend contributes construction plus feature introspection. The
introspection is what lets the setup dialog render itself: a webcam reports a
handful of features, a FLIR reports its whole node map, and the GUI treats both
the same way.

rig-verify: the Spinnaker node walk and the Ximea parameter probe are written
against the documented SDK surfaces and have not yet been run on hardware in
this repo, the same caveat the existing probe paths carry. Every SDK call is
individually guarded, so an unexpected model degrades to fewer features rather
than failing.
"""

from __future__ import annotations

import contextlib
import logging

from .features import (
    ACCESS_RO,
    GROUP_ACQUISITION,
    GROUP_COLOUR,
    GROUP_DIAGNOSTICS,
    GROUP_EXPOSURE,
    GROUP_FORMAT,
    GROUP_IDENTITY,
    GROUP_IO,
    GROUP_OTHER,
    GROUP_TRANSPORT,
    GROUP_TRIGGER,
    KIND_BOOL,
    KIND_COMMAND,
    KIND_ENUM,
    KIND_FLOAT,
    KIND_INT,
    KIND_STRING,
    TIER_CURATED,
    TIER_FULL,
    CameraFeature,
    apply_curated,
    enum_feature,
    number_feature,
)
from .registry import BackendSpec, register_backend

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- OpenCV ---

def _opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:                                   # pragma: no cover
        return False


def _opencv_list():
    from .factory import _list_opencv_cameras
    return _list_opencv_cameras()


def _opencv_create(unique_id: str, identifier: str, config: dict):
    from .opencv import OpenCVCamera
    width, height = config.get("width"), config.get("height")
    if width is None or height is None:
        raise ValueError(
            "OpenCV camera requires explicit 'width' and 'height' in config so "
            "a driver that quietly delivers a smaller frame can't misplace "
            "ROIs. Use OpenCVCamera.probe_supported_resolutions() to discover "
            "valid modes.")
    try:
        camera_id = int(identifier)
    except ValueError:
        camera_id = identifier                          # path-based id
    # ``capture_backend`` is a NAME ("dshow"/"msmf"/"v4l2") because that is
    # what a project file can carry across machines and OS versions; the cv2
    # constants are integers whose values are an implementation detail. An
    # unknown or absent name means "walk the platform's preference order",
    # which is the behaviour every camera had before the capability probe.
    from source.video.cameras.opencv import cv_backend_for as _cv_backend_for
    return OpenCVCamera(camera_id=camera_id, width=int(width),
                        height=int(height), fps=config.get("fps", 20),
                        capture_format=config.get("capture_format"),
                        cv_backend=_cv_backend_for(config.get("capture_backend")))


def _opencv_features(camera) -> list[CameraFeature]:
    """UVC exposes almost nothing portably, and saying so is the point.

    Every one of these is already a control in the Acquisition card above:
    the rate is its FPS picker, the transport format is its Pixel format
    picker, and width/height are its Resolution picker (the read-only rows
    here even said "Chosen by the resolution picker"). Shown a second time
    they were not extra capability, they were the same four settings in a
    second place with different widgets, and the rate row read 1 fps while
    the picker above it read 30, because nothing kept them in step.

    So they are reported at ``TIER_FULL``: still listed under "All features",
    where the point is that no option is unreachable, and absent from the
    curated panel, which is for what the card does NOT already own. A webcam
    therefore shows no curated rows at all, and the group hides itself.
    """
    feats = [
        number_feature("frame_rate", "Frame rate", group=GROUP_ACQUISITION,
                       unit="fps", minimum=1, maximum=240, live=True,
                       sdk_node="CAP_PROP_FPS", tier=TIER_FULL),
        enum_feature("capture_format", "Transport format",
                     [("mjpeg", "MJPG, compressed, full rate"),
                      ("yuv", "YUV, uncompressed, may cap rate")],
                     group=GROUP_FORMAT, sdk_node="CAP_PROP_FOURCC",
                     tier=TIER_FULL,
                     help="UVC transport encoding. Not a sensor pixel format."),
    ]
    for key, label, getter in (("width", "Width", "get_width"),
                               ("height", "Height", "get_height")):
        try:
            value = int(getattr(camera, getter)())
        except Exception:
            continue
        feats.append(CameraFeature(key=key, label=label, group=GROUP_FORMAT,
                                   kind=KIND_INT, access=ACCESS_RO,
                                   minimum=value, maximum=value,
                                   tier=TIER_FULL,
                                   help="Chosen by the resolution picker."))
    return apply_curated(feats)


# ------------------------------------------------------------ Spinnaker ---

def _spinnaker_available() -> bool:
    try:
        from .spinnaker import SPINNAKER_AVAILABLE
        return bool(SPINNAKER_AVAILABLE)
    except Exception:
        return False


def _spinnaker_list():
    from .spinnaker import list_available_cameras
    return list_available_cameras()


def _spinnaker_create(unique_id: str, identifier: str, config: dict):
    from .spinnaker import SpinnakerCamera
    cam = SpinnakerCamera(
        unique_id=unique_id,
        fps=config.get("fps", 30),
        exposure_us=config.get("exposure_us", 5000),
        gain_db=config.get("gain_db", 0),
        external_trigger=config.get("external_trigger", False),
        grayscale=config.get("grayscale", False),
        trigger=config.get("trigger"),
        line_output=config.get("line_output"),
    )
    # Scientific backends take resolution through the AOI path, not the
    # constructor, dropped here, the dialog's picker does nothing.
    width, height = config.get("width"), config.get("height")
    if width and height:
        try:
            cam.configure(roi=(0, 0, int(width), int(height)))
        except Exception as exc:
            logger.warning("Spinnaker: could not apply %sx%s: %s",
                           width, height, exc)
    return cam


# GenICam node -> canonical key. Anything not listed still appears, in the
# full tier, under its own vendor category.
_SPIN_KEYS = {
    "DeviceModelName": "model", "DeviceSerialNumber": "serial",
    "DeviceFirmwareVersion": "firmware", "DeviceTemperature": "temperature_c",
    "DeviceLinkCurrentThroughput": "link_throughput",
    "AcquisitionMode": "acquisition_mode",
    "AcquisitionFrameRateEnable": "frame_rate_enable",
    "AcquisitionFrameRate": "frame_rate",
    "AcquisitionResultingFrameRate": "resulting_frame_rate",
    "ExposureAuto": "exposure_auto", "ExposureTime": "exposure_us",
    "ExposureMode": "exposure_mode",
    "AutoExposureExposureTimeLowerLimit": "exposure_auto_min",
    "AutoExposureExposureTimeUpperLimit": "exposure_auto_max",
    "AutoExposureTargetGreyValue": "auto_target_grey",
    "GainAuto": "gain_auto", "Gain": "gain_db",
    "AutoGainLowerLimit": "gain_auto_min", "AutoGainUpperLimit": "gain_auto_max",
    "BlackLevel": "black_level",
    "PixelFormat": "pixel_format", "AdcBitDepth": "adc_bit_depth",
    "Width": "width", "Height": "height",
    "OffsetX": "offset_x", "OffsetY": "offset_y",
    "BinningHorizontal": "binning_h", "BinningVertical": "binning_v",
    "BinningHorizontalMode": "binning_mode",
    "DecimationHorizontal": "decimation_h", "DecimationVertical": "decimation_v",
    "ReverseX": "reverse_x", "ReverseY": "reverse_y",
    "TestPattern": "test_pattern",
    "BalanceWhiteAuto": "white_balance_auto",
    "Gamma": "gamma", "GammaEnable": "gamma_enable",
    "SharpeningEnable": "sharpness", "SaturationEnable": "saturation",
    "TriggerSelector": "trigger_selector", "TriggerMode": "trigger_mode",
    "TriggerSource": "trigger_source", "TriggerActivation": "trigger_activation",
    "TriggerDelay": "trigger_delay_us", "TriggerOverlap": "trigger_overlap",
    "TriggerSoftware": "trigger_software",
    "LineSelector": "line_selector", "LineMode": "line_mode",
    "LineSource": "line_source", "LineInverter": "line_inverter",
    "LineDebouncerTime": "line_debounce_us", "LineStatus": "line_status",
    "V3_3Enable": "supply_3v3",
    "CounterDuration": "strobe_width_us",
    "DeviceLinkThroughputLimit": "throughput_limit",
    "StreamBufferHandlingMode": "buffer_mode",
    "StreamBufferCountManual": "buffer_count",
    "GevSCPSPacketSize": "packet_size", "GevSCPD": "packet_delay",
    "ChunkModeActive": "chunk_enable", "UserSetSelector": "user_set",
}

# Nodes safe to push to a streaming camera.
_SPIN_LIVE = {
    "frame_rate", "frame_rate_enable", "exposure_us", "exposure_auto",
    "gain_db", "gain_auto", "black_level", "gamma", "gamma_enable",
    "sharpness", "saturation", "white_balance_auto", "reverse_x", "reverse_y",
    "test_pattern", "trigger_software", "exposure_auto_min",
    "exposure_auto_max", "gain_auto_min", "gain_auto_max", "auto_target_grey",
    "user_set", "indicator_led",
}

_SPIN_GROUPS = {
    "AnalogControl": GROUP_EXPOSURE, "AcquisitionControl": GROUP_ACQUISITION,
    "ImageFormatControl": GROUP_FORMAT, "DigitalIOControl": GROUP_IO,
    "TransportLayerControl": GROUP_TRANSPORT, "DeviceControl": GROUP_IDENTITY,
    "ChunkDataControl": GROUP_DIAGNOSTICS, "CounterAndTimerControl": GROUP_IO,
    "ColorTransformationControl": GROUP_COLOUR, "UserSetControl": GROUP_DIAGNOSTICS,
}


def _spinnaker_features(camera) -> list[CameraFeature]:
    import PySpin

    nodemap = getattr(camera, "_nodemap", None)
    if nodemap is None:
        return []

    feats: list[CameraFeature] = []
    seen: set[str] = set()

    def walk(node, category: str, depth: int = 0):
        if depth > 4:
            return
        try:
            cat = PySpin.CCategoryPtr(node)
            children = cat.GetFeatures()
        except Exception:
            return
        for child in children:
            try:
                if not PySpin.IsAvailable(child):
                    continue
                iface = child.GetPrincipalInterfaceType()
                name = child.GetName()
            except Exception:
                continue
            if iface == PySpin.intfICategory:
                with contextlib.suppress(Exception):
                    walk(child, PySpin.CCategoryPtr(child).GetName() or category,
                         depth + 1)
                continue
            if name in seen:
                continue
            seen.add(name)
            feat = _spin_feature(PySpin, child, name, category, iface)
            if feat is not None:
                feats.append(feat)

    try:
        walk(nodemap.GetNode("Root"), "")
    except Exception as exc:
        logger.warning("Spinnaker node walk failed: %s", exc)
        return []
    return apply_curated(feats)


def _spin_feature(PySpin, node, name, category, iface):
    key = _SPIN_KEYS.get(name, name)
    curated = name in _SPIN_KEYS
    group = _SPIN_GROUPS.get(category, GROUP_OTHER)
    try:
        writable = PySpin.IsWritable(node)
        readable = PySpin.IsReadable(node)
    except Exception:
        writable = readable = False
    access = "rw" if writable else ("ro" if readable else "wo")
    try:
        tip = node.GetToolTip() or node.GetDescription() or ""
    except Exception:
        tip = ""
    common = dict(key=key, label=_pretty(name), group=group, access=access,
                  live=key in _SPIN_LIVE, sdk_node=name, help=str(tip)[:240],
                  category=category or "Other",
                  tier=TIER_CURATED if curated else TIER_FULL)
    try:
        if iface == PySpin.intfIFloat:
            n = PySpin.CFloatPtr(node)
            return CameraFeature(kind=KIND_FLOAT, unit=_unit_for(name),
                                 minimum=n.GetMin() if readable else None,
                                 maximum=n.GetMax() if readable else None,
                                 **common)
        if iface == PySpin.intfIInteger:
            n = PySpin.CIntegerPtr(node)
            return CameraFeature(kind=KIND_INT, unit=_unit_for(name),
                                 minimum=n.GetMin() if readable else None,
                                 maximum=n.GetMax() if readable else None,
                                 increment=n.GetInc() if readable else None,
                                 **common)
        if iface == PySpin.intfIEnumeration:
            n = PySpin.CEnumerationPtr(node)
            options = []
            for entry in (n.GetEntries() if readable else []):
                try:
                    e = PySpin.CEnumEntryPtr(entry)
                    if PySpin.IsAvailable(e):
                        options.append((e.GetSymbolic(), _pretty(e.GetSymbolic())))
                except Exception:
                    continue
            return CameraFeature(kind=KIND_ENUM, options=tuple(options), **common)
        if iface == PySpin.intfIBoolean:
            return CameraFeature(kind=KIND_BOOL, **common)
        if iface == PySpin.intfICommand:
            return CameraFeature(kind=KIND_COMMAND, **common)
        if iface == PySpin.intfIString:
            return CameraFeature(kind=KIND_STRING, access=ACCESS_RO,
                                 **{k: v for k, v in common.items()
                                    if k != "access"})
    except Exception:
        return None
    return None


def _spinnaker_get(camera, key):
    """Read one node, dispatching on its declared GenICam type.

    Casting speculatively (float, then integer, then bool) and keeping the
    first cast that did not raise is wrong twice over: it depends on the
    vendor's pointer casts failing loudly, and it has no case at all for
    string nodes, which is what ``DeviceModelName`` and
    ``DeviceSerialNumber`` are, so model and serial always read back empty.
    """
    import PySpin
    node_name = _rev_lookup(_SPIN_KEYS, key)
    nodemap = getattr(camera, "_nodemap", None)
    if nodemap is None:
        return None
    node = nodemap.GetNode(node_name)
    if node is None or not PySpin.IsReadable(node):
        return None
    try:
        iface = node.GetPrincipalInterfaceType()
        if iface == PySpin.intfIFloat:
            return PySpin.CFloatPtr(node).GetValue()
        if iface == PySpin.intfIInteger:
            return PySpin.CIntegerPtr(node).GetValue()
        if iface == PySpin.intfIBoolean:
            return PySpin.CBooleanPtr(node).GetValue()
        if iface == PySpin.intfIString:
            return PySpin.CStringPtr(node).GetValue()
        if iface == PySpin.intfIEnumeration:
            return PySpin.CEnumerationPtr(node).GetCurrentEntry().GetSymbolic()
    except Exception as exc:
        logger.debug("Spinnaker read %s failed: %s", node_name, exc)
    return None


def _spinnaker_set(camera, key, value) -> bool:
    import PySpin
    node_name = _rev_lookup(_SPIN_KEYS, key)
    nodemap = getattr(camera, "_nodemap", None)
    if nodemap is None:
        return False
    node = nodemap.GetNode(node_name)
    if node is None or not PySpin.IsWritable(node):
        return False
    iface = node.GetPrincipalInterfaceType()
    if iface == PySpin.intfIFloat:
        n = PySpin.CFloatPtr(node)
        n.SetValue(max(n.GetMin(), min(n.GetMax(), float(value))))
    elif iface == PySpin.intfIInteger:
        n = PySpin.CIntegerPtr(node)
        n.SetValue(int(max(n.GetMin(), min(n.GetMax(), int(value)))))
    elif iface == PySpin.intfIBoolean:
        PySpin.CBooleanPtr(node).SetValue(bool(value))
    elif iface == PySpin.intfIEnumeration:
        n = PySpin.CEnumerationPtr(node)
        entry = n.GetEntryByName(str(value))
        if entry is None:
            return False
        n.SetIntValue(entry.GetValue())
    elif iface == PySpin.intfICommand:
        PySpin.CCommandPtr(node).Execute()
    else:
        return False
    return True


# ---------------------------------------------------------------- Ximea ---

def _ximea_available() -> bool:
    try:
        from .ximea import XIMEA_AVAILABLE
        return bool(XIMEA_AVAILABLE)
    except Exception:
        return False


def _ximea_list():
    from .ximea import list_available_cameras
    return list_available_cameras()


def _ximea_create(unique_id: str, identifier: str, config: dict):
    from .ximea import XimeaCamera
    cam = XimeaCamera(
        unique_id=unique_id,
        fps=config.get("fps", 30),
        exposure_us=config.get("exposure_us", 5000),
        gain_db=config.get("gain_db", 0),
        external_trigger=config.get("external_trigger", False),
        grayscale=config.get("grayscale", False),
        trigger=config.get("trigger"),
        line_output=config.get("line_output"),
    )
    width, height = config.get("width"), config.get("height")
    if width and height:
        try:
            cam.configure(roi=(0, 0, int(width), int(height)))
        except Exception as exc:
            logger.warning("Ximea: could not apply %sx%s: %s", width, height, exc)
    return cam


# (canonical key, xiAPI parameter, kind, group, unit, live)
_XIMEA_PARAMS = (
    ("model", "device_name", KIND_STRING, GROUP_IDENTITY, "", False),
    ("serial", "device_sn", KIND_STRING, GROUP_IDENTITY, "", False),
    ("temperature_c", "temp", KIND_FLOAT, GROUP_IDENTITY, "C", False),
    ("frame_rate", "framerate", KIND_FLOAT, GROUP_ACQUISITION, "fps", True),
    ("exposure_us", "exposure", KIND_INT, GROUP_EXPOSURE, "us", True),
    ("gain_db", "gain", KIND_FLOAT, GROUP_EXPOSURE, "dB", True),
    ("exposure_auto", "aeag", KIND_BOOL, GROUP_EXPOSURE, "", True),
    ("exposure_auto_max", "ae_max_limit", KIND_INT, GROUP_EXPOSURE, "us", True),
    ("gain_auto_max", "ag_max_limit", KIND_FLOAT, GROUP_EXPOSURE, "dB", True),
    ("width", "width", KIND_INT, GROUP_FORMAT, "px", False),
    ("height", "height", KIND_INT, GROUP_FORMAT, "px", False),
    ("offset_x", "offsetX", KIND_INT, GROUP_FORMAT, "px", False),
    ("offset_y", "offsetY", KIND_INT, GROUP_FORMAT, "px", False),
    ("decimation_h", "downsampling", KIND_INT, GROUP_FORMAT, "", False),
    ("gamma", "gammaY", KIND_FLOAT, GROUP_COLOUR, "", True),
    ("sharpness", "sharpness", KIND_FLOAT, GROUP_COLOUR, "", True),
    ("white_balance_auto", "auto_wb", KIND_BOOL, GROUP_COLOUR, "", True),
    ("wb_red", "wb_kr", KIND_FLOAT, GROUP_COLOUR, "", True),
    ("wb_blue", "wb_kb", KIND_FLOAT, GROUP_COLOUR, "", True),
    ("trigger_delay_us", "trigger_delay", KIND_INT, GROUP_TRIGGER, "us", False),
    ("trigger_burst", "acq_frame_burst_count", KIND_INT, GROUP_TRIGGER, "", False),
    ("bandwidth_limit", "limit_bandwidth", KIND_INT, GROUP_TRANSPORT, "Mb/s", False),
    ("buffer_count", "buffers_queue_size", KIND_INT, GROUP_TRANSPORT, "", False),
)

_XIMEA_ENUMS = {
    "image_data_format": ("pixel_format", GROUP_FORMAT,
                          ("XI_MONO8", "XI_MONO16", "XI_RAW8", "XI_RAW16",
                           "XI_RGB24", "XI_RGB32")),
    "trg_source": ("trigger_mode", GROUP_TRIGGER,
                   ("XI_TRG_OFF", "XI_TRG_EDGE_RISING", "XI_TRG_EDGE_FALLING",
                    "XI_TRG_SOFTWARE", "XI_TRG_LEVEL_HIGH", "XI_TRG_LEVEL_LOW")),
    "trg_selector": ("trigger_selector", GROUP_TRIGGER,
                     ("XI_TRG_SEL_FRAME_START", "XI_TRG_SEL_EXPOSURE_ACTIVE",
                      "XI_TRG_SEL_FRAME_BURST_START")),
    "gpo_mode": ("line_source", GROUP_IO,
                 ("XI_GPO_OFF", "XI_GPO_ON", "XI_GPO_FRAME_ACTIVE",
                  "XI_GPO_EXPOSURE_ACTIVE", "XI_GPO_EXPOSURE_PULSE")),
    "gpi_mode": ("line_mode", GROUP_IO,
                 ("XI_GPI_OFF", "XI_GPI_TRIGGER", "XI_GPI_EXT_EVENT")),
    "acq_timing_mode": ("acquisition_mode", GROUP_ACQUISITION,
                        ("XI_ACQ_TIMING_MODE_FREE_RUN",
                         "XI_ACQ_TIMING_MODE_FRAME_RATE",
                         "XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT")),
}


def _ximea_features(camera) -> list[CameraFeature]:
    cam = getattr(camera, "_cam", None)
    if cam is None:
        return []
    feats: list[CameraFeature] = []

    def probe(param, suffix):
        try:
            return float(cam.get_param(f"{param}:{suffix}"))
        except Exception:
            return None

    for key, param, kind, group, unit, live in _XIMEA_PARAMS:
        try:
            cam.get_param(param)
        except Exception:
            continue                                    # model lacks it
        if kind in (KIND_INT, KIND_FLOAT):
            feats.append(CameraFeature(
                key=key, label=_pretty(param), group=group, kind=kind,
                unit=unit, minimum=probe(param, "min"),
                maximum=probe(param, "max"), increment=probe(param, "inc"),
                live=live, sdk_node=f"XI_PRM_{param.upper()}",
                access=ACCESS_RO if kind is KIND_STRING else "rw"))
        elif kind is KIND_BOOL:
            feats.append(CameraFeature(key=key, label=_pretty(param),
                                       group=group, kind=KIND_BOOL, live=live,
                                       sdk_node=f"XI_PRM_{param.upper()}"))
        else:
            feats.append(CameraFeature(key=key, label=_pretty(param),
                                       group=group, kind=KIND_STRING,
                                       access=ACCESS_RO,
                                       sdk_node=f"XI_PRM_{param.upper()}"))

    for param, (key, group, options) in _XIMEA_ENUMS.items():
        try:
            cam.get_param(param)
        except Exception:
            continue
        feats.append(enum_feature(
            key, _pretty(param), [(o, _pretty(o)) for o in options],
            group=group, sdk_node=f"XI_PRM_{param.upper()}"))

    return apply_curated(feats)


_XIMEA_KEY_TO_PARAM = {k: p for k, p, *_ in _XIMEA_PARAMS}
_XIMEA_KEY_TO_PARAM.update({v[0]: p for p, v in _XIMEA_ENUMS.items()})


def _ximea_get(camera, key):
    cam = getattr(camera, "_cam", None)
    param = _XIMEA_KEY_TO_PARAM.get(key)
    if cam is None or param is None:
        return None
    return cam.get_param(param)


def _ximea_set(camera, key, value) -> bool:
    cam = getattr(camera, "_cam", None)
    param = _XIMEA_KEY_TO_PARAM.get(key)
    if cam is None or param is None:
        return False
    cam.set_param(param, value)
    return True


# --------------------------------------------------------------- helpers ---

def _pretty(name: str) -> str:
    """CamelCase / snake_case SDK names into something readable."""
    text = str(name)
    for prefix in ("XI_PRM_", "XI_", "CAP_PROP_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace("_", " ")
    out, prev_lower = [], False
    for ch in text:
        if ch.isupper() and prev_lower:
            out.append(" ")
        out.append(ch)
        prev_lower = ch.islower() or ch.isdigit()
    return "".join(out).strip().capitalize()


def _unit_for(node_name: str) -> str:
    lowered = node_name.lower()
    if "time" in lowered or "delay" in lowered or "duration" in lowered:
        return "us"
    if "gain" in lowered or "level" in lowered:
        return "dB"
    if "rate" in lowered:
        return "fps"
    if "throughput" in lowered:
        return "B/s"
    if "temperature" in lowered:
        return "C"
    return ""


def _rev_lookup(mapping: dict, key: str) -> str:
    for node, canonical in mapping.items():
        if canonical == key:
            return node
    return key


# -------------------------------------------------------------- register ---

def register_all() -> None:
    register_backend(BackendSpec(
        id="opencv", display_name="OpenCV / UVC", module_name="cv2",
        is_available=_opencv_available, list_cameras=_opencv_list,
        create_camera=_opencv_create, describe_features=_opencv_features))

    register_backend(BackendSpec(
        id="spinnaker", display_name="FLIR Spinnaker", module_name="PySpin",
        is_available=_spinnaker_available, list_cameras=_spinnaker_list,
        create_camera=_spinnaker_create, describe_features=_spinnaker_features,
        get_feature=_spinnaker_get, set_feature=_spinnaker_set))

    register_backend(BackendSpec(
        id="ximea", display_name="Ximea xiAPI", module_name="ximea.xiapi",
        is_available=_ximea_available, list_cameras=_ximea_list,
        create_camera=_ximea_create, describe_features=_ximea_features,
        get_feature=_ximea_get, set_feature=_ximea_set))
