"""Camera feature descriptors, the contract that lets the UI render itself.

A backend describes what its camera can do; the setup dialog renders whatever
it is handed. Nothing about a camera's option set is hardcoded in the GUI, so

  * a webcam shows three rows and a FLIR shows forty, without UI changes,
  * ranges come from the sensor instead of being guessed,
  * a new SDK gets a working panel by implementing ``describe_features``.

Two tiers share one renderer. ``TIER_CURATED`` is the short, ordered,
human-labelled set shown by default; ``TIER_FULL`` is the vendor's complete
node map, shown in the searchable "All features" tree.

``CURATED_ORDER`` is an *overlay*: it supplies preferred grouping, labels and
ordering for keys a backend happens to report. It never asserts that a camera
has a feature, a model that lacks one simply does not render that row.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# ---------------------------------------------------------------- constants

KIND_FLOAT = "float"
KIND_INT = "int"
KIND_ENUM = "enum"
KIND_BOOL = "bool"
KIND_COMMAND = "command"
KIND_STRING = "string"

ACCESS_RW = "rw"
ACCESS_RO = "ro"
ACCESS_WO = "wo"

TIER_CURATED = "curated"
TIER_FULL = "full"

# Group names, in the order the curated panel renders them. A group with no
# surviving features is not drawn.
GROUP_IDENTITY = "Identity & health"
GROUP_ACQUISITION = "Acquisition"
GROUP_EXPOSURE = "Exposure & gain"
GROUP_FORMAT = "Image format"
GROUP_COLOUR = "Colour"
GROUP_TRIGGER = "Trigger"
GROUP_IO = "Digital I/O & strobe"
GROUP_TRANSPORT = "Transport"
GROUP_DIAGNOSTICS = "Diagnostics"
GROUP_OTHER = "Other"

GROUP_ORDER = (
    GROUP_IDENTITY,
    GROUP_ACQUISITION,
    GROUP_EXPOSURE,
    GROUP_FORMAT,
    GROUP_COLOUR,
    GROUP_TRIGGER,
    GROUP_IO,
    GROUP_TRANSPORT,
    GROUP_DIAGNOSTICS,
    GROUP_OTHER,
)


@dataclass(frozen=True)
class CameraFeature:
    """One settable or readable property of a camera.

    ``live`` is the field the UI cares about most: True means the value can be
    pushed to a streaming camera, False means it is queued until the next
    connect. Backends must be honest here, a control that silently does
    nothing is worse than one that says "needs reconnect".
    """

    key: str
    label: str
    group: str = GROUP_OTHER
    kind: str = KIND_FLOAT
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    increment: float | None = None
    options: tuple[tuple[Any, str], ...] = ()
    access: str = ACCESS_RW
    live: bool = False
    tier: str = TIER_CURATED
    depends_on: tuple[tuple[str, str, Any], ...] = ()
    sdk_node: str = ""
    help: str = ""
    category: str = ""          # vendor's own category, for the full tree

    @property
    def writable(self) -> bool:
        return self.access in (ACCESS_RW, ACCESS_WO) and self.kind != KIND_COMMAND

    def clamp(self, value):
        """Constrain ``value`` to the feature's range and increment.

        Returns the value unchanged for kinds where clamping is meaningless.
        """
        if self.kind not in (KIND_FLOAT, KIND_INT):
            return value
        try:
            v = float(value)
        except (TypeError, ValueError):
            return value
        if self.minimum is not None:
            v = max(v, float(self.minimum))
        if self.maximum is not None:
            v = min(v, float(self.maximum))
        inc = self.increment
        if inc:
            base = float(self.minimum) if self.minimum is not None else 0.0
            steps = round((v - base) / float(inc))
            v = base + steps * float(inc)
            # Stepping can push us back outside the range at the top end.
            if self.maximum is not None and v > float(self.maximum):
                v -= float(inc)
            if self.minimum is not None and v < float(self.minimum):
                v = float(self.minimum)
        return round(v) if self.kind == KIND_INT else v

    def is_enabled_by(self, values: dict) -> bool:
        """True when every ``depends_on`` condition holds in ``values``."""
        for dep_key, op, expected in self.depends_on:
            actual = values.get(dep_key)
            if op == "==" and actual != expected:
                return False
            if op == "!=" and actual == expected:
                return False
            if op == "in" and actual not in expected:
                return False
            if op == "truthy" and not actual:
                return False
        return True


@dataclass(frozen=True)
class CuratedEntry:
    """Preferred presentation for a feature key, applied when present."""

    group: str
    label: str
    order: int
    help: str = ""


def _curated(rows) -> dict[str, CuratedEntry]:
    out: dict[str, CuratedEntry] = {}
    for i, (key, group, label, *rest) in enumerate(rows):
        out[key] = CuratedEntry(group=group, label=label, order=i,
                                help=rest[0] if rest else "")
    return out


# Backend-agnostic keys. A backend maps its own node onto one of these where
# the meaning matches, so the curated panel looks the same across vendors.
CURATED_ORDER = _curated([
    ("model",              GROUP_IDENTITY,   "Model"),
    ("serial",             GROUP_IDENTITY,   "Serial"),
    ("firmware",           GROUP_IDENTITY,   "Firmware"),
    ("sdk_version",        GROUP_IDENTITY,   "SDK version"),
    ("temperature_c",      GROUP_IDENTITY,   "Sensor temperature"),
    ("link_throughput",    GROUP_IDENTITY,   "Link throughput"),

    ("acquisition_mode",   GROUP_ACQUISITION, "Acquisition mode"),
    ("frame_rate_enable",  GROUP_ACQUISITION, "Limit frame rate"),
    ("frame_rate",         GROUP_ACQUISITION, "Frame rate"),
    ("resulting_frame_rate", GROUP_ACQUISITION, "Resulting rate",
     "What the camera can actually deliver given exposure and bandwidth."),

    ("exposure_auto",      GROUP_EXPOSURE,   "Exposure auto"),
    ("exposure_us",        GROUP_EXPOSURE,   "Exposure"),
    ("exposure_mode",      GROUP_EXPOSURE,   "Exposure mode"),
    ("exposure_auto_min",  GROUP_EXPOSURE,   "Auto exposure min"),
    ("exposure_auto_max",  GROUP_EXPOSURE,   "Auto exposure max"),
    ("auto_target_grey",   GROUP_EXPOSURE,   "Auto target grey"),
    ("gain_auto",          GROUP_EXPOSURE,   "Gain auto"),
    ("gain_db",            GROUP_EXPOSURE,   "Gain"),
    ("gain_auto_min",      GROUP_EXPOSURE,   "Auto gain min"),
    ("gain_auto_max",      GROUP_EXPOSURE,   "Auto gain max"),
    ("black_level",        GROUP_EXPOSURE,   "Black level"),

    ("pixel_format",       GROUP_FORMAT,     "Pixel format"),
    ("adc_bit_depth",      GROUP_FORMAT,     "ADC bit depth"),
    ("width",              GROUP_FORMAT,     "Width"),
    ("height",             GROUP_FORMAT,     "Height"),
    ("offset_x",           GROUP_FORMAT,     "Offset X"),
    ("offset_y",           GROUP_FORMAT,     "Offset Y"),
    ("binning_h",          GROUP_FORMAT,     "Binning horizontal"),
    ("binning_v",          GROUP_FORMAT,     "Binning vertical"),
    ("binning_mode",       GROUP_FORMAT,     "Binning mode"),
    ("decimation_h",       GROUP_FORMAT,     "Decimation horizontal"),
    ("decimation_v",       GROUP_FORMAT,     "Decimation vertical"),
    ("reverse_x",          GROUP_FORMAT,     "Flip horizontally"),
    ("reverse_y",          GROUP_FORMAT,     "Flip vertically"),
    ("test_pattern",       GROUP_FORMAT,     "Test pattern",
     "Generates a synthetic image so the pipeline can be checked without a scene."),
    ("capture_format",     GROUP_FORMAT,     "Transport format"),

    ("white_balance_auto", GROUP_COLOUR,     "White balance auto"),
    ("wb_red",             GROUP_COLOUR,     "White balance red"),
    ("wb_blue",            GROUP_COLOUR,     "White balance blue"),
    ("gamma_enable",       GROUP_COLOUR,     "Gamma enable"),
    ("gamma",              GROUP_COLOUR,     "Gamma"),
    ("sharpness",          GROUP_COLOUR,     "Sharpness"),
    ("saturation",         GROUP_COLOUR,     "Saturation"),
    ("colour_filter",      GROUP_COLOUR,     "Colour filter array"),

    ("trigger_selector",   GROUP_TRIGGER,    "Trigger selector"),
    ("trigger_mode",       GROUP_TRIGGER,    "Trigger"),
    ("trigger_source",     GROUP_TRIGGER,    "Source line"),
    ("trigger_activation", GROUP_TRIGGER,    "Edge"),
    ("trigger_delay_us",   GROUP_TRIGGER,    "Trigger delay"),
    ("trigger_overlap",    GROUP_TRIGGER,    "Trigger overlap"),
    ("trigger_burst",      GROUP_TRIGGER,    "Frames per trigger"),
    ("trigger_software",   GROUP_TRIGGER,    "Fire one frame"),

    ("line_selector",      GROUP_IO,         "Line"),
    ("line_mode",          GROUP_IO,         "Line mode"),
    ("line_source",        GROUP_IO,         "Line source"),
    ("line_inverter",      GROUP_IO,         "Invert polarity"),
    ("line_debounce_us",   GROUP_IO,         "Debounce"),
    ("line_status",        GROUP_IO,         "Line status"),
    ("supply_3v3",         GROUP_IO,         "3.3 V supply"),
    ("strobe_width_us",    GROUP_IO,         "Strobe pulse width",
     "Pulse width independent of exposure, via the camera's counter/timer."),
    ("indicator_led",      GROUP_IO,         "Status LED"),

    ("throughput_limit",   GROUP_TRANSPORT,  "Throughput limit"),
    ("buffer_mode",        GROUP_TRANSPORT,  "Buffer handling"),
    ("buffer_count",       GROUP_TRANSPORT,  "Buffer count"),
    ("packet_size",        GROUP_TRANSPORT,  "Packet size"),
    ("packet_delay",       GROUP_TRANSPORT,  "Packet delay"),
    ("bandwidth_limit",    GROUP_TRANSPORT,  "Bandwidth limit"),
    ("bandwidth_auto",     GROUP_TRANSPORT,  "Auto bandwidth"),

    ("dropped_frames",     GROUP_DIAGNOSTICS, "Dropped frames"),
    ("incomplete_frames",  GROUP_DIAGNOSTICS, "Incomplete frames"),
    ("timestamp_source",   GROUP_DIAGNOSTICS, "Timestamp source"),
    ("chunk_enable",       GROUP_DIAGNOSTICS, "Chunk data"),
    ("user_set",           GROUP_DIAGNOSTICS, "User set"),
])


def apply_curated(features) -> list[CameraFeature]:
    """Overlay preferred group/label/order onto whatever a backend reported.

    Features with a curated entry are returned first, in curated order.
    Anything else keeps its own group and is appended, so a vendor-specific
    feature is never dropped just because this module has not heard of it.
    """
    known, unknown = [], []
    for f in features:
        entry = CURATED_ORDER.get(f.key)
        if entry is None:
            unknown.append(f)
            continue
        known.append((entry.order, replace(
            f,
            group=f.group if f.group not in ("", GROUP_OTHER) else entry.group,
            label=f.label or entry.label,
            help=f.help or entry.help,
        )))
    known.sort(key=lambda t: t[0])
    return [f for _, f in known] + unknown


def group_features(features) -> list[tuple[str, list[CameraFeature]]]:
    """Bucket features into display groups, in GROUP_ORDER, dropping empties."""
    buckets: dict[str, list[CameraFeature]] = {}
    for f in features:
        buckets.setdefault(f.group or GROUP_OTHER, []).append(f)
    ordered = [(g, buckets.pop(g)) for g in GROUP_ORDER if g in buckets]
    # Vendor groups the overlay does not know about, alphabetically after.
    ordered.extend(sorted(buckets.items()))
    return ordered


# ------------------------------------------------------------------ helpers
# Small constructors so backends stay readable.


def enum_feature(key, label, options, *, group=GROUP_OTHER, live=False,
                 sdk_node="", tier=TIER_CURATED, access=ACCESS_RW,
                 depends_on=(), help="", category="") -> CameraFeature:
    return CameraFeature(key=key, label=label, group=group, kind=KIND_ENUM,
                         options=tuple(options), live=live, sdk_node=sdk_node,
                         tier=tier, access=access, depends_on=tuple(depends_on),
                         help=help, category=category)


def number_feature(key, label, *, group=GROUP_OTHER, unit="", minimum=None,
                   maximum=None, increment=None, kind=KIND_FLOAT, live=False,
                   sdk_node="", tier=TIER_CURATED, access=ACCESS_RW,
                   depends_on=(), help="", category="") -> CameraFeature:
    return CameraFeature(key=key, label=label, group=group, kind=kind,
                         unit=unit, minimum=minimum, maximum=maximum,
                         increment=increment, live=live, sdk_node=sdk_node,
                         tier=tier, access=access, depends_on=tuple(depends_on),
                         help=help, category=category)


