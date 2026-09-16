"""The ONE definition of ``*_video_data.txt``.

Three readers and three writers of the session file, none of them agreeing on
the column names, do not produce crashes, they produce silent wrong numbers.
An offline retrack writing ``actual_ts`` where the analysis reader looks for
``frame_ts_ms`` leaves the reader on its row counter, so every timestamp
becomes "the row number, in milliseconds", which then trips the teleport clamp
and collapses distance to zero with no error anywhere.

So: one column contract, one reader, one row formatter, and every legacy
dialect named and adapted explicitly rather than handled by accident.

The file is a TSV with two interleaved kinds of line:

* **header / event lines**: ``<seconds>\\t<row_type>\\t<subtype>\\t<content>``.
  Header info lines are the special case ``0.000\\tinfo\\t<key>\\t<value>``;
  everything else (``event``, ``state``, ``warning``, …) is a timestamped
  session event. Plus the standalone ``Zones\\t{json}`` line.
* **data rows**: one per acquired frame, positional, described by the
  ``# frame_number\\t…`` comment line that precedes them.

The data row is the alignment surface between the video timeline, the MCU
timeline and the event rows, so it stays 1:1 with acquired frames even for
multi-animal sessions (identities are packed into the cells, not into extra
rows).

This module has no Qt, no numpy and no pandas: it is imported by the recorder
(live, on the recording thread) and by the headless analysis engine alike.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── the contract ─────────────────────────────────────────────────────────────

#: Schema version written by the current recorder. v3 = base columns only,
#: v4 = base columns plus the declared metric columns.
SCHEMA_VERSION = 4

#: The positional per-frame columns, in order. Metric columns declared by
#: ``core.metrics_registry`` are APPENDED after ``stage``: never inserted,
#: so a reader that indexes by position still finds every column it knows.
BASE_COLUMNS: Tuple[str, ...] = (
    "frame_number",   # monotonic acquired-frame counter (NOT the CFR video slot)
    "frame_ts_ms",    # ms the frame was acquired, relative to stage start
    "pose_ts_ms",     # ms the pose's frame was captured, or NA
    "mcu_ts_ms",      # MCU/hardware event time on the same clock, or NA
    "speed",          # cm/s when calibrated (px_per_cm > 0), else px/s
    "location",       # zone name, or `none`
    "pose_array",     # JSON pose payload, or `none`
    "stage",          # live protocol stage name
)

#: Legacy column names → canonical ones. ``actual_ts`` was the acquisition
#: time; ``desired_ts`` was the CFR grid slot and is kept as an extra rather
#: than confused with it.
COLUMN_ALIASES: Dict[str, str] = {
    "actual_ts": "frame_ts_ms",
    "timestamp_ms": "frame_ts_ms",
    "frame_ts": "frame_ts_ms",
    "pose_ts": "pose_ts_ms",
    "zone": "location",
    "pose": "pose_array",
    "state": "stage",
}

#: Cells that mean "no value". ``na`` (tracker off) and ``none`` (tracker on,
#: nothing found) are DIFFERENT facts upstream, and :func:`iter_rows` keeps
#: them distinguishable via ``row["_tracker_off"]``; both read as absent here.
NA_VALUES = frozenset({"", "na", "n/a", "none", "null", "nan"})

#: The sentinel the recorder writes when the tracker is off, as opposed to
#: `none` which means "tracker on, nothing detected this frame".
TRACKER_OFF = "na"

#: Row types that mark a header/event line rather than a per-frame data row.
EVENT_ROW_TYPES = frozenset({
    "info", "event", "state", "print", "variable", "warning", "error",
    "trigger", "marker",
})

#: Dialects this module can read.
DIALECT_V4 = "v4"                 # current recorder, named column header
DIALECT_LEGACY_COLUMNS = "legacy_columns"   # `desired_ts`/`actual_ts` header
DIALECT_LEGACY_BMV = "legacy_bmv"           # `B {...}` / `M {...}` / `V {...}`
DIALECT_JSON_ROWS = "json_rows"             # pymazeRetrack: one JSON per line
#: The sibling port (``D:\\Ulm_Data\\pyBehaviorLab\\code``): ``#key   value``
#: header lines, a ``#columns`` line of SPACE-separated names, and a
#: multi-line JSON block between ``#zone_config_begin``/``_end``.
DIALECT_HASH_HEADER = "hash_header"
DIALECT_UNKNOWN = "unknown"

#: Sibling column name → ours. ``frame_fw_ms`` is the MCU firmware clock, which
#: is what ``mcu_ts_ms`` means here; ``elapsed`` (``MM:SS.mmm``) is the host
#: acquisition time and therefore the real ``frame_ts_ms``.
HASH_HEADER_COLUMNS: Dict[str, str] = {
    "frame": "frame_number",
    "elapsed": "frame_ts_ms",
    "frame_fw_ms": "mcu_ts_ms",
    "pose_fw_ms": "pose_ts_ms",
    "pose": "pose_array",
    "zone": "location",
    "state": "stage",
    # ── this rig's FIRST recorder, which most existing data is in ──
    #
    # Same header shape, different column names. None of them were mapped, so
    # every one of those recordings was read as having no timestamp column at
    # all: the clock fell back to the video, and a recording whose video had
    # been archived was blocked outright. `capture_elapsed` carries the same
    # MM:SS.mmm the current `elapsed` does.
    "frame_num": "frame_number",
    "capture_elapsed": "frame_ts_ms",
    "mcu_fw_capture_ms": "mcu_ts_ms",
    "mcu_fw_pose_ms": "pose_ts_ms",
    "speed_pxs": "speed",
    # ── and the recorder in between, which had ONE firmware clock ──
    #
    # Surveyed across 997 recordings: 529 use the `mcu_fw_capture_ms` spelling
    # above, 172 use `frame_fw_ms`, and 296 use plain `fw_ms`, which was
    # mapped by nothing. Those 296 carry the MCU clock on 99.98% of their rows
    # and were read as having none, so every one of them fell back to
    # aligning the task state by elapsed time.
    "fw_ms": "mcu_ts_ms",
}


#: The per-frame time column, 2D first then 3D. A 3D reconstruction writes
#: ``t_ms`` (ms since stage start, floor frame) where a 2D recording writes
#: ``frame_ts_ms``. Code that looks only for the 2D name concludes a perfectly
#: good 3D session has no timestamps at all.
TIME_COLUMNS: Tuple[str, ...] = ("frame_ts_ms", "t_ms")

#: The per-frame pose column, likewise. A 3D value is
#: ``[x, y, z, conf, reproj_px, n_views(, var_mm2)]`` in floor mm, so the two
#: are read differently, but "does this row carry a pose" is one question.
POSE_COLUMNS: Tuple[str, ...] = ("pose_array", "pose3d")


def timestamp_column(header: "VideoDataHeader") -> str:
    """Which column holds this file's per-frame time, or ``""``."""
    idx = header.index
    return next((c for c in TIME_COLUMNS if c in idx), "")


def pose_column(header: "VideoDataHeader") -> str:
    """Which column holds this file's per-frame pose, or ``""``."""
    idx = header.index
    return next((c for c in POSE_COLUMNS if c in idx), "")


def column_header_line(metric_columns: Sequence[str] = (),
                       extra_columns: Sequence[str] = ()) -> str:
    """The ``# frame_number…`` line, terminated by a newline.

    The recorder and every offline writer call this, which is what makes it
    impossible to emit a layout the reader does not know.
    """
    cols = list(BASE_COLUMNS) + list(metric_columns) + list(extra_columns)
    return "# " + "\t".join(cols) + "\n"


def info_line(key: str, value: Any) -> str:
    """A header info line, terminated by a newline."""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, separators=(",", ":"))
    return f"0.000\tinfo\t{key}\t{value}\n"


def zones_line(zones: Any) -> str:
    """The ``Zones<TAB>{json}`` line, terminated by a newline."""
    if isinstance(zones, str):
        return f"Zones\t{zones}\n"
    return f"Zones\t{json.dumps(zones, separators=(',', ':'))}\n"


# ── header ───────────────────────────────────────────────────────────────────

#: The order the recorder writes its info lines in. Rendering follows it so a
#: file written offline is diff-comparable with one written live.
_INFO_ORDER: Tuple[str, ...] = (
    "video_file", "subject_id", "box_id", "stage", "start_time", "units",
    "frame_clock", "target_fps", "resolution", "pose_resolution",
    "codec", "encoder", "metadata",
    # Zones line is emitted here, between `metadata` and `body_parts`.
    "body_parts", "tracker", "schema", "metric_columns", "metric_config",
)

_JSON_INFO_KEYS = frozenset({
    "units", "metadata", "body_parts", "tracker", "schema", "metric_columns",
    "metric_config",
})


@dataclass
class VideoDataHeader:
    """Everything above the first data row, parsed once.

    ``info`` holds the raw string values exactly as written; the typed
    properties below are the supported way to read them, so a caller never
    has to know whether a field was JSON-encoded.
    """
    info: Dict[str, str] = field(default_factory=dict)
    zones_raw: str = ""
    columns: Tuple[str, ...] = ()
    dialect: str = DIALECT_UNKNOWN
    path: str = ""
    #: Column names as they appeared in the file, before alias mapping.
    raw_columns: Tuple[str, ...] = ()

    # ── typed accessors ──────────────────────────────────────────────

    def _json(self, key: str, default):
        raw = self.info.get(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default

    @property
    def subject_id(self) -> str:
        sid = self.info.get("subject_id", "") or self.info.get("subject", "")
        if not sid:
            # The sibling port carries it inside its `pycontrol` JSON block.
            sid = str((self._json("pycontrol", {}) or {}).get("subject", ""))
        return sid

    @property
    def box_id(self) -> str:
        return self.info.get("box_id", "")

    @property
    def stage(self) -> str:
        return self.info.get("stage", "")

    @property
    def start_time(self) -> str:
        return self.info.get("start_time", "")

    @property
    def units(self) -> dict:
        return self._json("units", {}) or {}

    @property
    def px_per_cm(self) -> float:
        """Pixels per centimetre, or 0.0 when the session is uncalibrated.

        0 is not a fallback to guess around: it means every distance is in
        pixels and must be labelled as such.
        """
        try:
            return float(self.units.get("px_per_cm", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def frame_clock(self) -> str:
        """Where the frame timestamps came from, a property of the camera and
        backend for the whole session, so it is a header key, not a column."""
        return self.info.get("frame_clock", "")

    @property
    def target_fps(self) -> float:
        raw = self.info.get("target_fps", 0)
        if not raw:
            raw = (self._json("camera", {}) or {}).get("fps", 0)
        try:
            return float(raw or 0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def pose_resolution(self) -> Tuple[int, int]:
        """The space the POSE coordinates are in.

        Not the same as :attr:`resolution`, which describes the .mp4. Frames
        are tracked at the ROI's own size and the video is rescaled
        afterwards, so normalized zones must be multiplied by THIS to land
        where the poses are. ``(0, 0)`` for recordings written before the
        recorder declared it, the caller falls back to `resolution` and
        cannot tell whether that is right.
        """
        return self._wxh("pose_resolution")

    def _wxh(self, key: str) -> Tuple[int, int]:
        raw = str(self.info.get(key, "") or "")
        if "x" in raw.lower():
            try:
                w, h = raw.lower().split("x")
                return int(w), int(h)
            except (TypeError, ValueError):
                pass
        return (0, 0)

    @property
    def resolution(self) -> Tuple[int, int]:
        raw = self.info.get("resolution", "")
        if not raw:
            # The sibling port carries it inside its `camera` JSON block.
            raw = str((self._json("camera", {}) or {}).get("resolution", ""))
        if "x" in raw.lower():
            try:
                w, h = raw.lower().split("x")
                return int(w), int(h)
            except (TypeError, ValueError):
                pass
        return (0, 0)

    @property
    def body_parts(self) -> List[str]:
        bp = self._json("body_parts", None)
        if isinstance(bp, list):
            return [str(b) for b in bp]
        raw = self.info.get("body_parts", "")
        if raw and not raw.startswith("["):
            return [p.strip() for p in raw.split(",") if p.strip()]
        # The rig writes them inside its self-describing tracker block rather
        # than on a line of their own. Same fact, other dialect.
        named = self.tracker.get("bodyparts")
        if isinstance(named, list):
            return [str(b) for b in named]
        return []

    @property
    def skeleton(self) -> List[tuple]:
        """Connected pairs of part names, as the recording declares them.

        Recorded by the rig from what the model itself states (sleap-nn keeps
        the skeleton in ``training_config.yaml``) or from what the operator
        drew. Empty means the recording declares none, which is a fact about
        the model, not a gap to be filled by guessing an anatomy.
        """
        raw = self.tracker.get("skeleton")
        if not isinstance(raw, list):
            return []
        known = set(self.body_parts)
        out = []
        for edge in raw:
            if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                continue
            a, b = str(edge[0]), str(edge[1])
            if not known or (a in known and b in known):
                out.append((a, b))
        return out

    @property
    def tracker(self) -> dict:
        return self._json("tracker", {}) or {}

    @property
    def schema_version(self) -> int:
        s = self._json("schema", {}) or {}
        try:
            return int(s.get("version", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def metric_columns(self) -> List[dict]:
        mc = self._json("metric_columns", None)
        return mc if isinstance(mc, list) else []

    @property
    def metric_config(self) -> dict:
        return self._json("metric_config", {}) or {}

    @property
    def metadata(self) -> dict:
        return self._json("metadata", {}) or {}

    @property
    def zones(self):
        """The Zones payload as written, a list of zone dicts, or a name→dict
        mapping. Callers that want one shape should use :func:`zones_as_list`."""
        if not self.zones_raw:
            return None
        try:
            return json.loads(self.zones_raw)
        except (TypeError, ValueError):
            return None

    # ── derived ──────────────────────────────────────────────────────

    @property
    def index(self) -> Dict[str, int]:
        """Canonical column name → position."""
        return {name: i for i, name in enumerate(self.columns)}

    @property
    def is_3d(self) -> bool:
        return "pose3d" in self.columns or "t_ms" in self.columns

    def with_tracker(self, tracker: dict, body_parts: Sequence[str]) -> "VideoDataHeader":
        """A copy describing the SAME session retracked by another backend.

        The carry-over rule, in code: an offline retrack replaces exactly three
        fields and copies everything else verbatim. ``units`` (and with it
        ``px_per_cm``), ``start_time``, ``metadata``, ``metric_config`` and the
        zones all survive, losing them is what turned a calibrated session
        into an uncalibrated one on every retrack.
        """
        info = dict(self.info)
        info["tracker"] = json.dumps(tracker, separators=(",", ":"))
        info["body_parts"] = json.dumps(list(body_parts), separators=(",", ":"))
        info["schema"] = json.dumps({"version": SCHEMA_VERSION},
                                    separators=(",", ":"))
        return VideoDataHeader(info=info, zones_raw=self.zones_raw,
                               columns=tuple(BASE_COLUMNS), dialect=DIALECT_V4,
                               path=self.path, raw_columns=tuple(BASE_COLUMNS))

    # ── rendering ────────────────────────────────────────────────────

    def render(self) -> List[str]:
        """The header lines, in the recorder's order, each newline-terminated."""
        out: List[str] = []
        for key in _INFO_ORDER:
            if key == "body_parts" and self.zones_raw:
                out.append(zones_line(self.zones_raw))
            val = self.info.get(key)
            if val is None or val == "":
                continue
            out.append(f"0.000\tinfo\t{key}\t{val}\n")
        if self.zones_raw and not any(ln.startswith("Zones\t") for ln in out):
            out.append(zones_line(self.zones_raw))
        # Anything the writer set that this module does not know about still
        # travels, dropping unknown fields is how files lose their history.
        for key, val in self.info.items():
            if key in _INFO_ORDER or val in (None, ""):
                continue
            out.append(f"0.000\tinfo\t{key}\t{val}\n")
        out.append("# " + "\t".join(self.columns or BASE_COLUMNS) + "\n")
        return out

    @classmethod
    def parse(cls, path: str) -> "VideoDataHeader":
        """Read only the header of ``path``, stops at the first data row."""
        h = cls(path=path)
        info: Dict[str, str] = {}
        raw_cols: Tuple[str, ...] = ()
        dialect = DIALECT_UNKNOWN
        zone_block: Optional[List[str]] = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.rstrip("\n\r")
                    if not s.strip():
                        continue
                    # A multi-line JSON zone block. Collected verbatim; without
                    # this its opening `{` looks like a JSON data row and the
                    # whole file is mistaken for that dialect.
                    if zone_block is not None:
                        if s.startswith("#zone_config_end"):
                            h.zones_raw = _compact_json("".join(zone_block))
                            zone_block = None
                        else:
                            zone_block.append(s)
                        continue
                    if s.startswith("#zone_config_begin"):
                        zone_block = []
                        dialect = DIALECT_HASH_HEADER
                        continue
                    if s.startswith("Zones3D\t"):
                        h.zones_raw = s.split("\t", 1)[1]
                        continue
                    if s.startswith("Zones\t") or s.startswith("Zones "):
                        h.zones_raw = (s.split("\t", 1)[1] if "\t" in s
                                       else s[len("Zones "):])
                        continue
                    if s.startswith("#"):
                        cols = _parse_column_comment(s)
                        if cols:
                            raw_cols = cols
                            dialect = (DIALECT_V4 if "frame_ts_ms" in cols
                                       else DIALECT_LEGACY_COLUMNS)
                            continue
                        key, val = _parse_hash_header(s)
                        if key == "columns":
                            raw_cols = tuple(val.split())
                            dialect = DIALECT_HASH_HEADER
                        elif key:
                            info.setdefault(_HASH_INFO_KEYS.get(key, key), val)
                            if key in _HASH_INFO_KEYS or key in (
                                    "version", "session_id", "sync"):
                                dialect = DIALECT_HASH_HEADER
                        continue
                    if s.startswith(("B ", "M ", "V ")):
                        _absorb_bmv(s, info, h)
                        dialect = DIALECT_LEGACY_BMV
                        continue
                    if s.startswith("body_parts:"):
                        info["body_parts"] = json.dumps(
                            [p.strip() for p in s.split(":", 1)[1].split(",")
                             if p.strip()], separators=(",", ":"))
                        continue
                    parts = s.split("\t")
                    if len(parts) >= 4 and parts[1] == "info":
                        info[parts[2]] = parts[3]
                        continue
                    if parts and parts[0].strip().lower() == "time":
                        continue          # the `time type subtype content` legend
                    if s.lstrip().startswith("{"):
                        dialect = DIALECT_JSON_ROWS
                        break             # first JSON data row
                    if _looks_like_data_row(parts):
                        break             # first columnar data row
        except OSError:
            pass
        h.info = info
        h.raw_columns = raw_cols
        h.columns = _canonical_columns(raw_cols, dialect)
        h.dialect = dialect
        return h


def _parse_column_comment(line: str) -> Tuple[str, ...]:
    """``# a\\tb\\tc`` → ``("a", "b", "c")``; a prose comment → ``()``."""
    body = line.lstrip("#").strip()
    if "\t" not in body:
        return ()
    cols = tuple(c.strip() for c in body.split("\t") if c.strip())
    return cols if len(cols) >= 3 else ()


#: Sibling header keys → the names this module's accessors read.
_HASH_INFO_KEYS = {
    "session_start": "start_time",
    "box_id": "box_id",
    "video_file": "video_file",
    "units": "units",
    "tracker": "tracker",
    "camera": "camera",
    "pycontrol": "pycontrol",
    "video_codec": "video_codec",
}


def _parse_hash_header(line: str) -> Tuple[str, str]:
    """``#key   value`` → ``("key", "value")``; a rule or prose → ``("", "")``.

    The separator is runs of whitespace, and the value may itself be JSON with
    spaces in it, so the split is on the FIRST run only.
    """
    # A header key is flush against the hash (`#version`); a prose comment has
    # a space after it (`# pyBehaviorLab - video …`). Without that distinction
    # the banner line becomes a header key called "pyBehaviorLab".
    if not line.startswith("#") or line[1:2] in ("", " ", "\t", "#", "="):
        return ("", "")
    body = line.lstrip("#").rstrip()
    if not body:
        return ("", "")
    parts = body.split(None, 1)
    if len(parts) < 2:
        return ("", "")
    key = parts[0].strip()
    if not key or not key.replace("_", "").isalnum():
        return ("", "")
    return (key, parts[1].strip())


def _compact_json(text: str) -> str:
    """Pretty-printed JSON → one line, or "" when it will not parse."""
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except (TypeError, ValueError):
        return ""


def parse_elapsed(cell: Optional[str]) -> float:
    """``MM:SS.mmm`` (or ``HH:MM:SS.mmm``) → milliseconds.

    The sibling port writes its acquisition time this way. Read as a plain
    float it would be 0.0 for every frame of the first minute, a whole
    session that looks like it happened instantaneously.
    """
    if is_na(cell):
        return math.nan
    s = str(cell).strip()
    if ":" not in s:
        return parse_float(s)
    parts = s.split(":")
    try:
        seconds = float(parts[-1])
        for i, p in enumerate(reversed(parts[:-1]), start=1):
            seconds += float(p) * (60 ** i)
    except (TypeError, ValueError):
        return math.nan
    return seconds * 1000.0


def _canonical_columns(raw: Sequence[str], dialect: str) -> Tuple[str, ...]:
    if not raw:
        return tuple(BASE_COLUMNS) if dialect != DIALECT_JSON_ROWS else ()
    if dialect == DIALECT_HASH_HEADER:
        return tuple(HASH_HEADER_COLUMNS.get(c, c) for c in raw)
    return tuple(COLUMN_ALIASES.get(c, c) for c in raw)


def _absorb_bmv(line: str, info: Dict[str, str], h: "VideoDataHeader") -> None:
    """Legacy single-letter header lines: ``B``=session, ``M``=meta/zones,
    ``V``=crop box. Ignored entirely by the old analysis parser, which is why
    a raw-video "stub" written in this dialect analysed to nothing."""
    tag, _, payload = line.partition(" ")
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return
    if not isinstance(data, dict):
        return
    if tag == "M":
        zones = data.get("zones", data)
        if zones:
            h.zones_raw = json.dumps(zones, separators=(",", ":"))
        for k, v in data.items():
            if k != "zones":
                info.setdefault(k, v if isinstance(v, str) else
                                json.dumps(v, separators=(",", ":")))
    elif tag == "V":
        info.setdefault("crop", json.dumps(data, separators=(",", ":")))
        w, hgt = data.get("width"), data.get("height")
        if w and hgt:
            info.setdefault("resolution", f"{int(w)}x{int(hgt)}")
    else:                                   # B, session header
        for k, v in data.items():
            info.setdefault(k, v if isinstance(v, str) else
                            json.dumps(v, separators=(",", ":")))
        if "fps" in data:
            info.setdefault("target_fps", str(data["fps"]))


def _looks_like_data_row(parts: Sequence[str]) -> bool:
    if len(parts) < 2:
        return False
    head = parts[0].strip()
    if not head or not head.lstrip("-").isdigit():
        return False
    # An event row's second cell is a row-type word; a data row's is a number.
    return parts[1].strip().lower() not in EVENT_ROW_TYPES


# ── values ───────────────────────────────────────────────────────────────────

def is_na(cell: Optional[str]) -> bool:
    return cell is None or str(cell).strip().lower() in NA_VALUES


def parse_float(cell: Optional[str], default: float = math.nan) -> float:
    if is_na(cell):
        return default
    try:
        return float(str(cell).strip())
    except (TypeError, ValueError):
        return default


def parse_int(cell: Optional[str], default: int = -1) -> int:
    if is_na(cell):
        return default
    try:
        return int(float(str(cell).strip()))
    except (TypeError, ValueError):
        return default


def parse_pose(cell: Optional[str],
               names: Sequence[str] = ()) -> Optional[dict]:
    """The ``pose_array`` cell → a ``{body_part: [x, y, conf]}`` dict.

    Returns ``None`` for ``na``/``none``, absent, which is not the same as
    an empty pose dict (a frame where the detector ran and found nothing).

    **Two spellings, because the rig writes one and the re-tracker the
    other.** A retracked file names every point in the cell; the LIVE
    recorder writes a positional array, ``[[x,y,c], …]``, with the names
    declared once in the header, which is smaller by about a third over a
    75,000-frame session. A reader that understands only one of them returns
    ``None`` for the other, and a session recorded with pose then reads back
    as a session with no pose at all, silently.

    ``names`` supplies the header's ``body_parts`` for the positional form.
    Without them a positional cell cannot be interpreted, and that is
    reported rather than quietly dropped.
    """
    if is_na(cell):
        return None
    try:
        val = json.loads(cell)
    except (TypeError, ValueError):
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, list):
        return _named_pose(val, names)
    return None


def _named_pose(points: list, names: Sequence[str]) -> Optional[dict]:
    """A positional pose array against the body-part names it belongs to."""
    if not points:
        return {}
    # A multi-instance snapshot nests one array per animal. The single-animal
    # readers below want one animal, and the first is the one every existing
    # measure already assumed.
    if isinstance(points[0], list) and points[0] and isinstance(points[0][0], list):
        points = points[0]
    if not names:
        logger.warning(
            "a frame's pose is written positionally (%d points) but the file "
            "declares no body_parts, so the points cannot be named and the "
            "frame is being read as untracked", len(points))
        return None
    if len(points) != len(names):
        logger.warning(
            "a frame carries %d pose points but the header names %d body "
            "parts (%s), the extra points cannot be identified",
            len(points), len(names), ", ".join(names[:6]))
    out = {}
    for name, point in zip(names, points):
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            out[name] = list(point)
    return out


def format_pose(pose: Optional[dict]) -> str:
    if pose is None:
        return "none"
    return json.dumps(pose, separators=(",", ":"))


def zones_as_list(zones: Any, *, _inner: bool = False) -> List[dict]:
    """Any zone payload → a list of zone dicts, each carrying its ``name``.

    Three shapes occur: a plain list, a ``name → zone`` mapping, and the
    sibling port's ``{"scale": …, "arena": …, "zones": {…}}`` config wrapper.
    Read without unwrapping, that last one yields zones called "scale" and
    "zones" and none of the real ones.
    """
    if not zones:
        return []
    if isinstance(zones, dict) and not _inner:
        # The wrapper is recognised by HAVING a "zones" key, not by that key
        # holding anything. A recording written before any zone was drawn
        # carries `{"scale": {}, "arena": {}, "zones": {}}`, and reading that
        # as a plain name→zone mapping turned an empty block into three zones
        # called "scale", "arena" and "zones", which then appeared in the
        # Space column, in the zone picker, and as columns in the results.
        inner = zones.get("zones")
        if "zones" in zones and isinstance(inner, (dict, list)):
            # `_inner`: only the TOP level is the config wrapper. A zone
            # legitimately NAMED "zones" made the recursion treat the
            # name→zone mapping as a second wrapper and return nothing at
            # all, every zone in the file vanished because one of them had
            # an unlucky name.
            out = zones_as_list(inner, _inner=True)
            # `scale` and `arena` sit beside `zones` in that wrapper. Carry
            # them only when they actually describe a shape, inventing a
            # schema for a key we have only ever seen empty would be a guess.
            for key in ("scale", "arena"):
                extra = zones.get(key)
                if isinstance(extra, dict) and (extra.get("points")
                                                or extra.get("values")):
                    zz = dict(extra)
                    zz.setdefault("name", key)
                    zz.setdefault("type", key)
                    out.append(zz)
            return out
    if isinstance(zones, list):
        return [z for z in zones if isinstance(z, dict) and _has_geometry(z)]
    if isinstance(zones, dict):
        out = []
        for name, z in zones.items():
            if not isinstance(z, dict) or not _has_geometry(z):
                # A zone with no shape is not a zone. These arrive from a
                # block written before an empty `{"scale":{},"arena":{},
                # "zones":{}}` was read correctly, three entries called
                # scale, arena and zones, with empty point lists, which then
                # showed in the zone picker and produced columns of zeros.
                continue
            zz = dict(z)
            zz.setdefault("name", name)
            out.append(zz)
        return out
    return []


def zone_config_from_list(zones: Sequence[dict]) -> Dict[str, Any]:
    """A list of zone dicts → the ``{scale, arena, zones}`` block the rig writes.

    The inverse of :func:`zones_as_list`, and shaped after what the recorder
    actually writes rather than after what the key names suggest: a scale zone
    stays among the named zones, carrying ``"type": "scale"``, and the
    top-level ``scale`` and ``arena`` keys stay empty. Recordings written by
    the rig look like that, `"Scale": {"type": "scale", ...}` inside
    ``zones``: and a file rewritten here has to be indistinguishable from one
    the rig wrote, or the two halves of this system disagree about the same
    session.
    """
    named: Dict[str, Any] = {}
    for z in zones or []:
        if not isinstance(z, dict):
            continue
        name = str(z.get("name") or "").strip()
        if not name or not _has_geometry(z):
            # A zone with no geometry is not a zone. They arrive from an
            # analysis folder written before an empty `{"scale":{},
            # "arena":{},"zones":{}}` block was read correctly: it came back
            # as three zones called scale, arena and zones, and writing those
            # into a recording put three empty shapes in the operator's own
            # data file.
            continue
        named[name] = {k: v for k, v in z.items() if k != "name"}
    return {"scale": {}, "arena": {}, "zones": named}


def _has_geometry(zone: dict) -> bool:
    """Whether this zone describes a shape at all.

    Points for a polygon, line or scale; a centre with axes or a radius for an
    ellipse or circle written the compact way.
    """
    if zone.get("points"):
        return True
    return bool(zone.get("center") and (zone.get("semi_axes")
                                        or zone.get("radius_norm")))


def format_zone_config(zone_config: Dict[str, Any]) -> str:
    """The block's text, laid out the way the recorder lays it out.

    One zone per line, compact within the line. Matching the rig's shape means
    a file it wrote and a file this rewrote differ only in the zones, which is
    what a diff should show.
    """
    out: List[str] = ["{"]
    keys = list(zone_config.keys())
    for i, key in enumerate(keys):
        tail = "," if i < len(keys) - 1 else ""
        value = zone_config[key]
        if key == "zones" and isinstance(value, dict):
            out.append('  "zones": {')
            names = list(value.keys())
            for j, name in enumerate(names):
                sep = "," if j < len(names) - 1 else ""
                out.append('    "' + name + '": '
                           + json.dumps(value[name], separators=(",", ":"))
                           + sep)
            out.append("  }" + tail)
        else:
            out.append('  "' + key + '": '
                       + json.dumps(value, separators=(",", ":")) + tail)
    out.append("}")
    return "\n".join(out)


def _line_ending(path: str) -> str:
    """The line ending this file already uses.

    A recorder on Windows writes CRLF. Reading in the default text mode turns
    those into bare LF, so writing the lines back out rewrites the ending of
    every row in the file, one byte each, on tens of thousands of rows, in a
    file this is meant to be copying through untouched.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(65536)
    except OSError:
        return "\n"
    return "\r\n" if b"\r\n" in head else "\n"


def write_zone_config(path: str, zones: Sequence[dict]) -> None:
    """Put these zones INTO the recording, in its own header block.

    The analyser keeps an edit beside the recording, in its bundle, so the
    original file is never touched by accident. That is the right default and
    the wrong only option: a recording carried to another machine, or handed
    to a colleague, arrives without the zones anyone drew for it.

    Only the ``#zone_config_begin`` / ``#zone_config_end`` block changes. Every
    other byte, including all the per-frame rows and their line endings, is
    copied through unread. A file with no block gains one where the recorder
    would have put it, just before the tracker line, which is the first thing
    written after it.

    Written to a temporary file and moved into place, so an interrupted write
    cannot leave a recording half-rewritten.
    """
    block = format_zone_config(zone_config_from_list(zones)).split("\n")
    eol = _line_ending(path)
    tmp = path + ".zonetmp"
    wrote = False

    def _emit(dst) -> None:
        dst.write("#zone_config_begin" + eol)
        for line in block:
            dst.write(line + eol)
        dst.write("#zone_config_end" + eol)

    # newline="" on both sides: the endings this file uses are its own, and
    # every line not in the zone block goes through exactly as it arrived.
    src = open(path, "r", encoding="utf-8", errors="ignore", newline="")
    dst = open(tmp, "w", encoding="utf-8", newline="")
    try:
        skipping = False
        for raw in src:
            line = raw.rstrip("\r\n")
            if skipping:
                if line.startswith("#zone_config_end"):
                    skipping = False
                continue
            if line.startswith("#zone_config_begin"):
                _emit(dst)
                skipping = True
                wrote = True
                continue
            if not wrote and (line.startswith("#tracker")
                              or line.startswith("#columns")):
                _emit(dst)
                wrote = True
            dst.write(raw)
    finally:
        src.close()
        dst.close()

    if not wrote:                       # no header at all: leave it alone
        os.remove(tmp)
        raise ValueError(
            os.path.basename(path) + " has no header to write zones into")
    os.replace(tmp, path)


# ── reading ──────────────────────────────────────────────────────────────────

def format_row(columns: Sequence[str], values: Dict[str, Any]) -> str:
    """One data row, positionally correct by construction.

    Missing cells are written as ``na``, never silently shifted, which is the
    failure mode a hand-built f-string invites.
    """
    cells = []
    for name in columns:
        v = values.get(name, TRACKER_OFF)
        if v is None:
            v = "none"
        elif isinstance(v, float):
            v = "none" if math.isnan(v) else f"{v:.4f}".rstrip("0").rstrip(".") or "0"
        elif isinstance(v, (dict, list)):
            v = json.dumps(v, separators=(",", ":"))
        cells.append(str(v))
    return "\t".join(cells) + "\n"


def iter_rows(path: str, header: Optional[VideoDataHeader] = None
              ) -> Iterator[Dict[str, Any]]:
    """Yield one dict per per-frame data row, keyed by CANONICAL column name.

    Every dialect this module knows lands on the same keys, so a caller never
    branches on the file's age. Rows carry ``_index`` (0-based position among
    data rows) and ``_tracker_off`` (the ``na`` sentinel, distinct from a
    tracked frame with nothing found).
    """
    h = header or VideoDataHeader.parse(path)
    cols = h.columns or tuple(BASE_COLUMNS)
    idx = 0
    try:
        fh = open(path, "r", encoding="utf-8", errors="ignore")
    except OSError:
        return
    with fh:
        for line in fh:
            s = line.rstrip("\n\r")
            if not s.strip():
                continue
            if s.lstrip().startswith("{"):
                row = _json_row(s, idx)
                if row is not None:
                    yield row
                    idx += 1
                continue
            if s.startswith(("#", "Zones", "B ", "M ", "V ", "body_parts:")):
                continue
            parts = s.split("\t")
            if not _looks_like_data_row(parts):
                continue
            row: Dict[str, Any] = {"_index": idx, "_parts": parts}
            for i, name in enumerate(cols):
                row[name] = parts[i] if i < len(parts) else ""
            if h.dialect == DIALECT_HASH_HEADER:
                _normalise_hash_row(row)
            _name_pose_cell(row, h)
            row["_tracker_off"] = (
                str(row.get("pose_array", "")).strip().lower() == TRACKER_OFF)
            yield row
            idx += 1


def _name_pose_cell(row: Dict[str, Any], header) -> None:
    """Rewrite a positional pose cell into the named form, in place.

    Done once here rather than at each of the dozen places that read a pose,
    so a file written by the live recorder and a file written by the
    re-tracker present the same thing to everything downstream, which is the
    contract the two halves of the pipeline meet on.

    Costs nothing on a file that is already named: the cheap first character
    tells the two apart before any JSON is touched.
    """
    cell = row.get("pose_array")
    if not isinstance(cell, str):
        return
    text = cell.lstrip()
    if not text.startswith("["):
        return
    pose = parse_pose(cell, getattr(header, "body_parts", ()) or ())
    if pose:
        row["pose_array"] = format_pose(pose)


#: Timestamp passes, keyed by the file's identity on disk. Two callers want
#: this for the same recording at the same moment, the clock resolver and the
#: tab's header reader, and a recording is not re-read between them. Bounded,
#: because a long session is tens of thousands of floats.
_TS_CACHE: "OrderedDict[tuple, Tuple[float, ...]]" = OrderedDict()
_TS_CACHE_MAX = 64


def _file_key(path: str) -> Optional[tuple]:
    """Identity of a file's CONTENT, so an edited recording is re-read."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def timestamps(path: str, header: Optional[VideoDataHeader] = None
               ) -> List[float]:
    """Every data row's ``frame_ts_ms``, and nothing else, in one pass.
    One entry per data row, ``nan`` where the cell does not parse, so the
    length is the row count and the caller can have the duration from the ends
    without a second pass.

    Callers that want ONLY the time were paying for a whole row: ``iter_rows``
    builds a dict per line and normalises every field, and both the clock
    resolver and the tab's header reader ran it over every recording, the
    same rows parsed twice, 250,000 dicts to open thirty files. This reads the
    one column those callers asked for.
    """
    key = _file_key(path)
    if key is not None and key in _TS_CACHE:
        _TS_CACHE.move_to_end(key)
        return list(_TS_CACHE[key])
    out = _read_timestamps(path, header)
    if key is not None:
        _TS_CACHE[key] = tuple(out)
        while len(_TS_CACHE) > _TS_CACHE_MAX:
            _TS_CACHE.popitem(last=False)
    return out


def _read_timestamps(path: str, header: Optional[VideoDataHeader] = None
                     ) -> List[float]:
    """:func:`timestamps` without the cache."""
    h = header or VideoDataHeader.parse(path)
    col = timestamp_column(h)
    if not col:
        return []
    cols = h.columns or tuple(BASE_COLUMNS)
    if col not in cols:                    # unusual layout: the general reader
        return [parse_float(r.get(col)) for r in iter_rows(path, h)]
    i = cols.index(col)
    as_elapsed = h.dialect == DIALECT_HASH_HEADER
    out: List[float] = []
    try:
        fh = open(path, "r", encoding="utf-8", errors="ignore")
    except OSError:
        return []
    with fh:
        for line in fh:
            s = line.rstrip("\n\r")
            if not s.strip():
                continue
            if s.lstrip().startswith("{"):
                # A brace opens two different things: a JSON-per-line data row
                # (a dialect with no fixed column order, the general reader is
                # the honest answer there), and the zone-config block that
                # every current recording carries in its header. Asking
                # `_json_row` which one this is costs a parse on a handful of
                # lines; assuming the first was the second read every file the
                # slow way.
                if _json_row(s, len(out)) is not None:
                    return [parse_float(r.get(col)) for r in iter_rows(path, h)]
                continue
            if s.startswith(("#", "Zones", "B ", "M ", "V ", "body_parts:")):
                continue
            parts = s.split("\t")
            if not _looks_like_data_row(parts):
                continue
            raw = parts[i] if i < len(parts) else ""
            out.append(parse_elapsed(raw) if as_elapsed else parse_float(raw))
    return out


def _normalise_hash_row(row: Dict[str, Any]) -> None:
    """Bring a sibling-format row onto this module's value conventions.

    Two differences that are silent if unhandled: the elapsed time is
    ``MM:SS.mmm`` rather than a number, and an empty cell is written ``-``
    rather than ``none``.
    """
    ms = parse_elapsed(row.get("frame_ts_ms"))
    row["frame_ts_ms"] = "" if math.isnan(ms) else f"{ms:.3f}"
    for key in ("location", "stage", "events"):
        if str(row.get(key, "")).strip() == "-":
            row[key] = ""


def _json_row(line: str, idx: int) -> Optional[Dict[str, Any]]:
    """pymazeRetrack dialect: one JSON record per line."""
    try:
        rec = json.loads(line)
    except (TypeError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    pose = rec.get("pose_array", rec.get("pose", rec.get("poses")))
    return {
        "_index": idx,
        "_parts": [],
        "frame_number": rec.get("frame_number", rec.get("frame", idx)),
        "frame_ts_ms": rec.get("frame_ts_ms", rec.get("t", rec.get("time", ""))),
        "pose_ts_ms": "",
        "mcu_ts_ms": "",
        "speed": rec.get("speed", ""),
        "location": rec.get("location", rec.get("zone", "")),
        "pose_array": (json.dumps(pose, separators=(",", ":"))
                       if isinstance(pose, dict) else "none"),
        "stage": rec.get("stage", rec.get("state", "")),
        "_tracker_off": False,
    }


def iter_events(path: str) -> Iterator[Dict[str, Any]]:
    """Yield the timestamped session events (MCU firings, stage changes,
    warnings, the session footer), everything that is not a per-frame row."""
    try:
        fh = open(path, "r", encoding="utf-8", errors="ignore")
    except OSError:
        return
    with fh:
        for line in fh:
            s = line.rstrip("\n\r")
            if not s.strip() or s.startswith(("#", "Zones", "B ", "M ", "V ")):
                continue
            parts = s.split("\t")
            if len(parts) < 2:
                continue
            rtype = parts[1].strip().lower()
            if rtype not in EVENT_ROW_TYPES:
                continue
            ts = parse_float(parts[0], math.nan)
            if rtype == "info" and ts == 0.0:
                continue                       # header, not an event
            yield {"t_s": ts, "type": rtype,
                   "subtype": parts[2] if len(parts) > 2 else "",
                   "content": parts[3] if len(parts) > 3 else ""}


def count_data_rows(path: str) -> int:
    """How many per-frame rows the file holds. Used to cross-check the video's
    frame count, a mismatch means dropped frames, and must be reported rather
    than absorbed."""
    n = 0
    for _ in iter_rows(path):
        n += 1
    return n


# ── writing ──────────────────────────────────────────────────────────────────

def write_session(path: str, header: VideoDataHeader,
                  rows: Iterable[Dict[str, Any]],
                  events: Iterable[Dict[str, Any]] = ()) -> str:
    """Write a complete, valid session file.

    This is the only supported way to produce one offline. It cannot emit a
    column layout :func:`iter_rows` does not understand, because both sides
    read ``header.columns``.
    """
    cols = header.columns or tuple(BASE_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header.render():
            f.write(line)
        for ev in events:
            f.write(f"{float(ev.get('t_s', 0)):.3f}\t{ev.get('type', 'event')}\t"
                    f"{ev.get('subtype', '')}\t{ev.get('content', '')}\n")
        for row in rows:
            f.write(format_row(cols, row))
    return path


def header_for_new_session(*, subject_id: str = "", stage: str = "",
                           start_time: str = "", px_per_cm: float = 0.0,
                           fps: float = 0.0, resolution: Tuple[int, int] = (0, 0),
                           pose_resolution: Tuple[int, int] = (0, 0),
                           zones: Any = None, body_parts: Sequence[str] = (),
                           tracker: Optional[dict] = None,
                           frame_clock: str = "", video_file: str = "",
                           metadata: Optional[dict] = None,
                           extra: Optional[Dict[str, Any]] = None
                           ) -> VideoDataHeader:
    """Build a header for a session that has no sidecar of its own, a bare
    video being tracked offline.

    ``frame_clock`` is mandatory in spirit: it records whether the timestamps
    below are real acquisition times or a grid assumed from a frame rate, and
    an analysis that cannot tell the difference will happily report a number
    it has no right to.
    """
    info: Dict[str, str] = {}
    if video_file:
        info["video_file"] = video_file
    if subject_id:
        info["subject_id"] = subject_id
    if stage:
        info["stage"] = stage
    if start_time:
        info["start_time"] = start_time
    info["units"] = json.dumps(
        {"speed": "cm/s" if px_per_cm > 0 else "px/s", "pose": "px",
         "ts": "ms_since_stage_start", "px_per_cm": round(float(px_per_cm), 4)},
        separators=(",", ":"))
    if frame_clock:
        info["frame_clock"] = frame_clock
    if fps:
        info["target_fps"] = str(fps)
    if resolution and resolution[0]:
        info["resolution"] = f"{int(resolution[0])}x{int(resolution[1])}"
    if pose_resolution and pose_resolution[0]:
        info["pose_resolution"] = (f"{int(pose_resolution[0])}"
                                   f"x{int(pose_resolution[1])}")
    if metadata:
        info["metadata"] = json.dumps(metadata, separators=(",", ":"))
    if body_parts:
        info["body_parts"] = json.dumps(list(body_parts), separators=(",", ":"))
    if tracker:
        info["tracker"] = json.dumps(tracker, separators=(",", ":"))
        info["schema"] = json.dumps({"version": SCHEMA_VERSION},
                                    separators=(",", ":"))
    for k, v in (extra or {}).items():
        info[k] = v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
    zr = ""
    if zones:
        zr = zones if isinstance(zones, str) else json.dumps(
            zones, separators=(",", ":"))
    return VideoDataHeader(info=info, zones_raw=zr,
                           columns=tuple(BASE_COLUMNS), dialect=DIALECT_V4,
                           raw_columns=tuple(BASE_COLUMNS))
