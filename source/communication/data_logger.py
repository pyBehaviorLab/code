"""Behavioral data logging, events, state transitions, variable changes, analog.

The Data_logger writes one per-session ``.tsv`` (see ``open_data_file``) with a
tab-separated row per record, Time (ms since session start), Type (Event /
State / Print / Variable / …), Subtype (source or operation), Content, and an
inline JSON ``metadata`` header line carrying the task + hardware-def djb2
hashes.

Usage::

    logger = Data_logger(board=pycboard, print_func=print)
    logger.open_data_file("data/mcu_data", "Subject001", metadata=info)
    # Data is automatically logged during the session.
    logger.close_files()
"""

import os
import json
import logging
import time
import numpy as np
from datetime import datetime
from source.communication.message import MsgType, Datatuple

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------
#  Data_logger
# ----------------------------------------------------------------------------------------


def ms_to_readable_time(milliseconds):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    if minutes:
        return f"{minutes}:{seconds:02d}.{milliseconds:03d}"
    return f"{seconds}.{milliseconds:03d}"


class Data_logger:
    """Class for logging data from a pyControl setup to disk"""

    def __init__(self, board, print_func=None):
        self.board = board
        self.print_func = print_func
        self.reset()

    # Flush the TSV at most ~4 Hz instead of on every MCU batch. Error / warning /
    # framework-stop rows still flush eagerly, so error data is durable at once.
    _FLUSH_INTERVAL_S = 0.25

    def reset(self):
        self.data_file = None
        self.file_path = None
        self.subject_ID = None
        self.analog_writers = {}
        self.pre_run_prints = []
        self.end_datetime = None
        self.end_timestamp = None
        self._last_flush = 0.0

    def open_data_file(self, data_dir, subject_ID, datetime_now=None, box_ID=None, metadata=None, video_info=None, file_name=None):
        """Open file tsv/txt file for event data and write header information.
        If state machine uses analog inputs instantiate analog data writers.

        Args:
            data_dir: Directory to save data files
            subject_ID: Subject identifier
            datetime_now: Optional datetime for file naming
            box_ID: Optional Box ID for file naming
            metadata: Optional metadata dictionary to include in file
                      (should contain 'experimenter', 'project', and any
                       additional fields from the metadata manager)
            video_info: Optional dict with video recording info {recorded: bool, video_name: str, video_ts: str}
            file_name: Optional EXACT filename. When given, auto-naming and
                      the no-overwrite guard are skipped, the file is
                      overwritten. Used by the dry-run safety net to write a
                      fixed ``Box<N>.tsv`` that each run overwrites.
        """
        self.data_dir = data_dir
        # subject_ID can be empty, Record opens the TSV regardless so MCU events
        # always land somewhere; "unnamed" is the filename component when blank.
        self.subject_ID = subject_ID or ""
        self.metadata = metadata or {}

        if datetime_now is None:
            datetime_now = datetime.now()
        self.datetime_now = datetime_now  # Store for consistent naming
        self.end_timestamp = None

        # Filename format: ID-Box_ID-rest. Empty subject -> "unnamed".
        # Sanitize the subject_ID for the FILENAME ONLY (header keeps it verbatim):
        # replace path separators and control chars (tab/newline/CR/NUL) with "_"
        # so a stray "zfs/z" or pasted control char can't form an unopenable path.
        from source.datetime_formats import FILE_STEM_TS_FMT
        _raw_sid = (self.subject_ID or "unnamed")
        sid_for_file = "".join(
            "_" if c in ("/", "\\", "\t", "\n", "\r", "\x00") else c
            for c in _raw_sid
        )
        if file_name is None:
            ts_part = "-" + datetime_now.strftime(FILE_STEM_TS_FMT)
            if box_ID:
                file_name = f"{sid_for_file}-Box{box_ID}{ts_part}.tsv"
            else:
                file_name = f"{sid_for_file}{ts_part}.tsv"
            self.file_path = os.path.join(self.data_dir, file_name)
            # Refuse to silently overwrite an existing MCU TSV, a same-second
            # restart of the same box would otherwise clobber an earlier run.
            if os.path.exists(self.file_path):
                raise FileExistsError(
                    f"MCU TSV already exists: {self.file_path}. Two recordings "
                    f"would share one filename (same subject, same box, same "
                    f"second). Wait one second and retry."
                )
        else:
            # Explicit name (dry-run safety net), overwrite on purpose.
            self.file_path = os.path.join(self.data_dir, file_name)
        self.data_file = open(self.file_path, "w", encoding="utf-8", newline="\n")
        self.data_file.write(
            self.tsv_row_str(
                time="time", rtype="type", subtype="subtype", content="content"
            )  # Write header with row names.
        )
        self.write_info_line("experimenter", self.metadata.get('experimenter', ''))
        self.write_info_line("project", self.metadata.get('project', ''))
        self.write_info_line("task_name", self.board.sm_info.name)
        # Hashes written as 8-char zero-padded hex so one string cross-references
        # this TSV, the video _video_data.txt, and the on-disk
        # ``<project>/source/<hex>.py``.
        self.write_info_line("task_file_hash", f"{self.board.sm_info.task_hash:08x}")
        # Hardware-definition lineage, the MCU-held HD's djb2 hash (cross-refs
        # <project>/source/hd/<hex>.py). "00000000" if no HD is on the board.
        self.write_info_line("hardware_def_hash", f"{self.board.sm_info.hardware_def_hash:08x}")
        # Device-driver lineage, {device_file: hash} for every MCU driver, hex
        # cross-refs <project>/source/devices/<hex>.py. Empty dict if no devices.
        self.write_info_line("devices", json.dumps(
            {name: f"{h:08x}" for name, h in (self.board.sm_info.devices or {}).items()}))
        if box_ID:
            self.write_info_line("box_id", str(box_ID))
        self.write_info_line("framework_version", self.board.sm_info.framework_version)
        self.write_info_line("micropython_version", self.board.sm_info.micropython_version)
        self.write_info_line("subject_id", self.subject_ID)
        # Single master timestamp: the same datetime_now that produced the
        # filename stem, video file name, tracking header, and run_id, so every
        # output file of this session carries one identical timestamp.
        from source.datetime_formats import HEADER_TS_FMT
        self.write_info_line(
            "start_time", self.datetime_now.strftime(HEADER_TS_FMT))

        # ``video_info is None`` means the real video filename isn't known yet
        # (the encoder starts after this call); the caller then writes it via
        # ``write_video_info`` once the recorder reports its ``video_path``. Pass
        # an explicit ``{'recorded': False, ...}`` to assert "no video this run".
        if video_info is not None:
            self._write_video_block(video_info)

        # Write metadata as single JSON blob
        if metadata:
            self.write_info_line("metadata", json.dumps(metadata))

        self.write_to_file(self.pre_run_prints)
        self.pre_run_prints = []
        self.analog_writers = {
            ID: Analog_writer(ai["name"], ai["fs"], ai["dtype"], self.file_path)
            for ID, ai in self.board.sm_info.analog_inputs.items()
        }

    def write_info_line(self, subtype, content, time=0):
        self.data_file.write(self.tsv_row_str("info", time, subtype, content))

    def _write_video_block(self, video_info):
        """Write the video-info lines (recorded flag, filename,
        timestamp sidecar, and a JSON blob). Shared by ``open_data_file``
        and the deferred ``write_video_info``."""
        self.write_info_line("video_recorded", str(video_info.get('recorded', False)))
        if video_info.get('video_name'):
            self.write_info_line("video_file", video_info['video_name'])
        if video_info.get('video_ts'):
            self.write_info_line("video_timestamps", video_info['video_ts'])
        self.write_info_line("video", json.dumps(video_info))

    def write_video_info(self, video_info):
        """Record the video-info block after the encoder has started and
        reported its real ``video_path``. Called once per video file:
        operant once per run, maze once per stage recorder. No-op if the
        data file is closed or ``video_info`` is empty."""
        if not video_info or getattr(self, "data_file", None) is None:
            return
        try:
            self._write_video_block(video_info)
        except Exception as e:
            # A failed write leaves the TSV without its video block, breaking
            # analyzer file-resolution, surface it rather than hide it.
            logger.warning("data_logger: video-info block write failed: %r", e)

    @staticmethod
    def _sanitize_tsv_field(value) -> str:
        """Replace tab/newline/CR with a literal escape so a user-supplied
        subject_id or metadata value cannot inject a row split into the TSV.
        Applied to every TSV field so the column-count guarantee always holds."""
        if value is None:
            return ""
        s = str(value)
        return s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")

    def tsv_row_str(self, rtype, time, subtype="", content=""):
        # Output time in seconds with 3 decimal places for consistency with original
        # If time is int (milliseconds), convert to seconds; if it's string (header), keep as is
        time_str = f"{time / 1000:.3f}" if isinstance(time, int) else time
        # Sanitize every text field so a tab/newline cannot break the row format.
        return (
            f"{time_str}\t"
            f"{self._sanitize_tsv_field(rtype)}\t"
            f"{self._sanitize_tsv_field(subtype)}\t"
            f"{self._sanitize_tsv_field(content)}\n"
        )

    def close_files(self):
        if self.data_file:
            # If end_datetime wasn't set by STOPF/ERROR mid-stream, derive it from
            # the master start datetime + MCU elapsed time so stop stays on the same
            # clock as start. If the MCU disconnected, board.timestamp may be stale,
            # so fall back to wall-clock when the derived end drifts >5 s from now.
            from source.datetime_formats import HEADER_TS_FMT
            from datetime import timedelta
            if self.end_datetime is None:
                self.end_timestamp = (
                    self.board.timestamp if hasattr(self.board, 'timestamp') else 0)
                derived = None
                if getattr(self, "datetime_now", None) is not None:
                    derived = (
                        self.datetime_now
                        + timedelta(milliseconds=int(self.end_timestamp or 0))
                    )
                wall_now = datetime.now()
                if derived is not None:
                    drift_s = abs((wall_now - derived).total_seconds())
                    if drift_s <= 5.0:
                        self.end_datetime = derived
                    else:
                        # Stale MCU anchor, use wall_now and flag it in the header
                        # so the analyst knows end is host wall-clock, not MCU-derived.
                        self.end_datetime = wall_now
                        self.write_info_line(
                            "end_time_source",
                            "host_wall_clock_mcu_stale",
                        )
                else:
                    self.end_datetime = wall_now
            self.write_info_line(
                "end_time",
                self.end_datetime.strftime(HEADER_TS_FMT),
                self.end_timestamp,
            )
            self.data_file.close()
            self.data_file = None
        for analog_writer in self.analog_writers.values():
            analog_writer.close_files()
        self.analog_writers = {}

    def process_data(self, new_data):
        """If data_file is open new data is written to file.  If print_func is specified
        human readable data strings are passed to it."""
        if self.data_file:
            self.write_to_file(new_data)
        if self.print_func:
            # Use compact format for live output: D/P timestamp content (cleaner than TSV)
            self.print_func(self.data_to_string(new_data, compact=True), end="")

    def write_to_file(self, new_data):
        data_string = self.data_to_string(new_data)
        if data_string:
            self.data_file.write(data_string)
            # Flush on urgent rows (error/warning/framework-stop) or once per
            # interval; everything else is batched to keep the write path light.
            urgent = any(nd.type in (MsgType.ERROR, MsgType.WARNG, MsgType.STOPF) for nd in new_data)
            now = time.monotonic()
            if urgent or (now - self._last_flush) >= self._FLUSH_INTERVAL_S:
                self.data_file.flush()
                self._last_flush = now
        for nd in new_data:
            if nd.type == MsgType.ANLOG:
                writer_id, data = nd.content
                self.analog_writers[writer_id].save_analog_chunk(timestamp=nd.time, data_array=data)

    def data_to_string(self, new_data, prettify=False, max_len=60, compact=False):
        """Convert list of data tuples into a string.

        Args:
            new_data: List of Datatuple objects to format
            prettify: If True, format for GUI (readable timestamps, multiline vars)
            max_len: Maximum length for variables before wrapping (when prettify=True)
            compact: If True, use compact format (D/P timestamp content) with milliseconds

        Format modes:
            compact=True: 'D 1234 event_name' or 'P 1234 message'
            prettify=True: TSV with readable timestamps
            Both False: TSV with millisecond timestamps for file logging
        """
        # Use list accumulation + join instead of string += to avoid O(n^2).
        parts = []
        for nd in new_data:
            if compact:
                # Compact format: D/P timestamp content
                time_ms = nd.time
                if nd.type == MsgType.STATE:  # State entry.
                    if hasattr(self.board, 'sm_info') and self.board.sm_info and hasattr(self.board.sm_info, 'ID2name'):
                        parts.append(f"D {time_ms} {self.board.sm_info.ID2name[nd.content]}\n")
                    else:
                        parts.append(f"D {time_ms} state_{nd.content}\n")
                elif nd.type == MsgType.EVENT:  # Event.
                    if hasattr(self.board, 'sm_info') and self.board.sm_info and hasattr(self.board.sm_info, 'ID2name'):
                        parts.append(f"D {time_ms} {self.board.sm_info.ID2name[nd.content]}\n")
                    else:
                        parts.append(f"D {time_ms} event_{nd.content}\n")
                elif nd.type == MsgType.PRINT:  # User print output.
                    parts.append(f"P {time_ms} {nd.content}\n")
                elif nd.type == MsgType.VARBL:  # Variable.
                    variables_dict = json.loads(nd.content)
                    var_pairs = [f"{k}:{v}" for k, v in variables_dict.items()]
                    parts.append(f"P {time_ms} {' '.join(var_pairs)}\n")
                elif nd.type == MsgType.THRSH:  # Threshold
                    parts.append(f"P {time_ms} threshold:{nd.content}\n")
                elif nd.type == MsgType.WARNG:  # Warning
                    parts.append(f"P {time_ms} WARNING:{nd.content}\n")
                elif nd.type == MsgType.ERROR:  # Error
                    # Stop time on the same clock as start (datetime_now + MCU ms).
                    from datetime import timedelta
                    self.end_timestamp = nd.time
                    if getattr(self, "datetime_now", None) is not None:
                        self.end_datetime = (
                            self.datetime_now
                            + timedelta(milliseconds=int(nd.time or 0)))
                    else:
                        self.end_datetime = datetime.now()
                    parts.append(f"P {time_ms} ERROR:{nd.content}\n")
            else:
                # Original TSV format
                time = ms_to_readable_time(nd.time) if prettify else nd.time
                if nd.type == MsgType.STATE:  # State entry.
                    parts.append(self.tsv_row_str("state", time, content=self.board.sm_info.ID2name[nd.content]))
                elif nd.type == MsgType.EVENT:  # Event.
                    parts.append(self.tsv_row_str("event", time, nd.subtype, self.board.sm_info.ID2name[nd.content]))
                elif nd.type == MsgType.PRINT:  # User print output.
                    if prettify:
                        print_str = nd.content.replace("\n", "\n\t\t\t")
                    else:
                        print_str = nd.content.replace("\n", "|").replace("\r", "|")
                    parts.append(self.tsv_row_str("print", time, nd.subtype, content=print_str))
                elif nd.type == MsgType.VARBL:  # Variable.
                    var_str = nd.content
                    if prettify:
                        variables_dict = json.loads(nd.content)
                        if len(repr(variables_dict)) > max_len:  # Wrap variables across multiple lines.
                            var_str = "{\n"
                            for var_name, var_value in sorted(variables_dict.items(), key=lambda x: x[0].lower()):
                                var_str += f'\t\t\t"{var_name}": {var_value}\n'
                            var_str += "\t\t\t}"
                    parts.append(self.tsv_row_str("variable", time, nd.subtype, content=var_str))
                elif nd.type == MsgType.THRSH:  # Threshold
                    parts.append(self.tsv_row_str("threshold", time, nd.subtype, content=nd.content))
                elif nd.type == MsgType.WARNG:  # Warning
                    parts.append(self.tsv_row_str("warning", time, content=nd.content))
                elif nd.type in (MsgType.ERROR, MsgType.STOPF):  # Error or stop framework.
                    # Stop time on the SAME clock as start.
                    from datetime import timedelta
                    self.end_timestamp = nd.time
                    if getattr(self, "datetime_now", None) is not None:
                        self.end_datetime = (
                            self.datetime_now
                            + timedelta(milliseconds=int(nd.time or 0)))
                    else:
                        self.end_datetime = datetime.now()
                    if nd.type == MsgType.ERROR:
                        content = nd.content
                        if prettify:
                            content = f"\n\n{content}"
                        else:
                            content = content.replace("\n", "|").replace("\r", "|")
                        parts.append(self.tsv_row_str("error", time, content=content))
        return "".join(parts)

    def print_message(self, msg, source="u"):
        """Print a message to the log and data file. If called pre-run message is logged when
        data file is opened, if called post run message is logged to previously open data file."""
        new_data = [
            Datatuple(
                time=self.board.get_timestamp() if self.board.framework_running else self.board.timestamp,
                type=MsgType.PRINT,
                subtype=MsgType.PRINT.get_subtype(source),
                content=msg,
            )
        ]
        if self.board.framework_running:
            self.process_data(new_data)
            if self.board.data_consumers:
                for data_consumer in self.board.data_consumers:
                    data_consumer.process_data(new_data)
        else:
            self.print_func(self.data_to_string(new_data, prettify=True), end="")
            if self.board.timestamp == 0:  # Pre-run, store note to log when file opened.
                self.pre_run_prints += new_data
            elif self.file_path:  # Post-run, log note to previous data file.
                # Match the session file's encoding/newline (opened utf-8/\n)
                # so a post-run note never mixes \r\n into a \n file.
                with open(self.file_path, "a", encoding="utf-8", newline="\n") as data_file:
                    data_file.write(self.data_to_string(new_data))


# ----------------------------------------------------------------------------------------
#  Analog_writer
# ----------------------------------------------------------------------------------------


class Analog_writer:
    """Class for writing data from one analog input to disk."""

    def __init__(self, name, sampling_rate, data_type, session_filepath):
        self.name = name
        self.sampling_rate = sampling_rate
        self.data_type = data_type
        self.open_data_files(session_filepath)

    def open_data_files(self, session_filepath):
        ses_path_stem, file_ext = os.path.splitext(session_filepath)
        self.path_stem = ses_path_stem + f"_{self.name}"
        self.t_tempfile_path = self.path_stem + ".time.temp"
        self.d_tempfile_path = self.path_stem + f".data-1{self.data_type}.temp"
        self.time_tempfile = open(self.t_tempfile_path, "wb")
        self.data_tempfile = open(self.d_tempfile_path, "wb")
        self.next_chunk_start_time = 0

    def close_files(self):
        """Close data files. Convert temp files to numpy."""
        self.time_tempfile.close()
        self.data_tempfile.close()
        with open(self.t_tempfile_path, "rb") as f:
            times = np.frombuffer(f.read(), dtype="float64")
            np.save(self.path_stem + ".time.npy", times)
        with open(self.d_tempfile_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=self.data_type)
            np.save(self.path_stem + ".data.npy", data)
        os.remove(self.t_tempfile_path)
        os.remove(self.d_tempfile_path)

    def save_analog_chunk(self, timestamp, data_array):
        """Save a chunk of analog data to .pca data file."""
        if np.abs(self.next_chunk_start_time - timestamp / 1000) < 0.001:
            chunk_start_time = self.next_chunk_start_time
        else:
            chunk_start_time = timestamp / 1000
        times = (np.arange(len(data_array), dtype="float64") / self.sampling_rate) + chunk_start_time  # Seconds
        self.time_tempfile.write(times.tobytes())
        self.data_tempfile.write(data_array.tobytes())
        self.time_tempfile.flush()
        self.data_tempfile.flush()
        self.next_chunk_start_time = chunk_start_time + len(data_array) / self.sampling_rate
