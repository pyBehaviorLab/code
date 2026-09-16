"""Recording, encoders, frame log, drop log.

Holds the session-stem naming convention shared by every output file.
"""

import os
from datetime import datetime


def video_info_from_path(video_path) -> dict:
    """Build the MCU-TSV ``video_info`` dict from a recorder's real
    ``video_path`` (set by ``VideoRecorder.start_recording``, with the
    final ``.mp4``/``.avi`` extension already resolved).

    Single source of truth for the video-info block written to the TSV
    header and the runs JSON, both operant and maze derive it from the
    recorder rather than guessing the filename ahead of time. Returns the
    "no video" dict when ``video_path`` is empty.
    """
    if not video_path:
        return {'recorded': False, 'video_name': None, 'video_ts': None}
    name = os.path.basename(str(video_path))
    stem = name.rsplit('.', 1)[0]
    return {
        'recorded': True,
        'video_name': name,
        # Per-frame timestamp sidecar, FrameLog writes <stem>_video_data.txt
        # (there is no separate *_video_timestamps.txt file).
        'video_ts': stem + "_video_data.txt",
    }


def build_session_stem(subject_id: str, setup_id: int,
                       start_wall: datetime) -> str:
    """Single source of truth for the per-session filename stem.

    Used by MCU TSV (``<stem>.tsv``), video (``<stem>.mp4``), frame log
    (``<stem>_video_data.txt``), and any per-session sidecar. Keeps the
    four output files reconcilable from filename alone, even after they
    migrate between folders.
    """
    return f"{subject_id}-Box{int(setup_id)}-{start_wall.strftime('%Y-%m-%d-%H%M%S')}"
