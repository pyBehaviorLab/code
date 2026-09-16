"""McuRowMirror, mirror MCU STATE/EVENT messages into the tracking writer.

A pycboard ``data_consumer`` that walks each ``new_data`` batch the 10 ms
pyControl drain delivers and mirrors MCU STATE / EVENT messages into the
active ``_video_data.txt`` writer as v2 ``state`` / ``events`` column entries.

The actual state / event name is resolved via
``pycboard.sm_info.ID2name[content]``: NOT the trigger subtype
(``user`` / ``task`` / ``timer`` / …), which carries no information about
which event fired. Events whose names appear in
``FrameLog.SUPPRESS_EVENT_NAMES`` (e.g. ``zone_changed``) are filtered
inside the writer's ``note_event`` so the file stays free of redundant noise.

PRINT messages are NOT mirrored here, they land in the per-session
``.log`` file via pycboard's ``print_func`` already.

This mirror does NO clock work. The per-frame MCU framework timestamps
come from ``pycboard.fw_ms_at(capture_host_ns)`` in ``RecorderSink``,
monotonic extrapolation from the last-message anchor, valid once the
first anchored MCU message arrives (``_fw_anchored``).
"""

from __future__ import annotations


class McuRowMirror:
    """pycboard ``data_consumer`` that mirrors MCU STATE / EVENT names
    into a per-box tracking writer's ``state`` / ``events`` columns.

    ``writer_provider`` / ``sm_info_provider`` are late-bound zero-arg
    callables so the mirror survives the writer opening/closing AND the
    ``sm_info`` being reassigned on every task Upload/Reset. Either may
    return ``None`` when not yet attached, the mirror then does nothing.
    """

    __slots__ = ("_get_writer", "_get_sm_info")

    def __init__(self, writer_provider=None, sm_info_provider=None) -> None:
        self._get_writer = writer_provider
        self._get_sm_info = sm_info_provider

    def process_data(self, new_data):
        if not new_data:
            return
        try:
            from source.communication.message import MsgType as _MT
        except Exception:
            return  # without MsgType we can't tell timestamped messages apart
        writer = self._get_writer() if self._get_writer else None
        if writer is None:
            return
        sm_info = self._get_sm_info() if self._get_sm_info else None
        id2name = getattr(sm_info, "ID2name", None) if sm_info else None
        if id2name is None:
            return
        for nd in new_data:
            t = getattr(nd, "type", None)
            if t not in (_MT.STATE, _MT.EVENT):
                continue
            # Use the real name from sm_info.ID2name[content]; the
            # ``content`` field on EVENT/STATE Datatuples is the ID integer
            # set by ``pycboard.process_data``. Subtype carries only the
            # SOURCE (user/task/timer/…) and would lose the actual identity.
            content = getattr(nd, "content", None)
            if content is None:
                continue
            name = id2name.get(int(content))
            if not name:
                continue
            try:
                if t is _MT.STATE:
                    writer.note_state(name=name)
                else:
                    writer.note_event(name=name)
            except Exception:
                pass


__all__ = ["McuRowMirror"]
