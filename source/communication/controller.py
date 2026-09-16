"""Central MCU controller, thin per-box ``Pycboard`` registry.

Original pyControl is single-box: one ``Pycboard`` instance owned by the
GUI directly. This file ports that ownership to a multi-box GUI without
inventing a parallel API: ``MCUController`` is a dict-like registry of
``box_id -> Pycboard``. Callers use the Original ``Pycboard`` method
names, ``set_variable``, ``get_variable``, ``trigger_event``,
``process_data``, ``framework_running``, ``sm_info``, …, directly on
the board returned by indexing.

Usage::

    mcu[box_id].set_variable(name, value)
    mcu[box_id].framework_running
    for board in mcu.values():
        if board.framework_running:
            board.stop_framework()

Mirrors ``source.video.framebus.Pipeline`` (camera/tracking) in
position but NOT in shape: Pipeline owns the whole pipeline
and exposes its own thick API. MCUController is intentionally thin,
the per-box ``Pycboard`` already IS the API, identical to Original.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

from source.communication.pycboard import Pycboard

logger = logging.getLogger(__name__)


class MCUController:
    """Dict-like registry of per-box ``Pycboard`` instances.

    The ``boards`` attribute is public; iterating, indexing, ``in``, and
    ``len()`` all read through to it. The two genuinely-new operations
    are ``register`` (install a freshly-opened board) and ``unregister``
    (drop a closed board). Everything else is plain ``Pycboard`` usage,
    same method names as upstream pyControl.
    """

    def __init__(self, parent_window: Any = None) -> None:
        self.boards: Dict[int, Pycboard] = {}
        self._parent = parent_window

    # ── Registry ops (the only genuinely new names) ──────────────────

    def register(self, setup_id: int, board: Pycboard) -> None:
        """Install a freshly-opened ``Pycboard`` for ``box_id``. Idempotent,
        replaces any prior entry (closing it is the caller's job)."""
        self.boards[int(setup_id)] = board

    def unregister(self, setup_id: int) -> Optional[Pycboard]:
        """Remove and return the board for ``box_id`` (or None if absent).
        Closing the returned board is the caller's job."""
        return self.boards.pop(int(setup_id), None)

    # ── Dict-like access, callers use Pycboard methods directly ─────

    def __getitem__(self, setup_id: Any) -> Pycboard:
        return self.boards[int(setup_id)]

    def __contains__(self, setup_id: Any) -> bool:
        return int(setup_id) in self.boards

    def __iter__(self) -> Iterator[int]:
        return iter(self.boards)

    def __len__(self) -> int:
        return len(self.boards)

    def get(self, setup_id: Any, default: Optional[Pycboard] = None
            ) -> Optional[Pycboard]:
        """Same semantics as ``dict.get``, None when not registered."""
        return self.boards.get(int(setup_id), default)

    def keys(self):  return self.boards.keys()
    def values(self): return self.boards.values()
    def items(self):  return self.boards.items()

    # ── Whole-fleet shutdown helpers (loop convenience only) ─────────

    def stop_all(self) -> None:
        """Stop framework on every running board. Loops; per-board errors
        log and continue (one failing board doesn't block the rest)."""
        for bid, board in list(self.boards.items()):
            try:
                if board.framework_running:
                    board.stop_framework()
            except BaseException as e:
                logger.debug("MCU stop_all: box %s: %s", bid, e)

    def close_all(self) -> None:
        """Stop + close every board, then drop the registry."""
        for bid, board in list(self.boards.items()):
            try:
                if board.framework_running:
                    board.stop_framework()
            except BaseException as e:
                logger.debug("MCU close_all stop: box %s: %s", bid, e)
            try:
                board.close()
            except BaseException as e:
                logger.debug("MCU close_all close: box %s: %s", bid, e)
        self.boards.clear()


__all__ = ["MCUController"]
