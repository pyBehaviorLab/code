"""PoseSink wake-token purge must be per-box, not scorched-earth.

The worker snapshots pending frames, then purges leftover wake tokens.
A token for a box whose frame arrived BETWEEN the snapshot and the purge
must survive, eating it left that frame waiting out the full 0.5 s
``get()`` timeout before the next drain (latency, not loss). The None
shutdown sentinel must never be swallowed either.
"""
import queue

from source.video.framebus.pose_sink import PoseSink


def _sink():
    s = PoseSink.__new__(PoseSink)
    # Sinks hold one wake queue per worker; ``_wake`` is the read-only
    # accessor for the single-worker case, which PoseSink always is.
    s._nworkers = 1
    s._wakes = [queue.Queue()]
    return s


def test_a_late_arrivals_token_survives_the_purge():
    s = _sink()
    s._wake.put(1)          # drained this round
    s._wake.put(2)          # frame arrived after the drain snapshot
    s._purge_wake_tokens({1})
    assert s._wake.get_nowait() == 2
    assert s._wake.empty()


def test_drained_boxes_tokens_are_dropped():
    s = _sink()
    for tok in (1, 1, 3, 1):
        s._wake.put(tok)
    s._purge_wake_tokens({1, 3})
    assert s._wake.empty()


def test_the_shutdown_sentinel_is_never_swallowed():
    s = _sink()
    s._wake.put(1)
    s._wake.put(None)
    s._purge_wake_tokens({1})
    assert s._wake.get_nowait() is None
