"""Insurance test: the two DJB2 implementations in the codebase MUST agree.

Two copies exist intentionally:
  * ``source/communication/pycboard.py :: _djb2_file``: kept here so the
    MCU layer is fully self-contained (matches the
    ``feedback_mcu_layer_compact`` rule).
  * ``source/config/hashing.py :: djb2_int_from_file``: used by the
    snapshot store + GUI/runtime code.

Both compute the same 4-byte little-endian chunked DJB2 hash. A change
to one without the other would silently desync the host/snapshot hashes
from the MCU's view of the same file. This test fails immediately if the
implementations diverge.
"""
import tempfile
from pathlib import Path

from source.communication.pycboard import _djb2_file as mcu_djb2_int
from source.config.hashing import djb2_int_from_file, djb2_hex_from_file


def _write_temp(contents: bytes) -> Path:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    tf.write(contents)
    tf.close()
    return Path(tf.name)


def test_djb2_dual_impl_agree_on_empty_file():
    p = _write_temp(b"")
    try:
        assert mcu_djb2_int(p) == djb2_int_from_file(p)
    finally:
        p.unlink()


def test_djb2_dual_impl_agree_on_small_aligned():
    # Multiple of 4 bytes, no last-chunk padding issue.
    p = _write_temp(b"abcdEFGH" * 4)  # 32 bytes
    try:
        assert mcu_djb2_int(p) == djb2_int_from_file(p)
    finally:
        p.unlink()


def test_djb2_dual_impl_agree_on_unaligned_tail():
    # Length not divisible by 4, both impls handle the partial last
    # chunk identically (only via int.from_bytes(c, "little") of the
    # remaining bytes).
    p = _write_temp(b"hello world from pyBehaviorLab")
    try:
        assert mcu_djb2_int(p) == djb2_int_from_file(p)
    finally:
        p.unlink()


def test_djb2_dual_impl_agree_on_realistic_task_file():
    # Realistic-shape pyControl task: ~2 KB of typical Python text.
    src = b"""
from pyControl.utility import *
import hardware_definition as hw

states = ['init', 'wait_for_press', 'reward']
events = ['poke_left', 'poke_right', 'session_timer']
initial_state = 'init'

v.n_trials = 0
v.reward_duration = 250

def init(event):
    if event == 'entry':
        print('Initialising trial')
        timed_goto_state('wait_for_press', 1000)

def wait_for_press(event):
    if event == 'poke_left':
        goto_state('reward')

def reward(event):
    if event == 'entry':
        hw.reward_pin.value(1)
        timed_goto_state('init', v.reward_duration)
    elif event == 'exit':
        hw.reward_pin.value(0)
""" * 8  # padded to ~14 KB so we exercise multiple 4-byte chunks
    p = _write_temp(src)
    try:
        assert mcu_djb2_int(p) == djb2_int_from_file(p)
        # Hex form derived from the int form matches the snapshot-store
        # filename convention.
        assert djb2_hex_from_file(p) == f"{mcu_djb2_int(p):08x}"
    finally:
        p.unlink()


def test_djb2_known_vector():
    # Pin a known input/output so an algorithm tweak that happens to leave
    # the dual-impl in sync but produces different output from before is
    # still caught.
    p = _write_temp(b"pyBehaviorLab\n")
    try:
        h = mcu_djb2_int(p)
        # Recompute manually for sanity, 4-byte LE chunks of 14 bytes
        # = chunks (b"pyBe", b"havi", b"orLa", b"b\n")
        expected = 5381
        for chunk in (b"pyBe", b"havi", b"orLa", b"b\n"):
            expected = ((expected << 5) + expected + int.from_bytes(chunk, "little")) & 0xFFFFFFFF
        assert h == expected
        assert djb2_int_from_file(p) == expected
    finally:
        p.unlink()
