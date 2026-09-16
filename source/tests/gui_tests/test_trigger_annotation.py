"""T2, trigger video annotation (draw_triggers) + overlay/relay plumbing."""
from __future__ import annotations

import numpy as np
import pytest

from source.video.trigger_engine import RuleState, draw_triggers

cv2 = pytest.importorskip("cv2")


def _frame():
    return np.zeros((120, 160, 3), np.uint8)


def _rs(**kw):
    base = dict(id="r", name="reward", active=True, fired=False)
    base.update(kw)
    return RuleState(**base)


# ── drawing ────────────────────────────────────────────────────────────────

def test_chip_drawn_when_active():
    f = _frame()
    draw_triggers(f, [_rs(active=True, color="#39c5cf")])
    assert f.any()   # something was drawn


def test_no_states_is_noop():
    f = _frame()
    draw_triggers(f, [])
    assert not f.any()


def test_hidden_chip_not_drawn():
    f = _frame()
    draw_triggers(f, [_rs(show_on_video=False, highlight_geometry=False)])
    assert not f.any()


def test_geometry_circle_drawn_and_scaled():
    f = _frame()
    draw_triggers(f, [_rs(active=True, show_on_video=False,
                          geom={"type": "circle", "x": 40, "y": 30, "r": 10})],
                  scale_x=1.0, scale_y=1.0)
    # a ring around (40,30) → those rows have colour
    assert f[20:41, 30:51].any()


def test_geometry_respects_scale_offset():
    f = _frame()
    # source (100,100) with 0.5 scale and offset 20 → display (40,40)
    draw_triggers(f, [_rs(active=True, show_on_video=False,
                          geom={"type": "circle", "x": 100, "y": 100, "r": 20})],
                  scale_x=0.5, scale_y=0.5, off_x=20, off_y=20)
    assert f.any()


def test_fired_flash_and_inactive_dim_do_not_crash():
    f = _frame()
    draw_triggers(f, [_rs(active=True, fired=True), _rs(id="b", active=False)],
                  flash_on_fire=True)
    assert f.any()


def test_bad_geom_is_safe():
    f = _frame()
    draw_triggers(f, [_rs(active=True, geom={"type": "line", "a": None, "b": None})])
    # no crash; the chip still drew
    assert f.any()


# ── overlay state + relay ──────────────────────────────────────────────────

def test_overlay_state_triggers_default_empty():
    from source.video.framebus.types import OverlayState
    s = OverlayState()
    assert s.triggers == []
    assert s.trigger_chip_corner == "top_right"


def test_pipeline_fanout_relays_trigger_frame():
    from source.video.framebus.controller import Pipeline
    from source.video.trigger_engine import TriggerFrame
    pipe = Pipeline(target_fps=30)
    try:
        got = []
        unsub = pipe.on_trigger_frame(got.append)
        tf = TriggerFrame(setup_id=1, cam_frame_id=2, states=[_rs()])
        pipe._fanout_trigger_frame(tf)
        assert got and got[0] is tf
        unsub()
        pipe._fanout_trigger_frame(tf)
        assert len(got) == 1   # unsubscribed
    finally:
        pipe.shutdown()
