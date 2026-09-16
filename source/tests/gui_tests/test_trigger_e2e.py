"""End-to-end: a trigger authored into TrackingConfig.triggers fires a real MCU
event when a pose frame satisfies it, and its per-frame state reaches the GUI,
driving the actual Pipeline → PoseSink → MCUPusher/TrackingPushPolicy →
board.queue_trigger_event chain (the single evaluator, no parallel engine).
"""
from __future__ import annotations

from source.video.framebus.controller import Pipeline


class _FakeBoard:
    framework_running = True

    def __init__(self):
        self.fired = []

    def queue_trigger_event(self, name):
        self.fired.append(name)

    def queue_set_coordinates(self, name, val):
        pass

    def queue_trigger_intrinsic_event(self, name):
        pass


def _install(pipe, setup_id, triggers):
    board = _FakeBoard()
    pipe.push.register_box(setup_id, board)
    pipe.push._policies[setup_id].update(triggers=triggers)
    return board


def _pose(pipe, setup_id, speed, coords):
    pipe.push.on_pose_result(
        setup_id, cam_frame_id=1, pose_array=[], location=None, speed=speed,
        zones_by_body_part={}, raw_pose_dict=coords)


def test_speed_rule_fires_mcu_event_end_to_end():
    pipe = Pipeline(target_fps=30)
    try:
        board = _install(pipe, 1, [
            {"condition": "speed_gt", "threshold": 50, "event_name": "moving"}])
        _pose(pipe, 1, 10.0, {"centroid": (0, 0, 0.9)})    # slow → nothing
        assert board.fired == []
        _pose(pipe, 1, 120.0, {"centroid": (0, 0, 0.9)})   # fast → edge → fire
        assert board.fired == ["moving"]
        _pose(pipe, 1, 130.0, {"centroid": (0, 0, 0.9)})   # still fast → no re-fire
        assert board.fired == ["moving"]
    finally:
        pipe.shutdown()


def test_no_rules_fires_nothing():
    pipe = Pipeline(target_fps=30)
    try:
        board = _install(pipe, 1, [])
        _pose(pipe, 1, 200.0, {"centroid": (0, 0, 0.9)})
        assert board.fired == []
    finally:
        pipe.shutdown()


def test_event_not_fired_when_board_task_stopped():
    pipe = Pipeline(target_fps=30)
    try:
        board = _install(pipe, 1, [
            {"condition": "speed_gt", "threshold": 50, "event_name": "moving"}])
        board.framework_running = False
        _pose(pipe, 1, 200.0, {"centroid": (0, 0, 0.9)})
        assert board.fired == []
    finally:
        pipe.shutdown()


def test_pose_fanout_accepts_infer_done_ns():
    # PoseSink fans its full kwarg set to every subscriber, so the pose
    # fanout must accept infer_done_ns. A subscriber that does not raises
    # TypeError every frame, and the GUI overlay subscribers never run.
    pipe = Pipeline(target_fps=30)
    try:
        seen = []
        pipe.on_pose_result(lambda *a, **k: seen.append(a))
        # Call exactly as notify_subscribers does, with the extra kwargs.
        pipe._fanout_pose_result(
            1, 1, [], None, 0.0, {}, {"centroid": (0, 0, 0.9)},
            capture_host_ns=123, forecast_coords=None, infer_done_ns=456)
        assert seen                      # overlay subscriber ran (no TypeError)
    finally:
        pipe.shutdown()


def test_set_features_scale_reaches_policy():
    # px/mm cannot be derived from typed cfg.zones, so the GUI pushes it via
    # set_features_scale, without it, cm/s and mm thresholds do not scale.
    pipe = Pipeline(target_fps=30)
    try:
        pipe.push.register_box(1, _FakeBoard())
        pipe.set_features_scale(1, 2.5)
        assert pipe.push._policies[1]._features.px_per_mm == 2.5
    finally:
        pipe.shutdown()


def test_authored_trigger_reaches_policy_via_config():
    """An advanced trigger authored on the box's central TrackingConfig
    (the "Event & Trigger" tab) fires through apply_tracking_config, the
    real GUI→policy path, not a direct policy poke."""
    pipe = Pipeline(target_fps=30)
    try:
        board = _FakeBoard()
        pipe.push.register_box(1, board)
        pipe.update_tracking_config(1, triggers=[
            {"condition": "speed_gt", "threshold": 50, "event_name": "moving"}])
        pipe.apply_tracking_config(1)
        _pose(pipe, 1, 120.0, {"centroid": (0, 0, 0.9)})
        assert board.fired == ["moving"]
    finally:
        pipe.shutdown()


def test_authored_triggers_round_trip_through_config_json():
    """color / show_on_video / plot survive TrackingConfig to_json → from_json."""
    from source.video.framebus.types import TrackingConfig
    tc = TrackingConfig(setup_id=1, triggers=[
        {"condition": "speed_gt", "threshold": 42, "event_name": "run",
         "color": "#ff8800", "show_on_video": False, "plot": True}])
    back = TrackingConfig.from_json(tc.to_json(), setup_id=1)
    assert back.triggers == tc.triggers


def test_trigger_frame_relayed_from_policy_to_gui():
    """The policy's per-frame trigger state reaches the GUI relay (annotation +
    Session-Plot lane) via the pipeline fanout."""
    pipe = Pipeline(target_fps=30)
    try:
        frames = []
        pipe.on_trigger_frame(frames.append)
        _install(pipe, 1, [
            {"condition": "speed_gt", "threshold": 50, "event_name": "fast",
             "name": "running"}])
        _pose(pipe, 1, 100.0, {"centroid": (0, 0, 0.9)})
        assert frames
        s = frames[-1].states[0]
        assert s.name == "running" and s.active and s.fired
    finally:
        pipe.shutdown()
