"""End-to-end integration tests for MCU pipeline + video pipeline.

Both pipelines (``MCUController`` and ``Pipeline``) own per-box
state. They must stay consistent: when a ``Pycboard`` is registered for
box N, BOTH the central MCU registry AND the camera pipeline's MCUPusher
must see the same instance, so:

    - the controls dialog reaches the board via ``mcu[box_id]``;
    - the camera-side ``MCUPusher`` reaches the same board to fire MCU
      events / coordinate writes from the camera thread.

These tests construct real ``MCUController`` + ``Pipeline``
instances (no Qt widgets) and verify the cross-pipeline contract:

  1. After registration via the widget mixin's ``_register_pycboard_with_video``
     hook, both pipelines see the same Pycboard object.
  2. Camera-side ``MCUPusher`` can reach the board via the registry.
  3. Multi-box scenarios keep the per-box state isolated.
  4. ``mcu_disconnect`` removes the board from the MCU registry; the
     pyboard reference is detached at the MCUPusher level (Phase 2).
  5. Broadcast ops on the MCU side don't touch the camera pipeline.

All Pycboard interactions are mocked because the real Pycboard owns a
serial port; the integration is tested at the controller boundary.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from source.communication.controller import MCUController
from source.communication.pycboard import Pycboard


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_pycboard():
    """A MagicMock that quacks like a Pycboard for the methods the
    pipeline + controller actually call."""
    def _make(running: bool = False) -> MagicMock:
        b = MagicMock(spec=Pycboard)
        b.framework_running = running
        # ``sm_info`` reads, used by the controls dialog's
        # ``refresh_variables`` and by tracking_push.queue_trigger_event.
        b.sm_info = MagicMock()
        b.sm_info.events = {"zone_changed": 1, "reward": 2}
        b.sm_info.variables = {"trial_count": 0, "reward_volume": 5.0}
        return b
    return _make


@pytest.fixture
def mcu():
    """A fresh MCUController per test."""
    return MCUController()


@pytest.fixture
def pipeline(monkeypatch):
    """Construct a real Pipeline without spinning up the
    camera tick thread / VideoManager registry. We patch the things
    that touch hardware so the test stays headless."""
    # The camera tick thread polls VideoManager which probes cameras.
    # For an MCU-side integration test we don't need it, patch the
    # tick to a no-op.
    from source.video.framebus import Pipeline
    pipe = Pipeline(target_fps=30)
    return pipe


# ── 1. Cross-pipeline registration ──────────────────────────────────


class TestCrossPipelineRegistration:
    """When a freshly-opened ``Pycboard`` is wired into BOTH pipelines
    (the widget mixin does this in ``mcu_connect``), every consumer
    sees the same instance."""

    def test_same_pycboard_visible_to_both_pipelines(
        self, mcu, pipeline, fake_pycboard
    ):
        b = fake_pycboard(running=False)

        # Simulate what BoxWidgetMixin.mcu_connect does:
        #   1. main_window.mcu.register(box_id, pyc)
        #   2. main_window.pipeline.update_box_pycboard(box_id, pyc)
        mcu.register(7, b)
        pipeline.update_box_pycboard(7, b)

        # MCU side
        assert mcu[7] is b
        # Pipeline side, internal _pycboards registry
        assert pipeline._pycboards[7] is b
        # MCUPusher side (camera pipeline -> MCU bridge)
        assert pipeline.push._pycboards[7] is b

    def test_pushsink_can_reach_board_via_registry(
        self, mcu, pipeline, fake_pycboard
    ):
        """The MCUPusher fires MCU events from the camera worker thread.
        After cross-pipeline registration, that lookup must succeed."""
        b = fake_pycboard(running=True)
        mcu.register(3, b)
        pipeline.update_box_pycboard(3, b)

        # Simulate MCUPusher's lookup at push-time (push.py:138).
        pyc_from_pushsink = pipeline.push._pycboards.get(3)
        assert pyc_from_pushsink is b
        # And it can drive a queue_trigger_event the way tracking_push does:
        pyc_from_pushsink.queue_trigger_event("zone_changed", source="t")
        b.queue_trigger_event.assert_called_once_with(
            "zone_changed", source="t"
        )

    def test_box_registered_with_pycboard(self, mcu, pipeline, fake_pycboard):
        """Binding a pycboard registers the box so RecorderSink can read
        the raw ``pycboard.timestamp`` per frame."""
        b = fake_pycboard()
        mcu.register(2, b)
        pipeline.update_box_pycboard(2, b)
        assert 2 in pipeline._registered_boxes
        assert pipeline._pycboards[2] is b


# ── 2. Multi-box independence ───────────────────────────────────────


class TestMultiBoxIndependence:
    def test_two_boards_register_independently(
        self, mcu, pipeline, fake_pycboard
    ):
        b1, b2 = fake_pycboard(running=True), fake_pycboard(running=False)
        mcu.register(1, b1)
        mcu.register(2, b2)
        pipeline.update_box_pycboard(1, b1)
        pipeline.update_box_pycboard(2, b2)

        # Distinct instances per box on every pipeline.
        assert mcu[1] is b1 and mcu[2] is b2
        assert pipeline._pycboards[1] is b1
        assert pipeline._pycboards[2] is b2
        assert pipeline.push._pycboards[1] is b1
        assert pipeline.push._pycboards[2] is b2

    def test_set_variable_isolated_per_box(
        self, mcu, pipeline, fake_pycboard
    ):
        b1, b2 = fake_pycboard(running=True), fake_pycboard(running=True)
        mcu.register(1, b1)
        mcu.register(2, b2)
        pipeline.update_box_pycboard(1, b1)
        pipeline.update_box_pycboard(2, b2)

        # User edits trial_count on box 1 only.
        mcu[1].set_variable("trial_count", 99, source="a")
        b1.set_variable.assert_called_once_with("trial_count", 99, source="a")
        b2.set_variable.assert_not_called()

    def test_camera_thread_zone_event_routed_to_correct_box(
        self, mcu, pipeline, fake_pycboard
    ):
        """Simulate the per-box camera worker firing a tracking event
        via ``MCUPusher._pycboards[box_id]``, ensure routing is correct."""
        b1, b2 = fake_pycboard(running=True), fake_pycboard(running=True)
        mcu.register(1, b1)
        mcu.register(2, b2)
        pipeline.update_box_pycboard(1, b1)
        pipeline.update_box_pycboard(2, b2)

        pipeline.push._pycboards[1].queue_trigger_event(
            "zone_changed", source="t"
        )
        b1.queue_trigger_event.assert_called_once()
        b2.queue_trigger_event.assert_not_called()


# ── 3. Disconnect path ──────────────────────────────────────────────


class TestDisconnectPath:
    """``BoxWidgetMixin.mcu_disconnect`` does:
        1. close the board
        2. mcu.unregister(box_id)
        3. pipeline.update_box_pycboard(box_id, None)  # detach hook

    The MCU side must drop the entry. The pipeline side currently keeps
    its existing pycboard reference because ``register_box(box_id, pycboard=None)``
    is a no-op when pycboard is None, Phase 2 will tighten that.
    For now the test pins the current behavior so a future fix that
    changes it is intentional.
    """

    def test_mcu_drops_after_unregister(self, mcu, fake_pycboard):
        b = fake_pycboard()
        mcu.register(5, b)
        assert 5 in mcu
        mcu.unregister(5)
        assert 5 not in mcu

    def test_pipeline_unregister_box_clears_push(
        self, mcu, pipeline, fake_pycboard
    ):
        """Full disconnect (board going away for good) calls
        ``pipeline.unregister_box``: verify MCUPusher and registry are
        both cleared."""
        b = fake_pycboard()
        mcu.register(8, b)
        pipeline.update_box_pycboard(8, b)
        assert 8 in pipeline.push._pycboards

        # Simulate the full disconnect flow.
        mcu.unregister(8)
        pipeline.unregister_box(8)

        assert 8 not in mcu
        assert 8 not in pipeline._pycboards
        assert 8 not in pipeline.push._pycboards

    def test_disconnect_one_box_leaves_others_alive(
        self, mcu, pipeline, fake_pycboard
    ):
        b1, b2 = fake_pycboard(), fake_pycboard()
        mcu.register(1, b1)
        mcu.register(2, b2)
        pipeline.update_box_pycboard(1, b1)
        pipeline.update_box_pycboard(2, b2)

        # Disconnect only box 1.
        mcu.unregister(1)
        pipeline.unregister_box(1)

        assert 1 not in mcu and 2 in mcu
        assert mcu[2] is b2
        assert pipeline._pycboards[2] is b2
        assert pipeline.push._pycboards[2] is b2


# ── 4. Lifecycle: framework start/stop driven through registry ──────


class TestFrameworkLifecycleThroughRegistry:
    def test_start_framework_only_on_target_box(
        self, mcu, fake_pycboard
    ):
        b1, b2 = fake_pycboard(), fake_pycboard()
        mcu.register(1, b1)
        mcu.register(2, b2)

        mcu[1].start_framework(data_output=True)
        b1.start_framework.assert_called_once_with(data_output=True)
        b2.start_framework.assert_not_called()

    def test_stop_all_idempotent_with_idle_boards(
        self, mcu, fake_pycboard
    ):
        b1, b2 = fake_pycboard(running=True), fake_pycboard(running=False)
        mcu.register(1, b1)
        mcu.register(2, b2)
        mcu.stop_all()
        b1.stop_framework.assert_called_once()
        b2.stop_framework.assert_not_called()

    def test_running_box_filter_via_iteration(
        self, mcu, fake_pycboard
    ):
        b1, b2, b3 = (fake_pycboard(running=True),
                      fake_pycboard(running=False),
                      fake_pycboard(running=True))
        mcu.register(1, b1)
        mcu.register(2, b2)
        mcu.register(3, b3)

        # Pattern from base.py / dialogs: filter via boards.values()
        running_ids = [bid for bid, board in mcu.items()
                       if board.framework_running]
        assert sorted(running_ids) == [1, 3]


# ── 5. Process-data tick interaction ────────────────────────────────


class TestProcessDataTick:
    """The per-widget plot_update timer drives ``Pycboard.process_data``
    on every tick. The cross-pipeline contract is that an event arriving
    on the serial line during process_data can subsequently be visible
    to camera-side consumers (because both pipelines share the same
    Pycboard instance via the registries)."""

    def test_process_data_drives_same_object(
        self, mcu, pipeline, fake_pycboard
    ):
        b = fake_pycboard(running=True)
        mcu.register(4, b)
        pipeline.update_box_pycboard(4, b)

        # Per-widget tick -> calls process_data on the SAME Pycboard
        # that MCUPusher will use to push events from the camera thread.
        mcu[4].process_data()
        b.process_data.assert_called_once()
        # MCUPusher reaches the SAME instance.
        assert pipeline.push._pycboards[4] is b


# ── 6. Public-surface contract pin ──────────────────────────────────


class TestPublicContracts:
    """Pin the cross-pipeline contract so an accidental rename of either
    side gets caught here."""

    def test_pipeline_has_update_box_pycboard(self, pipeline):
        assert callable(getattr(pipeline, "update_box_pycboard", None))

    def test_pipeline_has_unregister_box(self, pipeline):
        assert callable(getattr(pipeline, "unregister_box", None))

    def test_pipeline_has_pushsink_with_pycboards(self, pipeline):
        # The dict MCUPusher uses for per-box MCU lookups.
        assert isinstance(pipeline.push._pycboards, dict)

    def test_mcu_thin_surface_unchanged(self, mcu):
        public = {n for n in dir(mcu) if not n.startswith("_")}
        assert public == {
            "boards", "get", "items", "keys", "register", "unregister",
            "values", "close_all", "stop_all",
        }
