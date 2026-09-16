"""Disconnect must not block the GUI thread waiting for camera threads.

``stop_camera`` runs on the Disconnect button's click. Joining the capture
thread there froze the UI for up to 3 s per camera, and a camera mid-reconnect
(0.5 s retry sleeps around eight device opens) hits that ceiling routinely.
"""
import threading
import time

from source.video.cameras.capture import VideoManager


class _SlowThread:
    """A camera thread that takes a while to notice it was stopped."""

    def __init__(self, unwind_s=1.0):
        self.camera_id = 7
        self.stopped = False
        self._unwind_s = unwind_s
        self._done = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self.stopped:
            time.sleep(0.01)
        time.sleep(self._unwind_s)
        self._done.set()

    def stop(self):
        self.stopped = True

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self._done.is_set()

    def is_alive(self):
        return not self._done.is_set()


def _manager_with(thread, camera_id=7, setup_id=1):
    vm = VideoManager.__new__(VideoManager)
    vm.cameras = {camera_id: thread}
    vm.box_camera_map = {setup_id: camera_id}
    vm._segment_processors = {}
    return vm


def test_stop_camera_returns_before_the_thread_unwinds():
    thread = _SlowThread(unwind_s=1.0)
    vm = _manager_with(thread)

    t0 = time.monotonic()
    vm.stop_camera(1)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.3, (
        f"stop_camera blocked the caller for {elapsed:.2f}s")
    assert thread.stopped, "the camera thread was never told to stop"
    assert 7 not in vm.cameras
    assert 1 not in vm.box_camera_map


def test_the_thread_is_still_reaped_in_the_background():
    thread = _SlowThread(unwind_s=0.2)
    vm = _manager_with(thread)
    vm.stop_camera(1)
    assert thread.wait(timeout=3.0), "camera thread was abandoned, not reaped"


def test_a_camera_shared_by_two_boxes_stops_only_once():
    thread = _SlowThread(unwind_s=0.05)
    vm = _manager_with(thread)
    vm.box_camera_map[2] = 7

    vm.stop_camera(1)
    assert not thread.stopped, "stopped the device while box 2 still uses it"
    assert 7 in vm.cameras

    vm.stop_camera(2)
    assert thread.stopped
    assert 7 not in vm.cameras
