"""
Pytest configuration and shared fixtures.

Provides common fixtures, mocks, and test utilities used across the test suite.
"""

import collections
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# board_tests/ and framework_tests/ are MicroPython scripts that run on
# the pyboard hardware (they import ``from pyControl.utility import *``
# and ``from devices import *``, which are MCU-side packages, not host).
# Skip them during host-side pytest collection so they don't show up as
# ``ERROR collecting`` noise.
collect_ignore_glob = [
    "board_tests/*.py",
    "framework_tests/*.py",
]


@pytest.fixture(autouse=True)
def _clean_data_dir_pollution():
    """Save-on-project pre-creates ``<code>/data/<project>/``, so tests that
    save projects pollute the real ``data/`` folder. Snapshot the dir before
    each test and remove anything new afterwards; pre-existing live data and
    log files survive because they exist before any test runs."""
    from source import paths as _app_paths
    data_root = Path(_app_paths.top_dir) / "data"
    if not data_root.is_dir():
        yield
        return
    before = set(p.name for p in data_root.iterdir())
    try:
        yield
    finally:
        try:
            for p in data_root.iterdir():
                if p.name not in before:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            p.unlink()
                        except OSError:
                            pass
        except Exception:
            pass


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Disable GUI warnings during tests
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# ============================================================================
# Directory and File Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_data_dir(temp_dir):
    """Create a test data directory structure"""
    data_dir = Path(temp_dir) / "test_data"
    data_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (data_dir / "recordings").mkdir(exist_ok=True)
    (data_dir / "logs").mkdir(exist_ok=True)
    (data_dir / "configs").mkdir(exist_ok=True)
    (data_dir / "metadata").mkdir(exist_ok=True)

    yield data_dir


@pytest.fixture
def sample_metadata_file(test_data_dir):
    """Create a sample cohort Excel file.

    Schema matches the current MetadataManager contract:
    required columns are ``Subject`` (string) and ``SetupID`` (int).
    Extra columns ride along for downstream tests.
    """
    data = {
        'Subject': ['SUB001', 'SUB002', 'SUB003'],
        'SetupID': [1, 2, 3],
        'Condition': ['Control', 'Treatment', 'Control'],
        'Age': [12, 15, 10],
        'Weight': [25.5, 28.3, 23.1]
    }
    df = pd.DataFrame(data)

    metadata_file = test_data_dir / "metadata" / "test_metadata.xlsx"
    df.to_excel(metadata_file, index=False)

    yield metadata_file


# ============================================================================
# Mock Objects for Hardware and GUI
# ============================================================================

@pytest.fixture
def mock_qapplication():
    """Mock QApplication for GUI tests"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_file_dialog():
    """Mock QFileDialog for file selection tests"""
    with patch('PySide6.QtWidgets.QFileDialog') as mock_dialog:
        mock_dialog.getSaveFileName.return_value = ('/tmp/test.json', '*.json')
        mock_dialog.getOpenFileName.return_value = ('/tmp/test.json', '*.json')
        yield mock_dialog


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_threads():
    """Ensure all threads are properly cleaned up after tests"""
    yield
    import threading
    # Wait for non-daemon threads to complete
    for thread in threading.enumerate():
        if thread != threading.current_thread() and not thread.daemon:
            thread.join(timeout=1.0)


@pytest.fixture(autouse=True)
def shutdown_leaked_pipelines():
    """Stop any Pipeline a test left running.

    A Pipeline's sinks own daemon worker threads, and the object graph is
    cyclic, so a leaked one is freed by the CYCLIC collector at an arbitrary
    later moment, potentially while those threads are still alive and
    touching its memory. On Windows that surfaces as heap corruption
    (0xc0000374) inside an unrelated test, which is close to undebuggable
    from where it lands.

    Shutting them down here keeps the failure at the test that leaked it,
    and keeps the collector's job harmless.
    """
    yield
    import sys
    mod = sys.modules.get("source.video.framebus.controller")
    if mod is None:
        return                      # nothing imported it, nothing to stop
    for pipe in mod.live_pipelines():
        try:
            pipe.shutdown()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def close_leaked_windows():
    """Close any parentless top-level window a test left open.

    Detached tabs and the Session Plot window deliberately have no Qt parent,
    so the operator can stack them behind the main window (see
    ``source/gui/window_behavior.py``). The cost is that Qt no longer destroys
    them with their opener: a test that creates one and only drops its
    reference leaves a live C++ widget whose Python wrapper is collected
    later, at an arbitrary point in an unrelated test. On Windows that is the
    same heap corruption (0xc0000374) the leaked-Pipeline fixture above exists
    to prevent.

    Same trade-off as that fixture: this is a safety net, so it hides leaks
    rather than failing on them. Turning it into an assertion is a one-line
    change if you would rather leaks be loud.
    """
    yield
    import sys
    mod = sys.modules.get("source.gui.window_behavior")
    if mod is None:
        return
    try:
        mod.close_independent_windows()
    except Exception:
        pass


def _live_top_levels():
    """(widget, cpp_pointer) for every top-level widget Qt currently holds.

    Identity is the C++ pointer, not the Python wrapper: PySide hands out a
    fresh wrapper for the same underlying widget on each call, so comparing
    wrappers, by ``id()`` or by set membership, silently treats every
    widget as new.
    """
    try:
        from PySide6 import QtWidgets
        from shiboken6 import getCppPointer, isValid
    except ImportError:
        return []
    app = QtWidgets.QApplication.instance()
    if app is None:
        return []
    out = []
    for w in app.topLevelWidgets():
        try:
            if isValid(w):
                out.append((w, getCppPointer(w)[0]))
        except (RuntimeError, TypeError):
            continue
    return out


# NOTE: defined LAST on purpose. Autouse fixtures tear down in reverse
# definition order, so this runs BEFORE the pipeline sweep above, once the
# widgets holding them are gone, the leaked Pipelines drop out of that
# WeakSet by themselves instead of being shut down again on every later test.
@pytest.fixture(autouse=True)
def destroy_widgets_created_by_this_test(request):
    """Fail a test that leaves top-level Qt widgets alive.

    A Qt widget is a C++ object. Dropping the last Python reference does not
    delete it, and ``deleteLater`` only *queues* a delete that never runs
    without an event loop, so a test that builds a dialog and returns leaks
    the entire widget tree for the rest of the session.

    Left alone this is quadratic, not merely untidy. Measured over the GUI
    suite, live top-level widgets climbed past 2500 and child widgets past
    35000, and because those widgets keep their ``Pipeline`` alive, the
    WeakSet in ``shutdown_leaked_pipelines`` grew with them, so every later
    test re-shut-down every pipeline every other test had leaked. The run
    degraded to minutes per test and never reached the end: a hang with no
    failing test to point at.

    That is why this asserts rather than quietly tidying up. The cost of a
    leak is paid by some unrelated test much later, so the leak has to be
    loud at the point it happens or it is undebuggable.

    The widgets are destroyed either way, BEFORE the assertion, a failing
    test must not also poison the rest of the run.

    Only widgets that appeared DURING the test are considered, so anything a
    session-scoped fixture owns is left alone. Mark a test
    ``@pytest.mark.leaks_widgets("why")`` to opt out.

    Everything new is destroyed, but only widgets of a class defined in this
    project are blamed. Qt makes parentless ``QFrame``/``QWidget`` top-levels
    of its own accord, combo popups, tooltip frames, and those appear and
    vanish outside any test's control; failing a test for one is a false
    positive it cannot act on. A leaked ``UnifiedTrackingDialog`` is the
    finding worth reporting, and its class lives under ``source.``.

    Deliberately does NOT call ``close()``. A close runs the widget's own
    ``closeEvent``, which is arbitrary application code:
    ``UnifiedTrackingDialog`` opens a modal "save zones?" prompt there, and
    driving that from a teardown with no event loop corrupted the heap
    (0xc0000374). ``deleteLater`` plus a DeferredDelete drain frees the C++
    side without running any of it, the widget's destructor still stops its
    own timers and children.
    """
    before = {ptr for _w, ptr in _live_top_levels()}
    yield
    leaked = [w for w, ptr in _live_top_levels() if ptr not in before]
    if not leaked:
        return

    kinds = collections.Counter()
    for w in leaked:
        try:
            cls = type(w)
            if cls.__module__.split(".")[0] in ("source", "tools"):
                kinds[cls.__name__] += 1
        except RuntimeError:
            continue

    # Destroy FIRST, so a failure here costs one test rather than the run.
    from PySide6 import QtCore, QtWidgets
    for w in leaked:
        try:
            w.deleteLater()
        except RuntimeError:
            pass          # C++ side already gone; nothing to free
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    if not kinds or request.node.get_closest_marker("leaks_widgets"):
        return

    listing = ", ".join(f"{n}x {name}" for name, n in kinds.most_common())
    raise AssertionError(
        f"this test left {sum(kinds.values())} top-level Qt widget(s) alive: "
        f"{listing}.\n"
        f"They have been destroyed so the rest of the run is unaffected, but "
        f"the test must clean up after itself, a leaked widget keeps its "
        f"whole tree, and any Pipeline it owns, alive for the session.\n"
        f"Fix: close/dispose the widget in a finally block or a fixture, "
        f"`w.close(); w.deleteLater(); QApplication.processEvents()`.\n"
        f"If the leak is genuinely intended, mark the test "
        f"`@pytest.mark.leaks_widgets(\"reason\")`.")


@pytest.fixture(autouse=True)
def _no_real_cameras_in_tests():
    """Every test sees a machine with no cameras, unless it says otherwise.

    Enumeration and the identity table are cached process-wide, so without
    this one test that touches real hardware hands the developer's webcams to
    every test after it: results then depend on ordering, on the machine, and
    on whether something was plugged in.

    The caches are seeded EMPTY rather than invalidated. Invalidating makes
    every test that enumerates re-walk the USB bus, measured at 4 minutes to
    14 for the suite, and leaves a window where a prewarm thread can refill
    them mid-test. An empty table is authoritative: nothing walks the bus, and
    identity lookups find nothing, which is what a test machine should look
    like. A test that wants cameras overrides the two caches itself.
    """
    from source.video.cameras import factory, identity
    identity._live_cache = []
    factory._enum_cache = []
    yield
    identity._live_cache = None
    factory._enum_cache = None
