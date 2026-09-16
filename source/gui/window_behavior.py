"""Secondary windows that the operator can put behind the main window.

The rule, and why it is not obvious
-----------------------------------
A ``QWidget`` created with a **parent** and given the ``Qt.Window`` flag is not
an independent window. On Windows it becomes an *owned* top-level (Win32
owner/owned relationship), and an owned window is permanently above its owner
in the z-order. No amount of clicking the main window will bring it forward:
the operator cannot send a detached tab or a plot window "back", and it covers
whatever is underneath for the rest of the session.

The same construction also makes some window managers treat the child as
input-blocking even when it is explicitly ``NonModal``, which is why the plot
window already carried a comment about forcing ``NonModal``, that addressed
the input symptom but not the stacking one, because both come from the
ownership, not the flags.

The fix is to give these windows **no Qt parent at all**. That costs two things
which this module supplies:

* **Lifetime.** With no parent, Qt will not delete the window when the main
  window goes away, and nothing keeps it alive either. Callers must hold a
  reference (they already do); this module additionally keeps a weak registry
  so shutdown can close whatever is still open.
* **Shutdown.** A parentless top-level does not close with its opener, and on
  some platforms a surviving visible window keeps the process alive. The main
  window's ``closeEvent`` calls :func:`close_independent_windows`.

Modal dialogs are deliberately NOT covered here. A modal dialog *should* block
the main window and sit above it; that is what modal means. This module is
only for windows the operator is meant to work alongside: detached tabs, the
Session Plot window, and anything similar added later.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator

from source.log import get_logger

logger = get_logger()

# Weak so a window that is closed and deleted drops out on its own; we never
# want this registry to be the thing keeping a window alive.
_registry: weakref.WeakSet = weakref.WeakSet()


def register_independent_window(win) -> None:
    """Track ``win`` so shutdown can close it.

    Call after constructing a parentless top-level. Safe to call twice.
    """
    try:
        _registry.add(win)
    except TypeError:
        # Not weak-referenceable, nothing to track, and not worth failing over.
        logger.debug("window_behavior: %r is not weak-referenceable", type(win))


def independent_windows() -> Iterator:
    """The still-live registered windows. Mainly for tests."""
    return iter(list(_registry))


def close_independent_windows() -> int:
    """Close every registered window. Returns how many were closed.

    Called from the main window's ``closeEvent``. Each close is guarded: these
    windows may already have been destroyed (``WA_DeleteOnClose``), in which
    case touching the Python wrapper raises ``RuntimeError``.
    """
    closed = 0
    for win in list(_registry):
        try:
            win.close()
            closed += 1
        except RuntimeError:
            # C++ side already gone; that is a successful outcome, not an error.
            continue
        except Exception as e:
            logger.debug("window_behavior: close failed for %r: %s", win, e)
    return closed
