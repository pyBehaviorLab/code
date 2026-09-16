"""The recordings in ``data/``, and only those.

A recursive glob for ``*_video_data.txt`` also finds what the analyser itself
wrote: every re-track leaves one inside ``<stem>.pbanalysis/poses/``. Those are
artifacts, not recordings, and tests that treat them as recordings assert
things about this analyser's own output while claiming to assert them about
the rig's.

One definition, imported by every test that needs it, so a new test cannot
quietly go back to globbing everything.
"""
from __future__ import annotations

import glob
import os
from typing import List

#: The suffix the bundle folder carries. Anything under one is derived.
BUNDLE_SUFFIX = ".pbanalysis"

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "data")


def recordings(root: str = "") -> List[str]:
    """Session files a rig wrote, newest-agnostic, sorted for stability."""
    base = root or _ROOT
    found = glob.glob(os.path.join(base, "**", "*_video_data.txt"),
                      recursive=True)
    return sorted(p for p in found if not _is_derived(p))


def _is_derived(path: str) -> bool:
    """Whether this file is something the analyser produced."""
    parts = os.path.normpath(path).split(os.sep)
    return any(part.endswith(BUNDLE_SUFFIX) for part in parts)
