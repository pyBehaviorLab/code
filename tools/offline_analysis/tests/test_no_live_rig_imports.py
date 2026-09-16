"""The analyser depends on nothing in the rig.

It must be liftable: delete ``tools/offline_analysis`` and the rig is
untouched; copy the folder somewhere else and it still runs. That holds only if
the dependency points one way, so nothing here imports ``source.*`` at all.

Stated on its own, the rule costs the two halves their shared code, and the
vendored copies then drift: a ``success`` green that no longer matches the
rig's, a zone geometry that cannot follow the rig's move to Cython. Importing
the rig's modules instead fixes the drift and costs the independence.

So the copies stay and the drift is GUARDED rather than hoped away:
``test_vendor_matches_rig.py`` compares every vendored file against the module
it came from, and these tests hold the perimeter around it.

The one exception is the tracker seam. Re-tracking a recording is the single
thing the analyser genuinely cannot do alone, and copying a detector stack
would be absurd. So ``engine/trackers.py`` may reach for the rig's *tracking
package only*, never at module scope, and answers "there is no tracker, and
here is why" when it is absent. That keeps the folder runnable with no rig
while letting it re-track when one is beside it.
"""
from __future__ import annotations

import ast
import pathlib

from tools.offline_analysis.tests.test_vendor_matches_rig import ORIGINS

_ANALYZER = pathlib.Path(__file__).resolve().parents[1]
_REPO = pathlib.Path(__file__).resolve().parents[3]

#: The only file allowed to name the rig, and only for detectors.
_TRACKER_SEAM = "engine/trackers.py"


def _imported_modules(path: pathlib.Path):
    """(lineno, dotted_name) for every import in a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        # A relative import names nothing outside the package, so it cannot
        # reach the rig and is not evidence either way.
        elif isinstance(node, ast.ImportFrom) and not node.level:
            out.append((node.lineno, node.module or ""))
    return out


def _analyser_files():
    here = pathlib.Path(__file__).name
    return [p for p in _ANALYZER.rglob("*.py")
            if p.name != here and "tests" not in p.relative_to(_ANALYZER).parts]


def test_the_analyser_imports_nothing_from_the_rig():
    offenders = []
    for path in _analyser_files():
        rel = path.relative_to(_ANALYZER).as_posix()
        if rel == _TRACKER_SEAM:
            continue
        for lineno, mod in _imported_modules(path):
            if mod.split(".")[0] == "source":
                offenders.append(f"{rel}:{lineno} imports {mod}")
    assert not offenders, (
        "the analyser must stand alone, vendor what it needs into "
        "tools/offline_analysis/vendor/, or route it through "
        f"{_TRACKER_SEAM} if it is a detector:\n  " + "\n  ".join(sorted(offenders)))


def test_the_tracker_seam_only_reaches_for_trackers():
    """The exception is narrow on purpose. It may name the rig's tracking
    package and nothing else, not the camera, not the frame bus, not the GUI.
    """
    seam = _ANALYZER / "engine" / "trackers.py"
    for lineno, mod in _imported_modules(seam):
        if mod.split(".")[0] != "source":
            continue
        assert mod == "source.video.tracking" or mod.startswith("source.video.tracking."), (
            f"{_TRACKER_SEAM}:{lineno} imports {mod}; the seam exists for "
            "detectors only")


def test_the_tracker_seam_never_imports_at_module_scope():
    """A top-level import would make the rig required again: the analyser
    would fail to start without it, including every part that needs no
    detector at all."""
    seam = _ANALYZER / "engine" / "trackers.py"
    tree = ast.parse(seam.read_text(encoding="utf-8"))
    for node in tree.body:
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for mod in mods:
            assert mod.split(".")[0] != "source", (
                f"{_TRACKER_SEAM}:{node.lineno} imports {mod} at module scope; "
                "the analyser would then need the rig just to start")


def test_the_rig_does_not_import_the_analyser():
    """Launching it is fine; importing it would make the folder undeletable.

    The rig's own tests are excluded, and deliberately: a couple of them read a
    recording back through the analyser's parser to prove the recorder and the
    reader agree, which is a contract worth testing from that side.
    """
    offenders = []
    for path in (_REPO / "source").rglob("*.py"):
        if "tests" in path.parts:
            continue
        for lineno, mod in _imported_modules(path):
            if mod.startswith("tools.offline_analysis"):
                offenders.append(
                    f"{path.relative_to(_REPO).as_posix()}:{lineno} imports {mod}")
    assert not offenders, (
        "the rig must not depend on the analyser:\n  " + "\n  ".join(sorted(offenders)))


def test_the_vendored_copies_are_where_the_perimeter_expects_them():
    """A vendored module that is not under vendor/ is one nobody will think to
    check for drift."""
    vendor = _ANALYZER / "vendor"
    assert vendor.is_dir(), "tools/offline_analysis/vendor/ is missing"
    for name in sorted(ORIGINS):
        assert (vendor / name).exists(), f"vendor/{name} is missing"


def test_no_loose_copies_outside_the_vendor_package():
    """Copies scattered through the package go unnoticed long enough to
    drift; under vendor/ the drift guard can see them."""
    loose = []
    for path in _analyser_files():
        rel = path.relative_to(_ANALYZER)
        if rel.parts[0] == "vendor":
            continue
        if path.name in ORIGINS:
            loose.append(rel.as_posix())
    assert not loose, (
        f"copies of vendored modules outside vendor/: {sorted(loose)}; the "
        "drift guard only looks inside vendor/")
