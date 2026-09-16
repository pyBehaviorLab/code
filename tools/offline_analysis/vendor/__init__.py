"""Copies of the rig modules the analyser needs, so it needs the rig for nothing.

The offline analyser is meant to be liftable: delete ``tools/offline_analysis``
and the rig is untouched, and copy it elsewhere and it still runs. That only
holds if the dependency points one way, so nothing under
``tools/offline_analysis`` imports ``source.*``: the perimeter test enforces it.

The cost of this is drift, and it is a cost we have paid before: an earlier
vendoring went stale (a ``success`` green that no longer matched the rig's for
months), which is why the blanket rule was dropped the first time. What makes
it safe now is ``tests/test_vendor_matches_rig.py``, which fails when a
vendored file stops agreeing with the module it was taken from while both live
in this repository. Away from the repo the guard simply does not run, which is
the right behaviour for a copy that has been deliberately lifted out.

Vendored here, and nothing else:

    theme.py             <- source/gui/theme.py
    style_builders.py    <- source/gui/style_builders.py
    zone_editor.py       <- source/gui/widgets/zone_editor.py
    numeric_line_edit.py <- source/gui/widgets/numeric_line_edit.py
    zone_overlay.py      <- source/gui/widgets/zone_overlay.py
    zone_coords.py       <- source/video/zones/coords.py
    zone_geometry.py     <- source/video/zones/geometry.py  (Cython path dropped)
    zone_io.py           <- source/video/zones/io.py

The zone editor is here for a reason worth stating: a zone drawn while
analysing has to obey the same rules as a zone drawn in the tracking dialog
before the session, same shapes, same scale line, and above all the same
stored form, normalized [0, 1] points with the frame size beside them, so a
zone does not move when the resolution does. The only way to be sure of that
is for it to be the same code, which makes the drift guard load-bearing
rather than tidy.

``icons/`` holds the two SVGs ``style_builders`` resolves relative to its own
file (the checkbox tick and the combo chevron). They travel with the module
rather than being pointed at from outside it, so the vendored copy runs
unmodified, which is the whole idea of a copy. Without them Qt silently draws
checkboxes and dropdowns with no glyph and says so only on stderr.
"""
