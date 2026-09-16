"""Offline analysis package, standalone GUI for analysing saved sessions.

Launched as:
    python -m tools.offline_analysis.app

Shares the rig's data and presentation layers by importing them directly,
``source.video.zones`` (schema, geometry, coords, io), ``source.gui.theme``,
``source.gui.style_builders`` and ``source.gui.widgets.zone_overlay``. It used
to vendor private copies of all seven so the folder could be lifted out of the
repo and run standalone; that property was dropped deliberately because the
copies drifted (the analyzer kept a stale "success" green for months) and
because ``zone_geometry`` could not be shared at all once the rig's moved to a
Cython accelerator.

Still does NOT touch the live-rig subsystems: no ``source.communication``, no
``source.video.cameras``, no ``source.video.framebus``, no ``pyControl``. The
rig GUI launches this as a subprocess and the two apps share no live state,
that separation is about runtime, and it stands.
"""
