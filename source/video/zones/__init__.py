"""Zone geometry, scale calibration and MCU triggering.

``schema.py`` is the zone shape as stored on disk, ``geometry.py`` the
point-in-zone maths (Cython-accelerated with a pure-Python fallback),
``coords.py`` the pixel↔normalised conversions, and ``triggering.py`` the
zone-change → MCU-event mapping.

Deliberately empty of imports: ``geometry.py`` picks its Cython or pure-Python
backend at import time, and re-exporting from here would force that choice on
consumers that only want the schema. Import the module you need directly::

    from source.video.zones.schema import Zone

This file exists so setuptools' (non-namespace) package discovery ships the
package at all, without it, ``source.video.zones`` is absent from any built
wheel even though it imports fine from a source checkout.
"""
