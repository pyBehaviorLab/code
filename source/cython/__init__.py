"""Cython acceleration package for pyBehaviorLab.

Provides optional compiled extensions for performance-critical paths.
Every function has a pure-Python fallback, the app runs identically
without Cython compiled, just slower.

Consumers import the compiled symbols directly from the submodules and
guard each call site with a local availability flag (set when the import
succeeds), e.g.::

    try:
        from source.cython.zone_math import cross_line_side
        _HAS_ZONE_MATH = True
    except ImportError:
        _HAS_ZONE_MATH = False

Modules:
    zone_math, point/segment/line geometry (zones triggering)
    _image_ops, illumination normalize + batch BGR→gray (tracking/capture)
    drawing_ops, DLC keypoint extraction for the video overlay
"""
