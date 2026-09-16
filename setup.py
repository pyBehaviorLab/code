# setup.py, Cython extension build for pyBehaviorLab (unified operant + maze).
#
# Usage:
#   python setup.py build_ext --inplace
#
# Called automatically at startup by source/cython/build.py.
# If Cython is not installed, attempts to compile from pre-generated .c files.

import os
from setuptools import setup, Extension

try:
    import numpy as np
    numpy_include = np.get_include()
except ImportError:
    numpy_include = ""

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext_suffix = ".pyx" if USE_CYTHON else ".c"

extensions = [
    # Zone trigger geometry, used by zone_manager.py per-frame (both modes).
    Extension(
        "source.cython.zone_math",
        sources=[f"source/cython/zone_math{ext_suffix}"],
        include_dirs=[numpy_include] if numpy_include else [],
    ),
    # Image processing, illumination normalization + batch grayscale.
    Extension(
        "source.cython._image_ops",
        sources=[f"source/cython/_image_ops{ext_suffix}"],
        include_dirs=[numpy_include] if numpy_include else [],
    ),
    # Video overlay drawing, DLC keypoint extraction (maze).
    Extension(
        "source.cython.drawing_ops",
        sources=[f"source/cython/drawing_ops{ext_suffix}"],
        include_dirs=[numpy_include] if numpy_include else [],
    ),
]

if USE_CYTHON:
    extensions = cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    )

setup(
    name="pybehaviorlab_cython",
    ext_modules=extensions,
)
