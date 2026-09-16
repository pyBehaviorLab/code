# source/cython/build.py
"""
Auto-build Cython extensions at startup if they are missing or stale.

Called from main.py BEFORE any imports that depend on Cython modules.
This ensures compiled .pyd/.so files exist and are newer than their .pyx
sources. If not, runs `python setup.py build_ext --inplace` automatically.

Fallback behavior:
    If Cython or NumPy is not installed, build is skipped with a warning.
    All consumer modules (tracker.py, zone_manager.py, video.py, pycboard.py,
    main_window.py) have try/except ImportError guards and fall back to
    equivalent pure-Python implementations, the app runs either way.
"""

import sys
import subprocess
import importlib
from pathlib import Path

# Root of the project (where setup.py lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Cython modules to check, must match extensions in setup.py
MODULES = [
    "source.cython.zone_math",
    "source.cython._image_ops",
    "source.cython.drawing_ops",
]

# Corresponding .pyx source files (relative to PROJECT_ROOT)
PYX_FILES = [
    "source/cython/zone_math.pyx",
    "source/cython/_image_ops.pyx",
    "source/cython/drawing_ops.pyx",
]


def _find_compiled(module_name: str) -> Path | None:
    """Locate the compiled .pyd/.so for a given module.

    Returns the Path to the compiled file, or None if the module
    cannot be imported (not yet compiled or import error).
    """
    try:
        mod = importlib.import_module(module_name)
        fpath = getattr(mod, "__file__", None)
        if fpath and Path(fpath).exists():
            return Path(fpath)
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def _is_stale(module_name: str, pyx_path: str) -> bool:
    """Check if a compiled module is missing or older than its .pyx source.

    Returns True if the module needs rebuilding:
      - Compiled file does not exist (never built)
      - .pyx source has a newer modification time than the compiled file
    """
    compiled = _find_compiled(module_name)
    if compiled is None:
        return True  # Not compiled at all
    pyx = PROJECT_ROOT / pyx_path
    if not pyx.exists():
        return False  # No source to compare, keep existing compiled file
    return pyx.stat().st_mtime > compiled.stat().st_mtime


def needs_build() -> bool:
    """Return True if any Cython module is missing or out of date."""
    return any(_is_stale(mod, pyx) for mod, pyx in zip(MODULES, PYX_FILES))


def build(verbose: bool = True) -> bool:
    """Invoke ``python setup.py build_ext --inplace`` as a subprocess.

    Returns True on success, False on failure. On failure, prints the
    last 20 lines of stderr for diagnostics (full output is often too long).
    """
    if verbose:
        print("[CYTHON] Building Cython extensions...")

    # ``--build-lib`` + ``--build-temp`` redirect setuptools' staging into
    # ``source/cython/_build/`` so the only artefacts at the project root are
    # the in-place ``.pyd`` files Python actually imports.
    try:
        result = subprocess.run(
            [
                sys.executable, "setup.py", "build_ext",
                "--inplace",
                "--build-lib",  "source/cython/_build/lib",
                "--build-temp", "source/cython/_build/temp",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            if verbose:
                print("[CYTHON] Build successful.")
            return True
        # On Windows a concurrent pyOperant/pyMaze instance can hold the
        # ``.pyd`` loaded as a DLL, so setuptools' overwrite step gets
        # WinError 5 (Access denied). If every required ``.pyd`` is already on
        # disk and importable, use the existing build.
        stderr = result.stderr or ""
        locked = ("Access is denied" in stderr) or ("WinError 5" in stderr)
        all_present = all(_find_compiled(mod) for mod in MODULES)
        if locked and all_present:
            print(
                "[CYTHON] Rebuild skipped, .pyd file locked by another "
                "Python process (concurrent pyOperant/pyMaze instance). "
                "Using existing compiled extensions."
            )
            return True
        print(f"[CYTHON] Build FAILED (exit code {result.returncode}):")
        # Truncate to last 20 lines, compiler output can be very long
        stderr_lines = stderr.strip().splitlines()
        for line in stderr_lines[-20:]:
            print(f"  {line}")
        return False
    except subprocess.TimeoutExpired:
        print("[CYTHON] Build timed out after 120s.")
        return False
    except FileNotFoundError:
        print("[CYTHON] Cannot find Python executable for build.")
        return False


def ensure_built(verbose: bool = True) -> bool:
    """Check all Cython modules and rebuild if any are stale.

    Call this once at application startup (from main.py) before importing
    any module that uses Cython extensions.

    Steps:
        1. Check if all .pyd/.so files exist and are up to date.
        2. If not, verify Cython and NumPy are installed (prerequisites).
        3. Run build(). On success, flush sys.modules so the next import
           picks up the freshly compiled versions instead of stale cached ones.

    Returns True if all modules are available after this call.
    """
    if not needs_build():
        if verbose:
            print("[CYTHON] All extensions up to date.")
        return True

    # Cython is required to compile .pyx -> .c -> .pyd
    try:
        import Cython
    except ImportError:
        print("[CYTHON] WARNING: Cython not installed. Extensions will not be built.")
        print("[CYTHON]   Install with: pip install cython")
        print("[CYTHON]   Falling back to pure-Python implementations.")
        return False

    # NumPy headers needed for _image_ops and drawing_ops compilation
    try:
        import numpy
    except ImportError:
        print("[CYTHON] WARNING: NumPy not installed. Cannot build extensions.")
        return False

    success = build(verbose=verbose)

    if success:
        # Evict stale module objects so the next `import` loads the new .pyd
        for mod_name in MODULES:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

    return success
