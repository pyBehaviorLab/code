"""
Canonical path constants for pyBehaveTrack.

All default directories are relative to the repo root.
Import from here instead of computing repo_root in every file.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Repo root: source/core/paths.py → source/core → source → repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Standard directories
MODELS_DIR = REPO_ROOT / "models"
DLC_MODELS_DIR = MODELS_DIR / "dlc"
SLEAP_MODELS_DIR = MODELS_DIR / "sleap"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PROJECTS_DIR = EXPERIMENTS_DIR / "projects"
PROTOCOLS_DIR = EXPERIMENTS_DIR / "protocols"
TRACKING_CONFIGS_DIR = EXPERIMENTS_DIR / "tracking_configs"
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
TEMPLATES_DIR = CONFIG_DIR / "templates"
HARDWARE_DIR = REPO_ROOT / "hardware"
HARDWARE_CONFIGS_DIR = HARDWARE_DIR / "configs"
FIRMWARE_DIR = HARDWARE_DIR / "firmware"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Category → canonical default directory, for file-dialog start paths.
# Keeps every "Browse…" button opening in the right place (models →
# models/, configs → config/, projects → experiments/projects, …).
_DIALOG_DIRS = {
    "model": MODELS_DIR,
    "dlc_model": DLC_MODELS_DIR,
    "sleap_model": SLEAP_MODELS_DIR,
    "config": EXPERIMENTS_DIR,
    "camera_config": EXPERIMENTS_DIR,
    "tracking_config": TRACKING_CONFIGS_DIR,
    "project": PROJECTS_DIR,
    "protocol": PROTOCOLS_DIR,
    "zones": EXPERIMENTS_DIR,
    "data": DATA_DIR,
    "analysis": DATA_DIR,
    "metadata": DATA_DIR,
    "hardware_config": HARDWARE_CONFIGS_DIR,
    "firmware": FIRMWARE_DIR,
}


def dialog_dir(category: str, current: str = "") -> str:
    """Return the start directory for a file dialog of the given category.

    Prefers ``current`` (the value already in the field, its own folder for
    a file, or the path itself for a directory) when it exists on disk, so
    re-browsing lands where the user last pointed. Otherwise falls back to the
    canonical folder for ``category`` (created if missing). Always returns a
    usable string path.
    """
    if current:
        p = Path(current)
        if p.exists():
            return str(p if p.is_dir() else p.parent)
        if p.parent.exists():
            return str(p.parent)
    target = _DIALOG_DIRS.get(category, REPO_ROOT)
    try:
        ensure_dir(target)
    except OSError:
        return str(REPO_ROOT)
    return str(target)


def atomic_write_json(path, data: Any, *, indent: int = 2) -> None:
    """Write ``data`` as JSON to ``path``, atomically.

    Write to a temp file in the SAME directory, then ``os.replace``, which is
    atomic on both POSIX and Windows. The parent directory is created if it is
    missing, and the temp file is removed on any failure.

    Same-directory matters: ``os.replace`` is only atomic within one
    filesystem, so a temp file in the system temp dir would silently become a
    copy-then-delete and reintroduce the torn-write window this exists to
    close.

    Shared because all three of its callers persist things a session cannot be
    rebuilt without, camera capability caches, lens calibration profiles and
    rig extrinsics. Three identical copies of the routine that decides whether
    a half-written calibration can exist is three places to fix it.
    """
    payload = json.dumps(data, indent=indent, ensure_ascii=False)
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
