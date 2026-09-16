"""Application settings (experiments/config/settings.json).

Qt-free, importable from the communication and config layers without
dragging in the GUI package.

Functions:
- get_setting(), set_setting(): Application settings
- user_folder(): Get user folder paths
- get/set_mcu_display_mode(): per-machine MCU label default
"""

import json
from pathlib import Path
from source.log import get_logger

logger = get_logger()

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

_default_settings = {
    "folders": {
        "api_classes": "api_classes",
        "controls_dialogs": "controls_dialogs",
        "devices": "devices",
        "hardware_definitions": "hardware_definitions",
        "data": "data",
        "experiments": "experiments",
        "tasks": "tasks"
    },
    # No "plotting" or "camera" block: every key in both was written here and
    # read nowhere. Plot history lengths live in gui/plotting.py; camera
    # defaults live in the per-camera CameraConfig the pipeline owns. Same for
    # GUI.ui_font_size. Settings with no reader are not configuration, they
    # are a promise the app does not keep.
    "GUI": {
        "log_font_size": 9,
        "theme": "system"
    },
    "display": {
        # Cap the per-tile display rate. null = follow the camera; an int
        # caps display refresh (capture is unaffected). Used by the GUI
        # display tick (``base.py::_paint_streaming_cameras_once``, driven
        # by ``process_timer``).
        "max_fps": None,
    },
    "video": {
        # Encoder preference for recording.
        #   use_gpu: "auto" (use hardware encoder if present), true (force),
        #            false (CPU only, predictable, no NVENC session cap).
        #   allow_cpu_fallback: when a hardware encoder is present but its
        #            writer won't open (typically the consumer-GPU NVENC
        #            concurrent-session cap being exceeded on a many-box
        #            rig), record on CPU libx264 so the box still records
        #            rather than losing video. false = strict hardware-only.
        #   prefer_hevc: use H.265 (smaller files) where available.
        #   quality_crf: constant-quality factor 0-51 (lower = better/larger).
        #            23 is a visually-lossless default for behaviour video.
        "use_gpu": "auto",
        "allow_cpu_fallback": True,
        "prefer_hevc": False,
        "quality_crf": 23,
    },
    "mcu": {
        # How MCUs are LABELLED in the picker / per-box COM field. One of
        # "hashed" (djb2 of USB serial, same on every PC/OS, default),
        # "native" (COMn / /dev/ttyACMn), "serial" (raw USB serial). This is
        # the per-machine default; a project may pin its own via
        # Meta.mcu_display_mode. Binding is always by serial regardless.
        "display_mode": "hashed",
    }
}

from source import paths as _paths
_settings_file = Path(_paths.settings_file)


def _migrate_legacy_settings_file() -> None:
    """One-time relocate of ``config/settings.json`` →
    ``experiments/config/settings.json``.

    If the new file is missing and one exists at the legacy path, copy it
    across and delete the old. Idempotent, no-op once relocated.
    """
    if _settings_file.exists():
        return
    legacy = Path(_paths.top_dir) / "config" / "settings.json"
    if not legacy.exists():
        return
    try:
        _settings_file.parent.mkdir(parents=True, exist_ok=True)
        _settings_file.write_bytes(legacy.read_bytes())
        try:
            legacy.unlink()
        except OSError:
            pass
        logger.info(
            "Migrated legacy settings.json from %s to %s",
            legacy, _settings_file,
        )
    except Exception as e:
        logger.warning(
            "settings.json migration failed (%s); keeping legacy file at %s",
            e, legacy,
        )


# settings.json parse cache: (st_mtime_ns, parsed dict). get_setting runs
# several times per record start (and from 1 Hz refresh paths), re-reading
# and re-parsing the file each call was pure I/O overhead.
_settings_cache = None


def _load_user_settings():
    """Parsed settings.json, cached by file mtime; None when absent/bad."""
    global _settings_cache
    try:
        mtime = _settings_file.stat().st_mtime_ns
    except OSError:
        _settings_cache = None
        return None
    if _settings_cache is not None and _settings_cache[0] == mtime:
        return _settings_cache[1]
    try:
        with open(_settings_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        return None
    _settings_cache = (mtime, data)
    return data


def get_setting(setting_type, setting_name=None, want_default=False):
    """Get a setting value from config or default."""
    if want_default:
        if setting_name:
            return _default_settings.get(setting_type, {}).get(setting_name)
        else:
            return _default_settings.get(setting_type, {})

    _migrate_legacy_settings_file()
    user_settings = _load_user_settings()
    if user_settings is not None:
        if setting_name:
            return user_settings.get(setting_type, {}).get(
                setting_name,
                _default_settings.get(setting_type, {}).get(setting_name)
            )
        return user_settings.get(setting_type,
                                 _default_settings.get(setting_type, {}))

    if setting_name:
        return _default_settings.get(setting_type, {}).get(setting_name)
    else:
        return _default_settings.get(setting_type, {})


def set_setting(setting_type, setting_name, value):
    """Set a setting value and save to config file."""
    _migrate_legacy_settings_file()
    _settings_file.parent.mkdir(parents=True, exist_ok=True)

    if _settings_file.exists():
        try:
            with open(_settings_file, 'r') as f:
                settings = json.load(f)
        except Exception:
            settings = _default_settings.copy()
    else:
        settings = _default_settings.copy()

    if setting_type not in settings:
        settings[setting_type] = {}
    settings[setting_type][setting_name] = value

    try:
        with open(_settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Setting saved: {setting_type}.{setting_name} = {value}")
        # Refresh the parse cache so a read straight after the write sees
        # the new value even if the filesystem's mtime is coarse.
        global _settings_cache
        try:
            _settings_cache = (_settings_file.stat().st_mtime_ns, settings)
        except OSError:
            _settings_cache = None
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")


def user_folder(folder_name):
    """Get the path to a user folder."""
    folder_path = get_setting("folders", folder_name)
    path = Path(folder_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created folder: {folder_path}")
    return str(path)


def get_mcu_display_mode():
    """Per-machine default for how MCUs are labelled (hashed/native/serial)."""
    return get_setting("mcu", "display_mode") or "hashed"


def set_mcu_display_mode(mode):
    """Persist the per-machine MCU label default."""
    set_setting("mcu", "display_mode", mode)
