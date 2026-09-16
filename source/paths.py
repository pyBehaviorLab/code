"""Process-wide on-disk path resolver.

``top_dir`` is the pyBehaviorLab project root (the parent of ``source/``).
Every reusable config/data folder constant below is derived from that
anchor, so the whole project tree can move to another machine and every
stored path still resolves.

Paths saved to disk are forward-slash strings relative to top_dir (see
``relpath_for_storage``); paths passed around in memory are absolute.

Source-level utilities (``log``, ``datetime_formats``, ``paths``) live
together here; the on-disk "config" folder is ``experiments/config/``.
"""

import os

# top_dir = parent of source/ (i.e. the project root).
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Core directories, derived from top_dir.
# ---------------------------------------------------------------------------
# Every reusable JSON config + template lives under ``experiments/config/``;
# per-user projects live under ``experiments/projects/``.

framework_dir        = os.path.join(top_dir, 'pyControl')
devices_dir          = os.path.join(top_dir, 'devices')
tasks_dir            = os.path.join(top_dir, 'tasks')
data_dir             = os.path.join(top_dir, 'data')
models_dir           = os.path.join(top_dir, 'models')
hardware_dir         = os.path.join(top_dir, 'hardware_definitions')
tools_dir            = os.path.join(top_dir, 'tools')

experiments_dir      = os.path.join(top_dir, 'experiments')
projects_dir         = os.path.join(experiments_dir, 'projects')

# All reusable JSON config + templates land under experiments/config/.
config_dir           = os.path.join(experiments_dir, 'config')
tracking_configs_dir = os.path.join(config_dir, 'tracking')
stats_templates_dir   = os.path.join(config_dir, 'stats_templates')

# Single source of truth for the live application settings file.
settings_file        = os.path.join(config_dir, 'settings.json')
