"""Stats canvas package.

Public surface (both modes import from here):

    from source.stats import StatsCanvas

The canvas is the QWidget added to the operant + maze main windows as
the "Statistics" tab. Per-task config lives in either:

  * ``tasks/<family>/stats.json`` or ``config.json``: co-located with
    the task (``stats.json`` is what tools/init_stats.py writes; both
    are probed, ``stats.json`` first)
  * ``experiments/config/stats_templates/*.json``: shared template library
"""

from source.stats.canvas import (  # noqa: F401
    StatsCanvas,
    BoxStatisticsCalculator,
    StatisticsAutoDetector,
    StatisticsDataConsumer,
    TemplateLoader,
)

__all__ = [
    "StatsCanvas",
    "BoxStatisticsCalculator",
    "StatisticsAutoDetector",
    "StatisticsDataConsumer",
    "TemplateLoader",
]
