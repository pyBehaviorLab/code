"""Base class + discovery for optional Python hooks attached to MCU
analysis templates.

Convention (mirrors ``api_classes/`` in the rig GUI): a script lives in
``tools/offline_analysis/scripts/<id>/`` with two files:

  template.json, declarative spec (params, counters, columns, plots)
  analysis.py, optional; defines a subclass of OfflineAnalysisHook
                    matching ``template.json["hook_class"]``

When ``hook_class`` is omitted (template-only), the generic engine in
``template_engine.py`` runs the analysis using the declared counters +
calculations alone, no Python needed.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from .core import AnalysisResult

logger = logging.getLogger(__name__)


class OfflineAnalysisHook:
    """Subclass to add custom logic to a template-driven MCU analysis.

    Default behaviour mirrors the generic template engine (the engine
    actually instantiates this base class when no hook is declared), so
    overriding any single method works without breaking the rest.
    """

    template: Dict[str, Any] = {}   # injected by the engine before run()

    # ── Lifecycle ─────────────────────────────────────────────────────
    def load(self, files: List[Path]) -> pd.DataFrame:
        """Parse N input files into a tall dataframe.

        Default: lazy concat of TSV/CSV reads, adapters that need a
        smarter parser override this.
        """
        frames = []
        for fp in files:
            try:
                df = pd.read_csv(fp, sep="\t", comment="#", header=None,
                                 dtype=str, engine="python", on_bad_lines="skip")
                df["__file__"] = str(fp)
                frames.append(df)
            except Exception as e:
                logger.warning("load failed for %s: %s", fp, e)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]
                ) -> AnalysisResult:
        """Compute per-file metrics. Default applies the template's
        counters + calculations row-by-row."""
        from .template_engine import apply_counters_calculations
        summary = apply_counters_calculations(df, self.template, params)
        return AnalysisResult(
            success=True, summary_df=summary,
            metadata={"engine": "template-generic"},
        )

    def plots(self, result: AnalysisResult) -> List[Any]:
        """Return matplotlib figures. Default uses the template's plot
        declarations (see template_engine.render_plots)."""
        from .template_engine import render_plots
        return render_plots(result.summary_df, self.template)

    def summarise(self, results: List[AnalysisResult]) -> Optional[pd.DataFrame]:
        """Multi-session aggregate. Default: concatenate summary_df rows."""
        rows = [r.summary_df for r in results
                if r.summary_df is not None]
        if not rows:
            return None
        return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def load_hook_class(script_dir: Path, class_name: str
                    ) -> Optional[Type[OfflineAnalysisHook]]:
    """Import ``analysis.py`` next to ``template.json`` and return the
    named hook class, or None if not found."""
    py = script_dir / "analysis.py"
    if not py.is_file():
        return None
    mod_name = f"tools.offline_analysis.scripts.{script_dir.name}.analysis"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, py)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name, None)
        if cls is None:
            logger.warning("hook class %s not found in %s", class_name, py)
            return None
        if not issubclass(cls, OfflineAnalysisHook):
            logger.warning("hook class %s is not OfflineAnalysisHook", class_name)
            return None
        return cls
    except Exception as e:
        logger.error("Failed to load hook %s: %s", py, e)
        return None


__all__ = ["OfflineAnalysisHook", "load_hook_class"]
