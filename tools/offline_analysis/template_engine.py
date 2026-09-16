"""Template engine, reads ``template.json``, validates, executes.

A template declares (per ``config/stats_templates/*.json`` schema,
extended with offline-only fields):

  id, name, version, description, auto_detect_keywords
  input        {type: SINGLE_FILE|MULTIPLE_FILES|FOLDER, file_filter, min_files}
  parameters   [{name, label, type: INT|FLOAT|STRING|BOOL|CHOICE,
                 default, min, max, choices, description}]
  counters     {name: {type: "print", match: "<string>"}}
  calculations {metric: "<safe python expr>"}
  columns      [{key, label, format?}]
  plots        [{name, type: "circular"|"bar"|"line", metric, ...}]
  hook_class   <optional Python class name in analysis.py>

This module does:
  * load + validate a ``template.json``
  * build a :class:`ScriptTemplate` you can hand to the UI
  * generic execution (no hook): counters + calculations + columns
  * generic plot rendering
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .core import (
    AnalysisScript, InputType, ParameterDef, ParamType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaded template
# ---------------------------------------------------------------------------

@dataclass
class ScriptTemplate:
    """Parsed template.json with optional resolved Python hook."""
    id: str
    name: str
    version: str
    description: str
    script_dir: Path
    raw: Dict[str, Any]
    has_hook: bool = False
    hook_class_name: str = ""
    # Mirror AnalysisScript so the UI can present it uniformly.
    as_script: Optional[AnalysisScript] = None
    readme: str = ""

    def to_analysis_script(self) -> AnalysisScript:
        return self.as_script or _build_analysis_script(self.raw)


# ---------------------------------------------------------------------------
# Discovery + load
# ---------------------------------------------------------------------------

def discover_templates(scripts_dir: Optional[Path] = None
                       ) -> List[ScriptTemplate]:
    """Walk ``tools/offline_analysis/scripts/*/template.json`` and load each."""
    if scripts_dir is None:
        scripts_dir = Path(__file__).resolve().parent / "scripts"
    out: List[ScriptTemplate] = []
    if not scripts_dir.is_dir():
        return out
    for sub in sorted(scripts_dir.iterdir()):
        if not sub.is_dir():
            continue
        tpl_path = sub / "template.json"
        if not tpl_path.is_file():
            continue
        try:
            tpl = load_template(tpl_path)
            out.append(tpl)
        except Exception as e:
            logger.error("Bad template %s: %s", tpl_path, e)
    return out


def load_template(path: Path) -> ScriptTemplate:
    """Load + lightly validate a single ``template.json``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "id" not in raw or "name" not in raw:
        raise ValueError(f"{path}: template missing id/name")
    script = _build_analysis_script(raw)
    readme = ""
    rp = path.parent / "README.md"
    if rp.is_file():
        try:
            readme = rp.read_text(encoding="utf-8")
        except Exception:
            readme = ""
    return ScriptTemplate(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        version=raw.get("version", "1.0"),
        description=raw.get("description", ""),
        script_dir=path.parent,
        raw=raw,
        has_hook=bool(raw.get("hook_class")),
        hook_class_name=raw.get("hook_class", ""),
        as_script=script,
        readme=readme,
    )


def _build_analysis_script(raw: Dict[str, Any]) -> AnalysisScript:
    inp = raw.get("input", {})
    itype = {
        "SINGLE_FILE":    InputType.SINGLE_FILE,
        "MULTIPLE_FILES": InputType.MULTIPLE_FILES,
        "FOLDER":         InputType.FOLDER,
    }.get(inp.get("type", "MULTIPLE_FILES"), InputType.MULTIPLE_FILES)
    params: List[ParameterDef] = []
    for p in raw.get("parameters", []) or []:
        ptype = {
            "INT": ParamType.INT, "FLOAT": ParamType.FLOAT,
            "STRING": ParamType.STRING, "BOOL": ParamType.BOOL,
            "CHOICE": ParamType.CHOICE,
        }.get(p.get("type", "STRING").upper(), ParamType.STRING)
        params.append(ParameterDef(
            name=p["name"],
            label=p.get("label", p["name"]),
            param_type=ptype,
            default=p.get("default"),
            description=p.get("description", ""),
            min_value=p.get("min"),
            max_value=p.get("max"),
            choices=p.get("choices"),
        ))
    cols  = raw.get("columns", []) or []
    plots = raw.get("plots", []) or []
    return AnalysisScript(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        description=raw.get("description", ""),
        input_type=itype,
        file_filter=inp.get("file_filter", "*"),
        parameters=params,
        has_summary_table=bool(cols),
        has_detail_table=False,
        has_plots=bool(plots),
        has_export=True,
    )


# ---------------------------------------------------------------------------
# Execution, generic counters + calculations
# ---------------------------------------------------------------------------

_SAFE_FUNCS = {
    "abs": abs, "max": max, "min": min, "sum": sum, "len": len,
    "round": round, "int": int, "float": float, "str": str,
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
}


def _safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    """Evaluate a calculation expression in a sandboxed namespace.

    Mirrors the live-stats ``BoxStatisticsCalculator``, restricts builtins
    so a template author can't shell out from a JSON file."""
    try:
        return eval(expr, {"__builtins__": {}}, {**_SAFE_FUNCS, **env})
    except ZeroDivisionError:
        return 0
    except Exception as e:
        logger.debug("calc '%s' failed: %s", expr, e)
        return None


def _count_matches_in_file(path: Path, counters: Dict[str, Dict[str, Any]]
                           ) -> Dict[str, int]:
    """Stream-count counter matches per file. Counters of type 'print'
    match any line whose text contains the substring."""
    counts = {name: 0 for name in counters}
    if not path.is_file():
        return counts
    matchers = []
    for name, spec in counters.items():
        if spec.get("type", "print") == "print":
            m = spec.get("match", "")
            if m:
                matchers.append((name, m))
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for name, needle in matchers:
                    if needle in line:
                        counts[name] += 1
    except Exception as e:
        logger.warning("count failed for %s: %s", path, e)
    return counts


def apply_counters_calculations(df: pd.DataFrame,
                                template: Dict[str, Any],
                                params: Dict[str, Any]) -> pd.DataFrame:
    """Build the summary dataframe from per-file counter rows.

    ``df`` is whatever ``hook.load()`` returned. The default loader stuffs
    the source path into ``__file__``; we group by that and re-count from
    the original file (faster + correct than reparsing the concatenated df).
    """
    counters = template.get("counters", {}) or {}
    calculations = template.get("calculations", {}) or {}
    columns = template.get("columns", []) or []
    rows: List[Dict[str, Any]] = []
    if "__file__" not in df.columns or df.empty:
        # Nothing to group on, return empty.
        return pd.DataFrame(columns=[c.get("key", "") for c in columns])
    for fp_str, _grp in df.groupby("__file__"):
        fp = Path(fp_str)
        counts = _count_matches_in_file(fp, counters) if counters else {}
        env: Dict[str, Any] = {**counts, **params, "file": fp.name}
        # subject/box/date pulled from filename if we can
        env.update(_parse_mcu_filename(fp.name))
        for metric, expr in calculations.items():
            env[metric] = _safe_eval(str(expr), env)
        rows.append(env)
    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    # Restrict + order columns per template, formatting numerics if asked
    if columns:
        keys = [c["key"] for c in columns if c.get("key") in df_out.columns]
        df_out = df_out.reindex(columns=keys)
        # Renames + formats
        renames: Dict[str, str] = {}
        for c in columns:
            k = c["key"]
            if c.get("label") and k in df_out.columns:
                renames[k] = c["label"]
            fmt = c.get("format")
            if fmt and k in df_out.columns:
                try:
                    df_out[k] = df_out[k].apply(
                        lambda v, _fmt=fmt: _fmt.format(v) if v is not None else "")
                except Exception:
                    pass
        if renames:
            df_out = df_out.rename(columns=renames)
    return df_out


_MCU_RE = re.compile(
    r"^(?P<subject>.+?)-Box(?P<box>\d+)-"
    r"(?P<date>\d{4}-\d{2}-\d{2})-(?P<hms>\d{6})\.tsv$"
)


def _parse_mcu_filename(name: str) -> Dict[str, str]:
    m = _MCU_RE.match(name)
    if not m:
        return {"subject_id": "", "box_id": "", "session_date": "",
                "session_time": "", "session": name}
    return {
        "subject_id":   m.group("subject"),
        "box_id":       m.group("box"),
        "session_date": m.group("date"),
        "session_time": m.group("hms"),
        "session":      f"{m.group('subject')}-{m.group('date')}",
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def render_plots(summary_df: Optional[pd.DataFrame],
                 template: Dict[str, Any]) -> List[Any]:
    """Build matplotlib figures from the template's plot declarations."""
    if summary_df is None or summary_df.empty:
        return []
    figures: List[Any] = []
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib unavailable: %s", e)
        return []
    for spec in template.get("plots", []) or []:
        kind = (spec.get("type") or "bar").lower()
        metric = spec.get("metric")
        title = spec.get("name") or metric or ""
        if not metric:
            continue
        # Match against either raw metric or its label
        col = metric
        if col not in summary_df.columns:
            for c in summary_df.columns:
                if str(c).lower() == metric.lower():
                    col = c
                    break
            else:
                continue
        try:
            if kind == "bar":
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(range(len(summary_df)),
                       pd.to_numeric(summary_df[col], errors="coerce"))
                ax.set_title(title); ax.set_ylabel(spec.get("ylabel", metric))
                figures.append(fig)
            elif kind == "line":
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(pd.to_numeric(summary_df[col], errors="coerce"),
                        marker="o")
                ax.set_title(title); ax.set_ylabel(spec.get("ylabel", metric))
                figures.append(fig)
            elif kind == "circular":
                vals = pd.to_numeric(summary_df[col], errors="coerce").dropna()
                pct = float(vals.mean()) if not vals.empty else 0.0
                fig, ax = plt.subplots(figsize=(3, 3),
                                       subplot_kw={"projection": "polar"})
                theta = (pct / 100.0) * 2 * math.pi
                ax.barh(0, theta, color="#a855f7")
                ax.set_yticks([]); ax.set_xticks([])
                ax.set_title(f"{title}: {pct:.1f}%")
                figures.append(fig)
            else:
                logger.debug("unknown plot type '%s'", kind)
        except Exception as e:
            logger.warning("plot '%s' failed: %s", title, e)
    return figures


# ---------------------------------------------------------------------------
# Run a template end-to-end
# ---------------------------------------------------------------------------

def run_template(tpl: ScriptTemplate, files: List[str],
                 params: Dict[str, Any]):
    """Execute a template (with or without hook). Returns AnalysisResult."""
    from .analysis_hook import OfflineAnalysisHook, load_hook_class
    cls: Optional[type] = None
    if tpl.has_hook:
        cls = load_hook_class(tpl.script_dir, tpl.hook_class_name)
    if cls is None:
        cls = OfflineAnalysisHook
    hook = cls()
    hook.template = tpl.raw
    paths = [Path(f) for f in files]
    try:
        df = hook.load(paths)
        result = hook.compute(df, params)
        if not result.figures:
            try:
                result.figures = hook.plots(result) or []
            except Exception as e:
                logger.warning("hook.plots failed: %s", e)
        return result
    except Exception as e:
        logger.exception("template '%s' run failed", tpl.id)
        from .core import AnalysisResult
        return AnalysisResult(success=False, error_message=str(e))


__all__ = [
    "ScriptTemplate", "discover_templates", "load_template",
    "apply_counters_calculations", "render_plots", "run_template",
]
