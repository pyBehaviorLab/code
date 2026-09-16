"""Base classes shared by every analysis domain.

The two domains (MCU log analysis, video-tracking analysis) keep separate
adapters and script registries, but they all return the same AnalysisResult
shape and use the same ParameterDef / AnalysisScript / ScriptRegistry
infrastructure for plug-in analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InputType(Enum):
    """Type of input data required by the script."""
    SINGLE_FILE = "single_file"
    MULTIPLE_FILES = "multiple_files"
    FOLDER = "folder"


class ParamType(Enum):
    """Type of parameter."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    CHOICE = "choice"


@dataclass
class ParameterDef:
    """Definition of a script parameter."""
    name: str
    label: str
    param_type: ParamType
    default: Any
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None


@dataclass
class AnalysisScript:
    """Definition of an analysis script."""
    id: str
    name: str
    description: str
    input_type: InputType
    file_filter: str  # e.g. "*.txt *.tsv *.csv"
    parameters: List[ParameterDef] = field(default_factory=list)
    has_summary_table: bool = False
    has_detail_table: bool = False
    has_plots: bool = False
    has_export: bool = False


class ScriptRegistry:
    """Registry of available analysis scripts.

    Each domain (mcu, video) gets its own subclass with its own _scripts
    dict, so registrations in one domain don't pollute the other.
    """

    _scripts: Dict[str, AnalysisScript] = {}

    @classmethod
    def register(cls, script: AnalysisScript) -> None:
        cls._scripts[script.id] = script

    @classmethod
    def get(cls, script_id: str) -> Optional[AnalysisScript]:
        return cls._scripts.get(script_id)

    @classmethod
    def all(cls) -> List[AnalysisScript]:
        return list(cls._scripts.values())

    @classmethod
    def ids(cls) -> List[str]:
        return list(cls._scripts.keys())


@dataclass
class AnalysisResult:
    """Canonical analysis-result payload returned by every adapter.

    Both MCU and video adapters produce this exact shape so GUI tabs can render
    results without knowing which domain ran them.
    """
    success: bool = False
    error_message: str = ""
    summary_df: Optional[Any] = None    # pandas.DataFrame (top-level rollup)
    detail_df: Optional[Any] = None     # pandas.DataFrame (per-row detail)
    figures: List[Any] = field(default_factory=list)   # matplotlib.Figure list
    metadata: Dict[str, Any] = field(default_factory=dict)
