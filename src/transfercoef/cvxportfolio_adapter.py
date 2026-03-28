from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import ProjectPaths

class CvxportfolioImportError(ImportError):
    """Raised when the local cvxportfolio source tree cannot be imported."""

def resolve_local_cvxportfolio_root(project_root: str | Path) -> Path:
    """Resolve the vendered cvxportfolio source root for this project."""

    return ProjectPaths.from_project_root(project_root).local_cvxportfolio_root

def ensure_local_cvxportfolio_on_path(project_root: str | Path) -> Path:
    """Add the local cvxportfolio source directory to ``sys.path`` if needed."""

    source_root = resolve_local_cvxportfolio_root(project_root)
    if not source_root.exists():
        raise CvxportfolioImportError(
            f"Local cvxportfolio source not found at {source_root}"
        )
    
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
    
    return source_root

def import_cvxportfolio(project_root: str | Path) -> ModuleType:
    """Import the local cvxportfolio from the local vendored source tree."""

    ensure_local_cvxportfolio_on_path(project_root)
    try:
        return importlib.import_module("cvxportfolio")
    except Exception as exc:
        raise CvxportfolioImportError(
            "Unable to import local cvxportfolio.  Ensure its Python dependencies "
            "are available in the target environment."
        ) from exc

def get_yahoo_finance_class(project_root: str | Path) -> type[Any]:
    """Return the local ``cvxportfolio.YahooFinance`` class."""

    cvxportfolio = import_cvxportfolio(project_root)
    return cvxportfolio.YahooFinance

def get_single_period_optimization_class(project_root: str | Path) -> type[Any]:
    """Return the local ``cvxportfolio.SinglePeriodOptimization`` class."""

    cvxportfolio = import_cvxportfolio(project_root)
    return cvxportfolio.SinglePeriodOptimization

def get_market_simulator_class(project_root: str | Path) -> type[Any]:
    """Return the local ``cvxportfolio.MarketSimulator`` class."""

    cvxportfolio = import_cvxportfolio(project_root)
    return cvxportfolio.MarketSimulator

def get_common_constraint_classes(project_root: str | Path) -> dict[str, type[Any]]:
    """Return commonly used local constraint classes for scenario construction."""

    cvxportfolio = import_cvxportfolio(project_root)
    return {
        "LongOnly": cvxportfolio.LongOnly,
        "leverageLimit": cvxportfolio.LeverageLimit,
        "MaxWeights": cvxportfolio.MaxWeights,
        "MinWeights": cvxportfolio.MinWeights,
        "TurnoverLimit": cvxportfolio.TurnoverLimit,
        "DollarNeutral": cvxportfolio.DollarNeutral,
    }

def get_common_objective_classes(project_root: str | Path) -> dict[str, type[Any]]:
    """Return commonly used local objective classes for portfolio construction."""

    cvxportfolio = import_cvxportfolio(project_root)
    return {
        "ReturnForecast": cvxportfolio.ReturnForecast,
        "CashReturn": cvxportfolio.CashReturn,
        "FullCovariance": cvxportfolio.FullCovariance,
        "DiagonalCovariance": cvxportfolio.DiagonalCovariance,
        "FactorModelCovariance": cvxportfolio.FactorModelCovariance,
    }