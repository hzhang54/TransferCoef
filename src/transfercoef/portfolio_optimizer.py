from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ScenarioConfig, SimulationConfig
from .cvxportfolio_adapter import (
    get_common_constraint_classes,
    get_common_objective_classes,
    get_single_period_optimization_class,
)

@dataclass(frozen=True)
class OptimizationInputs:
    """Inputs required to construct one portfolio optimization problem."""

    forecast_alpha: pd.Series
    convariance: pd.DataFrame
    risk_aversion: float
    previous_weights: pd.Series | None = None


@dataclass(frozen=True)
class OptimizationResult:
    """Output of one scenario optimization request."""
    scenario_name: str
    weights: pd.Series
    method: str
    metadata: dict[str, object]

def _common_assets(forecast_alpha: pd.Series, covariance: pd.DataFrame) -> list[str]:
    return [
        asset 
        for asset in forecast_alpha.index
        if asset in covariance.index and asset in covariance.columns
    ]    

def _prepare_optimization_data(
    forecast_alpha: pd.Series,
    covariance: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Align forecast alpha and covariance to their common asset universe."""

    assets = _common_assets(forecast_alpha, covariance)
    if not assets:
        raise ValueError("No common assets between forecast alpha and covariance.")
    
    aligned_forecast_alpha = forecast_alpha.loc[assets].astype(float)
    aligned_covariance = covariance.loc[assets, assets].astype(float)
    return aligned_forecast_alpha, aligned_covariance

def _normalize_weights(weights: pd.Series, leverage_limit: float | None = None) -> pd.Series:
    
    
    