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
    tracking_error_target: float | None = None
    frontier_mode: str | None = None

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
    """Normalize weights to a target gross leverage when requested."""

    normalized = weights.astype(float).copy()
    gross = float(normalized.abs().sum())
    if np.isclose(gross, 0.0):
        return normalized
    
    if leverage_limit is None:
        return normalized / gross
    
    return normalized * (float(leverage_limit) / gross)

def solve_unconstrained_weights(inputs: OptimizationInputs) -> pd.Series:
    """Compute unconstrained Markowitz weights via a closed-form solve."""

    aligned_forecast_alpha, aligned_covariance = _prepare_optimization_data(
        inputs.forecast_alpha, 
        inputs.convariance
    )

    alpha = aligned_forecast_alpha.to_numpy(dtype=float)
    covariance = aligned_covariance.to_numpy(dtype=float)

    ridge = 1e-8 * np.eye(len(aligned_forecast_alpha.index))
    raw_weights = np.linalg.solve(inputs.risk_aversion * covariance + ridge, alpha)
    weights = pd.Series(raw_weights, index=aligned_forecast_alpha.index, name="unconstrained_weights")    
    return _normalize_weights(weights)

def build_constraint_objects(
    project_root: str | Path,
    scenario: ScenarioConfig,
    covariance: pd.DataFrame | None = None,
    tracking_error_target: float | None = None,    
) -> list[object]:
    """Construct local cvxportfolio constraint objects for a scenario."""
    
    constraints = []
    classes = get_common_constraint_classes(project_root)
    objective_classes = get_common_objective_classes(project_root)

    if scenario.long_only:
        constraints.append(classes["LongOnly"]())
    if scenario.leverage_limit is not None:
        constraints.append(classes["LeverageLimit"](scenario.leverage_limit))
    if scenario.max_weight is not None:
        constraints.append(classes["MaxWeights"](scenario.max_weight))
    if scenario.min_weight is not None:
        constraints.append(classes["MinWeights"](scenario.min_weight))
    if scenario.turnover_limit is not None:
        constraints.append(classes["TurnoverLimit"](scenario.turnover_limit))
    if scenario.dollar_neutral is not None:
        constraints.append(classes["DollarNeutral"]())
    if tracking_error_target is not None:
        if tracking_error_target <= 0:
            raise ValueError("tracking_error_target must be positive when provided.")
        if covariance is None:
            raise ValueError("covariance is required when building a tracking-error constraint.")
        constraints.append(
            objective_classes["FullCovariance"](covariance) <= float(tracking_error_target) ** 2
        )

    return constraints
    
def build_single_period_policy(
    project_root: str | Path,
    scenario: ScenarioConfig,
    forecast_alpha: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
    tracking_error_target: float | None = None,
    frontier_mode: str | None = None,
) -> object:
    """Build a local cvxportfolio single-period policy for a scenario."""

    aligned_forecast_alpha, aligned_covariance = _prepare_optimization_data(
        forecast_alpha,
        covariance,
    )

    objective_classes = get_common_objective_classes(project_root)
    single_period_optimization = get_single_period_optimization_class(project_root)

    #objectives = (
    #    objective_classes["ReturnsForecast"](aligned_forecast_alpha)
    #    - risk_aversion * objective_classes["FullCovariance"](aligned_covariance)
    #)
    #constraints = build_constraint_objects(project_root, scenario)
    
    normalized_frontier_mode =  (frontier_mode or "hybrid").strip().lower().replace("-", "_")
    if normalized_frontier_mode not in {"hybrid", "pure_risk_budget"}:
        raise ValueError(
            "frontier_mode must be one of {'hybrid', 'pure_risk_budget'} when provided."
        )
    
    objective = objective_classes["ReturnsForecast"](aligned_forecast_alpha)
    if normalized_frontier_mode == "hybrid":
        objective = objective - risk_aversion * objective_classes["FullCovariance"](aligned_covariance)
    
    constraints = build_constraint_objects(
        project_root=project_root,
        scenario=scenario,
        covariance=aligned_covariance,
        tracking_error_target=tracking_error_target,
    )

    return single_period_optimization(
        objectives=objectives,
        constraints=constraints,
        include_cash_return=False,
    )


def build_holdings_from_weights(
    asset_names: list[str],
    portfolio_value: float = 1.0,
    previous_weights: pd.Series | None = None,
    cash_key: str = "cash",
) -> pd.Series:
    """Create a holdings vector suitable for ``cvxportfolio.Policy.execute``."""

    noncash_weights = pd.Series(0.0, index=asset_names, dtype=float)    
    if previous_weights is not None:
        common_assets = noncash_weights.index.intersection(previous_weights.index)
        noncash_weights.loc[common_assets] = previous_weights.loc[common_assets].astype(float)
    
    holdings = noncash_weights * float(portfolio_value)
    cash_value = float(portfolio_value) - float(holdings.sum())
    holdings.loc[cash_key] = cash_value
    return holdings

def extract_post_trade_weights(
    trades: pd.Series,
    holdings: pd.Series,
    cash_key: str = "cash",
) -> pd.Series:
    """Convert post-trade holdings into non-cash portfolio weights."""
    
    post_trade_holdings = (holdings + trades).astype(float)
    portfolio_value = float(post_trade_holdings.sum())
    if np.isclose(portfolio_value, 0.0):
        raise ValueError("Post-trade holdings sum to zero, cannot compute weights.")
    
    post_trade_weights = post_trade_holdings / portfolio_value
    return post_trade_weights.drop(labels=[cash_key], errors="ignore")
    
def execute_real_policy(
    project_root: str | Path,
    scenario: ScenarioConfig,
    inputs: OptimizationInputs,
    portfolio_value: float = 1.0,
    cash_key: str = "cash",
) -> tuple[pd.Series, object]:
    """Execute a real local cvxportfolio policy and return non-cash weights."""
    
    aligned_forecast_alpha, aligned_covariance = _prepare_optimization_data(
        inputs.forecast_alpha,
        inputs.convariance,
    )
    policy = build_single_period_policy(
        project_root=project_root,
        scenario=scenario,
        forecast_alpha=aligned_forecast_alpha,
        covariance=aligned_covariance,
        risk_aversion=inputs.risk_aversion,
        tracking_error_target=inputs.tracking_error_target,
        frontier_mode=inputs.frontier_mode,
    )
    
    holdings = build_holdings_from_weights(
        asset_names=list(aligned_forecast_alpha.index),
        portfolio_value=portfolio_value,
        previous_weights=inputs.previous_weights,
        cash_key=cash_key,
    )
    
    trades, _, _ = policy.execute(
        h=holdings,
        market_data=None,
        t=pd.Timestamp.utcnow(),
    )
    weights = extract_post_trade_weights(
        trades=trades,
        holdings=holdings,
        cash_key=cash_key,
    )
    return weights.rename(f"{scenario.name}_weights"), policy

def apply_simple_constraints(
    weights: pd.Series,
    scenario: ScenarioConfig,    
) -> pd.Series:
    """Apply a simple deterministic approximation of scenario constraints."""

    constrained = weights.astype(float).copy()
    
    if scenario.long_only:
        constrained = constrained.clip(lower=0.0)
    if scenario.max_weight is not None:
        constrained = constrained.clip(upper=scenario.max_weight)
    if scenario.min_weight is not None:
        constrained = constrained.clip(lower=scenario.min_weight)
    if scenario.dollar_neutral:
        constrained = constrained - constrained.mean()
    
    leverage_target = scenario.leverage_limit
    return _normalize_weights(constrained, leverage_limit=leverage_target)

def approximate_constrained_weights(
    unconstrained_weights: pd.Series,
    scenario: ScenarioConfig,
    previous_weights: pd.Series | None = None,
) -> pd.Series:
    """Approximate constrained weights before full cvxportfolio execution is wired in."""

    constrained = apply_simple_constraints(unconstrained_weights, scenario)
    
    if scenario.turnover_limit is not None and previous_weights is not None:
        common_index = constrained.index.intersection(previous_weights.index)
        adjusted = constrained.copy()
        target_change = adjusted.loc[common_index] - previous_weights.loc[common_index]
        gross_turnover = 0.5 * float(target_change.abs().sum())
        if gross_turnover > scenario.turnover_limit and gross_turnover > 0.0:
            scale = scenario.turnover_limit / gross_turnover
            adjusted.loc[common_index] = previous_weights.loc[common_index] + scale * target_change
            constrained = _normalize_weights(adjusted, leverage_limit=scenario.leverage_limit)

    return constrained.rename(f"{scenario.name}_weights")

def optimize_scenarios(
    project_root: str | Path,
    scenarios: ScenarioConfig,
    inputs: OptimizationInputs,
) -> OptimizationResult:
    """Optimize one scenario using the current project optimization contract."""

    unconstrained_weights = solve_unconstrained_weights(inputs)
    try:
        executed_weights, policy = execute_real_policy(
            project_root=project_root,
            scenario=scenario,
            inputs=inputs,
        )
        return OptimizationResult(
            scenario_name=scenario.name,
            weights=executed_weights,
            method="cvxportfolio_policy_execute",
            metadata={
                "policy": policy,
                "description": scenario.description,
                "fallback_used": False,
                "tracking_error_target": inputs.tracking_error_target,
                "frontier_mode": inputs.frontier_mode,
            },
        )
    except Exception as exc:
        if inputs.tracking_error_target is not None:
            raise RuntimeError(
                "Tracking-error constrained optimization requires successful local cvxportfolio execution; "
                "heuristic fallback is disabled for TE-constrained runs."
            ) from exc

        approximate_weights = approximate_constrained_weights(
            unconstrained_weights=unconstrained_weights,
            scenario=scenario,
            previous_weights=inputs.previous_weights,
        )
        return OptimizationResult(
            scenario_name=scenario.name,
            weights=approximate_weights,
            method="approximate_constrained_fallback",
            metadata={
                "policy": None,
                "description": scenario.description,
                "fallback_used": True,
                "fallback_reason": str(exc),
                "tracking_error_target": inputs.tracking_error_target,
                "frontier_mode": inputs.frontier_mode,
            },
        )
                
            
def optimize_all_scenarios(
    project_root: str | Path,
    scenarios: tuple[ScenarioConfig, ...] | list[ScenarioConfig],
    inputs: OptimizationInputs,
) -> dict[str, OptimizationResult]:
    """Optimize all configured scenarios and returns keyed by scenario name."""

    results: dict[str, OptimizationResult] = {}
    for scenario in scenarios:
        results[scenario.name] = optimize_scenarios(
            project_root=project_root,
            scenario=scenario,
            inputs=inputs,
        )
    return results

def build_optimization_inputs(
    forecast_alpha: pd.Series,
    covariance: pd.DataFrame,
    simulation_config: SimulationConfig,
    previous_weights: pd.Series | None = None,
    tracking_error_target: float | None = None,
) -> OptimizationInputs:
    """Create optimization inputs from the shared simulation configuration."""

    return OptimizationInputs(
        forecast_alpha=forecast_alpha,
        covariance=covariance,
        risk_aversion=simulation_config.risk_aversion,
        previous_weights=previous_weights,
        tracking_error_target=tracking_error_target,
        frontier_mode=simulation_config.frontier_mode,
    )
        


    
