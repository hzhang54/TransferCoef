from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transfer_coefficient import (
    align_series,
    build_transfer_coefficient_result,
    safe_correlation,
)

@dataclass(frozen=True)
class TrialDiagnostics:
    """Diagnostic computed for a single scenario in one simulation trial."""

    scenario_name: str
    tracking_error_target: float | None
    frontier_mode: str | None
    optimization_method: str | None
    solver_success: bool
    ex_post_ic: float
    alpha_over_ic_dispersion: float
    realized_return_over_sigma_dispersion: float
    transfer_coefficient: float
    risk_weighted_transfer_coefficient: float
    active_return: float
    active_risk: float
    realized_tracking_error: float
    information_ratio: float
    theoretical_information_ratio: float

    def to_series(self) -> pd.Series:
        """Convert the diagnostics record to a pandas series."""
        return pd.Series(
            {
                "scenario_name": self.scenario_name,
                "tracking_error_target": self.tracking_error_target,
                "frontier_mode": self.frontier_mode,
                "optimization_method": self.optimization_method,
                "solver_success": self.solver_success,
                "ex_post_ic": self.ex_post_ic,
                "alpha_over_ic_dispersion": self.alpha_over_ic_dispersion,
                "realized_return_over_sigma_dispersion": self.realized_return_over_sigma_dispersion,
                "transfer_coefficient": self.transfer_coefficient,
                "risk_weighted_transfer_coefficient": self.risk_weighted_transfer_coefficient,
                "active_return": self.active_return,
                "active_risk": self.active_risk,
                "realized_tracking_error": self.realized_tracking_error,
                "information_ratio": self.information_ratio,
                "theoretical_information_ratio": self.theoretical_information_ratio,
            }
        )

def compute_ex_post_ic(forecast_alpha: pd.Series, realized_return: pd.Series) -> float:
    """Compute the ex-post information coefficient."""

    return safe_correlation(forecast_alpha, realized_return)

def compute_alpha_over_ic_dispersion(
    forecast_alpha: pd.Series,
    information_coefficient: float,
) -> float:
    """Compute ``std(alpha_i / IC)`` for the simulated cross-section."""
    if np.isnan(information_coefficient) or np.isclose(information_coefficient, 0.0):
        return float("nan")
    
    return float((forecast_alpha.astype(float) / information_coefficient).std(ddof=0))

def compute_realized_return_over_sigma_dispersion(
    realized_returns: pd.Series,
    residual_volatilities: pd.Series,
) -> float:
    """Compute ``std(r_i / sigma_i)`` for the simulated cross-section."""
    
    aligned = pd.concat([realized_returns, residual_volatilities], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    
    returns = aligned.iloc[:, 0].to_numpy(dtype=float)
    sigmas = aligned.iloc[:, 1].to_numpy(dtype=float)
    
    if np.any(np.isclose(sigmas, 0.0)):
        return float("nan")
    
    standardized_returns = returns / sigmas
    return float(np.std(standardized_returns, ddof=0))
        
    

def compute_active_return(weights: pd.Series, realized_return: pd.Series) -> float:
    """Compute active return as the portfolio return from the weight vector."""

    aligned_weights, aligned_returns = align_series(weights, realized_return)
    if aligned_weights.empty:
        return float("nan")
    return float(np.dot(aligned_weights.to_numpy(dtype=float), aligned_returns.to_numpy(dtype=float)))

def compute_active_risk(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Compute active risk from weights and a covariance matrix."""

    common_assets = [asset for asset in weights.index if asset in covariance.index and asset in covariance.columns]
    if not common_assets:
        return float("nan")
    
    aligned_weights = weights.loc[common_assets].to_numpy(dtype=float)
    aligned_covariance = covariance.loc[common_assets, common_assets].to_numpy(dtype=float)
    variance = float(aligned_weights.T @ aligned_covariance @ aligned_weights)
    return float(np.sqrt(max(variance, 0.0)))
    

def compute_information_ratio(active_return: float, active_risk: float) -> float:
    """Compute information ratio from active return and active risk."""

    if np.isnan(active_return) or np.isnan(active_risk) or np.isclose(active_risk, 0.0):
        return float("nan")
    return float(active_return / active_risk)


def compute_theoretical_information_ratio(
    information_coefficient: float,
    transfer_coefficient: float,
    breadth: float,
) -> float:
    """Compute the theoretical information ratio from the fundamental law extension. """

    if breadth < 0:
        raise ValueError("Breadth must be non-negative.")
    return float(information_coefficient * transfer_coefficient * np.sqrt(breadth))

def build_trial_diagnostics(
    scenario_name: str,
    forecast_alpha: pd.Series,
    realized_returns: pd.Series,
    residual_volatilities: pd.Series,
    unconstrained_weights: pd.Series,
    constrained_weights: pd.Series,
    covariance: pd.DataFrame,
    breadth: float,
    risk_weights: pd.Series | None = None,
    tracking_error_target: float | None = None,
    frontier_mode: str | None = None,
    optimization_method: str | None = None,
    solver_success: bool | None = None,
) -> TrialDiagnostics:
    """Compute all single-trial diagnostics for one scenario."""

    ex_post_ic = compute_ex_post_ic(forecast_alpha, realized_returns)
    alpha_over_ic_dispersion = compute_alpha_over_ic_dispersion(
        forecast_alpha,
        ex_post_ic,
    )
    
    realized_return_over_sigma_dispersion = compute_realized_return_over_sigma_dispersion(
        realized_returns,
        residual_volatilities,
    )

    tc_result = build_transfer_coefficient_result(
        unconstrained_weights=unconstrained_weights,
        constrained_weights=constrained_weights,
        covariance=covariance,
        risk_weights=risk_weights,
    )
    
    active_return = compute_active_return(constrained_weights, realized_returns)
    active_risk = compute_active_risk(constrained_weights, covariance)
    realized_tracking_error = active_risk
    information_ratio = compute_information_ratio(active_return, active_risk)
    theoretical_information_ratio = compute_theoretical_information_ratio(
        information_coefficient=ex_post_ic,
        transfer_coefficient=tc_result.risk_weighted_tc,
        breadth=breadth,
    )
    return TrialDiagnostics(
        scenario_name=scenario_name,
        tracking_error_target=tracking_error_target,
        frontier_mode=frontier_mode,
        optimization_method=optimization_method,
        solver_success=solver_success,
        ex_post_ic=ex_post_ic,
        alpha_over_ic_dispersion=alpha_over_ic_dispersion,
        realized_return_over_sigma_dispersion=realized_return_over_sigma_dispersion,
        transfer_coefficient=tc_result.plain_tc,
        risk_weighted_transfer_coefficient=tc_result.risk_weighted_tc,
        active_return=active_return,
        active_risk=active_risk,
        realized_tracking_error=realized_tracking_error,
        information_ratio=information_ratio,
        theoretical_information_ratio=theoretical_information_ratio,
    )

def aggregate_trial_diagnostics(
    trial_diagnostics: list[TrialDiagnostics],
) -> pd.DataFrame:
    """Aggregate per-trial diagnostics by scenario into summary statistics."""

    if not trial_diagnostics:
        return pd.DataFrame()

    frame = pd.DataFrame([diagnostic.to_series() for diagnostic in trial_diagnostics])
    group_columns = ["scenario_name"]
    if "tracking_error_target" in frame.columns and frame["tracking_error_target"].notna().any():
        group_columns.extend(["tracking_error_target", "frontier_mode"])

    excluded_columns = set(group_columns + ["optimization_method"])
    numeric_columns = [
        column
#        for column in frame.columns
#        if column not in group_columns
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded_columns
    ]

    #aggregated = frame.groupby("scenario_name", dropna=False)[numeric_columns].agg(["mean", "std"])
    aggregated = frame.groupby(group_columns, dropna=False)[numeric_columns].agg(["mean", "std"])
    aggregated.columns = [f"{metric}_{statistic}" for metric, statistic in aggregated.columns]
    return aggregated.reset_index()

def build_table2_layout(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Transform aggregated diagnostics into a table 2-style row layout."""

    if summary_frame.empty:
        return pd.DataFrame()

    summary_frame = summary_frame.set_index("scenario_name")
    row_mapping = {
        "Mean ex post IC": "ex_post_ic_mean",
        "Std ex post IC": "ex_post_ic_std",
        "Mean std(alpha_i / IC)": "alpha_over_ic_dispersion_mean",
        "Std std(alpha_i / IC)": "alpha_over_ic_dispersion_std",
        "Mean std(r_i / sigma_i)": "realized_return_over_sigma_dispersion_mean",
        "Std std(r_i / sigma_i)": "realized_return_over_sigma_dispersion_std",
        "Mean TC": "transfer_coefficient_mean",
        "Std TC": "transfer_coefficient_std",
        "Mean risk-weighted TC": "risk_weighted_transfer_coefficient_mean",
        "Std risk-weighted TC": "risk_weighted_transfer_coefficient_std",
        "Mean active return": "active_return_mean",
        "Mean active risk": "active_risk_mean",
        "Mean realized tracking error": "realized_tracking_error_mean",
        "Std realized tracking error": "realized_tracking_error_std",
        "Mean ex post IR": "information_ratio_mean",
        "Std ex post IR": "information_ratio_std",
        "Mean theoretical IR": "theoretical_information_ratio_mean",
        "Std theoretical IR": "theoretical_information_ratio_std",
    }
    
    table_rows: dict[str, pd.Series] = {}
    for label, column_name in row_mapping.items():
        if column_name in summary_frame.columns:
            table_rows[label] = summary_frame[column_name]
    
    if not table_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(table_rows).T
    