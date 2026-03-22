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
    ex_post_ic: float
    transfer_coefficient: float
    risk_weighted_transfer_coefficient: float
    active_return: float
    information_ratio: float
    theoretical_information_ratio: float

    def to_series(self) -> pd.Series:
        """Convert the diagnostics record to a pandas series."""
        return pd.Series(
            {
                "scenario_name": self.scenario_name,
                "ex_post_ic": self.ex_post_ic,
                "transfer_coefficient": self.transfer_coefficient,
                "risk_weighted_transfer_coefficient": self.risk_weighted_transfer_coefficient,
                "active_return": self.active_return,
                "information_ratio": self.information_ratio,
                "theoretical_information_ratio": self.theoretical_information_ratio,
            }
        )

def compute_ex_post_ic(forecast_alpha: pd.Series, realized_return: pd.Series) -> float:
    """Compute the ex-post information coefficient."""

    return safe_correlation(forecast_alpha, realized_return)

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
    realized_return: pd.Series,
    unconstrained_weights: pd.Series,
    constrained_weights: pd.Series,
    covariance: pd.DataFrame,
    breadth: float,
    risk_weights: pd.Series | None = None,
) -> TrialDiagnostics:
    """Compute all single-trial diagnostics for one scenario."""

    ex_post_ic = compute_ex_post_ic(forecast_alpha, realized_return)
    tc_result = build_transfer_coefficient_result(
        unconstrained_weights=unconstrained_weights,
        constrained_weights=constrained_weights,
        covariance=covariance,
        risk_weights=risk_weights,
    )
    
    active_return = compute_active_return(constrained_weights, realized_return)
    active_risk = compute_active_risk(constrained_weights, covariance)
    information_ratio = compute_information_ratio(active_return, active_risk)
    theoretical_information_ratio = compute_theoretical_information_ratio(
        information_coefficient=ex_post_ic,
        transfer_coefficient=tc_result.risk_weighted_tc,
        breadth=breadth,
    )
    return TrialDiagnostics(
        scenario_name=scenario_name,
        ex_post_ic=ex_post_ic,
        transfer_coefficient=tc_result.plain_tc,
        risk_weighted_transfer_coefficient=tc_result.risk_weighted_tc,
        active_return=active_return,
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
    numeric_columns = [
        column
        for column in frame.columns
        if column != "scenario_name"
    ]

    aggregated = frame.groupby("scenario_name", dropna=False)[numeric_columns].agg(["mean", "std"])
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
        "Mean TC": "transfer_coefficient_mean",
        "Std TC": "transfer_coefficient_std",
        "Mean risk-weighted TC": "risk_weighted_transfer_coefficient_mean",
        "Std risk-weighted TC": "risk_weighted_transfer_coefficient_std",
        "Mean active return": "active_return_mean",
        "Mean active risk": "active_risk_mean",
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
    