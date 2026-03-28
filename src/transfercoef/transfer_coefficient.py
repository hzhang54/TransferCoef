from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TransferCoefficientResult:
    """Transfer coefficient values for one constrained portfolio comparison."""

    plain_tc: float
    risk_weighted_tc: float

    def to_series(self) -> pd.Series:
        """Convert transfer coefficient results to a pandas series."""

        return pd.Series(
            {
                "transfer_coefficient": self.plain_tc,
                "risk_weighted_transfer_coefficient": self.risk_weighted_tc,
            }
        )

def align_series(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two series on their shared index and drop missing values."""

    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return aligned.iloc[:, 0], aligned.iloc[:, 1]

def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    """Compute a stable Pearson correlation after alignment."""

    aligned_left, aligned_right = align_series(left, right)
    if len(aligned_left) < 2:
        return float("nan")
    
    left_values = aligned_left.to_numpy(dtype=float)
    right_values = aligned_right.to_numpy(dtype=float)

    if np.isclose(left_values.std(ddof=0), 0.0) or np.isclose(right_values.std(ddof=0), 0.0):
        return float("nan")
    
    return float(np.corrcoef(left_values, right_values)[0, 1])

def compute_plain_transfer_coefficient(
    unconstrained_weights: pd.Series,
    constrained_weights: pd.Series,
) -> float:
    """Compute the plain Pearson transfer coefficient."""

    return safe_correlation(unconstrained_weights, constrained_weights)
    
def compute_risk_weighted_transfer_coefficient(
    unconstrained_weights: pd.Series,
    constrained_weights: pd.Series,
    risk_weights: pd.Series,
) -> float:
    """Compute a risk-weighted transfer coefficient using non-negative weights."""

    aligned = pd.concat(
        [unconstrained_weights, constrained_weights, risk_weights],
        axis=1,
    ).dropna()

    if aligned.shape[0] < 1:
        return float("nan")
    
    left = aligned.iloc[:, 0].to_numpy(dtype=float)
    right = aligned.iloc[:, 1].to_numpy(dtype=float)
    weights = aligned.iloc[:, 2].to_numpy(dtype=float)

    if np.any(weights < 0):
        raise ValueError("risk_weights must be non-negative.")

    left_norm = np.sqrt(np.sum(weights * np.square(left)))
    right_norm = np.sqrt(np.sum(weights * np.square(right)))

    if np.isclose(left_norm, 0.0) or np.isclose(right_norm, 0.0):
        return float("nan")
    
    numerator = float(np.sum(weights * left * right))
    return numerator / float(left_norm * right_norm)


def inverse_variance_risk_weights(covariance: pd.DataFrame | pd.Series) -> pd.Series:
    """Create inverse-variance weights from a covariance matrix or variance vector."""
    if isinstance(covariance, pd.DataFrame):
        variance = pd.Series(np.diag(covariance.to_numpy(dtype=float)), index=covariance.index)
    else:
        variance = covariance.astype(float)
    
    safe_variance = variance.replace(0.0, np.nan)
    inv_var = 1.0 / safe_variance
    inv_var = inv_var.replace([np.inf, -np.inf], np.nan).dropna()
    return inv_var

        
def build_transfer_coefficient_result(
    unconstrained_weights: pd.Series,
    constrained_weights: pd.Series,
    covariance: pd.DataFrame | pd.Series | None = None,
    risk_weights: pd.Series | None = None,
) -> TransferCoefficientResult:
    """Compute both plain and risk-weighted transfer coefficients."""
    
    effective_risk_weights = risk_weights
    if effective_risk_weights is None:
        if covariance is None:
            raise ValueError("Either covariance or risk_weights must be provided.")
        effective_risk_weights = inverse_variance_risk_weights(covariance)
    
    plain_tc = compute_plain_transfer_coefficient(
        unconstrained_weights=unconstrained_weights,
        constrained_weights=constrained_weights,
    )
    risk_weighted_tc = compute_risk_weighted_transfer_coefficient(
        unconstrained_weights=unconstrained_weights,
        constrained_weights=constrained_weights,
        risk_weights=effective_risk_weights,
    )
    return TransferCoefficientResult(
        plain_tc=plain_tc,
        risk_weighted_tc=risk_weighted_tc,
    )
    