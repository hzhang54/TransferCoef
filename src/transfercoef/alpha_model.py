from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationConfig

@dataclass(frozen=True)
class AlphaSample:
    """Synthetic forecast alpha and realized residual returns for one trial."""

    score_vector: pd.Series
    residual_volatilities: pd.Series
    forecast_alpha: pd.Series
    realized_returns: pd.Series
    target_ic: float
    realized_forecast_to_return_corr: float
    alpha_over_ic_dispersion: float


@dataclass(frozen=True)
class AlphaGenerationInputs:
    """Inputs required to generate one synthetic cross-sectional alpha sample."""

    asset_names: list[str]
    residual_volatility: float
    target_ic: float
    alpha_volatility: float | None = None

def build_asset_names(num_assets: int, prefix: str = "asset") -> list[str]:
    """Create deterministic synthetic asset names."""

    return [f"{prefix}_{index:04d}" for index in range(num_assets)]

def _validate_target_ic(target_ic: float) -> None:
    if not -1.0 <= target_ic <= 1.0:
        raise ValueError("target_ic must lie in the interval [-1, 1].")

def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    """Compute a stable Pearson correlation for aligned series."""

    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.shape[0] < 2:
        return float("nan")
    
    left_values = aligned.iloc[:, 0].to_numpy(dtype=float)
    right_values = aligned.iloc[:, 1].to_numpy(dtype=float)
    
    left_std = left_values.std(ddof=0)
    right_std = right_values.std(ddof=0)
    
    if np.isclose(left_std, 0.0) or np.isclose(right_std, 0.0):
        return float("nan")

    return float(np.corrcoef(left_values, right_values)[0, 1])
    
def generate_standard_normal_score(
    asset_names: list[str],
    rng: np.random.Generator,
) -> pd.Series:
    """Generate the standardized cross-sectional score vector from Eq. (9)."""
    
    score_vector = rng.normal(loc=0.0, scale=1.0, size=len(asset_names))
    return pd.Series(score_vector, index=asset_names, name="score_vector")

def generate_residual_volatilities(
    asset_names: list[str],
    residual_volatility: float,
) -> pd.Series:
    """Generate the asset-level residual volatilities ``sigma_i``."""

    sigma = np.full(len(asset_names), residual_volatility, dtype=float)
    return pd.Series(sigma, index=asset_names, name="residual_volatility")

def generate_forecast_alpha(
    score_vector: pd.Series,
    residual_volatilities: pd.Series,
    target_ic: float,
) -> pd.Series:
    """Generate the forecast alpha using the paper-style Eq. (9) scaling."""

    _validate_target_ic(target_ic)
    
    aligned_sigma = residual_volatilities.loc[score_vector.index].astype(float)
    forecast_values = np.abs(target_ic) * aligned_sigma.to_numpy(dtype=float) * score_vector.to_numpy(dtype=float)
    return pd.Series(forecast_values, index=score_vector.index, name="forecast_alpha")

def generate_realized_returns(
    score_vector: pd.Series,
    residual_volatilities: pd.Series,
    target_ic: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Generate realized residual returns with expected cross-sectional IC."""

    _validate_target_ic(target_ic)
    
    residual_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(score_vector),
    )
    aligned_sigma = residual_volatilities.loc[score_vector.index].astype(float)
    realized_values = aligned_sigma.to_numpy(dtype=float) * (
        target_ic * score_vector.to_numpy(dtype=float) + 
        np.sqrt(max(0.0, 1.0 - target_ic**2)) * residual_noise
    )
    return pd.Series(realized_values, index=score_vector.index, name="realized_returns")


def compute_alpha_over_ic_dispersion(
    forecast_alpha: pd.Series,
    target_ic: float
) -> float:
    """Compute ``std(alpha_i) / IC`` for comparison to ``sigma_i`` in Eq. (9)."""
    
    if np.isclose(target_ic, 0.0):
        return float("nan")
    
    return float((forecast_alpha / target_ic).std(ddof=0))

def generate_alpha_sample(
    inputs: AlphaGenerationInputs,
    rng: np.random.Generator,
) -> AlphaSample:
    """Generate one synthetic alpha/forecast/return sample for a trial."""

    score_vector = generate_standard_normal_score(
        asset_names=inputs.asset_names,
        rng=rng,
    )

    residual_volatilities = generate_residual_volatilities(
        asset_names=inputs.asset_names,
        residual_volatility=inputs.residual_volatility,
    )

    forecast_alpha = generate_forecast_alpha(
        score_vector=score_vector,
        residual_volatilities=residual_volatilities,
        target_ic=inputs.target_ic,
    )

    realized_returns = generate_realized_returns(
        score_vector=score_vector,
        residual_volatilities=residual_volatilities,
        target_ic=inputs.target_ic,
        rng=rng,
    )

    return AlphaSample(
        score_vector=score_vector,
        residual_volatilities=residual_volatilities,
        forecast_alpha=forecast_alpha,
        realized_returns=realized_returns,
        target_ic=inputs.target_ic,
        realized_forecast_to_return_corr=_safe_correlation(
            forecast_alpha,
            realized_returns,
        ),
        alpha_over_ic_dispersion=compute_alpha_over_ic_dispersion(
            forecast_alpha=forecast_alpha,
            target_ic=inputs.target_ic,
        ),
    )

def build_alpha_inputs_from_simulation_config(
    simulation_config: SimulationConfig,
    asset_names: list[str] | None = None,
) -> AlphaGenerationInputs:
    """Create alpha generation inputs from the shared simulation configuration."""
    
    effective_asset_names = (
        build_asset_names(simulation_config.num_assets)
        if asset_names is None
        else asset_names
    )

    return AlphaGenerationInputs(
        asset_names=effective_asset_names,
        residual_volatility=simulation_config.residual_volatility,
        target_ic=simulation_config.target_ic,
        alpha_volatility=simulation_config.alpha_volatility,
    )

def generate_alpha_samples_from_config(
    simulation_config: SimulationConfig,
    rng: np.random.Generator,
    asset_names: list[str] | None = None,
) -> AlphaSample:

    """convenience wrapper to generate a synthetic alpha sample from config."""

    inputs = build_alpha_inputs_from_simulation_config(
        simulation_config=simulation_config,
        asset_names=asset_names,
    )
    return generate_alpha_sample(inputs=inputs, rng=rng)


def summarize_alpha_sample(sample: AlphaSample) -> pd.Series:
    """Return a compact summary of one generated alpha sample."""

    return pd.Series(
        {
            "target_ic": sample.target_ic,
            "forecast_to_return_corr": sample.realized_forecast_to_return_corr,
            "alpha_over_ic_dispersion": sample.alpha_over_ic_dispersion,
            "mean_residual_volatility": float(sample.residual_volatilities.mean()),
            "score_mean": float(sample.score_vector.mean()),
            "score_std": float(sample.score_vector.std(ddof=0)),
            "forecast_mean": float(sample.forecast_alpha.mean()),
            "forecast_std": float(sample.forecast_alpha.std(ddof=0)),
            "return_mean": float(sample.realized_returns.mean()),
            "return_std": float(sample.realized_returns.std(ddof=0)),
        },
        name="alpha_sample_summary",
    )
    