from __future__ import annotations

from dataclasses import dataclass

import numpy as np 
import pandas as pd

from .alpha_model import AlphaSample, generate_alpha_samples_from_config
from .config import AppConfig, ScenarioConfig
from .diagnostics import TrialDiagnostics, aggregate_trial_diagnostics, build_trial_diagnostics
from .portfolio_optimizer import (
    OptimizationInputs,
    OptimizationResults,
    optimize_all_scenarios,
    solve_unconstrained_weights,
)


@dataclass(frozen=True)
class MonteCarloTrialResult:
    """Complete output for one Monte Carlo trial across all scenarios."""
    
    trial_id: int
    alpha_sample: AlphaSample
    covariance: pd.DataFrame
    unconstrained_weights: pd.Series
    scenario_results: dict[str, OptimizationResults]
    scenario_diagnostics: dict[str, TrialDiagnostics]
    

@dataclass(frozen=True)
class MonteCarloRunResult:
    """Aggregate output for a full Monte Carlo run."""

    config: AppConfig
    trial_results: list[MonteCarloTrialResult]
    trial_diagnostics_frame: pd.DataFrame
    summary_frame: pd.DataFrame

def build_diagonal_covariance(
    residual_volatilities: pd.Series,
) -> pd.DataFrame:
    """Build a simple diagonal covariance matrix for synthetic trials."""

    variances = np.square(residual_volatilities.to_numpy(dtype=float))
    return pd.DataFrame(
        np.diag(variances),
        index=residual_volatilities.index,
        columns=residual_volatilities.index,
    )

def build_optimization_inputs_for_trial(
    alpha_sample: AlphaSample,
    covariance: pd.DataFrame,
    risk_aversion: float,
    previous_weights: pd.Series | None = None,
) -> OptimizationInputs:
    """Construct optimization inputs for one trial."""

    return OptimizationInputs(
        forecast_alpha=alpha_sample.forecast_alpha,
        covariance=covariance,
        risk_aversion=risk_aversion,
        previous_weights=previous_weights,
    )

def run_single_trial(
    config: AppConfig,
    trial_id: int,
    rng: np.random.Generator,
    previous_weights_by_scenario: dict[str, pd.Series] | None = None,
) -> MonteCarloTrialResult:
    """Run one synthetic Monte carlo trial across all configured scenarios."""

    alpha_sample = generate_alpha_samples_from_config(
        simulation_config=config.simulation,
        rng=rng,
    )
    covariance = build_diagonal_covariance(
        residual_volatilities=alpha_sample.residual_volatilities,
    )

    previous_weights_by_scenario = previous_weights_by_scenario or {}
    unconstrained_inputs = build_unconstrained_inputs_for_trial(
        alpha_sample=alpha_sample,
        covariance=covariance,
        risk_aversion=config.simulation.risk_aversion,
        previous_weights=None,
    )
    unconstrained_weights = solve_unconstrained_weights(unconstrained_inputs)
    
    scenario_results: dict[str, OptimizationResults] = {}
    scenario_diagnostics: dict[str, TrialDiagnostics] = {}
    
    for scenario in config.scenarios:
        optimization_inputs = build_optimization_inputs_for_trial(
            alpha_sample=alpha_sample,
            covariance=covariance,
            risk_aversion=config.simulation.risk_aversion,
            previous_weights=previous_weights_by_scenario.get(scenario.name),
        )
        optimization_result = optimize_all_scenarios(
            project_root=config.paths.project_root,
            scenarios=[scenario],
            inputs=optimization_inputs,
        )[scenario.name]
        scenario_results[scenario.name] = optimization_result    
        scenario_diagnostics[scenario.name] = build_trial_diagnostics(
            scenario_name=scenario.name,
            forecast_alpha=alpha_sample.forecast_alpha,
            realized_returns=alpha_sample.realized_returns,
            residual_volatilities=alpha_sample.residual_volatilities,
            unconstrained_weights=unconstrained_weights,
            constrained_weights=optimization_result.weights,
            covariance=covariance,
            breadth=config.simulation.breadth,
        )

    return MonteCarloTrialResult(
        trial_id=trial_id,
        alpha_sample=alpha_sample,
        covariance=covariance,
        unconstrained_weights=unconstrained_weights,
        scenario_results=scenario_results,
        scenario_diagnostics=scenario_diagnostics,
    )

def trial_diagnostics_to_frame(trial_results: list[MonteCarloTrialResult]) -> pd.DataFrame:
    """Flatten per-trial diagnostics into a long-format DataFrame."""

    records: list[dict[str, object]] = []
    for trial_result in trial_results:
        for scenario_name, diagnostics in trial_result.scenario_diagnostics.items():
            record = diagnostics.to_series().to_dict()
            record["trial_id"] = trial_result.trial_id
            record["scenario_name"] = scenario_name
            records.append(record)
    
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)
            
def summarize_trials(trial_results: list[MonteCarloTrialResult]) -> pd.DataFrame:
    """Aggregate diagnostics across all trials into scenario summaries."""

    diagnostics_list: list[TrialDiagnostics] = []
    for trial_result in trial_results:
        diagnostics_list.extend(trial_result.scenario_diagnostics.values())
    return aggregate_trial_diagnostics(diagnostics_list)



    def run_monte_carlo(config: AppConfig) -> MonteCarloRunResult:
        """Run the full synthetic Monte Carlo experiment for the configured scenarios."""

        rng = np.random.default_rng(config.simulation.random_seed)
        trial_results: list[MonteCarloTrialResult] = []
        previous_weights_by_scenario: dict[str, pd.Series] = {}

        for trial_id in range(config.simulation.num_trials):
            trial_result = run_single_trial(
                config=config,
                trial_id=trial_id,
                rng=rng,
                previous_weights_by_scenario=(
                    previous_weights_by_scenario
                    if config.simulation.include_turnover_path_dynamics
                    else None
                ),
            )
            trial_results.append(trial_result)
            
            if config.simulation.include_turnover_path_dynamics:
                previous_weights_by_scenario = {
                    scenario_name: result.weights
                    for scenario_name, result in trial_result.scenario_results.items()
                }

        trial_diagnostics_frame = trial_diagnostics_to_frame(trial_results)
        summary_frame = summarize_trials(trial_results)

        return MonteCarloRunResult(
            config=config,
            trial_results=trial_results,
            trial_diagnostics_frame=trial_diagnostics_frame,
            summary_frame=summary_frame,
        ) 

    def build_scenario_overview(config: AppConfig) -> pd.DataFrame:
        """Return a compact DataFrame describing configured scenarios."""
    
        rows: list[dict[str, object]] = []
        for scenario in config.scenarios:
            rows.append(_scenario_to_record(scenario))
        return pd.DataFrame(rows)

    def _scenario_to_record(scenario: ScenarioConfig) -> dict[str, object]:
        return {
            "name": scenario.name,
            "description": scenario.description,
            "long_only": scenario.long_only,
            "leverage_limit": scenario.leverage_limit,
            "max_weight": scenario.max_weight,
            "min_weight": scenario.min_weight,
            "turnover_limit": scenario.turnover_limit,
            "dollar_neutral": scenario.dollar_neutral,
        }

        
    
    