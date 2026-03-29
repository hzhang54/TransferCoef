from __future__ import annotations

from dataclasses import dataclass

import numpy as np 
import pandas as pd

from .alpha_model import AlphaSample, generate_alpha_samples_from_config
from .config import AppConfig, ScenarioConfig
from .diagnostics import TrialDiagnostics, build_trial_diagnostics
from .portfolio_optimizer import (
    OptimizationInputs,
    OptimizationResult,
    optimize_all_scenarios,
    solve_unconstrained_weights,
)

@dataclass(frozen=True)
class FrontierRunSpec:
    """One optimization run definition for a scenario and optional TE target."""

    run_key: str
    scenario_name: str
    tracking_error_target: float | None
    frontier_mode: str | None

@dataclass(frozen=True)
class MonteCarloTrialResult:
    """Complete output for one Monte Carlo trial across all scenario runs."""
    
    trial_id: int
    alpha_sample: AlphaSample
    covariance: pd.DataFrame
    unconstrained_weights: pd.Series
    scenario_results: dict[str, OptimizationResult]
    scenario_diagnostics: dict[str, TrialDiagnostics]
    run_specs: dict[str, FrontierRunSpec]


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
    tracking_error_target: float | None = None,
    frontier_mode: str | None = None,
) -> OptimizationInputs:
    """Construct optimization inputs for one trial."""

    return OptimizationInputs(
        forecast_alpha=alpha_sample.forecast_alpha,
        covariance=covariance,
        risk_aversion=risk_aversion,
        previous_weights=previous_weights,
        tracking_error_target=tracking_error_target,
        frontier_mode=frontier_mode,
    )

def build_frontier_run_specs(config: AppConfig) -> list[FrontierRunSpec]:
    """Expand configured scenarios into scenario/TE frontier run specifications."""

    frontier_enabled = config.simulation.enable_tracking_error_frontier
    frontier_mode = config.simulation.frontier_mode
    tracking_error_targets = config.simulation.tracking_error_targets
    precision = config.simulation.tracking_error_label_precision

    run_specs: list[FrontierRunSpec] = []
    for scenario in config.scenarios:
        if not frontier_enabled:
            run_specs.append(
                FrontierRunSpec(
                    run_key=scenario.name,
                    scenario_name=scenario.name,
                    tracking_error_target=None,
                    frontier_mode=None,
                )
            )
            continue

        for tracking_error_target in tracking_error_targets:
            run_specs.append(
                FrontierRunSpec(
                    run_key=(
                        f"{scenario.name}__te_{tracking_error_target:.{precision}f}"
                        f"__{frontier_mode}"
                    ),
                    scenario_name=scenario.name,
                    tracking_error_target=float(tracking_error_target),
                    frontier_mode=frontier_mode,
                )
            )
    return run_specs


def run_single_trial(
    config: AppConfig,
    trial_id: int,
    rng: np.random.Generator,
    previous_weights_by_run: dict[str, pd.Series] | None = None,
) -> MonteCarloTrialResult:
    """Run one synthetic Monte carlo trial across all configured scenario runs."""

    alpha_sample = generate_alpha_samples_from_config(
        simulation_config=config.simulation,
        rng=rng,
    )
    covariance = build_diagonal_covariance(
        residual_volatilities=alpha_sample.residual_volatilities,
    )

    previous_weights_by_run = previous_weights_by_run or {}
    unconstrained_inputs = build_optimization_inputs_for_trial(
        alpha_sample=alpha_sample,
        covariance=covariance,
        risk_aversion=config.simulation.risk_aversion,
        previous_weights=None,
    )
    unconstrained_weights = solve_unconstrained_weights(unconstrained_inputs)
    
    scenario_results: dict[str, OptimizationResult] = {}
    scenario_diagnostics: dict[str, TrialDiagnostics] = {}
    run_specs: dict[str, FrontierRunSpec] = {}
    scenario_map = {scenario.name: scenario for scenario in config.scenarios}
    
    for run_spec in build_frontier_run_specs(config):
        scenario = scenario_map[run_spec.scenario_name]
        optimization_inputs = build_optimization_inputs_for_trial(
            alpha_sample=alpha_sample,
            covariance=covariance,
            risk_aversion=config.simulation.risk_aversion,
            previous_weights=previous_weights_by_run.get(run_spec.run_key),
            tracking_error_target=run_spec.tracking_error_target,
            frontier_mode=run_spec.frontier_mode,
        )
        optimization_result = optimize_all_scenarios(
            project_root=config.paths.project_root,
            scenarios=[scenario],
            inputs=optimization_inputs,
        )[scenario.name]
        scenario_results[run_spec.run_key] = optimization_result    
        scenario_diagnostics[run_spec.run_key] = build_trial_diagnostics(
            scenario_name=run_spec.scenario_name,
            forecast_alpha=alpha_sample.forecast_alpha,
            realized_returns=alpha_sample.realized_returns,
            residual_volatilities=alpha_sample.residual_volatilities,
            unconstrained_weights=unconstrained_weights,
            constrained_weights=optimization_result.weights,
            covariance=covariance,
            breadth=config.simulation.breadth,
        )
        run_specs[run_spec.run_key] = run_spec

    return MonteCarloTrialResult(
        trial_id=trial_id,
        alpha_sample=alpha_sample,
        covariance=covariance,
        unconstrained_weights=unconstrained_weights,
        scenario_results=scenario_results,
        scenario_diagnostics=scenario_diagnostics,
        run_specs=run_specs,
    )

def trial_diagnostics_to_frame(trial_results: list[MonteCarloTrialResult]) -> pd.DataFrame:
    """Flatten per-trial diagnostics into a long-format DataFrame."""

    records: list[dict[str, object]] = []
    for trial_result in trial_results:
        for run_key, diagnostics in trial_result.scenario_diagnostics.items():
            run_spec = trial_result.run_specs[run_key]
            optimization_result = trial_result.scenario_results[run_key]
            record = diagnostics.to_series().to_dict()
            record["trial_id"] = trial_result.trial_id
            record["scenario_name"] = run_spec.scenario_name
            record["run_key"] = run_key
            record["tracking_error_target"] = run_spec.tracking_error_target
            record["frontier_mode"] = run_spec.frontier_mode
            record["optimization_method"] = optimization_result.method
            record["solver_success"] = not bool(optimization_result.metadata.get("fallback_used", False))
            records.append(record)
    
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)
            
def summarize_trials(trial_results: list[MonteCarloTrialResult]) -> pd.DataFrame:
    """Aggregate diagnostics across all trials into scenario summaries."""

#    diagnostics_list: list[TrialDiagnostics] = []
#    for trial_result in trial_results:
#        diagnostics_list.extend(trial_result.scenario_diagnostics.values())
#    return aggregate_trial_diagnostics(diagnostics_list)
    diagnostics_frame = trial_diagnostics_to_frame(trial_results)
    if diagnostics_frame.empty:
        return diagnostics_frame

    group_columns = ["scenario_name"]
    if diagnostics_frame["tracking_error_target"].notna().any():
        group_columns.extend(["tracking_error_target", "frontier_mode"])

    numeric_columns = diagnostics_frame.select_dtypes(include=["number"]).columns.difference(["trial_id"])
    aggregated = diagnostics_frame.groupby(group_columns, dropna=False)[list(numeric_columns)].agg(["mean", "std"])
    aggregated.columns = [f"{metric}_{statistic}" for metric, statistic in aggregated.columns]
    return aggregated.reset_index()


def run_monte_carlo(config: AppConfig) -> MonteCarloRunResult:
    """Run the full synthetic Monte Carlo experiment for the configured scenario runs."""

    rng = np.random.default_rng(config.simulation.random_seed)
    trial_results: list[MonteCarloTrialResult] = []
    previous_weights_by_run: dict[str, pd.Series] = {}

    for trial_id in range(config.simulation.num_trials):
        trial_result = run_single_trial(
            config=config,
            trial_id=trial_id,
            rng=rng,
            previous_weights_by_run=(
                previous_weights_by_run
                if config.simulation.include_turnover_path_dynamics
                else None
            ),
        )
        trial_results.append(trial_result)
        
        if config.simulation.include_turnover_path_dynamics:
            previous_weights_by_run = {
                run_key: result.weights
                for run_key, result in trial_result.scenario_results.items()
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
    """Return a compact DataFrame describing configured scenarios and frontier runs."""

#    rows: list[dict[str, object]] = []
#    for scenario in config.scenarios:
#        rows.append(_scenario_to_record(scenario))
    scenario_map = {scenario.name: scenario for scenario in config.scenarios}
    rows = [_scenario_to_record(scenario_map[run_spec.scenario_name], run_spec) for run_spec in build_frontier_run_specs(config)]
    return pd.DataFrame(rows)

def _scenario_to_record(scenario: ScenarioConfig, run_spec: FrontierRunSpec) -> dict[str, object]:
    return {
        "name": run_spec.run_key,
        "scenario_name": scenario.name,
        "description": scenario.description,
        "long_only": scenario.long_only,
        "leverage_limit": scenario.leverage_limit,
        "max_weight": scenario.max_weight,
        "min_weight": scenario.min_weight,
        "turnover_limit": scenario.turnover_limit,
        "dollar_neutral": scenario.dollar_neutral,
        "tracking_error_target": run_spec.tracking_error_target,
        "frontier_mode": run_spec.frontier_mode,
    }

        
    
    