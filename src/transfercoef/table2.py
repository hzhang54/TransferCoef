from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import AppConfig
from .diagnostics import build_table2_layout
from .monte_carlo import MonteCarloRunResult


@dataclass(frozen=True)
class Table2Summary:
    """Container for Table 2-style reporting outputs."""

    table_frame: pd.DataFrame
    scenario_summary_frame: pd.DataFrame
    trial_diagnostics_frame: pd.DataFrame

def create_table2_summary(run_result: MonteCarloRunResult) -> Table2Summary:
    """Build the Table 2-style summary artifacts from a Monte Carlo run."""

    table_frame = build_table2_layout(run_result.summary_frame)
    return Table2Summary(
        table_frame=table_frame,
        scenario_summary_frame=run_result.summary_frame.copy(),
        trial_diagnostics_frame=run_result.trial_diagnostics_frame.copy(),
    )

def format_table2_for_display(
    table_summary: pd.DataFrame,
    float_precision: int = 6,
) -> pd.DataFrame:
    """Return a display-friendly rounded version of the Table 2 summary."""

    if table_frame.empty:
        return table_frame.copy()

    return table_frame.astype(float).round(float_precision)

def scenario_summary_for_display(
    scenario_summary_frame: pd.DataFrame,
    float_precision: int = 6,
) -> pd.DataFrame:
    """Return a display-friendly rounded scenario summary."""
    if scenario_summary_frame.empty:
        return scenario_summary_frame.copy()

    display_frame = scenario_summary_frame.copy()
    numeric_columns = display_frame.select_dtypes(include=["number"]).columns
    display_frame.loc[:, numeric_columns] = display_frame.loc[:, numeric_columns].round(float_precision)
    return display_frame


def build_output_paths(config: AppConfig) -> dict[str, Path]:
    """Build standard output paths for Table 2 reporting artifacts."""

    base_name = config.output.table_name
    output_dir = config.paths.output_tables_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "table_csv": output_dir / f"{base_name}.csv",
        "scenario_summary_csv": output_dir / f"{base_name}_scenario_summary.csv",
        "trial_diagnostics_csv": output_dir / f"{base_name}_trial_diagnostics.csv",
    }

def export_table2_summary(
    summary: Table2Summary,
    config: AppConfig,
) -> dict[str, Path]:
    """Export Table 2 summary artifacts to CSV files."""

    output_paths = build_output_paths(config)
    
    if config.output.export_csv:
        summary.table_frame.to_csv(output_paths["table2_csv"])
        summary.scenario_summary_frame.to_csv(output_paths["scenario_summary_csv"], index=False)

    if config.output.export_trial_data:
        summary.trial_diagnostics_frame.to_csv(output_paths["trial_diagnostics_csv"], index=False)

    return output_paths

def build_table2_report(run_result: MonteCarloRunResult) -> dict[str, pd.DataFrame]:
    """Create all report DataFrames needed for a Table 2 review workflow."""

    summary = create_table2_summary(run_result)
    return {
        "table2": format_table2_for_display(
            summary.table_frame,
            float_precision=run_result.config.output.float_precision,
        ),
        "scenario_summary": scenario_summary_for_display(
            summary.scenario_summary_frame,
            float_precision=run_result.config.output.float_precision,
        ),
        "trial_diagnostics": summary.trial_diagnostics_frame.copy(),
    }
        
    
def summarize_table2_report(run_result: MonteCarloRunResult) -> pd.Series:
    """Return compact run metadata for a generated Table 2 report."""

    summary = create_table2_summary(run_result)
    return pd.Series(
        {
            "num_trials": len(run_result.trial_results),
            "num_scenarios": len(run_result.config.scenarios),
            "table_rows": int(summary.table_frame.shape[0]),
            "table_columns": int(summary.table_frame.shape[1]),
            "scenario_summary_rows": int(summary.scenario_summary_frame.shape[0]),
            "trial_diagnostics_rows": int(summary.trial_diagnostics_frame.shape[0]),
        },
        name="table2_report_summary",
    )
    