from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import AppConfig, Dataconfig, OutputConfig, SimulationConfig, build_default_config
from .monte_carlo import build_scenario_overview, run_monte_carlo 
from .table2 import (
    build_table2_report,
    create_table2_summary,
    export_table2_summary,
    summarize_table2_report,
)

def parse_args() -> argparse.Namespace:
    """parse command-line arguments for the Table 2 synthetic workflow."""
    
    parser = argparse.ArgumentParser(
        description="Run the Table 2 transfer coefficient Monte Carlo replication workflow.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Project root containing DESIGN.md, outputs/, and cvxportfolio-master/.",
    )
    # --num-trials argument of int type with no default, with help message "Override the default number of Monte Carlo trials."
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Override the default number of Monte Carlo trials.",
    )
    # --num-assets argument of int type with no default, with help message "Override the default number of assets in each synthetic cross-section."
    parser.add_argument(
        "--num-assets",
        type=int,
        default=None,
        help="Override the default number of assets in each synthetic cross-section.",
    )
    # --target-ic argument of float type with no default, with help message "Override the target information coefficient used for forecast generation."
    parser.add_argument(
        "--target-ic",
        type=float,
        default=None,
        help="Override the target information coefficient used for forecast generation.",
    )
    # --breadth argument of float type with no default, with help meesage "Override the breadth used in theoretical IR calculations."
    parser.add_argument(
        "--breadth",
        type=float,
        default=None,
        help="Override the breadth used in theoretical IR calculations.",
    )
    # --random-seed argument of int type with no default, with help message "Override the simulation random seed."
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the simulation random seed.",
    )
    # --table-name argument of string type with no default, with help message "Override the base output filename for Table 2 exports."
    parser.add_argument(
        "--table-name",
        default=None,
        help="Override the base output filename for Table 2 exports.",
    )
    # --no-export argument of string type with no default, with help message "Disable CSV export of the generated report artifacts."
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable CSV export of the generated report artifacts.",
    )
    # --include-turnover-path-dynamics argument of string type with no default, with help message "Carry previous scenario weights across trials when turnover limits are enabled."
    parser.add_argument(
        "--include-turnover-path-dynamics",
        action="store_true",
        help="Carry previous scenario weights across trials when turnover limits are enabled.",
    )
    # --enable-tracking-error-frontier argument of string type with no default, with help message "Enable paper-style TC-versus-TE frontier runs.  In the current cash-benchmark "
    parser.add_argument(
        "--enable-tracking-error-frontier",
        action="store_true",
        help=(
            "Enable paper-style TC-versus-TE frontier runs.  In the current cash-benchmark "
            "special case, TE is portfolio volatility under the residual covariance."
            ),
    )
    # --tracking-error-targets argument of string type with no default, with help message "Optional list of positive TE targets, e.g. --tracking-error-targets 0.02 0.04 0.06."
    parser.add_argument(
        "--tracking-error-targets",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of positive TE targets, e.g. --tracking-error-targets 0.02 0.04 0.06.",
    )
    # --frontier-mode argument with no default, two choices: hybrid or pure_risk_budget, with help message "Choose frontier optimzation mode: hybrid or pure_risk_budget.
    parser.add_argument(
        "--frontier-mode",
        choices=["hybrid", "pure_risk_budget"],
        default=None,
        help="Choose frontier optimzation mode: hybrid or pure_risk_budget.",
    )
    # --use-historical-calibration argument of string type with no default, with help message "Enable historical calibration mode in the config for later integration work."
    parser.add_argument(
        "--use-historical-calibration",
        action="store_true",
        help="Enable historical calibration mode in the config for later integration work.",
    )
    # --tickers argument with nargs "*" and help message "Optional ticker list for historical calibration mode."
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker list for historical calibration mode.",
    )
    return parser.parse_args()

def build_runtime_config(args: argparse.Namespace) -> AppConfig:
    """Build an application config using defaults plus CLI overrides."""

    base_config = build_default_config(args.project_root)

    tracking_error_targets = base_config.simulation.tracking_error_targets
    if args.tracking_error_targets is not None:
        if any(target <= 0 for target in args.tracking_error_targets):
            raise ValueError("All tracking error targets must be positive.")
        tracking_error_targets = tuple(sorted({float(target) for target in args.tracking_error_targets}))

    simulation = replace(
        base_config.simulation,
        num_trials=args.num_trials if args.num_trials is not None else base_config.simulation.num_trials,
        num_assets=args.num_assets if args.num_assets is not None else base_config.simulation.num_assets,
        target_ic=args.target_ic if args.target_ic is not None else base_config.simulation.target_ic,
        breadth=args.breadth if args.breadth is not None else base_config.simulation.breadth,
        random_seed=args.random_seed if args.random_seed is not None else base_config.simulation.random_seed,
        include_turnover_path_dynamics=(
            args.include_turnover_path_dynamics
            if args.include_turnover_path_dynamics
            else base_config.simulation.include_turnover_path_dynamics
        ),
        enable_tracking_error_frontier=(
            args.enable_tracking_error_frontier
            if args.enable_tracking_error_frontier
            else base_config.simulation.enable_tracking_error_frontier
        ),
        tracking_error_targets=tracking_error_targets,
        frontier_mode=args.frontier_mode if args.frontier_mode is not None else base_config.simulation.frontier_mode,
    )
    
    output = replace(
        base_config.output,
        table_name=args.table_name if args.table_name is not None else base_config.output.table_name,
        export_csv= False if args.no_export else base_config.output.export_csv,
    )
    
    data = replace(
        base_config.data,
        use_historical_calibration=args.use_historical_calibration,
        tickers=tuple(args.tickers) if args.tickers else base_config.data.tickers,
    )

    return AppConfig(
        paths=base_config.paths,
        data=data,
        simulation=simulation,
        output=output,
        scenarios=base_config.scenarios,
    )

def run_application(config: AppConfig) -> dict[str, pd.DataFrame | pd.Series | dict[str, Path]]:
    """Run the Monte Carlo workflow and optionally export the Table 2 artifacts."""

    run_result = run_monte_carlo(config)
    report_frames = build_table2_report(run_result)
    report_summary = summarize_table2_report(run_result)

    export_paths: dict[str, Path] = {}
    if config.output.export_csv or config.output.export_trial_data:
        export_paths = export_table2_summary(
            summary=create_table2_summary(run_result),
            config=config,
        )

    return {
        "scenario_overview": build_scenarios_overview(config),
        "table2": report_frames["table2"],
        "scenario_summary": report_frames["scenario_summary"],
        "trial_diagnostics": report_frames["trial_diagnostics"],
        "report_summary": report_summary,
        "export_paths": export_paths,
    }

def print_run_summary(results: dict[str, pd.DataFrame | pd.Series | dict[str, Path]]) -> None:
    """Print a concise console summary of a completed run."""
    
    report_summary = results["report_summary"]
    scenario_overview = results["scenario_overview"]
    table2 = results["table2"]
    export_paths = results["export_paths"]

    print("Scenario overview:")
    print(scenario_overview.to_string(index=False))
    print()

    print("Run summary:")
    if isinstance(report_summary, pd.Series):
        print(report_summary.to_string())
    print()

    print("Table 2-style summary:")
    if isinstance(table2, pd.DataFrame) and not table2.empty:
        print(table2.to_string())
    else:
        print("No Table 2 data available.")
    print()

    if export_paths:
        print("Exported files:")
        for label, path in export_paths.items():
            print(f"- {label}: {path}")
        
    
def main() -> None:
    """CLI entry point for the syntehtic Table 2 replication workflow."""

    args = parse_args()
    config = build_runtime_config(args)
    results = run_application(config)
    print_run_summary(results)

if __name__ == "__main__":
    main()
        
        
    

    