import sys
import pandas as pd
from unittest.mock import patch
from dataclasses import replace

from tests.test_support import add_src_to_path
project_root = add_src_to_path()

from transfercoef.config import build_default_config
from transfercoef.monte_carlo import run_monte_carlo
from transfercoef.portfolio_optimizer import OptimizationResult
from transfercoef.table2 import create_table2_summary, build_table2_report

def main():
    config = build_default_config(project_root)
    config = replace(
        config,
        simulation=replace(
            config.simulation,
            num_trials=2,
            num_assets=4,
            random_seed=3,
            enable_tracking_error_frontier=True,
            tracking_error_targets=(0.02, 0.04),
        ),
    )

    def fake_optimize_all_scenarios(project_root: str, scenarios: list, inputs):
        scenario = scenarios[0]
        weights = pd.Series([0.6, 0.4], index=["asset_1", "asset_2"], name=f"{scenario.name}_weights")
        return {
            scenario.name: OptimizationResult(
                scenario_name=scenario.name,
                weights=weights,
                method="mock_optimizer",
                metadata={"fallback_used": False},
            )
        }
    
    with patch("transfercoef.monte_carlo.optimize_all_scenarios", side_effect=fake_optimize_all_scenarios):
        run_result = run_monte_carlo(config)
    
    summary = create_table2_summary(run_result)
    
    print("PIVOT EMPTY:", summary.frontier_tc_pivot_frame.empty)
    print("PIVOT COLUMNS:", summary.frontier_tc_pivot_frame.columns.tolist())
    print("PIVOT HEAD:")
    print(summary.frontier_tc_pivot_frame.head())
    print("PIVOT VALUES dtype:")
    print(summary.frontier_tc_pivot_frame.dtypes)
    print("\nFRONTIER FRAME head (cols tracking_error_target, transfer_coefficient_mean):")
    print(summary.frontier_summary_frame[['tracking_error_target', 'transfer_coefficient_mean']].head())
    print("Is transfer_coefficient_mean all NaN?", summary.frontier_summary_frame['transfer_coefficient_mean'].isna().all())

if __name__ == "__main__":
    main()
