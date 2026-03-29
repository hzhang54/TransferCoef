from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.test_support import add_src_to_path, has_packages

HAS_DEPS = has_packages("numpy", "pandas")

if HAS_DEPS:
    import pandas as pd

project_root = add_src_to_path()

if HAS_DEPS:
    from transfercoef.config import build_default_config # noqa: E402
    from transfercoef.monte_carlo import run_monte_carlo # noqa: E402
    from transfercoef.portfolio_optimizer import OptimizationResult # noqa: E402
    from transfercoef.table2 import ( # noqa: E402
        build_output_paths,
        build_table2_report,
        create_table2_summary,
        format_table2_for_display,
        summarize_table2_report,
    )

@unittest.skipUnless(HAS_DEPS, "Requires numpy and pandas to run Table 2 tests.")
class Table2Tests(unittest.TestCase):
    def test_create_table2_summary_produces_non_empty_table(self) -> None:
        config = build_default_config(project_root)
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                num_trials=2,
                num_assets=5,
                random_seed=5,
            ),
        )

        run_result = run_monte_carlo(config)
        summary = create_table2_summary(run_result)

        self.assertFalse(summary.table_frame.empty)
        self.assertFalse(summary.scenario_summary_frame.empty)
        self.assertFalse(summary.trial_diagnostics_frame.empty)

    def test_format_table2_for_display_rounds_numeric_values(self) -> None:
        frame = pd.DataFrame({"scenario_a": [0.123456789]}, index=["Mean ex post IC"])

        rounded = format_table2_for_display(frame, float_precision=4)

        self.assertAlmostEqual(float(rounded.iloc[0,0]), 0.1235)

    def test_build_output_paths_uses_output_table_directory(self) -> None:
        config = build_default_config(project_root)
        paths = build_output_paths(config)
        
        self.assertIn("table2_csv", paths)
        self.assertTrue(str(paths["table2_csv"]).endswith("table2_replication.csv"))

    def test_summarize_table2_report_returns_expected_metadata(self) -> None:
        config = build_default_config(project_root)
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                num_trials=2,
                num_assets=4,
                random_seed=3
            ),
        )

        run_result = run_monte_carlo(config)
        summary = summarize_table2_report(run_result)
        report_frames = build_table2_report(run_result)

        self.assertEqual(int(summary["num_trials"]), 2)
        self.assertEqual(int(summary["num_scenarios"]), len(config.scenarios))
        self.assertIn("table2", report_frames)
        self.assertIn("scenario_summary", report_frames)
        self.assertIn("trial_diagnostics", report_frames)
    
    def test_frontier_summary_outputs_are_present_when_enabled(self) -> None:
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
            asset_index = inputs.forecast_alpha.index
            weight_values = [0.6, 0.4] + [0.0] * (len(asset_index) - 2)
            weights = pd.Series(weight_values, index=asset_index, name=f"{scenario.name}_weights")
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
        report_frames = build_table2_report(run_result)

        self.assertFalse(summary.frontier_summary_frame.empty)
        self.assertFalse(summary.frontier_tc_pivot_frame.empty)
        self.assertFalse(summary.frontier_risk_weighted_tc_pivot_frame.empty)

        self.assertIn("frontier_summary", report_frames)
        self.assertIn("frontier_tc_pivot", report_frames)
        self.assertIn("frontier_risk_weighted_tc_pivot", report_frames)

    
if __name__ == "__main__":
    unittest.main()