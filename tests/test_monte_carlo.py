from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.test_support import add_src_to_path, has_packages

HAS_DEPS = has_packages("numpy", "pandas")

project_root = add_src_to_path()

if HAS_DEPS:
    import pandas as pd
    
    from transfercoef.config import build_default_config # noqa: E402
    from transfercoef.monte_carlo import run_monte_carlo # noqa: E402
    from transfercoef.portfolio_optimizer import OptimizationResult # noqa: E402

@unittest.skipUnless(HAS_DEPS, "Requires numpy and pandas to run Monte Carlo tests.")
class MonteCarloTests(unittest.TestCase):
    def test_small_monte_carlo_run_produces_expected_shapes(self) -> None:
        config = build_default_config(project_root)
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                num_trials=3,
                num_assets=6,
                random_seed=11,
            ),
        )

        result = run_monte_carlo(config)

        self.assertEqual(len(result.trial_results), 3)
        self.assertFalse(result.trial_diagnostics_frame.empty)
        self.assertFalse(result.summary_frame.empty)
        self.assertIn("scenario_name", result.summary_frame.columns)

    def test_frontier_run_expansion_adds_te_columns(self) -> None:
        config = build_default_config(project_root)
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                num_trials=2,
                num_assets=4,
                random_seed=7,
                enable_tracking_error_frontier=True,
                tracking_error_targets=(0.02, 0.04),
                frontier_mode="hybrid",
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
            result = run_monte_carlo(config)

        self.assertIn("tracking_error_target", result.trial_diagnostics_frame.columns)
        self.assertIn("frontier_mode", result.trial_diagnostics_frame.columns)
        self.assertEqual(result.trial_diagnostics_frame["tracking_error_target"].nunique(), 2)
        self.assertGreaterEqual(len(result.trial_diagnostics_frame), 2 * len(config.scenarios) * 2)

if __name__ == "__main__":
    unittest.main()
        