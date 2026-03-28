from __future__ import annotations

import unittest

from tests.test_support import add_src_to_path, has_packages

HAS_DEPS = has_packages("pandas", "numpy")

if HAS_DEPS:
    import pandas as pd 

add_src_to_path()

if HAS_DEPS:
    from transfercoef.diagnostics import ( # noqa: E402
        aggregate_trial_diagnostics,
        build_trial_diagnostics,
        compute_realized_return_over_sigma_dispersion,
        compute_theoretical_information_ratio,
    )

@unittest.skipUnless(HAS_DEPS, "Requires numpy and pandas to run diagnostics tests.")
class DiagnosticsTests(unittest.TestCase):
    def test_theoretical_information_ratio_matches_formula(self) -> None:
        result = compute_theoretical_information_ratio(
            information_coefficient=0.05,
            transfer_coefficient=0.4,
            breadth=100.0,
        )

        self.assertAlmostEqual(result, 0.2)

    def test_build_trial_diagnostics_returns_expected_fields(self) -> None:
        forecast_alpha = pd.Series([0.03, -0.01, 0.02], index=["a", "b", "c"])
        realized_returns = pd.Series([0.02, -0.02, 0.01], index=["a", "b", "c"])
        residual_volatilities = pd.Series([0.2, 0.3, 0.4], index=["a", "b", "c"])
        unconstrained_weights = pd.Series([0.4, -0.2, 0.4], index=["a", "b", "c"])
        constrained_weights = pd.Series([0.3, 0.0, 0.7], index=["a", "b", "c"])
        covariance = pd.DataFrame(
            [
                [0.04, 0.0, 0.0],
                [0.0, 0.09, 0.0],
                [0.0, 0.0, 0.16],
            ],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        
        diagnostics = build_trial_diagnostics(
            scenario_name="long_only",
            forecast_alpha=forecast_alpha,
            realized_returns=realized_returns,
            residual_volatilities=residual_volatilities,
            unconstrained_weights=unconstrained_weights,
            constrained_weights=constrained_weights,
            covariance=covariance,
            breadth=25.0,
        )
        
        self.assertEqual(diagnostics.scenario_name, "long_only")
        self.assertTrue(pd.notna(diagnostics.ex_post_ic))
        self.assertTrue(pd.notna(diagnostics.alpha_over_ic_dispersion))
        self.assertTrue(pd.notna(diagnostics.realized_return_over_sigma_dispersion))
        self.assertTrue(pd.notna(diagnostics.transfer_coefficient))
        self.assertTrue(pd.notna(diagnostics.risk_weighted_transfer_coefficient))
        self.assertTrue(pd.notna(diagnostics.theoretical_information_ratio))
        self.assertIsNone(diagnostics.tracking_error_target)
        self.assertEqual(diagnostics.realized_tracking_error, diagnostics.active_risk)

    def test_compute_realized_return_over_sigma_dispersion(self) -> None:
        realized_returns = pd.Series([0.2, -0.3, 0.4], index=["a", "b", "c"])
        residual_volatilities = pd.Series([0.2, 0.3, 0.4], index=["a", "b", "c"])
        
        result = compute_realized_return_over_sigma_dispersion(
            realized_returns=realized_returns,
            residual_volatilities=residual_volatilities,
        )
        
        self.assertAlmostEqual(result, 0.9428090415820634, places=10)
        
    def test_aggregate_trial_diagnostics_groups_by_tracking_error(self) -> None:
        diagnostics = [
            build_trial_diagnostics(
                scenario_name="long_only",
                tracking_error_target=0.02,
                frontier_mode="hybrid",
                optimization_method="mock",
                solver_success=True,
                forecast_alpha=pd.Series([0.03, -0.01], index=["a", "b"]),
                realized_returns=pd.Series([0.02, -0.02], index=["a", "b"]),
                residual_volatilities=pd.Series([0.2, 0.3], index=["a", "b"]),
                unconstrained_weights=pd.Series([0.5, -0.5], index=["a", "b"]),
                constrained_weights=pd.Series([0.5, 0.5], index=["a", "b"]),
                covariance=pd.DataFrame(
                    [
                        [0.04, 0.0],
                        [0.0, 0.09],
                    ],
                    index=["a", "b"],
                    columns=["a", "b"],
                ),
                breadth=25.0,
            ),
            build_trial_diagnostics(
                scenario_name="long_only",
                tracking_error_target=0.04,
                frontier_mode="hybrid",
                optimization_method="mock",
                solver_success=True,
                forecast_alpha=pd.Series([0.03, -0.01], index=["a", "b"]),
                realized_returns=pd.Series([0.02, -0.02], index=["a", "b"]),
                residual_volatilities=pd.Series([0.2, 0.3], index=["a", "b"]),
                unconstrained_weights=pd.Series([0.5, -0.5], index=["a", "b"]),
                constrained_weights=pd.Series([0.5, 0.5], index=["a", "b"]),
                covariance=pd.DataFrame(
                    [
                        [0.04, 0.0],
                        [0.0, 0.09],
                    ],
                    index=["a", "b"],
                    columns=["a", "b"],
                ),
                breadth=25.0,
            ),
        ]

        aggregated = aggregate_trial_diagnostics(diagnostics)

        self.assertIn("tracking_error_target", aggregated.columns)
        self.assertEqual(len(aggregated), 2)
        
        
if __name__ == "__main__":
    unittest.main()