from __future__ import annotations

import unittest
from unittest.mock import patch
from tests.test_support import add_src_to_path, has_packages

HAS_DEPS = has_packages("numpy", "pandas")

if HAS_DEPS:
    import pandas as pd

add_src_to_path()

if HAS_DEPS:
    from transfercoef.config import ScenarioConfig #noqa: E402
    from transfercoef.portfolio_optimizer import ( #noqa: E402
        OptimizationInputs,
        _normalize_weights,
        apply_simple_constraints,
        approximate_constrained_weights,
        build_holdings_from_weights,
        extract_post_trade_weights,
        optimize_scenario,
        solve_unconstrained_weights,
    )

@unittest.skipUnless(HAS_DEPS, "requires numpy and pandas to run optimizer tests.")
class PortfolioOptimizerTests(unittest.TestCase):
    def test_normalize_weights_targets_gross_leverage(self) -> None:
        weights = pd.Series([2.0, -1.0, 1.0], index=["a", "b", "c"])

        normalized = _normalize_weights(weights, leverage_limit=1.5)

        self.assertAlmostEqual(float(normalized.abs().sum()), 1.5, places=10)

    def test_solve_unconstrained_weights_returns_common_assets_only(self) -> None:
        forecast_alpha = pd.Series([0.03, -0.01, 0.02], index=["a","b","c"])
        covariance = pd.DataFrame(
            [[0.04, 0.0],[0.0, 0.09]],
            index=["a","b"],
            columns=["a","b"]
        )
        inputs = OptimizationInputs(
            forecast_alpha=forecast_alpha,
            covariance=covariance,
            risk_aversion=1.0,
        )

        result = solve_unconstrained_weights(inputs)

        self.assertListEqual(list(result.index), ["a","b"])
        self.assertAlmostEqual(float(result.abs().sum()), 1.0, places=10)
        
    def test_apply_simple_constraints_enforces_long_only_and_cap(self) -> None:
        weights = pd.Series([0.7, -0.2, 0.1], index=["a","b","c"])
        scenario = ScenarioConfig(
            name="long_only_cap",
            description="test scenario",
            long_only=True,
            leverage_limit=1.0,
            max_weight=0.5,
        )

        constrained = apply_simple_constraints(weights, scenario)

        self.assertTrue((constrained >= 0.0).all())
        self.assertTrue((constrained <= 1.0).all())
        self.assertAlmostEqual(float(constrained.abs().sum()), 1.0, places=10)

    def test_approximate_constrained_weights_respects_turnover_limit(self) -> None:
        unconstrained = pd.Series([0.8, 0.2], index=["a","b"])
        previous = pd.Series([0.2, 0.8], index=["a","b"])
        scenario = ScenarioConfig(
            name="turnover",
            description="Turnover limited scenario",
            leverage_limit=1.0,
            turnover_limit=0.1,
        )

        constrained = approximate_constrained_weights(unconstrained, scenario, previous_weights=previous)
        turnover = 0.5 * float((constrained - previous).abs().sum())
        
        self.assertLessEqual(turnover, 0.1 + 1e-10)
            
    def test_build_holdings_from_weights_adds_cash_balance(self) -> None:
        previous_weights = pd.Series([0.25, 0.35], index=["a","b"])

        holdings = build_holdings_from_weights(
            asset_names=["a","b"],
            portfolio_value=100.0,
            previous_weights=previous_weights,
        )

        self.assertAlmostEqual(float(holdings.sum()), 100.0, places=10)
        self.assertIn("cash", holdings.index)
        self.assertAlmostEqual(holdings["cash"], 40.0, places=10)

    def test_extract_post_trade_weights_drops_cash(self) -> None:
        holdings = pd.Series([20.0, 30.0, 50.0], index=["a","b","cash"])
        trades = pd.Series([10.0, -10.0, 0.0], index=["a","b","cash"])
        
        weights = extract_post_trade_weights(trades=trades, holdings=holdings)

        self.assertListEqual(list(weights.index), ["a","b"])
        self.assertAlmostEqual(float(weights.sum()), 0.5, places=10)


    def test_te_constrained_runs_do_not_use_fallback(self) -> None:
        scenario = ScenarioConfig(name="long_only", description="Test scenario", long_only=True)
        inputs = OptimizationInputs(
            forecast_alpha=pd.Series([0.03, 0.01], index=["a","b"]),
            covariance=pd.DataFrame(
                [[0.04, 0.0],[0.0, 0.09]],
                index=["a","b"],
                columns=["a","b"]
            ),
            risk_aversion=1.0,
            tracking_error_target=0.05,
            frontier_mode="hybrid",
        )

        with patch("transfercoef.portfolio_optimizer.execute_real_policy", side_effect=RuntimeError("solver failed")):
            with self.assertRaises(RuntimeError):
                optimize_scenario(project_root=".", scenario=scenario, inputs=inputs)
        
    def test_non_te_runs_can_fallback(self) -> None:
        scenario = ScenarioConfig(name="long_only", description="Test scenario", long_only=True)
        inputs = OptimizationInputs(
            forecast_alpha=pd.Series([0.03, 0.01], index=["a","b"]),
            covariance=pd.DataFrame(
                [[0.04, 0.0],[0.0, 0.09]],
                index=["a","b"],
                columns=["a","b"]
            ),
            risk_aversion=1.0,
        )

        with patch("transfercoef.portfolio_optimizer.execute_real_policy", side_effect=RuntimeError("solver failed")):
            result = optimize_scenario(project_root=".", scenario=scenario, inputs=inputs)

        self.assertEqual(result.method, "approximate_constraints_fallback")
        self.assertTrue(bool(result.metadata["fallback_used"]))

if __name__ == "__main__":
    unittest.main()