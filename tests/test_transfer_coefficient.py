from __future__ import annotations

import unittest

from tests.test_support import add_src_to_path, has_packages

HAS_DEPS = has_packages("pandas")

if HAS_DEPS:
    import pandas as pd

add_src_to_path()

if HAS_DEPS:
    from transfercoef.transfer_coefficient import ( # noqa: E402
        build_transfer_coefficient_result,
        compute_plain_transfer_coefficient,
        compute_risk_weighted_transfer_coefficient,
        inverse_variance_risk_weights,
    )

@unittest.skipUnless(HAS_DEPS, "Requires pandas to run transfer coefficient tests.")
class TransferCoefficientTests(unittest.TestCase):
    def test_identical_weights_have_unit_transfer_coefficient(self) -> None:
        weights = pd.Series([0.2, -0.1, 0.3], index=["a", "b", "c"])
        covariance = pd.DataFrame(
            [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        result = build_transfer_coefficient_result(weights, weights, covariance=covariance)
        
        self.assertAlmostEqual(result.plain_tc, 1.0, places=10)
        self.assertAlmostEqual(result.risk_weighted_tc, 1.0, places=10)

    def test_inverse_variance_risk_weights_match_diagonal_inverse(self) -> None:
        covariance = pd.DataFrame(
            [[0.25, 0.0], [0.0, 1.0]],
            index=["a", "b"],
            columns=["a", "b"],
        )
        
        result = inverse_variance_risk_weights(covariance)
        
        self.assertAlmostEqual(result["a"], 4.0)
        self.assertAlmostEqual(result["b"], 1.0)

    def test_risk_weighted_transfer_coefficient_uses_supplied_weights(self) -> None:
        unconstrained = pd.Series([0.5, 0.5], index=["a", "b"])
        constrained = pd.Series([0.5, -0.5], index=["a", "b"])
        risk_weights = pd.Series([1.0, 1.0], index=["a", "b"])
        
        plain_tc = compute_plain_transfer_coefficient(unconstrained, constrained)
        weighted_tc = compute_risk_weighted_transfer_coefficient(
            unconstrained, constrained, risk_weights
        )
        
        self.assertAlmostEqual(plain_tc, 0.0, places=10)
        self.assertAlmostEqual(weighted_tc, 0.0, places=10)

if __name__ == "__main__":
    unittest.main()