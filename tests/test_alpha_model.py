from __future__ import annotations

import unittest

from tests.test_support import add_src_to_path, has_packages

HAS_DEPS = has_packages("numpy")

if HAS_DEPS:
    import numpy as np

add_src_to_path()

if HAS_DEPS:
    from transfercoef.alpha_model import ( # noqa: E402
        AlphaGenerationInputs,
        build_asset_names,
        compute_alpha_over_ic_dispersion,
        generate_alpha_sample,
        generate_forecast_alpha,
        generate_residual_volatilities,
        generate_standard_normal_score,
    )

@unittest.skipUnless(HAS_DEPS, "Requires numpy to run alpha model tests.")
class AlphaModelTests(unittest.TestCase):
    def test_generate_standard_normal_score_respects_requested_shape(self) -> None:
        rng = np.random.default_rng(123)
        asset_names = build_asset_names(5)

        result = generate_standard_normal_score(asset_names, rng=rng)

        self.assertEqual(len(result), 5)
        self.assertListEqual(list(result.index), asset_names)
    
    def test_generate_forecast_alpha_rejects_invalid_ic(self) -> None:
        score_vector = generate_standard_normal_score(build_asset_names(3), rng=np.random.default_rng(123))
        residual_volatilities = generate_residual_volatilities(build_asset_names(3), residual_volatility=0.10)
        
        with self.assertRaises(ValueError):
            generate_forecast_alpha(
                score_vector=score_vector,
                residual_volatilities=residual_volatilities,
                target_ic=1.5,
            )

    def test_generate_alpha_sample_produces_eq9_style_dispersion(self) -> None:
        rng = np.random.default_rng(123)
        inputs = AlphaGenerationInputs(
            asset_names=build_asset_names(50),
            alpha_volatility=0.02,
            residual_volatility=0.10,
            target_ic=0.10,
        )

        sample = generate_alpha_sample(inputs=inputs, rng=rng)

        self.assertEqual(len(sample.score_vector), 50)
        self.assertTrue(np.isfinite(sample.realized_forecast_to_return_corr))
        self.assertTrue(np.isfinite(sample.alpha_over_ic_dispersion))
        self.assertAlmostEqual(
            sample.alpha_over_ic_dispersion,
            float(sample.residual_volatilities.mean()),
            places=2,
        )

    def test_compute_alpha_over_ic_dispersion_scales_by_ic(self) -> None:
        forecast_alpha = np.array([0.01, -0.01, 0.02, -0.02])
        dispersion = compute_alpha_over_ic_dispersion(
            forecast_alpha=np.array(forecast_alpha), # type: ignore[arg-type]
            target_ic=0.5,
        )

        self.assertTrue(np.isfinite(dispersion))

if __name__ == "__main__":
    unittest.main()
