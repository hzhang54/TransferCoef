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

