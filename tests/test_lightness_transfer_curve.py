import unittest

import numpy as np
from scipy.interpolate import PchipInterpolator

from texture_map_toolbox.api.luma import fit_monotonic_lightness_transfer_curve


class LightnessTransferCurveTests(unittest.TestCase):
    def test_fit_returns_monotonic_control_points(self):
        source = np.linspace(0.0, 1.0, 257)
        target = np.linspace(0.0, 1.0, 1025) ** 2

        control_points = fit_monotonic_lightness_transfer_curve(
            source,
            target,
            quantile_count=128,
        )

        self.assertGreaterEqual(control_points.shape[0], 2)
        self.assertEqual(control_points.shape[1], 2)
        self.assertTrue(np.all(np.diff(control_points[:, 0]) > 0.0))
        self.assertTrue(np.all(np.diff(control_points[:, 1]) >= -1e-12))
        self.assertAlmostEqual(float(control_points[0, 0]), 0.0)
        self.assertAlmostEqual(float(control_points[-1, 0]), 1.0)

    def test_fit_reduces_quantile_mismatch(self):
        rng = np.random.default_rng(0)
        source = rng.beta(2.0, 5.0, 4096)
        target = rng.beta(5.0, 2.0, 3072)

        control_points = fit_monotonic_lightness_transfer_curve(
            source,
            target,
            quantile_count=256,
        )
        interpolator = PchipInterpolator(control_points[:, 0], control_points[:, 1], extrapolate=True)
        transformed = np.clip(interpolator(source), 0.0, 1.0)

        quantiles = np.linspace(0.0, 1.0, 65)
        baseline_error = float(
            np.mean(np.abs(np.quantile(source, quantiles) - np.quantile(target, quantiles)))
        )
        transformed_error = float(
            np.mean(np.abs(np.quantile(transformed, quantiles) - np.quantile(target, quantiles)))
        )

        self.assertLess(transformed_error, baseline_error * 0.1)

    def test_fit_handles_repeated_source_values(self):
        source = np.full(128, 0.5, dtype=np.float64)
        target = np.linspace(0.0, 1.0, 513, dtype=np.float64)

        control_points = fit_monotonic_lightness_transfer_curve(
            source,
            target,
            quantile_count=64,
        )

        self.assertGreaterEqual(control_points.shape[0], 2)
        self.assertTrue(np.all(np.diff(control_points[:, 0]) > 0.0))
        self.assertTrue(np.all(np.diff(control_points[:, 1]) >= -1e-12))
        self.assertTrue(np.all((control_points[:, 1] >= 0.0) & (control_points[:, 1] <= 1.0)))


if __name__ == "__main__":
    unittest.main()