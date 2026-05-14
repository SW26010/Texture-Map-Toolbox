import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from texture_map_toolbox.core.luma import (
    DITHER_STRENGTH,
    LumaExecutionRequest,
    apply_precurve_dither,
    infer_input_quantization_step,
    resolve_dither_strength,
    run_luma_workflow,
)


class PrecurveDitherTests(unittest.TestCase):
    def test_default_dither_strength_is_auto(self):
        self.assertIsNone(DITHER_STRENGTH)

    def test_infers_uint8_step_from_code_value_range(self):
        dtype_name, bit_depth, step = infer_input_quantization_step(
            np.zeros((2, 3, 3), dtype=np.uint8)
        )

        self.assertEqual(dtype_name, "uint8")
        self.assertEqual(bit_depth, 8)
        self.assertAlmostEqual(step, 1.0 / 255.0)

    def test_infers_uint16_step_from_code_value_range(self):
        dtype_name, bit_depth, step = infer_input_quantization_step(
            np.zeros((2, 3, 3), dtype=np.uint16)
        )

        self.assertEqual(dtype_name, "uint16")
        self.assertEqual(bit_depth, 16)
        self.assertAlmostEqual(step, 1.0 / 65535.0)

    def test_auto_dither_strength_matches_half_input_step(self):
        loaded_image = SimpleNamespace(input_quantization_step=1.0 / 255.0)

        strength, source = resolve_dither_strength(None, loaded_image)

        self.assertEqual(source, "auto-half-input-step")
        self.assertAlmostEqual(strength, 0.5 / 255.0)

    def test_workflow_auto_dither_uses_input_image_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "gradient.png"
            gradient = np.linspace(0, 255, 16, dtype=np.uint8)
            rgb = np.dstack(np.meshgrid(gradient, gradient, gradient, indexing="xy")).astype(np.uint8)
            Image.fromarray(rgb[:, :, :3], mode="RGB").save(image_path)

            result = run_luma_workflow(
                LumaExecutionRequest(
                    image_path=str(image_path),
                    algorithm="fast",
                    evaluate_result=False,
                    show_plots=False,
                )
            )

        self.assertEqual(result.input_dtype, "uint8")
        self.assertEqual(result.input_bit_depth, 8)
        self.assertAlmostEqual(result.input_quantization_step, 1.0 / 255.0)
        self.assertAlmostEqual(result.dither_strength, 0.5 / 255.0)

    def test_apply_precurve_dither_only_changes_valid_pixels(self):
        strength = 0.5 / 255.0
        y_image = np.full((6, 6), 0.5, dtype=np.float64)
        valid_mask = np.array(
            [
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
            ],
            dtype=bool,
        )

        dithered = apply_precurve_dither(y_image, valid_mask, strength)
        delta = dithered - y_image

        self.assertTrue(np.any(np.abs(delta[valid_mask]) > 0.0))
        self.assertTrue(np.all(delta[~valid_mask] == 0.0))
        self.assertLessEqual(np.max(np.abs(delta[valid_mask])), strength)

    def test_apply_precurve_dither_is_deterministic_per_seed(self):
        strength = 0.5 / 255.0
        y_image = np.full((8, 8), 0.5, dtype=np.float64)
        valid_mask = np.ones((8, 8), dtype=bool)

        first = apply_precurve_dither(y_image, valid_mask, strength, random_seed=7)
        second = apply_precurve_dither(y_image, valid_mask, strength, random_seed=7)
        third = apply_precurve_dither(y_image, valid_mask, strength, random_seed=11)

        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))
