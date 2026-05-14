import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from texture_map_toolbox.core.luma import LumaImageExportOptions, save_luma_output_image


class ImageExportOptionTests(unittest.TestCase):
    def test_save_luma_output_image_can_write_png_with_current_mask_alpha(self):
        rgb_float = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
            ],
            dtype=np.float64,
        )
        current_mask = np.array(
            [
                [True, False],
                [True, True],
            ],
            dtype=bool,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "masked.png"
            save_luma_output_image(
                rgb_float,
                str(output_path),
                export_options=LumaImageExportOptions(format_name="png", alpha_mode="current-mask"),
                current_mask=current_mask,
            )

            with Image.open(output_path) as exported_image:
                exported = np.asarray(exported_image)

            self.assertEqual(exported.shape[-1], 4)
            self.assertTrue(
                np.array_equal(
                    exported[:, :, 3],
                    np.array(
                        [
                            [255, 0],
                            [255, 255],
                        ],
                        dtype=np.uint8,
                    ),
                )
            )

    def test_save_luma_output_image_can_write_16bit_lightness_grayscale_png(self):
        lightness = np.array(
            [
                [0.0, 0.5],
                [1.0, 0.25],
            ],
            dtype=np.float64,
        )
        rgb_float = np.repeat(lightness[:, :, None], 3, axis=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "lightness.png"
            save_luma_output_image(
                rgb_float,
                str(output_path),
                export_options=LumaImageExportOptions(
                    format_name="png",
                    bit_depth=16,
                    output_mode="lightness-grayscale",
                ),
                lightness_image=lightness,
            )

            with Image.open(output_path) as exported_image:
                exported = np.asarray(exported_image)

            self.assertEqual(exported.dtype, np.uint16)
            self.assertEqual(exported.ndim, 2)
            self.assertEqual(int(exported[0, 0]), 0)
            self.assertEqual(int(exported[1, 0]), 65535)
            self.assertAlmostEqual(exported[0, 1] / 65535.0, 0.5, delta=1.0 / 65535.0)

    def test_save_luma_output_image_rejects_16bit_color_export(self):
        rgb_float = np.array(
            [
                [[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]],
            ],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "color16.png"
            with self.assertRaisesRegex(ValueError, "single-channel L grayscale"):
                save_luma_output_image(
                    rgb_float,
                    str(output_path),
                    export_options=LumaImageExportOptions(format_name="png", bit_depth=16),
                )


if __name__ == "__main__":
    unittest.main()