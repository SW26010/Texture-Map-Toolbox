import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from texture_map_toolbox.api.luma import LumaExecutionRequest, load_image_data, run_luma_workflow
from texture_map_toolbox.gui.qt_editor import build_qt_editor, launch_qt_editor


class AlphaInputValidationTests(unittest.TestCase):
    def _write_rgb_image(self, path: Path, *, suffix: str):
        rgb = np.array(
            [
                [[255, 32, 32], [32, 255, 32], [32, 32, 255]],
                [[255, 255, 32], [255, 32, 255], [32, 255, 255]],
            ],
            dtype=np.uint8,
        )
        image = Image.fromarray(rgb, mode="RGB")
        if suffix.lower() in {".jpg", ".jpeg"}:
            image.save(path, quality=100, subsampling=0)
            return
        image.save(path)

    def _write_rgba_png(self, path: Path, alpha_plane: np.ndarray):
        rgb = np.array(
            [
                [[255, 32, 32], [32, 255, 32], [32, 32, 255]],
                [[255, 255, 32], [255, 32, 255], [32, 255, 255]],
            ],
            dtype=np.uint8,
        )
        rgba = np.dstack([rgb, np.asarray(alpha_plane, dtype=np.uint8)])
        Image.fromarray(rgba, mode="RGBA").save(path)

    def _write_grayscale_mask(self, path: Path, values: np.ndarray):
        Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)

    def _write_auto_mask_source(self, path: Path):
        rgb = np.full((8, 8, 3), 10, dtype=np.uint8)
        rgb[2:6, 2:6] = np.array([220, 140, 60], dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path)

    def _write_multi_seed_mask_source(self, path: Path):
        rgb = np.full((8, 8, 3), 220, dtype=np.uint8)
        rgb[0:2, 0:2] = np.array([10, 10, 10], dtype=np.uint8)
        rgb[0:2, 6:8] = np.array([40, 40, 40], dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path)

    def test_png_with_fully_opaque_alpha_emits_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "opaque.png"
            self._write_rgba_png(image_path, np.full((2, 3), 255, dtype=np.uint8))

            loaded_image = load_image_data(str(image_path))

            self.assertEqual(loaded_image.alpha_source, "implicit-opaque")
            self.assertTrue(loaded_image.valid_mask.all())
            self.assertTrue(loaded_image.mask_prompt_required)
            self.assertTrue(any("fully opaque" in warning for warning in loaded_image.image_warnings))

    def test_jpeg_without_alpha_emits_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "opaque.jpg"
            self._write_rgb_image(image_path, suffix=".jpg")

            loaded_image = load_image_data(str(image_path))

            self.assertEqual(loaded_image.alpha_source, "implicit-opaque")
            self.assertTrue(loaded_image.valid_mask.all())
            self.assertTrue(loaded_image.mask_prompt_required)
            self.assertTrue(any("JPEG" in warning for warning in loaded_image.image_warnings))

    def test_auto_detect_mask_uses_border_connected_background(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "auto-mask-source.png"
            self._write_auto_mask_source(image_path)

            loaded_image = load_image_data(str(image_path), auto_detect_mask=True)

            self.assertEqual(loaded_image.alpha_source, "auto-detected")
            self.assertFalse(loaded_image.mask_prompt_required)
            self.assertFalse(loaded_image.valid_mask[0, 0])
            self.assertTrue(loaded_image.valid_mask[3, 3])
            self.assertTrue(any("auto-detected border mask" in warning for warning in loaded_image.image_warnings))

    def test_seed_mask_uses_selected_connected_region(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "seed-mask-source.png"
            self._write_auto_mask_source(image_path)

            loaded_image = load_image_data(str(image_path), mask_seed_point=(0, 0))

            self.assertEqual(loaded_image.alpha_source, "interactive-seed")
            self.assertFalse(loaded_image.mask_prompt_required)
            self.assertFalse(loaded_image.valid_mask[0, 0])
            self.assertTrue(loaded_image.valid_mask[3, 3])
            self.assertTrue(any("connected-region mask" in warning for warning in loaded_image.image_warnings))

    def test_seed_mask_supports_multiple_selected_regions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "multi-seed-mask-source.png"
            self._write_multi_seed_mask_source(image_path)

            loaded_image = load_image_data(
                str(image_path),
                mask_seed_points=((0, 0), (0, 7)),
                mask_color_tolerance=0,
            )

            self.assertEqual(loaded_image.alpha_source, "interactive-seed")
            self.assertFalse(loaded_image.mask_prompt_required)
            self.assertFalse(loaded_image.valid_mask[0, 0])
            self.assertFalse(loaded_image.valid_mask[0, 7])
            self.assertTrue(loaded_image.valid_mask[4, 4])

    def test_seed_mask_region_offset_can_expand_or_shrink_selected_area(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "multi-seed-mask-source.png"
            self._write_multi_seed_mask_source(image_path)

            expanded_image = load_image_data(
                str(image_path),
                mask_seed_points=((0, 0),),
                mask_color_tolerance=0,
                mask_region_offset=1,
            )
            shrunk_image = load_image_data(
                str(image_path),
                mask_seed_points=((0, 0),),
                mask_color_tolerance=0,
                mask_region_offset=-1,
            )

            self.assertFalse(expanded_image.valid_mask[2, 2])
            self.assertTrue(expanded_image.valid_mask[4, 4])
            self.assertTrue(shrunk_image.valid_mask.all())

    def test_qt_editor_lightness_plot_shows_dashed_source_histogram_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            self._write_rgb_image(image_path, suffix=".png")

            window = build_qt_editor(str(image_path))
            try:
                baseline_x, baseline_y = window.lightness_plot._baseline_hist_item.getData()
                self.assertIsNotNone(baseline_x)
                self.assertIsNotNone(baseline_y)
                self.assertGreater(np.asarray(baseline_x).size, 0)
                self.assertGreater(np.asarray(baseline_y).size, 0)
                self.assertEqual(
                    window.lightness_plot._baseline_hist_item.opts["pen"].style(),
                    QtCore.Qt.PenStyle.DashLine,
                )
            finally:
                window.close()

    def test_external_alpha_mask_overrides_embedded_alpha(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgba_png(image_path, np.full((2, 3), 255, dtype=np.uint8))
            self._write_grayscale_mask(
                mask_path,
                np.array(
                    [
                        [0, 255, 0],
                        [255, 64, 0],
                    ],
                    dtype=np.uint8,
                ),
            )

            loaded_image = load_image_data(str(image_path), alpha_mask_path=str(mask_path))

            expected_mask = np.array(
                [
                    [False, True, False],
                    [True, True, False],
                ],
                dtype=bool,
            )
            self.assertEqual(loaded_image.alpha_source, "external-mask")
            self.assertEqual(loaded_image.alpha_mask_path, str(mask_path.resolve()))
            self.assertTrue(np.array_equal(loaded_image.valid_mask, expected_mask))

    def test_alpha_mask_size_must_match_source_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgba_png(image_path, np.full((2, 3), 255, dtype=np.uint8))
            self._write_grayscale_mask(mask_path, np.zeros((3, 2), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "must match the source image size exactly"):
                load_image_data(str(image_path), alpha_mask_path=str(mask_path))

    def test_run_luma_workflow_propagates_alpha_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "opaque.jpg"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgb_image(image_path, suffix=".jpg")
            self._write_grayscale_mask(mask_path, np.full((2, 3), 255, dtype=np.uint8))

            result = run_luma_workflow(
                LumaExecutionRequest(
                    image_path=str(image_path),
                    alpha_mask_path=str(mask_path),
                    algorithm="fast",
                    show_plots=False,
                    evaluate_result=False,
                )
            )

            self.assertEqual(result.alpha_source, "external-mask")
            self.assertEqual(result.alpha_mask_path, str(mask_path.resolve()))
            self.assertTrue(any("JPEG" in warning for warning in result.image_warnings))

    def test_qt_editor_launch_can_continue_without_extra_mask_after_seed_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "opaque.jpg"
            self._write_rgb_image(image_path, suffix=".jpg")

            window = None
            dialog = mock.Mock()
            dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
            dialog.continue_without_mask_requested.return_value = True
            dialog.selected_mask_seed_point.return_value = None
            dialog.selected_mask_seed_points.return_value = ()
            dialog.selected_mask_color_tolerance.return_value = 0
            dialog.selected_mask_region_offset.return_value = 0
            with mock.patch("texture_map_toolbox.gui.qt_editor.QtSeedMaskSelectionDialog", return_value=dialog) as dialog_cls:
                window = launch_qt_editor(str(image_path), run_event_loop=False)
                try:
                    dialog_cls.assert_called_once()
                    self.assertIsNotNone(window)
                    self.assertEqual(window.alpha_source, "implicit-opaque")
                    self.assertTrue(window.valid_mask.all())
                finally:
                    if window is not None:
                        window.close()

    def test_qt_editor_launch_can_use_seed_mask_from_prompt_dialog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "multi-seed-mask-source.png"
            self._write_multi_seed_mask_source(image_path)

            window = None
            dialog = mock.Mock()
            dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
            dialog.continue_without_mask_requested.return_value = False
            dialog.selected_mask_seed_point.return_value = (0, 0)
            dialog.selected_mask_seed_points.return_value = ((0, 0), (0, 7))
            dialog.selected_mask_color_tolerance.return_value = 0
            dialog.selected_mask_region_offset.return_value = 1
            with mock.patch("texture_map_toolbox.gui.qt_editor.QtSeedMaskSelectionDialog", return_value=dialog) as dialog_cls:
                window = launch_qt_editor(str(image_path), run_event_loop=False)
                try:
                    dialog_cls.assert_called_once()
                    self.assertIsNotNone(window)
                    self.assertEqual(window.alpha_source, "interactive-seed")
                    self.assertFalse(window.valid_mask[0, 0])
                    self.assertFalse(window.valid_mask[0, 7])
                    self.assertFalse(window.valid_mask[2, 2])
                    self.assertTrue(window.valid_mask[4, 4])
                finally:
                    if window is not None:
                        window.close()


if __name__ == "__main__":
    unittest.main()