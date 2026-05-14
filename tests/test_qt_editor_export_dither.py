import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from PySide6 import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from texture_map_toolbox.core.luma import LumaImageExportOptions
from texture_map_toolbox.gui.qt_editor import QtImageExportDialog, build_qt_editor


class QtEditorExportDitherTests(unittest.TestCase):
    def _write_rgb_image(self, path: Path):
        Image.fromarray(
            np.array(
                [
                    [[255, 32, 32], [128, 128, 128], [32, 32, 255]],
                    [[255, 255, 32], [32, 255, 32], [255, 32, 255]],
                ],
                dtype=np.uint8,
            ),
            mode="RGB",
        ).save(path)

    def _write_rgba_image(self, path: Path, alpha_plane: np.ndarray):
        rgb = np.array(
            [
                [[255, 32, 32], [128, 128, 128], [32, 32, 255]],
                [[255, 255, 32], [32, 255, 32], [255, 32, 255]],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(np.dstack([rgb, np.asarray(alpha_plane, dtype=np.uint8)]), mode="RGBA").save(path)

    def _write_grayscale_mask(self, path: Path, values: np.ndarray):
        Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)

    def _build_export_window(
        self,
        image_path: Path,
        *,
        alpha_mask_path: Path | None = None,
        dither_strength: float = 0.25,
    ):
        return build_qt_editor(
            str(image_path),
            alpha_mask_path=None if alpha_mask_path is None else str(alpha_mask_path),
            dither_strength=dither_strength,
        )

    def _build_export_dialog(self, output_path: Path, *, dither_strength: float, output_mode: str = "color"):
        dialog = mock.Mock()
        dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        dialog.selected_output_path.return_value = str(output_path)
        dialog.selected_export_options.return_value = LumaImageExportOptions(
            format_name="png",
            bit_depth=8,
            color_space="srgb",
            output_mode=output_mode,
            alpha_mode="source-alpha",
        )
        dialog.selected_dither_strength.return_value = dither_strength
        return dialog

    def _mock_export_pipeline(self, output_path: Path, *, dither_strength: float, output_mode: str = "color"):
        return mock.patch(
            "texture_map_toolbox.gui.qt_editor.QtImageExportDialog",
            return_value=self._build_export_dialog(
                output_path,
                dither_strength=dither_strength,
                output_mode=output_mode,
            ),
        ), mock.patch(
            "texture_map_toolbox.gui.qt_editor.save_luma_output_image",
        ), mock.patch(
            "texture_map_toolbox.gui.qt_editor.reconstruct_from_state_curves",
            return_value=(
                np.zeros((2, 3, 3), dtype=np.float32),
                np.zeros((2, 3, 3), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
                0,
            ),
        ), mock.patch(
            "texture_map_toolbox.gui.qt_editor.evaluate_reconstruction",
            return_value=(np.zeros((2, 3, 3), dtype=np.uint8), 42.0, np.zeros((2, 3), dtype=np.float32), {"mean": 0.0}),
        )

    def test_export_dialog_explains_half_8bit_step_reference(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = QtImageExportDialog(
            None,
            initial_output_path="output.png",
            initial_export_options=LumaImageExportOptions(format_name="png"),
            initial_dither_strength=0.5 / 255.0,
            source_bit_depth=8,
            source_quantization_step=1.0 / 255.0,
        )
        try:
            help_text = dialog.dither_help_label.text()
            self.assertIn("8-bit", help_text)
            self.assertIn("0.001961", help_text)
            self.assertIn("0.5 / 255", help_text)
        finally:
            dialog.close()
            _ = app

    def test_export_dialog_uses_save_dialog_filter_to_choose_format(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = QtImageExportDialog(
            None,
            initial_output_path="output.png",
            initial_export_options=LumaImageExportOptions(format_name="png"),
            initial_dither_strength=0.0,
        )
        try:
            with mock.patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=("output-image", "JPEG Files (*.jpg *.jpeg)"),
            ):
                dialog._browse_output_path()

            self.assertEqual(dialog.selected_export_options().format_name, "jpeg")
            self.assertTrue(dialog.selected_output_path().endswith(".jpg"))
        finally:
            dialog.close()
            _ = app

    def test_export_dialog_preserves_positive_dither_strength(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            output_path = Path(temp_dir) / "output.png"
            self._write_rgb_image(image_path)

            window = self._build_export_window(image_path, dither_strength=0.25)
            try:
                save_dialog_patch, save_image_patch, reconstruct_patch, evaluate_patch = self._mock_export_pipeline(
                    output_path,
                    dither_strength=0.25,
                )
                with save_dialog_patch, save_image_patch, reconstruct_patch as reconstruct_mock, evaluate_patch:
                    window.export_image_button.click()
                    QtWidgets.QApplication.processEvents()

                self.assertEqual(reconstruct_mock.call_args.kwargs["dither_strength"], 0.25)
                self.assertIn("dither=on", window.status_label.text())
            finally:
                window.close()

    def test_export_dialog_can_disable_dither_for_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            output_path = Path(temp_dir) / "output.png"
            self._write_rgb_image(image_path)

            window = self._build_export_window(image_path, dither_strength=0.25)
            try:
                save_dialog_patch, save_image_patch, reconstruct_patch, evaluate_patch = self._mock_export_pipeline(
                    output_path,
                    dither_strength=0.0,
                )
                with save_dialog_patch, save_image_patch, reconstruct_patch as reconstruct_mock, evaluate_patch:
                    window.export_image_button.click()
                    QtWidgets.QApplication.processEvents()

                self.assertEqual(reconstruct_mock.call_args.kwargs["dither_strength"], 0.0)
                self.assertIn("dither=off", window.status_label.text())
            finally:
                window.close()

    def test_export_dialog_passes_selected_export_options_to_image_saver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            output_path = Path(temp_dir) / "output.png"
            self._write_rgb_image(image_path)

            window = self._build_export_window(image_path, dither_strength=0.25)
            try:
                save_dialog_patch, save_image_patch, reconstruct_patch, evaluate_patch = self._mock_export_pipeline(
                    output_path,
                    dither_strength=0.125,
                    output_mode="lightness-grayscale",
                )
                with save_dialog_patch, save_image_patch as save_mock, reconstruct_patch, evaluate_patch:
                    window.export_image_button.click()
                    QtWidgets.QApplication.processEvents()

                self.assertEqual(save_mock.call_args.args[1], str(output_path))
                self.assertEqual(save_mock.call_args.kwargs["export_options"].output_mode, "lightness-grayscale")
                self.assertIsNotNone(save_mock.call_args.kwargs["lightness_image"])
                self.assertTrue(np.array_equal(save_mock.call_args.kwargs["current_mask"], window.valid_mask))
            finally:
                window.close()

    def test_source_alpha_export_uses_embedded_png_alpha_instead_of_external_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            mask_path = Path(temp_dir) / "mask.png"
            output_path = Path(temp_dir) / "output.png"
            embedded_alpha = np.array(
                [
                    [255, 128, 0],
                    [64, 255, 192],
                ],
                dtype=np.uint8,
            )
            external_mask = np.array(
                [
                    [0, 255, 0],
                    [255, 255, 0],
                ],
                dtype=np.uint8,
            )
            self._write_rgba_image(image_path, embedded_alpha)
            self._write_grayscale_mask(mask_path, external_mask)

            window = self._build_export_window(image_path, alpha_mask_path=mask_path, dither_strength=0.25)
            try:
                save_dialog_patch, save_image_patch, reconstruct_patch, evaluate_patch = self._mock_export_pipeline(
                    output_path,
                    dither_strength=0.0,
                )
                with save_dialog_patch, save_image_patch as save_mock, reconstruct_patch, evaluate_patch:
                    window.export_image_button.click()
                    QtWidgets.QApplication.processEvents()

                self.assertTrue(
                    np.allclose(
                        save_mock.call_args.kwargs["source_alpha_float"],
                        embedded_alpha.astype(np.float64) / 255.0,
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        save_mock.call_args.kwargs["current_mask"],
                        external_mask > 0,
                    )
                )
            finally:
                window.close()