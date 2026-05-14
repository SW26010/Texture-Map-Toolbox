import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from PySide6 import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from texture_map_toolbox.gui.qt_editor import build_qt_editor


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

    def _build_export_window(self, image_path: Path, *, dither_strength: float = 0.25):
        return build_qt_editor(str(image_path), dither_strength=dither_strength)

    def _mock_export_pipeline(self, output_path: Path):
        return mock.patch(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(output_path), "PNG Files (*.png)"),
        ), mock.patch(
            "texture_map_toolbox.gui.qt_editor.save_luma_output_image",
        ), mock.patch(
            "texture_map_toolbox.gui.qt_editor.reconstruct_from_state_curves",
            return_value=(np.zeros((2, 3, 3), dtype=np.float32), None, np.zeros((2, 3), dtype=np.float32), 0),
        ), mock.patch(
            "texture_map_toolbox.gui.qt_editor.evaluate_reconstruction",
            return_value=(np.zeros((2, 3, 3), dtype=np.uint8), 42.0, np.zeros((2, 3), dtype=np.float32), {"mean": 0.0}),
        )

    def test_export_checkbox_defaults_to_enabled_when_dither_strength_is_positive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            output_path = Path(temp_dir) / "output.png"
            self._write_rgb_image(image_path)

            window = self._build_export_window(image_path, dither_strength=0.25)
            try:
                self.assertIsNotNone(window.export_dither_checkbox)
                self.assertTrue(window.export_dither_checkbox.isChecked())
                save_dialog_patch, save_image_patch, reconstruct_patch, evaluate_patch = self._mock_export_pipeline(
                    output_path,
                )
                with save_dialog_patch, save_image_patch, reconstruct_patch as reconstruct_mock, evaluate_patch:
                    window.export_image_button.click()
                    QtWidgets.QApplication.processEvents()

                self.assertEqual(reconstruct_mock.call_args.kwargs["dither_strength"], 0.25)
                self.assertIn("dither=on", window.status_label.text())
            finally:
                window.close()

    def test_export_checkbox_can_disable_dither_for_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            output_path = Path(temp_dir) / "output.png"
            self._write_rgb_image(image_path)

            window = self._build_export_window(image_path, dither_strength=0.25)
            try:
                self.assertIsNotNone(window.export_dither_checkbox)
                window.export_dither_checkbox.setChecked(False)
                save_dialog_patch, save_image_patch, reconstruct_patch, evaluate_patch = self._mock_export_pipeline(
                    output_path,
                )
                with save_dialog_patch, save_image_patch, reconstruct_patch as reconstruct_mock, evaluate_patch:
                    window.export_image_button.click()
                    QtWidgets.QApplication.processEvents()

                self.assertEqual(reconstruct_mock.call_args.kwargs["dither_strength"], 0.0)
                self.assertIn("dither=off", window.status_label.text())
            finally:
                window.close()