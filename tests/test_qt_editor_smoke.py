import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from PySide6 import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from texture_map_toolbox import __main__ as package_main
from texture_map_toolbox.api.luma import resolve_input_image_path
from texture_map_toolbox.cli import editor as editor_cli
from texture_map_toolbox.gui.qt_editor import (
    QtTargetImagePickerDialog,
    _ensure_qt_application,
    build_qt_editor,
    build_qt_editor_launcher,
)


SAMPLE_IMAGE = resolve_input_image_path(None)


class QtEditorSmokeTests(unittest.TestCase):
    def _write_rgba_image(self, path: Path):
        rgb = np.array(
            [
                [[255, 32, 32], [32, 255, 32], [32, 32, 255]],
                [[255, 255, 32], [255, 32, 255], [32, 255, 255]],
            ],
            dtype=np.uint8,
        )
        alpha = np.array(
            [
                [0, 255, 255],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(path)

    def _write_grayscale_mask(self, path: Path):
        Image.fromarray(
            np.array(
                [
                    [0, 255, 255],
                    [0, 255, 255],
                ],
                dtype=np.uint8,
            ),
            mode="L",
        ).save(path)

    def _write_auto_mask_source(self, path: Path):
        rgb = np.full((8, 8, 3), 10, dtype=np.uint8)
        rgb[2:6, 2:6] = np.array([220, 140, 60], dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path)

    def _write_multi_seed_mask_source(self, path: Path):
        rgb = np.full((8, 8, 3), 220, dtype=np.uint8)
        rgb[0:2, 0:2] = np.array([10, 10, 10], dtype=np.uint8)
        rgb[0:2, 6:8] = np.array([40, 40, 40], dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path)

    def test_qt_launcher_builds_and_opens_editor_offscreen(self):
        launcher = build_qt_editor_launcher()
        try:
            self.assertEqual(launcher.windowTitle(), "Texture-Map-Toolbox Launcher")
            launcher.image_path_edit.setText(SAMPLE_IMAGE)
            editor_window = launcher.launch_selected_editor(show_window=False)
            self.assertIsNotNone(editor_window)
            try:
                self.assertEqual(editor_window.image_path, resolve_input_image_path(SAMPLE_IMAGE))
                self.assertEqual(editor_window._preview_buf.shape[-1], 3)
            finally:
                editor_window.close()
        finally:
            launcher.close()

    def test_qt_launcher_closes_after_opening_editor_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            self._write_rgba_image(image_path)

            launcher = build_qt_editor_launcher(image_path=str(image_path))
            editor_window = None
            try:
                launcher.show()
                editor_window = launcher.launch_selected_editor(show_window=True)
                self.assertIsNotNone(editor_window)
                self.assertFalse(launcher.isVisible())
            finally:
                if editor_window is not None:
                    editor_window.close()
                launcher.close()

    def test_qt_editor_builds_offscreen(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            self.assertEqual(window.windowTitle(), "Texture-Map-Toolbox Qt MVP")
            self.assertEqual(window._preview_buf.shape[-1], 3)
            self.assertGreater(window.lightness_plot.control_item._positions.shape[0], 1)
            self.assertGreater(window.chroma_plot.control_item._positions.shape[0], 1)
            self.assertGreater(window.hue_plot.control_item._positions.shape[0], 1)
        finally:
            window.close()

    def test_editor_cli_dispatches_to_qt_backend(self):
        with mock.patch("texture_map_toolbox.gui.qt_editor.launch_qt_editor") as launch_qt_editor:
            args = editor_cli.parse_args([
                SAMPLE_IMAGE,
                "--backend",
                "qt",
            ])
            exit_code = editor_cli.execute_cli(args)

        self.assertEqual(exit_code, 0)
        launch_qt_editor.assert_called_once()

    def test_editor_cli_without_image_launches_qt_launcher(self):
        with mock.patch("texture_map_toolbox.gui.qt_editor.launch_qt_editor_launcher") as launch_qt_editor_launcher:
            args = editor_cli.parse_args([
                "--backend",
                "qt",
            ])
            exit_code = editor_cli.execute_cli(args)

        self.assertEqual(exit_code, 0)
        launch_qt_editor_launcher.assert_called_once()

    def test_package_main_without_args_launches_qt_launcher(self):
        with mock.patch("texture_map_toolbox.gui.qt_editor.launch_qt_editor_launcher") as launch_qt_editor_launcher:
            exit_code = package_main.main([])

        self.assertEqual(exit_code, 0)
        launch_qt_editor_launcher.assert_called_once()

    def test_qt_editor_loads_per_curve_target_images(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            window.apply_lightness_target_image(SAMPLE_IMAGE)
            self.assertIn("lightness", window.target_curve_image_paths)
            self.assertIsNotNone(window._lightness_reference_histogram)
            self.assertTrue((window.ctrl_y[0][1:] - window.ctrl_y[0][:-1] >= -1e-12).all())

            window.apply_chroma_target_image(SAMPLE_IMAGE)
            self.assertIn("chroma", window.target_curve_image_paths)
            self.assertTrue((window.ctrl_x[1] == window.base_model.key_y).all())
            self.assertTrue((window.ctrl_y[1] == window.base_model.key_c).all())

            window.apply_hue_target_image(SAMPLE_IMAGE)
            self.assertIn("hue", window.target_curve_image_paths)
            self.assertTrue((window.ctrl_x[2] == window.base_model.key_y).all())
            self.assertTrue((window.ctrl_y[2] == window.base_model.key_h).all())
        finally:
            window.close()

    def test_qt_editor_applies_single_target_image_to_multiple_curves(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            window.apply_target_image_selection(
                SAMPLE_IMAGE,
                apply_lightness=True,
                apply_chroma=True,
                apply_hue=True,
            )
            self.assertEqual(
                set(window.target_curve_image_paths),
                {"lightness", "chroma", "hue"},
            )
        finally:
            window.close()

    def test_qt_target_image_selection_uses_external_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "target.png"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgba_image(target_path)
            self._write_grayscale_mask(mask_path)

            window = build_qt_editor(SAMPLE_IMAGE)
            try:
                loaded_image, _, _ = window._load_target_curve_model(
                    str(target_path),
                    alpha_mask_path=str(mask_path),
                    show_warnings=False,
                )
                self.assertEqual(loaded_image.alpha_source, "external-mask")
                self.assertEqual(loaded_image.alpha_mask_path, str(mask_path.resolve()))

                window.apply_target_image_selection(
                    str(target_path),
                    alpha_mask_path=str(mask_path),
                    apply_lightness=True,
                    apply_chroma=False,
                    apply_hue=False,
                    show_warnings=False,
                )
                self.assertEqual(window.target_curve_mask_paths["lightness"], str(mask_path.resolve()))
            finally:
                window.close()

    def test_qt_target_picker_shows_live_image_and_mask_previews(self):
        _ensure_qt_application()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "target.png"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgba_image(image_path)
            self._write_grayscale_mask(mask_path)

            dialog = QtTargetImagePickerDialog(
                None,
                initial_image_path=str(image_path),
                initial_mask_path=str(mask_path),
                initial_mask_mode="external",
            )
            try:
                self.assertTrue(dialog.load_mask_radio.isChecked())
                self.assertIsNotNone(dialog.image_preview_label._pixmap)
                self.assertIsNotNone(dialog.mask_preview_label._pixmap)
                self.assertIn("Shape: 3 x 2", dialog.image_preview_info_label.text())
                self.assertIn("Matches selected image size", dialog.mask_preview_info_label.text())
                self.assertEqual(dialog.selected_mask_path(), str(mask_path))
            finally:
                dialog.close()

    def test_qt_target_picker_seed_mode_shows_mask_preview_after_click(self):
        _ensure_qt_application()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "auto-mask-source.png"
            self._write_auto_mask_source(image_path)

            dialog = QtTargetImagePickerDialog(
                None,
                initial_image_path=str(image_path),
                initial_mask_mode="interactive-seed",
            )
            try:
                self.assertTrue(dialog.pick_region_radio.isChecked())
                dialog._handle_image_preview_click(0, 0)
                self.assertIsNotNone(dialog.mask_preview_label._pixmap)
                self.assertIn("connected-region", dialog.mask_preview_info_label.text())
                self.assertEqual(dialog.selected_mask_seed_point(), (0, 0))
            finally:
                dialog.close()

    def test_qt_target_picker_seed_mode_supports_multiple_seeds_and_sliders(self):
        _ensure_qt_application()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "multi-seed-mask-source.png"
            self._write_multi_seed_mask_source(image_path)

            dialog = QtTargetImagePickerDialog(
                None,
                initial_image_path=str(image_path),
                initial_mask_mode="interactive-seed",
            )
            try:
                dialog.seed_mask_controls.color_tolerance_slider.setValue(0)
                dialog.seed_mask_controls.region_offset_slider.setValue(1)
                dialog._handle_image_preview_click(0, 0)
                dialog._handle_image_preview_click(0, 7)

                self.assertEqual(dialog.selected_mask_seed_points(), ((0, 0), (0, 7)))
                self.assertEqual(dialog.image_preview_label._marker_points, ((0, 0), (0, 7)))
                self.assertIn("Color tolerance: 0", dialog.mask_preview_info_label.text())
                self.assertIn("Region offset: +1 px", dialog.mask_preview_info_label.text())
            finally:
                dialog.close()

    def test_qt_target_picker_browse_mask_updates_live_preview(self):
        _ensure_qt_application()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "target.png"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgba_image(image_path)
            self._write_grayscale_mask(mask_path)

            dialog = QtTargetImagePickerDialog(None, initial_image_path=str(image_path))
            try:
                with mock.patch(
                    "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                    return_value=(str(mask_path), "Images (*.png)"),
                ):
                    dialog._browse_mask()
                self.assertTrue(dialog.load_mask_radio.isChecked())
                self.assertEqual(dialog.selected_mask_path(), str(mask_path))
                self.assertIsNotNone(dialog.mask_preview_label._pixmap)
            finally:
                dialog.close()

    def test_qt_launcher_shows_live_image_and_mask_previews(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgba_image(image_path)
            self._write_grayscale_mask(mask_path)

            launcher = build_qt_editor_launcher(image_path=str(image_path), alpha_mask_path=str(mask_path))
            try:
                self.assertTrue(launcher.load_mask_radio.isChecked())
                self.assertIsNotNone(launcher.input_image_preview_label._pixmap)
                self.assertIsNotNone(launcher.mask_preview_label._pixmap)
                self.assertIn("Shape: 3 x 2", launcher.input_image_info_label.text())
                self.assertIn("Matches selected image size", launcher.mask_preview_info_label.text())
                self.assertEqual(launcher.alpha_mask_path_edit.text(), str(mask_path))
            finally:
                launcher.close()

    def test_qt_launcher_browse_mask_updates_live_preview(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            mask_path = Path(temp_dir) / "mask.png"
            self._write_rgba_image(image_path)
            self._write_grayscale_mask(mask_path)

            launcher = build_qt_editor_launcher(image_path=str(image_path))
            try:
                with mock.patch(
                    "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                    return_value=(str(mask_path), "Images (*.png)"),
                ):
                    launcher._browse_alpha_mask()
                self.assertTrue(launcher.load_mask_radio.isChecked())
                self.assertEqual(launcher.alpha_mask_path_edit.text(), str(mask_path))
                self.assertIsNotNone(launcher.mask_preview_label._pixmap)
            finally:
                launcher.close()

    def test_qt_launcher_seed_mode_previews_and_opens_editor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "multi-seed-mask-source.png"
            self._write_multi_seed_mask_source(image_path)

            launcher = build_qt_editor_launcher(image_path=str(image_path))
            editor_window = None
            try:
                launcher.pick_region_radio.setChecked(True)
                launcher.seed_mask_controls.color_tolerance_slider.setValue(0)
                launcher.seed_mask_controls.region_offset_slider.setValue(1)
                launcher._handle_image_preview_click(0, 0)
                launcher._handle_image_preview_click(0, 7)
                self.assertIn("connected-region", launcher.mask_preview_info_label.text())
                self.assertEqual(launcher.input_image_preview_label._marker_points, ((0, 0), (0, 7)))
                editor_window = launcher.launch_selected_editor(show_window=False)
                self.assertIsNotNone(editor_window)
                self.assertEqual(editor_window.alpha_source, "interactive-seed")
                self.assertFalse(editor_window.valid_mask[0, 0])
                self.assertFalse(editor_window.valid_mask[0, 7])
                self.assertFalse(editor_window.valid_mask[2, 2])
                self.assertTrue(editor_window.valid_mask[4, 4])
            finally:
                if editor_window is not None:
                    editor_window.close()
                launcher.close()


if __name__ == "__main__":
    unittest.main()