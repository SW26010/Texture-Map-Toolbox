import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from texture_map_toolbox import __main__ as package_main
from texture_map_toolbox.api.luma import resolve_input_image_path
from texture_map_toolbox.cli import editor as editor_cli
from texture_map_toolbox.core.luma import STATE_CURVE_CTRL_POINTS
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

    def _sample_plot_background_pixel_rgb(self, plot, x_value: float, y_value: float) -> tuple[int, int, int]:
        overlay_items = [
            plot.control_item,
            plot._curve_item,
            plot._default_curve_item,
            plot._hist_item,
            plot._baseline_hist_item,
            plot._reference_hist_item,
        ]
        visibility = [item.isVisible() for item in overlay_items]
        try:
            for item in overlay_items:
                item.setVisible(False)
            plot.repaint()
            QtWidgets.QApplication.processEvents()
            image = plot.viewport().grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        finally:
            for item, is_visible in zip(overlay_items, visibility):
                item.setVisible(is_visible)
            plot.repaint()
            QtWidgets.QApplication.processEvents()

        scene_point = plot.getPlotItem().vb.mapViewToScene(QtCore.QPointF(x_value, y_value))
        widget_point = plot.mapFromScene(scene_point)
        pixel_x = max(0, min(image.width() - 1, int(round(widget_point.x()))))
        pixel_y = max(0, min(image.height() - 1, int(round(widget_point.y()))))
        color = QtGui.QColor(image.pixel(pixel_x, pixel_y))
        return color.red(), color.green(), color.blue()

    def _assert_plot_background_matches_y_direction(
        self,
        plot,
        background_rgb: np.ndarray,
        y_range: tuple[float, float],
    ):
        x_index = background_rgb.shape[1] // 2
        y_min, y_max = y_range
        for fraction in (0.1, 0.9):
            y_value = y_min + (y_max - y_min) * fraction
            actual_rgb = np.asarray(
                self._sample_plot_background_pixel_rgb(plot, 0.5, y_value),
                dtype=np.float64,
            )
            row_index = int(round(fraction * (background_rgb.shape[0] - 1)))
            expected_same = background_rgb[row_index, x_index].astype(np.float64)
            expected_flipped = background_rgb[(background_rgb.shape[0] - 1) - row_index, x_index].astype(np.float64)
            same_distance = float(np.linalg.norm(actual_rgb - expected_same))
            flipped_distance = float(np.linalg.norm(actual_rgb - expected_flipped))
            self.assertLess(
                same_distance,
                flipped_distance,
                msg=(
                    f"plot background y mapping looks flipped at fraction {fraction}: "
                    f"actual={tuple(actual_rgb.astype(int))}, "
                    f"expected_same={tuple(expected_same.astype(int))}, "
                    f"expected_flipped={tuple(expected_flipped.astype(int))}"
                ),
            )

    def test_qt_launcher_builds_and_opens_editor_offscreen(self):
        launcher = build_qt_editor_launcher()
        try:
            self.assertEqual(launcher.windowTitle(), "Texture-Map-Toolbox Launcher")
            self.assertTrue(launcher.pick_region_radio.isChecked())
            self.assertEqual(launcher.seed_mask_controls.color_tolerance(), 0)
            self.assertEqual(launcher.seed_mask_controls.region_offset(), 0)
            launcher.image_path_edit.setText(SAMPLE_IMAGE)
            launcher.use_image_alpha_radio.setChecked(True)
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
                launcher.use_image_alpha_radio.setChecked(True)
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
            self.assertEqual(window.export_image_button.text(), "Export Image")
            self.assertGreater(window.lightness_plot.control_item._positions.shape[0], 1)
            self.assertEqual(window.chroma_plot.control_item._positions.shape[0], STATE_CURVE_CTRL_POINTS)
            self.assertEqual(window.hue_plot.control_item._positions.shape[0], STATE_CURVE_CTRL_POINTS)
        finally:
            window.close()

    def test_qt_editor_can_export_full_resolution_image(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "exported-preview.png"
                with mock.patch(
                    "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), "PNG Files (*.png)"),
                ), mock.patch(
                    "texture_map_toolbox.gui.qt_editor.save_luma_output_image",
                ) as save_output_image:
                    window.export_image_button.click()
                    QtWidgets.QApplication.processEvents()

                save_output_image.assert_called_once()
                saved_image, saved_path = save_output_image.call_args.args
                self.assertEqual(saved_path, str(output_path))
                self.assertEqual(saved_image.dtype, np.uint8)
                self.assertEqual(saved_image.shape[-1], 3)
                self.assertEqual(window.output_image_path, str(output_path))
                self.assertIn("Exported image:", window.status_label.text())
        finally:
            window.close()

    def test_qt_editor_defaults_use_sparse_handles_but_exact_baseline_curves(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            self.assertEqual(window.ctrl_x[1].size, STATE_CURVE_CTRL_POINTS)
            self.assertEqual(window.ctrl_x[2].size, STATE_CURVE_CTRL_POINTS)
            self.assertEqual(window.curve_point_count_spinboxes[1].value(), STATE_CURVE_CTRL_POINTS)
            self.assertFalse(any(window._curve_override_enabled))

            state_curves = window._build_state_curves()
            self.assertEqual(state_curves.chroma_points.shape[0], window.base_model.key_y.size)
            self.assertEqual(state_curves.hue_points.shape[0], window.base_model.key_y.size)
        finally:
            window.close()

    def test_qt_editor_curve_backgrounds_follow_view_y_direction(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            window.show()
            QtWidgets.QApplication.processEvents()
            state_curves = window._build_state_curves()

            self._assert_plot_background_matches_y_direction(
                window.lightness_plot,
                window._build_lightness_background(),
                (0.0, 1.0),
            )
            self._assert_plot_background_matches_y_direction(
                window.chroma_plot,
                window._build_chroma_background(state_curves),
                window.chroma_ylim,
            )
            self._assert_plot_background_matches_y_direction(
                window.hue_plot,
                window._build_hue_background(state_curves),
                (0.0, 360.0),
            )
        finally:
            window.close()

    def test_qt_editor_hue_wraps_and_display_window_can_shift(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            window.show()
            QtWidgets.QApplication.processEvents()

            hue_points = np.array(
                [
                    [0.0, 350.0],
                    [0.5, 10.0],
                    [1.0, 20.0],
                ],
                dtype=np.float64,
            )
            window._apply_control_points(2, hue_points, rerender=False, override_enabled=True)

            sampled_hue = window._sample_effective_curve_values(
                2,
                np.array([0.125, 0.25, 0.375], dtype=np.float64),
            )
            self.assertTrue(np.all((sampled_hue >= 0.0) & (sampled_hue < 360.0)))
            self.assertGreater(float(sampled_hue[0]), 350.0)
            self.assertLess(float(sampled_hue[1]), 20.0)

            window._set_hue_display_start(300)

            self.assertEqual(window.hue_display_start_slider.value(), 300)
            self.assertEqual(window._current_hue_display_range(), (300.0, 660.0))
            displayed_positions = window.hue_plot.control_item.positions()
            self.assertTrue(
                np.allclose(
                    displayed_positions[:, 1],
                    window._map_hue_values_to_display_window(window.ctrl_y[2]),
                )
            )
            self.assertGreater(float(np.max(displayed_positions[:, 1])), 360.0)
            self._assert_plot_background_matches_y_direction(
                window.hue_plot,
                window._build_hue_background(window._build_state_curves()),
                window._current_hue_display_range(),
            )
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
            self.assertTrue(window._curve_override_enabled[0])
            self.assertEqual(window.ctrl_x[0].size, STATE_CURVE_CTRL_POINTS)
            self.assertTrue((window.ctrl_y[0][1:] - window.ctrl_y[0][:-1] >= -1e-12).all())

            window.apply_chroma_target_image(SAMPLE_IMAGE)
            self.assertIn("chroma", window.target_curve_image_paths)
            self.assertTrue(window._curve_override_enabled[1])
            self.assertEqual(window.ctrl_x[1].size, STATE_CURVE_CTRL_POINTS)
            self.assertTrue(np.allclose(window.ctrl_x[1], np.linspace(0.0, 1.0, STATE_CURVE_CTRL_POINTS)))

            window.apply_hue_target_image(SAMPLE_IMAGE)
            self.assertIn("hue", window.target_curve_image_paths)
            self.assertTrue(window._curve_override_enabled[2])
            self.assertEqual(window.ctrl_x[2].size, STATE_CURVE_CTRL_POINTS)
            self.assertTrue(np.allclose(window.ctrl_x[2], np.linspace(0.0, 1.0, STATE_CURVE_CTRL_POINTS)))
        finally:
            window.close()

    def test_qt_editor_curve_controls_can_resample_drag_x_and_reset_default(self):
        window = build_qt_editor(SAMPLE_IMAGE)
        try:
            window._set_curve_point_count(1, 6)
            self.assertEqual(window.ctrl_x[1].size, 6)
            self.assertEqual(window.curve_point_count_spinboxes[1].value(), 6)
            self.assertFalse(window._curve_override_enabled[1])

            edited_points = np.column_stack([window.ctrl_x[1], window.ctrl_y[1]])
            edited_points[2, 0] = 0.42
            edited_points[2, 1] = edited_points[2, 1] + 0.05
            window._on_curve_points_changed(1, edited_points)

            self.assertTrue(window._curve_override_enabled[1])
            self.assertAlmostEqual(window.ctrl_x[1][2], 0.42, places=6)
            self.assertEqual(window._build_state_curves().chroma_points.shape[0], 6)

            window._reset_curve_to_default(1)

            self.assertFalse(window._curve_override_enabled[1])
            self.assertEqual(window.ctrl_x[1].size, 6)
            self.assertEqual(window._build_state_curves().chroma_points.shape[0], window.base_model.key_y.size)
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

    def test_qt_target_picker_defaults_to_pick_region_with_zero_mask_inputs(self):
        _ensure_qt_application()
        dialog = QtTargetImagePickerDialog(None)
        try:
            self.assertTrue(dialog.pick_region_radio.isChecked())
            self.assertEqual(dialog.selected_mask_color_tolerance(), 0)
            self.assertEqual(dialog.selected_mask_region_offset(), 0)
        finally:
            dialog.close()

    def test_qt_target_picker_seed_mode_supports_multiple_seeds_and_numeric_mask_inputs(self):
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
                self.assertIsInstance(dialog.seed_mask_controls.color_tolerance_spinbox, QtWidgets.QSpinBox)
                dialog.seed_mask_controls.color_tolerance_spinbox.setValue(0)
                self.assertIsInstance(dialog.seed_mask_controls.region_offset_spinbox, QtWidgets.QSpinBox)
                dialog.seed_mask_controls.region_offset_spinbox.setValue(1)
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
                launcher.seed_mask_controls.color_tolerance_spinbox.setValue(0)
                launcher.seed_mask_controls.region_offset_spinbox.setValue(1)
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