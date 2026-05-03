import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from texture_map_toolbox import __main__ as package_main
from texture_map_toolbox.api.luma import resolve_input_image_path
from texture_map_toolbox.cli import editor as editor_cli
from texture_map_toolbox.gui.qt_editor import build_qt_editor, build_qt_editor_launcher


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


if __name__ == "__main__":
    unittest.main()