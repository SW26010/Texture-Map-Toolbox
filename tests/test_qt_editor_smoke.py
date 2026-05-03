import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from texture_map_toolbox.api.luma import resolve_input_image_path
from texture_map_toolbox.cli import editor as editor_cli
from texture_map_toolbox.gui.qt_editor import build_qt_editor


SAMPLE_IMAGE = resolve_input_image_path(None)


class QtEditorSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()