import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from texture_map_toolbox import __main__ as package_main
from texture_map_toolbox.api.luma import (
    LumaExecutionRequest,
    resolve_input_image_path,
    run_luma_workflow,
)
from texture_map_toolbox.cli.main import main as cli_main
from texture_map_toolbox.gui.editor import launch_editor
from texture_map_toolbox.gui.luma_plots import plot_analysis, plot_comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "data" / "mtmtPonyTail_custom.png"


class LumaSmokeTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_default_sample_resolution_uses_custom_asset(self):
        self.assertEqual(Path(resolve_input_image_path(None)), SAMPLE_IMAGE.resolve())

    def test_original_workflow_runs(self):
        result = run_luma_workflow(
            LumaExecutionRequest(
                image_path=str(SAMPLE_IMAGE),
                algorithm="original",
                show_plots=False,
            )
        )
        self.assertEqual(result.algorithm, "original")
        self.assertIsNotNone(result.recolored_rgb_int)
        self.assertIsNotNone(result.psnr)
        self.assertEqual(result.source_image_shape, result.output_image_shape)

    def test_fast_workflow_runs(self):
        result = run_luma_workflow(
            LumaExecutionRequest(
                image_path=str(SAMPLE_IMAGE),
                algorithm="fast",
                show_plots=False,
            )
        )
        self.assertEqual(result.algorithm, "fast")
        self.assertIsNotNone(result.recolored_rgb_int)
        self.assertIsNotNone(result.preview_lut_size)
        self.assertGreater(result.output_scale, 0.0)

    def test_package_entrypoint_runs_cli(self):
        exit_code = package_main.main([
            "luma",
            str(SAMPLE_IMAGE),
            "--algorithm",
            "fast",
            "--no-plots",
            "--skip-evaluation",
        ])
        self.assertEqual(exit_code, 0)

    def test_cli_generates_output_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_image = Path(temp_dir) / "preview.png"
            result_json = Path(temp_dir) / "summary.json"

            exit_code = cli_main(
                [
                    "luma",
                    str(SAMPLE_IMAGE),
                    "--algorithm",
                    "original",
                    "--no-plots",
                    "--output-image",
                    str(output_image),
                    "--result-json",
                    str(result_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_image.exists())
            self.assertTrue(result_json.exists())

            payload = json.loads(result_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["algorithm"], "original")
            self.assertIn("psnr", payload)

    def test_cli_request_json_runs_fast_workflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request_json = Path(temp_dir) / "request.json"
            output_image = Path(temp_dir) / "preview.png"
            result_json = Path(temp_dir) / "summary.json"
            request_json.write_text(
                json.dumps(
                    {
                        "image_path": str(SAMPLE_IMAGE),
                        "algorithm": "fast",
                        "show_plots": False,
                        "output_image_path": str(output_image),
                        "result_json_path": str(result_json),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            exit_code = cli_main(["luma", "--request-json", str(request_json), "--skip-evaluation"])
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_image.exists())
            self.assertTrue(result_json.exists())

            payload = json.loads(result_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["algorithm"], "fast")

    def test_plot_helpers_smoke(self):
        result = run_luma_workflow(
            LumaExecutionRequest(
                image_path=str(SAMPLE_IMAGE),
                algorithm="original",
                show_plots=False,
            )
        )
        analysis_figure = plot_analysis(result.y_samples, result.model)
        comparison_figure = plot_comparison(
            result.rgb_float,
            result.y_eval,
            result.recolored_rgb_int,
            result.valid_mask,
            result.psnr,
            result.delta_e_image,
        )
        self.assertGreaterEqual(len(analysis_figure.axes), 4)
        self.assertGreaterEqual(len(comparison_figure.axes), 4)

    def test_editor_launch_export_and_full_resolution_render(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            curve_output_path = Path(temp_dir) / "curves.json"
            editor = launch_editor(
                str(SAMPLE_IMAGE),
                curve_output_path=str(curve_output_path),
            )
            editor._save_curves()
            editor._render_full_resolution()

            self.assertTrue(curve_output_path.exists())
            payload = json.loads(curve_output_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload.keys()), ["chroma", "hue", "lightness"])


if __name__ == "__main__":
    unittest.main()