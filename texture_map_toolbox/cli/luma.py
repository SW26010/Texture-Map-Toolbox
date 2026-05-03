"""CLI adapter for the Oklch luma workflows."""

import argparse

import matplotlib.pyplot as plt

from texture_map_toolbox.api.luma import (
    LumaExecutionRequest,
    LumaExecutionResult,
    SUPPORTED_LUMA_ALGORITHMS,
    load_luma_request_json,
    normalize_luma_execution_request,
    run_luma_workflow,
    save_luma_output_image,
    summarize_luma_result,
    write_luma_summary_json,
)
from texture_map_toolbox.gui.matplotlib_runtime import show_figures
from texture_map_toolbox.gui.luma_plots import plot_analysis, plot_comparison


def print_luma_summary(result: LumaExecutionResult):
    """打印主流程摘要。"""
    summary = summarize_luma_result(result)
    print(f"Algorithm: {summary['algorithm']}")
    print("Input axis: original Oklch Lightness (L0)")
    print(f"Curve source: {summary['curve_source']}")
    print(
        "Image shape: "
        f"source={tuple(summary['source_image_shape'])}, "
        f"output={tuple(summary['output_image_shape'])}, "
        f"scale={summary['output_scale']:.4f}"
    )
    print(f"Quantile keypoints: {summary['keypoints']}")
    print(
        "State-curve control points: "
        f"L={summary['state_curve_points']['lightness']}, "
        f"C={summary['state_curve_points']['chroma']}, "
        f"h={summary['state_curve_points']['hue']}"
    )
    print(f"Pre-curve dither strength: {summary['dither_strength']:.6f}")
    if summary["preview_lut_size"] is not None:
        print(f"Preview LUT size: {summary['preview_lut_size']}")
    print(f"Gamut-compressed pixels: {summary['gamut_compressed_pixels']}")
    if result.gamut_compressed_lut_entries is not None:
        print(f"Gamut-compressed LUT entries: {result.gamut_compressed_lut_entries}")
    if result.psnr is not None:
        print(f"PSNR (Peak Signal-to-Noise Ratio): {result.psnr:.2f} dB")
    if result.delta_e_stats is not None:
        for key, value in result.delta_e_stats.items():
            print(f"Delta E 2000 ({key.replace('_', ' ').title()}): {value:.2f}")
    else:
        print("Evaluation: skipped")


def build_luma_request_from_args(args: argparse.Namespace) -> LumaExecutionRequest:
    """将 CLI 参数折叠成统一请求对象。"""
    if args.request_json:
        request = load_luma_request_json(args.request_json)
    else:
        algorithm = args.algorithm or "original"
        request = LumaExecutionRequest(
            algorithm=algorithm,
            evaluate_result=algorithm == "original",
        )

    if args.image_path is not None:
        request.image_path = args.image_path
    if args.curve_path is not None:
        request.curve_path = args.curve_path
    if args.algorithm is not None:
        request.algorithm = args.algorithm
    if args.dither_strength is not None:
        request.dither_strength = args.dither_strength
    if args.preview_scale is not None:
        request.preview_scale = args.preview_scale
    if args.preview_lut_size is not None:
        request.preview_lut_size = args.preview_lut_size
    if args.output_image is not None:
        request.output_image_path = args.output_image
    if args.result_json is not None:
        request.result_json_path = args.result_json
    if args.skip_evaluation:
        request.evaluate_result = False
    if args.no_plots:
        request.show_plots = False

    return normalize_luma_execution_request(request)


def configure_cli_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向已有 parser 注入主流程 CLI 参数。"""
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Input image path. Required unless a local sample image is available.",
    )
    parser.add_argument(
        "--request-json",
        help="Optional JSON file containing a serialized luma execution request.",
    )
    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_LUMA_ALGORITHMS,
        help="Select the algorithm: `original` keeps the offline high-quality path, `fast` uses the shared preview LUT path.",
    )
    parser.add_argument(
        "--curves",
        dest="curve_path",
        help="Optional JSON file containing lightness/chroma/hue control points.",
    )
    parser.add_argument(
        "--dither-strength",
        type=float,
        help="Optional pre-curve dither amplitude applied on the input lightness axis.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        help="Preview downsample ratio used by the shared fast algorithm.",
    )
    parser.add_argument(
        "--preview-lut-size",
        type=int,
        help="Preview LUT size used by the shared fast algorithm.",
    )
    parser.add_argument(
        "--output-image",
        help="Optional output image path used to save the recolored result.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not open matplotlib comparison or analysis plots.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip PSNR and Delta E evaluation to keep the run generation-only.",
    )
    parser.add_argument(
        "--result-json",
        "--summary-json",
        dest="result_json",
        help="Optional JSON file path used to write the run result summary.",
    )
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """构建主流程命令行 parser。"""
    return configure_cli_parser(
        argparse.ArgumentParser(
            description="Run the Oklch luma workflow through the original offline algorithm or the shared fast preview LUT algorithm."
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    return build_arg_parser().parse_args(argv)


def execute_cli(args: argparse.Namespace) -> int:
    """执行主流程 CLI。"""
    request = build_luma_request_from_args(args)
    result = run_luma_workflow(request)
    print_luma_summary(result)

    if request.output_image_path:
        save_luma_output_image(result.recolored_rgb_int, request.output_image_path)
        result.output_image_path = request.output_image_path
        print(f"Output image: {request.output_image_path}")

    if request.result_json_path:
        write_luma_summary_json(result, request.result_json_path)
        print(f"Result JSON: {request.result_json_path}")

    if request.show_plots:
        if result.recolored_rgb_int is not None and result.delta_e_image is not None and result.psnr is not None:
            plot_comparison(
                result.rgb_float,
                result.y_eval,
                result.recolored_rgb_int,
                result.valid_mask,
                result.psnr,
                result.delta_e_image,
            )
        plot_analysis(result.y_samples, result.model)
        show_figures()

    return 0


def main(argv: list[str] | None = None) -> int:
    """主流程 CLI 入口。"""
    return execute_cli(parse_args(argv))


__all__ = [
    "build_arg_parser",
    "build_luma_request_from_args",
    "configure_cli_parser",
    "execute_cli",
    "main",
    "parse_args",
    "print_luma_summary",
]