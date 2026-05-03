"""CLI adapter for the Oklch state curve editor."""

import argparse

from texture_map_toolbox.api.luma import DITHER_STRENGTH, resolve_input_image_path
from texture_map_toolbox.gui.editor import launch_editor


def configure_cli_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向已有 parser 注入编辑器 CLI 参数。"""
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Input image path. Required unless a local sample image is available.",
    )
    parser.add_argument(
        "--curves",
        dest="curve_path",
        help="Optional JSON file containing initial Lt/Ct/ht control points.",
    )
    parser.add_argument(
        "--curve-output",
        dest="curve_output_path",
        help="Optional JSON file path used when exporting curves from the editor.",
    )
    parser.add_argument(
        "--dither-strength",
        type=float,
        default=DITHER_STRENGTH,
        help="Optional pre-curve dither amplitude applied on the input lightness axis.",
    )
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """构建编辑器命令行 parser。"""
    return configure_cli_parser(
        argparse.ArgumentParser(description="Interactive Oklch state-curve editor.")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析编辑器命令行参数。"""
    return build_arg_parser().parse_args(argv)


def execute_cli(args: argparse.Namespace) -> int:
    """执行编辑器 CLI。"""
    resolved_image_path = resolve_input_image_path(args.image_path)
    print(f"Loading: {resolved_image_path}")
    print("Building Oklch base model done. Opening editor...")
    editor = launch_editor(
        resolved_image_path,
        curve_path=args.curve_path,
        curve_output_path=args.curve_output_path,
        dither_strength=args.dither_strength,
    )
    editor.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    """编辑器 CLI 入口。"""
    return execute_cli(parse_args(argv))


__all__ = [
    "build_arg_parser",
    "configure_cli_parser",
    "execute_cli",
    "main",
    "parse_args",
]