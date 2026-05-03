"""Unified CLI entrypoint for Texture-Map-Toolbox."""

import argparse
import sys

from texture_map_toolbox.cli import editor, luma


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m texture_map_toolbox",
        description="Unified CLI for the Texture-Map-Toolbox Oklch workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    luma_parser = subparsers.add_parser(
        "luma",
        help="Build and evaluate an Oklch-based luma color map.",
    )
    luma.configure_cli_parser(luma_parser)
    luma_parser.set_defaults(command_func=luma.execute_cli)

    editor_parser = subparsers.add_parser(
        "editor",
        help="Open the Oklch state-curve editor.",
    )
    editor.configure_cli_parser(editor_parser)
    editor_parser.set_defaults(command_func=editor.execute_cli)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the unified CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from texture_map_toolbox.gui.qt_editor import launch_qt_editor_launcher

        launch_qt_editor_launcher()
        return 0

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.command_func(args)


if __name__ == "__main__":
    raise SystemExit(main())