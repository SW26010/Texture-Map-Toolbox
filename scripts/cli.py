"""Unified CLI entrypoint for Texture-Map-Toolbox."""

import argparse

if __package__ in {None, ""}:
    import hsl_curve_editor
    import luma_color_map
else:
    from . import hsl_curve_editor, luma_color_map


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.cli",
        description="Unified CLI for the Texture-Map-Toolbox luma LUT workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    luma_parser = subparsers.add_parser(
        "luma",
        help="Build and evaluate an Oklch-based luma color map.",
    )
    luma_color_map.configure_cli_parser(luma_parser)
    luma_parser.set_defaults(command_func=luma_color_map.execute_cli)

    editor_parser = subparsers.add_parser(
        "editor",
        help="Open the Oklch state-curve editor.",
    )
    hsl_curve_editor.configure_cli_parser(editor_parser)
    editor_parser.set_defaults(command_func=hsl_curve_editor.execute_cli)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the unified CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.command_func(args)


if __name__ == "__main__":
    raise SystemExit(main())