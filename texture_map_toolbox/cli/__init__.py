"""CLI entrypoints for Texture-Map-Toolbox."""


def build_arg_parser():
	"""Lazily resolve the unified CLI parser."""
	from .main import build_arg_parser as _build_arg_parser

	return _build_arg_parser()


def main(argv=None):
	"""Lazily resolve the unified CLI entrypoint."""
	from .main import main as _main

	return _main(argv)


__all__ = ["build_arg_parser", "main"]