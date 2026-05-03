"""Runtime helpers for matplotlib-backed GUI flows."""

from matplotlib import pyplot as plt
from matplotlib.rcsetup import interactive_bk


_INTERACTIVE_BACKENDS = {backend.lower() for backend in interactive_bk}


def backend_is_interactive() -> bool:
    """Return whether the active matplotlib backend can open interactive windows."""
    return plt.get_backend().lower() in _INTERACTIVE_BACKENDS


def show_figures(block: bool = True) -> bool:
    """Show figures only when the current backend supports interactive windows."""
    if not backend_is_interactive():
        return False
    plt.show(block=block)
    return True


__all__ = ["backend_is_interactive", "show_figures"]