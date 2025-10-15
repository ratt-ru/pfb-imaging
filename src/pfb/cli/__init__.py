"""Lightweight CLI for pfb-imaging."""

import typer
from typing import ParamSpec, TypeVar, Callable
from functools import wraps

# Main app
app = typer.Typer(
    name="hipfb",
    help="pfb-imaging: Radio interferometric imaging suite based on a preconditioned forward-backward approach",
    no_args_is_help=True,
)

# Signature inheritance utility
P = ParamSpec('P')
R = TypeVar('R')

def inherit_signature(base_func: Callable[P, R]) -> Callable[[Callable], Callable[P, R]]:
    """Inherit signature from base function."""
    def decorator(wrapper_func: Callable) -> Callable[P, R]:
        @wraps(base_func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            return wrapper_func(*args, **kwargs)
        return inner
    return decorator

# Import and register commands
from pfb.cli.restore import restore
app.command(name="restore")(restore)

from pfb.cli.grid import grid
app.command(name='grid')(grid)

from pfb.cli.hci import hci
app.command(name='hci')(hci)

from pfb.cli.sara import sara
app.command(name='sara')(sara)

__all__ = ["app", "inherit_signature"]