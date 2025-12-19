"""Lightweight CLI for pfb-imaging."""

import typer
from typing import ParamSpec, TypeVar, Callable
from functools import wraps

# Main app
app = typer.Typer(
    name="pfb",
    help="pfb-imaging: Radio interferometric imaging suite based on a preconditioned forward-backward approach",
    no_args_is_help=True,
)

# Import and register commands
from pfb_imaging.cli.degrid import degrid
app.command(name='degrid')(degrid)

from pfb_imaging.cli.fluxtractor import fluxtractor
app.command(name='fluxtractor')(fluxtractor)

from pfb_imaging.cli.grid import grid
app.command(name='grid')(grid)

from pfb_imaging.cli.hci import hci
app.command(name='hci')(hci)

from pfb_imaging.cli.init import init
app.command(name='init')(init)

from pfb_imaging.cli.kclean import kclean
app.command(name='kclean')(kclean)

from pfb_imaging.cli.model2comps import model2comps
app.command(name='model2comps')(model2comps)

from pfb_imaging.cli.restore import restore
app.command(name="restore")(restore)

from pfb_imaging.cli.sara import sara
app.command(name='sara')(sara)



__all__ = ["app"]