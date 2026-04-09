"""Lightweight CLI for pfb-imaging."""

import typer

print(
    """
    ███████████  ███████████ ███████████
   ░░███░░░░░███░░███░░░░░░█░░███░░░░░███
    ░███    ░███ ░███   █ ░  ░███    ░███
    ░██████████  ░███████    ░██████████
    ░███░░░░░░   ░███░░░█    ░███░░░░░███
    ░███         ░███  ░     ░███    ░███
    █████        █████       ███████████
   ░░░░░        ░░░░░       ░░░░░░░░░░░
    """
)

# Main app
app = typer.Typer(
    name="pfb",
    help="pfb-imaging: Radio interferometric imaging suite based on a preconditioned forward-backward approach",
    no_args_is_help=True,
)

# Import and register commands
from pfb_imaging.cli.degrid import degrid  # noqa: E402

app.command(name="degrid")(degrid)

from pfb_imaging.cli.fluxtractor import fluxtractor  # noqa: E402

app.command(name="fluxtractor")(fluxtractor)

from pfb_imaging.cli.grid import grid  # noqa: E402

app.command(name="grid")(grid)

from pfb_imaging.cli.hci import hci  # noqa: E402

app.command(name="hci")(hci)

from pfb_imaging.cli.init import init  # noqa: E402

app.command(name="init")(init)

from pfb_imaging.cli.kclean import kclean  # noqa: E402

app.command(name="kclean")(kclean)

from pfb_imaging.cli.model2comps import model2comps  # noqa: E402

app.command(name="model2comps")(model2comps)

from pfb_imaging.cli.restore import restore  # noqa: E402

app.command(name="restore")(restore)

from pfb_imaging.cli.sara import sara  # noqa: E402

app.command(name="sara")(sara)

from pfb_imaging.cli.deconv import deconv  # noqa: E402

app.command(name="deconv")(deconv)

__all__ = ["app"]
