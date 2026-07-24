"""Lightweight CLI for pfb-imaging."""

import typer
import typer.core

from pfb_imaging.cli._deprecation import warn_deprecated

# Subcommands retired in pfb-imaging v0.1.0, mapped to their replacement. The
# callback below prints a loud banner when one of these is invoked; the commands
# themselves still run for now.
DEPRECATED_COMMANDS = {
    "init": "pfb imager",
    "grid": "pfb imager",
    "kclean": "pfb deconv",
    "sara": "pfb deconv",
    "fluxtractor": "pfb deconv",
    "model2comps": "pfbspec model2comps  (pfb-model-spec package)",
}


class LogoGroup(typer.core.TyperGroup):
    """Custom Typer group that prints the logo before help."""

    def format_help(self, ctx, formatter):
        typer.echo(text)
        super().format_help(ctx, formatter)


# Main app
app = typer.Typer(
    name="pfb",
    cls=LogoGroup,
    help="pfb-imaging: Radio interferometric imaging suite based on a preconditioned forward-backward approach",
    no_args_is_help=True,
)

text = """
    ███████████  ███████████ ███████████
   ░░███░░░░░███░░███░░░░░░█░░███░░░░░███
    ░███    ░███ ░███   █ ░  ░███    ░███
    ░██████████  ░███████    ░██████████
    ░███░░░░░░   ░███░░░█    ░███░░░░░███
    ░███         ░███  ░     ░███    ░███
    █████        █████       ███████████
   ░░░░░        ░░░░░       ░░░░░░░░░░░
    """


@app.callback()
def main(ctx: typer.Context):
    """Radio interferometric imaging suite based on a preconditioned forward-backward approach."""
    typer.echo(text, err=True)
    if ctx.invoked_subcommand in DEPRECATED_COMMANDS:
        warn_deprecated(ctx.invoked_subcommand, replacement=DEPRECATED_COMMANDS[ctx.invoked_subcommand])


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

from pfb_imaging.cli.imager import imager  # noqa: E402

app.command(name="imager")(imager)

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
