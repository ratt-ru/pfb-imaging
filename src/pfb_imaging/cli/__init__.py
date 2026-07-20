"""Lightweight CLI for pfb-imaging."""

import typer
import typer.core


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
def main():
    """Radio interferometric imaging suite based on a preconditioned forward-backward approach."""
    typer.echo(text, err=True)


# Import and register commands
from pfb_imaging.cli.degrid import degrid  # noqa: E402

app.command(name="degrid")(degrid)

from pfb_imaging.cli.hci import hci  # noqa: E402

app.command(name="hci")(hci)

from pfb_imaging.cli.imager import imager  # noqa: E402

app.command(name="imager")(imager)

from pfb_imaging.cli.restore import restore  # noqa: E402

app.command(name="restore")(restore)

from pfb_imaging.cli.deconv import deconv  # noqa: E402

app.command(name="deconv")(deconv)

__all__ = ["app"]
