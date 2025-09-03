# flake8: noqa
import click
from importlib.metadata import version

from pfb import logo
from pfb.workers.init import init
from pfb.workers.grid import grid
from pfb.workers.degrid import degrid
from pfb.workers.kclean import kclean
from pfb.workers.restore import restore
from pfb.workers.model2comps import model2comps
from pfb.workers.fluxtractor import fluxtractor
from pfb.workers.hci import hci
from pfb.workers.smoovie import smoovie
from pfb.workers.sara import sara

@click.group()
@click.version_option(version=version("pfb-imaging"), prog_name='pfb')
def cli():
    logo()
    pass


# Add all the subcommands to the main CLI group.
for cmd in (init, grid, degrid, kclean, restore, model2comps, fluxtractor, hci, smoovie, sara):
    cli.add_command(cmd)

if __name__ == '__main__':
    cli()
