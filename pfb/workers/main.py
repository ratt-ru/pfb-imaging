# flake8: noqa
import click
from importlib.metadata import version

from pfb import logo

@click.group()
@click.version_option(version=version("pfb-imaging"), prog_name='pfb')
def cli():
    logo()
    pass


from pfb.workers import (init, grid, degrid, kclean,
                         restore, model2comps,
                         fluxtractor, hci, smoovie, sara)

if __name__ == '__main__':
    cli()
