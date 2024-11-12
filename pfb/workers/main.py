# flake8: noqa
import click
from pfb import logo

@click.group()
def cli():
    logo()
    pass


from pfb.workers import (init, grid, degrid, kclean,
                         restore, model2comps,
                         fluxtractor, hci, smoovie, sara)

if __name__ == '__main__':
    cli()
