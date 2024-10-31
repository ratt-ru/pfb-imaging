# flake8: noqa
import click
from pfb import logo
logo()

@click.group()
def cli():
    pass


from pfb.workers import (init, grid, degrid, klean,
                         restore, model2comps,
                         fluxtractor, hci, smoovie, sara)

if __name__ == '__main__':
    cli()
