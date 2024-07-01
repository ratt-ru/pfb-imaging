# flake8: noqa
import click
from pfb import logo
logo()

@click.group()
def cli():
    pass


from pfb.workers import (init, grid, degrid, klean,
                         restore, spotless, model2comps,
                         fluxmop, hci, smoovie, sara)

if __name__ == '__main__':
    from pfb.workers.spotless import spotless
    cli.add_command(spotless)
    cli()
