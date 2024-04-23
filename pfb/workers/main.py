# flake8: noqa
import click


@click.group()
def cli():
    pass


from pfb.workers import (init, grid, degrid, klean,
                         restore, spotless, model2comps,
                         fluxmop, fastim, smoovie)
