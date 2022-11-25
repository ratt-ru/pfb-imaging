# flake8: noqa
import click


@click.group()
def cli():
    pass


from pfb.workers import (init, grid, degrid,
                         clean, restore, fwdbwd, spotless)
