# flake8: noqa
import click


@click.group()
def cli():
    pass


from pfb.workers import (init, init_ims, grid, degrid,
                         clean, restore, fwdbwd)
