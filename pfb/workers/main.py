# flake8: noqa
import click


@click.group()
def cli():
    pass


from pfb.workers import init, grid, degrid, clean, restore
from pfb.workers.deconv import fwdbwd
from pfb.workers.post import spifit, binterp
