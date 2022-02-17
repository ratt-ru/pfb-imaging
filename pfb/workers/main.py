# flake8: noqa
import click


@click.group()
def cli():
    pass


from pfb.workers.post import restore
from pfb.workers import init, grid, degrid
from pfb.workers.post import spifit, binterp
from pfb.workers.deconv import nnls, clean, forward, backward
from pfb.workers.weighting import imweight
from pfb.workers.misc import transcols, plot1gc, sim_noise
