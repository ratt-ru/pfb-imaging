# flake8: noqa
import click


@click.group()
def cli():
    pass


from pfb.workers.post import restore
from pfb.workers.grid import dirty, psf, predict, residual
from pfb.workers.post import spifit, binterp
from pfb.workers.deconv import nnls, clean, forward
from pfb.workers.weighting import imweight
from pfb.workers.misc import jones2col, transcols, plot1gc