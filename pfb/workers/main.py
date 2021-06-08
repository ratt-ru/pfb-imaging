# flake8: noqa
import click


@click.group()
def cli():
    pass


from pfb.workers.post import restore
from pfb.workers.grid import dirty, psf, predict
from pfb.workers.post import spi_fitter, fluxmop
from pfb.workers.deconv import nnls, clean
from pfb.workers.weighting import imweight