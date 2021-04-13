import click


@click.group()
def cli():
    pass

from pfb.workers.post import restore
from pfb.workers.post import spi_fitter
from pfb.workers.grid import dirty, psf, dirty_and_psf, weighting
