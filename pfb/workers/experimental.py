# flake8: noqa
import click


@click.group()
def cli():
    pass

from pfb.workers.misc import (gainspector, ift2qc, fledges, hthresh,
                              bsmooth, delay_init, forward, backward,
                              restimator, zarr2fits)
