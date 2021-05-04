# flake8: noqa

from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
log = pyscilog.get_logger('RESTORE')


@cli.command()
@click.option('-ms', '--ms', required=True,
              help="List of paths to measurement sets")
def psf():
    pass
