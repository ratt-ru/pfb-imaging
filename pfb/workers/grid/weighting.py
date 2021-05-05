# flake8: noqa

from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
log = pyscilog.get_logger('RESTORE')


@cli.command()
@click.option('-model', '--model', required=True,
              help="Path to model image cube")
def weighting():
    pass
