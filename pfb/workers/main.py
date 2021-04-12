import click


@click.group()
def cli():
    pass

from pfb.workers.post import restore
# cli.add_command(restore)

from pfb.workers.post import spi_fitter
# cli.add_command(spi_fitter)