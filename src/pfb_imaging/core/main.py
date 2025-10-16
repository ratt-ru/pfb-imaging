# flake8: noqa
import sys
import os
import click
from importlib.metadata import version
from pfb_imaging import logo
import logging

@click.group()
@click.version_option(version=version("pfb-imaging"), prog_name='pfb')
@click.pass_context
def cli(ctx):  # ctx is passed through to workers
    logo()
    # this may be required for numba parallelism
    # find python and set LD_LIBRARY_PATH
    paths = sys.path
    ppath = [paths[i] for i in range(len(paths)) if 'pfb/bin' in paths[i]]
    if len(ppath):
        ldpath = ppath[0].replace('bin', 'lib')
        ldcurrent = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ["LD_LIBRARY_PATH"] = ':'.join([ldpath, ldcurrent]).rstrip(':')
    else:
        raise RuntimeError("Could not set LD_LIBRARY_PATH for TBB")
    
    os.environ["JAX_ENABLE_X64"] = 'True'
    os.environ["JAX_LOGGING_LEVEL"] = 'INFO'
    # these are passed through when initialising the Ray cluster
    ctx.ensure_object(dict)
    ctx.obj['env_vars'] = {
        "LD_LIBRARY_PATH": f'{ldpath}:{ldcurrent}',
        "JAX_ENABLE_X64": 'True',
        "JAX_LOGGING_LEVEL": "ERROR",
        "PYTHONWARNINGS": "ignore:.*CUDA-enabled jaxlib is not installed.*"
    }
    pass

from pfb_imaging.workers.init import init
from pfb_imaging.workers.grid import grid
from pfb_imaging.workers.degrid import degrid
from pfb_imaging.workers.kclean import kclean
from pfb_imaging.workers.restore import restore
from pfb_imaging.workers.model2comps import model2comps
from pfb_imaging.workers.fluxtractor import fluxtractor
from pfb_imaging.workers.hci import hci
from pfb_imaging.workers.smoovie import smoovie
from pfb_imaging.workers.sara import sara

# Add all the subcommands to the main CLI group.
for cmd in (init, grid, degrid, kclean, restore, model2comps, fluxtractor, hci, smoovie, sara):
    cli.add_command(cmd)

if __name__ == '__main__':
    cli()
