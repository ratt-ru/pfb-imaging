# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('PLOT1GC')


@cli.command()
@click.option('-g', '--gains', required=True,
              help='Path to qcal gains.')
@click.option('-o', '--output-filename', type=str,
              help="Basename of output.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int, default=1,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int, default=1,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=int,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
def plot1gc(**kw):
    '''
    Plot effective gains produced my QuartiCal
    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _plot1gc(**args)

def _plot1gc(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)

    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 4, 'font.family': 'serif'})
    import numpy as np
    from daskms.experimental.zarr import xds_from_zarr
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    Gs = xds_from_zarr(args.gains)

    ncorr = 2
    for s, G in enumerate(Gs):
        for c in range(ncorr):
            fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(16, 12))
            for i, ax in enumerate(axs.ravel()):
                if i < 58:
                    g = G.gains.values[:, :, i, 0, c]

                    im = ax.imshow(np.abs(g), cmap='inferno', interpolation=None)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.1, width=0.1, labelsize=0.1, pad=0.1)
                else:
                    ax.axis('off')

            fig.tight_layout()

            plt.savefig(args.output_filename + f"_corr{c}_scan{s}_abs.png",
                        dpi=500, bbox_inches='tight')

            fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(16, 12))
            gref = G.gains.values[:, :, 57, 0, c]
            for i, ax in enumerate(axs.ravel()):
                if i < 58:
                    g = G.gains.values[:, :, i, 0, c] * gref.conj()

                    im = ax.imshow(np.unwrap(np.unwrap(np.angle(g), axis=0), axis=1),
                                cmap='inferno', interpolation=None)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.5, width=0.5, labelsize=0.5, pad=0.5)
                else:
                    ax.axis('off')

            fig.tight_layout()

            plt.savefig(args.output_filename + f"_corr{c}_scan{s}_phase.png",
                        dpi=500, bbox_inches='tight')

            fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(16, 12))
            for i, ax in enumerate(axs.ravel()):
                if i < 58:
                    g = G.jhj.values[:, :, i, 0, c]

                    im = ax.imshow(np.abs(g), cmap='inferno', interpolation=None)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.5, width=0.5, labelsize=0.5, pad=0.5)
                else:
                    ax.axis('off')

            fig.tight_layout()

            plt.savefig(args.output_filename + f"_corr{c}_scan{s}_jhj.png",
                        dpi=500, bbox_inches='tight')

if __name__=='__main__':
    main()

# rmse = np.sqrt(np.vdot(res, res).real/res.size)