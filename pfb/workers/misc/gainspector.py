# flake8: noqa
from contextlib import ExitStack
from pfb.workers.experimental import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('GAINSPECTOR')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.gainspector["inputs"].keys():
    defaults[key] = schema.gainspector["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.gainspector)
def gainspector(**kw):
    '''
    Plot effective gains produced my QuartiCal
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}.log')

    from glob import glob
    if opts.gain_dir is not None:
        gt = glob(opts.gain_dir)
        try:
            assert len(gt) > 0
            opts.gain_dir = gt
        except Exception as e:
            raise ValueError(f"No gain table  at {opts.gain_dir}")

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _gainspector(**opts)

def _gainspector(**kw):
    opts = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(opts.gain_dir, list) and not \
        isinstance(opts.gain_dir, ListConfig):
        opts.gain_dir = [opts.gain_dir]
    OmegaConf.set_struct(opts, True)

    # import matplotlib as mpl
    # mpl.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    import numpy as np
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from daskms.experimental.zarr import xds_from_zarr
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import xarray as xr

    Gs = []
    for gain in opts.gain_dir:
        # why do we distinguish between :: and /?
        try:
            G = xds_from_zarr(f'{gain}::{opts.gain_term}')
        except:
            G = xds_from_zarr(f'{gain}/{opts.gain_term}')
        for g in G:
            Gs.append(g)

    if opts.join_times:
        Gs = [xr.concat(Gs, dim='gain_t')]

    for s, G in enumerate(Gs):
        gain = G.gains.sortby('gain_t')
        try:
            flag = G.gain_flags.sortby('gain_t')
        except:
            flag = None
            print('No gain flags found', file=log)
        ntime, nchan, nant, ndir, ncorr = gain.shape
        if opts.ref_ant is not None:
            if opts.ref_ant == -1:
                ref_ant = nant-1
            else:
                ref_ant = opts.ref_ant
            gref = gain[:, :, ref_ant]
        else:
            gref = np.ones((ntime, nchan, ndir, ncorr))
        for c in [0,1]:
            ntot = ntime + nchan
            tlength = int(np.ceil(11 * ntime/ntot))
            fig, axs = plt.subplots(nrows=nant, ncols=1,
                                    figsize=(10, nant*tlength))
            for i, ax in enumerate(axs.ravel()):
                if i < nant:
                    if opts.mode == 'ampphase':
                        g = np.abs(gain.values[:, :, i, 0, c])
                    elif opts.mode == 'reim':
                        g = np.real(gain.values[:, :, i, 0, c])
                    else:
                        raise ValueError(f'Unknown mode {opts.mode}')
                    if flag is not None:
                        f = flag.values[:, :, i, 0]
                        It, If = np.where(f)
                        g[It, If] = np.nan

                    if opts.vmina is not None:
                        glow = opts.vmina
                        ghigh = opts.vmaxa
                    else:
                        gmed = np.nanmedian(g)
                        gstd = np.nanstd(g)
                        glow = gmed - opts.vlow * gstd
                        ghigh = gmed + opts.vhigh * gstd

                    im = ax.imshow(g,
                                   cmap='inferno', interpolation=None,
                                   vmin=glow, vmax=ghigh)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.1, width=0.1,
                                      labelsize=10.0, pad=0.1)
                else:
                    ax.axis('off')

            fig.tight_layout()
            name = 'real' if opts.mode == 'reim' else 'abs'
            plt.savefig(opts.output_filename + f"_corr{c}_scan{s}_{name}.png",
                        dpi=100, bbox_inches='tight')
            plt.close()

            fig, axs = plt.subplots(nrows=nant, ncols=1,
                                    figsize=(10, nant*tlength))

            for i, ax in enumerate(axs.ravel()):
                if i < nant:
                    if opts.mode == 'ampphase':
                        g = (gain.values[:, :, i, 0, c] *
                             gref[:, :, 0, c].conj())
                        g = np.unwrap(np.unwrap(np.angle(g), axis=0), axis=1)
                    elif opts.mode == 'reim':
                        g = np.imag(gain.values[:, :, i, 0, c])
                    else:
                        raise ValueError(f'Unknown mode {opts.mode}')

                    if flag is not None:
                        f = flag.values[:, :, i, 0]
                        It, If = np.where(f)
                        g[It, If] = np.nan

                    if opts.vminp is not None:
                        glow = opts.vminp
                        ghigh = opts.vmaxp
                    else:
                        gmed = np.nanmedian(g)
                        gstd = np.nanstd(g)
                        glow = gmed - opts.vlow * gstd
                        ghigh = gmed + opts.vhigh * gstd

                    im = ax.imshow(g, cmap='inferno', interpolation=None,
                                   vmin=glow, vmax=ghigh)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.5, width=0.5,
                                      labelsize=10, pad=0.5)
                else:
                    ax.axis('off')

            fig.tight_layout()

            name = 'imag' if opts.mode == 'reim' else 'phase'
            plt.savefig(opts.output_filename + f"_corr{c}_scan{s}_{name}.png",
                        dpi=100, bbox_inches='tight')
            plt.close()
            try:
                jhj = G.jhj.sortby('gain_t')
                fig, axs = plt.subplots(nrows=nant, ncols=1,
                                        figsize=(10, nant*tlength))
                for i, ax in enumerate(axs.ravel()):
                    if i < nant:
                        g = jhj.values[:, :, i, 0, c]
                        if flag is not None:
                            f = flag.values[:, :, i, 0]
                            It, If = np.where(f)
                            g[It, If] = np.nan

                        im = ax.imshow(np.abs(g), cmap='inferno',
                                       interpolation=None)
                        ax.set_title(f"Antenna: {i}")
                        ax.axis('off')

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("bottom", size="10%", pad=0.01)
                        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                        cb.outline.set_visible(False)
                        cb.ax.tick_params(length=0.5, width=0.5,
                                          labelsize=10, pad=0.5)
                    else:
                        ax.axis('off')

                fig.tight_layout()

                plt.savefig(opts.output_filename + f"_corr{c}_scan{s}_jhj.png",
                            dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                continue

