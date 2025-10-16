import typer
from pathlib import Path
from typing_extensions import Annotated
from hip_cargo import stimela_cab, stimela_output


@stimela_cab(
    name="pfb_restore",
    info="Restore clean components to residual images",
)
@stimela_output(
    name="mfs_image",
    dtype="File",
    info="{current.output_filename}_{current.product}_{current.suffix}_image_mfs.fits",
    required=False,
)
@stimela_output(
    name="image",
    dtype="File",
    info="{current.output_filename}_{current.product}_{current.suffix}_image.fits",
    required=False,
)
def restore(
    model_name: Annotated[str, typer.Option(help="Name of model in dds")] = "MODEL",
    residual_name: Annotated[str, typer.Option(help="Name of residual in dds")] = "RESIDUAL",
    suffix: Annotated[str, typer.Option(help="Can be used to specify a custom name for the image space data products")] = "main",
    outputs: Annotated[str, typer.Option(help="Output products (m)odel, (r)esidual, (i)mage, (c)lean beam, (d)irty, (f)ft_residuals (amplitude and phase will be produced). Use captitals to produce corresponding cubes.")] = "mMrRiI",
    overwrite: Annotated[bool, typer.Option(help="Allow overwriting fits files")] = True,
    gausspar: Annotated[list[float] | None, typer.Option(help="Gaussian parameters (e-major, e-minor, position-angle) specifying the resolution to restore images to. The major and minor axes need to be specified in units of arcseconds and the position-angle in degrees. The default resolution is the native resolution in each imaging band. This parameter can be used to homogenise the resolution of the cubes. Set to (0,0,0) to use the resolution of the lowest band.")] = None,
    inflate_factor: Annotated[float, typer.Option(help="Inflate the intrinsic resolution of the uniformly blurred image by this amount.")] = 1.5,
    drop_bands: Annotated[list[int] | None, typer.Option(help="List of bands to discard")] = None,
    host_address: Annotated[str | None, typer.Option(help="Address where the distributed client lives. Uses LocalCluster if no address is provided and scheduler is set to distributed.")] = None,
    nworkers: Annotated[int, typer.Option(help="Number of worker processes. Use with distributed scheduler.")] = 1,
    nthreads: Annotated[int | None, typer.Option(help="Number of threads used to scale vertically (eg. for FFTs and gridding). Each dask thread can in principle spawn this many threads. Will attempt to use half the available threads by default.")] = None,
    direct_to_workers: Annotated[bool, typer.Option(help="Connect direct to workers i.e. bypass scheduler. Faster but then the dashboard isn't very useful.")] = True,
    log_level: Annotated[str, typer.Option(help="Logging level")] = "error",  # choices: ['error', 'warning', 'info', 'debug']
    output_filename: Annotated[str | None, typer.Option(help="Basename of output")] = None,
    log_directory: Annotated[str | None, typer.Option(help="Directory to write logs and performance reports to.")] = None,
    product: Annotated[str, typer.Option(help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.")] = "I",
    fits_output_folder: Annotated[str | None, typer.Option(help="Optional path to write fits files to. Set to output-filename if not provided. The same naming conventions apply.")] = None,
    fits_mfs: Annotated[bool, typer.Option(help="Output MFS fits files")] = True,
    fits_cubes: Annotated[bool, typer.Option(help="Output fits cubes")] = True,
):
    # Lazy import heavy dependencies
    from pfb_imaging.workers.restore import restore as restore_impl

    if output_filename is None:
        raise typer.BadParameter("output-filename is required")
    
    return restore_impl(
        model_name=model_name,
        residual_name=residual_name,
        suffix=suffix,
        outputs=outputs,
        overwrite=overwrite,
        gausspar=gausspar,
        inflate_factor=inflate_factor,
        drop_bands=drop_bands,
        host_address=host_address,
        nworkers=nworkers,
        nthreads=nthreads,
        direct_to_workers=direct_to_workers,
        log_level=log_level,
        output_filename=output_filename,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
        fits_mfs=fits_mfs,
        fits_cubes=fits_cubes,
    )