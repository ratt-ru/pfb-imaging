import typer
from pathlib import Path
from typing_extensions import Annotated
from hip_cargo import stimela_cab, stimela_output

app = typer.Typer()

@app.command()
@stimela_cab(
    name="pfb_restore",
    info="Pfb Restore",
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
    gausspar: Annotated[list[float] | None, typer.Option(None, help="Gaussian parameters (e-major, e-minor, position-angle) specifying the resolution to restore images to. The major and minor axes need to be specified in units of arcseconds and the position-angle in degrees. The default resolution is the native resolution in each imaging band. This parameter can be used to homogenise the resolution of the cubes. Set to (0,0,0) to use the resolution of the lowest band.")] = None,
    inflate_factor: Annotated[float, typer.Option(help="Inflate the intrinsic resolution of the uniformly blurred image by this amount.")] = 1.5,
    drop_bands: Annotated[list[int] | None, typer.Option(None, help="List of bands to discard")] = None,
    host_address: Annotated[str | None, typer.Option(None, help="Address where the distributed client lives. Uses LocalCluster if no address is provided and scheduler is set to distributed.")] = None,
    nworkers: Annotated[int, typer.Option(help="Number of worker processes. Use with distributed scheduler.")] = 1,
    nthreads: Annotated[int | None, typer.Option(None, help="Number of threads used to scale vertically (eg. for FFTs and gridding). Each dask thread can in principle spawn this many threads. Will attempt to use half the available threads by default.")] = None,
    direct_to_workers: Annotated[bool, typer.Option(help="Connect direct to workers i.e. bypass scheduler. Faster but then the dashboard isn't very useful.")] = True,
    log_level: Annotated[str, typer.Option(help="")] = "error",  # choices: ['error', 'warning', 'info', 'debug']
    output_filename: Annotated[str | None, typer.Option(..., help="Basename of output")] = None,
    log_directory: Annotated[str | None, typer.Option(None, help="Directory to write logs and performance reports to.")] = None,
    product: Annotated[str, typer.Option(help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.")] = "I",
    fits_output_folder: Annotated[str | None, typer.Option(None, help="Optional path to write fits files to. Set to output-filename if not provided. The same naming conventions apply.")] = None,
    fits_mfs: Annotated[bool, typer.Option(help="Output MFS fits files")] = True,
    fits_cubes: Annotated[bool, typer.Option(help="Output fits cubes")] = True,
):
    """
    Pfb Restore
    
    Args:
        model_name: Name of model in dds
        residual_name: Name of residual in dds
        suffix: Can be used to specify a custom name for the image space data products
        outputs: Output products (m)odel, (r)esidual, (i)mage, (c)lean beam, (d)irty, (f)ft_residuals (amplitude and phase will be produced). Use captitals to produce corresponding cubes.
        overwrite: Allow overwriting fits files
        gausspar: Gaussian parameters (e-major, e-minor, position-angle) specifying the resolution to restore images to. The major and minor axes need to be specified in units of arcseconds and the position-angle in degrees. The default resolution is the native resolution in each imaging band. This parameter can be used to homogenise the resolution of the cubes. Set to (0,0,0) to use the resolution of the lowest band.
        inflate_factor: Inflate the intrinsic resolution of the uniformly blurred image by this amount.
        drop_bands: List of bands to discard
        host_address: Address where the distributed client lives. Uses LocalCluster if no address is provided and scheduler is set to distributed.
        nworkers: Number of worker processes. Use with distributed scheduler.
        nthreads: Number of threads used to scale vertically (eg. for FFTs and gridding). Each dask thread can in principle spawn this many threads. Will attempt to use half the available threads by default.
        direct_to_workers: Connect direct to workers i.e. bypass scheduler. Faster but then the dashboard isn't very useful.
        log_level: 
        output_filename: Basename of output
        log_directory: Directory to write logs and performance reports to.
        product: String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.
        fits_output_folder: Optional path to write fits files to. Set to output-filename if not provided. The same naming conventions apply.
        fits_mfs: Output MFS fits files
        fits_cubes: Output fits cubes
    """
    # TODO: Implement function
    # Lazy import heavy dependencies here
    # from pfb.operators import my_function
    pass


if __name__ == "__main__":
    app()