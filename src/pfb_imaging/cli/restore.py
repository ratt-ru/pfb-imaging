from pathlib import Path
from typing import Annotated, NewType
from typing import Literal

import typer

from hip_cargo.utils.decorators import stimela_cab, stimela_output

File = NewType("File", Path)


@stimela_cab(
    name="restore",
    info="",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="File",
    name="mfs-image",
    info="",
)
@stimela_output(
    dtype="File",
    name="image",
    info="",
)
def restore(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
        ),
    ],
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of model in dds",
        ),
    ] = "MODEL",
    residual_name: Annotated[
        str,
        typer.Option(
            help="Name of residual in dds",
        ),
    ] = "RESIDUAL",
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products",
        ),
    ] = "main",
    outputs: Annotated[
        str,
        typer.Option(
            help="Output products (m)odel, (r)esidual, (i)mage, (c)lean beam, (d)irty, (f)ft_residuals (amplitude and phase will be produced). "
            "Use captitals to produce corresponding cubes.",
        ),
    ] = "mMrRiI",
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwriting fits files",
        ),
    ] = True,
    gausspar: Annotated[
        str | None,
        typer.Option(
            help="Gaussian parameters (e-major, e-minor, position-angle) specifying the resolution to restore images to. "
            "The major and minor axes need to be specified in units of arcseconds and the position-angle in degrees. "
            "The default resolution is the native resolution in each imaging band. "
            "This parameter can be used to homogenise the resolution of the cubes. "
            "Set to (0,0,0) to use the resolution of the lowest band.. "
            "Stimela dtype: List[float]",
        ),
    ] = None,
    inflate_factor: Annotated[
        float,
        typer.Option(
            help="Inflate the intrinsic resolution of the uniformly blurred image by this amount.",
        ),
    ] = 1.5,
    drop_bands: Annotated[
        str | None,
        typer.Option(
            help="List of bands to discard. Stimela dtype: List[int]",
        ),
    ] = None,
    host_address: Annotated[
        str | None,
        typer.Option(
            help="Address where the distributed client lives. "
            "Uses LocalCluster if no address is provided and scheduler is set to distributed.",
        ),
    ] = None,
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of worker processes. Use with distributed scheduler.",
        ),
    ] = 1,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads used to scale vertically (eg. "
            "for FFTs and gridding). "
            "Each dask thread can in principle spawn this many threads. "
            "Will attempt to use half the available threads by default.",
        ),
    ] = None,
    direct_to_workers: Annotated[
        bool,
        typer.Option(
            help="Connect direct to workers i.e. bypass scheduler. Faster but then the dashboard isn't very useful.",
        ),
    ] = True,
    log_level: Annotated[
        Literal["error", "warning", "info", "debug"],
        typer.Option(
            help="",
        ),
    ] = "error",
    log_directory: Annotated[
        str | None,
        typer.Option(
            help="Directory to write logs and performance reports to.",
        ),
    ] = None,
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
        ),
    ] = "I",
    fits_output_folder: Annotated[
        str | None,
        typer.Option(
            help="Optional path to write fits files to. "
            "Set to output-filename if not provided. "
            "The same naming conventions apply.",
        ),
    ] = None,
    fits_mfs: Annotated[
        bool,
        typer.Option(
            help="Output MFS fits files",
        ),
    ] = True,
    fits_cubes: Annotated[
        bool,
        typer.Option(
            help="Output fits cubes",
        ),
    ] = True,
):
    # Lazy import the core implementation
    from pfb_imaging.workers.restore import restore as restore_core  # noqa: E402

    # Parse gausspar if provided as comma-separated string
    gausspar_list = None
    if gausspar is not None:
        gausspar_list = [float(x.strip()) for x in gausspar.split(",")]

    # Parse drop_bands if provided as comma-separated string
    drop_bands_list = None
    if drop_bands is not None:
        drop_bands_list = [int(x.strip()) for x in drop_bands.split(",")]

    # Call the core function with all parameters
    restore_core(
        output_filename,
        model_name=model_name,
        residual_name=residual_name,
        suffix=suffix,
        outputs=outputs,
        overwrite=overwrite,
        gausspar=gausspar_list,
        inflate_factor=inflate_factor,
        drop_bands=drop_bands_list,
        host_address=host_address,
        nworkers=nworkers,
        nthreads=nthreads,
        direct_to_workers=direct_to_workers,
        log_level=log_level,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
        fits_mfs=fits_mfs,
        fits_cubes=fits_cubes,
    )
