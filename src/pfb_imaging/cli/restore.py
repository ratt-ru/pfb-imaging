from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo.utils.decorators import stimela_cab, stimela_output

File = NewType("File", Path)


@stimela_cab(
    name="restore",
    info="",
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
            help="Output products (m)odel, (r)esidual, (i)mage, (c)lean beam, (d)irty, (f)ft_residuals. "
            "Use capitals to produce corresponding cubes.",
        ),
    ] = "mMrRiI",
    gausspar: Annotated[
        str | None,
        typer.Option(
            help="Gaussian parameters (e-major, e-minor, position-angle) specifying restoring resolution. "
            "The major and minor axes need to be specified in units of arcseconds. "
            "The position-angle should be in degrees. "
            "The default resolution is the native resolution in each imaging band. "
            "This parameter can be used to homogenise the resolution of the cubes. "
            "Set to (0,0,0) to use the resolution of the lowest band. "
            "Stimela dtype: List[float]",
        ),
    ] = None,
    drop_bands: Annotated[
        str | None,
        typer.Option(
            help="List of bands to discard. Stimela dtype: List[int]",
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
            help="Number of threads used to scale vertically (for FFTs and gridding). "
            "Each dask thread can in principle spawn this many threads. "
            "Will attempt to use half the available threads by default.",
        ),
    ] = None,
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
):
    # Lazy import the core implementation
    from pfb_imaging.core.restore import restore as restore_core  # noqa: E402

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
        gausspar=gausspar_list,
        drop_bands=drop_bands_list,
        nworkers=nworkers,
        nthreads=nthreads,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
    )
