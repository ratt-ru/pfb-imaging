from pathlib import Path
from typing import Annotated, NewType
from typing import Literal

import typer

from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="model2comps",
    info="",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="mds-out",
    info="",
)
def model2comps(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwrite of existing model",
        ),
    ] = False,
    mds: Annotated[
        str | None,
        typer.Option(
            help="An optional input model to append to the model. "
            "Will be rendered to the same resolution as the model in the dds.",
        ),
    ] = None,
    from_fits: Annotated[
        str | None,
        typer.Option(
            help="An optional fits input model. An mds will be created that matches the model resolution.",
        ),
    ] = None,
    nbasisf: Annotated[
        int | None,
        typer.Option(
            help="Order of interpolating polynomial for frequency axis. One less than the number of bands by default.",
        ),
    ] = None,
    nbasist: Annotated[
        int,
        typer.Option(
            help="Order of interpolating polynomial for time axis. This is hypothetical for the time being.",
        ),
    ] = 1,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads",
        ),
    ] = None,
    fit_mode: Annotated[
        str,
        typer.Option(
            help="",
        ),
    ] = "Legendre",
    min_val: Annotated[
        float | None,
        typer.Option(
            help="Only fit components above this flux level",
        ),
    ] = None,
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products",
        ),
    ] = "main",
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of model in mds",
        ),
    ] = "MODEL",
    use_wsum: Annotated[
        bool,
        typer.Option(
            help="Use wsum as weights during fit",
        ),
    ] = True,
    sigmasq: Annotated[
        float,
        typer.Option(
            help="Multiple of the identity to add to the hessian for stability",
        ),
    ] = 1e-10,
    model_out: Annotated[
        str | None,
        typer.Option(
            help="Optional explicit output name. Otherwise the default naming convention is used.",
        ),
    ] = None,
    out_format: Annotated[
        Literal["zarr", "json"],
        typer.Option(
            help="Format to dump model to.",
        ),
    ] = "zarr",
    out_freqs: Annotated[
        str | None,
        typer.Option(
            help="A string flow:fhigh:step of frequencies in hertz where the output cube needs to be evaluated.",
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
    from pfb_imaging.core.model2comps import model2comps as model2comps_core  # noqa: E402

    # Call the core function with all parameters
    model2comps_core(
        output_filename,
        overwrite=overwrite,
        mds=mds,
        from_fits=from_fits,
        nbasisf=nbasisf,
        nbasist=nbasist,
        nthreads=nthreads,
        fit_mode=fit_mode,
        min_val=min_val,
        suffix=suffix,
        model_name=model_name,
        use_wsum=use_wsum,
        sigmasq=sigmasq,
        model_out=model_out,
        out_format=out_format,
        out_freqs=out_freqs,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
    )
