from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import ListInt, parse_list_int, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
URI = NewType("URI", Path)


@stimela_cab(
    name="init",
    info="Initialise Stokes data products.",
    image="ghcr.io/ratt-ru/pfb-imaging:dependabotprs",
)
@stimela_output(
    dtype="Directory",
    name="xds-out",
    info="Output dataset directory.",
    implicit="{current.output-filename}_{current.product}.xds",
    must_exist=False,
)
def init(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=Path,
            help="Path to measurement set",
        ),
    ],
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
        ),
    ],
    scans: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of SCAN_NUMBERS to image. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI.",
        ),
    ] = None,
    ddids: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of DATA_DESC_ID's to images. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI.",
        ),
    ] = None,
    fields: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of FIELD_ID's to image. Defaults to all. Input as comma separated list 0,2 if running from CLI.",
        ),
    ] = None,
    freq_range: Annotated[
        str | None,
        typer.Option(
            help="Frequency range to image in Hz. Specify as a string with colon delimiter ('1e9:1.1e9').",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwrite of output xds",
        ),
    ] = False,
    data_column: Annotated[
        str,
        typer.Option(
            help="Data column to image. "
            "Must be the same across MSs. "
            "Simple arithmetic is supported ('CORRECTED_DATA-MODEL_DATA'). "
            "When gains are present this column will be corrected.",
        ),
    ] = "DATA",
    weight_column: Annotated[
        str | None,
        typer.Option(
            help="Column containing natural weights. Must be the same across MSs",
        ),
    ] = None,
    sigma_column: Annotated[
        str | None,
        typer.Option(
            help="Column containing standard devations. "
            "Will be used to initialise natural weights if detected. "
            "Must be the same across MSs",
        ),
    ] = None,
    flag_column: Annotated[
        str,
        typer.Option(
            help="Column containing data flags. Must be the same across MSs",
        ),
    ] = "FLAG",
    gain_table: Annotated[
        list[URI] | None,
        typer.Option(
            parser=Path,
            help="Path to Quartical gain table containing NET gains. "
            "There must be a table for each MS and glob(ms) and glob(gt) should match up when running from CLI.",
        ),
    ] = None,
    integrations_per_image: Annotated[
        int,
        typer.Option(
            help="Number of time integrations per image. Default (-1, 0, None) -> dataset per scan.",
        ),
    ] = -1,
    channels_per_image: Annotated[
        int,
        typer.Option(
            help="Number of channels per image for degridding resolution. "
            "Any of (-1, 0, None) implies single dataset per spw.",
        ),
    ] = -1,
    precision: Annotated[
        Literal["single", "double"],
        typer.Option(
            help="Gridding precision",
        ),
    ] = "double",
    bda_decorr: Annotated[
        float,
        typer.Option(
            help="BDA decorrelation factor. Only has an effect if less than one",
        ),
    ] = 1.0,
    max_field_of_view: Annotated[
        float,
        typer.Option(
            help="The maximum field of view that will be considered. Used to compute decorrelation due to BDA.",
        ),
    ] = 3.0,
    beam_model: Annotated[
        str | None,
        typer.Option(
            help="Beam model to use. Only katbeam currently supported.",
        ),
    ] = None,
    chan_average: Annotated[
        int,
        typer.Option(
            help="Average this number if channels together",
        ),
    ] = 1,
    progressbar: Annotated[
        bool,
        typer.Option(
            help="Display progress. Use --no-progressbar to deactivate.",
        ),
    ] = True,
    check_ants: Annotated[
        bool,
        typer.Option(
            help="Check that ANTENNA1 and ANTENNA2 tables are consistent with the ANTENNA table.",
        ),
    ] = False,
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
    wgt_mode: Annotated[
        Literal["l2", "minvar"],
        typer.Option(
            help="Controls how the Stokes weights are computed. "
            "l2 -> use standard Gaussian formula. "
            "minvar -> use minimum between correlations (wsclean Stokes I style).",
        ),
    ] = "l2",
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        {"stimela": {"skip": True}},
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        {"stimela": {"skip": True}},
    ] = False,
):
    """
    Initialise Stokes data products.
    """
    if backend == "native" or backend == "auto":
        try:
            # Lazy import the core implementation
            from pfb_imaging.core.init import init as init_core  # noqa: E402

            # Call the core function with all parameters
            init_core(
                ms,
                output_filename,
                scans=scans,
                ddids=ddids,
                fields=fields,
                freq_range=freq_range,
                overwrite=overwrite,
                data_column=data_column,
                weight_column=weight_column,
                sigma_column=sigma_column,
                flag_column=flag_column,
                gain_table=gain_table,
                integrations_per_image=integrations_per_image,
                channels_per_image=channels_per_image,
                precision=precision,
                bda_decorr=bda_decorr,
                max_field_of_view=max_field_of_view,
                beam_model=beam_model,
                chan_average=chan_average,
                progressbar=progressbar,
                check_ants=check_ants,
                log_directory=log_directory,
                product=product,
                nworkers=nworkers,
                nthreads=nthreads,
                wgt_mode=wgt_mode,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Fall back to container execution
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    run_in_container(
        init,
        dict(
            ms=ms,
            output_filename=output_filename,
            scans=scans,
            ddids=ddids,
            fields=fields,
            freq_range=freq_range,
            overwrite=overwrite,
            data_column=data_column,
            weight_column=weight_column,
            sigma_column=sigma_column,
            flag_column=flag_column,
            gain_table=gain_table,
            integrations_per_image=integrations_per_image,
            channels_per_image=channels_per_image,
            precision=precision,
            bda_decorr=bda_decorr,
            max_field_of_view=max_field_of_view,
            beam_model=beam_model,
            chan_average=chan_average,
            progressbar=progressbar,
            check_ants=check_ants,
            log_directory=log_directory,
            product=product,
            nworkers=nworkers,
            nthreads=nthreads,
            wgt_mode=wgt_mode,
        ),
        backend=backend,
        always_pull_images=always_pull_images,
    )
