from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import ListInt, StimelaMeta, parse_list_int, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
URI = NewType("URI", Path)


@stimela_cab(
    name="init",
    info="Initialise Stokes data products.",
)
@stimela_output(
    dtype="Directory",
    name="xds-out",
    info="Output dataset directory.",
    implicit="{current.output-filename}_{current.product}.xds",
    must_exist=False,
)
@stimela_output(
    dtype="Directory",
    name="log-directory",
    info="Directory to write logs and performance reports to.",
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
def init(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=Path,
            help="Path to measurement set",
            rich_help_panel="Input",
        ),
    ],
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
            rich_help_panel="Naming",
        ),
    ],
    scans: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of SCAN_NUMBERS to image. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    ddids: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of DATA_DESC_ID's to images. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    fields: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="List of FIELD_ID's to image. Defaults to all. Input as comma separated list 0,2 if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    freq_range: Annotated[
        str | None,
        typer.Option(
            help="Frequency range to image in Hz. Specify as a string with colon delimiter ('1e9:1.1e9').",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwrite of output xds",
            rich_help_panel="Control",
        ),
    ] = False,
    data_column: Annotated[
        str,
        typer.Option(
            help="Data column to image. "
            "Must be the same across MSs. "
            "Simple arithmetic is supported ('CORRECTED_DATA-MODEL_DATA'). "
            "When gains are present this column will be corrected.",
            rich_help_panel="Data Selection",
        ),
    ] = "DATA",
    weight_column: Annotated[
        str | None,
        typer.Option(
            help="Column containing natural weights. Must be the same across MSs",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    sigma_column: Annotated[
        str | None,
        typer.Option(
            help="Column containing standard devations. "
            "Will be used to initialise natural weights if detected. "
            "Must be the same across MSs",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    flag_column: Annotated[
        str,
        typer.Option(
            help="Column containing data flags. Must be the same across MSs",
            rich_help_panel="Data Selection",
        ),
    ] = "FLAG",
    gain_table: Annotated[
        list[URI] | None,
        typer.Option(
            parser=Path,
            help="Path to Quartical gain table containing NET gains. "
            "There must be a table for each MS and glob(ms) and glob(gt) should match up when running from CLI.",
            rich_help_panel="Input",
        ),
    ] = None,
    integrations_per_image: Annotated[
        int,
        typer.Option(
            help="Number of time integrations per image. Default (-1, 0, None) -> dataset per scan.",
            rich_help_panel="Imaging",
        ),
    ] = -1,
    channels_per_image: Annotated[
        int,
        typer.Option(
            help="Number of channels per image for degridding resolution. "
            "Any of (-1, 0, None) implies single dataset per spw.",
            rich_help_panel="Imaging",
        ),
    ] = -1,
    precision: Annotated[
        Literal["single", "double"],
        typer.Option(
            help="Gridding precision",
            rich_help_panel="Imaging",
        ),
    ] = "double",
    bda_decorr: Annotated[
        float,
        typer.Option(
            help="BDA decorrelation factor. Only has an effect if less than one",
            rich_help_panel="Averaging",
        ),
    ] = 1.0,
    max_field_of_view: Annotated[
        float,
        typer.Option(
            help="The maximum field of view that will be considered. Used to compute decorrelation due to BDA.",
            rich_help_panel="Averaging",
        ),
    ] = 3.0,
    beam_model: Annotated[
        str | None,
        typer.Option(
            help="Beam model to use. Only katbeam currently supported.",
            rich_help_panel="Input",
        ),
    ] = None,
    chan_average: Annotated[
        int,
        typer.Option(
            help="Average this number if channels together",
            rich_help_panel="Averaging",
        ),
    ] = 1,
    progressbar: Annotated[
        bool,
        typer.Option(
            help="Display progress. Use --no-progressbar to deactivate.",
            rich_help_panel="Reporting",
        ),
    ] = True,
    check_ants: Annotated[
        bool,
        typer.Option(
            help="Check that ANTENNA1 and ANTENNA2 tables are consistent with the ANTENNA table.",
            rich_help_panel="Control",
        ),
    ] = False,
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of worker processes. Use with distributed scheduler.",
            rich_help_panel="Performance",
        ),
    ] = 1,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads used to scale vertically (for FFTs and gridding). "
            "Each dask thread can in principle spawn this many threads. "
            "Will attempt to use half the available threads by default.",
            rich_help_panel="Performance",
        ),
    ] = None,
    wgt_mode: Annotated[
        Literal["l2", "minvar"],
        typer.Option(
            help="Controls how the Stokes weights are computed. "
            "l2 -> use standard Gaussian formula. "
            "minvar -> use minimum between correlations (wsclean Stokes I style).",
            rich_help_panel="Weighting",
        ),
    ] = "l2",
    log_directory: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="Directory to write logs and performance reports to.",
            rich_help_panel="Output",
        ),
        StimelaMeta(
            mkdir=False,
            path_policies={
                "write_parent": True,
            },
        ),
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
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
                product=product,
                nworkers=nworkers,
                nthreads=nthreads,
                wgt_mode=wgt_mode,
                log_directory=log_directory,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("pfb-imaging")
    if image is None:
        raise RuntimeError("No Container URL in pfb-imaging metadata.")

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
            product=product,
            nworkers=nworkers,
            nthreads=nthreads,
            wgt_mode=wgt_mode,
            log_directory=log_directory,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
