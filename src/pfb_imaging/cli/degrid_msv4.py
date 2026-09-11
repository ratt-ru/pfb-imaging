from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListStr,
    StimelaMeta,
    parse_list_str,
    parse_upath,
    stimela_cab,
    stimela_output,
)

Directory = NewType("Directory", Path)
URI = NewType("URI", Path)


@stimela_cab(
    name="degrid_msv4",
    info="Degrid visibilities from a component model into MSv4 measurement sets. "
    "The model image needs to be in component format.",
)
@stimela_output(
    dtype="Directory",
    name="log-directory",
    info="Directory to write logs and performance reports to.",
    must_exist=False,
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
@stimela_output(
    dtype="Directory",
    name="numba-cache-dir",
    info="Implicit output ensuring the numba cache location is mounted. "
    "The cache defaults to a per-user directory under the system temp directory. "
    "Override it by setting the NUMBA_CACHE_DIR environment variable.",
    implicit="/tmp/numba",
    must_exist=False,
    mkdir=False,
    path_policies={"write_parent": True},
)
def degrid_msv4(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=parse_upath,
            help="Path to measurement set.",
            rich_help_panel="Input",
        ),
        StimelaMeta(
            writable=True,
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
    channels_per_chunk: Annotated[
        int,
        typer.Option(
            ...,
            help="Number of channels per degridding chunk. "
            "Required. "
            "The model is re-rendered once per chunk so this sets how finely the model spectrum is sampled. "
            "Narrower chunks cost more FFTs at the same visibility count.",
            rich_help_panel="Chunking",
        ),
    ],
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products. "
            "This is useful for distinguishing runs with different imaging paramaters. "
            "For example, different image sizes of robustness factors.",
            rich_help_panel="Naming",
        ),
    ] = "main",
    mds: Annotated[
        str | None,
        typer.Option(
            help="Optional path to mds to use for degridding. "
            "By default it is inferred from output-filename and suffix. "
            "Both the deconv name and the model2comps name are tried.",
            rich_help_panel="Input",
        ),
    ] = None,
    model_column: Annotated[
        str,
        typer.Option(
            help="Column to write model data to",
            rich_help_panel="Output",
        ),
    ] = "MODEL_DATA",
    product: Annotated[
        str,
        typer.Option(
            help="Stokes product to degrid. "
            "Must name exactly one product and must match the model's own stokes attribute. "
            "The genesis mds spec stores a single Stokes plane.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
    scan_names: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="List of scan names to degrid. "
            "Defaults to all. "
            "These are MSv4 scan_name values, not SCAN_NUMBER integers. "
            "Input as a comma separated list if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    spw_names: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="List of spectral window names to degrid. "
            "Defaults to all. "
            "These are MSv4 spectral_window_name values, not DATA_DESC_ID integers. "
            "Input as a comma separated list if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    field_names: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="List of field names to degrid. "
            "Defaults to all. "
            "These are MSv4 field_name values, not FIELD_ID integers. "
            "Input as a comma separated list if running from CLI.",
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
    data_group: Annotated[
        str,
        typer.Option(
            help="MSv4 data group used to resolve the 'DATA' column to its correlated_data variable. "
            "Also selects the field_and_source subtable.",
            rich_help_panel="Data Selection",
        ),
    ] = "base",
    partition_columns: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="Columns to partition the MSv4 store by (xarray-ms PARTITION_SCHEMA). "
            "Defaults to FIELD_ID,DATA_DESC_ID,SCAN_NUMBER; other instruments may need SOURCE_ID. "
            "Input as a comma separated list if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    integrations_per_chunk: Annotated[
        int,
        typer.Option(
            help="Number of time integrations per degridding chunk. "
            "Default -1 degrids the whole partition in one chunk. "
            "This is a memory and parallelism knob only.",
            rich_help_panel="Chunking",
        ),
    ] = -1,
    accumulate: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto model column",
            rich_help_panel="Output",
        ),
    ] = False,
    region_file: Annotated[
        str | None,
        typer.Option(
            help="A region file containing regions that need to be converted to separate measurement set columns. "
            "Each region in the file will end up in a separate column labelled as model-column{#}. "
            "The remainder of the fields goes into model-column.",
            rich_help_panel="Input",
        ),
    ] = None,
    epsilon: Annotated[
        float,
        typer.Option(
            help="Gridder accuracy",
            rich_help_panel="WGridder",
        ),
    ] = 1e-07,
    do_wgridding: Annotated[
        bool,
        typer.Option(
            help="Perform w-correction via improved w-stacking",
            rich_help_panel="WGridder",
        ),
    ] = True,
    ray_address: Annotated[
        str,
        typer.Option(
            help="Address of the ray cluster to connect to. If not provided, will run locally.",
            rich_help_panel="Performance",
        ),
    ] = "local",
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
    progressbar: Annotated[
        bool,
        typer.Option(
            help="Display progress. Use --no-progressbar to deactivate.",
            rich_help_panel="Reporting",
        ),
    ] = True,
    log_directory: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Directory to write logs and performance reports to.",
            rich_help_panel="Output",
        ),
        StimelaMeta(
            must_exist=False,
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
    Degrid visibilities from a component model into MSv4 measurement sets.
    The model image needs to be in component format.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                degrid_msv4,
                dict(
                    ms=ms,
                    output_filename=output_filename,
                    channels_per_chunk=channels_per_chunk,
                    suffix=suffix,
                    mds=mds,
                    model_column=model_column,
                    product=product,
                    scan_names=scan_names,
                    spw_names=spw_names,
                    field_names=field_names,
                    freq_range=freq_range,
                    data_group=data_group,
                    partition_columns=partition_columns,
                    integrations_per_chunk=integrations_per_chunk,
                    accumulate=accumulate,
                    region_file=region_file,
                    epsilon=epsilon,
                    do_wgridding=do_wgridding,
                    ray_address=ray_address,
                    nworkers=nworkers,
                    nthreads=nthreads,
                    progressbar=progressbar,
                    log_directory=log_directory,
                ),
            )

            # Lazy import the core implementation
            from pfb_imaging.core.degrid_msv4 import degrid_msv4 as degrid_msv4_core  # noqa: E402

            # Call the core function with all parameters
            degrid_msv4_core(
                ms,
                output_filename,
                channels_per_chunk,
                suffix=suffix,
                mds=mds,
                model_column=model_column,
                product=product,
                scan_names=scan_names,
                spw_names=spw_names,
                field_names=field_names,
                freq_range=freq_range,
                data_group=data_group,
                partition_columns=partition_columns,
                integrations_per_chunk=integrations_per_chunk,
                accumulate=accumulate,
                region_file=region_file,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                ray_address=ray_address,
                nworkers=nworkers,
                nthreads=nthreads,
                progressbar=progressbar,
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
        degrid_msv4,
        dict(
            ms=ms,
            output_filename=output_filename,
            channels_per_chunk=channels_per_chunk,
            suffix=suffix,
            mds=mds,
            model_column=model_column,
            product=product,
            scan_names=scan_names,
            spw_names=spw_names,
            field_names=field_names,
            freq_range=freq_range,
            data_group=data_group,
            partition_columns=partition_columns,
            integrations_per_chunk=integrations_per_chunk,
            accumulate=accumulate,
            region_file=region_file,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            ray_address=ray_address,
            nworkers=nworkers,
            nthreads=nthreads,
            progressbar=progressbar,
            log_directory=log_directory,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
