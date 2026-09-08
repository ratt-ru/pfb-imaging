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
    name="imager",
    info="Initialise Stokes data products.",
)
@stimela_output(
    dtype="Directory",
    name="dt-out",
    info="Output imaging DataTree directory.",
    implicit="{current.output-filename}_{current.product}.dt",
    must_exist=True,
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
    "The cache defaults to a per-user directory under /tmp. "
    "Override it by setting the NUMBA_CACHE_DIR environment variable.",
    implicit="/tmp/numba-cache",
    must_exist=False,
    mkdir=False,
    path_policies={"write_parent": True},
)
@stimela_output(
    dtype="Directory",
    name="beam-cache-dir",
    info="Implicit output ensuring the beam cache location is mounted. "
    "The cache defaults to a per-user directory under /tmp. "
    "Override it by setting the MBEAMS_CACHE_DIR environment variable.",
    implicit="/tmp/mbeams-cache",
    must_exist=False,
    path_policies={"write_parent": True},
)
def imager(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=parse_upath,
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
    scan_names: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="List of SCAN_NUMBERS to image. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    spw_names: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="List of DATA_DESC_ID's to images. "
            "Defaults to all. "
            "Input as comma separated list 0,2 if running from CLI.",
            rich_help_panel="Data Selection",
        ),
    ] = None,
    field_names: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
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
            parser=parse_upath,
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
    concat_row: Annotated[
        bool,
        typer.Option(
            help="Concatenate datasets by row",
            rich_help_panel="Imaging",
        ),
    ] = True,
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
            help="Beam model to use. "
            "Either katbeam or a MeerKAT band name. "
            "One of U, L, S0 or S4 initialises a BeamWizard from the meerkat-beams band cache.",
            rich_help_panel="Input",
        ),
    ] = None,
    phase_dir: Annotated[
        str | None,
        typer.Option(
            help="Rephase visibilities to this phase center. "
            "Should be a string in the format 'HH:MM:SS,DD:MM:SS' (note the , delimiter). "
            "Defaults to the barycentre of the selected fields when more than one field is selected.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            help="Predefined celestial objects known to astropy. "
            "Or a string in the format 'HH:MM:SS,DD:MM:SS' (note the , delimiter)",
            rich_help_panel="Imaging",
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
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
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
    wgt_mode: Annotated[
        Literal["l2", "minvar"],
        typer.Option(
            help="Controls how the Stokes weights are computed. "
            "l2 -> use standard Gaussian formula. "
            "minvar -> use minimum between correlations (wsclean Stokes I style).",
            rich_help_panel="Weighting",
        ),
    ] = "l2",
    weight_grouping: Annotated[
        Literal["per-band-time", "mfs", "per-band", "per-time"],
        typer.Option(
            help="How uv counts are grouped before forming imaging weights. "
            "per-band-time -> each output image weighted on its own counts. "
            "mfs/per-time -> sum counts over bands within a time. "
            "per-band -> sum counts over time within a band.",
            rich_help_panel="Weighting",
        ),
    ] = "per-band-time",
    robustness: Annotated[
        float | None,
        typer.Option(
            help="Briggs robustness in [-2, 2]. Values > 2 (or None) use natural weighting.",
            rich_help_panel="Weighting",
        ),
    ] = None,
    field_of_view: Annotated[
        float | None,
        typer.Option(
            help="Field of view in degrees. Used to set the image size.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    super_resolution_factor: Annotated[
        float,
        typer.Option(
            help="Pixels per resolution element relative to Nyquist.",
            rich_help_panel="Imaging",
        ),
    ] = 2.0,
    cell_size: Annotated[
        float | None,
        typer.Option(
            help="Cell size in arcseconds. Overrides super-resolution-factor if set.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    nx: Annotated[
        int | None,
        typer.Option(
            help="Number of pixels in x. Computed from field-of-view if not set.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    ny: Annotated[
        int | None,
        typer.Option(
            help="Number of pixels in y. Computed from field-of-view if not set.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    psf_oversize: Annotated[
        float,
        typer.Option(
            help="PSF size relative to the image size.",
            rich_help_panel="Imaging",
        ),
    ] = 1.4,
    filter_counts_level: Annotated[
        float,
        typer.Option(
            help="Replace uv cells with counts below median/level to avoid upweighting near-empty cells.",
            rich_help_panel="Weighting",
        ),
    ] = 5.0,
    npix_super: Annotated[
        int,
        typer.Option(
            help="Box half-size in pixels for super-uniform weighting. 0 -> standard uniform.",
            rich_help_panel="Weighting",
        ),
    ] = 0,
    epsilon: Annotated[
        float,
        typer.Option(
            help="Gridder accuracy.",
            rich_help_panel="Imaging",
        ),
    ] = 1e-07,
    do_wgridding: Annotated[
        bool,
        typer.Option(
            help="Use w-gridding (recommended for wide fields).",
            rich_help_panel="Imaging",
        ),
    ] = True,
    double_accum: Annotated[
        bool,
        typer.Option(
            help="Use double precision accumulation when gridding.",
            rich_help_panel="Imaging",
        ),
    ] = True,
    keep_scratch: Annotated[
        bool,
        typer.Option(
            help="Keep the .scratch store of pass-1 averaged data for re-gridding without re-reading the MS.",
            rich_help_panel="Control",
        ),
    ] = True,
    fits_mfs: Annotated[
        bool,
        typer.Option(
            help="Write MFS FITS images.",
            rich_help_panel="Output",
        ),
    ] = True,
    fits_cubes: Annotated[
        bool,
        typer.Option(
            help="Write FITS image cubes.",
            rich_help_panel="Output",
        ),
    ] = True,
    psf: Annotated[
        bool,
        typer.Option(
            help="Compute the PSF data products. "
            "Switch off for a quicklook dirty image that skips the PSF gridding. "
            "A tree written without the PSF cannot be deconvolved.",
            rich_help_panel="Output",
        ),
    ] = True,
    beam: Annotated[
        bool,
        typer.Option(
            help="Write FITS images of the effective beam. "
            "The beam is always stored in the DataTree regardless of this flag.",
            rich_help_panel="Output",
        ),
    ] = True,
    fits_per_partition: Annotated[
        bool,
        typer.Option(
            help="Write FITS images for each data partition into a partitions subdirectory. "
            "Useful to sanity check field and beam orientations.",
            rich_help_panel="Output",
        ),
    ] = False,
    fits_output_folder: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Directory to write FITS files to. Required when output is on a non-file protocol.",
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
    Initialise Stokes data products.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                imager,
                dict(
                    ms=ms,
                    output_filename=output_filename,
                    scan_names=scan_names,
                    spw_names=spw_names,
                    field_names=field_names,
                    freq_range=freq_range,
                    overwrite=overwrite,
                    data_column=data_column,
                    data_group=data_group,
                    partition_columns=partition_columns,
                    weight_column=weight_column,
                    sigma_column=sigma_column,
                    flag_column=flag_column,
                    gain_table=gain_table,
                    integrations_per_image=integrations_per_image,
                    channels_per_image=channels_per_image,
                    concat_row=concat_row,
                    precision=precision,
                    bda_decorr=bda_decorr,
                    max_field_of_view=max_field_of_view,
                    beam_model=beam_model,
                    phase_dir=phase_dir,
                    target=target,
                    chan_average=chan_average,
                    progressbar=progressbar,
                    product=product,
                    ray_address=ray_address,
                    nworkers=nworkers,
                    nthreads=nthreads,
                    wgt_mode=wgt_mode,
                    weight_grouping=weight_grouping,
                    robustness=robustness,
                    field_of_view=field_of_view,
                    super_resolution_factor=super_resolution_factor,
                    cell_size=cell_size,
                    nx=nx,
                    ny=ny,
                    psf_oversize=psf_oversize,
                    filter_counts_level=filter_counts_level,
                    npix_super=npix_super,
                    epsilon=epsilon,
                    do_wgridding=do_wgridding,
                    double_accum=double_accum,
                    keep_scratch=keep_scratch,
                    fits_mfs=fits_mfs,
                    fits_cubes=fits_cubes,
                    psf=psf,
                    beam=beam,
                    fits_per_partition=fits_per_partition,
                    fits_output_folder=fits_output_folder,
                    log_directory=log_directory,
                ),
            )

            # Lazy import the core implementation
            from pfb_imaging.core.imager import imager as imager_core  # noqa: E402

            # Call the core function with all parameters
            imager_core(
                ms,
                output_filename,
                scan_names=scan_names,
                spw_names=spw_names,
                field_names=field_names,
                freq_range=freq_range,
                overwrite=overwrite,
                data_column=data_column,
                data_group=data_group,
                partition_columns=partition_columns,
                weight_column=weight_column,
                sigma_column=sigma_column,
                flag_column=flag_column,
                gain_table=gain_table,
                integrations_per_image=integrations_per_image,
                channels_per_image=channels_per_image,
                concat_row=concat_row,
                precision=precision,
                bda_decorr=bda_decorr,
                max_field_of_view=max_field_of_view,
                beam_model=beam_model,
                phase_dir=phase_dir,
                target=target,
                chan_average=chan_average,
                progressbar=progressbar,
                product=product,
                ray_address=ray_address,
                nworkers=nworkers,
                nthreads=nthreads,
                wgt_mode=wgt_mode,
                weight_grouping=weight_grouping,
                robustness=robustness,
                field_of_view=field_of_view,
                super_resolution_factor=super_resolution_factor,
                cell_size=cell_size,
                nx=nx,
                ny=ny,
                psf_oversize=psf_oversize,
                filter_counts_level=filter_counts_level,
                npix_super=npix_super,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                keep_scratch=keep_scratch,
                fits_mfs=fits_mfs,
                fits_cubes=fits_cubes,
                psf=psf,
                beam=beam,
                fits_per_partition=fits_per_partition,
                fits_output_folder=fits_output_folder,
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
        imager,
        dict(
            ms=ms,
            output_filename=output_filename,
            scan_names=scan_names,
            spw_names=spw_names,
            field_names=field_names,
            freq_range=freq_range,
            overwrite=overwrite,
            data_column=data_column,
            data_group=data_group,
            partition_columns=partition_columns,
            weight_column=weight_column,
            sigma_column=sigma_column,
            flag_column=flag_column,
            gain_table=gain_table,
            integrations_per_image=integrations_per_image,
            channels_per_image=channels_per_image,
            concat_row=concat_row,
            precision=precision,
            bda_decorr=bda_decorr,
            max_field_of_view=max_field_of_view,
            beam_model=beam_model,
            phase_dir=phase_dir,
            target=target,
            chan_average=chan_average,
            progressbar=progressbar,
            product=product,
            ray_address=ray_address,
            nworkers=nworkers,
            nthreads=nthreads,
            wgt_mode=wgt_mode,
            weight_grouping=weight_grouping,
            robustness=robustness,
            field_of_view=field_of_view,
            super_resolution_factor=super_resolution_factor,
            cell_size=cell_size,
            nx=nx,
            ny=ny,
            psf_oversize=psf_oversize,
            filter_counts_level=filter_counts_level,
            npix_super=npix_super,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            keep_scratch=keep_scratch,
            fits_mfs=fits_mfs,
            fits_cubes=fits_cubes,
            psf=psf,
            beam=beam,
            fits_per_partition=fits_per_partition,
            fits_output_folder=fits_output_folder,
            log_directory=log_directory,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
