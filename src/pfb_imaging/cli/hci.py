from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import ListInt, StimelaMeta, parse_list_int, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
URI = NewType("URI", Path)


@stimela_cab(
    name="hci",
    info="High cadence imaging algorithm.",
)
@stimela_output(
    dtype="Directory",
    name="output-dataset",
    info="Basename of output.",
    required=True,
    policies={"positional": True},
    must_exist=True,
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
@stimela_output(
    dtype="Directory",
    name="log-directory",
    info="Directory to write logs and performance reports to.",
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
@stimela_output(
    dtype="Directory",
    name="temp-dir",
    info="A temporary directory to store ephemeral files.",
    metadata={"rich_help_panel": "Output"},
)
def hci(
    ms: Annotated[
        list[URI],
        typer.Option(
            ...,
            parser=Path,
            help="Path to measurement set",
            rich_help_panel="Input",
        ),
    ],
    output_dataset: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=Path,
            help="Basename of output.",
            rich_help_panel="Output",
        ),
        StimelaMeta(
            must_exist=True,
            mkdir=False,
            path_policies={
                "write_parent": True,
            },
        ),
    ],
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
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
    transfer_model_from: Annotated[
        str | None,
        typer.Option(
            help="Name of dataset to use for model initialisation",
            rich_help_panel="Input",
        ),
    ] = None,
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
    model_column: Annotated[
        str | None,
        typer.Option(
            help="Model column to subtract from corrected data column. Useful to avoid high time res degridding.",
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
            parser=Path,
            help="Path to Quartical gain table containing NET gains. "
            "There must be a table for each MS. "
            "glob(ms) and glob(gt) should match up when running from CLI.",
            rich_help_panel="Input",
        ),
    ] = None,
    max_simul_chunks: Annotated[
        int,
        typer.Option(
            help="Maximum number of chunks to process simultaneously.",
            rich_help_panel="Performance",
        ),
    ] = 4,
    images_per_chunk: Annotated[
        int,
        typer.Option(
            help="Number of images per chunk.",
            rich_help_panel="Performance",
        ),
    ] = 16,
    integrations_per_image: Annotated[
        int,
        typer.Option(
            help="Number of time integrations per image. Default (-1, 0, None) -> dataset per scan.",
            rich_help_panel="Imaging",
        ),
    ] = 1,
    channels_per_image: Annotated[
        int,
        typer.Option(
            help="Number of channels per image for gridding resolution. "
            "Default of (-1, 0, None) implies a single dataset per spw.",
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
    beam_model: Annotated[
        URI | None,
        typer.Option(
            parser=Path,
            help="Path to beam model (bds produced by suricat-beams).",
            rich_help_panel="Input",
        ),
    ] = None,
    field_of_view: Annotated[
        float,
        typer.Option(
            help="Field of view in degrees",
            rich_help_panel="Imaging",
        ),
    ] = 1.0,
    super_resolution_factor: Annotated[
        float,
        typer.Option(
            help="Will over-sample Nyquist by this factor at max frequency",
            rich_help_panel="Imaging",
        ),
    ] = 1,
    cell_size: Annotated[
        float | None,
        typer.Option(
            help="Cell size in arc-seconds",
            rich_help_panel="Imaging",
        ),
    ] = None,
    nx: Annotated[
        int | None,
        typer.Option(
            help="Number of x pixels",
            rich_help_panel="Imaging",
        ),
    ] = None,
    ny: Annotated[
        int | None,
        typer.Option(
            help="Number of y pixels",
            rich_help_panel="Imaging",
        ),
    ] = None,
    psf_relative_size: Annotated[
        float | None,
        typer.Option(
            help="Relative size of the PSF in pixels. "
            "A value of 1.0 means the PSF will be the same size as the image. "
            "The default is to make the PSF just big enough to extract the PSF parameters. "
            "Ideally requires double sized PSF.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    robustness: Annotated[
        float | None,
        typer.Option(
            help="Robustness factor for Briggs weighting. None means natural",
            rich_help_panel="Weighting",
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
    l2_reweight_dof: Annotated[
        float | None,
        typer.Option(
            help="The degrees of freedom parameter for L2 reweighting. "
            "The default (None) means no reweighting. "
            "A sensible value for this parameter depends on the level of RFI in the data. "
            "Small values result in aggressive reweighting. "
            "This should be avoided if the model is still incomplete.",
            rich_help_panel="Weighting",
        ),
    ] = None,
    eta: Annotated[
        float,
        typer.Option(
            help="Value to add to Hessian to make it invertible. "
            "Smaller values tend to fit more noise and make the inversion less stable.",
            rich_help_panel="Imaging",
        ),
    ] = 1e-05,
    psf_out: Annotated[
        bool,
        typer.Option(
            help="Whether to produce output PSF's or not.",
            rich_help_panel="Output",
        ),
    ] = False,
    weight_grid_out: Annotated[
        bool,
        typer.Option(
            help="Whether to produce an image of the weighted grid or not.",
            rich_help_panel="Output",
        ),
    ] = False,
    natural_grad: Annotated[
        bool,
        typer.Option(
            help="Compute naural gradient",
            rich_help_panel="Imaging",
        ),
    ] = False,
    check_ants: Annotated[
        bool,
        typer.Option(
            help="Check that ANTENNA1 and ANTENNA2 tables are consistent with the ANTENNA table.",
            rich_help_panel="Control",
        ),
    ] = False,
    inject_transients: Annotated[
        str | None,
        typer.Option(
            help="YAML file containing transients to inject into the data.",
            rich_help_panel="Imaging",
        ),
    ] = None,
    filter_counts_level: Annotated[
        float,
        typer.Option(
            help="Set minimum counts in the uniform weighting grid to the median divided by this value. "
            "This is useful to avoid artificially up-weighting nearly empty uv-cells.",
            rich_help_panel="Weighting",
        ),
    ] = 10.0,
    npix_super: Annotated[
        int,
        typer.Option(
            help="Half-size in pixels of the box used for super-uniform weighting. "
            "Each visibility is normalised by the sum of counts over a (2*npix_super+1)^2 box around its uv-cell. "
            "0 (default) recovers standard uniform weighting. "
            "Combines with robustness to give super-robust weighting.",
            rich_help_panel="Weighting",
        ),
    ] = 0,
    min_padding: Annotated[
        float,
        typer.Option(
            help="Minimum padding to be applied during gridding.",
            rich_help_panel="Imaging",
        ),
    ] = 2.0,
    phase_dir: Annotated[
        str | None,
        typer.Option(
            help="Rephase visibilities to this phase center. "
            "Should be a string in the format 'HH:MM:SS,DD:MM:SS' (note the , delimiter)",
            rich_help_panel="Imaging",
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
    double_accum: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto grid using double precision. Only has an affect when using single precision.",
            rich_help_panel="WGridder",
        ),
    ] = True,
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
    cg_tol: Annotated[
        float,
        typer.Option(
            help="Tolerance of conjugate gradient algorithm",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 0.001,
    cg_maxit: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for conjugate gradient algorithm",
            rich_help_panel="ConjugateGradient",
        ),
    ] = 150,
    object_store_memory: Annotated[
        float | None,
        typer.Option(
            help="Object store memory (in GB) when using the distributed scheduler.",
            rich_help_panel="Performance",
        ),
    ] = None,
    cube_to_fits: Annotated[
        bool,
        typer.Option(
            help="Whether to convert the output cube to FITS format.",
            rich_help_panel="Output",
        ),
    ] = False,
    wgt_mode: Annotated[
        Literal["l2", "minvar"],
        typer.Option(
            help="Controls how the Stokes weights are computed. "
            "l2 -> use standard Gaussian formula. "
            "minvar -> use minimum between correlations (wsclean Stokes I style).",
            rich_help_panel="Weighting",
        ),
    ] = "l2",
    obs_label: Annotated[
        str | None,
        typer.Option(
            help="Optional observation label to include in the stacked cube.",
            rich_help_panel="Output",
        ),
    ] = None,
    flag_excess_rms: Annotated[
        float,
        typer.Option(
            help="Flag data with RMS values exceeding the median by this factor.",
            rich_help_panel="Imaging",
        ),
    ] = 1.5,
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
    temp_dir: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="A temporary directory to store ephemeral files.",
            rich_help_panel="Output",
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
    High cadence imaging algorithm.
    """
    if backend == "native" or backend == "auto":
        try:
            # Lazy import the core implementation
            from pfb_imaging.core.hci import hci as hci_core  # noqa: E402

            # Call the core function with all parameters
            hci_core(
                ms,
                output_dataset,
                product=product,
                scans=scans,
                ddids=ddids,
                fields=fields,
                freq_range=freq_range,
                overwrite=overwrite,
                transfer_model_from=transfer_model_from,
                data_column=data_column,
                model_column=model_column,
                weight_column=weight_column,
                sigma_column=sigma_column,
                flag_column=flag_column,
                gain_table=gain_table,
                max_simul_chunks=max_simul_chunks,
                images_per_chunk=images_per_chunk,
                integrations_per_image=integrations_per_image,
                channels_per_image=channels_per_image,
                precision=precision,
                beam_model=beam_model,
                field_of_view=field_of_view,
                super_resolution_factor=super_resolution_factor,
                cell_size=cell_size,
                nx=nx,
                ny=ny,
                psf_relative_size=psf_relative_size,
                robustness=robustness,
                target=target,
                l2_reweight_dof=l2_reweight_dof,
                eta=eta,
                psf_out=psf_out,
                weight_grid_out=weight_grid_out,
                natural_grad=natural_grad,
                check_ants=check_ants,
                inject_transients=inject_transients,
                filter_counts_level=filter_counts_level,
                npix_super=npix_super,
                min_padding=min_padding,
                phase_dir=phase_dir,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                nworkers=nworkers,
                nthreads=nthreads,
                cg_tol=cg_tol,
                cg_maxit=cg_maxit,
                object_store_memory=object_store_memory,
                cube_to_fits=cube_to_fits,
                wgt_mode=wgt_mode,
                obs_label=obs_label,
                flag_excess_rms=flag_excess_rms,
                log_directory=log_directory,
                temp_dir=temp_dir,
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
        hci,
        dict(
            ms=ms,
            product=product,
            scans=scans,
            ddids=ddids,
            fields=fields,
            freq_range=freq_range,
            overwrite=overwrite,
            transfer_model_from=transfer_model_from,
            data_column=data_column,
            model_column=model_column,
            weight_column=weight_column,
            sigma_column=sigma_column,
            flag_column=flag_column,
            gain_table=gain_table,
            max_simul_chunks=max_simul_chunks,
            images_per_chunk=images_per_chunk,
            integrations_per_image=integrations_per_image,
            channels_per_image=channels_per_image,
            precision=precision,
            beam_model=beam_model,
            field_of_view=field_of_view,
            super_resolution_factor=super_resolution_factor,
            cell_size=cell_size,
            nx=nx,
            ny=ny,
            psf_relative_size=psf_relative_size,
            robustness=robustness,
            target=target,
            l2_reweight_dof=l2_reweight_dof,
            eta=eta,
            psf_out=psf_out,
            weight_grid_out=weight_grid_out,
            natural_grad=natural_grad,
            check_ants=check_ants,
            inject_transients=inject_transients,
            filter_counts_level=filter_counts_level,
            npix_super=npix_super,
            min_padding=min_padding,
            phase_dir=phase_dir,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            nworkers=nworkers,
            nthreads=nthreads,
            cg_tol=cg_tol,
            cg_maxit=cg_maxit,
            object_store_memory=object_store_memory,
            cube_to_fits=cube_to_fits,
            wgt_mode=wgt_mode,
            obs_label=obs_label,
            flag_excess_rms=flag_excess_rms,
            output_dataset=output_dataset,
            log_directory=log_directory,
            temp_dir=temp_dir,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
