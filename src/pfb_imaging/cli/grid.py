from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="grid",
    info="Initialise image data products.",
)
@stimela_output(
    dtype="Directory",
    name="dds-out",
    info="Output dataset directory.",
    implicit="{current.output-filename}_{current.product}_{current.suffix}.dds",
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
@stimela_output(
    dtype="Directory",
    name="fits-output-folder",
    info="Optional path to write fits files to. "
    "Set to output-filename if not provided. "
    "The same naming conventions apply.",
    mkdir=False,
    path_policies={"write_parent": True},
    metadata={"rich_help_panel": "Output"},
)
def grid(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
            rich_help_panel="Naming",
        ),
    ],
    xds: Annotated[
        str | None,
        typer.Option(
            help="Optional explicit path to xds. Set using output-filename and suffix by default.",
            rich_help_panel="Input",
        ),
    ] = None,
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products. "
            "This is useful for distinguishing runs with different imaging paramaters. "
            "For example, different image sizes of robustness factors.",
            rich_help_panel="Naming",
        ),
    ] = "main",
    concat_row: Annotated[
        bool,
        typer.Option(
            help="Concatenate datasets by row",
            rich_help_panel="Imaging",
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwriting of image space data products. Specify suffix to create a new data set.",
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
    use_best_model: Annotated[
        bool,
        typer.Option(
            help="If this flag is set MODEL_BEST will be used as the model unless transfer-model-from is specified. "
            "By default MODEL will be used as the model.",
            rich_help_panel="Control",
        ),
    ] = False,
    robustness: Annotated[
        float | None,
        typer.Option(
            help="Robustness factor for Briggs weighting. None means natural",
            rich_help_panel="Weighting",
        ),
    ] = None,
    dirty: Annotated[
        bool,
        typer.Option(
            help="Compute the dirty image",
            rich_help_panel="Output",
        ),
    ] = True,
    psf: Annotated[
        bool,
        typer.Option(
            help="Compute the PSF",
            rich_help_panel="Output",
        ),
    ] = True,
    residual: Annotated[
        bool,
        typer.Option(
            help="Compute the residual (only if model is present)",
            rich_help_panel="Output",
        ),
    ] = True,
    noise: Annotated[
        bool,
        typer.Option(
            help="Compute noise map by sampling from weights",
            rich_help_panel="Output",
        ),
    ] = True,
    beam: Annotated[
        bool,
        typer.Option(
            help="Interpolate average beam pattern",
            rich_help_panel="Output",
        ),
    ] = True,
    weight: Annotated[
        bool,
        typer.Option(
            help="Compute effectve image space weights",
            rich_help_panel="Output",
        ),
    ] = True,
    psf_oversize: Annotated[
        float,
        typer.Option(
            help="Size of PSF relative to dirty image",
            rich_help_panel="Imaging",
        ),
    ] = 1.4,
    field_of_view: Annotated[
        float | None,
        typer.Option(
            help="Field of view in degrees",
            rich_help_panel="Imaging",
        ),
    ] = None,
    super_resolution_factor: Annotated[
        float,
        typer.Option(
            help="Will over-sample Nyquist by this factor at max frequency",
            rich_help_panel="Imaging",
        ),
    ] = 2,
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
    filter_counts_level: Annotated[
        float,
        typer.Option(
            help="Set minimum counts in the uniform weighting grid to the median divided by this value. "
            "This is useful to avoid artificially up-weighting nearly empty uv-cells.",
            rich_help_panel="Weighting",
        ),
    ] = 5.0,
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
    product: Annotated[
        str,
        typer.Option(
            help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.",
            rich_help_panel="Data Selection",
        ),
    ] = "I",
    fits_mfs: Annotated[
        bool,
        typer.Option(
            help="Output MFS fits files",
            rich_help_panel="Fits",
        ),
    ] = True,
    fits_cubes: Annotated[
        bool,
        typer.Option(
            help="Output fits cubes",
            rich_help_panel="Fits",
        ),
    ] = True,
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
    fits_output_folder: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="Optional path to write fits files to. "
            "Set to output-filename if not provided. "
            "The same naming conventions apply.",
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
    Initialise image data products.
    """
    if backend == "native" or backend == "auto":
        try:
            # Lazy import the core implementation
            from pfb_imaging.core.grid import grid as grid_core  # noqa: E402

            # Call the core function with all parameters
            grid_core(
                output_filename,
                xds=xds,
                suffix=suffix,
                concat_row=concat_row,
                overwrite=overwrite,
                transfer_model_from=transfer_model_from,
                use_best_model=use_best_model,
                robustness=robustness,
                dirty=dirty,
                psf=psf,
                residual=residual,
                noise=noise,
                beam=beam,
                weight=weight,
                psf_oversize=psf_oversize,
                field_of_view=field_of_view,
                super_resolution_factor=super_resolution_factor,
                cell_size=cell_size,
                nx=nx,
                ny=ny,
                filter_counts_level=filter_counts_level,
                target=target,
                l2_reweight_dof=l2_reweight_dof,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                double_accum=double_accum,
                nworkers=nworkers,
                nthreads=nthreads,
                product=product,
                fits_mfs=fits_mfs,
                fits_cubes=fits_cubes,
                log_directory=log_directory,
                fits_output_folder=fits_output_folder,
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
        grid,
        dict(
            output_filename=output_filename,
            xds=xds,
            suffix=suffix,
            concat_row=concat_row,
            overwrite=overwrite,
            transfer_model_from=transfer_model_from,
            use_best_model=use_best_model,
            robustness=robustness,
            dirty=dirty,
            psf=psf,
            residual=residual,
            noise=noise,
            beam=beam,
            weight=weight,
            psf_oversize=psf_oversize,
            field_of_view=field_of_view,
            super_resolution_factor=super_resolution_factor,
            cell_size=cell_size,
            nx=nx,
            ny=ny,
            filter_counts_level=filter_counts_level,
            target=target,
            l2_reweight_dof=l2_reweight_dof,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
            nworkers=nworkers,
            nthreads=nthreads,
            product=product,
            fits_mfs=fits_mfs,
            fits_cubes=fits_cubes,
            log_directory=log_directory,
            fits_output_folder=fits_output_folder,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
