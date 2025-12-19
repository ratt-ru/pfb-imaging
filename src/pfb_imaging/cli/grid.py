from pathlib import Path
from typing import Annotated, NewType
from typing import Literal

import typer

from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="grid",
    info="",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="dds-out",
    info="",
)
def grid(
    output_filename: Annotated[
        str,
        typer.Option(
            ...,
            help="Basename of output",
        ),
    ],
    xds: Annotated[
        str | None,
        typer.Option(
            help="Optional explicit path to xds. Set using output-filename and suffix by default.",
        ),
    ] = None,
    suffix: Annotated[
        str,
        typer.Option(
            help="Can be used to specify a custom name for the image space data products",
        ),
    ] = "main",
    concat_row: Annotated[
        bool,
        typer.Option(
            help="Concatenate datasets by row",
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwriting of image space data products. Specify suffix to create a new data set.",
        ),
    ] = False,
    transfer_model_from: Annotated[
        str | None,
        typer.Option(
            help="Name of dataset to use for model initialisation",
        ),
    ] = None,
    use_best_model: Annotated[
        bool,
        typer.Option(
            help="If this flag is set MODEL_BEST will be used as the model unless transfer-model-from is specified. "
            "By default MODEL will be used as the model.",
        ),
    ] = False,
    robustness: Annotated[
        float | None,
        typer.Option(
            help="Robustness factor for Briggs weighting. None means natural",
        ),
    ] = None,
    dirty: Annotated[
        bool,
        typer.Option(
            help="Compute the dirty image",
        ),
    ] = True,
    psf: Annotated[
        bool,
        typer.Option(
            help="Compute the PSF",
        ),
    ] = True,
    residual: Annotated[
        bool,
        typer.Option(
            help="Compute the residual (only if model is present)",
        ),
    ] = True,
    noise: Annotated[
        bool,
        typer.Option(
            help="Compute noise map by sampling from weights",
        ),
    ] = True,
    beam: Annotated[
        bool,
        typer.Option(
            help="Interpolate average beam pattern",
        ),
    ] = True,
    psf_oversize: Annotated[
        float,
        typer.Option(
            help="Size of PSF relative to dirty image",
        ),
    ] = 1.4,
    weight: Annotated[
        bool,
        typer.Option(
            help="Compute effectve image space weights",
        ),
    ] = True,
    field_of_view: Annotated[
        float | None,
        typer.Option(
            help="Field of view in degrees",
        ),
    ] = None,
    super_resolution_factor: Annotated[
        float,
        typer.Option(
            help="Will over-sample Nyquist by this factor at max frequency",
        ),
    ] = 2,
    cell_size: Annotated[
        float | None,
        typer.Option(
            help="Cell size in arc-seconds",
        ),
    ] = None,
    nx: Annotated[
        int | None,
        typer.Option(
            help="Number of x pixels",
        ),
    ] = None,
    ny: Annotated[
        int | None,
        typer.Option(
            help="Number of y pixels",
        ),
    ] = None,
    filter_counts_level: Annotated[
        float,
        typer.Option(
            help="Set minimum counts in the uniform weighting grid to the median divided by this value. "
            "This is useful to avoid artificially up-weighting nearly empty uv-cells.",
        ),
    ] = 5.0,
    target: Annotated[
        str | None,
        typer.Option(
            help="This can be predefined celestial objects known to astropy or a string in the format 'HH:MM:SS,DD:MM:SS' (note the , delimiter)",
        ),
    ] = None,
    l2_reweight_dof: Annotated[
        float | None,
        typer.Option(
            help="The degrees of freedom parameter for L2 reweighting. "
            "The default (None) means no reweighting. "
            "A sensible value for this parameter depends on the level of RFI in the data. "
            "Small values (eg. "
            "2) result in aggressive reweighting and should be avoided if the model is still incomplete.",
        ),
    ] = None,
    epsilon: Annotated[
        float,
        typer.Option(
            help="Gridder accuracy",
        ),
    ] = 1e-07,
    do_wgridding: Annotated[
        bool,
        typer.Option(
            help="Perform w-correction via improved w-stacking",
        ),
    ] = True,
    double_accum: Annotated[
        bool,
        typer.Option(
            help="Accumulate onto grid using double precision. Only has an affect when using single precision.",
        ),
    ] = True,
    host_address: Annotated[
        str | None,
        typer.Option(
            help="Address where the distributed client lives. Uses LocalCluster if no address is provided.",
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
        psf_oversize=psf_oversize,
        weight=weight,
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
        host_address=host_address,
        nworkers=nworkers,
        nthreads=nthreads,
        log_level=log_level,
        log_directory=log_directory,
        product=product,
        fits_output_folder=fits_output_folder,
        fits_mfs=fits_mfs,
        fits_cubes=fits_cubes,
    )
