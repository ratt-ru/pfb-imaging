import typer
from pathlib import Path
from typing_extensions import Annotated
from hip_cargo import stimela_cab, stimela_output

@stimela_cab(
    name="pfb_hci",
    info="Pfb Hci",
    policies="{'pass_missing_as_none': True}",
)
@stimela_output(
    name="dir_out",
    dtype="Directory",
    info="{current.output_filename}_{current.product}.fds",
    required=False,
)
def hci(
    ms: Annotated[list[str] | None, typer.Option(help="Path to measurement set")] = None,
    output_filename: Annotated[str | None, typer.Option(help="Basename of output")] = None,
    log_directory: Annotated[str | None, typer.Option(help="Directory to write logs and performance reports to.")] = None,
    product: Annotated[str, typer.Option(help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.")] = "I",
    scans: Annotated[list[int] | None, typer.Option(help="List of SCAN_NUMBERS to image. Defaults to all. Input as comma separated list 0,2 if running from CLI")] = None,
    ddids: Annotated[list[int] | None, typer.Option(help="List of DATA_DESC_ID's to images. Defaults to all. Input as comma separated list 0,2 if running from CLI")] = None,
    fields: Annotated[list[int] | None, typer.Option(help="List of FIELD_ID's to image. Defaults to all. Input as comma separated list 0,2 if running from CLI")] = None,
    freq_range: Annotated[str | None, typer.Option(help="Frequency range to image in Hz. Specify as a string with colon delimiter eg. '1e9:1.1e9'")] = None,
    overwrite: Annotated[bool, typer.Option(help="Allow overwrite of output xds")] = False,
    transfer_model_from: Annotated[str | None, typer.Option(help="Name of dataset to use for model initialisation")] = None,
    data_column: Annotated[str, typer.Option(help="Data column to image. Must be the same across MSs. Simple arithmetic is supported eg. 'CORRECTED_DATA-MODEL_DATA'. When gains are present this column will be corrected.")] = "DATA",
    model_column: Annotated[str | None, typer.Option(help="Model column to subtract from corrected data column. Useful to avoid high time res degridding.")] = None,
    weight_column: Annotated[str | None, typer.Option(help="Column containing natural weights. Must be the same across MSs")] = None,
    sigma_column: Annotated[str | None, typer.Option(help="Column containing standard devations. Will be used to initialise natural weights if detected. Must be the same across MSs")] = None,
    flag_column: Annotated[str, typer.Option(help="Column containing data flags. Must be the same across MSs")] = "FLAG",
    gain_table: Annotated[list[str] | None, typer.Option(help="Path to Quartical gain table containing NET gains. There must be a table for each MS and glob(ms) and glob(gt) should match up when running from CLI.")] = None,
    integrations_per_image: Annotated[int, typer.Option(help="Number of time integrations per image. Default (-1, 0, None) -> dataset per scan.")] = 1,
    channels_per_image: Annotated[int, typer.Option(help="Number of channels per image for gridding resolution. Default of (-1, 0, None) implies a single dataset per spw.")] = -1,
    precision: Annotated[str, typer.Option(help="Gridding precision")] = "double",  # choices: ['single', 'double']
    beam_model: Annotated[str | None, typer.Option(help="Path to beam model as an xarray dataset backed by zarr")] = None,
    field_of_view: Annotated[float, typer.Option(help="Field of view in degrees")] = 1.0,
    super_resolution_factor: Annotated[float, typer.Option(help="Will over-sample Nyquist by this factor at max frequency")] = 1.4,
    cell_size: Annotated[float | None, typer.Option(help="Cell size in arc-seconds")] = None,
    nx: Annotated[int | None, typer.Option(help="Number of x pixels")] = None,
    ny: Annotated[int | None, typer.Option(help="Number of y pixels")] = None,
    psf_relative_size: Annotated[float | None, typer.Option(help="Relative size of the PSF in pixels. A value of 1.0 means the PSF will be the same size as the image. The default is to make the PSF just big enough to extract the PSF parameters. If the natural gradient is required the PSF needs to ideally be twice the size of the image.")] = None,
    robustness: Annotated[float | None, typer.Option(help="Robustness factor for Briggs weighting. None means natural")] = None,
    target: Annotated[str | None, typer.Option(help="This can be predefined celestial objects known to astropy or a string in the format 'HH:MM:SS,DD:MM:SS' (note the , delimiter)")] = None,
    l2_reweight_dof: Annotated[float | None, typer.Option(help="The degrees of freedom parameter for L2 reweighting. The default (None) means no reweighting. A sensible value for this parameter depends on the level of RFI in the data. Small values (eg. 2) result in aggressive reweighting and should be avoided if the model is still incomplete.")] = None,
    progressbar: Annotated[bool, typer.Option(help="Display progress. Use --no-progressbar to deactivate.")] = True,
    output_format: Annotated[str, typer.Option(help="zarr or fits output")] = "zarr",
    eta: Annotated[float, typer.Option(help="Value to add to Hessian to make it invertible. Smaller values tend to fit more noise and make the inversion less stable.")] = "1e-5",
    psf_out: Annotated[bool, typer.Option(help="Whether to produce output PSF's or not.")] = False,
    weight_grid_out: Annotated[bool, typer.Option(help="Whether to produce an image of the weighted grid or not.")] = False,
    natural_grad: Annotated[bool, typer.Option(help="Compute naural gradient")] = False,
    check_ants: Annotated[bool, typer.Option(help="Check that ANTENNA1 and ANTENNA2 tables are consistent with the ANTENNA table.")] = False,
    inject_transients: Annotated[str | None, typer.Option(help="YAML file containing transients to inject into the data.")] = None,
    filter_counts_level: Annotated[float, typer.Option(help="Set minimum counts in the uniform weighting grid to the median divided by this value. This is useful to avoid artificially up-weighting nearly empty uv-cells.")] = 10.0,
    min_padding: Annotated[float, typer.Option(help="Minimum padding to be applied during gridding.")] = 2.0,
    phase_dir: Annotated[str | None, typer.Option(help="Rephase visibilities to this phase center. Should be a string in the format 'HH:MM:SS,DD:MM:SS' (note the , delimiter)")] = None,
    stack: Annotated[bool, typer.Option(help="Stack everything into a single xarray dataset")] = False,
    epsilon: Annotated[float, typer.Option(help="Gridder accuracy")] = "1e-7",
    do_wgridding: Annotated[bool, typer.Option(help="Perform w-correction via improved w-stacking")] = True,
    double_accum: Annotated[bool, typer.Option(help="Accumulate onto grid using double precision. Only has an affect when using single precision.")] = True,
    host_address: Annotated[str | None, typer.Option(help="Address where the distributed client lives. Uses LocalCluster if no address is provided and scheduler is set to distributed.")] = None,
    nworkers: Annotated[int, typer.Option(help="Number of worker processes. Use with distributed scheduler.")] = 1,
    nthreads: Annotated[int | None, typer.Option(help="Number of threads used to scale vertically (eg. for FFTs and gridding). Each dask thread can in principle spawn this many threads. Will attempt to use half the available threads by default.")] = None,
    direct_to_workers: Annotated[bool, typer.Option(help="Connect direct to workers i.e. bypass scheduler. Faster but then the dashboard isn't very useful.")] = True,
    log_level: Annotated[str, typer.Option(help="")] = "error",  # choices: ['error', 'warning', 'info', 'debug']
    cg_tol: Annotated[float, typer.Option(help="Tolreance of conjugate gradient algorithm")] = "1e-3",
    cg_maxit: Annotated[int, typer.Option(help="Maximum iterations for conjugate gradient algorithm")] = 150,
    cg_minit: Annotated[int, typer.Option(help="Minimum iterations for conjugate gradient algorithm")] = 10,
    cg_verbose: Annotated[int, typer.Option(help="Verbosity of conjugate gradient algorithm. Set to > 1 for debugging, 0 for silence.")] = 1,
    cg_report_freq: Annotated[int, typer.Option(help="Report frequency of conjugate gradient algorithm.")] = 10,
    backtrack: Annotated[bool, typer.Option(help="Ensure residual decreases at every iteration.")] = False,
):
    # TODO: Implement function
    # Lazy import heavy dependencies here
    # from pfb_imaging.operators import my_function
    pass