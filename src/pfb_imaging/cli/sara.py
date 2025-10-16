import typer
from pathlib import Path
from typing_extensions import Annotated
from hip_cargo import stimela_cab, stimela_output

@stimela_cab(
    name="pfb_sara",
    info="Pfb Sara",
    policies="{'pass_missing_as_none': True}",
)
@stimela_output(
    name="dds_out",
    dtype="Directory",
    info="{current.output_filename}_{current.product}_{current.suffix}.dds",
    required=False,
)
@stimela_output(
    name="mds_out",
    dtype="Directory",
    info="{current.output_filename}_{current.product}_{current.suffix}_model.mds",
    required=False,
)
def sara(
    suffix: Annotated[str, typer.Option(help="Can be used to specify a custom name for the image space data products", rich_help_panel="OUTPUTS")] = "main",
    bases: Annotated[str, typer.Option(help="Wavelet bases to use. Give as comma separated str eg.", rich_help_panel="SARA")] = "self,db1,db2,db3",
    nlevels: Annotated[int, typer.Option(help="Wavelet decomposition level", rich_help_panel="SARA")] = 2,
    l1_reweight_from: Annotated[int, typer.Option(help="L1 reweighting will kick in either at convergence or after this many iterations. Set to a negative value to disbale L1 reweighting.", rich_help_panel="SARA")] = 5,
    hess_norm: Annotated[float | None, typer.Option(help="Spectral norm of hessian approximation", rich_help_panel="SARA")] = None,
    hess_approx: Annotated[str, typer.Option(help="Which Hessian approximation to use. Set to wgt for vis space approximation, psf for zero-padded image space approximation and direct for direct inversion.", rich_help_panel="PFB")] = "psf",  # choices: ['wgt', 'psf', 'direct']
    rmsfactor: Annotated[float, typer.Option(help="By default will threshold by rmsfactor*rms at every iteration", rich_help_panel="SARA")] = 1.0,
    eta: Annotated[float, typer.Option(help="Will use eta*wsum to regularise the inversion of the Hessian approximation.", rich_help_panel="PFB")] = 1.0,
    gamma: Annotated[float, typer.Option(help="Step size of update", rich_help_panel="PFB")] = 1.0,
    alpha: Annotated[float, typer.Option(help="Controls how aggressively the l1reweighting is applied. Larger values correspond to more agressive reweighting.", rich_help_panel="SARA")] = 2,
    nbasisf: Annotated[int | None, typer.Option(help="Number of basis functions to use while fitting the frequency axis. Default is to use the number of non-null imaging bands i.e. interpolation.", rich_help_panel="MODEL")] = None,
    positivity: Annotated[int, typer.Option(help="How to apply positivity constraint 0 -> no positivity, 1 -> normal positivity constraint 2 -> strong positivity i.e. all pixels in a band > 0", rich_help_panel="SARA")] = 1,
    niter: Annotated[int, typer.Option(help="Number of iterations.", rich_help_panel="PFB")] = 10,
    nthreads: Annotated[int | None, typer.Option(help="Total number of threads to use. Defaults to half the total number available.", rich_help_panel="PARALLEL")] = None,
    tol: Annotated[float, typer.Option(help="Tolerance at which to terminate algorithm. Will stop when norm(x-xp)/norm(x) < tol", rich_help_panel="PFB")] = "5e-4",
    diverge_count: Annotated[int, typer.Option(help="Will terminate the algorithm if the rms increases this many times. Set to larger than niter to disable this check.", rich_help_panel="PFB")] = 5,
    rms_outside_model: Annotated[bool, typer.Option(help="Mask residual where model is non-zero when computing rms. This is not recommended for largely non-empty fields.")] = False,
    init_factor: Annotated[float, typer.Option(help="Reduce the regularisation strength by this fraction at the outset.", rich_help_panel="PFB")] = 0.5,
    verbosity: Annotated[int, typer.Option(help="Set to larger than 1 to report timings during residual computation", rich_help_panel="DEBUG")] = 1,
    epsilon: Annotated[float, typer.Option(help="Gridder accuracy", rich_help_panel="GRIDDING")] = "1e-7",
    do_wgridding: Annotated[bool, typer.Option(help="Perform w-correction via improved w-stacking", rich_help_panel="GRIDDING")] = True,
    double_accum: Annotated[bool, typer.Option(help="Accumulate onto grid using double precision. Only has an affect when using single precision.", rich_help_panel="GRIDDING")] = True,
    pd_tol: Annotated[list[float], typer.Option(help="Tolreance of primal dual algorithm", rich_help_panel="PRIMAL-DUAL")] = ['3e-4'],
    pd_maxit: Annotated[int, typer.Option(help="Maximum iterations for primal dual algorithm", rich_help_panel="PRIMAL-DUAL")] = 450,
    pd_verbose: Annotated[int, typer.Option(help="Verbosity of primal dual algorithm. Set to > 1 for debugging, 0 for silence", rich_help_panel="PRIMAL-DUAL")] = 1,
    pd_report_freq: Annotated[int, typer.Option(help="Report frequency of primal dual algorithm", rich_help_panel="PRIMAL-DUAL")] = 50,
    pm_tol: Annotated[float, typer.Option(help="Tolreance of power method", rich_help_panel="POWER-METHOD")] = "1e-3",
    pm_maxit: Annotated[int, typer.Option(help="Maximum iterations for power method", rich_help_panel="POWER-METHOD")] = 100,
    pm_verbose: Annotated[int, typer.Option(help="Verbosity of primal power method. Set to > 1 for debugging, 0 for silence", rich_help_panel="POWER-METHOD")] = 1,
    pm_report_freq: Annotated[int, typer.Option(help="Report frequency of power method algorithm", rich_help_panel="POWER-METHOD")] = 100,
    cg_tol: Annotated[float, typer.Option(help="Tolerance of conjugate gradient algorithm", rich_help_panel="CONJUGATE-GRADIENT")] = "1e-3",
    cg_maxit: Annotated[int, typer.Option(help="Maximum iterations for conjugate gradient algorithm", rich_help_panel="CONJUGATE-GRADIENT")] = 150,
    cg_minit: Annotated[int, typer.Option(help="Minimum iterations for conjugate gradient algorithm", rich_help_panel="CONJUGATE-GRADIENT")] = 10,
    cg_verbose: Annotated[int, typer.Option(help="Verbosity of conjugate gradient algorithm. Set to > 1 for debugging, 0 for silence", rich_help_panel="CONJUGATE-GRADIENT")] = 1,
    cg_report_freq: Annotated[int, typer.Option(help="Report frequency of conjugate gradient algorithm", rich_help_panel="CONJUGATE-GRADIENT")] = 10,
    backtrack: Annotated[bool, typer.Option(help="Ensure residual decreases at every iteration", rich_help_panel="CONJUGATE-GRADIENT")] = False,
    output_filename: Annotated[str | None, typer.Option(help="Basename of output", rich_help_panel="OUTPUTS")] = None,
    log_directory: Annotated[str | None, typer.Option(help="Directory to write logs and performance reports to.", rich_help_panel="OUTPUTS")] = None,
    product: Annotated[str, typer.Option(help="String specifying which Stokes products to produce. Outputs are always be alphabetically ordered.", rich_help_panel="OUTPUTS")] = "I",
    fits_output_folder: Annotated[str | None, typer.Option(help="Optional path to write fits files to. Set to output-filename if not provided. The same naming conventions apply.", rich_help_panel="OUTPUTS")] = None,
    fits_mfs: Annotated[bool, typer.Option(help="Output MFS fits files", rich_help_panel="OUTPUTS")] = True,
    fits_cubes: Annotated[bool, typer.Option(help="Output fits cubes", rich_help_panel="OUTPUTS")] = True,
):
    # TODO: Implement function
    # Lazy import heavy dependencies here
    # from pfb_imaging.operators import my_function
    pass