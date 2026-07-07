"""Minor-cycle preset factories: assemble PFBSolver from CLI options.

Contributors add a deconvolution algorithm by writing a Regulariser and
registering a factory here ("kclean" will slot in as OneShot + ClarkProx).
"""

import numpy as np

from pfb_imaging.deconv.pfb import PFBSolver
from pfb_imaging.operators.hessian import HessTreeRay
from pfb_imaging.operators.psi import IdentityPsi, PsiNocopytRay
from pfb_imaging.opt.forward_backward import ForwardBackward
from pfb_imaging.opt.pcg import PCG
from pfb_imaging.opt.primal_dual import PrimalDual
from pfb_imaging.prox.l1 import L1
from pfb_imaging.prox.l21 import L21
from pfb_imaging.prox.positivity import positivity_prox


def _build_hess(partitions_per_band, geometry, opts):
    """HessTreeRay with the legacy total-wsum normalisation convention."""
    wsum_b = np.array([sum(float(np.sum(p["wsum"])) for p in parts) for parts in partitions_per_band])
    wsum_tot = wsum_b.sum()
    etas = opts["eta"] * wsum_b / wsum_tot
    return HessTreeRay(
        partitions_per_band,
        geometry["nx"],
        geometry["ny"],
        geometry["nx_psf"],
        geometry["ny_psf"],
        etas=etas,
        nthreads=opts["nthreads"],
        wsums=wsum_tot,
        cg_tol=opts["cg_tol"],
        cg_maxit=opts["cg_maxit"],
        cg_verbose=opts["cg_verbose"],
    )


def _build_backward(opts):
    """Backward solver from opt_backend; primal_prox from the positivity mode."""
    pprox = positivity_prox(opts["positivity"])
    if opts["opt_backend"] == "primal-dual":
        return PrimalDual(
            tol=opts["pd_tol"],
            maxit=opts["pd_maxit"],
            verbosity=opts["pd_verbose"],
            report_freq=opts["pd_report_freq"],
            gamma=opts["gamma"],
            primal_prox=pprox,
        )
    if opts["opt_backend"] == "forward-backward":
        return ForwardBackward(
            tol=opts["fb_tol"],
            maxit=opts["fb_maxit"],
            verbosity=opts["fb_verbose"],
            report_freq=opts["fb_report_freq"],
            gamma=opts["gamma"],
            acceleration=opts["acceleration"],
            primal_prox=pprox,
        )
    raise ValueError(f"Unknown opt_backend '{opts['opt_backend']}'")


def _common_kwargs(model, update, opts):
    return dict(
        model=model,
        update=update,
        gamma=opts["gamma"],
        hessnorm=opts["hess_norm"],
        l1_reweight_from=opts["l1_reweight_from"],
        pm_tol=opts["pm_tol"],
        pm_maxit=opts["pm_maxit"],
        pm_verbose=opts["pm_verbose"],
        pm_report_freq=opts["pm_report_freq"],
        verbosity=opts["verbosity"],
    )


def make_sara(partitions_per_band, geometry, model, update, opts):
    """SARA: l21 over a wavelet dictionary, PD or FB backward."""
    nband = len(partitions_per_band)
    bases = tuple(opts["bases"]) if not isinstance(opts["bases"], str) else tuple(opts["bases"].split(","))
    psi = PsiNocopytRay(nband, geometry["nx"], geometry["ny"], bases, opts["nlevels"], opts["nthreads"])
    reg = L21(psi, bases, rmsfactor=opts["rmsfactor"], alpha=opts["alpha"])
    hess = _build_hess(partitions_per_band, geometry, opts)
    fwd = PCG(
        tol=opts["cg_tol"], maxit=opts["cg_maxit"], verbosity=opts["cg_verbose"], report_freq=opts["cg_report_freq"]
    )
    return PFBSolver(hess, fwd, _build_backward(opts), reg, **_common_kwargs(model, update, opts))


def make_ista(partitions_per_band, geometry, model, update, opts):
    """ISTA: image-domain l1, forward-backward without acceleration."""
    nband = len(partitions_per_band)
    reg = L1(IdentityPsi(nband, geometry["nx"], geometry["ny"]))
    hess = _build_hess(partitions_per_band, geometry, opts)
    fwd = PCG(
        tol=opts["cg_tol"], maxit=opts["cg_maxit"], verbosity=opts["cg_verbose"], report_freq=opts["cg_report_freq"]
    )
    bwd = ForwardBackward(
        tol=opts["fb_tol"],
        maxit=opts["fb_maxit"],
        verbosity=opts["fb_verbose"],
        report_freq=opts["fb_report_freq"],
        gamma=opts["gamma"],
        acceleration=False,
        primal_prox=positivity_prox(opts["positivity"]),
    )
    return PFBSolver(hess, fwd, bwd, reg, **_common_kwargs(model, update, opts))


PRESETS = {"sara": make_sara, "ista": make_ista}
