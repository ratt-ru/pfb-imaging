import logging
import tracemalloc

import numpy as np
import pytest

from pfb_imaging.utils import misc
from pfb_imaging.utils.misc import fitcleanbeam

pmp = pytest.mark.parametrize


def _gauss_psf(n, sig_px, sidelobe=0.05):
    """Elliptical Gaussian main lobe plus low ripples, peak 1 at (n//2, n//2)."""
    x = np.arange(n) - n // 2
    xx, yy = np.meshgrid(x, x, indexing="ij")
    r2 = xx**2 / sig_px**2 + yy**2 / (0.7 * sig_px) ** 2
    psf = np.exp(-0.5 * r2) + sidelobe * np.cos(xx / 3.0) * np.exp(-0.5 * r2 / 16)
    return psf[None] / psf.max()


def _embed(psf, n):
    """Zero-pad a centred PSF to n x n, keeping its peak at (n//2, n//2)."""
    out = np.zeros((1, n, n))
    m = psf.shape[-1]
    lo = n // 2 - m // 2
    out[:, lo : lo + m, lo : lo + m] = psf
    return out


# 20 px: the 10-sigma fit disc (~200 px) lies far outside a small first window,
# so the window has to grow to contain it, not just the main lobe
@pmp("sig_px", [3.0, 20.0])
def test_fitcleanbeam_window_is_exact(sig_px):
    """Same fit points in the same order give bitwise the same parameters."""
    small = _gauss_psf(512, sig_px)
    big = _embed(small, 2048)
    np.testing.assert_array_equal(fitcleanbeam(big), fitcleanbeam(small))


def test_fitcleanbeam_memory_is_bounded():
    """The fit allocates for the fit region, not the grid (#339)."""
    big = _embed(_gauss_psf(512, 3.0), 2048)
    fitcleanbeam(_gauss_psf(64, 3.0))  # jit-compile outside the measurement
    tracemalloc.start()
    fitcleanbeam(big)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < big.nbytes, f"peak {peak / big.nbytes:.1f} x the PSF"


def test_fitcleanbeam_fit_warning_is_logged(monkeypatch, caplog):
    # it was a bare print, so it never reached the log file (#348)
    def flagged(func, x0, **kwargs):
        return x0, 0.0, {"warnflag": 2, "task": "ABNORMAL"}

    monkeypatch.setattr(misc, "fmin_l_bfgs_b", flagged)
    monkeypatch.setattr(misc.log, "propagate", True)  # the app logger stops propagation at "pfb"
    with caplog.at_level(logging.WARNING, logger="pfb.MISC"):
        fitcleanbeam(_gauss_psf(64, 3.0))
    assert any("ABNORMAL" in r.getMessage() and r.name == "pfb.MISC" for r in caplog.records)
