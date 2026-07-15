import pytest
from radiomesh.generated import _stokes_expr as se

from pfb_imaging.utils.stokes import stokes_expr_funcs


def test_selects_diag_l2_in_iquv_order():
    vis_fns, wgt_fns = stokes_expr_funcs("VQI", "linear", "4", "l2", 5)
    # always ordered I, Q, U, V regardless of product string order
    assert vis_fns == (
        se.LINEAR_VIS_DIAGJONES_I,
        se.LINEAR_VIS_DIAGJONES_Q,
        se.LINEAR_VIS_DIAGJONES_V,
    )
    assert wgt_fns == (
        se.LINEAR_WEIGHT_DIAGJONES_I,
        se.LINEAR_WEIGHT_DIAGJONES_Q,
        se.LINEAR_WEIGHT_DIAGJONES_V,
    )


def test_selects_minvar_weights():
    vis_fns, wgt_fns = stokes_expr_funcs("I", "circular", "4", "minvar", 5)
    assert vis_fns == (se.CIRCULAR_VIS_DIAGJONES_I,)
    assert wgt_fns == (se.CIRCULAR_WEIGHT_MINVAR_DIAGJONES_I,)


def test_selects_full_jones():
    vis_fns, wgt_fns = stokes_expr_funcs("IQUV", "linear", "4", "l2", 6)
    assert vis_fns[0] is se.LINEAR_VIS_JONES_I
    assert wgt_fns[3] is se.LINEAR_WEIGHT_JONES_V
    assert len(vis_fns) == len(wgt_fns) == 4


@pytest.mark.parametrize(
    ("product", "pol", "nc", "wgt_mode", "jones_ndim", "exc"),
    [
        ("I", "elliptical", "4", "l2", 5, ValueError),  # unknown pol
        ("I", "linear", "3", "l2", 5, ValueError),  # unsupported nc
        ("I", "linear", "4", "huber", 5, ValueError),  # unknown mode
        ("Q", "circular", "2", "l2", 5, ValueError),  # Q needs 4 corr (circular)
        ("U", "linear", "2", "l2", 5, ValueError),  # U needs 4 corr
        ("V", "linear", "2", "l2", 5, ValueError),  # V needs 4 corr (linear)
        ("IX", "linear", "4", "l2", 5, ValueError),  # unknown product letter
        ("I", "linear", "1", "l2", 5, ValueError),  # 1 corr unsupported
        ("I", "linear", "4", "minvar", 6, NotImplementedError),  # minvar full-jones
        ("I", "linear", "2", "l2", 6, ValueError),  # full jones needs 4 corr
        ("I", "linear", "4", "l2", 4, ValueError),  # bad jones ndim
    ],
)
def test_selector_validation(product, pol, nc, wgt_mode, jones_ndim, exc):
    with pytest.raises(exc):
        stokes_expr_funcs(product, pol, nc, wgt_mode, jones_ndim)
