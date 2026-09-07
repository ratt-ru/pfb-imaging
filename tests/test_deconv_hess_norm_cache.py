"""hess_norm is lambda_max(M), so its cache is only valid for the M that produced it.

The .dt carries hess_norm in the band attrs and deconv reuses it across runs. Every
option that defines M must therefore take part in the cache key -- otherwise changing
--eta, --eta-mode or the frequency prior silently reuses a norm for a different
operator, and the primal-dual step sizes are set from it.
"""

import inspect
import json

from pfb_imaging.core.deconv import _M_OPTS, _cached_hess_norm, _m_signature, deconv

BASE = {
    "eta": 1e-3,
    "eta_mode": None,
    "eta_cap": 100.0,
    "gp_length_scale": None,
    "gp_cap": 10.0,
    "niter": 10,  # not an M option; must be ignored
}


def _attrs(opts, value=1.68):
    return {"hess_norm": value, "hess_norm_opts": _m_signature(opts)}


def test_the_m_option_set_is_pinned():
    """Change-detector, not a proof.

    Whether an option defines M is a fact about ``presets._build_hess``, which
    no assertion here can read. This pins the set so that adding a knob to the
    preconditioner has to touch this file, and the next reader has to decide
    whether the new knob belongs in the key.
    """
    assert set(_M_OPTS) == {"eta", "eta_mode", "eta_cap", "gp_length_scale", "gp_cap"}


def test_every_key_name_is_a_real_deconv_option():
    """A name that no longer exists keys on None for every run.

    ``_m_signature`` reads the options with ``.get``, so renaming an option in
    the CLI without renaming it here does not raise -- the entry silently
    becomes None for everybody, the signature stops discriminating on it, and
    the cache starts handing out a norm for a different M. That is the failure
    mode the tautological version of this test could not see.
    """
    params = set(inspect.signature(deconv).parameters)
    missing = [k for k in _M_OPTS if k not in params]
    assert not missing, f"_M_OPTS names that are not deconv options: {missing}"


def test_signature_ignores_options_that_do_not_define_m():
    assert _m_signature(BASE) == _m_signature({**BASE, "niter": 99})


def test_signature_is_order_independent():
    reversed_opts = dict(reversed(list(BASE.items())))
    assert _m_signature(BASE) == _m_signature(reversed_opts)


def test_matching_options_reuse_the_cached_value():
    assert _cached_hess_norm(_attrs(BASE), BASE) == 1.68


def test_missing_hess_norm_returns_none():
    assert _cached_hess_norm({}, BASE) is None


def test_a_tree_written_before_the_signature_existed_is_re_estimated():
    """Re-estimating is the safe direction; reusing an unlabelled norm is not."""
    assert _cached_hess_norm({"hess_norm": 1.68}, BASE) is None


def test_changing_any_m_option_invalidates_the_cache():
    attrs = _attrs(BASE)
    for key, changed in (
        ("eta", 1e-2),
        ("eta_mode", "radial"),
        ("eta_cap", 1000.0),
        ("gp_length_scale", 0.5),
        ("gp_cap", 100.0),
    ):
        assert _cached_hess_norm(attrs, {**BASE, key: changed}) is None, f"{key} did not invalidate"


def test_the_signature_is_json_serialisable_for_zarr_attrs():
    """Zarr attrs must round-trip through JSON; eta_mode is None on the default path."""
    assert json.loads(_m_signature(BASE))["eta_mode"] is None
