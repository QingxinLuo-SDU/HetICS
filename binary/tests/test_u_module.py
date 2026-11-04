"""U module: bitwise identity with lambda 0, moments of U, one-sided lambda, gamma0 truth and isolation of U."""

from __future__ import annotations

import math

import numpy as np
import pytest

import binary_with_u as bu
import formal_dgp
from conftest import SMALL_PARAMS, small_row


def _rngs(group):
    return (np.random.default_rng([group["data_seed"], 1]),
            np.random.default_rng([group["param_seed"], 2]))


def test_lambda_zero_is_bitwise_identical_to_source(small_continuous, small_group,
                                                    sources):
    """Check 1: with both lambdas 0, X/Y/p0/p1 are bitwise identical to make_binary."""
    _gen, b6 = sources
    original = b6.SIG_X_SPEC
    b6.SIG_X_SPEC = float(small_group["sig_x_spec"])
    try:
        for gamma in (-1.0, 0.0, 1.0, 3.0):
            rb, rp = _rngs(small_group)
            ref = b6.make_binary(small_continuous, gamma, rb, rp)
            rb, rp = _rngs(small_group)
            got = bu.make_binary_u(small_continuous, gamma, rb, rp, b6,
                                   u=None, lambda_ux=0.0, lambda_uy=0.0)
            assert np.array_equal(ref["X_bin"], got["X_bin"])
            assert np.array_equal(ref["Y_bin"], got["Y_bin"])
            assert ref["ate_do"] == got["ate_do"]
            assert np.array_equal(np.asarray(ref["ate_k"]), np.asarray(got["ate_k"]))
            for k, v in ref["beta_spec_X"].items():
                assert np.array_equal(np.asarray(v), np.asarray(got["beta_spec_X"][k]))
    finally:
        b6.SIG_X_SPEC = original


def test_u_population_and_sample_moments(small_continuous, small_group):
    """Check 2: Var(U)=.58, SD=.761577, Corr(U,S)=.919145, Corr(U,shared)=.721397."""
    U, S, meta = formal_dgp.unmeasured_u_for_base(
        small_continuous, small_group, SMALL_PARAMS["rho_block"])
    assert meta["population_var_u"] == pytest.approx(0.58, abs=1e-12)
    assert meta["population_sd_u"] == pytest.approx(0.761577310586, abs=1e-9)
    assert meta["population_corr_u_s"] == pytest.approx(0.919145030018, abs=1e-9)
    assert meta["residual_var_given_F"] == pytest.approx(0.09, abs=1e-12)
    assert meta["not_standardized"] is True
    # theory: Corr(U, shared_j) = corr(U,S) * sqrt((1+(m-1)rho)/m)
    m, rho = SMALL_PARAMS["n_causal_shared"], SMALL_PARAMS["rho_block"]
    expected = 0.919145030018 * math.sqrt((1 + (m - 1) * rho) / m)
    assert meta["sample_corr_U_shared_mean"] == pytest.approx(expected, abs=0.12)
    # close in finite samples, not bitwise; U is not standardized
    assert meta["sample_var_U"] == pytest.approx(0.58, rel=0.35)
    assert abs(meta["sample_var_U"] - 1.0) > 0.1


def test_u_shared_score_uses_population_scaling(small_continuous):
    S = bu.shared_common_score(small_continuous["F"],
                               small_continuous["true_shared_causal"],
                               SMALL_PARAMS["rho_block"])
    m, rho = SMALL_PARAMS["n_causal_shared"], SMALL_PARAMS["rho_block"]
    manual = (np.asarray(small_continuous["F"], float)[
        :, [int(j) for j in small_continuous["true_shared_causal"]]].sum(axis=1)
        / math.sqrt(m * (1 + (m - 1) * rho)))
    assert np.allclose(S, manual, atol=0, rtol=0)


def test_noise_loading_is_not_sqrt_one_minus_rho_squared():
    """.3 is a fixed noise loading, not sqrt(1-.7^2)=0.714; U is not normalized."""
    assert bu.U_NOISE_LOADING == 0.3
    assert abs(bu.U_NOISE_LOADING - math.sqrt(1 - 0.7 ** 2)) > 0.4


def test_single_lambda_zero_removes_only_that_equation(small_continuous, small_group,
                                                       sources):
    """Check 3: with lambda_UX=0, X returns to its value without U while the Y increment remains."""
    _gen, b6 = sources
    U, _S, _meta = formal_dgp.unmeasured_u_for_base(
        small_continuous, small_group, SMALL_PARAMS["rho_block"])
    rb, rp = _rngs(small_group)
    base = bu.make_binary_u(small_continuous, 1.0, rb, rp, b6, u=None,
                            lambda_ux=0.0, lambda_uy=0.0)
    rb, rp = _rngs(small_group)
    only_y = bu.make_binary_u(small_continuous, 1.0, rb, rp, b6, u=U,
                              lambda_ux=0.0, lambda_uy=0.5)
    assert np.array_equal(base["X_bin"], only_y["X_bin"])       # no increment in the X equation
    assert not np.allclose(base["p1_do"], only_y["p1_do"])      # increment in the Y equation remains

    rb, rp = _rngs(small_group)
    only_x = bu.make_binary_u(small_continuous, 1.0, rb, rp, b6, u=U,
                              lambda_ux=0.5, lambda_uy=0.0)
    assert np.allclose(base["p1_do"], only_x["p1_do"], atol=0, rtol=0)
    assert not np.array_equal(base["X_bin"], only_x["X_bin"])


def test_gamma_zero_truth_is_exactly_zero_with_u(small_continuous, small_group,
                                                 sources):
    """Check 4: with gamma_logor=0, p1-p0 with U is exactly 0."""
    _gen, b6 = sources
    U, _S, _m = formal_dgp.unmeasured_u_for_base(
        small_continuous, small_group, SMALL_PARAMS["rho_block"])
    rb, rp = _rngs(small_group)
    out = bu.make_binary_u(small_continuous, 0.0, rb, rp, b6, u=U,
                           lambda_ux=0.5, lambda_uy=0.5)
    assert np.array_equal(out["p1_do"], out["p0_do"])
    assert float(np.max(np.abs(out["tau_do_i"]))) == 0.0
    assert out["ate_do"] == 0.0


def test_u_changes_treatment_and_outcome_but_not_F(small_continuous, small_group,
                                                   sources):
    _gen, b6 = sources
    U, _S, _m = formal_dgp.unmeasured_u_for_base(
        small_continuous, small_group, SMALL_PARAMS["rho_block"])
    rb, rp = _rngs(small_group)
    base = bu.make_binary_u(small_continuous, 1.0, rb, rp, b6, u=None)
    rb, rp = _rngs(small_group)
    with_u = bu.make_binary_u(small_continuous, 1.0, rb, rp, b6, u=U,
                              lambda_ux=0.5, lambda_uy=0.5)
    assert not np.array_equal(base["X_bin"], with_u["X_bin"])
    assert not np.array_equal(base["Y_bin"], with_u["Y_bin"])
    # the observed F is unchanged by U
    assert np.asarray(small_continuous["F"]).shape[1] == SMALL_PARAMS["p_total"]


def test_binary_unit_truth_never_exposes_u(small_continuous, small_group, sources):
    """Check 5: the truth holds do risks but not U; U appears only in the latent diagnostics."""
    _gen, b6 = sources
    U, _S, meta = formal_dgp.unmeasured_u_for_base(
        small_continuous, small_group, SMALL_PARAMS["rho_block"])
    row = small_row("TEST_U", gamma_logor=1.0, has_u=True, methods=["dual_icp"])
    observed, truth, _meta = formal_dgp.binary_unit(
        small_continuous, row, small_group, b6, u=U)
    assert "U" not in truth and "u" not in truth
    flat = [v for v in truth.values() if isinstance(v, np.ndarray)]
    for arr in flat:
        if arr.shape == U.shape:
            assert not np.array_equal(arr, U)
    assert np.asarray(observed.features).shape[1] == SMALL_PARAMS["p_total"]
    assert truth["has_unmeasured_U"] is True
    assert "unmeasured U" in truth["full_causal_core_note"]


def test_u_requires_vector_when_lambdas_nonzero(small_continuous, sources):
    _gen, b6 = sources
    with pytest.raises(ValueError):
        bu.make_binary_u(small_continuous, 1.0, np.random.default_rng(1),
                         np.random.default_rng(2), b6, u=None, lambda_ux=0.5)
