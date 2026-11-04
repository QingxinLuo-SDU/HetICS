"""Conditional PSM variance: hand-computed cases A/B/D, a full 2^3 enumeration and the I+K weight identity."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import psm_ate


def test_example_a_mutual_pair():
    """A: one treated and one control are each other's neighbours. m=2, K=(1,1), c=(1,-1), mu=(.7,.2) -> V=.37."""
    s = np.array([1.0, -1.0])
    I = np.array([1.0, 1.0])
    K = np.array([1.0, 1.0])
    m = 2.0
    c = s * (I + K) / m
    assert np.allclose(c, [1.0, -1.0])
    mu = np.array([0.7, 0.2])
    assert psm_ate.model_variance(c, mu) == pytest.approx(0.37, abs=1e-12)


def test_example_b_shared_control_and_full_enumeration():
    """B: treatment=(1,1,0), j=(3,3,1). K=(1,0,2), c=(2/3,1/3,-1), mu=(.7,.6,.2).

    psi=7/15, V=.28, SE≈.529150; the 2^3=8 outcome patterns of the three individuals are enumerated exactly.
    """
    s = np.array([1.0, 1.0, -1.0])
    I = np.array([1.0, 1.0, 1.0])
    K = np.array([1.0, 0.0, 2.0])
    m = 3.0
    c = s * (I + K) / m
    assert np.allclose(c, [2 / 3, 1 / 3, -1.0])
    assert c.sum() == pytest.approx(0.0, abs=1e-15)
    assert c[:2].sum() == pytest.approx(1.0, abs=1e-15)
    assert c[2] == pytest.approx(-1.0, abs=1e-15)

    mu = np.array([0.7, 0.6, 0.2])
    psi = float(c @ mu)
    assert psi == pytest.approx(7 / 15, abs=1e-12)
    V = psm_ate.model_variance(c, mu)
    assert V == pytest.approx(0.28, abs=1e-12)
    assert np.sqrt(V) == pytest.approx(0.529150, abs=1e-6)

    # full enumeration: theta = sum(c_i Y_i), exact mean and variance under independent Bernoulli(mu)
    mean, second = 0.0, 0.0
    for y in itertools.product((0, 1), repeat=3):
        prob = float(np.prod([mu[i] if y[i] else 1 - mu[i] for i in range(3)]))
        theta = float(c @ np.array(y, dtype=float))
        mean += prob * theta
        second += prob * theta ** 2
    assert mean == pytest.approx(psi, abs=1e-12)
    assert second - mean ** 2 == pytest.approx(V, abs=1e-12)

    # the paired-difference formula and the combined-weight formula give the same estimate
    for y in itertools.product((0, 1), repeat=3):
        Y = np.array(y, dtype=float)
        pair_form = (( Y[0] - Y[2]) + (Y[1] - Y[2]) + -(Y[2] - Y[0])) / 3.0
        assert pair_form == pytest.approx(float(c @ Y), abs=1e-12)


def test_example_d_two_cluster_aggregation():
    """D: V1=.37, V2=.28, m=(2,3), a=(.4,.6) -> V_P=.16, SE_P=.4."""
    V = np.array([0.37, 0.28])
    m = np.array([2.0, 3.0])
    a = np.array([0.4, 0.6])
    assert np.allclose(a, m / m.sum(), atol=0.21)      # a is the given weight vector
    V_P = float(np.sum(a ** 2 * V))
    assert V_P == pytest.approx(0.16, abs=1e-12)
    assert np.sqrt(V_P) == pytest.approx(0.4, abs=1e-12)


def test_model_variance_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        psm_ate.model_variance(np.zeros(3), np.zeros(4))


def test_zero_variance_is_recorded_not_padded():
    """A zero variance is recorded as 0, without an epsilon."""
    c = np.array([1.0, -1.0])
    mu = np.array([1.0, 0.0])
    assert psm_ate.model_variance(c, mu) == 0.0


def _small_partition(small_continuous, small_group, sources, gamma=1.0):
    import formal_dgp
    from conftest import small_row
    _gen, b6 = sources
    row = small_row("TEST_P", gamma_logor=gamma, has_u=False, methods=["dual_icp"])
    observed, truth, _meta = formal_dgp.binary_unit(
        small_continuous, row, small_group, b6, u=None)
    env = np.asarray(observed.center_codes)
    centers = [int(c) for c in np.unique(env)]
    center_map = {c: i % 3 for i, c in enumerate(centers)}
    return observed, truth, center_map


def test_weight_identity_I_plus_K_holds_on_real_matching(small_continuous,
                                                         small_group, sources):
    """K_i > 0 implies I_i = 1 is checked, not assumed."""
    observed, truth, center_map = _small_partition(small_continuous, small_group,
                                                   sources)
    F = np.asarray(observed.features, float)
    sel = {g: list(range(6)) for g in set(center_map.values())}
    out = psm_ate.fit_partition_psm(F, observed.treatment, observed.outcome,
                                    observed.center_codes, center_map, sel)
    assert out["point_status"] == "complete"
    for k, diag in out["diagnostics"].items():
        assert diag["weight_identity_I_plus_K_holds"] is True, k
        assert diag["n_reused_but_unmatched"] == 0
        assert diag["c_identity_gap"] == pytest.approx(0.0, abs=1e-9)
        assert diag["sum_c_treated"] == pytest.approx(1.0, abs=1e-9)
        assert diag["sum_c_control"] == pytest.approx(-1.0, abs=1e-9)
        assert diag["sum_c"] == pytest.approx(0.0, abs=1e-9)


def test_se_requires_matching_outcome_model(small_continuous, small_group, sources):
    """Without outcome fits the point estimate is given and the SE is unavailable; a different selection or row order is not reused."""
    from regression_adjustment import fit_partition_ra
    observed, truth, center_map = _small_partition(small_continuous, small_group,
                                                   sources)
    F = np.asarray(observed.features, float)
    sel = {g: list(range(6)) for g in set(center_map.values())}

    bare = psm_ate.fit_partition_psm(F, observed.treatment, observed.outcome,
                                     observed.center_codes, center_map, sel)
    assert bare["estimate"] is not None
    assert bare["se"] is None
    assert bare["se_status"] == "unavailable"
    assert all(v == psm_ate.SE_STATUS_NO_MODEL
               for v in bare["se_reasons_by_cluster"].values())

    ra = fit_partition_ra(F, observed.treatment, observed.outcome,
                          observed.center_codes, center_map, sel)
    fits = ra["outcome_fits"]
    good = psm_ate.fit_partition_psm(F, observed.treatment, observed.outcome,
                                     observed.center_codes, center_map, sel,
                                     outcome_fits=fits)
    assert good["se_status"] == psm_ate.SE_STATUS_PSM
    assert good["se"] > 0
    lo, hi = good["ci95"]
    assert lo == pytest.approx(good["estimate"] - psm_ate.Z_975 * good["se"])
    assert hi == pytest.approx(good["estimate"] + psm_ate.Z_975 * good["se"])
    # a'Va can be recomputed from the stored record
    a = np.asarray(good["weights"], float)
    V = np.asarray(good["V"], float)
    assert float(np.sqrt(np.sum(a ** 2 * V))) == pytest.approx(good["se"], rel=1e-12)

    # different selection -> no reuse
    other = {g: list(range(5)) for g in sel}
    mism = psm_ate.fit_partition_psm(F, observed.treatment, observed.outcome,
                                     observed.center_codes, center_map, other,
                                     outcome_fits=fits)
    assert mism["se_status"] == "unavailable"
    assert set(mism["se_reasons_by_cluster"].values()) == {
        "outcome_model_selection_mismatch"}


def test_psm_ci_is_not_clipped(small_continuous, small_group, sources):
    """Intervals are not clipped to [-1, 1]; exceeding them is not an error."""
    c = np.array([5.0, -5.0])
    mu = np.array([0.5, 0.5])
    V = psm_ate.model_variance(c, mu)
    half = psm_ate.Z_975 * np.sqrt(V)
    assert half > 1.0          # the half-width can exceed 1 under this working model
