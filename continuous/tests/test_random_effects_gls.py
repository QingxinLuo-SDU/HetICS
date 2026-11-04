"""REML random-effects combination."""
import numpy as np
import pytest

from random_effects_gls import fit_reml_gls, gls_mean


def test_gls_matches_closed_form():
    theta = np.array([1.0, 1.4, 0.8])
    v = np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])
    mu, var_mu, weights = gls_mean(theta, v, tau2=0.0)
    w = np.linalg.solve(v, np.ones(3))
    w /= w.sum()
    assert weights == pytest.approx(w)
    assert mu == pytest.approx(float(w @ theta))
    assert var_mu == pytest.approx(1.0 / (np.ones(3) @ np.linalg.solve(v, np.ones(3))))


def test_gls_weights_sum_to_one():
    v = np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])
    _, _, weights = gls_mean(np.array([1.0, 1.4, 0.8]), v, tau2=0.05)
    assert weights.sum() == pytest.approx(1.0)


def test_ignoring_cross_cluster_correlation_changes_the_answer():
    """Off-diagonal sampling covariances change mu and its variance."""
    theta = np.array([1.0, 1.4, 0.8])
    v = np.array([[0.04, 0.03, 0.02], [0.03, 0.09, 0.03], [0.02, 0.03, 0.16]])
    mu_full, var_full, _ = gls_mean(theta, v, tau2=0.0)
    mu_diag, var_diag, _ = gls_mean(theta, np.diag(np.diag(v)), tau2=0.0)
    assert abs(mu_full - mu_diag) > 1e-6
    assert abs(var_full - var_diag) > 1e-6


def test_equal_effects_allow_tau2_boundary_zero():
    result = fit_reml_gls(np.array([1., 1., 1.]), np.diag([0.04, 0.05, 0.06]))
    assert result.status == "ok"
    assert result.tau2 >= 0.0
    assert result.tau2 == pytest.approx(0.0, abs=1e-8)
    assert result.boundary_tau2 is True


def test_spread_effects_give_positive_tau2():
    v = np.diag([0.01, 0.01, 0.01])
    result = fit_reml_gls(np.array([0.2, 1.0, 1.8]), v)
    assert result.status == "ok"
    assert result.tau2 > 0.1
    assert result.boundary_tau2 is False


def test_primary_result_has_no_q_selected_model():
    result = fit_reml_gls(np.array([1., 1.2, 0.9]), np.eye(3) * 0.05)
    assert not hasattr(result, "selected_by_q")
    assert "sel_model" not in result.intervals_exploratory


def test_all_interval_fields_are_labelled_exploratory():
    result = fit_reml_gls(np.array([1.0, 1.3, 0.7]), np.eye(3) * 0.04)
    assert result.intervals_exploratory
    assert all(name.endswith("_exploratory")
               for name in result.intervals_exploratory)
    assert "primary_ci" not in result.intervals_exploratory


def test_singular_covariance_is_unavailable_not_crash():
    v = np.zeros((3, 3))
    result = fit_reml_gls(np.array([1.0, 1.1, 0.9]), v)
    assert result.status == "unavailable"
    assert result.error_code == "reml_failed"


def test_nonfinite_theta_is_unavailable():
    result = fit_reml_gls(np.array([1.0, np.nan, 0.9]), np.eye(3) * 0.04)
    assert result.status == "unavailable"
    assert result.error_code in ("theta_vector_incomplete", "reml_failed")
