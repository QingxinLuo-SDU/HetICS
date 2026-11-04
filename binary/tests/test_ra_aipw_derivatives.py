"""Analytic derivatives: the RA and AIPW breads agree with finite differences (A = -mean(du/dlambda))."""

from __future__ import annotations

import numpy as np
import pytest

from aipw_sandwich import build_design, equations_and_bread
from regression_adjustment import build_ra_design, ra_equations_and_bread


def _finite_difference_bread(fun, lam, *, h=1e-6):
    """A_hat = -mean(du/dlambda) by central differences, one parameter at a time."""
    lam = np.asarray(lam, dtype=float)
    base = fun(lam)
    n, k = base.shape
    A = np.zeros((k, lam.size), dtype=float)
    for j in range(lam.size):
        step = np.zeros_like(lam)
        step[j] = h
        plus = fun(lam + step).mean(axis=0)
        minus = fun(lam - step).mean(axis=0)
        A[:, j] = -(plus - minus) / (2 * h)
    return A


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(20260918)
    n, p = 400, 3
    F = rng.normal(size=(n, p))
    env = np.repeat([0, 1, 2], n // 3 + 1)[:n]
    X = (rng.random(n) < 0.45).astype(int)
    Y = (rng.random(n) < 0.35).astype(int)
    return F, X, Y, env


def test_ra_bread_matches_finite_difference(toy):
    F, X, Y, env = toy
    T, T1, T0, _x = build_ra_design(F, env, X, [0, 1, 2])
    q = T.shape[1]
    rng = np.random.default_rng(7)
    b = rng.normal(scale=0.3, size=q)
    theta = 0.11
    lam = np.concatenate([b, [theta]])

    def rows(vec):
        U, _A = ra_equations_and_bread(T, T1, T0, Y, vec[:q], float(vec[q]))
        return U

    _U, A = ra_equations_and_bread(T, T1, T0, Y, b, theta)
    A_fd = _finite_difference_bread(rows, lam)
    assert np.allclose(A, A_fd, atol=1e-6), np.abs(A - A_fd).max()
    assert A[q, q] == 1.0
    assert np.allclose(A[:q, q], 0.0)


def test_aipw_bread_matches_finite_difference(toy):
    F, X, Y, env = toy
    R = build_design(F, env, [0, 1, 2])
    q = R.shape[1]
    rng = np.random.default_rng(11)
    alpha = rng.normal(scale=0.25, size=q)
    beta0 = rng.normal(scale=0.25, size=q)
    beta1 = rng.normal(scale=0.25, size=q)
    theta = 0.07
    lam = np.concatenate([alpha, beta0, beta1, [theta]])

    def rows(vec):
        U, _A = equations_and_bread(R, X, Y, vec[:q], vec[q:2 * q],
                                    vec[2 * q:3 * q], float(vec[3 * q]))
        return U

    _U, A = equations_and_bread(R, X, Y, alpha, beta0, beta1, theta)
    A_fd = _finite_difference_bread(rows, lam)
    assert np.allclose(A, A_fd, atol=1e-5), np.abs(A - A_fd).max()
    # the effect row keeps all nuisance cross terms (not only the diagonal)
    assert not np.allclose(A[3 * q, 0:q], 0.0)
    assert not np.allclose(A[3 * q, q:2 * q], 0.0)
    assert not np.allclose(A[3 * q, 2 * q:3 * q], 0.0)
    assert A[3 * q, 3 * q] == 1.0


def test_aipw_rejects_boundary_propensity(toy):
    F, X, Y, env = toy
    R = build_design(F, env, [0, 1])
    q = R.shape[1]
    huge = np.zeros(q)
    huge[0] = 1e4                      # intercept pushed to the boundary -> e reaches 1 numerically
    with pytest.raises(ValueError):
        equations_and_bread(R, X, Y, huge, np.zeros(q), np.zeros(q), 0.0)


def test_ra_design_layout_is_intercept_centers_x_then_selected(toy):
    F, X, Y, env = toy
    T, T1, T0, x_col = build_ra_design(F, env, X, [2, 0])
    assert np.allclose(T[:, 0], 1.0)
    assert np.array_equal(T1[:, x_col], np.ones(T.shape[0]))
    assert np.array_equal(T0[:, x_col], np.zeros(T.shape[0]))
    # only the X column changes in the counterfactual designs
    others = [j for j in range(T.shape[1]) if j != x_col]
    assert np.array_equal(T[:, others], T1[:, others])
    assert np.array_equal(T[:, others], T0[:, others])
