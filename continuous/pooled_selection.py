"""pooled_selection.py — covariate selection of the comparators, applied once to all centers pooled.

pooled_lasso  Lasso(Y ~ F): StandardScaler, 3-fold LassoCV and the 1-SE rule (run_scenarioA_main._pooled_lasso).
pooled_pds    post-double-selection (Belloni, Chernozhukov and Hansen 2014): Lasso(Y ~ F) union Lasso(X ~ F),
              both with the pooled_lasso function rather than the plug-in penalty.
pooled_oal    outcome-adaptive lasso (Shortreed and Ertefaie 2017), adapted to a continuous exposure:
              1. unpenalized OLS Y ~ 1 + X + F_s gives the outcome coefficients b_j;
              2. adaptive lasso of the exposure model with penalty weights |b_j|^(-gamma);
              3. delta grid {-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49}, lambda_n = n^delta,
                 gamma = 2(g - delta + 1), g = 2;
              4. the lambda minimizing the weighted absolute mean difference (wAMD) is chosen; ties take
                 the first grid value.
              Continuous-exposure adaptation: a Gaussian working model for the exposure with the residual
              variance of the unpenalized OLS (df n - p - 1) and an unpenalized intercept; wAMD sums
              |b_j| |corr_w(F_s,j, X)| over all p covariates with stabilized generalized propensity score
              weights f(X) / f(X | F_S) under normal working models, without truncation.
hdps          hdPS proxy: the k covariates with the largest product of exposure and outcome Spearman
              correlations, k = max(|pooled_lasso|, 10) (run_scenarioA_main._hdps).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

import run_scenarioA_main as off

OAL_DELTAS = (-10.0, -5.0, -2.0, -1.0, -0.75, -0.5, -0.25, 0.25, 0.49)
OAL_GAMMA_FACTOR = 2.0
POOLED_METHODS = ("pooled_lasso", "pooled_pds", "pooled_oal", "hdps")


def oal_gamma(delta: float, gamma_factor: float = OAL_GAMMA_FACTOR) -> float:
    """Pairing rule lambda_n * n^(gamma/2 - 1) = n^g, i.e. gamma = 2(g - delta + 1)."""
    return 2.0 * (gamma_factor - delta + 1.0)


def outcome_coefficients(Fs, X, Y) -> np.ndarray:
    """Coefficients of F in the unpenalized OLS Y ~ 1 + X + F_s."""
    n = Fs.shape[0]
    design = np.column_stack([np.ones(n), np.asarray(X, float), Fs])
    coef, *_ = np.linalg.lstsq(design, np.asarray(Y, float), rcond=None)
    return coef[2:]


NEGLIGIBLE_PENALTY = 1e-8     # penalties below this fraction of the KKT bound are treated as zero


def weighted_lasso_gaussian(Fs, xc, log_w, lam, sigma2):
    """min ||xc - F_s a||^2 / (2 sigma^2) + lam sum_j w_j |a_j| with w_j = exp(log_w_j); F_s and xc centered.

    Equivalent to 1/2 ||xc - F_s a||^2 + sum_j pi_j |a_j| with pi_j = sigma^2 lam w_j. Because lam spans
    n^-10 to n^0.49 and the weights span many orders of magnitude, the columns are split into three groups:
      * pi_j > ||F_j|| ||xc||: a_j = 0 by the KKT conditions;
      * pi_j <= 1e-8 ||F_j|| ||xc||: treated as unpenalized and projected out exactly (Frisch-Waugh);
      * otherwise: an ordinary lasso on the projected residuals with columns rescaled by
        s_j = pi_min / pi_j <= 1 and alpha = pi_min / n.
    """
    n, p = Fs.shape
    alpha = np.zeros(p)
    log_bound = np.log(np.sqrt((Fs ** 2).sum(axis=0)) * np.linalg.norm(xc) + 1e-300)
    log_pen = np.log(sigma2 * lam) + log_w            # log pi_j; infinite weight removes the column
    keep = log_pen <= log_bound
    free = np.flatnonzero(keep & (log_pen <= log_bound + np.log(NEGLIGIBLE_PENALTY)))
    pen = np.flatnonzero(keep & (log_pen > log_bound + np.log(NEGLIGIBLE_PENALTY)))
    if free.size == 0 and pen.size == 0:
        return alpha
    if free.size:
        U = Fs[:, free]
        proj = lambda M: M - U @ np.linalg.lstsq(U, M, rcond=None)[0]
        x_r, F_r = proj(xc), (proj(Fs[:, pen]) if pen.size else np.empty((n, 0)))
    else:
        x_r, F_r = xc, Fs[:, pen]
    if pen.size:
        log_min = log_pen[pen].min()
        scale = np.exp(log_min - log_pen[pen])        # <= 1
        model = Lasso(alpha=float(np.exp(log_min)) / n, fit_intercept=False,
                      precompute=True, max_iter=100_000, tol=1e-10, selection="cyclic")
        model.fit(F_r * scale, x_r)
        alpha[pen] = model.coef_ * scale
    if free.size:
        rest = xc - Fs[:, pen] @ alpha[pen] if pen.size else xc
        alpha[free] = np.linalg.lstsq(Fs[:, free], rest, rcond=None)[0]
    return alpha


def weighted_abs_corr(Fs, x, log_weight) -> np.ndarray:
    """|corr_w(F_s,j, x)| for j = 1..p; weights are normalized after subtracting the largest log weight."""
    w = np.exp(log_weight - np.max(log_weight))
    w = w / w.sum()
    fbar = w @ Fs
    xbar = float(w @ x)
    fd, xd = Fs - fbar, x - xbar
    cov = (w * xd) @ fd
    var_f = w @ (fd ** 2)
    var_x = float(w @ (xd ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(var_f * var_x)
    return np.abs(np.nan_to_num(corr, nan=0.0))


def _normal_logpdf(x, mean, var):
    return -0.5 * np.log(2.0 * np.pi * var) - (x - mean) ** 2 / (2.0 * var)


def pooled_oal(F, X, Y, deltas=OAL_DELTAS, gamma_factor=OAL_GAMMA_FACTOR):
    Fs = StandardScaler().fit_transform(F)           # same preprocessing as pooled_lasso
    x = np.asarray(X, float)
    n, p = Fs.shape
    beta_t = outcome_coefficients(Fs, x, Y)
    with np.errstate(divide="ignore"):
        log_abs_beta = np.log(np.abs(beta_t))         # b_j = 0 gives an infinite weight
    xc = x - x.mean()
    ols, *_ = np.linalg.lstsq(Fs, xc, rcond=None)
    sigma2 = float(np.sum((xc - Fs @ ols) ** 2) / max(n - p - 1, 1))
    x_mean, x_var = float(x.mean()), float(x.var())

    wamd, sizes, fits = [], [], []
    for delta in deltas:
        g = oal_gamma(delta, gamma_factor)
        log_w = -g * log_abs_beta
        a = weighted_lasso_gaussian(Fs, xc, log_w, float(n) ** delta, sigma2)
        fitted = x_mean + Fs @ a
        tau2 = float(np.mean((x - fitted) ** 2))
        log_gps_w = _normal_logpdf(x, x_mean, x_var) - _normal_logpdf(x, fitted, tau2)
        wamd.append(float(np.sum(np.abs(beta_t) * weighted_abs_corr(Fs, x, log_gps_w))))
        sizes.append(int(np.count_nonzero(a)))
        fits.append(a)
    best = int(np.argmin(wamd))                       # ties take the first grid value
    selected = sorted(int(j) for j in np.flatnonzero(fits[best]))
    return selected, {"oal_delta": float(deltas[best]),
                      "oal_gamma": oal_gamma(deltas[best], gamma_factor),
                      "oal_wamd_min": wamd[best],
                      "oal_sizes": " ".join(map(str, sizes)),
                      "oal_wamd": " ".join(f"{v:.6g}" for v in wamd)}


def select_pooled(F, X, Y, methods=POOLED_METHODS):
    """Pooled selected sets, timings and diagnostics; PDS reuses the outcome-side pooled lasso."""
    sets, t_select, diag = {}, {}, {}
    t0 = time.perf_counter()
    lasso = sorted(off._pooled_lasso(F, Y))
    t_select["pooled_lasso"] = time.perf_counter() - t0
    sets["pooled_lasso"] = lasso
    if "pooled_pds" in methods:
        t0 = time.perf_counter()
        s_x = off._pooled_lasso(F, X)
        sets["pooled_pds"] = sorted(set(lasso) | s_x)
        t_select["pooled_pds"] = t_select["pooled_lasso"] + time.perf_counter() - t0
        diag["pooled_pds"] = {"pds_n_from_y": len(lasso), "pds_n_from_x": len(s_x)}
    if "pooled_oal" in methods:
        t0 = time.perf_counter()
        sets["pooled_oal"], diag["pooled_oal"] = pooled_oal(F, X, Y)
        t_select["pooled_oal"] = time.perf_counter() - t0
    if "hdps" in methods:
        t0 = time.perf_counter()
        sets["hdps"] = sorted(off._hdps(F, X, Y, k_top=max(len(lasso), 10)))
        t_select["hdps"] = time.perf_counter() - t0
    return ({m: sets[m] for m in methods}, {m: t_select[m] for m in methods},
            {m: diag.get(m, {}) for m in methods})


__all__ = ["OAL_DELTAS", "OAL_GAMMA_FACTOR", "POOLED_METHODS", "oal_gamma", "pooled_oal", "select_pooled",
           "outcome_coefficients", "weighted_lasso_gaussian", "weighted_abs_corr"]
