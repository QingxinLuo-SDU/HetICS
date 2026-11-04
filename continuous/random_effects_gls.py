"""random_effects_gls.py — REML random-effects combination of the cluster-specific estimates.

Model: theta_k = mu + u_k, E(u_k) = 0, Var(u_k) = tau2, with known sampling covariance V.
  V_tau = V + tau2 I,  W = V_tau^{-1},  mu_hat = 1'W theta / 1'W 1.
tau2 maximizes the restricted likelihood on [0, upper]; the boundary 0 is evaluated as a candidate.
With K = 3 the intervals for mu are exploratory.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from v4_types import MetaAnalysisResult, unavailable_meta

_JITTER_FLOOR = 1e-300


def gls_mean(theta, sampling_cov, tau2: float):
    """GLS mean, its variance and the normalized weights for a given tau2."""
    th = np.asarray(theta, dtype=float).ravel()
    k = th.size
    vt = np.asarray(sampling_cov, dtype=float) + float(tau2) * np.eye(k)
    iv_one = np.linalg.solve(vt, np.ones(k))
    denom = float(iv_one.sum())
    if not np.isfinite(denom) or abs(denom) < _JITTER_FLOOR:
        raise np.linalg.LinAlgError("1'W1 is not available")
    weights = iv_one / denom
    return float(weights @ th), float(1.0 / denom), weights


def _nll_reml(theta, sampling_cov, tau2: float):
    """Negative restricted log-likelihood (including log|V| and log(1'W1))."""
    th = np.asarray(theta, dtype=float).ravel()
    k = th.size
    vt = np.asarray(sampling_cov, dtype=float) + float(tau2) * np.eye(k)
    sign_v, logdet_v = np.linalg.slogdet(vt)
    if sign_v <= 0 or not np.isfinite(logdet_v):
        return np.inf, None
    iv_one = np.linalg.solve(vt, np.ones(k))
    s = float(iv_one.sum())
    if not np.isfinite(s) or s <= 0:
        return np.inf, None
    mu = float(iv_one @ th / s)
    resid = th - mu
    quad = float(resid @ np.linalg.solve(vt, resid))
    nll = 0.5 * (logdet_v + np.log(s) + quad)
    if not np.isfinite(nll):
        return np.inf, None
    return nll, (vt, mu, resid, quad, s)


def _tau2_upper(theta, sampling_cov) -> float:
    th = np.asarray(theta, dtype=float).ravel()
    return max(1.0, 10.0 * (float(np.var(th, ddof=1))
                            + float(np.max(np.diag(sampling_cov)))))


def _valid_inputs(theta, sampling_cov):
    """(th, v, None) or (None, None, error_code)."""
    th = np.asarray(theta, dtype=float).ravel()
    v = np.asarray(sampling_cov, dtype=float)
    if th.size == 0 or not np.isfinite(th).all():
        return None, None, "theta_vector_incomplete"
    if v.shape != (th.size, th.size) or not np.isfinite(v).all():
        return None, None, "covariance_invalid"
    if np.any(np.diag(v) <= 0):
        return None, None, "reml_failed"
    return th, v, None


def fit_reml_gls(theta, sampling_cov) -> MetaAnalysisResult:
    """REML estimate of mu; tau2 optimized on [0, upper] with the boundary 0 also evaluated."""
    th, v, bad = _valid_inputs(theta, sampling_cov)
    if bad is not None:
        return unavailable_meta("reml_gls", bad, int(np.asarray(theta).size))
    k = th.size
    upper = _tau2_upper(th, v)

    nll0, pack0 = _nll_reml(th, v, 0.0)
    candidates = [(nll0, 0.0, pack0)]
    converged = True
    try:
        res = optimize.minimize_scalar(
            lambda t: _nll_reml(th, v, max(float(t), 0.0))[0],
            bounds=(0.0, upper), method="bounded",
            options={"xatol": 1e-10},
        )
        converged = bool(res.success)
        tau_int = max(float(res.x), 0.0)
        nll_int, pack_int = _nll_reml(th, v, tau_int)
        candidates.append((nll_int, tau_int, pack_int))
    except Exception:                                     # noqa: BLE001
        converged = False

    candidates = [c for c in candidates if np.isfinite(c[0]) and c[2] is not None]
    if not candidates:
        return unavailable_meta("reml_gls", "reml_failed", k)
    nll, tau2, pack = min(candidates, key=lambda c: c[0])
    vt, mu, resid, quad, s = pack
    var_mu = float(1.0 / s)
    weights = np.linalg.solve(vt, np.ones(k)) / s
    return MetaAnalysisResult(
        method="reml_gls", mu=mu, var_mu=var_mu, tau2=float(tau2),
        weights=weights, converged=converged,
        boundary_tau2=bool(tau2 <= 1e-12), objective=float(nll),
        intervals_exploratory=_exploratory_intervals(vt, mu, var_mu, resid, k),
        status="ok", error_code=None,
    )


def _exploratory_intervals(vt, mu, var_mu, resid, k) -> dict:
    """Wald and modified Knapp-Hartung intervals for mu (exploratory)."""
    se = float(np.sqrt(max(var_mu, 0.0)))
    z = float(stats.norm.ppf(0.975))
    out = {"ci_wald_exploratory": (mu - z * se, mu + z * se)}
    if k > 1:
        quad = float(resid @ np.linalg.solve(vt, resid))
        q_hk = quad / (k - 1)
        var_mkh = max(1.0, q_hk) * var_mu
        crit_t = float(stats.t.ppf(0.975, k - 1))
        out["ci_mkh_exploratory"] = (mu - crit_t * np.sqrt(var_mkh),
                                     mu + crit_t * np.sqrt(var_mkh))
    return out
