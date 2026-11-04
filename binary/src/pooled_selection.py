"""pooled_selection.py — selection rules of the pooled comparators (one adjustment set from all centers).

Every rule pools all centers, ignores centers and latent clusters, and returns one set S; covariates are
standardized (StandardScaler, ddof=0).

pooled_lasso: L1-logistic Y ~ F (vendor step6b_binary.pooled_logistic_lasso: 3-fold CV, neg_log_loss,
  1-SE rule with the strongest penalty).

hdps: hdPS proxy; the top k covariates by |Spearman(F_j, X)| x |Spearman(F_j, Y)|,
  k = min(p, max(|S_lasso|, 10)).

pooled_pds (Belloni, Chernozhukov & Hansen 2014): S = S_Y ∪ S_X with S_Y = L1-logistic Y ~ F and
  S_X = L1-logistic X ~ F, both from the pooled lasso function (CV 1-SE penalty instead of the BCH plug-in
  penalty), so S_Y is the pooled lasso selection.

pooled_oal (Shortreed & Ertefaie 2017, Biometrics 73:1111-1122):
  1. unpenalized logistic Y ~ X + F (F standardized), coefficients β̃_j of F;
  2. treatment model X ~ F: minimize Σ_i[−a_i η_i + log(1+e^{η_i})] + λ_n Σ_j ω_j|α_j|, ω_j = |β̃_j|^(−γ)
     (summed negative log-likelihood, unpenalized intercept; equation (3));
  3. λ_n ∈ {n^δ: δ ∈ −10, −5, −2, −1, −0.75, −0.5, −0.25, 0.25, 0.49} (the 9 values of the authors' code;
     section 4.1 of the paper prints 8 of them, without −2), with γ = 2(3 − δ) so that λ_n·n^(γ/2−1) = n²;
  4. for each value, IPTW weights τ = A/π̂ + (1−A)/(1−π̂) (equation (4), ATE) and
     wAMD = Σ_j |β̃_j|·|weighted mean_j(treated) − weighted mean_j(control)|; the value with the smallest
     wAMD is chosen (ties: first in the grid);
  5. S = covariates with nonzero coefficients in that treatment model.
  Differences from the published implementation: a logistic outcome model (binary outcome); the 9-value
  grid; the penalized problem is solved exactly by proximal Newton with coordinate descent (lqa gives no
  exact zeros); the propensity score is not truncated.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy.stats import rankdata

OAL_DELTAS = (-10.0, -5.0, -2.0, -1.0, -0.75, -0.5, -0.25, 0.25, 0.49)
OAL_GAMMA_CONVERGENCE = 2.0          # λ_n · n^(γ/2 − 1) = n^2


def oal_gamma(delta: float) -> float:
    """γ such that n^δ · n^(γ/2−1) = n^2, i.e. γ/2 − 1 + δ = 2 and γ = 2(3 − δ)."""
    return 2.0 * (OAL_GAMMA_CONVERGENCE - float(delta) + 1.0)


def _expit(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _nll(eta, y) -> float:
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


_CHUNK = 16384                                      # row blocks, to avoid n×p temporaries


def _col_abs_sum(X):
    out = np.zeros(X.shape[1])
    for i in range(0, X.shape[0], _CHUNK):
        out += np.abs(X[i:i + _CHUNK]).sum(axis=0)
    return out


def _weighted_gram(Xa, w):
    k = Xa.shape[1]
    G = np.zeros((k, k))
    for i in range(0, Xa.shape[0], _CHUNK):
        Xc = Xa[i:i + _CHUNK]
        G += Xc.T @ (Xc * w[i:i + _CHUNK, None])
    return G


# ---------------------------------------------------------------------------
# Weighted L1 logistic regression (unpenalized intercept; pen_j may be 0, tiny, huge or inf)
# ---------------------------------------------------------------------------
def logistic_l1(X, y, pen, *, beta0=None, b0=None, tol=1e-10, max_outer=200,
                max_sweeps=2000) -> dict:
    """min_{b0,β}  Σ_i [log(1+e^{η_i}) − y_i η_i] + Σ_j pen_j |β_j|, η = b0 + Xβ.

    Proximal Newton: the outer loop takes a quadratic approximation at the current point (IRLS Hessian), the
    inner loop solves the quadratic plus weighted L1 problem by coordinate descent on the weighted Gram
    matrix, and a backtracking line search runs on the true objective. Covariates with pen_j ≥ Σ_i|x_ij| are
    zero by the KKT conditions (|∂ℓ/∂β_j| ≤ Σ_i|x_ij| always holds) and are not solved for.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    pen = np.asarray(pen, float)
    n, p = X.shape
    colabs = _col_abs_sum(X)
    with np.errstate(invalid="ignore"):
        free = pen < colabs                     # inf / nan / too large -> False
    act = np.flatnonzero(free)
    Xa = X if act.size == p else X[:, act]          # no copy when every covariate is free
    pa = pen[act]
    k = act.size
    theta = np.zeros(k + 1)
    ybar = min(max(y.mean(), 1e-6), 1 - 1e-6)
    theta[0] = math.log(ybar / (1 - ybar)) if b0 is None else float(b0)
    if beta0 is not None:
        theta[1:] = np.asarray(beta0, float)[act]
    penvec = np.concatenate([[0.0], pa])

    def objective(th):
        return _nll(th[0] + Xa @ th[1:], y) + float(np.sum(penvec * np.abs(th)))

    obj = objective(theta)
    converged, n_outer = False, 0
    for n_outer in range(1, max_outer + 1):
        eta = theta[0] + Xa @ theta[1:]
        mu = _expit(eta)
        w = np.maximum(mu * (1.0 - mu), 1e-12)
        r = mu - y
        g = np.concatenate([[r.sum()], Xa.T @ r])
        H = np.empty((k + 1, k + 1))
        H[0, 0] = w.sum()
        H[0, 1:] = H[1:, 0] = Xa.T @ w
        H[1:, 1:] = _weighted_gram(Xa, w)
        if not np.any(penvec[1:] > 0):          # unpenalized: plain Newton step
            new = theta - np.linalg.solve(H, g)
        else:
            new = theta.copy()
            v = np.zeros(k + 1)                 # v = H (new − theta)
            diag = np.diag(H).copy()
            for _sweep in range(max_sweeps):
                max_change = 0.0
                for j in range(k + 1):
                    hjj = diag[j]
                    rj = g[j] + v[j] - hjj * (new[j] - theta[j])
                    z = hjj * theta[j] - rj
                    if j == 0:
                        t = z / hjj
                    else:
                        lam = penvec[j]
                        t = (z - lam) / hjj if z > lam else ((z + lam) / hjj if z < -lam else 0.0)
                    delta = t - new[j]
                    if delta != 0.0:
                        v += H[j] * delta          # H is symmetric: use the contiguous row
                        new[j] = t
                        change = hjj * delta * delta
                        if change > max_change:
                            max_change = change
                if max_change < tol * tol:
                    break
        d = new - theta
        decrease = float(g @ d + penvec @ (np.abs(new) - np.abs(theta)))
        step, accepted = 1.0, False
        for _ in range(40):
            cand = theta + step * d
            cobj = objective(cand)
            if cobj <= obj + 1e-4 * step * decrease + 1e-12 * abs(obj):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        moved = float(np.max(np.abs(step * d))) if d.size else 0.0
        theta, prev = cand, obj
        obj = cobj
        if moved < tol or abs(prev - obj) <= 1e-14 * max(1.0, abs(obj)):
            converged = True
            break

    beta = np.zeros(p)
    beta[act] = theta[1:]
    beta[np.abs(beta) == 0.0] = 0.0
    # KKT check over all p covariates: |∇_j + pen_j sign| if nonzero, max(0, |∇_j| − pen_j) if zero
    mu = _expit(theta[0] + X @ beta)
    grad = X.T @ (mu - y)
    nz = beta != 0
    kkt_nz = np.abs(grad[nz] + pen[nz] * np.sign(beta[nz]))
    with np.errstate(invalid="ignore"):
        kkt_z = np.maximum(0.0, np.abs(grad[~nz]) - pen[~nz])
    scale = max(1.0, float(np.max(np.abs(grad))) if grad.size else 1.0)
    kkt = max(float(kkt_nz.max()) if kkt_nz.size else 0.0,
              float(np.nanmax(kkt_z)) if kkt_z.size else 0.0) / scale
    return {"intercept": float(theta[0]), "beta": beta, "converged": converged,
            "n_outer": n_outer, "objective": obj, "n_free": int(k),
            "n_fixed_zero_by_kkt": int(p - k), "kkt_violation_rel": kkt,
            "intercept_grad": float(abs((mu - y).sum()))}


def logistic_mle(Z, y, **kw) -> dict:
    """Unpenalized logistic regression (step 1 of OAL)."""
    return logistic_l1(Z, y, np.zeros(np.asarray(Z).shape[1]), **kw)


# ---------------------------------------------------------------------------
# Pooled selection rules
# ---------------------------------------------------------------------------
def _standardize(F):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(np.asarray(F, float))


def _hdps_topk(F, X, Y, k_top):
    """hdPS proxy: score_j = |Spearman(F_j, X)| * |Spearman(F_j, Y)|, top k_top covariates."""
    Frnk = rankdata(F, axis=0).astype(float)
    Frnk -= Frnk.mean(axis=0)
    Xr = rankdata(X).astype(float); Xr -= Xr.mean()
    Yr = rankdata(Y).astype(float); Yr -= Yr.mean()
    denF = np.sqrt((Frnk ** 2).sum(axis=0)) + 1e-12
    rX = np.abs(Frnk.T @ Xr) / (denF * (np.sqrt((Xr ** 2).sum()) + 1e-12))
    rY = np.abs(Frnk.T @ Yr) / (denF * (np.sqrt((Yr ** 2).sum()) + 1e-12))
    return set(np.argsort(rX * rY)[::-1][:k_top].tolist())


def _warning_texts(caught) -> list[str]:
    return [f"{w.category.__name__}: {w.message}" for w in caught]


def pooled_lasso(F, Y, b6) -> dict:
    block = {"method": "pooled_lasso", "definition": "pooled_logistic_lasso_3fold_1se"}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            sel = b6.pooled_logistic_lasso(np.asarray(F, float), np.asarray(Y).astype(int))
            block["elapsed"] = time.perf_counter() - t0
        block["warnings"] = _warning_texts(caught)
        selected = sorted(int(v) for v in sel)
        block.update(status="complete", selected=selected, n_selected=len(selected),
                     legal_empty=bool(len(selected) == 0))
    except Exception as error:                              # noqa: BLE001
        block.update(status="failed", selected=None, reason=f"{type(error).__name__}: {error}")
    return block


def hdps(F, X, Y, lasso_selected) -> dict:
    """hdPS proxy with k = min(p, max(|pooled lasso selection|, 10))."""
    block = {"method": "hdps", "definition": "two_sided_spearman_product_topk_proxy"}
    if lasso_selected is None:
        block.update(status="failed", selected=None, k_top=None, reason="upstream_dependency_failed",
                     upstream="pooled_logistic_lasso")
        return block
    p = int(np.asarray(F).shape[1])
    k_top = min(p, max(len(lasso_selected), 10))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            sel = _hdps_topk(np.asarray(F, float), np.asarray(X).astype(int), np.asarray(Y).astype(int), k_top)
            block["elapsed"] = time.perf_counter() - t0
        block["warnings"] = _warning_texts(caught)
        block.update(status="complete", selected=sorted(int(v) for v in sel), k_top=int(k_top),
                     k_rule="min(p, max(|pooled_selected|, 10))", pooled_n_selected=len(lasso_selected))
    except Exception as error:                              # noqa: BLE001
        block.update(status="failed", selected=None, k_top=int(k_top), reason=f"{type(error).__name__}: {error}")
    return block


def pooled_pds(F, X, Y, b6, *, s_y=None) -> dict:
    """S = S_Y ∪ S_X, both from the pooled lasso selection function; a given s_y is reused."""
    block = {"method": "pooled_pds",
             "definition": "pooled_logistic_lasso(Y~F) UNION pooled_logistic_lasso(X~F); "
                           "3fold neg_log_loss 1-SE (same function as pooled_lasso)",
             "deviation": "penalty by CV 1-SE as in pooled_lasso, not BCH plug-in"}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if s_y is None:
            s_y = b6.pooled_logistic_lasso(np.asarray(F, float), np.asarray(Y).astype(int))
            block["s_y_source"] = "recomputed"
        else:
            block["s_y_source"] = "reused_pooled_lasso_selection"
        s_x = b6.pooled_logistic_lasso(np.asarray(F, float), np.asarray(X).astype(int))
    s_y = sorted(int(j) for j in s_y)
    s_x = sorted(int(j) for j in s_x)
    union = sorted(set(s_y) | set(s_x))
    block.update(status="complete", selected=union, n_selected=len(union),
                 s_y=s_y, s_x=s_x, n_s_y=len(s_y), n_s_x=len(s_x),
                 n_only_x=len(set(s_x) - set(s_y)), legal_empty=not union,
                 elapsed=time.perf_counter() - t0,
                 warnings=sorted({f"{w.category.__name__}: {w.message}" for w in caught}))
    return block


def pooled_oal(F, X, Y, *, deltas=OAL_DELTAS) -> dict:
    """OAL of Shortreed & Ertefaie (2017), equations (3) and (4), on the 9-value grid of the authors' code."""
    t0 = time.perf_counter()
    Fs = _standardize(F)
    A = np.asarray(X).astype(float)
    Yv = np.asarray(Y).astype(float)
    n, p = Fs.shape
    Z = np.column_stack([A, Fs])
    out_fit = logistic_mle(Z, Yv)
    del Z
    beta_t = out_fit["beta"][1:]
    abs_bt = np.abs(beta_t)
    with np.errstate(divide="ignore"):
        log_bt = np.log(abs_bt)                     # β̃ = 0 gives an infinite penalty (excluded)
    grid, fits = [], {}
    warm_beta, warm_b0 = None, None
    for delta in sorted(deltas, reverse=True):      # warm starts from the sparsest value
        gam = oal_gamma(delta)
        log_pen = delta * math.log(n) - gam * log_bt
        with np.errstate(over="ignore"):
            pen = np.where(np.isfinite(log_pen) & (log_pen < 700), np.exp(np.minimum(log_pen, 700)), np.inf)
        fit = logistic_l1(Fs, A, pen, beta0=warm_beta, b0=warm_b0)
        warm_beta, warm_b0 = fit["beta"], fit["intercept"]
        e = _expit(fit["intercept"] + Fs @ fit["beta"])
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.where(A == 1.0, 1.0 / e, 1.0 / (1.0 - e))
        if np.all(np.isfinite(tau)):
            m1 = Fs.T @ (tau * A) / np.sum(tau * A)
            m0 = Fs.T @ (tau * (1.0 - A)) / np.sum(tau * (1.0 - A))
            wamd = float(np.sum(abs_bt * np.abs(m1 - m0)))
        else:                                       # π̂ of exactly 0 or 1: weights undefined, value skipped
            wamd = float("inf")
        sel = np.flatnonzero(fit["beta"] != 0.0)
        fits[delta] = sel
        grid.append({"delta": delta, "gamma": gam, "lambda_n": float(n) ** delta,
                     "wamd": wamd, "n_selected": int(sel.size),
                     "n_fixed_zero_by_kkt": fit["n_fixed_zero_by_kkt"],
                     "converged": fit["converged"], "n_outer": fit["n_outer"],
                     "kkt_violation_rel": fit["kkt_violation_rel"],
                     "ps_min": float(e.min()), "ps_max": float(e.max())})
    grid.sort(key=lambda r: r["delta"])             # ascending grid order
    wamds = [r["wamd"] for r in grid]
    best = int(np.argmin(wamds))                    # ties: first value of the grid
    chosen = grid[best]
    selected = sorted(int(j) for j in fits[chosen["delta"]])
    ok = all(r["converged"] for r in grid) and out_fit["converged"]
    return {"method": "pooled_oal", "status": "complete", "selected": selected,
            "n_selected": len(selected), "legal_empty": not selected,
            "definition": "Shortreed&Ertefaie2017 eq(3)-(4); lambda_n=n^delta, "
                          "gamma=2(3-delta), wAMD(ATE IPTW) argmin",
            "chosen_delta": chosen["delta"], "chosen_gamma": chosen["gamma"],
            "grid": grid, "all_converged": ok,
            "outcome_model": {"converged": out_fit["converged"], "n_outer": out_fit["n_outer"],
                              "beta_tilde_abs_min": float(abs_bt.min()),
                              "beta_tilde_abs_max": float(abs_bt.max()),
                              "treatment_coef": float(out_fit["beta"][0])},
            "elapsed": time.perf_counter() - t0}


__all__ = ["OAL_DELTAS", "oal_gamma", "logistic_l1", "logistic_mle",
           "pooled_lasso", "hdps", "pooled_pds", "pooled_oal"]
