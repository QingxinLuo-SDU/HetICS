"""step6_methods.py — fixed-X knockoff candidates and the within-cluster partial Chow ICP.

prep                     within-center centering and unit-norm columns
fixed_x_knockoff         equicorrelated fixed-X knockoffs (requires n >= 2p)
lcd_statistics           lasso coefficient differences W from LassoLarsIC with BIC
knockoff_plus            knockoff+ selection from W
icp_filter_bh            partial Chow test with BH adjustment and the fallback rule
knockoff_family_cluster  exposure-side, outcome-side and dual candidates of one cluster, each followed by one ICP
"""
import time

import numpy as np
from sklearn.linear_model import LassoLarsIC

from chow_test import bh_adjust, chow_pvalues

Q_KNOCK = 0.10          # knockoff+ target FDR level
ALPHA_ICP = 0.05        # BH threshold of the ICP; the runners set the configured value


def prep(F_k, Y_k, env_k):
    F = F_k.astype(float).copy()
    Y = Y_k.astype(float).copy()
    for e in np.unique(env_k):
        m = env_k == e
        Y[m] -= Y[m].mean()
        F[m] -= F[m].mean(axis=0, keepdims=True)
    Y -= Y.mean()
    nrm = np.linalg.norm(F, axis=0)
    nrm[nrm < 1e-12] = 1.0
    F /= nrm
    return F, Y


def fixed_x_knockoff(F):
    n, p = F.shape
    assert n >= 2 * p, f"fixed-X knockoffs need n >= 2p, got n={n}, p={p}"
    Sigma = F.T @ F
    Sinv = np.linalg.inv(Sigma + 1e-8 * np.eye(p))
    lam_min = float(np.linalg.eigvalsh(Sigma)[0])
    s = np.full(p, min(1.0, 2.0 * max(lam_min, 0.0)))
    Ds = np.diag(s)
    M = 2 * Ds - Ds @ Sinv @ Ds
    w, V = np.linalg.eigh((M + M.T) / 2)
    w = np.clip(w, 0, None)
    C = (V * np.sqrt(w)) @ V.T
    R = np.random.default_rng(0).standard_normal((n, p))
    R_perp = R - F @ (Sinv @ (F.T @ R))
    U, _ = np.linalg.qr(R_perp)
    U = U[:, :p]
    return F @ (np.eye(p) - Sinv @ Ds) + U @ C


def lcd_statistics(Fp, Ft, resp):
    p = Fp.shape[1]
    beta = LassoLarsIC(criterion="bic").fit(np.hstack([Fp, Ft]), resp).coef_
    return np.abs(beta[:p]) - np.abs(beta[p:])


def knockoff_plus(W, q=Q_KNOCK):
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    tau = np.inf
    for t in ts:
        if (1 + np.sum(W <= -t)) / max(1, np.sum(W >= t)) <= q:
            tau = t
            break
    return set(np.where(W >= tau)[0].tolist())


def icp_filter_bh(F_k, Y_k, env_k, screened):
    """Returns (retained candidates, raw p-values keyed by position in screened).

    A candidate is retained when its BH-adjusted p-value is at least ALPHA_ICP. When every candidate is
    rejected, the max(1, floor(|screened|/2)) candidates with the largest raw p-values are retained.
    """
    if not screened:
        return [], {}
    raw = chow_pvalues(F_k[:, screened], Y_k, env_k)
    local_js = sorted(raw)
    pv = np.array([raw[j] for j in local_js])
    pv_bh = bh_adjust(pv)
    selected = [screened[lj] for i, lj in enumerate(local_js) if not bool(pv_bh[i] < ALPHA_ICP)]
    if not selected:
        by_stab = sorted(((screened[lj], float(pv[i])) for i, lj in enumerate(local_js)), key=lambda x: -x[1])
        selected = [c for c, _ in by_stab[:max(1, len(screened) // 2)]]
    return sorted(selected), raw


def knockoff_family_cluster(F_k, Y_k, X_k, env_k, n_centers_k):
    """Returns ({method: (selected, seconds)}, detail) for xside_icp, yside_icp and dual_icp.

    One knockoff copy serves the three methods; each method's time includes its construction. The dual
    candidate set is the union of the two one-sided sets and goes through one ICP. Single-center clusters
    skip the ICP. detail holds W, the candidates before the ICP and every ICP call with its p-values
    (icp_calls is filled only when all three ICP calls ran).
    """
    t0 = time.perf_counter()
    Fp, Yp = prep(F_k, Y_k, env_k)
    _, Xp = prep(F_k, X_k, env_k)
    Ft = fixed_x_knockoff(Fp)
    t_con = time.perf_counter() - t0

    t0 = time.perf_counter(); W_x = lcd_statistics(Fp, Ft, Xp); S_X = knockoff_plus(W_x); t_lx = time.perf_counter() - t0
    t0 = time.perf_counter(); W_y = lcd_statistics(Fp, Ft, Yp); S_Y = knockoff_plus(W_y); t_ly = time.perf_counter() - t0

    calls = []

    def _icp(cand):
        c = sorted(cand)
        if not c:
            return [], 0.0
        if n_centers_k <= 1:
            return c, 0.0
        t0 = time.perf_counter()
        out, pv = icp_filter_bh(F_k, Y_k, env_k, c)
        calls.append({"pre": [int(j) for j in c], "post": [int(j) for j in out],
                      "pvalues": {int(k): float(v) for k, v in pv.items()}})
        return out, time.perf_counter() - t0

    sx, tx = _icp(S_X)
    sy, ty = _icp(S_Y)
    sd, td = _icp(S_X | S_Y)
    family = {
        "xside_icp": (sx, t_con + t_lx + tx),
        "yside_icp": (sy, t_con + t_ly + ty),
        "dual_icp":  (sd, t_con + t_lx + t_ly + td),
    }
    by_dir = dict(zip(("x", "y", "dual"), calls)) if len(calls) == 3 else {}
    detail = {
        "W": {"x": W_x.tolist(), "y": W_y.tolist()},
        "pre_icp": {"x": sorted(S_X), "y": sorted(S_Y), "dual": sorted(S_X | S_Y)},
        "icp_calls": {d: by_dir.get(d) for d in ("x", "y", "dual")},
        "icp_call_count": len(calls),
        "n_centers_k": int(n_centers_k),
    }
    return family, detail
