"""step6_methods.py — building blocks of the exposure-side and outcome-side knockoff selection.

prep              within-center centering and unit-norm columns
fixed_x_knockoff  equicorrelated fixed-X knockoffs (requires n >= 2p)
lcd_select        lasso coefficient difference statistics (LassoLarsIC with BIC) and the knockoff+ threshold
"""
import numpy as np
from sklearn.linear_model import LassoLarsIC

Q_KNOCK = 0.10          # knockoff+ target FDR level


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


def lcd_select(Fp, Ft, resp, q=Q_KNOCK):
    p = Fp.shape[1]
    G = np.hstack([Fp, Ft])
    beta = LassoLarsIC(criterion="bic").fit(G, resp).coef_
    W = np.abs(beta[:p]) - np.abs(beta[p:])
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    tau = np.inf
    for t in ts:
        if (1 + np.sum(W <= -t)) / max(1, np.sum(W >= t)) <= q:
            tau = t
            break
    return set(np.where(W >= tau)[0].tolist())
