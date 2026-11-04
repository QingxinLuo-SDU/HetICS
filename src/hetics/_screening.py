"""_screening.py — input checks, fixed-X knockoff candidates and the partial Chow invariance selection."""
import math
import warnings

import numpy as np
from sklearn.linear_model import LassoLarsIC

from ._chow import bh_adjust, chow_pvalues


def check_inputs(F, X, Y, centers, feature_names, *, binary):
    """Validated float arrays, integer center codes, the original center labels and the covariate names."""
    if feature_names is None and hasattr(F, "columns"):
        feature_names = list(F.columns)
    F, X, Y, centers = (np.asarray(v) for v in (F, X, Y, centers))
    if F.ndim != 2:
        raise ValueError("F must be a two-dimensional array (n, p)")
    n, p = F.shape
    for name, v in (("X", X), ("Y", Y), ("centers", centers)):
        if v.ndim != 1 or v.shape[0] != n:
            raise ValueError(f"{name} must be one-dimensional with {n} rows, the rows of F")
    F, X, Y = F.astype(float), X.astype(float), Y.astype(float)
    if not (np.isfinite(F).all() and np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("F, X and Y must be finite; handle missing values before fitting")
    if binary and not (np.isin(X, (0, 1)).all() and np.isin(Y, (0, 1)).all()):
        raise ValueError("X and Y must be coded 0/1")
    if not binary and (np.ptp(X) == 0 or np.ptp(Y) == 0):
        raise ValueError("X and Y must vary")
    if centers.dtype == object and any(c is None for c in centers):
        raise ValueError("centers has missing identifiers")
    if centers.dtype.kind == "f" and not np.isfinite(centers).all():
        raise ValueError("centers has missing identifiers")
    constant = np.flatnonzero(np.ptp(F, axis=0) == 0).tolist()
    if constant:
        raise ValueError(f"constant columns of F: {constant}")
    seen, duplicate = {}, []
    for j in range(p):
        key = F[:, j].tobytes()
        if key in seen:
            duplicate.append((seen[key], j))
        seen.setdefault(key, j)
    if duplicate:
        raise ValueError(f"duplicate columns of F: {duplicate}")
    if feature_names is not None and len(feature_names) != p:
        raise ValueError(f"feature_names must have {p} entries")
    labels, env = np.unique(centers, return_inverse=True)
    return F, X, Y, env.astype(np.int64), labels, (None if feature_names is None else list(feature_names))


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
    Sigma = F.T @ F
    Sinv = np.linalg.inv(Sigma + 1e-8 * np.eye(p))
    lam_min = float(np.linalg.eigvalsh(Sigma)[0])
    if lam_min <= 1e-10:
        raise ValueError("covariates are collinear within a center cluster after removing center means; "
                         "fixed-X knockoffs have no power (check constant or redundant columns in that cluster)")
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


def lcd_select(Fp, Ft, resp, q):
    """Lasso coefficient differences (LassoLarsIC with BIC) and the knockoff+ threshold.

    Knockoff+ selects no covariate or at least ceil(1/q) of them.
    """
    p = Fp.shape[1]
    beta = LassoLarsIC(criterion="bic").fit(np.hstack([Fp, Ft]), resp).coef_
    W = np.abs(beta[:p]) - np.abs(beta[p:])
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    tau = np.inf
    for t in ts:
        if (1 + np.sum(W <= -t)) / max(1, np.sum(W >= t)) <= q:
            tau = t
            break
    return set(np.where(W >= tau)[0].tolist())


def knockoff_candidates(F_k, Y_k, X_k, env_k, q):
    """Exposure-side and outcome-side candidate sets of one cluster from one knockoff construction."""
    Fp, Yp = prep(F_k, Y_k, env_k)
    _, Xp = prep(F_k, X_k, env_k)
    Ft = fixed_x_knockoff(Fp)
    S_x, S_y = lcd_select(Fp, Ft, Xp, q), lcd_select(Fp, Ft, Yp, q)
    if not S_x | S_y:
        warnings.warn(f"knockoff+ selected no candidate in a cluster: at q={q} it selects none or at least "
                      f"{math.ceil(1 / q)} covariates, so the adjustment set of that cluster is empty and its effect "
                      "estimate is unadjusted for covariates", stacklevel=3)
    return S_x, S_y


def icp_select(F_k, Y_k, env_k, candidates, n_centers_k, alpha, *, fallback):
    """Partial Chow test with BH adjustment; a candidate is retained when its adjusted p-value is at least alpha.

    With fallback=True and every candidate rejected, the max(1, floor(|C|/2)) candidates with the largest raw
    p-values are retained. A single-center cluster has no cross-center comparison and keeps all candidates.
    Returns (retained, detail).
    """
    cand = sorted(int(j) for j in candidates)
    if not cand:
        return [], {"status": "empty_candidate", "fallback_used": False}
    if n_centers_k <= 1:
        return cand, {"status": "single_center_cluster", "fallback_used": False}
    raw = chow_pvalues(F_k[:, cand], Y_k, env_k)
    local = sorted(raw)
    pv = np.array([raw[j] for j in local], dtype=float)
    pv_bh = bh_adjust(pv)
    kept = [cand[j] for i, j in enumerate(local) if not bool(pv_bh[i] < alpha)]
    used = False
    if not kept and fallback:
        by_stab = sorted(((cand[j], float(pv[i])) for i, j in enumerate(local)), key=lambda x: -x[1])
        kept = [c for c, _ in by_stab[:max(1, len(cand) // 2)]]
        used = True
    return sorted(kept), {"status": "ok", "fallback_used": used,
                          "pvalues": {cand[j]: float(pv[i]) for i, j in enumerate(local)},
                          "pvalues_bh": {cand[j]: float(pv_bh[i]) for i, j in enumerate(local)}}


def _cluster_members(env, center_to_cluster):
    if len(center_to_cluster) == 0:
        raise ValueError("center_to_cluster is empty")
    observed_centers = [c.item() if isinstance(c, np.generic) else c for c in np.unique(env)]
    mapping = {int(c): int(k) for c, k in center_to_cluster.items()}
    missing = [c for c in observed_centers if int(c) not in mapping]
    if missing:
        raise ValueError(f"center_to_cluster lacks centers: {missing}")
    members = {}
    for center in observed_centers:
        members.setdefault(mapping[int(center)], []).append(center)
    return {k: members[k] for k in sorted(members)}


def cluster_members(env, center_to_cluster):
    """Centers of each cluster, in ascending cluster order."""
    return _cluster_members(np.asarray(env), center_to_cluster)
