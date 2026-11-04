"""_continuous.py — HetICS-RA for a continuous exposure and outcome."""
import numpy as np
from scipy import stats

from ._reml import fit_reml_gls
from ._screening import check_inputs, icp_select, knockoff_candidates
from ._seam import run_seam_clustering

Z_CRIT = 1.959963984540054


def _na(code):
    return dict(status="unavailable", error_code=code)


def _cr1_se(D, resid, bread, j, env, w=None):
    """Center-clustered CR1: bread (sum_c s_c s_c') bread x M/(M-1) x (n-1)/(n-p)."""
    n, p = D.shape
    s = D * (resid if w is None else resid * w)[:, None]
    cs = np.array([s[env == c].sum(axis=0) for c in np.unique(env)])
    m = cs.shape[0]
    v = (bread @ (cs.T @ cs) @ bread)[j, j] * m / (m - 1) * (n - 1) / (n - p)
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan")


def _block(theta, se, t_crit):
    if not (np.isfinite(theta) and np.isfinite(se)):
        return _na("theta_nonfinite")
    return dict(status="ok", theta=float(theta), se=float(se),
                ci_t=[float(theta - t_crit * se), float(theta + t_crit * se)],
                ci_z=[float(theta - Z_CRIT * se), float(theta + Z_CRIT * se)])


def regression_adjustment(X, Y, W, env):
    """OLS Y ~ 1 + center indicators + X + W; coefficient of X with center-clustered CR1 SE and t_{M-1} interval."""
    m = np.unique(env).size
    t_crit = float(stats.t.ppf(0.975, m - 1))
    dummies = (env[:, None] == np.unique(env)[None, 1:]).astype(float)
    D = np.column_stack([np.ones(X.size), dummies, X, W])
    j = 1 + dummies.shape[1]
    try:
        bread = np.linalg.inv(D.T @ D)
    except np.linalg.LinAlgError:
        return _na("design_singular")
    beta = bread @ (D.T @ Y)
    return _block(beta[j], _cr1_se(D, Y - D @ beta, bread, j, env), t_crit)


def fit_ra(F, X, Y, centers, *, K, sparsity=None, seed=0, q=0.10, alpha=0.001, feature_names=None):
    """HetICS-RA: center clustering, dual-side selection with invariance selection, and RA within clusters.

    F is the (n, p) array of baseline covariates (no intercept or center dummies); X, Y and centers are vectors of
    length n. K is the prespecified number of center clusters. sparsity is the sEAM support size (default
    round(0.20 p), as in the paper's simulations). q and alpha are the knockoff+ level and the BH threshold of
    the invariance selection.

    Returns a dict with "clusters" (centers, candidate and selected covariates as column indices of F, RA estimate,
    SE and intervals), "overall" (REML combination of the cluster estimates; its intervals are exploratory) and
    "clustering" (sEAM diagnostics). Raises ValueError when a cluster is too small for fixed-X knockoffs or the
    invariance selection, and RuntimeError when an estimate is unavailable.
    """
    F, X, Y, env, labels, names = check_inputs(F, X, Y, centers, feature_names, binary=False)
    n, p = F.shape
    if sparsity is None:
        sparsity = max(1, int(np.floor(0.20 * p + 0.5)))
    if not (K >= 2 and labels.size >= 2 * K):
        raise ValueError(f"need K >= 2 and at least 2 K centers; K={K}, {labels.size} centers")
    if not 1 <= sparsity <= p:
        raise ValueError(f"sparsity must be in [1, {p}]")
    grouping = run_seam_clustering(F, Y, env, K=K, sparsity=sparsity, random_state=seed,
                                   max_iter=300, tol=1e-4, lipschitz_method="exact")
    codes = np.unique(env)
    hat = np.asarray(grouping["center_cluster_hat"])
    clusters = []
    for k in range(K):
        members = codes[hat == k]
        take = np.isin(env, members)
        if members.size < 2 or take.sum() <= 2 * p + 1:
            raise ValueError(f"cluster {k} has {members.size} centers and {int(take.sum())} observations; each cluster "
                             f"needs at least 2 centers and more than {2 * p + 1} observations")
        Fk, Xk, Yk, ek = F[take], X[take], Y[take], env[take]
        sx, sy = knockoff_candidates(Fk, Yk, Xk, ek, q)
        selected, detail = icp_select(Fk, Yk, ek, sx | sy, members.size, alpha, fallback=False)
        ra = regression_adjustment(Xk, Yk, Fk[:, selected], ek)
        if ra["status"] != "ok" or not ra["se"] > 0:
            raise RuntimeError(f"cluster {k}: RA estimate unavailable: {ra}")
        clusters.append({"cluster": k, "centers": labels[members].tolist(), "n": int(take.sum()),
                         "candidates_exposure": sorted(sx), "candidates_outcome": sorted(sy),
                         "selected": selected,
                         "selected_names": None if names is None else [names[j] for j in selected],
                         "invariance": detail, **{c: ra[c] for c in ("theta", "se", "ci_t", "ci_z")}})
    meta = fit_reml_gls([c["theta"] for c in clusters], np.diag([c["se"] ** 2 for c in clusters]))
    overall = {"status": meta.status, "error_code": meta.error_code, "mu": meta.mu, "se": float(np.sqrt(meta.var_mu)),
               "tau2": meta.tau2, "weights": np.asarray(meta.weights).tolist(), "converged": meta.converged,
               "boundary_tau2": meta.boundary_tau2, "intervals_exploratory": meta.intervals_exploratory}
    return {"method": "HetICS-RA", "clusters": clusters, "overall": overall, "clustering": grouping["diagnostics"],
            "settings": {"K": K, "sparsity": sparsity, "seed": seed, "q": q, "alpha": alpha}}
