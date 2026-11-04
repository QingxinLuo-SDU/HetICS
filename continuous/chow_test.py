"""chow_test.py — partial Chow test of equal covariate-outcome slopes across centers, and BH adjustment.

For covariate j, the restricted model Y ~ center intercepts + beta F_j is compared with the unrestricted
model Y ~ center intercepts + beta_m F_j (one slope per center) by an F test with E - 1 and n - 2E degrees
of freedom, where E is the number of centers.
"""
import numpy as np
from scipy.stats import f as f_dist


def chow_pvalue(F: np.ndarray, Y: np.ndarray, env: np.ndarray, j: int) -> float:
    """p-value of the partial Chow test for column j of F; 1.0 when the test cannot be computed."""
    centers = np.unique(env)
    E, n = len(centers), len(Y)

    x_feats = F[:, j][:, None]
    q = x_feats.shape[1]

    cidx = np.searchsorted(centers, env)
    FE = np.zeros((n, E))
    FE[np.arange(n), cidx] = 1.0

    # Restricted model: common slope and center intercepts.
    X_r = np.column_stack([FE, x_feats])
    try:
        br, _, _, _ = np.linalg.lstsq(X_r, Y, rcond=None)
        RSS_r = float(np.sum((Y - X_r @ br) ** 2))
    except Exception:
        return 1.0

    # Unrestricted model: one slope per center and center intercepts.
    interact = np.hstack([FE * x_feats[:, k : k + 1] for k in range(q)])
    X_u = np.column_stack([FE, interact])
    n_u = E + E * q
    if n - n_u <= 0:
        return 1.0
    try:
        bu, _, _, _ = np.linalg.lstsq(X_u, Y, rcond=None)
        RSS_u = float(np.sum((Y - X_u @ bu) ** 2))
    except Exception:
        return 1.0

    df1 = q * (E - 1)
    df2 = n - n_u
    if df2 <= 0 or RSS_u < 1e-15:
        return 1.0

    F_stat = max(0.0, (RSS_r - RSS_u) / df1 / (RSS_u / df2))
    return float(1.0 - f_dist.cdf(F_stat, df1, df2))


def chow_pvalues(F: np.ndarray, Y: np.ndarray, env: np.ndarray) -> dict:
    """Partial Chow p-values for every column of F, keyed by column index."""
    return {j: chow_pvalue(F, Y, env, j) for j in range(F.shape[1])}


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([])
    order = np.argsort(pvals)
    pv_sorted = pvals[order]
    scaled = pv_sorted * m / np.arange(1, m + 1)
    bh_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    pv_bh = np.empty(m)
    pv_bh[order] = bh_sorted
    return pv_bh
