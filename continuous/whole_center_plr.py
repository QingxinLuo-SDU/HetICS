"""whole_center_plr.py — cluster-specific Robinson (PLR) estimator with two-fold whole-center cross-fitting.

Folds are formed by centers: all individuals of a center share a fold. The nuisance functions
E[X | W, k] and E[Y | W, k] are fitted on the training centers only (standardization and ridge tuning
included) and predicted for the held-out centers; center indicators are not used as features. The
estimator is unavailable when a fold is empty, the cluster has fewer than two centers, the residual
denominator is zero or predictions are incomplete. The standard error is the center-clustered sandwich.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v4_types import ClusterPLRResult, unavailable_plr

N_FOLDS = 2
RIDGE_ALPHAS = np.logspace(-4, 4, 17)
Z_CRIT = 1.959963984540054


def assign_center_folds(env: np.ndarray, fold_seed: int) -> dict:
    """Assign centers alternately to two folds after a random permutation drawn from fold_seed.

    Returns an empty dict when there are fewer than two centers.
    """
    centers = np.unique(np.asarray(env)).astype(int)
    if centers.size < N_FOLDS:
        return {}
    order = np.random.default_rng(fold_seed).permutation(centers)
    return {int(center): int(pos % N_FOLDS) for pos, center in enumerate(order)}


def _ridge_predict(w_train, target_train, w_valid):
    """Training-fold mean for an empty adjustment set; otherwise standardization and RidgeCV."""
    if w_train.shape[1] == 0:
        return np.full(w_valid.shape[0], float(np.mean(target_train)))
    n_train = w_train.shape[0]
    cv = 3 if n_train >= 9 else 2
    if n_train < 4:
        return np.full(w_valid.shape[0], float(np.mean(target_train)))
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS, cv=cv))
    model.fit(w_train, target_train)
    return np.asarray(model.predict(w_valid), dtype=float)


def fit_whole_center_plr(X, Y, W, env, fold_seed: int) -> ClusterPLRResult:
    """Two-fold whole-center PLR within one cluster; W has shape (n, |S|) and may have no columns."""
    x_all = np.asarray(X, dtype=float).ravel()
    y_all = np.asarray(Y, dtype=float).ravel()
    w_all = np.asarray(W, dtype=float)
    if w_all.ndim == 1:
        w_all = w_all.reshape(-1, 1)
    env_all = np.asarray(env).astype(int).ravel()

    if not (x_all.size == y_all.size == w_all.shape[0] == env_all.size):
        raise ValueError("X, Y, W and env differ in length")
    if not (np.isfinite(x_all).all() and np.isfinite(y_all).all()
            and np.isfinite(w_all).all()):
        return unavailable_plr("nuisance_failed")

    fold_by_center = assign_center_folds(env_all, fold_seed)
    if not fold_by_center:
        return unavailable_plr("fold_not_runnable")
    fold_of_row = np.array([fold_by_center[int(e)] for e in env_all])
    if any((fold_of_row == f).sum() == 0 for f in range(N_FOLDS)):
        return unavailable_plr("fold_not_runnable", fold_by_center)

    x_hat = np.full(x_all.size, np.nan)
    y_hat = np.full(y_all.size, np.nan)
    for fold in range(N_FOLDS):
        valid = fold_of_row == fold
        train = ~valid
        if train.sum() == 0 or valid.sum() == 0:
            return unavailable_plr("fold_not_runnable", fold_by_center)
        try:
            x_hat[valid] = _ridge_predict(w_all[train], x_all[train], w_all[valid])
            y_hat[valid] = _ridge_predict(w_all[train], y_all[train], w_all[valid])
        except Exception:                                  # noqa: BLE001
            return unavailable_plr("nuisance_failed", fold_by_center)

    if not (np.isfinite(x_hat).all() and np.isfinite(y_hat).all()):
        return unavailable_plr("nuisance_failed", fold_by_center)

    x_res = x_all - x_hat
    y_res = y_all - y_hat
    denom = float(x_res @ x_res)
    if not np.isfinite(denom) or denom <= 0.0:
        return unavailable_plr("theta_nonfinite", fold_by_center)
    theta = float((x_res @ y_res) / denom)
    if not np.isfinite(theta):
        return unavailable_plr("theta_nonfinite", fold_by_center)

    # Center-clustered sandwich.
    centers = sorted({int(e) for e in env_all})
    m_centers = len(centers)
    if m_centers < 2:
        return unavailable_plr("fold_not_runnable", fold_by_center)
    resid = y_res - theta * x_res
    center_scores = {c: float(x_res[env_all == c] @ resid[env_all == c])
                     for c in centers}
    scores = np.fromiter(center_scores.values(), dtype=float, count=m_centers)
    variance = float((m_centers / (m_centers - 1.0))
                     * float(((scores - scores.mean()) ** 2).sum()) / denom ** 2)
    if not np.isfinite(variance) or variance < 0.0:
        return unavailable_plr("theta_nonfinite", fold_by_center)
    se = float(np.sqrt(max(variance, 0.0)))
    t_crit = float(stats.t.ppf(0.975, m_centers - 1))
    return ClusterPLRResult(
        theta=theta, se=se, variance=variance,
        ci_z=(theta - Z_CRIT * se, theta + Z_CRIT * se),
        ci_t=(theta - t_crit * se, theta + t_crit * se),
        center_scores=center_scores, fold_by_center=dict(fold_by_center),
        status="ok", error_code=None,
    )


def fold_diagnostics_rows(env, fold_by_center: dict) -> list:
    """Training and validation centers and rows of each fold, with an overlap check."""
    env_all = np.asarray(env).astype(int).ravel()
    rows = []
    for fold in range(N_FOLDS):
        valid_centers = {c for c, f in fold_by_center.items() if f == fold}
        train_centers = {c for c, f in fold_by_center.items() if f != fold}
        valid_rows = int(np.isin(env_all, list(valid_centers)).sum())
        rows.append(dict(
            fold=fold,
            n_train_centers=len(train_centers), n_valid_centers=len(valid_centers),
            n_train_rows=int(env_all.size - valid_rows), n_valid_rows=valid_rows,
            center_overlap=len(train_centers & valid_centers),
            leakage_free=bool(not (train_centers & valid_centers)),
        ))
    return rows
