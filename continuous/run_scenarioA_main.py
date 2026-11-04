"""run_scenarioA_main.py — base simulation parameters and the pooled selection rules of two comparators.

K_TRUE, BASE_PARAMS     reference parameters of the latent-cluster DGP (overridden by step6x_config)
_p_scaled_params        variable counts scaled with p for the number-of-covariates axis
_pooled_lasso           Lasso(Y ~ F) on standardized covariates, 3-fold LassoCV and the 1-SE rule
_hdps                   hdPS proxy: top-k product of absolute exposure and outcome Spearman correlations
"""
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

K_TRUE = 3
BASE_PARAMS = {
    "n_centers":         12,
    "n_per_center":      2000,
    "n_per_center_var":  0.1,
    "p_total":           500,
    "K":                 K_TRUE,
    "n_Z":               20,
    "n_causal_shared":   25,
    "n_causal_specific": 5,
    "n_spurious":        50,
    "rho_max":           3.0,
    "rho_min":           1.0,
    "delta_spec":        0.5,
    "gamma_XY":          1.0,
    "kappa":             0.0,
    "f_X":               "linear",
    "f_Y":               "tanh",
    "rho_block":         0.6,
    "rho_cross":         0.3,
    "beta_mag_lo":       2.0,
    "beta_mag_hi":       3.0,
}


def _round_half_up(x: float) -> int:
    """Round half up (Python's round uses banker's rounding)."""
    return int(np.floor(x + 0.5))


def _p_scaled_params(p: int, base: dict) -> tuple:
    """Variable counts proportional to p; returns (params, sEAM sparsity)."""
    K        = base["K"]
    n_Z      = max(1, _round_half_up(0.04 * p))
    n_shared = max(1, _round_half_up(0.05 * p))
    n_spec   = max(1, _round_half_up(0.01 * p))
    n_spur   = max(1, _round_half_up(0.10 * p))
    sparsity = max(_round_half_up(0.20 * p), n_shared + K * n_spec + 5)
    params   = {**base, "p_total": p, "n_Z": n_Z, "n_causal_shared": n_shared,
                "n_causal_specific": n_spec, "n_spurious": n_spur}
    return params, sparsity


def _pooled_lasso(F, Y):
    """Lasso selection with the 1-SE rule: the largest lambda with CV-MSE <= MSE_min + SE_min."""
    sc = StandardScaler()
    Fs = sc.fit_transform(F)
    m  = LassoCV(cv=3, max_iter=5000).fit(Fs, Y)

    n_folds   = m.mse_path_.shape[1]
    mse_mean  = m.mse_path_.mean(axis=1)
    best_idx  = int(np.argmin(mse_mean))
    se_best   = float(m.mse_path_[best_idx].std()) / np.sqrt(n_folds)
    threshold = mse_mean[best_idx] + se_best

    candidates = np.where(mse_mean <= threshold)[0]
    alpha_1se  = (float(m.alphas_[candidates].max()) if len(candidates)
                  else float(m.alpha_))
    coef = Lasso(alpha=alpha_1se, max_iter=5000).fit(Fs, Y).coef_
    return set(np.where(coef != 0)[0].tolist())


def _hdps(F, X, Y, k_top):
    """hdPS proxy: score_j = |Spearman(F_j, X)| * |Spearman(F_j, Y)|, top k_top covariates."""
    Frnk = rankdata(F, axis=0).astype(float)
    Frnk -= Frnk.mean(axis=0)
    Xr = rankdata(X).astype(float); Xr -= Xr.mean()
    Yr = rankdata(Y).astype(float); Yr -= Yr.mean()
    denF = np.sqrt((Frnk ** 2).sum(axis=0)) + 1e-12
    rX = np.abs(Frnk.T @ Xr) / (denF * (np.sqrt((Xr ** 2).sum()) + 1e-12))
    rY = np.abs(Frnk.T @ Yr) / (denF * (np.sqrt((Yr ** 2).sum()) + 1e-12))
    return set(np.argsort(rX * rY)[::-1][:k_top].tolist())
