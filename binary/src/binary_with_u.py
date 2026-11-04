"""binary_with_u.py — binary layer with an optional single unmeasured confounder U.

`make_binary` of vendor/step6_pkg/step6b_binary.py repeated line by line, with `lambda_U * U` added to the
two logits; the constants and helpers of the source module are reused (`_sigmoid` / `draw_spec_x_coefs` /
`spec_x_score` / `SIG_X` / `SIG_X_SPEC` / `SIG_Y` / `C0`).

Random streams:
  * `rng_bin`  = default_rng([data_seed, 1]), called in the same order and number as in the source function
    (`random(n)` for X, then `random(n)` for Y), so the result is bitwise identical when lambda is 0;
  * `rng_param`= default_rng([param_seed, 2]), only for `draw_spec_x_coefs`;
  * epsilon has a separate stream default_rng(u_epsilon_seed).

U (fixed loadings, not standardized again):
    S_i = sum_j shared_ij / sqrt(m * (1 + (m-1) * rho_block))
    eps ~ N(0,1) independent
    U   = 0.7 * S + 0.3 * eps          # loadings, not correlations
    eta_X  += 0.5 * U ;  eta_Y(a) += 0.5 * U
Population values: Var(U)=0.58, SD(U)=0.761577, Corr(U,S)=0.919145, Corr(U, shared_j)=0.721397,
residual variance given F 0.09.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# fixed constants (not scanned)
U_SHARED_LOADING = 0.7
U_NOISE_LOADING = 0.3
LAMBDA_UX = 0.5
LAMBDA_UY = 0.5


def shared_common_score(F, shared_idx, rho_block: float) -> np.ndarray:
    """S_i: common score of the shared block, standardized by its known population variance (not the sample SD)."""
    idx = [int(j) for j in shared_idx]
    m = len(idx)
    if m <= 0:
        raise ValueError("number of shared covariates must be positive")
    denom = math.sqrt(m * (1.0 + (m - 1) * float(rho_block)))
    return np.asarray(F, float)[:, idx].sum(axis=1) / denom


def draw_unmeasured_u(S: np.ndarray, rng_eps,
                      *, shared_loading: float = U_SHARED_LOADING,
                      noise_loading: float = U_NOISE_LOADING) -> tuple[np.ndarray, dict]:
    """U = a*S + b*eps on its own scale; U is not standardized and b is not replaced by sqrt(1-a^2)."""
    S = np.asarray(S, float)
    eps = rng_eps.standard_normal(S.shape[0])
    U = float(shared_loading) * S + float(noise_loading) * eps
    meta = {
        "u_shared_loading": float(shared_loading),
        "u_noise_loading": float(noise_loading),
        "population_var_u": float(shared_loading) ** 2 + float(noise_loading) ** 2,
        "population_sd_u": math.sqrt(float(shared_loading) ** 2
                                     + float(noise_loading) ** 2),
        "population_corr_u_s": float(shared_loading) / math.sqrt(
            float(shared_loading) ** 2 + float(noise_loading) ** 2),
        "residual_var_given_F": float(noise_loading) ** 2,
        "sample_var_S": float(np.var(S)),
        "sample_var_U": float(np.var(U)),
        "sample_sd_U": float(np.std(U)),
        "sample_corr_U_S": float(np.corrcoef(U, S)[0, 1]),
        "sample_mean_U": float(U.mean()),
        "not_standardized": True,
    }
    return U, meta


def make_binary_u(data, gamma_logor: float, rng_bin, rng_param, b6,
                  *, u: np.ndarray | None = None,
                  lambda_ux: float = 0.0, lambda_uy: float = 0.0) -> dict[str, Any]:
    """`step6b_binary.make_binary` repeated line by line, plus `lambda*U` in the two logits.

    With `lambda_ux == lambda_uy == 0` the result is bitwise identical to the source function (tested).
    """
    F = np.asarray(data["F"], float)
    Xc = np.asarray(data["X_treat"], float)
    Yc = np.asarray(data["Y"], float)
    cs = np.asarray(data["true_cluster_sample"], int)
    idx_spec = data["true_cluster_specific_causal"]
    K = len(idx_spec)

    if u is None:
        if float(lambda_ux) != 0.0 or float(lambda_uy) != 0.0:
            raise ValueError("nonzero lambda without a U vector")
        U = np.zeros(F.shape[0], dtype=float)
    else:
        U = np.asarray(u, float)
        if U.shape != (F.shape[0],):
            raise ValueError(f"U has shape {U.shape}, sample size is {F.shape[0]}")

    # outcome baseline signal (as in the source function; z_y is not standardized again after adding U)
    y_baseline_signal = Yc - float(data["gamma_XY"]) * Xc
    zx = (Xc - Xc.mean()) / max(Xc.std(), 1e-12)
    z_y_baseline_signal = ((y_baseline_signal - y_baseline_signal.mean())
                           / max(y_baseline_signal.std(), 1e-12))

    # own-cluster spec score (unchanged)
    coef = b6.draw_spec_x_coefs(data, rng_param)
    raw = b6.spec_x_score(F, cs, idx_spec, coef)
    u_spec = np.zeros_like(raw)
    beta_spec_X_eff = {}
    for k in range(K):
        msk = cs == k
        sd_k = max(float(raw[msk].std()), 1e-12) if msk.any() else 1.0
        if msk.any():
            u_spec[msk] = (raw[msk] - float(raw[msk].mean())) / sd_k
        beta_spec_X_eff[k] = b6.SIG_X_SPEC * coef[k] / sd_k

    eta_x = b6.SIG_X * zx + b6.SIG_X_SPEC * u_spec + float(lambda_ux) * U
    Xb = (rng_bin.random(len(zx)) < b6._sigmoid(eta_x)).astype(int)

    eta1 = b6.C0 + gamma_logor + b6.SIG_Y * z_y_baseline_signal + float(lambda_uy) * U
    eta0 = b6.C0 + b6.SIG_Y * z_y_baseline_signal + float(lambda_uy) * U
    eta = np.where(Xb == 1, eta1, eta0)
    Yb = (rng_bin.random(len(eta)) < b6._sigmoid(eta)).astype(int)

    # per-individual do truth with U: the intervention sets a only, U is neither redrawn nor removed from the outcome
    p0_do = b6._sigmoid(eta0)
    p1_do = b6._sigmoid(eta1)
    diff = p1_do - p0_do
    ate_do = float(diff.mean())
    n_k = np.array([int((cs == k).sum()) for k in range(K)])
    ate_k = np.array([float(diff[cs == k].mean()) if n_k[k] else float("nan")
                      for k in range(K)])
    ate_do_nk = float(np.nansum(ate_k * n_k) / max(n_k.sum(), 1))
    return dict(X_bin=Xb, Y_bin=Yb, ate_do=ate_do, ate_k=ate_k, n_k=n_k,
                ate_do_nk=ate_do_nk, beta_spec_X=coef,
                beta_spec_X_eff=beta_spec_X_eff,
                p0_do=p0_do, p1_do=p1_do, tau_do_i=diff,
                lambda_UX=float(lambda_ux), lambda_UY=float(lambda_uy))


__all__ = ["U_SHARED_LOADING", "U_NOISE_LOADING", "LAMBDA_UX", "LAMBDA_UY",
           "shared_common_score", "draw_unmeasured_u", "make_binary_u"]
