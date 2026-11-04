"""latent_cluster_dgp_specx_u.py — specx DGP with an unmeasured confounder U.

The specx data are generated first and U is then added to the exposure and the outcome:

    S_i = sum_j shared_ij / sqrt(m * (1 + (m - 1) * rho_block))
    eps ~ N(0, 1) from a separate random stream
    U   = 0.7 * S + 0.3 * eps            # loadings; U is not standardized
    exposure  X += 0.5 * U
    outcome   Y += 0.5 * U

Adding U after generation equals adding it in the structural equations: in specx, Y_e is linear in X_e with a
common coefficient gamma_XY, and all columns of F are generated before X and Y, so
    X' = X + lambda_x U,    Y' = Y + gamma_XY * lambda_x U + lambda_y U.
The shared block has unit variances and common correlation rho_block, so Var(S) = 1.

The treatment effect remains gamma_XY. U is not in F, so A_core_by_cluster is the target set among the
observed covariates but no longer sufficient: adjusting for the shared covariates absorbs 0.7 S, and 0.3 eps
remains as unmeasured confounding.

With lambda_ux = lambda_uy = 0 the output equals specx exactly (checked by verify_specx_u_dgp.py).
"""
from __future__ import annotations

import inspect
import math

import numpy as np

from latent_cluster_dgp_specx import generate_latent_cluster_dgp_specx

SCENARIO_TAG = "specx_u"

# Fixed constants (identical in the binary branch).
U_SHARED_LOADING = 0.7
U_NOISE_LOADING = 0.3
LAMBDA_UX = 0.5
LAMBDA_UY = 0.5
# Separate stream of eps: default_rng([data_seed, U_EPS_STREAM]).
U_EPS_STREAM = 20261003
# rho_block defaults to the default of the specx signature.
RHO_BLOCK_DEFAULT = float(
    inspect.signature(generate_latent_cluster_dgp_specx).parameters["rho_block"].default)


def shared_common_score(F, shared_idx, rho_block: float) -> np.ndarray:
    """Common score of the shared block, scaled by its known population variance."""
    idx = [int(j) for j in shared_idx]
    m = len(idx)
    if m <= 0:
        raise ValueError("the number of shared covariates must be positive")
    denom = math.sqrt(m * (1.0 + (m - 1) * float(rho_block)))
    return np.asarray(F, float)[:, idx].sum(axis=1) / denom


def u_eps_rng(data_seed: int) -> np.random.Generator:
    """Random stream of eps, determined by data_seed and a fixed label."""
    return np.random.default_rng([int(data_seed), U_EPS_STREAM])


def generate_latent_cluster_dgp_specx_u(*, seed, param_seed=None,
                                        lambda_ux: float = 0.0,
                                        lambda_uy: float = 0.0,
                                        u_shared_loading: float = U_SHARED_LOADING,
                                        u_noise_loading: float = U_NOISE_LOADING,
                                        **specx_kwargs) -> dict:
    """specx with the unmeasured confounder U; other arguments are passed to specx."""
    data = generate_latent_cluster_dgp_specx(seed=seed, param_seed=param_seed,
                                             **specx_kwargs)
    F = np.asarray(data["F"], float)
    rho_block = float(specx_kwargs.get("rho_block", RHO_BLOCK_DEFAULT))
    S = shared_common_score(F, data["true_shared_causal"], rho_block)
    eps = u_eps_rng(seed).standard_normal(F.shape[0])
    U = float(u_shared_loading) * S + float(u_noise_loading) * eps

    gamma = float(data["gamma_XY"])
    lx, ly = float(lambda_ux), float(lambda_uy)
    X = np.asarray(data["X_treat"], float)
    Y = np.asarray(data["Y"], float)
    if lx != 0.0 or ly != 0.0:
        X = X + lx * U
        Y = Y + gamma * lx * U + ly * U

    a, b = float(u_shared_loading), float(u_noise_loading)
    out = dict(data)
    out.update(
        X_treat=X, Y=Y, U=U, U_S=S,
        lambda_UX=lx, lambda_UY=ly,
        scenario_tag=SCENARIO_TAG,
        U_meta={
            "u_shared_loading": a, "u_noise_loading": b,
            "population_var_u": a * a + b * b,
            "population_corr_u_s": a / math.sqrt(a * a + b * b),
            "residual_var_given_F": b * b,
            "sample_var_S": float(np.var(S)), "sample_var_U": float(np.var(U)),
            "sample_corr_U_S": float(np.corrcoef(U, S)[0, 1]),
            "not_standardized": True,
            "a_core_sufficient": (lx == 0.0 and ly == 0.0),
        })
    out["params"] = dict(data["params"], scenario=SCENARIO_TAG, lambda_ux=lx,
                         lambda_uy=ly, u_shared_loading=a, u_noise_loading=b,
                         rho_block_for_S=rho_block)
    return out


__all__ = ["SCENARIO_TAG", "U_SHARED_LOADING", "U_NOISE_LOADING", "LAMBDA_UX",
           "LAMBDA_UY", "U_EPS_STREAM", "shared_common_score", "u_eps_rng",
           "generate_latent_cluster_dgp_specx_u"]
