"""
latent_cluster_dgp_formal.py — Formal Simulation DGP
=====================================================
Latent-cluster DGP with two parameters added to the base design:

  delta_spec : float (default 0.2)
      Cluster-specific leakage strength.  In cluster k' ≠ k, observations
      receive an additional contribution
          delta_spec * eta_e * C_spec_k @ beta_spec_Y[k]
      where eta_e ~ Uniform(rho_min/rho_max, 1.0) is drawn independently
      per center, introducing inter-center instability in leakage magnitude
      (magnitude range [delta_spec/3, delta_spec] at rho_max=3, rho_min=1).
      delta_spec=0.0 → no leakage across clusters.

  n_per_center_var : float (default 0.1)
      Half-width of per-center sample-size variation.
      n_e = max(10, round(n_per_center * Uniform(1-var, 1+var))).
      n_per_center_var=0.0 → all centers have exactly n_per_center rows.

Defaults (p = 500):
  n_Z              = 20   (4% × p)
  n_causal_shared  = 25   (5% × p)
  n_causal_specific = 5   (1% × p, per cluster)
  n_spurious        = 50  (10% × p)
  n_centers         = 12
  n_per_center      = 2000  (±10% variation → n_per_center_var=0.1)
  K                 = 3
  rho_max           = 3.0,  rho_min = 1.0
  delta_spec        = 0.2
  gamma_XY          = 1.0

The return dict also contains:
  "n_per_center_actual" : list[int]  actual n_e per center
  "delta_spec"          : float

Two-level seed design:

  param_seed : int | None (default None)
      Separate RNG stream for the *parameter instance*: per-center sample
      sizes, causal coefficients (beta_shared_X/Y, beta_spec_Y), spurious
      Dirichlet mixing weights, and per-center leakage scales eta_e.
      With a fixed param_seed these center-level random parameters are
      identical across all data-generation seeds, so each sweep point uses
      ONE parameter instance × N data seeds.
      param_seed=None -> parameter draws share the data RNG stream.

  The return dict also exposes "beta_shared_X".
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dgp_helpers import _block_cov, _causal_func


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sample_beta(n: int, rng: np.random.Generator, n_total: int, kappa: float,
                 mag_lo: float = 0.5, mag_hi: float = 1.0,
                 signed: bool = True) -> np.ndarray:
    """Sample n causal coefficients.  mag ~ Uniform(mag_lo, mag_hi).

    Default magnitude band (0.5, 1.0).
    signed=True (default): random ±1 signs; signed=False: all-positive magnitudes.
    """
    scale = n_total ** (-kappa)
    if signed:
        return rng.choice([-1.0, 1.0], size=n) * (mag_lo + (mag_hi - mag_lo) * rng.random(n)) * scale
    return (mag_lo + (mag_hi - mag_lo) * rng.random(n)) * scale


def _cluster_rho_list(n_cpc: int, rho_max: float, rho_min: float) -> list:
    """Sign-reversing rho pattern within one cluster of n_cpc centers."""
    if n_cpc == 1:
        return [rho_max]
    if n_cpc == 2:
        return [+rho_max, -rho_min]
    return [+rho_max] + [+rho_min] * (n_cpc - 2) + [-rho_min]


# ── Main DGP ──────────────────────────────────────────────────────────────────

def generate_latent_cluster_dgp(
    n_centers:           int   = 12,
    n_per_center:        int   = 2000,
    n_per_center_var:    float = 0.1,
    p_total:             int   = 500,
    K:                   int   = 3,
    n_Z:                 int   = 20,
    n_causal_shared:     int   = 25,
    n_causal_specific:   int   = 5,
    n_spurious:          int   = 50,
    rho_max:             float = 3.0,
    rho_min:             float = 1.0,
    delta_spec:          float = 0.2,
    nu_sd:               float = 0.4,
    alpha_ZX:            float = 1.0,
    alpha_ZY:            float = 1.0,
    gamma_XY:            float = 1.0,
    sigma_Z:             float = 1.0,
    kappa:               float = 0.0,
    f_X:                 str   = "linear",
    f_Y:                 str   = "tanh",
    rho_block:           float = 0.3,
    rho_cross:           float = 0.0,
    beta_mag_lo:         float = 0.5,
    beta_mag_hi:         float = 1.0,
    signed_beta:         bool  = True,
    seed:                int   = 42,
    centers_per_cluster: list  = None,
    param_seed:          int   = None,
) -> dict:
    rng = np.random.default_rng(seed)
    # Two-layer seeds: rng_param drives the parameter instance (sample sizes,
    # betas, spurious weights, eta_e); rng drives the data. When param_seed is
    # None, rng_param aliases rng so the draw sequence is unchanged.
    rng_param = rng if param_seed is None else np.random.default_rng(param_seed)

    if rho_min < 1.0:
        raise ValueError("rho_min must be >= 1.0")

    # ── Cluster assignment ────────────────────────────────────────────────────
    if centers_per_cluster is not None:
        centers_per_cluster = list(centers_per_cluster)
        if len(centers_per_cluster) != K:
            raise ValueError(f"len(centers_per_cluster)={len(centers_per_cluster)} != K={K}")
        if sum(centers_per_cluster) != n_centers:
            raise ValueError(f"sum(centers_per_cluster)={sum(centers_per_cluster)} != n_centers={n_centers}")
        if any(c < 1 for c in centers_per_cluster):
            raise ValueError("All entries of centers_per_cluster must be >= 1.")
        n_cpc_list = centers_per_cluster
    else:
        if n_centers % K != 0:
            raise ValueError(f"n_centers={n_centers} must be divisible by K={K}.")
        n_cpc = n_centers // K
        if n_cpc < 2:
            raise ValueError(f"Need >=2 centers/cluster; got n_centers/K={n_cpc}")
        n_cpc_list = [n_cpc] * K

    cluster_start = [0] * K
    for k in range(1, K):
        cluster_start[k] = cluster_start[k - 1] + n_cpc_list[k - 1]

    center_to_cluster = {}
    for k in range(K):
        for i in range(n_cpc_list[k]):
            center_to_cluster[cluster_start[k] + i] = k

    # ── Per-center sample sizes (±n_per_center_var variation) ────────────────
    if n_per_center_var > 0.0:
        n_per_center_actual = [
            max(10, int(round(n_per_center * rng_param.uniform(
                1.0 - n_per_center_var, 1.0 + n_per_center_var))))
            for _ in range(n_centers)
        ]
    else:
        n_per_center_actual = [n_per_center] * n_centers

    n_total = sum(n_per_center_actual)

    # ── Column layout ─────────────────────────────────────────────────────────
    n_noise = p_total - n_Z - n_causal_shared - K * n_causal_specific - n_spurious
    if n_noise < 0:
        raise ValueError(
            f"p_total={p_total} too small; need at least {p_total - n_noise} columns.")

    cur = 0
    idx_Z        = list(range(cur, cur + n_Z));             cur += n_Z
    idx_shared   = list(range(cur, cur + n_causal_shared)); cur += n_causal_shared
    idx_specific = {}
    for k in range(K):
        idx_specific[k] = list(range(cur, cur + n_causal_specific))
        cur += n_causal_specific
    idx_spurious = list(range(cur, cur + n_spurious));      cur += n_spurious
    idx_noise    = list(range(cur, cur + n_noise))

    true_cluster_arr = np.array([center_to_cluster[e] for e in range(n_centers)])

    # ── Causal coefficients (sampled once, invariant across centers) ──────────
    beta_shared_X = _sample_beta(n_causal_shared,  rng_param, n_total, kappa, beta_mag_lo, beta_mag_hi, signed_beta)
    beta_shared_Y = _sample_beta(n_causal_shared,  rng_param, n_total, kappa, beta_mag_lo, beta_mag_hi, signed_beta)
    beta_spec_Y   = {k: _sample_beta(n_causal_specific, rng_param, n_total, kappa, beta_mag_lo, beta_mag_hi, signed_beta)
                     for k in range(K)}

    Sigma_sh = (_block_cov(n_causal_shared, rho_block)
                if n_causal_shared > 1 else np.eye(max(n_causal_shared, 1)))

    F_chunks, X_chunks, Y_chunks, env_chunks = [], [], [], []
    all_spur_weights = []

    for e in range(n_centers):
        k       = center_to_cluster[e]
        n_cpc_k = n_cpc_list[k]
        e_in_k  = e - cluster_start[k]
        rho_e   = _cluster_rho_list(n_cpc_k, rho_max, rho_min)[e_in_k]
        n_e     = n_per_center_actual[e]

        Z_e   = rng.standard_normal((n_e, n_Z)) * sigma_Z
        Csh_e = (rng.multivariate_normal(np.zeros(n_causal_shared), Sigma_sh, n_e)
                 if n_causal_shared > 0 else np.empty((n_e, 0)))
        Cspec = {}
        for kk in range(K):
            if n_causal_specific > 0:
                if rho_cross > 0.0 and n_causal_shared > 0:
                    # C_spec_j = rho_cross * (normalized Shared projection) + sqrt(1-rho_cross²) * independent noise
                    # column-wise normalization preserves variance structure; backward-compat when rho_cross=0
                    U_sh = Csh_e / (np.linalg.norm(Csh_e, axis=0, keepdims=True) + 1e-10)
                    proj = U_sh[:, :n_causal_specific]
                    indep = rng.standard_normal((n_e, n_causal_specific))
                    Cspec[kk] = rho_cross * proj + np.sqrt(1.0 - rho_cross ** 2) * indep
                else:
                    Cspec[kk] = rng.standard_normal((n_e, n_causal_specific))
            else:
                Cspec[kk] = np.empty((n_e, 0))

        nu_e = rng.standard_normal((n_e, n_spurious)) * nu_sd

        # Z-mix spurious: each variable driven by distinct Dirichlet mix of Z
        Cspur_e   = np.zeros((n_e, n_spurious))
        weights_e = np.zeros((n_spurious, n_Z))
        for s in range(n_spurious):
            w_s           = rng_param.dirichlet(np.ones(n_Z))
            weights_e[s]  = w_s
            Cspur_e[:, s] = rho_e * (Z_e @ w_s) + nu_e[:, s]
        all_spur_weights.append(weights_e)

        Cnoise_e = (rng.standard_normal((n_e, n_noise))
                    if n_noise > 0 else np.empty((n_e, 0)))

        Cspec_cols = (np.concatenate([Cspec[kk] for kk in range(K)], axis=1)
                      if n_causal_specific > 0 else np.empty((n_e, 0)))
        F_e = np.concatenate([Z_e, Csh_e, Cspec_cols, Cspur_e, Cnoise_e], axis=1)

        Z_agg = Z_e.sum(axis=1)
        X_e   = (alpha_ZX * Z_agg
                 + (_causal_func(Csh_e, f_X) @ beta_shared_X if n_causal_shared > 0 else 0.0)
                 + rng.standard_normal(n_e))

        # ── Y: own cluster effect (full) + leakage from other clusters ────────
        Y_shared = (_causal_func(Csh_e, f_Y) @ beta_shared_Y
                    if n_causal_shared > 0 else 0.0)
        Y_spec   = (_causal_func(Cspec[k], f_Y) @ beta_spec_Y[k]
                    if n_causal_specific > 0 else 0.0)

        # Leakage: C_spec_k2 -> Y for k2 != k, strength = delta_spec * eta_e
        # eta_e ~ Uniform(rho_min/rho_max, 1.0) varies per center (instability)
        if delta_spec > 0.0 and n_causal_specific > 0:
            for k2 in range(K):
                if k2 != k:
                    eta_e  = float(rng_param.uniform(rho_min / rho_max, 1.0))
                    Y_spec = Y_spec + delta_spec * eta_e * (
                        _causal_func(Cspec[k2], f_Y) @ beta_spec_Y[k2])

        Y_e = (gamma_XY * X_e + alpha_ZY * Z_agg + Y_shared + Y_spec
               + rng.standard_normal(n_e))

        F_chunks.append(F_e);  X_chunks.append(X_e)
        Y_chunks.append(Y_e);  env_chunks.append(np.full(n_e, e, dtype=int))

    F       = np.vstack(F_chunks)
    X_treat = np.concatenate(X_chunks)
    Y       = np.concatenate(Y_chunks)
    env     = np.concatenate(env_chunks)
    true_cluster_sample = np.array([center_to_cluster[e] for e in env])

    true_causal_by_cluster = {
        k: sorted(idx_shared + idx_specific[k]) for k in range(K)
    }

    return {
        "F":                            F,
        "X_treat":                      X_treat,
        "Y":                            Y,
        "env":                          env,
        "true_cluster":                 true_cluster_arr,
        "true_cluster_sample":          true_cluster_sample,
        "center_to_cluster":            center_to_cluster,
        "true_shared_causal":           idx_shared,
        "true_cluster_specific_causal": idx_specific,
        "true_causal_by_cluster":       true_causal_by_cluster,
        "spurious_vars":                idx_spurious,
        "noise_vars":                   idx_noise,
        "idx_Z":                        idx_Z,
        "gamma_XY":                     gamma_XY,
        "beta_shared_X":                beta_shared_X,
        "beta_shared_Y":                beta_shared_Y,
        "beta_spec_Y":                  beta_spec_Y,
        "scenario_tag":                 "formal",
        "spur_weights":                 all_spur_weights,
        "n_per_center_actual":          n_per_center_actual,
        "delta_spec":                   delta_spec,
        "params": {
            "n_centers": n_centers, "n_per_center": n_per_center,
            "n_per_center_var": n_per_center_var, "p_total": p_total,
            "K": K, "n_Z": n_Z, "n_causal_shared": n_causal_shared,
            "n_causal_specific": n_causal_specific, "n_spurious": n_spurious,
            "n_noise": n_noise, "rho_max": rho_max, "rho_min": rho_min,
            "delta_spec": delta_spec, "rho_cross": rho_cross, "nu_sd": nu_sd,
            "alpha_ZX": alpha_ZX, "alpha_ZY": alpha_ZY,
            "gamma_XY": gamma_XY, "kappa": kappa,
            "f_X": f_X, "f_Y": f_Y, "seed": seed, "param_seed": param_seed,
            "centers_per_cluster": n_cpc_list,
            "beta_mag_lo": beta_mag_lo, "beta_mag_hi": beta_mag_hi,
            "signed_beta": signed_beta,
            "scenario": "formal",
        },
    }
