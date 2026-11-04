"""verify_specx_u_dgp.py — checks of the DGP with the unmeasured confounder U.

  A  with lambda = 0, F / X_treat / Y / env equal the specx DGP
  B  U is a confounder: within clusters, given A_core, the change of the OLS gamma when U is added must exceed
     10 times the change with U switched off
  C  construction: Var(S) = 1, Corr(U, S) = 0.919, Var(U) = 0.58; the increments of X and Y equal lambda_x U and
     (gamma lambda_x + lambda_y) U
  D  other clusters' specific covariates remain non-confounders: given A_core and U, adding them changes gamma
     by less than 0.05

All criteria are paired differences within one DGP; the linear fit of the tanh outcome has a baseline
misspecification error common to both adjustment sets.

Usage: python verify_specx_u_dgp.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

os.environ.setdefault("SCENA_BETA_ANCHOR", "1.0,1.25")

import step6x_config as cfg                                         # noqa: E402
import latent_cluster_dgp_specx_u as dgpu                           # noqa: E402
from latent_cluster_dgp_specx import generate_latent_cluster_dgp_specx  # noqa: E402

SEED, PARAM_SEED = 20261003, 77


def _anchor_params() -> dict:
    p = dict(cfg.BASE_PARAMS)
    return p


def _gamma_hat(X, Y, adj) -> float:
    """Coefficient of X in Y ~ 1 + X + adj."""
    design = np.column_stack([np.ones_like(X), X] + ([adj] if adj.size else []))
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)
    return float(coef[1])


def _per_cluster_diff(data, extra_cols_fn, base_cols_fn) -> float:
    """Per true cluster gamma(base) - gamma(base + extra); mean absolute value over clusters."""
    F = np.asarray(data["F"], float)
    X, Y = np.asarray(data["X_treat"], float), np.asarray(data["Y"], float)
    cs = np.asarray(data["true_cluster_sample"], int)
    diffs = []
    for k in sorted(set(cs.tolist())):
        m = cs == k
        base = base_cols_fn(data, k, F[m])
        extra = extra_cols_fn(data, k, F[m], m)
        g0 = _gamma_hat(X[m], Y[m], base)
        g1 = _gamma_hat(X[m], Y[m], np.column_stack([base, extra]))
        diffs.append(g0 - g1)
    return float(np.mean(np.abs(diffs))), [round(d, 4) for d in diffs]


def _core_cols(data, k, Fk):
    return Fk[:, sorted(int(j) for j in data["A_core_by_cluster"][k])]


def _u_col(data, k, Fk, mask):
    return np.asarray(data["U"], float)[mask][:, None]


def _outer_spec_cols(data, k, Fk, mask):
    spec = data["true_cluster_specific_causal"]
    outer = sorted(int(j) for k2, cols in spec.items() if int(k2) != int(k) for j in cols)
    return Fk[:, outer]


def _core_plus_u(data, k, Fk):
    m = np.asarray(data["true_cluster_sample"], int) == k
    return np.column_stack([_core_cols(data, k, Fk), np.asarray(data["U"])[m]])


def main() -> int:
    params = _anchor_params()
    lam = dict(lambda_ux=dgpu.LAMBDA_UX, lambda_uy=dgpu.LAMBDA_UY)
    off = dgpu.generate_latent_cluster_dgp_specx_u(seed=SEED, param_seed=PARAM_SEED,
                                                   **params)
    on = dgpu.generate_latent_cluster_dgp_specx_u(seed=SEED, param_seed=PARAM_SEED,
                                                  **lam, **params)
    base = generate_latent_cluster_dgp_specx(seed=SEED, param_seed=PARAM_SEED, **params)
    results = {}

    # A
    a_ok = all(np.array_equal(np.asarray(off[k]), np.asarray(base[k]))
               for k in ("F", "X_treat", "Y", "env"))
    a_ok &= np.array_equal(np.asarray(on["F"]), np.asarray(base["F"]))
    results["A_bitwise_when_off"] = a_ok
    print(f"[{'PASS' if a_ok else 'FAIL'}] A  lambda = 0 reproduces specx; F is unchanged with U")

    # B
    d_on, per_on = _per_cluster_diff(on, _u_col, _core_cols)
    d_off, per_off = _per_cluster_diff(off, _u_col, _core_cols)
    b_ok = d_on > 10 * d_off and d_on > 0.01
    results["B_u_is_confounder"] = b_ok
    lx, ly, b_load = dgpu.LAMBDA_UX, dgpu.LAMBDA_UY, dgpu.U_NOISE_LOADING
    predicted = (lx * b_load) * (ly * b_load) / (1 + (lx * b_load) ** 2)
    print(f"[{'PASS' if b_ok else 'FAIL'}] B  shift of gamma from omitting U given A_core: with U {d_on:.4f}"
          f" (per cluster {per_on}), without U {d_off:.4f}; theoretical value {predicted:.4f}")

    # C
    meta = on["U_meta"]
    U = np.asarray(on["U"])
    dX = np.asarray(on["X_treat"]) - np.asarray(base["X_treat"])
    dY = np.asarray(on["Y"]) - np.asarray(base["Y"])
    g = float(base["gamma_XY"])
    c_checks = {
        "Var(S)=1": abs(meta["sample_var_S"] - 1.0) < 0.05,
        "Corr(U,S)=0.919": abs(meta["sample_corr_U_S"] - meta["population_corr_u_s"]) < 0.01,
        "Var(U)=0.58": abs(meta["sample_var_U"] - meta["population_var_u"]) < 0.03,
        "dX=lx*U": np.allclose(dX, lx * U, atol=1e-12),
        "dY=(g*lx+ly)*U": np.allclose(dY, (g * lx + ly) * U, atol=1e-12),
    }
    c_ok = all(c_checks.values())
    results["C_frozen_construction"] = c_ok
    print(f"[{'PASS' if c_ok else 'FAIL'}] C  " + "  ".join(
        f"{k}:{'ok' if v else 'FAIL'}" for k, v in c_checks.items()))
    print(f"       Var(S)={meta['sample_var_S']:.4f}  Corr(U,S)={meta['sample_corr_U_S']:.4f}"
          f"  Var(U)={meta['sample_var_U']:.4f}")

    # D
    d_outer, per_outer = _per_cluster_diff(on, _outer_spec_cols, _core_plus_u)
    d_ok = d_outer < 0.05
    results["D_outer_spec_still_not_confounder"] = d_ok
    print(f"[{'PASS' if d_ok else 'FAIL'}] D  change of gamma when other clusters' spec is added to A_core and U: "
          f"{d_outer:.4f} (per cluster {per_outer}) < 0.05")

    print()
    if all(results.values()):
        print("All four checks passed.")
        return 0
    print("Failed:", [k for k, v in results.items() if not v])
    return 1


if __name__ == "__main__":
    sys.exit(main())
