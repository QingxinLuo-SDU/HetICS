#!/usr/bin/env python
"""
verify_specx_dgp.py — checks of the specx DGP.

  A. alpha_spec_X = 0 reproduces latent_cluster_dgp_formal exactly.
  B. alpha_spec_X = 1: the own cluster's specific covariates affect both X and Y (confounders).
  C. Asymmetric structure: own-cluster specific covariates enter X with slopes that are stable across centers;
     other clusters' specific covariates do not enter X.
  D. Omitting other clusters' specific covariates causes no bias; omitting the own cluster's causes bias.

Usage: python verify_specx_dgp.py
"""
import sys

import numpy as np

from latent_cluster_dgp_formal import generate_latent_cluster_dgp as gen_formal
from latent_cluster_dgp_specx import generate_latent_cluster_dgp_specx as gen_specx

SMALL = dict(n_centers=6, n_per_center=3000, p_total=120, K=3, n_Z=5,
             n_causal_shared=8, n_causal_specific=4, n_spurious=10,
             rho_block=0.6, rho_cross=0.3, delta_spec=0.5,
             beta_mag_lo=2.0, beta_mag_hi=3.0)


def _mean_abs_corr(F, idx, v, mask):
    if not len(idx):
        return float("nan")
    return float(np.mean([abs(np.corrcoef(F[mask, j], v[mask])[0, 1]) for j in idx]))


def test_a_bitwise_identity():
    """alpha_spec_X = 0 equals the formal DGP."""
    off = gen_formal(**SMALL, seed=7, param_seed=13)
    new = gen_specx(**SMALL, seed=7, param_seed=13, alpha_spec_X=0.0)
    for key in ("F", "X_treat", "Y", "env"):
        assert np.array_equal(off[key], new[key]), f"[A] {key} differs"
    for key in ("true_shared_causal", "idx_Z", "spurious_vars", "noise_vars"):
        assert list(off[key]) == list(new[key]), f"[A] {key} differs"
    print("[A] alpha_spec_X = 0 equals the formal DGP: passed")


def test_b_spec_is_confounder():
    """alpha_spec_X = 1: within a cluster the specific covariates correlate with X and Y."""
    off = gen_formal(**SMALL, seed=7, param_seed=13)
    new = gen_specx(**SMALL, seed=7, param_seed=13, alpha_spec_X=1.0)

    rows = []
    for tag, d in (("formal", off), ("specx", new)):
        env, c2k = d["env"], d["center_to_cluster"]
        k0 = [e for e in np.unique(env) if c2k[int(e)] == 0]
        m = np.isin(env, k0)
        sp0 = d["true_cluster_specific_causal"][0]
        rows.append((tag,
                     _mean_abs_corr(d["F"], sp0, d["X_treat"], m),
                     _mean_abs_corr(d["F"], sp0, d["Y"], m),
                     _mean_abs_corr(d["F"], d["true_shared_causal"], d["X_treat"], m)))

    print("\n   mean |corr| in cluster 0     spec~X    spec~Y   shared~X")
    for tag, cx, cy, csh in rows:
        print(f"   {tag:6s}                    {cx:.4f}    {cy:.4f}    {csh:.4f}")

    off_cx, new_cx = rows[0][1], rows[1][1]
    assert off_cx < 0.05, f"[B] spec~X should be near 0 in the formal DGP, found {off_cx:.4f}"
    assert new_cx > 5 * off_cx, f"[B] specx shows no spec -> X effect: {new_cx:.4f}"

    # beta_spec_X and beta_shared_X are drawn from the same magnitude band.
    b_sp = float(np.mean([np.abs(new["beta_spec_X"][k]).mean() for k in range(SMALL["K"])]))
    b_sh = float(np.abs(new["beta_shared_X"]).mean())
    print(f"   mean |beta|:  spec_X={b_sp:.3f}   shared_X={b_sh:.3f}")
    assert 0.5 < b_sp / b_sh < 2.0, f"[B] coefficient magnitudes differ: {b_sp:.3f} vs {b_sh:.3f}"

    # Omitting the specific covariates: no change in the formal DGP, a large change in specx.
    print("\n   OLS estimate of gamma in cluster 0 (true value 1.0):")
    for tag, d in (("formal", off), ("specx", new)):
        env, c2k, F, X, Y = d["env"], d["center_to_cluster"], d["F"], d["X_treat"], d["Y"]
        m = np.isin(env, [e for e in np.unique(env) if c2k[int(e)] == 0])
        base = sorted(set(d["idx_Z"]) | set(d["true_shared_causal"]))
        full = sorted(set(base) | {j for k in range(SMALL["K"])
                                   for j in d["true_cluster_specific_causal"][k]})

        def ols_gamma(cols):
            W = np.column_stack([X[m], F[m][:, cols], np.ones(m.sum())])
            return float(np.linalg.lstsq(W, Y[m], rcond=None)[0][0])

        g_wo, g_w = ols_gamma(base), ols_gamma(full)
        # The criterion is the difference between the two adjustment sets within one DGP; the linear fit of
        # the tanh outcome has a baseline misspecification error common to both.
        delta = abs(g_wo - g_w)
        print(f"   {tag:6s}  without spec: {g_wo:.4f} (bias {g_wo-1:+.4f})"
              f"   with spec: {g_w:.4f} (bias {g_w-1:+.4f})"
              f"   difference {delta:.4f}")
        if tag == "formal":
            off_delta = delta
        else:
            new_delta = delta

    assert off_delta < 0.05, (
        f"[B] in the formal DGP adjusting for spec should not change the estimate; difference {off_delta:.4f}")
    assert new_delta > 10 * off_delta, (
        f"[B] in specx omitting spec should cause clear bias; difference {new_delta:.4f} "
        f"vs {off_delta:.4f} in the formal DGP")
    print(f"[B] spec is a confounder in specx: omission changes the estimate by {new_delta:.3f} "
          f"({off_delta:.3f} in the formal DGP): passed")


def test_c_asymmetric_spec_structure():
    """Own-cluster spec enters X and Y with stable slopes; other clusters' spec enters only Y."""
    d = gen_specx(**SMALL, seed=7, param_seed=13, alpha_spec_X=1.0)
    F, X, Y, env = d["F"], d["X_treat"], d["Y"], d["env"]
    c2k = d["center_to_cluster"]
    k0 = [int(e) for e in np.unique(env) if c2k[int(e)] == 0]
    own = d["true_cluster_specific_causal"][0]
    foreign = d["true_cluster_specific_causal"][1]

    def slopes(idx, resp):
        """Single-covariate slopes on resp in each center of cluster 0, shape (len(idx), n_center)."""
        out = []
        for j in idx:
            sl = []
            for e in k0:
                m = env == e
                fj = F[m, j] - F[m, j].mean()
                sl.append(float(fj @ (resp[m] - resp[m].mean()) / (fj @ fj)))
            out.append(sl)
        return np.array(out)

    sx_own, sx_for = slopes(own, X), slopes(foreign, X)
    sy_for = slopes(foreign, Y)
    mag_own = float(np.abs(sx_own.mean(axis=1)).mean())
    mag_for = float(np.abs(sx_for.mean(axis=1)).mean())
    cv_own = float(np.mean(np.abs(sx_own.std(axis=1) / (np.abs(sx_own.mean(axis=1)) + 1e-12))))
    cv_for_y = float(np.mean(np.abs(sy_for.std(axis=1) / (np.abs(sy_for.mean(axis=1)) + 1e-12))))

    print(f"\n   |mean slope on X| in cluster 0: own spec={mag_own:.3f}   other spec={mag_for:.3f}")
    print(f"   coefficient of variation of slopes: own spec on X={cv_own:.3f}   other spec on Y={cv_for_y:.3f}")

    assert mag_for < 0.15 * mag_own, (
        f"[C] other clusters' spec should not enter X; slope {mag_for:.3f} vs {mag_own:.3f}")
    assert cv_own < 0.25, f"[C] own-cluster slopes on X should be stable, CV={cv_own:.3f}"
    print("[C] asymmetric structure: passed")


def test_d_foreign_spec_is_not_confounder():
    """Omitting other clusters' spec causes no bias."""
    d = gen_specx(**SMALL, seed=7, param_seed=13, alpha_spec_X=1.0)
    env, c2k, F, X, Y = d["env"], d["center_to_cluster"], d["F"], d["X_treat"], d["Y"]
    m = np.isin(env, [e for e in np.unique(env) if c2k[int(e)] == 0])
    K = SMALL["K"]
    spec = d["true_cluster_specific_causal"]
    A_core = sorted(set(d["idx_Z"]) | set(d["true_shared_causal"]) | set(spec[0]))
    A_safe = sorted(set(A_core) | {j for k in range(K) for j in spec[k]})
    base = sorted(set(d["idx_Z"]) | set(d["true_shared_causal"]))     # without the own cluster's spec

    def ols_gamma(cols):
        W = np.column_stack([X[m], F[m][:, cols], np.ones(m.sum())])
        return float(np.linalg.lstsq(W, Y[m], rcond=None)[0][0])

    g_base, g_core, g_safe = ols_gamma(base), ols_gamma(A_core), ols_gamma(A_safe)
    print("\n   OLS estimate of gamma in cluster 0 (true value 1.0):")
    print(f"   without own spec : {g_base:.4f} (bias {g_base-1:+.4f})")
    print(f"   A_core           : {g_core:.4f} (bias {g_core-1:+.4f})")
    print(f"   A_safe           : {g_safe:.4f} (bias {g_safe-1:+.4f})")

    d_core_safe = abs(g_core - g_safe)
    d_base_core = abs(g_base - g_core)
    assert d_base_core > 0.10, (
        f"[D] omitting the own cluster's spec should cause clear bias; difference {d_base_core:.4f}")
    assert d_core_safe < 0.05, (
        f"[D] other clusters' spec are not confounders; difference {d_core_safe:.4f}")
    print(f"[D] own spec is a confounder (difference {d_base_core:.3f}), "
          f"other spec is not (difference {d_core_safe:.3f}): passed")


def main():
    print("=" * 66)
    print("Checks of latent_cluster_dgp_specx")
    print("=" * 66)
    try:
        test_a_bitwise_identity()
        test_b_spec_is_confounder()
        test_c_asymmetric_spec_structure()
        test_d_foreign_spec_is_not_confounder()
    except AssertionError as exc:
        print(f"\nCheck failed: {exc}")
        sys.exit(1)
    print("\nAll four checks passed.")


if __name__ == "__main__":
    main()
