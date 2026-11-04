"""seam_support_methods.py — within-cluster invariance selection and evaluation helpers.

icp_detail                partial Chow test with BH adjustment for a candidate set, with per-variable detail
align_clusters_hungarian  matching of learned to true center clusters (evaluation only)
truth_index, truth_type_of  covariate roles in the simulated data
"""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from chow_test import bh_adjust, chow_pvalues


def align_clusters_hungarian(true_cluster, center_cluster_hat, K):
    """Map learned cluster labels to true labels by maximal center overlap.

    Both arguments are center-level label vectors. Returns {learned cluster: true cluster}. The mapping is
    used for evaluation; estimation uses the learned partition.
    """
    t = np.asarray(true_cluster).ravel()
    e = np.asarray(center_cluster_hat).ravel()
    assert t.shape == e.shape, f"label vectors differ in length: {t.shape} vs {e.shape}"

    C = np.zeros((K, K), dtype=float)          # C[learned, true] = number of shared centers
    for a, b in zip(e, t):
        if 0 <= int(a) < K and 0 <= int(b) < K:
            C[int(a), int(b)] += 1.0
    r, c = linear_sum_assignment(-C)
    return {int(i): int(j) for i, j in zip(r, c)}


def icp_detail(F_k, Y_k, env_k, candidates, n_centers_k, alpha):
    """Partial Chow test with BH adjustment for every candidate in one cluster.

    A candidate is retained when its BH-adjusted p-value is at least alpha.
      kept_strict    retained candidates (may be empty)
      kept_official  kept_strict, or the max(1, floor(|C|/2)) candidates with the largest raw p-values when
                     every candidate is rejected
    status: ok; empty_candidate (empty candidate set, returned empty); single_center_cluster (no cross-center
    comparison, all candidates retained); no_pvalues (no p-values, all candidates retained).
    """
    cand = sorted(int(j) for j in candidates)

    base = {
        "kept_strict": [], "kept_official": [], "detail": [],
        "icp_fallback_fired": False, "t_icp": 0.0, "n_pre": len(cand),
    }
    if not cand:
        return {**base, "status": "empty_candidate"}
    if n_centers_k <= 1:
        return {**base, "status": "single_center_cluster",
                "kept_strict": list(cand), "kept_official": list(cand)}

    t0 = time.perf_counter()
    raw = chow_pvalues(F_k[:, cand], Y_k, env_k)
    t_icp = time.perf_counter() - t0

    if not raw:
        return {**base, "status": "no_pvalues", "t_icp": t_icp,
                "kept_strict": list(cand), "kept_official": list(cand)}

    local_js = sorted(raw.keys())
    pv_raw = np.array([raw[j] for j in local_js], dtype=float)
    pv_bh = bh_adjust(pv_raw)

    kept_strict = [cand[lj] for i, lj in enumerate(local_js) if not bool(pv_bh[i] < alpha)]

    if kept_strict:
        kept_official, fired = list(kept_strict), False
    else:
        by_stab = sorted(((cand[lj], float(pv_raw[i])) for i, lj in enumerate(local_js)),
                         key=lambda x: -x[1])
        kept_official = sorted(c for c, _ in by_stab[:max(1, len(cand) // 2)])
        fired = True

    detail = [{"var": cand[lj], "pv_raw": float(pv_raw[i]), "pv_bh": float(pv_bh[i]),
               "kept": bool(not (pv_bh[i] < alpha))}
              for i, lj in enumerate(local_js)]

    return {"status": "ok", "kept_strict": sorted(kept_strict),
            "kept_official": sorted(kept_official), "detail": detail,
            "icp_fallback_fired": fired, "t_icp": t_icp, "n_pre": len(cand)}


def truth_index(data):
    """Index sets of the covariate roles in simulated data."""
    spec_by_k = data["true_cluster_specific_causal"]
    return {
        "Z":         set(int(j) for j in data["idx_Z"]),
        "shared":    set(int(j) for j in data["true_shared_causal"]),
        "spec":      set(int(j) for v in spec_by_k.values() for j in v),
        "spec_by_k": {int(k): set(int(j) for j in v) for k, v in spec_by_k.items()},
        "spur":      set(int(j) for j in data["spurious_vars"]),
        "noise":     set(int(j) for j in data["noise_vars"]),
    }


def truth_type_of(j, T):
    """Role of covariate j; priority Z > shared > spec > spur > noise > unknown."""
    j = int(j)
    for name in ("Z", "shared", "spec", "spur", "noise"):
        if j in T[name]:
            return name
    return "unknown"
