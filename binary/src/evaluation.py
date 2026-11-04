"""evaluation.py — truth per analysis cluster, the two coverages, covariate families and failure denominators.

Three levels of truth:
  * true latent cluster k   mean(tau_do_i) of its members, internal check only;
  * learned analysis cluster g   mean(tau_do_i) of its actual members (target of RA and AIPW);
  * PSM target of method m   mean(tau_do_i) of the individuals the method retains, overall weighted by
    the retained sizes.

The two coverages are never merged:
    ci_valid, cover_true=(L<=tau_target<=U), cover_zero=(L<=0<=U),
    reject_zero=(L>0 or U<0); when the CI is invalid the last three are None (NA), not False.
    gamma=0  -> type I error = mean(reject_zero) = 1 - coverage_true (same valid denominator)
    gamma!=0 -> power        = mean(reject_zero) = 1 - coverage_zero (not 1 - cover_true)
"""

from __future__ import annotations

from typing import Any

import numpy as np

FAMILIES = ("Z", "shared", "spec_own", "spec_foreign", "spur", "noise")


# ---------------------------------------------------------------------------
# Truth targets
# ---------------------------------------------------------------------------
def cluster_targets(truth: dict, members_rows: dict[int, np.ndarray]) -> dict:
    """Full-sample targets of the learned analysis clusters and the size-weighted overall target."""
    tau = np.asarray(truth["tau_do_i"], dtype=float)
    per_cluster, sizes = {}, {}
    for g, rows in members_rows.items():
        rows = np.asarray(rows, dtype=np.int64)
        sizes[int(g)] = int(rows.size)
        per_cluster[int(g)] = float(tau[rows].mean()) if rows.size else float("nan")
    n_total = sum(sizes.values())
    overall = (float(sum(per_cluster[g] * sizes[g] for g in per_cluster if sizes[g])
                     / n_total) if n_total else float("nan"))
    return {"by_cluster": per_cluster, "sizes": sizes,
            "overall": overall, "scope": "analysis_cluster_full_members"}


def true_cluster_targets(truth: dict) -> dict:
    """Reference truth of the true latent clusters (internal check only, not a target of any method)."""
    tau = np.asarray(truth["tau_do_i"], dtype=float)
    tcs = np.asarray(truth["true_cluster_sample"], dtype=int)
    out, sizes = {}, {}
    for k in sorted(set(tcs.tolist())):
        mask = tcs == k
        sizes[int(k)] = int(mask.sum())
        out[int(k)] = float(tau[mask].mean()) if mask.any() else float("nan")
    return {"by_true_cluster": out, "sizes": sizes,
            "overall": float(tau.mean()), "scope": "true_latent_cluster"}


def retained_targets(truth: dict, matched_rows_by_cluster: dict[int, np.ndarray],
                     weights: list[float] | None,
                     cluster_order: list | None = None) -> dict:
    """Retained-population targets of PSM per cluster; overall weighted by the retained sizes (the PSM weights).

    The weights are aligned by `cluster_order`: clusters whose fit failed have no entry in `matching_detail`,
    so pairing the weights with `sorted(per_cluster)` would misalign them.
    """
    tau = np.asarray(truth["tau_do_i"], dtype=float)
    per_cluster, sizes = {}, {}
    for g, rows in matched_rows_by_cluster.items():
        rows = np.asarray(rows, dtype=np.int64)
        sizes[int(g)] = int(rows.size)
        per_cluster[int(g)] = float(tau[rows].mean()) if rows.size else float("nan")
    order = ([int(g) for g in cluster_order] if cluster_order is not None
             else sorted(per_cluster))
    if weights is None:
        total = sum(sizes.values())
        weights = [sizes.get(g, 0) / total if total else 0.0 for g in order]
    if len(weights) != len(order):
        raise ValueError(f"{len(weights)} PSM weights for {len(order)} clusters in cluster_order; "
                         "no positional matching")
    overall = float(sum(w * per_cluster[g] for w, g in zip(weights, order)
                        if w > 0 and g in per_cluster
                        and np.isfinite(per_cluster[g])))
    return {"by_cluster": per_cluster, "sizes": sizes, "overall": overall,
            "weight_order": order,
            "scope": "psm_caliper_retained_members"}


# ---------------------------------------------------------------------------
# Effect rows per replication
# ---------------------------------------------------------------------------
def effect_rows(result: dict, estimator: str, truth: dict, targets: dict,
                *, full_targets: dict, target_different: bool) -> list[dict]:
    """Rows of one (method, estimator) result per cluster and overall."""
    rows: list[dict] = []
    cluster_order = list(result.get("cluster_order") or [])
    theta = list(result.get("theta") or [])
    V = result.get("V")
    weights = result.get("weights")

    def _ci_fields(estimate, se, ci, target):
        out = {"ci_valid": False, "cover_true": None, "cover_zero": None,
               "reject_zero": None, "ci_low": None, "ci_high": None,
               "ci_width": None, "invalid_reason": None}
        if ci is None or se is None or not np.isfinite(se) or se <= 0:
            out["invalid_reason"] = ("se_unavailable" if se is None
                                     else "nonpositive_or_nonfinite_se")
            return out
        lo, hi = float(ci[0]), float(ci[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
            out["invalid_reason"] = "nonfinite_or_inverted_interval"
            return out
        out.update(ci_valid=True, ci_low=lo, ci_high=hi, ci_width=hi - lo,
                   cover_zero=bool(lo <= 0.0 <= hi),
                   reject_zero=bool(lo > 0.0 or hi < 0.0))
        out["cover_true"] = (None if target is None or not np.isfinite(target)
                             else bool(lo <= float(target) <= hi))
        return out

    def _error_fields(estimate, target):
        if estimate is None or target is None or not np.isfinite(target):
            return {"error": None, "abs_error": None}
        err = float(estimate) - float(target)
        return {"error": err, "abs_error": abs(err)}

    # --- clusters ---
    for pos, g in enumerate(cluster_order):
        est = theta[pos] if pos < len(theta) else None
        est = None if est is None or not np.isfinite(est) else float(est)
        target = targets["by_cluster"].get(int(g))
        se_g = None
        if isinstance(V, list) and pos < len(V) and V[pos] is not None \
                and np.isfinite(V[pos]) and V[pos] >= 0:
            se_g = float(np.sqrt(V[pos]))
        elif isinstance(V, np.ndarray) and V.ndim == 2 and pos < V.shape[0]:
            val = float(V[pos, pos])
            se_g = float(np.sqrt(val)) if np.isfinite(val) and val >= 0 else None
        ci_g = None
        if est is not None and se_g is not None and se_g > 0:
            ci_g = [est - 1.95996398454 * se_g, est + 1.95996398454 * se_g]
        row = {
            "estimator": estimator, "analysis_cluster": int(g),
            "target_scope": targets["scope"], "stage": "cluster",
            "estimate": est, "se": se_g,
            "truth_target": target, "n_target": targets["sizes"].get(int(g)),
            "point_status": ("complete" if est is not None else "unavailable"),
            "se_status": result.get("se_status"),
            "target_different": bool(target_different),
        }
        row.update(_error_fields(est, target))
        row.update(_ci_fields(est, se_g, ci_g, target))
        rows.append(row)

    # --- overall ---
    est = result.get("estimate")
    est = None if est is None or not np.isfinite(est) else float(est)
    target = targets["overall"]
    row = {
        "estimator": estimator, "analysis_cluster": -1,
        "target_scope": targets["scope"], "stage": "overall",
        "estimate": est, "se": result.get("se"),
        "truth_target": target,
        "n_target": sum(targets["sizes"].values()),
        "point_status": result.get("point_status"),
        "se_status": result.get("se_status"),
        "estimand": result.get("estimand"),
        "weights": weights,
        "target_different": bool(target_different),
        "raw_gap_to_full": (None if est is None else est - float(full_targets["overall"])),
        "target_shift": float(target) - float(full_targets["overall"]),
        "full_target": float(full_targets["overall"]),
    }
    row.update(_error_fields(est, target))
    row.update(_ci_fields(est, result.get("se"), result.get("ci95"), target))
    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Covariate families
# ---------------------------------------------------------------------------
def family_membership(truth: dict, p: int, own_cluster: int | None) -> dict[str, set]:
    spec = truth["true_cluster_specific_causal"]
    spec_own = set(spec.get(int(own_cluster), [])) if own_cluster is not None else set()
    spec_all = set(truth["spec_all"])
    return {
        "Z": set(truth["idx_Z"]),
        "shared": set(truth["true_shared_causal"]),
        "spec_own": spec_own,
        "spec_foreign": spec_all - spec_own,
        "spur": set(truth["spurious_vars"]),
        "noise": set(truth["noise_vars"]),
    }


def selection_metrics(selected, truth: dict, p: int, own_cluster: int | None) -> dict:
    """Recall and false inclusion by covariate family per cluster; the observed_core denominator excludes the unmeasured U."""
    sel = set(int(j) for j in (selected or []))
    fam = family_membership(truth, p, own_cluster)
    out: dict[str, Any] = {"n_selected": len(sel)}
    for name, members in fam.items():
        hit = len(sel & members)
        out[f"n_{name}"] = len(members)
        out[f"sel_{name}"] = hit
        out[f"recall_{name}"] = (hit / len(members)) if members else None
    core = set(truth["observed_core_by_cluster"].get(int(own_cluster), [])) \
        if own_cluster is not None else set()
    out["observed_core_size"] = len(core)
    out["observed_core_hit"] = len(sel & core)
    out["observed_core_recall"] = (len(sel & core) / len(core)) if core else None
    out["observed_core_missing"] = sorted(core - sel)
    out["n_false_positive"] = len(sel - core)
    out["core_denominator_note"] = (
        "the denominator is the observed core; the unmeasured U is not counted, so a measured recall of 100% "
        "does not imply sufficient adjustment" if truth.get("has_unmeasured_U") else "no U")
    return out


__all__ = ["FAMILIES", "cluster_targets", "true_cluster_targets", "retained_targets",
           "effect_rows", "family_membership", "selection_metrics"]
