"""psm_ate.py — propensity score matching: both directions, one nearest neighbour, with replacement, logit-PS caliper.

Each individual is matched to the nearest individual of the other treatment group on the propensity score
fitted within the cluster; only pairs within the caliper on the logit scale are kept. Matching in both
directions targets the ATE; when the caliper drops individuals, the target is the retained sample.

    caliper_k = 0.20 * std(lp, ddof=0)      lp = logit propensity score (fitted on the whole cluster)
    distance  = |lp_i - lp_j|, accepted if <= caliper; ties go to the smallest original row index
    tau_i     = (2X_i-1) * (Y_i - Y_j(i))
    theta_k   = mean_{i in M_k}(tau_i)      M_k = matched original individuals

The caliper changes the target population. Only when everyone is matched is the estimand
`full_sample_ate`; otherwise it is `caliper_retained_sample_ate`, aggregated with weights m_k/sum(m_k) and
compared with the truth of the matched population `ate_do_matched`; `raw_gap_to_full_ate` relative to the
full population is kept separately.

Standard errors use a conditional Bernoulli outcome working model on the fixed matching graph:

    c_i = s_i (I_i + K_i) / m_k        theta_k = sum_i c_i Y_i   (exact identity)
    V_k = sum_i c_i^2 mu_hat_i (1 - mu_hat_i)
    CI  = theta_k ± 1.95996398454 * sqrt(V_k)        (not clipped to [-1, 1])

mu_hat is the prediction at the observed treatment of the unpenalized logistic outcome model with the same
design, selection, partition and row order as RA, fitted once on the original individuals (the duplicated
rows of matching with replacement are not new data). Overall a_k = m_k / sum_l m_l, V = diag(V_k),
SE_P = sqrt(a'Va).

This approximates independent Bernoulli outcome noise given the matching graph:
`se_method=psm_conditional_bernoulli_fixed_graph_v1`, `variance_target=fixed_graph_outcome_noise`,
`conditioning=working_design`. It is not the Abadie-Imbens matching variance; it ignores the variability of
the PS estimate, of the matching graph and retained set, and of label and selection learning, and does not
guarantee 95% coverage of the causal truth.

It does not reuse the AIPW or RA sandwich or the RA SE, does not treat pair differences of reused matches as
independent, adds no HC1 or degrees-of-freedom factor, and adds no epsilon to a zero variance.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aipw_sandwich import (_check_selected, _fit_one, _statsmodels_module,
                           build_design, statsmodels_import_error)
from screening_bridge import cluster_members

CALIPER_K = 0.20
SE_METHOD_PSM = "psm_conditional_bernoulli_fixed_graph_v1"
SE_STATUS_PSM = "candidate_model_se_available"
SE_STATUS_NO_MODEL = "outcome_model_not_supplied"
VARIANCE_TARGET_PSM = "fixed_graph_outcome_noise"
Z_975 = 1.95996398454
ESTIMAND_FULL = "full_sample_ate"
ESTIMAND_RETAINED = "caliper_retained_sample_ate"


def nearest_within_caliper(lp, X, caliper, row_ids=None):
    """Matching in both directions with one nearest neighbour; returns `match` of length n (-1: none in the caliper).

    The one-dimensional scores are sorted and searchsorted for the neighbours on both sides, without an n x n
    distance matrix or a fixed index window. Ties go to the smallest original row index (`row_ids`; position
    when not given), so a permutation of the rows does not change the matching.
    """
    lp = np.asarray(lp, dtype=float)
    X = np.asarray(X).astype(int)
    n = lp.shape[0]
    if X.shape != (n,):
        raise ValueError("X and lp must have the same length")
    ids = (np.arange(n, dtype=np.int64) if row_ids is None
           else np.asarray(row_ids, dtype=np.int64))
    if ids.shape != (n,):
        raise ValueError("row_ids must have the length of lp")
    caliper = float(caliper)
    if not np.isfinite(caliper) or caliper < 0:
        raise ValueError(f"caliper must be a finite nonnegative number, got {caliper}")

    pools = {}
    for arm in (0, 1):
        idx = np.flatnonzero(X == arm)
        order = np.lexsort((ids[idx], lp[idx]))     # by lp, then by original row index
        pools[arm] = (lp[idx][order], idx[order])

    match = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        s, oidx = pools[1 - X[i]]
        if s.size == 0:
            continue
        v = lp[i]
        pos = int(np.searchsorted(s, v))
        near = [jj for jj in (pos - 1, pos) if 0 <= jj < s.size]
        if not near:
            continue
        dists = [abs(s[jj] - v) for jj in near]
        best = min(dists)
        if best > caliper:
            continue
        # candidates at the smallest distance can only be these two neighbouring score values; within the run
        # of each value oidx is ascending, so the first entry has the smallest original row index. The run is
        # found from the matched index, not from v ± best (floating-point arithmetic could miss ties).
        candidates = []
        for jj, d in zip(near, dists):
            if d == best:
                start = int(np.searchsorted(s, s[jj], side="left"))
                candidates.append(int(oidx[start]))
        match[i] = min(candidates, key=lambda j: int(ids[j]))
    return match


def _fit_ps_cluster(F_k, X_k, env_k, selected, *, backend="statsmodels_glm"):
    """Unpenalized logistic propensity model on the whole cluster; returns (lp, diagnostics)."""
    sm = _statsmodels_module()
    if backend == "statsmodels_glm" and sm is None:
        raise RuntimeError("backend statsmodels_glm was requested but statsmodels is unavailable: "
                           f"{statsmodels_import_error()} (no automatic switch to IRLS)")
    if backend == "irls_numpy":
        sm = None
    elif backend != "statsmodels_glm":
        raise ValueError(f"unknown logistic backend: {backend}")

    R = build_design(F_k, env_k, selected)
    n_k, q = R.shape
    if not np.any(X_k == 1) or not np.any(X_k == 0):
        raise RuntimeError("one treatment arm is missing in the cluster; PSM not estimable")
    if n_k <= q:
        raise RuntimeError(f"cluster sample size {n_k} is too small for {q} parameters")
    rank = int(np.linalg.matrix_rank(R))
    if rank < q:
        raise RuntimeError(f"PS design matrix is not of full column rank (rank={rank}<{q})")

    fit, messages = _fit_one("propensity", X_k, R, sm)
    gamma = np.asarray(fit.params, dtype=float)
    lp = R @ gamma                                  # logit PS, the linear predictor
    if not np.all(np.isfinite(lp)):
        raise RuntimeError("logit propensity scores are not finite")
    e = np.asarray(fit.predict(R), dtype=float)
    if np.any(e <= 0.0) or np.any(e >= 1.0):
        raise RuntimeError("propensity predictions reach 0 or 1 numerically (no default clipping)")
    diagnostics = {
        "n_k": int(n_k), "q": int(q), "design_rank": rank,
        "n_treated": int(np.sum(X_k == 1)), "n_control": int(np.sum(X_k == 0)),
        "e_min": float(e.min()), "e_max": float(e.max()),
        "lp_std_ddof0": float(np.std(lp, ddof=0)),
        "converged": bool(getattr(fit, "converged", True)),
        "fit_warnings": messages, "penalty": "none", "clipping_applied": False,
        "fitter": "statsmodels_glm" if sm is not None else "irls_numpy",
    }
    return lp, diagnostics


def _smd(F_sub, X_sub, weights, ref_var):
    """Weighted SMD; the denominator is the square root of the mean unweighted variance of the original groups."""
    out = []
    for j in range(F_sub.shape[1]):
        col = F_sub[:, j]
        m1 = np.average(col[X_sub == 1], weights=weights[X_sub == 1]) \
            if np.any((X_sub == 1) & (weights > 0)) else np.nan
        m0 = np.average(col[X_sub == 0], weights=weights[X_sub == 0]) \
            if np.any((X_sub == 0) & (weights > 0)) else np.nan
        denom = np.sqrt(ref_var[j])
        if denom <= 0:
            out.append(0.0 if (np.isfinite(m1) and np.isfinite(m0) and m1 == m0)
                       else float("nan"))
        else:
            out.append(float((m1 - m0) / denom))
    return out


def _psm_cluster(F_k, X_k, Y_k, env_k, row_ids, selected, *, backend):
    """PS fit, caliper matching in both directions, and estimate and diagnostics of the retained population in one cluster."""
    lp, diag = _fit_ps_cluster(F_k, X_k, env_k, selected, backend=backend)
    caliper = CALIPER_K * float(np.std(lp, ddof=0))
    match = nearest_within_caliper(lp, X_k, caliper, row_ids=row_ids)
    matched = match >= 0
    m_k = int(matched.sum())

    sign = 2.0 * X_k.astype(float) - 1.0
    if m_k:
        tau = sign[matched] * (Y_k[matched].astype(float)
                               - Y_k[match[matched]].astype(float))
        theta_k = float(np.mean(tau))
        # check with the reuse counts: theta = m^-1 sum (2X-1)(1+K) Y
        K = np.bincount(match[matched], minlength=X_k.shape[0]).astype(float)
        weights = np.where(matched, 1.0 + K, 0.0)
        theta_alt = float(np.sum(sign * weights * Y_k.astype(float)) / m_k)
        if abs(theta_alt - theta_k) > 1e-9 * max(1.0, abs(theta_k)):
            raise RuntimeError(f"the two point-estimate formulas disagree: {theta_k} vs {theta_alt}")
        # combined weights c_i = s_i (I_i + K_i) / m_k. The code uses I_i (1 + K_i); the two are equal if and
        # only if K_i > 0 implies I_i = 1, which is checked and recorded for every unit.
        I = matched.astype(float)
        c_vec = sign * (I + K) / m_k
        theta_c = float(np.sum(c_vec * Y_k.astype(float)))
        weight_identity_holds = bool(np.array_equal(I + K, weights))
        k_positive_without_match = int(np.sum((K > 0) & (~matched)))
        identity_gap = abs(theta_c - theta_k)
    else:
        theta_k, K = float("nan"), np.zeros(X_k.shape[0])
        weights = np.zeros(X_k.shape[0])
        theta_alt = float("nan")
        c_vec = np.zeros(X_k.shape[0])
        theta_c = float("nan")
        weight_identity_holds = True
        k_positive_without_match = 0
        identity_gap = float("nan")

    ref_var = (F_k[X_k == 1].var(axis=0) + F_k[X_k == 0].var(axis=0)) / 2.0
    sel_cols = np.asarray(selected, dtype=int)
    F_sel = F_k[:, sel_cols] if sel_cols.size else F_k[:, :0]
    ref_sel = ref_var[sel_cols] if sel_cols.size else np.zeros(0)
    before = _smd(F_sel, X_k, np.ones(X_k.shape[0]), ref_sel) if sel_cols.size else []
    after = _smd(F_sel, X_k, weights, ref_sel) if (sel_cols.size and m_k) else []

    dist = np.where(matched, np.abs(lp - lp[np.where(matched, match, 0)]), np.nan)
    cross = (env_k != env_k[np.where(matched, match, 0)]) & matched
    detail = {
        "row_id": [int(v) for v in row_ids],
        "matched": [bool(v) for v in matched],
        "match_row_id": [int(row_ids[match[i]]) if matched[i] else None
                         for i in range(match.shape[0])],
        "lp": [float(v) for v in lp],
        "distance": [None if not matched[i] else float(dist[i])
                     for i in range(match.shape[0])],
        "reuse_count_K": [int(v) for v in K],
        "weight": [float(v) for v in weights],
        "treatment": [int(v) for v in X_k],
        "center": [int(v) for v in env_k],
    }
    diag.update({
        "caliper_k": float(caliper), "caliper_rule": "0.20*std(logit_ps, ddof=0)",
        "n_original": int(X_k.shape[0]), "n_matched": m_k,
        "retention_rate": float(m_k / max(X_k.shape[0], 1)),
        "retention_treated": float(np.sum(matched & (X_k == 1))
                                   / max(int(np.sum(X_k == 1)), 1)),
        "retention_control": float(np.sum(matched & (X_k == 0))
                                   / max(int(np.sum(X_k == 0)), 1)),
        "retention_by_center": {int(c): float(np.mean(matched[env_k == c]))
                                for c in np.unique(env_k)},
        "cross_center_match_rate": (float(np.sum(cross) / m_k) if m_k else None),
        "distance_quantiles": (None if not m_k else
                               {q: float(np.quantile(dist[matched], q))
                                for q in (0.5, 0.9, 0.99)}),
        "distance_max": (None if not m_k else float(np.nanmax(dist[matched]))),
        "weight_sum_treated": float(np.sum(weights[X_k == 1])),
        "weight_sum_control": float(np.sum(weights[X_k == 0])),
        "theta_alt_reuse_formula": theta_alt,
        "theta_from_c_weights": theta_c,
        "c_identity_gap": identity_gap,
        "weight_identity_I_plus_K_holds": weight_identity_holds,
        "n_reused_but_unmatched": k_positive_without_match,
        "sum_c": (float(np.sum(c_vec)) if m_k else None),
        "sum_c_treated": (float(np.sum(c_vec[X_k == 1])) if m_k else None),
        "sum_c_control": (float(np.sum(c_vec[X_k == 0])) if m_k else None),
        "sum_c_squared": (float(np.sum(c_vec ** 2)) if m_k else None),
        "max_abs_c": (float(np.max(np.abs(c_vec))) if m_k else None),
        "max_reuse_K": int(K.max()) if K.size else 0,
        "smd_before": before, "smd_after": after,
        "smd_max_before": (float(np.nanmax(np.abs(before))) if before else None),
        "smd_max_after": (float(np.nanmax(np.abs(after))) if after else None),
        "unmatched_are_legal_caliper_losses": True,
    })
    detail["c_weight"] = [float(v) for v in c_vec]
    return theta_k, m_k, matched, detail, diag, c_vec


def model_variance(c_vec, mu_hat) -> float:
    """V = sum_i c_i^2 mu_hat_i (1 - mu_hat_i)."""
    c = np.asarray(c_vec, dtype=float)
    mu = np.asarray(mu_hat, dtype=float)
    if c.shape != mu.shape:
        raise ValueError(f"c and mu_hat differ in shape: {c.shape} vs {mu.shape}")
    return float(np.sum(c ** 2 * mu * (1.0 - mu)))


def fit_partition_psm(F, X, Y, env, center_to_cluster, selected_by_cluster,
                      *, backend="statsmodels_glm",
                      outcome_fits: Mapping | None = None) -> dict[str, Any]:
    """Caliper PSM in both directions on a fixed partition and fixed selections, with the conditional outcome-model SE.

    `outcome_fits` are the per-cluster outcome fits of `regression_adjustment.fit_partition_ra` with the same
    selection, partition and row order; without them the point estimate is still given and the SE is unavailable.
    """
    base = {"estimator": "psm_caliper", "se": None, "ci95": None, "V": None,
            "se_status": "unavailable", "se_method": SE_METHOD_PSM,
            "variance_target": VARIANCE_TARGET_PSM,
            "conditioning": "working_design"}
    if center_to_cluster is None:
        return {**base, "status": "abstain", "estimand": ESTIMAND_FULL,
                "cluster_order": [], "n_original": [], "n_target": [], "theta": [],
                "weights": None, "estimate": None, "point_status": "unavailable",
                "reason": "upstream_abstain", "errors_by_cluster": {},
                "diagnostics": {}, "matching_detail": {}}

    F = np.asarray(F, dtype=float)
    X = np.asarray(X).astype(int)
    Y = np.asarray(Y).astype(int)
    env = np.asarray(env)
    n_total, p = F.shape
    if not isinstance(selected_by_cluster, Mapping):
        raise ValueError("selected_by_cluster must map clusters to selections")

    members = cluster_members(env, center_to_cluster)
    cluster_order = sorted(members)
    selected = {k: _check_selected(selected_by_cluster[k], p) for k in cluster_order}

    n_original: list[int] = []
    n_target: list[int] = []
    theta: list[float] = []
    errors: dict[int, str] = {}
    diagnostics: dict[Any, Any] = {}
    matching_detail: dict[Any, Any] = {}
    matched_row_ids: list[int] = []
    c_by_cluster: dict[int, np.ndarray] = {}
    rows_by_cluster: dict[int, np.ndarray] = {}

    for k in cluster_order:
        rows = np.flatnonzero(np.isin(env, members[k]))
        n_k = int(rows.size)
        n_original.append(n_k)
        if n_k == 0:
            errors[k] = "cluster has no observations"
            n_target.append(0)
            theta.append(float("nan"))
            continue
        try:
            theta_k, m_k, matched, detail, diag, c_vec = _psm_cluster(
                F[rows], X[rows], Y[rows], env[rows], rows, selected[k],
                backend=backend)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            errors[k] = f"{type(error).__name__}: {error}"
            n_target.append(0)
            theta.append(float("nan"))
            diagnostics[k] = {"status": "unavailable", "reason": errors[k],
                              "n_k": n_k, "selected": selected[k],
                              "failure_kind": "ps_fit_or_input_failure"}
            continue
        diag.update({"status": "complete", "selected": selected[k], "theta_k": theta_k})
        diagnostics[k] = diag
        matching_detail[k] = detail
        n_target.append(m_k)
        theta.append(theta_k)
        c_by_cluster[k] = c_vec
        rows_by_cluster[k] = rows
        matched_row_ids.extend(int(v) for v in rows[matched])

    point_ok = not errors and sum(n_target) > 0
    all_matched = point_ok and all(m == n for m, n in zip(n_target, n_original))
    estimand = ESTIMAND_FULL if all_matched else ESTIMAND_RETAINED
    result: dict[str, Any] = {
        **base,
        "status": "complete" if point_ok else "incomplete",
        "estimand": estimand,
        "cluster_order": cluster_order,
        "n_original": n_original, "n_target": n_target,
        "n": n_original,
        "theta": theta, "weights": None, "estimate": None,
        "point_status": "complete" if point_ok else "unavailable",
        "errors_by_cluster": errors, "diagnostics": diagnostics,
        "matching_detail": matching_detail,
        "matched_row_ids": sorted(matched_row_ids),
        "full_ate_comparable": bool(all_matched),
        "retention_overall": (float(sum(n_target) / max(sum(n_original), 1))
                              if n_original else None),
        "weight_basis": ("n_k/N (original sizes)" if all_matched
                         else "m_k/sum(m_k) (retained by the caliper)"),
    }
    if not point_ok:
        result["reason"] = ("all_clusters_have_zero_matches" if not errors
                            else "cluster_ps_fit_unavailable")
        return result

    base_w = np.asarray(n_original if all_matched else n_target, dtype=float)
    if base_w.sum() <= 0:
        result["point_status"] = "unavailable"
        result["reason"] = "zero_total_weight"
        return result
    a = base_w / base_w.sum()
    theta_arr = np.asarray(theta, dtype=float)
    usable = np.isfinite(theta_arr) | (a == 0.0)
    if not np.all(usable):
        result["point_status"] = "unavailable"
        result["reason"] = "nonfinite_cluster_effect_with_positive_weight"
        return result
    result["weights"] = a.tolist()
    result["estimate"] = float(np.nansum(np.where(a > 0, a * theta_arr, 0.0)))

    # ---- conditional outcome-model variance ----
    se_reasons: dict[int, str] = {}
    V_diag = np.full(len(cluster_order), np.nan, dtype=float)
    for pos, k in enumerate(cluster_order):
        if a[pos] == 0.0:
            V_diag[pos] = 0.0                       # cluster without matches: weight 0
            se_reasons[k] = "zero_weight_no_matched_targets"
            continue
        fit = None if outcome_fits is None else outcome_fits.get(k)
        if fit is None:
            se_reasons[k] = SE_STATUS_NO_MODEL
            continue
        if fit.get("fit_status") != "complete":
            se_reasons[k] = "outcome_model_fit_failed"
            continue
        if list(fit.get("selected", [])) != list(selected[k]):
            se_reasons[k] = "outcome_model_selection_mismatch"
            continue
        if not np.array_equal(np.asarray(fit.get("rows")), rows_by_cluster[k]):
            se_reasons[k] = "outcome_model_row_alignment_mismatch"
            continue
        mu = np.asarray(fit["mu_hat"], dtype=float)
        if mu.shape != c_by_cluster[k].shape or not np.all(np.isfinite(mu)):
            se_reasons[k] = "invalid_prediction"
            continue
        diag_k = diagnostics.get(k, {})
        if not diag_k.get("weight_identity_I_plus_K_holds", True):
            se_reasons[k] = "weight_identity_violated"
            continue
        gap = diag_k.get("c_identity_gap")
        if gap is not None and np.isfinite(gap) and gap > 1e-9 * max(1.0, abs(theta_arr[pos])):
            se_reasons[k] = "c_weight_point_identity_violated"
            continue
        try:
            V_k = model_variance(c_by_cluster[k], mu)
        except ValueError as error:
            se_reasons[k] = f"nonfinite_variance: {error}"
            continue
        if not np.isfinite(V_k):
            se_reasons[k] = "nonfinite_variance"
            continue
        V_diag[pos] = V_k
        contrib = c_by_cluster[k] ** 2 * mu * (1.0 - mu)
        total = float(contrib.sum())
        diagnostics[k].update({
            "V_model": V_k,
            "se_model": float(np.sqrt(V_k)),
            "mu_hat_min": float(mu.min()), "mu_hat_max": float(mu.max()),
            "max_variance_share": (float(contrib.max() / total) if total > 0 else None),
            "se_method": SE_METHOD_PSM,
            "zero_model_variance": bool(V_k == 0.0),
        })
        if V_k == 0.0:
            se_reasons[k] = "zero_model_variance"

    result["V"] = V_diag.tolist()
    result["se_reasons_by_cluster"] = se_reasons
    positive = a > 0.0
    if np.all(np.isfinite(V_diag[positive])):
        var_total = float(np.sum((a[positive] ** 2) * V_diag[positive]))
        se = float(np.sqrt(var_total)) if var_total >= 0 else float("nan")
        result["se"] = se
        if np.isfinite(se) and se > 0:
            est = result["estimate"]
            result["ci95"] = [est - Z_975 * se, est + Z_975 * se]
            result["se_status"] = SE_STATUS_PSM
        else:
            result["ci95"] = None
            result["se_status"] = "unavailable"
            result["se_reason"] = "zero_model_variance"
    else:
        result["se_status"] = "unavailable"
        result["se_reason"] = "positive_weight_cluster_without_model_variance"
    return result


__all__ = ["CALIPER_K", "SE_METHOD_PSM", "SE_STATUS_PSM", "SE_STATUS_NO_MODEL",
           "VARIANCE_TARGET_PSM", "Z_975", "ESTIMAND_FULL", "ESTIMAND_RETAINED",
           "nearest_within_caliper", "model_variance", "fit_partition_psm"]
