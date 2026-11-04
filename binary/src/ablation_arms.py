"""ablation_arms.py — the two HetICS ablation arms and the reference dual_icp.

| arm | screening groups | candidates | ICP | estimation |
|---|---|---|---|---|
| dual_icp (reference) | frozen learned partition | S_X ∪ S_Y | within-cluster ICP across centers | stored final sets re-estimated on the regenerated data |
| dual_no_icp | same | stored candidates before the ICP, unchanged | skipped | as the reference |
| dual_global_icp | all centers in one screening group | fixed-X/BIC dual candidates on all observations | the same ICP with the hospitals as environments | frozen learned partition; the global set is used in every analysis cluster |

Candidates and ICP both go through screening_comparison.screen_family(..., "fixedx_bic"), i.e. vendor
step6_methods.knockoff_family_cluster (within-center centering and scaling, deterministic fixed-X knockoffs,
LassoLarsIC/BIC, knockoff+ at q = 0.10, partial Chow test with BH at m6.ALPHA_ICP and the fallback rule when
every candidate is rejected). Estimation calls run_formal.estimate_selections.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

import evaluation as ev
import screening_comparison as sc
from aipw_sandwich import fit_partition_aipw

ABLATION_ARMS = ("dual_icp", "dual_no_icp", "dual_global_icp")
ICP_ALPHA = 0.001          # icp_alpha of formal_v5.json / formal_r2.json; run_ablation sets it in every worker


def _by_cluster(text: str | None) -> dict[int, list[int] | None] | None:
    obj = json.loads(text) if text not in (None, "") else None
    if obj is None:
        return None
    return {int(k): (None if v is None else sorted(int(j) for j in v)) for k, v in obj.items()}


def cached_selection(name: str, cache_row: dict, members_rows: dict[int, np.ndarray]) -> dict:
    """dual_icp (stored final sets) and dual_no_icp (stored candidates before the ICP), taken from the unit cache."""
    assert name in ("dual_icp", "dual_no_icp"), name
    by = _by_cluster(cache_row["dual_sel"] if name == "dual_icp" else cache_row["dual_pre"])
    block: dict[str, Any] = {
        "source": "cached_original_unit", "original_dual_status": cache_row["dual_status"],
        "candidate_definition": "fixed_x_bic_S_X_union_S_Y", "screening_scope": "learned_cluster",
        "estimation_scope": "learned_cluster",
    }
    if name == "dual_no_icp":
        block.update(icp_invoked=False, icp_skipped_for_ablation=True, icp_pvalues="not_applicable",
                     note="ICP skipped for the ablation; not 'all candidates retained by the test'")
    else:
        block.update(icp="official_once_cached_final")
    if by is None:
        return {**block, "status": "unavailable", "reason": "cached_selection_missing",
                "selected_by_cluster": None}
    if set(by) != set(members_rows):
        return {**block, "status": "unavailable",
                "reason": f"cached_clusters_{sorted(by)}_vs_partition_{sorted(members_rows)}",
                "selected_by_cluster": None}
    missing = sorted(g for g, v in by.items() if v is None)
    if missing:                       # the original candidates failed in a cluster: arm not estimable
        return {**block, "status": "unavailable", "reason": f"original_candidate_missing_clusters_{missing}",
                "selected_by_cluster": None}
    return {**block, "status": "complete", "selected_by_cluster": {g: by[g] for g in sorted(by)}}


def icp_diagnostics(call: dict | None, pre: list[int], post: list[int], m6) -> dict:
    """Per-covariate p-values of one ICP call recorded by screen_family, at the threshold m6.ALPHA_ICP."""
    alpha = float(m6.ALPHA_ICP)
    out: dict[str, Any] = {"alpha_used": alpha, "n_candidates": len(pre), "candidates": list(pre),
                           "selected": list(post), "icp_invoked": bool(pre)}
    if not pre:
        out["branch"] = "empty_candidate_set_icp_not_invoked"
        return out
    if call is None:
        # screen_family records the calls per direction only when all three ICP calls ran; the ICP
        # itself ran and its selection is used, but the per-covariate p-values are not available
        out["branch"] = "icp_ran_pvalue_record_unavailable"
        return out
    raw = call.get("pvalues")
    if not raw:
        out["branch"] = "no_pvalues_return_candidates"
        return out
    screened = list(call["pre"])
    local = sorted(int(j) for j in raw)
    pv = np.array([raw[j] if j in raw else raw[str(j)] for j in local], dtype=float)
    with np.errstate(invalid="ignore"):
        bh = m6.bh_adjust(pv)
    kept = [screened[j] for i, j in enumerate(local) if not bool(bh[i] < alpha)]
    out.update(raw_pvalues={int(screened[j]): float(pv[i]) for i, j in enumerate(local)},
               bh_pvalues={int(screened[j]): float(bh[i]) for i, j in enumerate(local)},
               n_tested=len(local), n_untestable=len(screened) - len(local),
               n_rejected=len(local) - len(kept),
               nonfinite_pvalues=[int(screened[local[i]]) for i in np.flatnonzero(~np.isfinite(pv))])
    if not kept and post:
        out["branch"] = "all_rejected_stability_fallback"      # keep the max(1, |C|//2) largest raw p-values
    else:
        out["branch"] = "bh_filter_applied"
        out["kept_matches_rule"] = sorted(kept) == sorted(post)
    return out


def global_selection(F, X, Y, env, members_rows: dict[int, np.ndarray], *, m6, task) -> dict:
    """All centers form one screening group (hospitals as ICP environments); the set is used in every learned cluster."""
    centers = sorted(int(c) for c in np.unique(env))
    n_env = len(centers)
    block: dict[str, Any] = {"screening_scope": "global", "estimation_scope": "learned_cluster",
                             "icp_environments": "original_hospital_ids", "n_icp_environments": n_env,
                             "candidate_definition": "fixed_x_bic_S_X_union_S_Y_global"}
    if not members_rows:
        return {**block, "status": "unavailable", "reason": "upstream_abstain", "selected_by_cluster": None}
    if n_env < 2:
        raise RuntimeError(f"global ICP has {n_env} < 2 environments; hospital codes are missing")
    t0 = time.perf_counter()
    fam = sc.screen_family(F, X, Y, env, {c: 0 for c in centers}, "fixedx_bic",
                           m6=m6, task=task, config={})
    block["elapsed"] = time.perf_counter() - t0
    d = fam["directions"]["dual_icp"]
    detail = fam["detail_by_cluster"].get(0) or {}
    if d["status"] != "complete" or 0 not in d["selected_by_cluster"]:
        return {**block, "status": "failed",
                "reason": (d.get("errors_by_cluster") or {}).get(0, "global_candidate_failed"),
                "selected_by_cluster": None}
    if int(detail.get("n_centers_k", -1)) != n_env:
        raise RuntimeError(f"global candidates record {detail.get('n_centers_k')} centers, expected {n_env}")
    pre = list(d["pre_icp_by_cluster"][0] or [])
    sel = list(d["selected_by_cluster"][0])
    call = (detail.get("icp_calls") or {}).get("dual")
    block.update(status="complete", global_pre_icp=pre, global_selected=sel,
                 global_pre_icp_x=list(detail["pre_icp"]["x"]), global_pre_icp_y=list(detail["pre_icp"]["y"]),
                 icp_detail=icp_diagnostics(call, pre, sel, m6),
                 pre_icp_by_cluster={g: pre for g in sorted(members_rows)},
                 selected_by_cluster={g: sel for g in sorted(members_rows)})
    return block


def _unavailable(reason: str) -> list[dict]:
    return [{"estimator": "aipw", "stage": "overall", "analysis_cluster": -1,
             "point_status": "unavailable", "se_status": "unavailable", "reason": reason,
             "ci_valid": False, "cover_true": None, "cover_zero": None, "reject_zero": None}]


def estimate_all(F, X, Y, env, truth, selections, center_map, members_rows, *, backend):
    """RA, AIPW and PSM through run_formal.estimate_selections."""
    from run_formal import estimate_selections
    effects, diag, _matching, timings, errors = estimate_selections(
        F, X, Y, env, truth, selections, center_map, None, members_rows, backend=backend)
    return effects, diag, errors, timings


def estimate_aipw(F, X, Y, env, truth, selections, center_map, members_rows, *, backend):
    """AIPW only; identical to the AIPW branch of run_formal.estimate_selections."""
    effects, diag, errors, timings = {}, {}, [], {}
    for name, block in selections.items():
        sel = block.get("selected_by_cluster")
        if sel is None or center_map is None:
            effects[name] = {"aipw": _unavailable(block.get("reason", "upstream_abstain"))}
            continue
        t0 = time.perf_counter()
        try:
            res = fit_partition_aipw(F, X, Y, env, center_map, sel, backend=backend)
        except Exception as error:                                   # noqa: BLE001
            errors.append(f"aipw[{name}]: {type(error).__name__}: {error}")
            effects[name] = {"aipw": _unavailable(f"{type(error).__name__}: {error}")}
            continue
        res["elapsed"] = time.perf_counter() - t0
        res["estimator"] = "aipw"
        timings[f"aipw::{name}"] = res["elapsed"]
        arm_full = ev.cluster_targets(truth, members_rows)
        effects[name] = {"aipw": ev.effect_rows(res, "aipw", truth, arm_full,
                                                full_targets=arm_full, target_different=False)}
        diag[name] = {"aipw": {k: res.get(k) for k in ("status", "point_status", "se_status", "estimand",
                                                       "elapsed", "errors_by_cluster",
                                                       "se_reasons_by_cluster", "V", "cluster_order",
                                                       "weights")}}
        diag[name]["aipw"]["diagnostics"] = {str(k): v for k, v in (res.get("diagnostics") or {}).items()}
    return effects, diag, errors, timings
