"""pooled_arms.py — pooled comparators for binary outcomes: selection and estimation both pool all centers.

Each selection is used once on all centers for RA, AIPW and PSM (design vector [1, center dummies, W], the
same implementation as the formal units); the target is the ATE of the whole sample, without any partition.

  pooled_lasso_fp     pooled L1-logistic selection (pooled_selection.pooled_lasso)
  pooled_pds_fp       post-double selection (pooled_selection.pooled_pds)
  pooled_oal_fp       outcome-adaptive lasso (pooled_selection.pooled_oal)
  hdps_fp             hdPS proxy (pooled_selection.hdps, k from the pooled lasso selection)
  oracle_safe_pooled  A_safe, the union of the observed core sets of the true clusters (reference: pooled
                      estimation with the best pooled selection)

Selections stored in a cache (one column per arm) are used when present; otherwise they are computed when
allow_recompute is set.
"""
from __future__ import annotations

import json
import time

import numpy as np

import evaluation as ev
import pooled_selection as ps

POOLED_ARMS = ("pooled_lasso_fp", "pooled_pds_fp", "pooled_oal_fp", "hdps_fp", "oracle_safe_pooled")
CACHE_COLUMN = {"pooled_lasso_fp": "pooled_lasso", "pooled_pds_fp": "pooled_pds",
                "pooled_oal_fp": "pooled_oal", "hdps_fp": "hdps"}


def _cached(cache_row: dict, name: str) -> list[int] | None:
    text = (cache_row or {}).get(CACHE_COLUMN[name], "")
    return None if text in ("", None) else sorted(int(j) for j in json.loads(text))


def pooled_selections(F, X, Y, env, truth, cache_row, b6, *, allow_recompute: bool) -> tuple[dict, dict]:
    """The five selections (cache first). Returns (selections, timings); a block is {status, selected, source, ...}."""
    sel: dict[str, dict] = {}
    t: dict[str, float] = {}
    for name in ("pooled_lasso_fp", "pooled_pds_fp", "pooled_oal_fp", "hdps_fp"):
        got = _cached(cache_row, name)
        if got is not None:
            sel[name] = {"status": "complete", "selected": got, "source": "cache"}
            continue
        if not allow_recompute:
            sel[name] = {"status": "unavailable", "reason": "selection_cache_missing", "source": "none"}
            continue
        t0 = time.perf_counter()
        lasso = (sel.get("pooled_lasso_fp") or {}).get("selected")
        if name == "pooled_lasso_fp":
            blk = ps.pooled_lasso(F, Y, b6)
        elif name == "pooled_pds_fp":
            blk = ps.pooled_pds(F, X, Y, b6, s_y=lasso)
        elif name == "pooled_oal_fp":
            blk = ps.pooled_oal(F, X, Y)
        else:
            blk = ps.hdps(F, X, Y, lasso)
        t[f"select::{name}"] = time.perf_counter() - t0
        if blk.get("status") != "complete":
            sel[name] = {"status": "unavailable", "reason": blk.get("reason"), "source": "recomputed"}
        else:
            sel[name] = {"status": "complete", "selected": sorted(int(j) for j in blk["selected"]),
                         "source": "recomputed"}
    safe = sorted({int(j) for v in truth["observed_core_by_cluster"].values() for j in v})
    sel["oracle_safe_pooled"] = {"status": "complete", "selected": safe, "source": "truth_A_safe"}
    for blk in sel.values():
        blk.update(screening_scope="global", estimation_scope="all_centers_pooled")
    return sel, t


def feature_eval(selections: dict, truth: dict, p: int) -> dict:
    """Each selection evaluated against the core set of every true cluster."""
    true_ids = sorted(int(k) for k in truth["observed_core_by_cluster"])
    return {name: {str(k): ev.selection_metrics(blk["selected"], truth, p, k) for k in true_ids}
            for name, blk in selections.items() if blk.get("status") == "complete"}


def _unavailable(reason: str, estimator: str = "aipw") -> list[dict]:
    return [{"estimator": estimator, "stage": "overall", "analysis_cluster": -1, "point_status": "unavailable",
             "se_status": "unavailable", "reason": reason, "ci_valid": False, "cover_true": None,
             "cover_zero": None, "reject_zero": None}]


def estimate_pooled(F, X, Y, env, truth, selections, *, backend) -> tuple[dict, dict, list, dict]:
    """RA, AIPW and PSM of each selection on all centers (one cluster made of every center).

    Calls run_formal.estimate_selections, the implementation of the formal units.
    """
    from run_formal import ESTIMATOR_ORDER, estimate_selections
    one = {int(c): 0 for c in np.unique(env)}
    members_rows = {0: np.arange(np.asarray(env).shape[0])}
    ok = {name: {"selected_by_cluster": {0: list(blk["selected"])}}
          for name, blk in selections.items() if blk.get("status") == "complete"}
    effects, diag, _matching, timings, errors = estimate_selections(
        F, X, Y, env, truth, ok, one, None, members_rows, backend=backend)
    for name, blk in selections.items():
        if name not in ok:
            effects[name] = {est: _unavailable(blk.get("reason", "selection_unavailable"), est)
                             for est in ESTIMATOR_ORDER}
    return effects, diag, errors, timings
