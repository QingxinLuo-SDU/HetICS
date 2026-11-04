"""Pooled comparators on the LABELED_PARAMS fixture (9 centers x 800, p = 100, K = 3).

The selection cache is computed on the fixture with the same functions and passed to the runner. Checked:
cache first, pooled estimation = direct AIPW on all centers as one group, the gate (dual replay and recomputed
pooled lasso = cache), and the missing-cache and abstention branches.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import ablation_arms as aa
import formal_dgp
import pooled_arms as pa
import pooled_selection as ps
import run_pooled as rp
import screening_comparison as sc
from aipw_sandwich import fit_partition_aipw
from conftest import LABELED_PARAMS, small_row
from run_formal import estimate_selections, unit_path
from screening_bridge import load_screening


@pytest.fixture(scope="module")
def m6():
    mod = load_screening(formal_dgp.STEP6_DIR)
    original_alpha = mod.ALPHA_ICP
    mod.ALPHA_ICP = aa.ICP_ALPHA
    try:
        yield mod
    finally:
        mod.ALPHA_ICP = original_alpha


@pytest.fixture(scope="module")
def unit(labeled_continuous, small_group, sources, m6):
    _gen, b6 = sources
    row = small_row("TEST_P", gamma_logor=1.0, has_u=False, methods=["dual_icp"])
    row.update({k: str(v) for k, v in LABELED_PARAMS.items() if k not in ("centers_per_cluster", "gamma_XY")})
    row.update(SIG_Y="1", K="3", design_revision="pooled_test")
    observed, truth, _meta = formal_dgp.binary_unit(labeled_continuous, row, small_group, b6)
    F = np.asarray(observed.features, dtype=float)
    X = np.asarray(observed.treatment).astype(int)
    Y = np.asarray(observed.outcome).astype(int)
    env = np.asarray(observed.center_codes)
    centers = sorted(int(c) for c in np.unique(env))
    cmap = {c: int(k) for c, k in zip(centers, np.asarray(labeled_continuous["true_cluster"]))}
    members_rows = {g: np.flatnonzero(np.isin(env, [c for c in centers if cmap[c] == g]))
                    for g in sorted(set(cmap.values()))}
    fam = sc.screen_family(F, X, Y, env, cmap, "fixedx_bic", m6=m6, task=None, config={})
    d = fam["directions"]["dual_icp"]
    orig_sel = {"dual_icp": {"selected_by_cluster": {int(k): v for k, v in d["selected_by_cluster"].items()}}}
    effects, *_ = estimate_selections(F, X, Y, env, truth, orig_sel, cmap, None, members_rows,
                                      backend="statsmodels_glm")
    abl_row = {"unit_key": "TEST_P__p001__d01", "dual_status": d["status"],
               "dual_pre": json.dumps({str(k): v for k, v in d["pre_icp_by_cluster"].items()}),
               "dual_sel": json.dumps({str(k): v for k, v in d["selected_by_cluster"].items()}),
               "orig_aipw": json.dumps({"dual_icp": effects["dual_icp"]["aipw"]}), "center_map_available": "True"}
    lasso = ps.pooled_lasso(F, Y, b6)["selected"]
    pds = ps.pooled_pds(F, X, Y, b6, s_y=lasso)["selected"]
    oal = ps.pooled_oal(F, X, Y)["selected"]
    hd = ps.hdps(F, X, Y, lasso)["selected"]
    sel_row = {"unit_key": "TEST_P__p001__d01", "center_map_available": "True",
               "pooled_lasso": json.dumps(sorted(lasso)), "pooled_pds": json.dumps(sorted(pds)),
               "pooled_oal": json.dumps(sorted(oal)), "hdps": json.dumps(sorted(hd))}
    lab = {"unit_key": "TEST_P__p001__d01", "ate_do": repr(float(truth["ate_do"])), "n_rows": str(env.size),
           "center_map_available": "True", "labels_frozen": "True", "source_machine": "test",
           "center_map": json.dumps({str(c): g for c, g in cmap.items()}),
           "center_order": json.dumps(centers), "frozen_labels": json.dumps([cmap[c] for c in centers])}
    return dict(F=F, X=X, Y=Y, env=env, truth=truth, abl_row=abl_row, sel_row=sel_row, lab=lab, row=row,
                group=small_group, b6=b6, continuous=labeled_continuous, lasso=sorted(lasso))


def _overall(rec, arm):
    return [r for r in rec["effects"][arm]["aipw"] if r["stage"] == "overall"][0]


def test_unit_uses_cache_and_pools_estimation(unit, m6):
    rec = rp.pooled_unit(unit["continuous"], unit["row"], unit["group"], unit["lab"], unit["sel_row"],
                         unit["abl_row"], unit["b6"], m6, None, None, backend="statsmodels_glm",
                         gate=False, allow_recompute=False)
    assert set(rec["effects"]) == set(pa.POOLED_ARMS)
    for name, col in pa.CACHE_COLUMN.items():
        blk = rec["selections"][name]
        assert blk["status"] == "complete" and blk["estimation_scope"] == "all_centers_pooled"
        assert blk["source"] == "cache" and blk["selected"] == json.loads(unit["sel_row"][col])
    tau_mean = float(np.mean(unit["truth"]["tau_do_i"]))
    for arm in pa.POOLED_ARMS:
        o = _overall(rec, arm)
        assert o["point_status"] == "complete" and o["truth_target"] == pytest.approx(tau_mean), arm
        assert set(rec["effects"][arm]) == {"ra_standardized", "aipw", "psm_caliper"}, arm
        for est in ("ra_standardized", "psm_caliper"):
            r = [x for x in rec["effects"][arm][est] if x["stage"] == "overall"][0]
            assert r["point_status"] == "complete", (arm, est)
        assert {str(k) for k in rec["feature_selection"][arm]} == {"0", "1", "2"}      # per true cluster
    # pooled estimation = direct AIPW with all centers in one group
    one = {int(c): 0 for c in np.unique(unit["env"])}
    direct = fit_partition_aipw(unit["F"], unit["X"], unit["Y"], unit["env"], one,
                                {0: rec["selections"]["pooled_pds_fp"]["selected"]}, backend="statsmodels_glm")
    assert _overall(rec, "pooled_pds_fp")["estimate"] == direct["estimate"]
    safe = sorted({int(j) for v in unit["truth"]["observed_core_by_cluster"].values() for j in v})
    assert rec["selections"]["oracle_safe_pooled"]["selected"] == safe


def test_gate_passes_on_same_machine(unit, m6):
    rec = rp.pooled_unit(unit["continuous"], unit["row"], unit["group"], unit["lab"], unit["sel_row"],
                         unit["abl_row"], unit["b6"], m6, None, None, backend="statsmodels_glm",
                         gate=True, allow_recompute=False)
    res = rp.gate_check(rec, unit["abl_row"], unit["sel_row"])
    assert res["pass"], res["problems"]
    assert res["max_abs_diff"] == 0.0 and res["pooled_arms_without_estimate"] == []
    # a modified cache selection must fail the gate
    bad = dict(unit["sel_row"], pooled_lasso=json.dumps(unit["lasso"][:-1]))
    assert not rp.gate_check(rec, unit["abl_row"], bad)["pass"]


def test_missing_cache_and_abstain_branches(unit, m6):
    empty = dict(unit["sel_row"], pooled_pds="", pooled_oal="", hdps="")
    rec = rp.pooled_unit(unit["continuous"], unit["row"], unit["group"], unit["lab"], empty, unit["abl_row"],
                         unit["b6"], m6, None, None, backend="statsmodels_glm", gate=False, allow_recompute=False)
    for arm in ("pooled_pds_fp", "pooled_oal_fp", "hdps_fp"):
        assert rec["selections"][arm]["reason"] == "selection_cache_missing"
        assert _overall(rec, arm)["point_status"] == "unavailable"
    assert _overall(rec, "pooled_lasso_fp")["point_status"] == "complete"
    # abstaining labels: the pooled comparators do not need the partition and are recomputed
    lab = dict(unit["lab"], center_map_available="False")
    rec = rp.pooled_unit(unit["continuous"], unit["row"], unit["group"], lab, empty, unit["abl_row"],
                         unit["b6"], m6, None, None, backend="statsmodels_glm", gate=False, allow_recompute=False)
    assert rec["label_abstain"] is True
    assert rec["selections"]["pooled_pds_fp"]["source"] == "recomputed"
    assert rec["selections"]["pooled_pds_fp"]["selected"] == json.loads(unit["sel_row"]["pooled_pds"])
    assert rec["selections"]["hdps_fp"]["selected"] == json.loads(unit["sel_row"]["hdps"])
    assert all(_overall(rec, a)["point_status"] == "complete" for a in pa.POOLED_ARMS)


def test_resume_and_version(tmp_path, unit):
    path = unit_path(str(tmp_path), unit["row"], 1, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump({"unit_complete": True, "output_version": "binary-ablation-v5-20261027"}, h)
    assert not rp.pooled_unit_complete(path)
    with open(path, "w", encoding="utf-8") as h:
        json.dump({"unit_complete": True, "output_version": rp.POOLED_VERSION}, h)
    assert rp.pooled_unit_complete(path)
