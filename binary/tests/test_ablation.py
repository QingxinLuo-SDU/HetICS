"""The two ablation arms on the LABELED_PARAMS fixture (9 centers x 800, p = 100, K = 3).

Each cluster has 2400 rows >= 2p, so fixed-X knockoffs run; all centers have 7200 rows. The unit cache is computed
on the fixture with the same functions (screen_family + estimate_selections) and passed to the ablation runner:
the runner leaves the results unchanged, the two arms follow their definitions, and the failure branches carry
clear statuses and denominators.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import ablation_arms as aa
import formal_dgp
import run_ablation as ra
import screening_comparison as sc
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
    row.update(SIG_Y="1", K="3", design_revision="ablation_test")
    observed, truth, _meta = formal_dgp.binary_unit(labeled_continuous, row, small_group, b6)
    F = np.asarray(observed.features, dtype=float)
    X = np.asarray(observed.treatment).astype(int)
    Y = np.asarray(observed.outcome).astype(int)
    env = np.asarray(observed.center_codes)
    centers = sorted(int(c) for c in np.unique(env))
    cmap = {c: int(k) for c, k in zip(centers, np.asarray(labeled_continuous["true_cluster"]))}
    members_rows = {g: np.flatnonzero(np.isin(env, [c for c in centers if cmap[c] == g]))
                    for g in sorted(set(cmap.values()))}
    # the reference unit: dual screening and the three estimators
    fam = sc.screen_family(F, X, Y, env, cmap, "fixedx_bic", m6=m6, task=None, config={})
    d = fam["directions"]["dual_icp"]
    orig_sel = {"dual_icp": {"selected_by_cluster": {int(k): v for k, v in d["selected_by_cluster"].items()}}}
    effects, *_ = estimate_selections(F, X, Y, env, truth, orig_sel, cmap, None, members_rows,
                                      backend="statsmodels_glm")
    cache_row = {"unit_key": "TEST_P__p001__d01", "dual_status": d["status"],
                 "dual_pre": json.dumps({str(k): v for k, v in d["pre_icp_by_cluster"].items()}),
                 "dual_sel": json.dumps({str(k): v for k, v in d["selected_by_cluster"].items()}),
                 "orig_aipw": json.dumps({"dual_icp": effects["dual_icp"]["aipw"]}),
                 "center_map_available": "True"}
    lab = {"unit_key": "TEST_P__p001__d01", "ate_do": repr(float(truth["ate_do"])), "n_rows": str(env.size),
           "center_map_available": "True", "labels_frozen": "True", "source_machine": "test",
           "center_map": json.dumps({str(c): g for c, g in cmap.items()}),
           "center_order": json.dumps(centers), "frozen_labels": json.dumps([cmap[c] for c in centers])}
    return dict(F=F, X=X, Y=Y, env=env, truth=truth, cmap=cmap, members_rows=members_rows, fam=fam,
                cache_row=cache_row, lab=lab, row=row, group=small_group, b6=b6,
                continuous=labeled_continuous)


def test_no_icp_is_exactly_the_cached_candidates(unit):
    blk = aa.cached_selection("dual_no_icp", unit["cache_row"], unit["members_rows"])
    pre = unit["fam"]["directions"]["dual_icp"]["pre_icp_by_cluster"]
    assert blk["status"] == "complete"
    assert blk["selected_by_cluster"] == {int(k): sorted(v) for k, v in pre.items()}
    assert blk["icp_invoked"] is False and blk["icp_skipped_for_ablation"] is True
    assert blk["icp_pvalues"] == "not_applicable"
    # the final dual sets are subsets of the candidates (the ICP only removes, and the fallback keeps candidates)
    fin = aa.cached_selection("dual_icp", unit["cache_row"], unit["members_rows"])["selected_by_cluster"]
    assert all(set(fin[g]) <= set(blk["selected_by_cluster"][g]) for g in fin)


def test_global_arm_one_group_all_hospitals_as_environments(unit, m6):
    blk = aa.global_selection(unit["F"], unit["X"], unit["Y"], unit["env"], unit["members_rows"], m6=m6, task=None)
    n_env = len(np.unique(unit["env"]))
    assert blk["status"] == "complete" and n_env == LABELED_PARAMS["n_centers"]
    assert blk["n_icp_environments"] == n_env != 1
    assert blk["screening_scope"] == "global" and blk["estimation_scope"] == "learned_cluster"
    sets = list(blk["selected_by_cluster"].values())
    assert sorted(blk["selected_by_cluster"]) == sorted(unit["members_rows"])     # estimation partition = learned partition
    assert all(s == sets[0] for s in sets) and sets[0] == blk["global_selected"]
    assert set(blk["global_selected"]) <= set(blk["global_pre_icp"])
    assert blk["global_pre_icp"] == sorted(set(blk["global_pre_icp_x"]) | set(blk["global_pre_icp_y"]))
    assert blk["icp_detail"]["alpha_used"] == aa.ICP_ALPHA
    # identical to a direct call with every center mapped to group 0
    direct = sc.screen_family(unit["F"], unit["X"], unit["Y"], unit["env"],
                              {int(c): 0 for c in np.unique(unit["env"])}, "fixedx_bic", m6=m6, task=None, config={})
    assert direct["directions"]["dual_icp"]["selected_by_cluster"][0] == blk["global_selected"]


def test_global_arm_refuses_single_environment(unit, m6):
    env1 = np.zeros_like(unit["env"])
    with pytest.raises(RuntimeError, match="environments"):
        aa.global_selection(unit["F"], unit["X"], unit["Y"], env1, unit["members_rows"], m6=m6, task=None)


def test_three_estimators_keep_original_aipw_path_and_gate_passes(unit, m6):
    rec = ra.ablation_unit(unit["continuous"], unit["row"], unit["group"], unit["lab"], unit["cache_row"],
                           unit["b6"], m6, None, None, backend="statsmodels_glm", gate=True)
    res = ra.gate_compare(rec, unit["cache_row"])
    assert res["pass"], res["problems"]
    assert res["max_abs_diff"] == 0.0                       # same machine, same data: bitwise identical
    assert rec["estimators"] == ["ra_standardized", "aipw", "psm_caliper"] and set(rec["effects"]) == set(aa.ABLATION_ARMS)
    for arm in aa.ABLATION_ARMS:
        overall = [r for r in rec["effects"][arm]["aipw"] if r["stage"] == "overall"]
        assert overall and overall[0]["point_status"] == "complete", arm
        assert {str(k) for k in rec["feature_selection"][arm]} == {str(g) for g in unit["members_rows"]}
    assert rec["regeneration_checks"]["regeneration_equal"] is True
    assert rec["icp_alpha_used"] == aa.ICP_ALPHA


def test_status_branches(unit, m6):
    mr = unit["members_rows"]
    # candidates missing in a cluster -> arm not estimable
    bad = dict(unit["cache_row"], dual_pre=json.dumps({"0": [1, 2], "1": None, "2": [3]}))
    blk = aa.cached_selection("dual_no_icp", bad, mr)
    assert blk["status"] == "unavailable" and "missing_clusters_[1]" in blk["reason"]
    # cached clusters do not match the partition -> not estimable
    blk = aa.cached_selection("dual_no_icp", dict(unit["cache_row"], dual_pre=json.dumps({"0": [1]})), mr)
    assert blk["status"] == "unavailable" and "cached_clusters" in blk["reason"]
    # abstaining labels -> no arm is estimable and the rows give the reason
    lab = dict(unit["lab"], center_map_available="False")
    rec = ra.ablation_unit(unit["continuous"], unit["row"], unit["group"], lab, unit["cache_row"],
                           unit["b6"], m6, None, None, backend="statsmodels_glm", gate=False)
    for arm in aa.ABLATION_ARMS:
        assert rec["selections"][arm]["reason"] == "upstream_abstain"
        assert rec["effects"][arm]["aipw"][0]["point_status"] == "unavailable"
    # branches of the ICP diagnostics
    off = m6
    assert aa.icp_diagnostics(None, [], [], off)["branch"] == "empty_candidate_set_icp_not_invoked"
    assert aa.icp_diagnostics(None, [1, 2], [1, 2], off)["branch"] == "icp_ran_pvalue_record_unavailable"
    assert aa.icp_diagnostics({"pre": [5, 7], "pvalues": {}}, [5, 7], [5, 7], off)["branch"] == \
        "no_pvalues_return_candidates"
    allrej = aa.icp_diagnostics({"pre": [5, 7], "pvalues": {0: 1e-9, 1: 1e-8}}, [5, 7], [7], off)
    assert allrej["branch"] == "all_rejected_stability_fallback" and allrej["n_rejected"] == 2
    norm = aa.icp_diagnostics({"pre": [5, 7], "pvalues": {0: 1e-9, 1: 0.5}}, [5, 7], [7], off)
    assert norm["branch"] == "bh_filter_applied" and norm["kept_matches_rule"] is True


def test_resume_skips_complete_and_redoes_old_version(tmp_path, unit):
    row = unit["row"]
    path = unit_path(str(tmp_path), row, 1, 1)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump({"unit_complete": True, "output_version": "another-version"}, h)
    assert not ra.ablation_unit_complete(path)              # other versions or experiments are not complete
    with open(path, "w", encoding="utf-8") as h:
        json.dump({"unit_complete": True, "output_version": ra.ABLATION_VERSION}, h)
    assert ra.ablation_unit_complete(path)
    with open(path, "w", encoding="utf-8") as h:
        h.write("{trunc")                                    # truncated after a power loss
    assert not ra.ablation_unit_complete(path)


def test_release_recomputation_matches_original_cached_selections(unit, m6):
    rec = ra.ablation_unit(unit["continuous"], unit["row"], unit["group"], unit["lab"], {},
                          unit["b6"], m6, None, None, backend="statsmodels_glm",
                          gate=False, allow_recompute=True)
    for arm, field in (("dual_icp", "dual_sel"), ("dual_no_icp", "dual_pre")):
        assert rec["selected_by_cluster"][arm] == json.loads(unit["cache_row"][field])
        assert rec["selections"][arm]["source"] == "recomputed_original_screen_family"
    original_aipw = json.loads(unit["cache_row"]["orig_aipw"])["dual_icp"]
    for got, expected in zip(rec["effects"]["dual_icp"]["aipw"], original_aipw, strict=True):
        assert got["estimate"] == expected["estimate"]
        assert got["se"] == expected["se"]
    with pytest.raises(ValueError, match="gate validation requires"):
        ra.ablation_unit(unit["continuous"], unit["row"], unit["group"], unit["lab"], {},
                         unit["b6"], m6, None, None, backend="statsmodels_glm",
                         gate=True, allow_recompute=True)
