"""Arms of the formal units: ICP rule, alignment of learned to true clusters, and the hdPS k rule."""

from __future__ import annotations

import numpy as np
import pytest

import arms
import formal_dgp
import pooled_selection as ps
from conftest import small_row
from screening_bridge import load_screening


@pytest.fixture(scope="module")
def m6():
    return load_screening(formal_dgp.STEP6_DIR)


@pytest.fixture(scope="module")
def small_unit(small_continuous, small_group, sources):
    _gen, b6 = sources
    row = small_row("TEST_P", gamma_logor=1.0, has_u=False, methods=["dual_icp"])
    observed, truth, meta = formal_dgp.binary_unit(small_continuous, row,
                                                   small_group, b6, u=None)
    return observed, truth, meta


def test_icp_filter_bh_rule_and_fallback(small_unit, m6):
    from chow_test import bh_adjust, chow_pvalues
    observed, _truth, _meta = small_unit
    env = np.asarray(observed.center_codes)
    centers = [int(c) for c in np.unique(env)][:3]
    mask = np.isin(env, centers)
    F = np.asarray(observed.features, float)[mask]
    Y = np.asarray(observed.outcome).astype(float)[mask]
    E = env[mask]
    assert m6.icp_filter_bh(F, Y, E, []) == ([], {})

    cand = [0, 1, 2, 3, 4, 40, 41]
    selected, raw = m6.icp_filter_bh(F, Y, E, cand)
    pv = np.array([raw[j] for j in range(len(cand))])
    assert raw == chow_pvalues(F[:, cand], Y, E)
    kept = [cand[i] for i, b in enumerate(bh_adjust(pv)) if b >= m6.ALPHA_ICP]
    if kept:
        assert selected == sorted(kept)
    else:                                   # every candidate rejected: half with the largest raw p-values
        top = sorted(range(len(cand)), key=lambda i: -pv[i])[:max(1, len(cand) // 2)]
        assert selected == sorted(cand[i] for i in top)

    old = m6.ALPHA_ICP
    try:
        m6.ALPHA_ICP = 1.0 + 1e-9           # every BH p-value is below the threshold
        selected, raw = m6.icp_filter_bh(F, Y, E, cand)
        top = sorted(range(len(cand)), key=lambda i: -raw[i])[:len(cand) // 2]
        assert selected == sorted(cand[i] for i in top)
    finally:
        m6.ALPHA_ICP = old


def test_alignment_perfect_and_tie_break():
    # perfect correspondence: analysis clusters 0/1/2 fall in true clusters 2/0/1
    tcs = np.array([2] * 10 + [0] * 10 + [1] * 10)
    members = {0: np.arange(0, 10), 1: np.arange(10, 20), 2: np.arange(20, 30)}
    out = arms.align_clusters_to_truth(members, tcs, 3)
    assert out["mapping"] == {0: 2, 1: 0, 2: 1}
    assert out["total_overlap"] == 30
    assert out["tie_broken"] is False
    assert all(v == 1.0 for v in out["purity"].values())
    assert all(v is False for v in out["is_mixed_cluster"].values())

    # global tie: equal overlap everywhere -> lexicographically smallest mapping (0, 1, 2)
    tcs = np.array([0, 1, 2] * 9)
    members = {0: np.arange(0, 9), 1: np.arange(9, 18), 2: np.arange(18, 27)}
    out = arms.align_clusters_to_truth(members, tcs, 3)
    assert out["tie_broken"] is True
    assert out["n_optimal_solutions"] == 6
    assert out["mapping"] == {0: 0, 1: 1, 2: 2}


def test_alignment_is_globally_optimal_not_row_greedy():
    """Row-wise greedy matching would map both analysis clusters to true cluster 0."""
    tcs = np.array([0] * 20 + [1] * 20)
    members = {0: np.arange(0, 12), 1: np.concatenate([np.arange(12, 20),
                                                       np.arange(20, 40)])}
    out = arms.align_clusters_to_truth(members, tcs, 3)
    assert len(set(out["mapping"].values())) == 2      # one to one
    assert out["mapping"][0] == 0 and out["mapping"][1] == 1
    assert out["is_mixed_cluster"][1] is True
    assert out["purity"][1] == pytest.approx(20 / 28)


def test_hdps_k_rule_and_upstream_failure(small_unit, sources):
    _gen, b6 = sources
    observed, _truth, _meta = small_unit
    F = np.asarray(observed.features, float)
    X = np.asarray(observed.treatment).astype(int)
    Y = np.asarray(observed.outcome).astype(int)

    pooled = ps.pooled_lasso(F, Y, b6)
    assert pooled["status"] == "complete"
    hdps = ps.hdps(F, X, Y, pooled["selected"])
    expected_k = min(F.shape[1], max(len(pooled["selected"]), 10))
    assert hdps["k_top"] == expected_k
    assert hdps["k_rule"] == "min(p, max(|pooled_selected|, 10))"
    assert len(hdps["selected"]) == expected_k
    assert hdps["definition"].endswith("topk_proxy")

    class _Broken:
        @staticmethod
        def pooled_logistic_lasso(*_a, **_k):
            raise RuntimeError("upstream failure")
    broken = ps.pooled_lasso(F, Y, _Broken())
    assert broken["status"] == "failed"
    failed = ps.hdps(F, X, Y, broken["selected"])
    assert failed["status"] == "failed" and failed["reason"] == "upstream_dependency_failed"
    assert failed["selected"] is None      # a failed upstream fit is not an empty selection


def test_assumption_list_is_carried():
    assert "fixed_x_knockoff_uses_linear_bic_score_on_binary_response" in \
        arms.ASSUMPTIONS_NOT_COVERED
    assert "official_icp_partial_chow_not_exact_under_binary_outcome" in \
        arms.ASSUMPTIONS_NOT_COVERED
