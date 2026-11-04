"""Selection metrics of the continuous combination (synthetic records only).

Recall is the minimum over non-empty learned clusters of the recall of S_k*; precision is computed on the union
of the cluster-specific sets against S_safe and is missing for an empty union; all-center arms compute the
recall against every S_k* and take the minimum.
"""
import math

import pytest

from run_ra_cont import truth_sets, selection_row

T = truth_sets(200)            # Z 0-7, shared 8-17, spec 18-19 / 20-21 / 22-23, spur 24-43, noise 44-199


def _cl(k, k_true, sel, M_k=4):
    S = set(sel)
    core = T["core"].get(k_true, set())
    return dict(k=k, k_true=k_true, M_k=M_k, selected=sorted(S),
                recall_core=(len(S & core) / len(core) if core else float("nan")),
                n_spur=len(S & T["spur"]), n_noise=len(S & T["noise"]))


def test_layout_matches_proportional_blocks():
    assert sorted(T["core"][0]) == list(range(0, 18)) + [18, 19]
    assert sorted(T["safe"]) == list(range(24))
    assert (min(T["spur"]), max(T["spur"]), len(T["noise"])) == (24, 43, 156)


def test_cluster_arm_min_recall_union_precision_and_rates():
    base = list(range(18))
    arm = dict(scope="learned_cluster", clusters=[
        _cl(0, 0, base + [18, 19]),                 # all target covariates
        _cl(1, 2, base + [22, 24, 50]),             # misses one own spec, adds one spurious and one noise covariate
        _cl(2, 1, base + [20, 21, 22])])            # adds one other-cluster spec (in S_safe)
    r = selection_row(arm, T)
    assert r["recall_min"] == pytest.approx(19 / 20)
    assert r["all_core"] == 0.0
    union = set(base) | {18, 19, 20, 21, 22, 24, 50}
    assert r["precision"] == pytest.approx(len(union & T["safe"]) / len(union))
    assert r["spur_rate"] == pytest.approx(1 / 20) and r["noise_rate"] == pytest.approx(1 / 156)
    assert r["n_union"] == len(union) and r["empty_selection"] == 0.0


def test_empty_learned_cluster_is_ignored():
    arm = dict(scope="learned_cluster", clusters=[
        _cl(0, 0, list(range(20))), _cl(1, 1, list(range(18)) + [20, 21]), _cl(2, 2, [], M_k=0)])
    assert selection_row(arm, T)["recall_min"] == 1.0


def test_unmapped_nonempty_cluster_makes_recall_missing():
    arm = dict(scope="learned_cluster", clusters=[_cl(0, 0, list(range(20))), _cl(1, -1, [0, 1])])
    r = selection_row(arm, T)
    assert math.isnan(r["recall_min"]) and math.isnan(r["all_core"]) and r["n_clusters_unmapped"] == 1


def test_empty_union_gives_missing_precision_not_zero():
    arm = dict(scope="learned_cluster", clusters=[_cl(0, 0, []), _cl(1, 1, []), _cl(2, 2, [])])
    r = selection_row(arm, T)
    assert r["recall_min"] == 0.0 and math.isnan(r["precision"]) and r["empty_selection"] == 1.0


def test_pooled_arm_recall_is_min_over_true_clusters_not_safe_set():
    S = set(range(18)) | {18, 19, 20, 21}                      # lacks the two spec of cluster 2
    arm = dict(scope="all_centers_pooled", selected=sorted(S),
               recall_core=len(S & T["safe"]) / len(T["safe"]),  # stored against S_safe
               n_spur=0, n_noise=0)
    r = selection_row(arm, T)
    assert r["recall_min"] == pytest.approx(18 / 20)          # minimum over S_k*, not 22/24
    assert r["precision"] == 1.0


def test_oracle_arms_without_selected_are_skipped():
    oracle = dict(scope="learned_cluster", clusters=[{k: v for k, v in _cl(0, 0, list(range(20))).items()
                                                      if k != "selected"}])
    assert selection_row(oracle, T) is None
    assert selection_row(dict(scope="all_centers_pooled", recall_core=1.0, n_spur=0, n_noise=0), T) is None


def test_summary_is_two_level_and_skips_missing_precision():
    import pandas as pd
    from run_ra_cont import summarize_selection
    rows = []
    for p, recs in ((0, [1.0, 1.0, 0.5]), (1, [1.0])):              # parameter seed 0 three times, seed 1 once
        for d, rv in enumerate(recs):
            rows.append(dict(experiment="grid", combo_id="A05", method="dual_icp", scope="learned_cluster",
                             param_index=p, data_index=d, true_gamma=1.5, p=500, recall_min=rv,
                             all_core=float(rv == 1.0), precision=(float("nan") if (p, d) == (0, 2) else 0.8),
                             spur_rate=0.0, noise_rate=0.01, n_union=70.0, empty_selection=0.0))
    s = summarize_selection(pd.DataFrame(rows)).iloc[0]
    assert s["recall_min"] == pytest.approx(((1 + 1 + .5) / 3 + 1.0) / 2)   # within seeds, then equally across seeds
    assert s["precision"] == pytest.approx(0.8)
    assert (s["n_reps"], s["n_recall_defined"], s["n_precision_defined"], s["n_param_seeds"]) == (4, 4, 3, 2)


def test_mismatch_with_stored_counts_fails_loudly():
    bad = _cl(0, 0, list(range(20)))
    bad["n_spur"] = 3                                          # inconsistent with the truth sets: must raise
    with pytest.raises(AssertionError):
        selection_row(dict(scope="learned_cluster", clusters=[bad]), T)
