"""Calibration selection through sEAM, knockoff candidates and strict ICP."""
import numpy as np
import pytest

import calibration_selection_pipeline as pipe
import hypothesis_calibration_config as hcc

K = 3


def make_data(n_centers=6, n_per_center=8, p=4, K_true=K, seed=0):
    """Small synthetic data; the values only serve to count calls."""
    rng = np.random.default_rng(seed)
    env = np.repeat(np.arange(n_centers), n_per_center)
    n = env.size
    per_cluster = n_centers // K_true
    center_to_cluster = {c: c // per_cluster for c in range(n_centers)}
    return {
        "F": rng.standard_normal((n, p)),
        "Y": rng.standard_normal(n),
        "X_treat": rng.standard_normal(n),
        "env": env,
        "true_cluster": np.array([center_to_cluster[c] for c in range(n_centers)]),
        "true_cluster_sample": np.array([center_to_cluster[int(e)] for e in env]),
        "center_to_cluster": center_to_cluster,
        "A_core_by_cluster": {k: [0, 1] for k in range(K_true)},
        "A_safe": [0, 1, 2],
        "idx_Z": [0],
        "true_shared_causal": [1],
        "true_cluster_specific_causal": {0: [2], 1: [2], 2: [2]},
        "spurious_vars": [3],
        "noise_vars": [],
    }


def make_point(n_centers=6):
    return {"point_id": "M02_G0", "seam_sparsity": 3, "n_centers": n_centers,
            "centers_per_cluster": n_centers // K, "gamma_true": 0.0,
            "alpha_icp": hcc.ALPHA_ICP}


class Recorder:
    """Counts the calls of each building block and records the alpha passed to the ICP."""

    def __init__(self):
        self.calls = {}
        self.icp_alphas = []
        self.icp_cands = []

    def bump(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def count(self, name):
        return self.calls.get(name, 0)


def install_fakes(monkeypatch, rec, labels=None, y_candidates=(0, 1),
                  x_candidates=(1, 3), kept=(0, 1), icp_status="ok"):
    """Replace the building blocks by counting fakes.

    The LCD fake alternates outcome-side and exposure-side candidates: each cluster computes S_y, then S_x.
    """
    n_centers = 6
    labels = labels if labels is not None else [0, 0, 1, 1, 2, 2]

    def fake_seam(F, Y, env, K, sparsity, max_iter, tol, random_state,
                  lipschitz_method):
        rec.bump("seam")
        weights = np.full((n_centers, K), 0.1)
        for i, k in enumerate(labels):
            weights[i, k] = 0.9
        return {"center_cluster_hat": np.asarray(labels),
                "weights": weights,
                "beta_by_cluster": np.zeros((K, F.shape[1])),
                "diagnostics": {"n_iter": 7, "converged": True,
                                "log_likelihood": [-3.0, -2.0]}}

    def fake_prep(F_k, resp, env_k):
        rec.bump("prep")
        return np.asarray(F_k, dtype=float), np.asarray(resp, dtype=float)

    def fake_knockoff(Fp):
        rec.bump("knockoff")
        return np.asarray(Fp, dtype=float) * 0.5

    def fake_lcd(Fp, Ft, resp, q=0.1):
        rec.bump("lcd")
        return set(y_candidates if rec.count("lcd") % 2 == 1 else x_candidates)

    def fake_icp(F_k, Y_k, env_k, cand, n_centers_k, alpha=None):
        rec.bump("icp")
        rec.icp_alphas.append(alpha)
        rec.icp_cands.append(set(cand))
        keep = set(kept) & set(cand)
        return {"kept_strict": sorted(keep), "kept_official": list(cand),
                "detail": [{"var": int(j), "pv_raw": 0.4, "pv_bh": 0.4,
                            "kept": int(j) in keep} for j in sorted(cand)],
                "icp_fallback_fired": False, "t_icp": 0.0, "n_pre": len(cand),
                "status": icp_status}

    monkeypatch.setattr(pipe, "run_seam_clustering", fake_seam)
    monkeypatch.setattr(pipe, "prep", fake_prep)
    monkeypatch.setattr(pipe, "fixed_x_knockoff", fake_knockoff)
    monkeypatch.setattr(pipe, "lcd_select", fake_lcd)
    monkeypatch.setattr(pipe, "icp_detail", fake_icp)
    return rec


def test_seam_runs_exactly_once_per_dataset(monkeypatch):
    rec = install_fakes(monkeypatch, Recorder())
    pipe.fit_calibration_selection(make_data(), make_point(), 123)
    assert rec.count("seam") == 1


def test_knockoff_is_built_once_per_cluster(monkeypatch):
    rec = install_fakes(monkeypatch, Recorder())
    out = pipe.fit_calibration_selection(make_data(), make_point(), 123)
    n_non_empty = sum(1 for mem in out.members.values() if mem)
    assert n_non_empty == 3
    assert rec.count("knockoff") == n_non_empty
    assert rec.count("lcd") == 2 * n_non_empty         # outcome side and exposure side
    assert rec.count("prep") == 2 * n_non_empty
    assert rec.count("icp") == n_non_empty


def test_dual_candidate_set_is_the_union(monkeypatch):
    install_fakes(monkeypatch, Recorder(), y_candidates=(0, 1), x_candidates=(1, 3))
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    for k in range(K):
        assert out.channel("dual").selected_pre_icp[k] == {0, 1, 3}


def test_unknown_channel_is_rejected(monkeypatch):
    install_fakes(monkeypatch, Recorder())
    with pytest.raises(ValueError, match="unknown channels"):
        pipe.fit_calibration_selection(make_data(), make_point(), 1, channels=("xside",))


def test_empty_cluster_costs_no_knockoff(monkeypatch):
    rec = install_fakes(monkeypatch, Recorder(), labels=[0, 0, 0, 1, 1, 1])
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    assert out.channel("dual").cluster_status[2] == "empty_cluster"
    assert rec.count("knockoff") == 2
    assert rec.count("icp") == 2


def test_result_keeps_pre_and_post_icp_sets(monkeypatch):
    install_fakes(monkeypatch, Recorder(), y_candidates=(0, 1, 3),
                  x_candidates=(0, 1, 3), kept=(0, 1))
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    for k in range(K):
        assert out.channel("dual").selected_pre_icp[k] == {0, 1, 3}
        assert out.channel("dual").selected_post_icp[k] == {0, 1}
    assert out.icp_rule == "strict"


def test_only_the_strict_icp_rule_is_reported(monkeypatch):
    install_fakes(monkeypatch, Recorder())
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    assert not hasattr(out.channel("dual"), "kept_official")
    assert "kept_official" not in out.diagnostics


def test_icp_alpha_reaches_icp_detail(monkeypatch):
    rec = install_fakes(monkeypatch, Recorder())
    pipe.fit_calibration_selection(make_data(), make_point(), 1)
    assert rec.icp_alphas
    assert set(rec.icp_alphas) == {hcc.ALPHA_ICP}


def test_explicit_alpha_overrides_the_design_default(monkeypatch):
    rec = install_fakes(monkeypatch, Recorder())
    pipe.fit_calibration_selection(make_data(), make_point(), 1, alpha_icp=1e-6)
    assert set(rec.icp_alphas) == {1e-6}


def test_label_permutation_is_recovered_by_hungarian(monkeypatch):
    install_fakes(monkeypatch, Recorder(), labels=[2, 2, 0, 0, 1, 1])
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    assert out.k2true == {2: 0, 0: 1, 1: 2}
    assert out.label_exact is True
    assert out.seam_ari == pytest.approx(1.0)


def test_own_cluster_core_is_taken_from_the_matched_true_cluster(monkeypatch):
    install_fakes(monkeypatch, Recorder(), labels=[2, 2, 0, 0, 1, 1],
                  y_candidates=(0, 1), x_candidates=(0, 1), kept=(0, 1))
    data = make_data()
    data["A_core_by_cluster"] = {0: [0], 1: [1], 2: [3]}
    out = pipe.fit_calibration_selection(data, make_point(), 1)
    ch = out.channel("dual")
    # learned cluster 2 <-> true cluster 0 with A_core {0}; selected {0, 1} gives recall 1
    assert ch.a_core_recall[2] == pytest.approx(1.0)
    # learned cluster 1 <-> true cluster 2 with A_core {3}; selected {0, 1} gives recall 0
    assert ch.a_core_recall[1] == pytest.approx(0.0)


def test_unmapped_cluster_is_marked(monkeypatch):
    install_fakes(monkeypatch, Recorder())
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    assert pipe._cluster_status_for(99, [1, 2], out.k2true) == "cluster_unmapped"


def test_single_center_cluster_is_flagged(monkeypatch):
    rec = install_fakes(monkeypatch, Recorder(), labels=[0, 0, 0, 0, 1, 2])
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    ch = out.channel("dual")
    assert ch.cluster_status[1] == "single_center_cluster"
    assert ch.cluster_status[2] == "single_center_cluster"
    assert rec.count("icp") == 1


def test_knockoff_prerequisite_failure_is_a_status_not_an_exception(monkeypatch):
    install_fakes(monkeypatch, Recorder())

    def boom(Fp):
        raise AssertionError("fixed-X knockoffs need n >= 2p")

    monkeypatch.setattr(pipe, "fixed_x_knockoff", boom)
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    ch = out.channel("dual")
    assert set(ch.cluster_status.values()) == {"knockoff_prerequisite_failed"}
    assert ch.status == "unavailable"
    assert all(not s for s in ch.selected_post_icp.values())


def test_empty_candidate_set_is_not_called_ok(monkeypatch):
    install_fakes(monkeypatch, Recorder(), y_candidates=(), x_candidates=(),
                  kept=(), icp_status="empty_candidate")
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    ch = out.channel("dual")
    assert set(ch.cluster_status.values()) == {"empty_candidate"}
    assert ch.n_clusters_ok == 0


def test_icp_without_pvalues_is_flagged(monkeypatch):
    install_fakes(monkeypatch, Recorder(), icp_status="no_pvalues")
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    assert set(out.channel("dual").cluster_status.values()) == {"icp_no_pvalues"}


def test_rejected_candidates_stay_rejected(monkeypatch):
    install_fakes(monkeypatch, Recorder(), y_candidates=(0, 1, 3),
                  x_candidates=(0, 1, 3), kept=(), icp_status="empty_candidate")
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    for k in range(K):
        assert out.channel("dual").selected_post_icp[k] == set()


def test_diagnostics_carry_seam_margin_and_iteration_evidence(monkeypatch):
    install_fakes(monkeypatch, Recorder())
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    for name in ("seam_margin_min", "seam_margin_mean", "seam_margin_min_center",
                 "seam_n_centers_low_margin", "seam_n_iter", "seam_converged",
                 "t_seam", "t_select"):
        assert name in out.diagnostics
    assert out.diagnostics["seam_margin_min"] == pytest.approx(0.8)


def test_members_and_center_labels_agree(monkeypatch):
    install_fakes(monkeypatch, Recorder(), labels=[0, 0, 1, 1, 2, 2])
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    assert out.center_labels == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    assert out.members == {0: [0, 1], 1: [2, 3], 2: [4, 5]}


def test_icp_detail_rows_carry_the_channel(monkeypatch):
    install_fakes(monkeypatch, Recorder())
    out = pipe.fit_calibration_selection(make_data(), make_point(), 1)
    rows = out.channel("dual").icp_detail_rows
    assert rows and {r["channel"] for r in rows} == {"dual"}
