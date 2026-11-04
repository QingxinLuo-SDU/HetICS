"""Logic of the continuous runner and estimators (no simulation data). Requires SCENA_BETA_ANCHOR=1.0,1.25."""
import math

import numpy as np
import pandas as pd
import pytest

import step6x_config as cfg

pytestmark = pytest.mark.skipif(cfg.BETA_TAG != "b1-125", reason="requires SCENA_BETA_ANCHOR=1.0,1.25")


def _multicenter(rng, n_centers=4, n=1500, q=8, conf=0.3, gamma=1.5):
    env = np.repeat(np.arange(n_centers), n)
    W = rng.standard_normal((env.size, q))
    X = W @ np.full(q, conf / np.sqrt(q)) + rng.standard_normal(env.size)
    Y = gamma * X + W @ np.ones(q) + 0.3 * env + rng.standard_normal(env.size)
    return X, Y, W, env


def test_buckets_partition_the_design_exactly_once():
    import run_ra_cont as r
    full = r.all_tasks()
    parts = r.all_tasks(0, 417) + r.all_tasks(417, 629) + r.all_tasks(629, 1000)
    assert len(full) == len(r._combos()) * 1000
    assert sorted(parts) == sorted(full) and len(set(parts)) == len(parts)


def test_combo_filter_selects_axes_only():
    import run_ra_cont as r
    sub = r.all_tasks(0, 1000, "H")
    assert sub and all(c.startswith("H") for c, _p, _d in sub)
    assert len(r.all_tasks(0, 1000, "A,B")) == sum(1 for c, _p, _d in r.all_tasks() if c[0] in "AB")
    assert set(r.all_tasks(417, 629, "H")) <= set(r.all_tasks(417, 629))


def test_rep_complete_requires_version_and_all_arms(tmp_path):
    import run_ra_cont as r
    from ra_core import CLUSTER_ARMS, POOLED_ARMS, RA_VERSION
    f = tmp_path / "p00_d000.json.gz"
    arms = {m: {} for m in CLUSTER_ARMS + POOLED_ARMS}
    r.write_atomic(f, {"output_version": RA_VERSION, "arms": arms})
    assert r.rep_complete(f)
    r.write_atomic(f, {"output_version": RA_VERSION, "arms": {m: {} for m in CLUSTER_ARMS}})
    assert not r.rep_complete(f)
    r.write_atomic(f, {"output_version": "old", "arms": arms})
    assert not r.rep_complete(f)
    f.write_bytes(b"\x1f\x8b broken")
    assert not r.rep_complete(f)


def test_ra_is_the_ols_coefficient_with_center_dummies():
    from ra_core import extra_estimators
    X, Y, W, env = _multicenter(np.random.default_rng(0))
    out = extra_estimators(X, Y, W, env)
    D = np.column_stack([np.ones(X.size), (env[:, None] == np.arange(1, 4)).astype(float), X, W])
    beta = np.linalg.lstsq(D, Y, rcond=None)[0]
    assert out["ra"]["status"] == "ok" and out["ra"]["theta"] == pytest.approx(beta[4], abs=1e-10)
    lo, hi = out["ra"]["ci_t"]
    assert hi - lo == pytest.approx(2 * float(__import__("scipy").stats.t.ppf(0.975, 3)) * out["ra"]["se"])


def test_ra_and_gps_cover_truth_when_confounding_is_mild():
    from ra_core import extra_estimators
    rng = np.random.default_rng(1)
    ra, gps, gpt = [], [], []
    for _ in range(40):
        out = extra_estimators(*_multicenter(rng))
        ra.append(out["ra"]["theta"]); gps.append(out["gps_ipw"]["theta"]); gpt.append(out["gps_ipw_trunc"]["theta"])
    assert abs(np.mean(ra) - 1.5) < 0.01
    assert abs(np.mean(gps) - 1.5) < 0.05
    # truncation trades bias for variance; the unadjusted bias here is about 0.78
    assert abs(np.mean(gpt) - 1.5) < 0.2


def test_single_center_is_unavailable_not_fallback():
    from ra_core import ESTIMATORS, extra_estimators
    X, Y, W, env = _multicenter(np.random.default_rng(2), n_centers=1)
    out = extra_estimators(X, Y, W, env)
    assert set(out) == set(ESTIMATORS[1:]) and all(b["status"] == "unavailable" for b in out.values())


def test_mu_needs_all_three_clusters():
    from ra_core import aggregate
    ok = [dict(status="ok", theta=t, se=0.02) for t in (1.48, 1.50, 1.52)]
    res = aggregate(ok)
    assert res["status"] == "ok" and 1.48 < res["reml_gls"]["mu"] < 1.52
    assert aggregate(ok[:2] + [dict(status="unavailable")])["status"] == "unavailable"
    assert aggregate(ok[:2])["status"] == "unavailable"


def test_summarize_theta_two_level():
    import run_ra_cont as r
    rows = []
    for p in range(2):
        for d, th in enumerate((1.0, 2.0) if p == 0 else (1.5, 1.5)):
            rows.append(dict(experiment="grid", combo_id="A05", param_index=p, data_index=d, true_gamma=1.5,
                             method="dual_icp", scope="learned_cluster", k=0, k_true=0, estimator="ra",
                             status="ok", theta_hat=th, se=0.1, covered_t=(p == 1), reject0_t=True,
                             n_selected=10, recall_core=1.0, n_spur=0, n_noise=0))
    s = r.summarize_theta(pd.DataFrame(rows)).iloc[0]
    assert s.bias == pytest.approx(0.0)
    assert s.rmse == pytest.approx(math.sqrt(0.25 / 2))
    assert s.coverage_t == pytest.approx(0.5) and s.n_param_seeds == 2 and s.n_ok == 4
