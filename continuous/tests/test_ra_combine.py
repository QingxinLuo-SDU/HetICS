"""Point-by-point combination equals reading everything at once; duplicates count once; other output versions are skipped."""
import gzip
import json
from argparse import Namespace

import pandas as pd
import pytest

from ra_core import RA_VERSION
from run_ra_cont import (cmd_combine, load_raw, summarize_mu, summarize_selection, summarize_theta,
                         truth_sets)

T = truth_sets(500)


def _est(theta):
    return {"ra": dict(status="ok", theta=theta, se=0.1, ci_t=[theta - 0.2, theta + 0.2], covered_t=True,
                       reject0_t=True)}


def _write(root, cid, p, d, theta, version=RA_VERSION):
    clusters = [dict(k=k, k_true=k, M_k=4, selected=sorted(T["core"][k]), n_spur=0, n_noise=0, recall_core=1.0,
                     n_selected=len(T["core"][k]), est=_est(theta + 0.01 * k)) for k in range(3)]
    arms = {"dual_icp": dict(scope="per_learned_cluster", clusters=clusters,
                             mu={"ra": dict(status="ok", reml_gls=dict(mu=theta, var_mu=0.01, tau2=0.0),
                                            fgls=dict(mu=theta), dl=dict(mu=theta))}),
            "pooled_lasso_fp": dict(scope="all_centers_pooled", selected=sorted(T["safe"]), recall_core=1.0,
                                    n_spur=0, n_noise=0, est=_est(theta))}
    rec = dict(output_version=version, experiment="grid", combo_id=cid, param_index=p, data_index=d,
               true_gamma=1.5, p=500, n_centers=12, arms=arms)
    f = root / "grid" / cid / f"p{p:02d}_d{d:03d}.json.gz"
    f.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(f, "wt", encoding="utf-8") as h:
        json.dump(rec, h)


@pytest.mark.parametrize("workers", [1, 2])
def test_by_point_equals_whole_load(tmp_path, workers):
    a, b = tmp_path / "m1" / "runs_ra", tmp_path / "m2" / "runs_ra"
    for d in range(3):
        _write(a, "A00", 0, d, 1.5 + 0.1 * d)
    for p in range(2):
        _write(a, "B00", p, 0, 1.4 + 0.2 * p)
    _write(b, "A00", 0, 0, 1.5)                    # duplicate replicate: counted once
    _write(b, "A00", 1, 0, 1.7)
    _write(b, "A00", 1, 1, 9.9, version="old")     # other output version: skipped
    dest = tmp_path / "out"
    assert cmd_combine(Namespace(runs=[str(a), str(b)], dest=str(dest), workers=workers)) == 5   # fewer than 1000 replicates

    th, mu, sel = load_raw([a, b], with_selection=True, version=RA_VERSION)
    assert len(th.drop_duplicates(["combo_id", "param_index", "data_index"])) == 6
    keys = {"summary_theta": (summarize_theta(th), ["combo_id", "method", "estimator"]),
            "summary_mu": (summarize_mu(mu), ["combo_id", "method", "estimator"]),
            "summary_selection": (summarize_selection(sel), ["combo_id", "method"])}
    for name, (whole, k) in keys.items():
        got = pd.read_csv(dest / f"{name}.csv").sort_values(k).reset_index(drop=True)
        want = whole.sort_values(k).reset_index(drop=True)
        pd.testing.assert_frame_equal(got[want.columns], want, check_dtype=False)
    assert len(pd.read_csv(dest / "raw_theta.csv.gz")) == len(th)   # appended gzip members read as one file
    cnt = pd.read_csv(dest / "counts_check.csv").set_index("combo_id").n_reps.to_dict()
    assert cnt == {"A00": 4, "B00": 2}
