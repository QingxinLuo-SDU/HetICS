"""merge_pooled.py end to end: tar.gz / directory / zip inputs, version filter, failures, duplicates, gate
directories, arms outside POOLED_ARMS and the summary definitions."""
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "merge"))
import merge_pooled as mp  # noqa: E402

HDPS_V, POOL_V = "binary-fully-pooled-fpx-v1-20261031", "binary-pooled-v3-20261104"


def unit(point, rep, err, cover, reject, arms=("hdps_fp",), version=HDPS_V, complete=True, gamma=0.5):
    fam = {"n_Z": 20, "sel_Z": 20, "n_shared": 25, "sel_shared": 25, "n_spec_own": 5, "n_spec_foreign": 10,
           "n_spur": 50, "sel_spur": 10, "n_noise": 390, "sel_noise": 39, "n_selected": 100}
    fs = {str(k): dict(fam, sel_spec_own=own, sel_spec_foreign=5 - own, observed_core_recall=rec)
          for k, own, rec in ((0, 3, 0.96), (1, 4, 0.90), (2, 5, 1.0))}       # 20 + 25 + 5 = 50 safe in every cluster
    row = lambda stage: {"stage": stage, "estimate": 0.1 + err, "error": err, "se": 0.01, "ci_valid": True,  # noqa: E731
                         "cover_true": cover, "reject_zero": reject, "point_status": "complete"}
    return {"output_version": version, "unit_key": f"{point}__p{rep:03d}__d01", "point_id": point, "param_rep": rep,
            "data_rep": 1, "gamma_logor": gamma, "unit_complete": complete,
            "effects": {a: {e: [row("cluster"), row("overall")] for e in ("ra_standardized", "aipw", "psm_caliper")}
                        for a in arms},
            "feature_selection": {a: fs for a in arms}}


def add(tf, path, rec):
    data = json.dumps(rec).encode()
    info = tarfile.TarInfo(path)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_merge_end_to_end(tmp_path):
    base = "runs_pooled/w1/r1/units/P004/"
    with tarfile.open(tmp_path / "w1_r1.tar.gz", "w:gz") as tf:
        add(tf, base + "P004__p001__d01.json", unit("P004", 1, 0.01, True, True, arms=("hdps_fp", "other_arm")))
        add(tf, base + "P004__p002__d01.json", unit("P004", 2, -0.02, False, True))
        add(tf, base + "P004__p003__d01.json", unit("P004", 3, 0.04, True, False))
        add(tf, base + "P004__p004__d01.json", unit("P004", 4, 0.0, True, True, version="old-version"))
        add(tf, base + "P004__p005__d01.json", unit("P004", 5, 0.0, True, True, complete=False))
        add(tf, "runs_pooled/w1/r1/gate/units/P004/P004__p006__d01.json", unit("P004", 6, 9.9, True, True))
    dup = tmp_path / "dir/runs_pooled/w2/r1/units/P004"
    dup.mkdir(parents=True)
    (dup / "P004__p001__d01.json").write_text(json.dumps(unit("P004", 1, 0.01, True, True)))
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:gz") as tf:
        add(tf, "runs_pooled/w3/r2/units/P041/P041__p001__d01.json",
            unit("P041", 1, 0.05, True, True, arms=("pooled_lasso_fp",), version=POOL_V, gamma=0.0))
    with zipfile.ZipFile(tmp_path / "export.zip", "w") as z:
        z.writestr("w3_r2.tar.gz", inner.getvalue())

    out = tmp_path / "out"
    assert mp.main(["--inputs", str(tmp_path / "w1_r1.tar.gz"), str(tmp_path / "dir"),
                    str(tmp_path / "export.zip"), "--out", str(out)]) == 0
    qa = json.loads((out / "qa.json").read_text(encoding="utf-8"))
    assert qa["units::hdps"] == 3 and qa["units::pooled"] == 1
    assert qa["duplicate::hdps"] == 1 and qa["max_duplicate_aipw_diff"] == 0.0
    assert qa["failed::hdps"] == 1 and qa["skip_version::old-version"] == 1

    est = pd.read_csv(out / "est_pooled.csv")
    assert set(est.method) == {"hdps_fp", "pooled_lasso_fp"}                     # arms outside POOLED_ARMS dropped
    a = est[(est.method == "hdps_fp") & (est.estimator == "aipw")].iloc[0]
    errs = np.array([0.01, -0.02, 0.04])
    assert a.n_valid == 3 and np.isclose(a.bias, errs.mean()) and np.isclose(a.empirical_se, errs.std(ddof=1))
    assert np.isclose(a.mse, a.bias ** 2 + a.empirical_se ** 2)
    assert np.isclose(a.coverage, 2 / 3) and np.isclose(a.power, 2 / 3) and np.isnan(a.type1_error)
    b = est[est.method == "pooled_lasso_fp"].iloc[0]
    assert np.isclose(b.type1_error, 1.0) and np.isnan(b.power)                 # gamma = 0 gives the type I error

    sel = pd.read_csv(out / "sel_pooled.csv")
    s = sel[sel.method == "hdps_fp"].iloc[0]
    assert s.n_units == 3 and np.isclose(s.core_recall, 0.90) and np.isclose(s.precision, 0.50)
    assert np.isclose(s.spurious_recall, 10 / 50) and np.isclose(s.noise_rate, 39 / 390) and s.adjustment_set_size == 100
    cnt = pd.read_csv(out / "counts.csv")
    assert cnt.set_index("point_id").loc["P004", "n_units"] == 3 and not cnt.complete.any()
