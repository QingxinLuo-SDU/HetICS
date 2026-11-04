"""merge_ablation.py on synthetic unit records: estimation summaries, union-based selection metrics and pairing."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "merge"))
import merge_ablation as ma  # noqa: E402
from frozen_rerun import stage_rows  # noqa: E402


def unit(row, rep, errors, selected, version="binary-ablation-v6-20261028"):
    n_safe = int(row["n_Z"]) + int(row["n_causal_shared"]) + int(row["K"]) * int(row["n_causal_specific"])
    fam = {"n_Z": int(row["n_Z"]), "n_shared": int(row["n_causal_shared"]), "n_spec_own": int(row["n_causal_specific"]),
           "n_spec_foreign": n_safe - int(row["n_Z"]) - int(row["n_causal_shared"]) - int(row["n_causal_specific"]),
           "n_spur": int(row["n_spurious"]), "n_noise": int(row["n_noise"]), "observed_core_recall": 1.0}
    eff = lambda e: [{"stage": s, "point_status": "complete", "error": e, "ci_valid": True,  # noqa: E731
                      "cover_true": abs(e) < 0.05, "reject_zero": True} for s in ("cluster", "overall")]
    arms = ma.SOURCES["ablation"]["arms"]
    return {"output_version": version, "unit_complete": True, "point_id": row["point_id"],
            "unit_key": f"{row['point_id']}__p{rep:03d}__d01",
            "effects": {a: {e: eff(errors[a]) for e in ma.ESTIMATORS} for a in arms},
            "selections": {a: {"status": "complete"} for a in arms},
            "feature_selection": {a: {"0": fam, "1": fam} for a in arms},
            "selected_by_cluster": {a: {"0": selected[a], "1": selected[a]} for a in arms}}


def test_merge_ablation_from_unit_directory(tmp_path):
    row = stage_rows()["P001"]
    n_safe = int(row["n_Z"]) + int(row["n_causal_shared"]) + int(row["K"]) * int(row["n_causal_specific"])
    spur = n_safe                                                      # first spurious covariate
    sel = {"dual_icp": [0, 1], "dual_no_icp": [0, 1, spur], "dual_global_icp": [0]}
    units = tmp_path / "runs_ablation" / "w1" / "r1" / "units" / "P001"
    units.mkdir(parents=True)
    for rep, (d, n, g) in enumerate([(0.01, 0.02, -0.03), (0.03, 0.01, 0.07)], start=1):
        rec = unit(row, rep, {"dual_icp": d, "dual_no_icp": n, "dual_global_icp": g}, sel)
        (units / f"{rec['unit_key']}.json").write_text(json.dumps(rec), encoding="utf-8")
    out = tmp_path / "out"
    assert ma.main(["--source", "ablation", "--inputs", str(tmp_path / "runs_ablation"), "--out", str(out)]) == 0

    est = pd.read_csv(out / "est.csv")
    d = est[(est.method == "dual_icp") & (est.estimator == "aipw") & (est.level == "overall")].iloc[0]
    assert d.n_valid == 2 and np.isclose(d.bias, 0.02) and np.isclose(d.empirical_se, np.std([0.01, 0.03], ddof=1))
    s = pd.read_csv(out / "sel.csv").set_index("method")
    assert s.loc["dual_icp", "precision"] == 1.0 and np.isclose(s.loc["dual_no_icp", "precision"], 2 / 3)
    assert np.isclose(s.loc["dual_no_icp", "spurious_recall"], 1 / int(row["n_spurious"]))
    p = pd.read_csv(out / "paired.csv").set_index("method")
    assert np.isclose(p.loc["dual_global_icp", "d_sq_error"], np.mean([0.03 ** 2 - 0.01 ** 2, 0.07 ** 2 - 0.03 ** 2]))
    qa = json.loads((out / "merge_qa.json").read_text(encoding="utf-8"))
    assert qa["units"] == 2 and qa["points_not_n_reps"] == {"P001": 2}


def test_merge_ablation_rejects_other_versions(tmp_path):
    row = stage_rows()["P001"]
    sel = {a: [0] for a in ma.SOURCES["ablation"]["arms"]}
    units = tmp_path / "r1" / "units" / "P001"
    units.mkdir(parents=True)
    rec = unit(row, 1, {a: 0.0 for a in sel}, sel, version="other")
    (units / f"{rec['unit_key']}.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SystemExit):
        ma.main(["--source", "ablation", "--inputs", str(tmp_path), "--out", str(tmp_path / "out")])
