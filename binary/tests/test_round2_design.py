# -*- coding: utf-8 -*-
"""Design and wiring of round 2 (calibration and sensitivity, 59 points).

The three most important checks:
1. structure names equal those of round 1: seeds derive from structure_id; naming by this table would stop the
   U points from sharing F with their round-1 partners without U, and the C/S points with the round-1 gamma=+1
   points;
2. gamma of the S points comes only from the frozen file; without it the S points do not run;
3. SIG_Y=1 is actually applied.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import design_table as dt          # noqa: E402
import formal_dgp                  # noqa: E402
import run_formal                  # noqa: E402
import screening_bridge            # noqa: E402

CAL_ARMS = {"dual_icp", "oracle_reference"}


def _cfg():
    with open(os.path.join(_ROOT, "config", "formal_r2.json"), encoding="utf-8") as h:
        return json.load(h)


@pytest.fixture
def no_frozen(tmp_path, monkeypatch):
    """Tests never read the package results/ (it may hold a frozen value); they point to a missing file."""
    path = str(tmp_path / "gamma_small_frozen.json")
    monkeypatch.setattr(dt, "GAMMA_SMALL_FROZEN_PATH", path)
    return path


def _frozen(path, value="0.045"):
    with open(path, "w", encoding="utf-8") as h:
        json.dump({"status": "frozen", "design_revision": dt.DESIGN_REVISION,
                   "gamma_logor_str": value, "gamma_logor": float(value)}, h)


def test_run_table_is_round2_59(no_frozen):
    rows = dt.load_rows()
    assert len(rows) == 59
    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    assert tiers == {"calibration_added_null": 12, "calibration_small_gamma": 13,
                     "delta_sensitivity": 7, "correlation_sensitivity": 7,
                     "unmeasured_sensitivity": 20}
    assert {r["SIG_Y"] for r in rows} == {"1"}
    assert {r["n_reps"] for r in rows} == {"500"}
    assert {(r["param_reps"], r["data_reps"]) for r in rows} == {("500", "1")}
    assert {r["design_revision"] for r in rows} == {dt.DESIGN_REVISION}
    for r in rows:
        if r["calibration_member"] == "true":
            assert set(dt.methods_of(r)) == CAL_ARMS, r["point_id"]
        else:
            assert len(dt.methods_of(r)) == 4, r["point_id"]
    assert dt.unresolved_points(rows) == [f"S{i:02d}" for i in range(1, 14)]


def test_round2_rows_derive_from_v4(no_frozen):
    """Provenance: every row equals its v4 source row field by field, except the registered fields."""
    v4 = {r["point_id"]: r for r in dt.load_legacy_rows()}
    common = {"SIG_Y", "n_reps", "param_reps", "data_reps", "design_revision",
              "planned_estimator_attempts"}
    for new in dt.load_rows():
        pid = new["point_id"]
        if pid.startswith("S"):
            src = v4[new["pair_reference_point_id"]]
            allowed = common | {"point_id", "axis_membership", "tier", "scan_parameter",
                                "scan_value", "gamma_logor", "status", "calibration_member",
                                "report_panels", "execution_methods", "n_methods",
                                "pair_reference_point_id"}
            assert src["gamma_logor"] == "0"
            assert dt.structure_signature(new) == dt.structure_signature(src)
        else:
            src = v4[pid]
            allowed = common | ({"execution_methods", "n_methods"}
                                if pid.startswith("C") else set())
        diff = {k for k in new if new[k] != src[k]}
        assert diff <= allowed, f"{pid} changes unregistered fields: {sorted(diff - allowed)}"
        assert int(new["planned_estimator_attempts"]) == int(new["n_methods"]) * 3 * 500


def test_structure_ids_match_round1_and_pairs_share_seeds(no_frozen):
    rows = dt.load_rows()
    m = dt.structure_of(rows)
    assert m["U001"] == m["U006"] == m["S01"] == "STRUCT_P001"
    assert m["C001"] == m["S02"] == m["U015"] == "STRUCT_P029"
    assert m["U020"] == "STRUCT_P034" and m["C012"] == m["S13"] == "STRUCT_P040"
    legacy = dt.structure_of(dt.load_legacy_rows())
    for r in rows:
        pair = r["pair_reference_point_id"].strip()
        if pair:
            assert m[r["point_id"]] == legacy[pair], r["point_id"]
            assert dt.base_seeds(m[r["point_id"]], 7, 1) == dt.base_seeds(legacy[pair], 7, 1)


def test_static_checks_pass(no_frozen):
    c = dt.static_checks()
    assert c["problems"] == [] and c["status"] == "ok"
    assert c["n_rows"] == 59 and c["planned_units"] == 29_500
    assert c["planned_attempts"] == 279_000 == c["planned_estimator_attempts_total"]
    assert c["calibration_pairing"] == "cross_round"
    assert len(c["small_gamma_structures"]) == 13
    v4 = dt.static_checks(dt.load_legacy_rows())
    assert v4["problems"] == [] and v4["n_structures"] == 47


def test_unregistered_design_revision_is_rejected(no_frozen):
    rows = [dict(r) for r in dt.load_rows()]
    for r in rows:
        r["design_revision"] = "v6_unregistered"
    with pytest.raises(RuntimeError, match="no registered expectations"):
        dt.static_checks(rows)


def test_frozen_gamma_resolves_all_s_rows(no_frozen):
    _frozen(no_frozen, "0.045")
    rows = dt.load_rows()
    assert dt.unresolved_points(rows) == []
    assert {r["gamma_logor"] for r in rows if r["tier"] == dt.GAMMA_SMALL_TIER} == {"0.045"}
    c = dt.static_checks(rows)
    assert c["problems"] == [] and c["small_gamma_value"] == ["0.045"]
    with open(no_frozen, "w", encoding="utf-8") as h:
        json.dump({"status": "frozen", "design_revision": "another design",
                   "gamma_logor_str": "0.05"}, h)
    with pytest.raises(RuntimeError, match="valid frozen record"):
        dt.load_rows()


def test_runner_refuses_uncalibrated_s_points(no_frozen, tmp_path):
    rc = run_formal.main(["--out-dir", str(tmp_path / "run"), "--points", "S01,C001",
                          "--param-reps", "1-2", "--workers", "1"])
    assert rc == 7
    assert not os.path.isdir(str(tmp_path / "run" / "units"))


def test_verify_complete_counts_only_complete_units(no_frozen, tmp_path):
    out = str(tmp_path / "run")
    row = {"point_id": "C001"}
    run_formal.write_atomic(run_formal.unit_path(out, row, 1, 1),
                            {"output_version": run_formal.OUTPUT_VERSION, "unit_complete": True})
    run_formal.write_atomic(run_formal.unit_path(out, row, 2, 1),
                            {"output_version": run_formal.OUTPUT_VERSION,
                             "unit_complete": False, "status": "failed"})
    args = ["--out-dir", out, "--points", "C001", "--param-reps", "1-2", "--verify-complete"]
    assert run_formal.main(args) == 5
    rep = json.load(open(os.path.join(out, "verify_complete.json"), encoding="utf-8"))
    assert rep["points"]["C001"] == {"complete": 1, "failed_or_corrupt": 1, "expected": 2}
    run_formal.write_atomic(run_formal.unit_path(out, row, 2, 1),
                            {"output_version": run_formal.OUTPUT_VERSION, "unit_complete": True})
    assert run_formal.main(args) == 0


def test_config_matches_table():
    cfg = _cfg()
    assert cfg["icp_alpha"] == 0.001
    assert cfg["param_reps"] == list(range(1, 501)) and cfg["data_reps"] == [1]
    assert cfg["save_matching_detail"] != "full"
    assert cfg["run_table_sha256"] == dt.sha256_of(dt.RUN_TABLE_PATH) == dt.RUN_TABLE_SHA256
    assert cfg["design_revision"] == dt.DESIGN_REVISION
    assert cfg["output_version"] == dt.IMPLEMENTATION_VERSION
    cal = cfg["gamma_small_calibration"]
    assert not set(cal["param_reps"]) & set(cfg["param_reps"]), "calibration and formal param_reps must not overlap"
    assert cal["template_point"] == "S01"


def _tiny(point_id="P041"):
    rows = dt.load_rows()
    row = next(r for r in rows if r["point_id"] == point_id)
    sid = dt.structure_of(rows)[point_id]
    block = dict(dt.build_structures(rows)[sid])
    block["params"] = {**block["params"], "n_per_center": 300, "p_total": 200,
                       "n_per_center_var": 0.0}
    group = {"structure_id": sid, "base_group_key": f"{sid}__p001__d01",
             "param_rep": 1, "data_rep": 1, "param_seed": 7, "data_seed": 7,
             "u_epsilon_seed": 7, "sig_x_spec": block["sig_x_spec"]}
    return row, group, {sid: block}


def test_sig_y_is_actually_applied_and_restored(no_frozen):
    row, group, structures = _tiny()
    _gen, b6 = formal_dgp.load_sources()
    assert float(b6.SIG_Y) == 2.0, "the vendored constant keeps its value 2.0"
    continuous, _ = formal_dgp.generate_base(group, structures)
    _o, t1, m1 = formal_dgp.binary_unit(continuous, row, group, b6)
    assert m1["sig_y_applied"] == 1.0 and float(b6.SIG_Y) == 2.0
    _o, t2, _m = formal_dgp.binary_unit(continuous, {**row, "SIG_Y": "2"}, group, b6)
    assert t1["ate_do"] - t2["ate_do"] > 1e-3, "SIG_Y is not applied: ate_do is nearly equal for 1 and 2"


def test_sig_x_and_c0_guards_fire(no_frozen):
    row, group, structures = _tiny()
    _gen, b6 = formal_dgp.load_sources()
    continuous, _ = formal_dgp.generate_base(group, structures)
    for field in ("SIG_X", "C0"):
        with pytest.raises(RuntimeError, match=field):
            formal_dgp.binary_unit(continuous, {**row, field: str(float(row[field]) + 0.5)},
                                   group, b6)


def test_icp_alpha_injection():
    m6 = screening_bridge.load_screening(formal_dgp.STEP6_DIR)
    assert m6.ALPHA_ICP == 0.05, "vendored default stays 0.05"
    assert 0.0 < float(_cfg()["icp_alpha"]) <= 0.05
