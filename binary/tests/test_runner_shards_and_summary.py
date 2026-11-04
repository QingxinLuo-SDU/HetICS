"""Runner and summary: shard markers, resume, failure records, denominators and the two coverages.

Built from constructed unit records without the DGP; the bookkeeping is checked, not numbers.
"""

from __future__ import annotations

import json
import os

import pytest

import design_table as dt
import run_formal
import summarize_formal as sf


def _effect_row(**kw):
    row = {"estimator": "ra_standardized", "analysis_cluster": -1,
           "stage": "overall", "target_scope": "analysis_cluster_full_members",
           "estimate": None, "se": None, "truth_target": 0.0, "error": None,
           "abs_error": None, "ci_valid": False, "ci_low": None, "ci_high": None,
           "ci_width": None, "cover_true": None, "cover_zero": None,
           "reject_zero": None, "invalid_reason": None,
           "point_status": "complete", "se_status": "ok", "estimand": "full",
           "target_different": False, "raw_gap_to_full": None,
           "target_shift": 0.0, "n_target": 100}
    row.update(kw)
    return row


def _unit(point_id, param_rep, data_rep, *, gamma, rows, complete=True):
    return {
        "output_version": run_formal.OUTPUT_VERSION,
        "design_revision": dt.DESIGN_REVISION,
        "unit_key": f"{point_id}__p{param_rep:03d}__d{data_rep:02d}",
        "point_id": point_id, "scenario": "observed_confounders", "tier": "main",
        "report_panels": ["main"], "param_rep": param_rep, "data_rep": data_rep,
        "structure_id": "STRUCT_X", "unit_complete": complete,
        "errors": [], "observed_shape": {"n_rows": 100, "p": 10, "n_centers": 3},
        "unit_meta": {"prevalence_x": 0.5, "prevalence_y": 0.3},
        "timings": {"unit_total": 1.0, "label": 0.5},
        "label_status": {"center_map_available": True},
        "cluster_diagnostics": {"n_analysis_clusters": 3},
        "truth": {"gamma_logor": gamma, "ate_do": 0.0},
        "effects": {"dual_icp": {"ra_standardized": rows}},
        "estimator_diagnostics": {}, "selections": {}, "selected_by_cluster": {},
        "feature_selection": {},
    }


@pytest.fixture
def run_dir(tmp_path):
    return str(tmp_path / "run")


def test_resume_skips_complete_and_redoes_incomplete(run_dir):
    row = {"point_id": "P001"}
    good = run_formal.unit_path(run_dir, row, 1, 1)
    run_formal.write_atomic(good, {"output_version": run_formal.OUTPUT_VERSION,
                                   "unit_complete": True})
    assert run_formal.unit_is_complete(good) is True

    bad = run_formal.unit_path(run_dir, row, 2, 1)
    run_formal.write_atomic(bad, {"output_version": run_formal.OUTPUT_VERSION,
                                  "unit_complete": False, "status": "failed",
                                  "reason": "RuntimeError: intentional failure"})
    assert run_formal.unit_is_complete(bad) is False      # failed units are rerun

    stale = run_formal.unit_path(run_dir, row, 3, 1)
    run_formal.write_atomic(stale, {"output_version": "another implementation version",
                                    "unit_complete": True})
    assert run_formal.unit_is_complete(stale) is False    # another implementation version is not reused

    missing = run_formal.unit_path(run_dir, row, 4, 1)
    assert run_formal.unit_is_complete(missing) is False
    assert not os.path.exists(missing + ".tmp")           # atomic writes leave no temporary file


def test_shard_marker_needs_all_planned_data_reps(run_dir):
    row = {"point_id": "P001"}
    run_formal.write_atomic(run_formal.unit_path(run_dir, row, 1, 1),
                            {"output_version": run_formal.OUTPUT_VERSION,
                             "unit_complete": True})
    # only data_rep=1 of 10 planned -> no complete marker
    markers = run_formal.write_shard_markers(run_dir, [row], [1], [1], 10)
    assert markers["shard_complete_written"] == 0
    assert markers["shard_incomplete"] == 1

    for d in range(2, 11):
        run_formal.write_atomic(run_formal.unit_path(run_dir, row, 1, d),
                                {"output_version": run_formal.OUTPUT_VERSION,
                                 "unit_complete": True})
    markers = run_formal.write_shard_markers(run_dir, [row], [1], list(range(1, 11)), 10)
    assert markers["shard_complete_written"] == 1


def test_failed_unit_enters_ledger_not_silently_dropped(run_dir, monkeypatch):
    units = [_unit("P001", 1, 1, gamma=0.0, rows=[_effect_row()]),
             {"output_version": run_formal.OUTPUT_VERSION, "unit_key": "P001__p002__d01",
              "point_id": "P001", "unit_complete": False, "status": "failed",
              "reason": "RuntimeError: intentional failure", "param_rep": 2, "data_rep": 1}]
    for rec in units:
        run_formal.write_atomic(
            os.path.join(run_dir, "units", "P001", f"{rec['unit_key']}.json"), rec)
    loaded = sf.load_units(run_dir)
    assert len(loaded) == 2
    tables = sf.build_tables(loaded)
    assert len(tables["unit_status"]) == 2
    assert any(r.get("fatal") for r in tables["errors"])
    assert sum(1 for r in tables["unit_status"] if r["unit_complete"]) == 1


def test_summary_denominators_and_two_coverages():
    """gamma=0 reports the type I error; gamma=+1 the power = 1 - coverage_zero."""
    rows = []
    # 10 parameter blocks with 1 replication each: 6 intervals cover 0, 4 exclude it. At gamma=0 the truth is
    # exactly 0, so cover_true and cover_zero coincide; `type1 = 1 - coverage_true` holds only then.
    for i in range(10):
        reject = i < 4
        rows.append({"point_id": "PX", "method": "dual_icp", "param_rep": i + 1,
                     "data_rep": 1,
                     **_effect_row(estimate=0.02 * (i + 1), se=0.01,
                                   truth_target=0.0,
                                   error=0.02 * (i + 1),
                                   ci_valid=True, ci_low=-0.01, ci_high=0.03,
                                   cover_true=not reject,
                                   cover_zero=not reject, reject_zero=reject)})
    design = {"PX": {"gamma_logor": "0", "scenario": "s", "tier": "main",
                     "report_panels": "main"}}
    summary = sf.summarize(rows, design, planned_reps=10)
    assert len(summary) == 1
    rec = summary[0]
    assert rec["n_planned_stage"] == 10
    assert rec["n_attempted"] == 10
    assert rec["n_valid_ci"] == 10
    assert rec["n_failed"] == 0
    assert rec["type1_error"] == pytest.approx(0.4)
    assert rec["power"] is None
    assert rec["coverage_zero"] == pytest.approx(0.6)
    assert rec["type1_error"] == pytest.approx(1 - rec["coverage_true"])
    assert rec["n_param_blocks"] == 10
    assert rec["bias_mcse"] is not None

    # the same numbers read at gamma=+1: the power comes from coverage_zero only; rows with a nonzero truth make
    # the two coverages differ
    alt = []
    for i in range(10):
        reject = i < 4
        alt.append({"point_id": "PX", "method": "dual_icp", "param_rep": i + 1,
                    "data_rep": 1,
                    **_effect_row(estimate=0.3, se=0.05, truth_target=0.3,
                                  error=0.0, ci_valid=True,
                                  ci_low=0.2, ci_high=0.4,
                                  cover_true=True,
                                  cover_zero=not reject, reject_zero=reject)})
    design["PX"]["gamma_logor"] = "1"
    rec = sf.summarize(alt, design, planned_reps=10)[0]
    assert rec["type1_error"] is None
    assert rec["coverage_true"] == pytest.approx(1.0)
    assert rec["power"] == pytest.approx(1 - rec["coverage_zero"])
    assert rec["power"] != pytest.approx(1 - rec["coverage_true"])


def test_invalid_ci_is_not_counted_as_not_rejecting():
    rows = []
    for i in range(10):
        valid = i < 4
        rows.append({"point_id": "PX", "method": "dual_icp", "param_rep": i + 1,
                     "data_rep": 1,
                     **_effect_row(estimate=0.1, se=(0.01 if valid else None),
                                   truth_target=0.1, error=0.0,
                                   ci_valid=valid,
                                   ci_low=(0.08 if valid else None),
                                   ci_high=(0.12 if valid else None),
                                   cover_true=(True if valid else None),
                                   cover_zero=(False if valid else None),
                                   reject_zero=(True if valid else None),
                                   invalid_reason=(None if valid else "se_unavailable"))})
    design = {"PX": {"gamma_logor": "1", "scenario": "s", "tier": "main",
                     "report_panels": "main"}}
    rec = sf.summarize(rows, design, planned_reps=10)[0]
    assert rec["n_attempted"] == 10
    assert rec["n_valid_ci"] == 4
    assert rec["ci_availability"] == pytest.approx(0.4)
    # the rejection-rate denominator is the valid intervals, not 10; failures do not count as non-rejections
    assert rec["power"] == pytest.approx(1.0)
    assert rec["n_failed"] == 0                    # point estimates exist; only the SE is unavailable


def test_unexecuted_repetitions_are_not_failures():
    rows = [{"point_id": "PX", "method": "dual_icp", "param_rep": 1, "data_rep": 1,
             **_effect_row(estimate=0.1, se=0.01, error=0.0, ci_valid=True,
                           ci_low=0.05, ci_high=0.15, cover_true=True,
                           cover_zero=False, reject_zero=True)}]
    design = {"PX": {"gamma_logor": "1", "scenario": "s", "tier": "main",
                     "report_panels": "main"}}
    rec = sf.summarize(rows, design, planned_reps=1000)[0]
    assert rec["n_planned_stage"] == 1000
    assert rec["n_attempted"] == 1
    assert rec["n_failed"] == 0            # the 999 replications not executed yet are not failures


def test_paired_diff_marks_target_different():
    common = {"point_id": "PX", "param_rep": 1, "data_rep": 1}
    rows = [
        {**common, "method": "dual_icp", **_effect_row(estimator="psm_caliper",
         target_scope="psm_caliper_retained_members", error=0.01,
         target_different=True)},
        {**common, "method": "xside_icp", **_effect_row(estimator="psm_caliper",
         target_scope="psm_caliper_retained_members", error=0.03,
         target_different=True)},
        {**common, "method": "xside_icp", **_effect_row(error=0.02)},
        {**common, "method": "dual_icp", **_effect_row(error=0.01)},
    ]
    diffs = sf.paired_diffs(rows)
    by_est = {(d["method"], d["estimator"]): d for d in diffs}
    assert by_est[("xside_icp", "ra_standardized")]["comparable_same_target"] is True
    psm = by_est[("xside_icp", "psm_caliper")]
    assert psm["comparable_same_target"] is False
    assert "target-different" in psm["comparability_note"]
    assert psm["error_diff"] == pytest.approx(0.02)


def test_anomaly_report_flags_are_leads_not_conclusions():
    summary = [{"point_id": "PX", "scenario": "s", "tier": "main",
                "gamma_logor": 0.0, "method": "dual_icp",
                "estimator": "ra_standardized", "stage": "overall",
                "n_planned_stage": 10, "n_attempted": 10, "n_valid_ci": 10,
                "point_availability": 1.0, "ci_availability": 1.0,
                "bias": 0.1, "rmse": 0.2, "coverage_true": 0.5,
                "coverage_zero": 0.5, "reject_zero_rate": 0.5,
                "type1_error": 0.5, "mean_model_se": 0.01, "error_sd": 0.2}]
    flags = sf.anomaly_report(summary)
    assert len(flags) == 1
    assert "coverage_true" in flags[0]["flags"]
    assert "rejection rate at zero" in flags[0]["flags"]
    assert "few replications" in flags[0]["caveat"]
