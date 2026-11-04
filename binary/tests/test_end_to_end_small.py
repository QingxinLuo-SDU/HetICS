"""Small end-to-end run of a formal unit: every arm x 3 estimators, with labels, support sets, ICP, truth and
both coverages. The dimensions are far below the formal design; the test checks the interfaces only."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import arms
import formal_dgp
import run_formal
from conftest import LABELED_PARAMS, PKG_ROOT, small_row
from screening_bridge import load_screening

ALL_METHODS = list(arms.ALL_ARMS)


@pytest.fixture(scope="module")
def config():
    with open(os.path.join(PKG_ROOT, "config", "formal_r2.json"),
              encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["save_matching_detail"] = "summary"
    return cfg


@pytest.fixture(scope="module")
def unit_record(labeled_continuous, small_group, sources, config):
    _gen, b6 = sources
    m6 = load_screening(formal_dgp.STEP6_DIR)
    row = small_row("TEST_P", gamma_logor=1.0, has_u=False, methods=ALL_METHODS)
    row["p_total"] = str(LABELED_PARAMS["p_total"])
    record, _matching = run_formal.run_unit(
        labeled_continuous, row, small_group,
        {"TEST": {"params": LABELED_PARAMS, "sig_x_spec": 0.6}},
        m6, b6, config, None, None)
    return record


def test_unit_completes_with_all_arms(unit_record):
    assert unit_record["unit_complete"] is True
    assert set(unit_record["execution_methods"]) == set(ALL_METHODS)
    assert unit_record["errors"] == []
    assert set(unit_record["effects"]) == set(ALL_METHODS)


def test_label_stage_and_support_export(unit_record):
    status = unit_record["label_status"]
    assert status["primary_arm"] == "multiview_pool_composite_guarded"
    if status.get("center_map_available"):
        assert unit_record["support_status"] == "complete"
        support = unit_record["support_by_cluster"]
        assert support, "support sets come from the nonzero original columns of primary.parameters.beta"
        p = unit_record["observed_shape"]["p"]
        for g, cols in support.items():
            assert len(cols) <= 25
            assert all(0 <= int(j) < p for j in cols)
    else:
        # ineligible labels: no estimate that needs the partition is available
        assert unit_record["support_status"] == "label_unavailable"
        for name in arms.FIXEDX_ARMS + arms.ORACLE_ARMS:
            rows = unit_record["effects"][name]["ra_standardized"]
            assert all(r["point_status"] == "unavailable" for r in rows)


def test_oracle_arm_definitions(unit_record):
    sel = unit_record["selected_by_cluster"]
    truth = unit_record["truth"]
    if sel.get("oracle_min") is None:
        pytest.skip("labels abstained")
    for g, cols in sel["oracle_min"].items():
        assert set(cols) <= set(truth["a_safe"])
    assert unit_record["selections"]["oracle_min"]["partition_role"] == \
        "learned_partition"
    assert unit_record["selections"]["oracle_reference"]["partition_role"] == \
        "true_partition"


def test_reference_uses_true_partition_independently(unit_record):
    ref = unit_record["selections"][arms.REFERENCE_ARM]
    assert ref["status"] == "complete"
    assert ref["adjustment_role"] == "true_observed_core"
    rows = unit_record["effects"][arms.REFERENCE_ARM]["ra_standardized"]
    assert any(r["stage"] == "overall" for r in rows)


def test_double_coverage_fields_are_na_not_false(unit_record):
    seen_valid = False
    for name, by_est in unit_record["effects"].items():
        for est, rows in by_est.items():
            for row in rows:
                if row.get("ci_valid"):
                    seen_valid = True
                    assert row["cover_zero"] in (True, False)
                    assert row["reject_zero"] in (True, False)
                    assert row["cover_zero"] != row["reject_zero"]
                else:
                    assert row["cover_true"] is None
                    assert row["cover_zero"] is None
                    assert row["reject_zero"] is None
    assert seen_valid, "at least one valid interval is needed"


def test_targets_are_per_analysis_cluster_and_psm_is_retained(unit_record):
    for name, by_est in unit_record["effects"].items():
        for est in ("ra_standardized", "aipw"):
            for row in by_est[est]:
                if row.get("target_scope"):
                    assert row["target_scope"] == "analysis_cluster_full_members"
        for row in by_est["psm_caliper"]:
            if row.get("target_scope"):
                assert row["target_scope"] == "psm_caliper_retained_members"
            if row["stage"] == "overall" and row.get("estimate") is not None:
                assert row["raw_gap_to_full"] is not None
                assert row["target_shift"] is not None


def test_psm_se_comes_from_shared_outcome_model(unit_record):
    diag = unit_record["estimator_diagnostics"]
    for name, block in diag.items():
        psm = block["psm_caliper"]
        if psm["se_status"] == "candidate_model_se_available":
            V = np.asarray(psm["V"], dtype=float)
            a = np.asarray(psm["weights"], dtype=float)
            se = float(np.sqrt(np.sum(a ** 2 * V)))
            overall = [r for r in unit_record["effects"][name]["psm_caliper"]
                       if r["stage"] == "overall"][0]
            assert se == pytest.approx(overall["se"], rel=1e-9)


def test_feature_selection_uses_observed_core_denominator(unit_record):
    for name, per_cluster in unit_record["feature_selection"].items():
        for g, metrics in per_cluster.items():
            assert metrics["observed_core_size"] > 0
            assert 0.0 <= metrics["observed_core_recall"] <= 1.0
            assert metrics["n_selected"] >= metrics["observed_core_hit"]


def test_u_scenario_unit_runs_and_keeps_u_out_of_methods(labeled_continuous,
                                                         small_group, sources,
                                                         config):
    _gen, b6 = sources
    m6 = load_screening(formal_dgp.STEP6_DIR)
    U, _S, u_meta = formal_dgp.unmeasured_u_for_base(
        labeled_continuous, small_group, LABELED_PARAMS["rho_block"])
    row = small_row("TEST_U", gamma_logor=1.0, has_u=True,
                    methods=list(arms.FIXEDX_ARMS) + list(arms.ORACLE_ARMS))
    record, _m = run_formal.run_unit(
        labeled_continuous, row, small_group,
        {"TEST": {"params": LABELED_PARAMS, "sig_x_spec": 0.6}},
        m6, b6, config, U, u_meta)
    assert record["unit_complete"] is True
    assert arms.REFERENCE_ARM not in record["effects"]      # no reference arm with U
    assert record["latent_confounder_diagnostics"]["not_standardized"] is True
    assert record["observed_shape"]["p"] == LABELED_PARAMS["p_total"]
    # the output distinguishes measured core recall from sufficient adjustment
    for per_cluster in record["feature_selection"].values():
        for metrics in per_cluster.values():
            assert "does not imply sufficient adjustment" in metrics["core_denominator_note"]


def test_real_record_serialises_and_summarises(unit_record, tmp_path):
    """A real record is written atomically as JSON and summarized into the long tables."""
    import summarize_formal as sf

    run_dir = str(tmp_path / "run")
    for param_rep in (1, 2):
        rec = dict(unit_record)
        rec["param_rep"] = param_rep
        rec["point_id"] = "P006"
        rec["unit_key"] = f"P006__p{param_rep:03d}__d01"
        rec["scenario"] = "observed_confounders"
        rec["tier"] = "main"
        rec["report_panels"] = ["main"]
        run_formal.write_atomic(
            os.path.join(run_dir, "units", "P006", f"{rec['unit_key']}.json"), rec)

    loaded = sf.load_units(run_dir)
    assert len(loaded) == 2
    tables = sf.build_tables(loaded)
    for name in ("unit_status", "label_status", "cluster_diagnostics",
                 "truth_effects", "selection_raw", "feature_selection",
                 "effect_raw", "probability_diagnostics"):
        assert tables[name], f"{name} should not be empty"
    # rows for every arm x 3 estimators x (cluster + overall)
    methods = {r["method"] for r in tables["effect_raw"]}
    assert methods == set(ALL_METHODS)
    estimators = {r["estimator"] for r in tables["effect_raw"]}
    assert estimators == {"ra_standardized", "aipw", "psm_caliper"}

    rc = sf.main(["--out-dir", run_dir, "--planned-reps", "10"])
    assert rc == 0
    tables_dir = os.path.join(run_dir, "tables")
    assert os.path.isfile(os.path.join(tables_dir, "summary.csv"))
    with open(os.path.join(tables_dir, "resource_report.json"),
              encoding="utf-8") as handle:
        resource = json.load(handle)
    assert resource["n_units"] == 2
    assert "label" in resource["stage_seconds"]
    assert resource["per_unit_bytes_estimate"] > 0
