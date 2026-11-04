"""Design table checks: 86 rows, 47 structures, arms, arithmetic and pairing identities (no DGP)."""

from __future__ import annotations

import design_table as dt
import pytest


@pytest.fixture(scope="module")
def rows():
    return dt.load_legacy_rows()


def test_run_table_checksum_matches_frozen_design():
    assert dt.sha256_of(dt.RUN_TABLE_PATH) == dt.RUN_TABLE_SHA256


def test_static_checks_all_pass(rows):
    checks = dt.static_checks(rows)
    assert checks["problems"] == []
    assert checks["status"] == "ok"


def test_counts(rows):
    checks = dt.static_checks(rows)
    assert checks["n_rows"] == 86
    assert checks["n_structures"] == 47
    assert checks["calibration_member"] == 26
    assert checks["has_unmeasured_U"] == 20
    assert checks["method_slots_per_rep"] == 334
    assert checks["planned_estimator_attempts_total"] == 1_002_000
    assert checks["first_batch_units"] == 860
    assert checks["first_batch_attempts"] == 10_020
    assert checks["remainder_if_first_batch_counts"]["units"] == 85_140
    assert checks["remainder_if_first_batch_counts"]["attempts"] == 991_980


def test_structure_group_sizes_match_sop(rows):
    """Structure sizes: 28 single + 12 pairs + 6 triples + 1 group of 16."""
    checks = dt.static_checks(rows)
    assert checks["structure_group_sizes"] == {1: 28, 2: 12, 3: 6, 16: 1}
    assert 28 * 1 + 12 * 2 + 6 * 3 + 1 * 16 == 86
    assert 28 + 12 + 6 + 1 == 47


def test_execution_grouping(rows):
    by_n = {}
    for row in rows:
        by_n[int(row["n_methods"])] = by_n.get(int(row["n_methods"]), 0) + 1
    assert by_n == {4: 60, 5: 14, 2: 12}
    assert 40 * 4 + 14 * 5 + 12 * 2 + 20 * 4 == 334


def test_calibration_rows_execute_reference(rows):
    for row in rows:
        methods = set(dt.methods_of(row))
        if row["calibration_member"] == "true":
            assert "oracle_reference" in methods, row["point_id"]
            assert "dual_icp" in methods
        else:
            assert "oracle_reference" not in methods, row["point_id"]


def test_u_rows_do_not_add_a_thirteenth_arm(rows):
    for row in rows:
        if row["has_unmeasured_U"] != "true":
            continue
        methods = set(dt.methods_of(row))
        assert len(methods) == 4
        assert "oracle_reference" not in methods
        assert float(row["u_shared_loading"]) == 0.7
        assert float(row["u_noise_loading"]) == 0.3
        assert float(row["lambda_UX"]) == 0.5
        assert float(row["lambda_UY"]) == 0.5


def test_paired_points_share_base_seeds(rows):
    """U <-> no-U and calibration gamma0 <-> +1 pairs share param/data seeds (how the pairing is implemented)."""
    mapping = dt.structure_of(rows)
    by_id = {r["point_id"]: r for r in rows}
    checked = 0
    for row in rows:
        pair = row["pair_reference_point_id"].strip()
        if not pair:
            continue
        assert mapping[row["point_id"]] == mapping[pair]
        a = dt.base_seeds(mapping[row["point_id"]], 7, 3)
        b = dt.base_seeds(mapping[pair], 7, 3)
        assert a == b
        checked += 1
    assert checked == 32
    # gamma changes the binary layer only and is not part of the structure identity
    assert by_id["P001"]["gamma_logor"] != by_id["P006"]["gamma_logor"]
    assert mapping["P001"] == mapping["P006"] == mapping["U001"]


def test_algorithm_seed_is_per_analysis_identity(rows):
    """Algorithm seeds derive from point_id, so different gamma or U scenarios learn their labels anew."""
    a = dt.algorithm_seed_base("P001", 3, 2)
    b = dt.algorithm_seed_base("P006", 3, 2)
    c = dt.algorithm_seed_base("U001", 3, 2)
    assert len({a, b, c}) == 3


def test_u_epsilon_stream_is_structure_scoped(rows):
    mapping = dt.structure_of(rows)
    s1 = dt.u_epsilon_seed(mapping["U001"], 4, 1)
    s2 = dt.u_epsilon_seed(mapping["U006"], 4, 1)
    assert s1 == s2                      # the U points of one structure share the same U
    assert s1 != dt.u_epsilon_seed(mapping["U009"], 4, 1)
    assert s1 != dt.base_seeds(mapping["U001"], 4, 1)["data_seed"]


def test_base_groups_first_batch_shape(rows):
    groups = dt.build_base_groups(rows, list(range(1, 11)), [1])
    assert len(groups) == 47 * 10
    covered = sum(len(g["point_ids"]) for g in groups)
    assert covered == 86 * 10 == 860


def test_dgp_params_reject_non_unit_continuous_gamma(rows):
    bad = dict(rows[0])
    bad["gamma_XY_continuous"] = "2"
    with pytest.raises(ValueError):
        dt.dgp_params(bad)


def test_point_subset_does_not_change_structure_seeds(rows):
    """A --points subset run gets the same structure_id and seeds as a full run.

    (Collapsing the structures after filtering the rows would name STRUCT_ after the first point of the subset
    and change the seeds.)
    """
    full = dt.build_base_groups(rows, [1], [1])
    anchor = [g for g in full if g["structure_id"] == "STRUCT_P001"][0]
    assert len(anchor["point_ids"]) == 16
    keep = {"P006"}
    subset = [{**g, "point_ids": [p for p in g["point_ids"] if p in keep]}
              for g in full if any(p in keep for p in g["point_ids"])]
    assert len(subset) == 1
    assert subset[0]["structure_id"] == "STRUCT_P001"
    assert subset[0]["param_seed"] == anchor["param_seed"]
    assert subset[0]["data_seed"] == anchor["data_seed"]
    assert subset[0]["u_epsilon_seed"] == anchor["u_epsilon_seed"]
