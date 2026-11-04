"""Calibration design and independent parameter and data seed identities."""
import numpy as np
import pytest

import hypothesis_calibration_config as hcc
import hypothesis_calibration_seeds as hcs
import step6x_config as cfg
import step6x_config_v4 as cfg4
from v4_seeds import seeds_for_key
from v4_types import SeedKey

pytestmark = pytest.mark.usefixtures("legacy_beta_environment")


def test_design_is_exactly_seven_center_levels_times_two_gammas():
    points = hcc.build_calibration_points()
    assert len(points) == 14
    assert hcc.CENTERS_PER_CLUSTER == (3, 4, 5, 6, 8, 10, 12)
    assert hcc.GAMMA_VALUES == (0.0, 0.035)
    assert sorted({(p["centers_per_cluster"], p["gamma_true"]) for p in points}) == \
        sorted([(m, g) for m in hcc.CENTERS_PER_CLUSTER for g in hcc.GAMMA_VALUES])


def test_total_centers_is_three_times_centers_per_cluster():
    for p in hcc.build_calibration_points():
        assert p["n_centers"] == 3 * p["centers_per_cluster"]
        assert p["dgp_params"]["n_centers"] == p["n_centers"]
        assert p["dgp_params"]["K"] == 3 == hcc.K_TRUE
    assert [p["n_centers"] for p in hcc.build_calibration_points()
            if p["gamma_true"] == 0.0] == [9, 12, 15, 18, 24, 30, 36]


def test_every_center_has_exactly_ten_thousand_people():
    for p in hcc.build_calibration_points():
        assert p["dgp_params"]["n_per_center"] == 10000
        assert p["dgp_params"]["n_per_center_var"] == 0.0


def test_delta_spec_is_zero_and_beta_batch_is_b2_3():
    for p in hcc.build_calibration_points():
        assert p["dgp_params"]["delta_spec"] == 0.0
        assert p["beta_batch"] == "b2-3" == hcc.BETA_BATCH


def test_repetition_structure_is_ten_param_seeds_by_hundred_data_seeds():
    assert hcc.N_PARAM_SEEDS == 10
    assert hcc.N_DATA_SEEDS_PER_PARAM == 100
    for p in hcc.build_calibration_points():
        assert p["n_param_seeds"] == 10
        assert p["n_data_seeds_per_param"] == 100


def test_point_ids_are_unique_and_stable_strings():
    points = hcc.build_calibration_points()
    ids = [p["point_id"] for p in points]
    assert len(set(ids)) == 14
    assert "M03_G0" in ids and "M04_G0035" in ids and "M12_G0035" in ids
    # repeated construction gives the same order and identities
    assert ids == [p["point_id"] for p in hcc.build_calibration_points()]


def test_only_the_six_agreed_fields_differ_from_the_anchor():
    """Only these six parameters differ from the anchor defaults."""
    base = dict(cfg.BASE_PARAMS)
    overridden = {"n_centers", "n_per_center", "n_per_center_var", "gamma_XY",
                  "beta_mag_lo", "beta_mag_hi"}
    for p in hcc.build_calibration_points():
        pr = p["dgp_params"]
        assert set(pr) == set(base)
        for name, value in base.items():
            if name in overridden:
                continue
            assert pr[name] == value, f"{p['point_id']} changes {name}"


def test_points_are_deep_copies_and_cannot_leak_into_the_anchor():
    points = hcc.build_calibration_points()
    points[0]["dgp_params"]["p_total"] = -1
    assert cfg.BASE_PARAMS["p_total"] != -1
    assert hcc.build_calibration_points()[0]["dgp_params"]["p_total"] != -1


def test_two_arms_and_oracle_is_not_ranked():
    assert hcc.ARMS == ("dual_icp", "oracle_reference")
    assert hcc.PRIMARY_ARM == "dual_icp"
    assert hcc.DIAGNOSTIC_ARM == "oracle_reference"
    assert hcc.ARM_ROLES["oracle_reference"] == "diagnostic"
    assert hcc.ARM_ROLES["dual_icp"] == "primary"
    assert hcc.RANKED_ARMS == ("dual_icp",)
    assert hcc.DIAGNOSTIC_ARM not in hcc.RANKED_ARMS
    assert set(hcc.ARM_CHANNEL) == set(hcc.RANKED_ARMS)
    assert hcc.ARM_CHANNEL["dual_icp"] == "dual"


def test_output_version_is_separate_from_the_regular_simulation():
    assert hcc.OUTPUT_VERSION == "scenA-cont-wccf-reml-testcal-2.0-b1-125"
    assert "1.0" not in hcc.OUTPUT_VERSION.rsplit("-", 2)[0]
    assert hcc.OUTPUT_VERSION != cfg4.OUTPUT_VERSION
    assert not cfg4.OUTPUT_VERSION.startswith(hcc.OUTPUT_VERSION)


def test_wrong_beta_batch_environment_is_rejected():
    with pytest.raises(ValueError):
        hcc.build_calibration_points(beta_batch="b2-24")


def _point(m, gamma):
    return next(p for p in hcc.build_calibration_points()
                if p["centers_per_cluster"] == m and p["gamma_true"] == gamma)


def test_gamma_does_not_enter_the_seed_derivation():
    """The two gamma levels of a level share their seeds."""
    for m in hcc.CENTERS_PER_CLUSTER:
        for param_index, data_index in ((0, 0), (3, 47), (9, 49)):
            a = hcs.calibration_seeds(_point(m, 0.0), param_index, data_index)
            b = hcs.calibration_seeds(_point(m, hcc.GAMMA_VALUES[1]), param_index, data_index)
            assert a == b


def test_changing_m_changes_the_data_stream():
    a = hcs.calibration_seeds(_point(4, 0.0), 0, 0)
    b = hcs.calibration_seeds(_point(5, 0.0), 0, 0)
    assert a.data_seed != b.data_seed
    assert a.param_seed != b.param_seed


def test_changing_param_index_changes_both_layers():
    a = hcs.calibration_seeds(_point(10, hcc.GAMMA_VALUES[1]), 0, 0)
    b = hcs.calibration_seeds(_point(10, hcc.GAMMA_VALUES[1]), 1, 0)
    assert a.param_seed != b.param_seed
    assert a.data_seed != b.data_seed


def test_changing_data_index_changes_data_but_not_parameter_instance():
    a = hcs.calibration_seeds(_point(10, hcc.GAMMA_VALUES[1]), 2, 0)
    b = hcs.calibration_seeds(_point(10, hcc.GAMMA_VALUES[1]), 2, 1)
    assert a.param_seed == b.param_seed          # same parameter instance
    assert a.data_seed != b.data_seed
    assert a.fold_seed != b.fold_seed
    assert a.algorithm_seed != b.algorithm_seed


def test_four_streams_are_distinct_within_one_key():
    s = hcs.calibration_seeds(_point(6, 0.0), 4, 12)
    assert len({s.param_seed, s.data_seed, s.fold_seed, s.algorithm_seed}) == 4


def test_calibration_namespace_never_collides_with_the_regular_design():
    """Calibration seeds never coincide with those of the 65-point design."""
    regular = set()
    for combo_id in ("A05", "E00", "E01", "E02"):
        for param_index in range(hcc.N_PARAM_SEEDS):
            for data_index in (0, 1, 49):
                bundle = seeds_for_key(SeedKey(
                    beta_batch="b2-3", combo_id=combo_id,
                    param_index=param_index, data_index=data_index,
                    seed_group_id=combo_id))
                regular.update((bundle.param_seed, bundle.data_seed,
                                bundle.fold_seed, bundle.algorithm_seed))
    for p in hcc.build_calibration_points():
        for param_index in range(hcc.N_PARAM_SEEDS):
            for data_index in (0, 1, 49):
                b = hcs.calibration_seeds(p, param_index, data_index)
                assert not regular & {b.param_seed, b.data_seed,
                                      b.fold_seed, b.algorithm_seed}


def test_seed_identity_row_records_semantic_identity_and_integers():
    point = _point(12, hcc.GAMMA_VALUES[1])
    bundle = hcs.calibration_seeds(point, 7, 45)
    row = hcs.calibration_seed_row(point, 7, 45, bundle)
    assert row["point_id"] == point["point_id"]
    assert row["centers_per_cluster"] == 12
    assert row["gamma_true"] == hcc.GAMMA_VALUES[1]
    assert row["param_index"] == 7 and row["data_index"] == 45
    for name in ("param_seed", "data_seed", "fold_seed", "algorithm_seed"):
        assert row[name] == getattr(bundle, name)
        assert isinstance(row[name], int)


def test_seeds_are_deterministic_across_calls():
    point = _point(8, 0.0)
    assert hcs.calibration_seeds(point, 1, 2) == hcs.calibration_seeds(point, 1, 2)


def test_seed_values_fit_in_uint32():
    for p in hcc.build_calibration_points()[:4]:
        b = hcs.calibration_seeds(p, 0, 0)
        for value in (b.param_seed, b.data_seed, b.fold_seed, b.algorithm_seed):
            assert 0 <= value < 2 ** 32
            assert not isinstance(value, np.integer)
