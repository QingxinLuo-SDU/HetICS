"""Two-level seeds and the continuous simulation design."""
import pytest

import step6x_config_v4 as cfg4
from step6x_config_v4 import build_combos_v4
from v4_seeds import seeds_for_key
from v4_types import SeedKey


def test_parameter_seed_is_shared_by_100_data_instances():
    keys = [SeedKey("b2-3", "A05", 0, j, "A05") for j in range(100)]
    bundles = [seeds_for_key(k) for k in keys]
    assert len({b.param_seed for b in bundles}) == 1
    assert len({b.data_seed for b in bundles}) == 100


def test_delta_points_share_common_random_numbers_with_anchor():
    anchor = seeds_for_key(SeedKey("b2-3", "A05", 2, 7, "A05"))
    delta = seeds_for_key(SeedKey("b2-3", "G06", 2, 7, "A05"))
    assert (anchor.param_seed, anchor.data_seed) == (delta.param_seed, delta.data_seed)
    assert (anchor.fold_seed, anchor.algorithm_seed) == (delta.fold_seed,
                                                         delta.algorithm_seed)


def test_param_seeds_differ_across_parameter_instances():
    seeds = {seeds_for_key(SeedKey("b2-3", "A05", p, 0, "A05")).param_seed
             for p in range(10)}
    assert len(seeds) == 10


def test_four_streams_are_independent_within_one_realization():
    b = seeds_for_key(SeedKey("b2-3", "A05", 3, 11, "A05"))
    assert len({b.param_seed, b.data_seed, b.fold_seed, b.algorithm_seed}) == 4


def test_beta_batches_do_not_collide():
    a = seeds_for_key(SeedKey("b2-3", "A05", 0, 0, "A05"))
    b = seeds_for_key(SeedKey("b2-24", "A05", 0, 0, "A05"))
    assert a.data_seed != b.data_seed


def test_seeds_are_deterministic():
    key = SeedKey("b2-3", "E03", 4, 42, "E03")
    assert seeds_for_key(key) == seeds_for_key(key)


@pytest.mark.usefixtures("legacy_beta_environment")
def test_v4_grid_is_58_main_plus_7_sensitivity():
    combos = build_combos_v4("b2-3")
    assert len(combos) == 65
    assert sum(c["analysis_role"] == "main" for c in combos) == 58
    assert [c["dgp_params"]["delta_spec"] for c in combos
            if c["analysis_role"] == "delta_spec_sensitivity"] == [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
    assert all(c["dgp_params"]["delta_spec"] == 0.0
               for c in combos if c["analysis_role"] == "main")


@pytest.mark.usefixtures("legacy_beta_environment")
def test_delta_rows_carry_anchor_seed_group():
    combos = build_combos_v4("b2-3")
    assert all(c["seed_group_id"] == "A05"
               for c in combos if c["sweep"] == "G")
    assert all(c["seed_group_id"] == c["combo_id"]
               for c in combos if c["sweep"] != "G")


def test_repetition_contract_is_ten_by_hundred_and_two_folds():
    assert (cfg4.N_PARAM_SEEDS, cfg4.N_DATA_SEEDS_PER_PARAM) == (10, 100)
    assert cfg4.N_FOLDS == 2
    assert cfg4.PRIMARY_METHOD == "dual_icp"
    assert cfg4.OUTPUT_VERSION.startswith("scenA-cont-wccf-reml-4.3-")


def test_method_arms_are_the_reported_ones():
    import step6x_config as cfg3
    assert cfg3.METHODS == ("xside_icp", "dual_icp", "yside_icp", "oracle_min")
    assert cfg3.ICP_METHODS == ("xside_icp", "dual_icp", "yside_icp")


def test_version_bump_makes_old_chunks_unrecognisable():
    import step6x_config_v4 as cfg4
    assert "4.1" not in cfg4.OUTPUT_VERSION
    assert "4.0" not in cfg4.OUTPUT_VERSION
