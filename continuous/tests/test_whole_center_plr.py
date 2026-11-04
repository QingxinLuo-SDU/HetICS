"""Whole-center cross-fitting and cluster-robust PLR uncertainty."""
import numpy as np
import pytest

from whole_center_plr import assign_center_folds, fit_whole_center_plr


def test_each_center_is_in_exactly_one_fold():
    env = np.repeat(np.arange(5), 4)
    folds = assign_center_folds(env, fold_seed=17)
    assert set(folds) == set(range(5))
    assert set(folds.values()) == {0, 1}
    counts = np.bincount(list(folds.values()), minlength=2)
    assert abs(int(counts[0]) - int(counts[1])) <= 1


def test_fold_assignment_is_deterministic():
    env = np.repeat(np.arange(6), 3)
    assert assign_center_folds(env, 91) == assign_center_folds(env, 91)


def test_all_individuals_of_a_center_share_its_fold():
    env = np.repeat(np.arange(4), 5)
    folds = assign_center_folds(env, 5)
    per_row = np.array([folds[int(e)] for e in env])
    for center in np.unique(env):
        assert len(set(per_row[env == center].tolist())) == 1


def test_plr_recovers_exact_linear_effect_with_empty_adjustment_set():
    env = np.repeat(np.arange(4), 6)
    x0 = np.tile(np.array([-3., -2., -1., 1., 2., 3.]), 4)
    y = 2.0 * x0
    result = fit_whole_center_plr(x0, y, np.empty((24, 0)), env, fold_seed=3)
    assert result.status == "ok"
    assert result.theta == pytest.approx(2.0, abs=1e-10)
    assert result.variance >= 0.0


def test_one_center_cannot_fall_back_to_row_level_split():
    env = np.zeros(8, dtype=int)
    result = fit_whole_center_plr(np.arange(8.), np.arange(8.),
                                  np.empty((8, 0)), env, fold_seed=3)
    assert result.status == "unavailable"
    assert result.error_code == "fold_not_runnable"


def test_training_and_validation_centers_never_intersect():
    env = np.repeat(np.arange(6), 4)
    x0 = np.tile(np.array([-2., -1., 1., 2.]), 6)
    y = 1.5 * x0
    result = fit_whole_center_plr(x0, y, np.empty((24, 0)), env, fold_seed=11)
    assert result.status == "ok"
    assert set(result.fold_by_center) == set(range(6))
    for fold in (0, 1):
        train = {c for c, f in result.fold_by_center.items() if f != fold}
        valid = {c for c, f in result.fold_by_center.items() if f == fold}
        assert train and valid
        assert not (train & valid)


def test_center_scores_cover_every_center_and_sum_to_zero():
    env = np.repeat(np.arange(4), 6)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(24)
    y = 0.8 * x0 + 0.1 * rng.standard_normal(24)
    result = fit_whole_center_plr(x0, y, np.empty((24, 0)), env, fold_seed=7)
    assert set(result.center_scores) == set(range(4))
    # theta solves the moment condition, so the center scores sum to zero
    assert sum(result.center_scores.values()) == pytest.approx(0.0, abs=1e-8)


def test_sandwich_se_and_two_interval_flavours_are_both_reported():
    env = np.repeat(np.arange(6), 8)
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(48)
    y = 1.2 * x0 + 0.3 * rng.standard_normal(48)
    result = fit_whole_center_plr(x0, y, np.empty((48, 0)), env, fold_seed=2)
    assert result.status == "ok"
    assert result.se == pytest.approx(np.sqrt(result.variance))
    # the t interval (M - 1 = 5 df) is wider than the normal interval; both are reported
    assert (result.ci_t[1] - result.ci_t[0]) > (result.ci_z[1] - result.ci_z[0])
    assert result.ci_z[0] < result.theta < result.ci_z[1]


def test_zero_variance_treatment_is_unavailable_not_crash():
    env = np.repeat(np.arange(4), 4)
    x0 = np.ones(16)          # zero after residualization, so the denominator is zero
    y = np.arange(16.)
    result = fit_whole_center_plr(x0, y, np.empty((16, 0)), env, fold_seed=4)
    assert result.status == "unavailable"
    assert result.error_code in ("theta_nonfinite", "nuisance_failed")


def test_adjustment_set_removes_confounding_bias():
    """W confounds X and Y: adjusted theta is close to 1.0, the unadjusted one is biased upward."""
    rng = np.random.default_rng(3)
    env = np.repeat(np.arange(8), 40)
    w = rng.standard_normal((320, 1))
    x0 = 1.5 * w[:, 0] + rng.standard_normal(320)
    y = 1.0 * x0 + 2.0 * w[:, 0] + 0.5 * rng.standard_normal(320)
    adjusted = fit_whole_center_plr(x0, y, w, env, fold_seed=8)
    naive = fit_whole_center_plr(x0, y, np.empty((320, 0)), env, fold_seed=8)
    assert adjusted.status == "ok"
    assert abs(adjusted.theta - 1.0) < 0.1
    assert naive.theta - 1.0 > 0.3


def test_module_never_imports_row_level_kfold():
    import inspect

    import whole_center_plr as mod
    src = inspect.getsource(mod)
    assert "KFold" not in src
    assert "n_folds=5" not in src
