"""tests/test_perf_e3_loglik_reuse.py — E3 (reuse of the per-individual log-likelihood of the accepted candidate).

`classification_objective` recomputes `eta` and `outcome[:,None]*eta - logaddexp(0,eta)` with `proposed`, while
the M step has just computed the same array with the same parameters (they differ only in `pi`, which enters
neither the prediction nor the per-individual log-likelihood). The same intermediate array is reused; the two
reductions (weighted sum and per-center `np.add.at`) and the formula of the objective are unchanged.

Covered: unchanged public return shapes, bitwise equality of both reductions, use of the accepted array only after
backtracking, no array on failure, unchanged error semantics, and the four callers (fixed-label refit, CEM outer
loop, warmup, full EM).
"""

from __future__ import annotations

import numpy as np
import pytest

from bniseam_test import model
from bniseam_test.cem_objectives import (
    _classification_objective_from_loglik,
    classification_objective,
    dense_center_codes,
)
from bniseam_test.cem_optimizer import refit_fixed_labels, zero_parameters_from_labels
from bniseam_test.cem_types import CEMConfig
from bniseam_test.model import (
    FitConfig,
    _hard_label_m_step_with_loglik,
    generalized_m_step,
    hard_label_m_step,
    linear_predictor,
)
from bniseam_test.multiview_types import ObservedDataView


def _bits(value) -> list[int]:
    return np.asarray(value, dtype=np.float64).reshape(-1).view(np.uint64).tolist()


def _fixture(n_centers: int = 6, per_center: int = 120, p: int = 8, k: int = 2):
    rng = np.random.default_rng(20260919)
    n = n_centers * per_center
    codes = np.repeat(np.arange(n_centers), per_center).astype(np.int64)
    features = rng.standard_normal((n, p))
    labels = np.array([i % k for i in range(n_centers)], dtype=np.int64)
    signal = features[:, :3] @ np.array([0.8, -0.6, 0.4]) + 0.5 * labels[codes]
    treatment = (rng.random(n) < 0.5).astype(float)
    probability = 1.0 / (1.0 + np.exp(-(signal + 0.7 * treatment)))
    outcome = (rng.random(n) < probability).astype(float)
    observed = ObservedDataView(features=features, treatment=treatment, outcome=outcome,
                                center_codes=codes,
                                centers=np.arange(n_centers, dtype=np.int64))
    return observed, labels, k


def _fit_config(max_iter: int = 12, step_size: float = 1.0,
                tolerance: float = 0.0):
    """Tolerance 0 by default, as in the E2 test: the number of steps does not depend on the data."""
    return FitConfig(support_size=4, step_size=step_size, max_iter=max_iter,
                     warmup_steps=0, parameter_tolerance=tolerance,
                     objective_relative_tolerance=tolerance)


def _reference_row_loglik(parameters, observed):
    eta = linear_predictor(parameters, observed.treatment, observed.features)
    return observed.outcome[:, None] * eta - np.logaddexp(0.0, eta)


# --------------------------------------------------------------------------- #
# Core: both paths give bitwise identical objective values
# --------------------------------------------------------------------------- #

def test_e3_objective_from_loglik_is_bitwise_identical():
    observed, labels, k = _fixture()
    parameters = zero_parameters_from_labels(observed, labels, k)
    parameters.beta[0, :3] = [0.4, -0.3, 0.2]
    parameters.beta[1, 2:5] = [0.1, 0.5, -0.2]
    parameters.gamma = 0.6
    parameters.intercepts = np.array([0.2, -0.1])
    codes = dense_center_codes(observed)

    expected = classification_objective(observed, parameters, labels)
    actual = _classification_objective_from_loglik(
        observed, parameters, labels, _reference_row_loglik(parameters, observed), codes)
    assert _bits(actual) == _bits(expected)


def test_e3_objective_uses_the_updated_pi():
    """The log(pi) term uses the updated pi, not the pi of the parameters used for loglik."""
    observed, labels, k = _fixture()
    parameters = zero_parameters_from_labels(observed, labels, k)
    codes = dense_center_codes(observed)
    row_loglik = _reference_row_loglik(parameters, observed)

    shifted = parameters.clone()
    shifted.pi = np.array([0.7, 0.3])
    assert _bits(_classification_objective_from_loglik(
        observed, shifted, labels, row_loglik, codes)) == _bits(
        classification_objective(observed, shifted, labels))
    # two different pi give different objective values (the term is not ignored)
    assert _classification_objective_from_loglik(
        observed, shifted, labels, row_loglik, codes) != \
        _classification_objective_from_loglik(
            observed, parameters, labels, row_loglik, codes)


def test_e3_m_step_returns_the_accepted_candidates_loglik():
    """The returned array is the per-individual log-likelihood of the accepted candidate, bit by bit."""
    observed, labels, k = _fixture()
    config = _fit_config()
    parameters = zero_parameters_from_labels(observed, labels, k)
    codes = dense_center_codes(observed)

    proposed, info, loglik = _hard_label_m_step_with_loglik(
        parameters, labels, observed.treatment, observed.features, observed.outcome,
        codes, config)
    assert not info.backtracking_failed
    assert loglik is not None
    # proposed and the accepted candidate differ only in pi, which does not enter loglik
    assert _bits(loglik) == _bits(_reference_row_loglik(proposed, observed))


def test_e3_backtracking_returns_the_accepted_not_the_last_rejected(monkeypatch):
    """A real backtracking: the first step size is rejected and the second accepted; the array belongs to the latter."""
    observed, labels, k = _fixture()
    config = _fit_config()
    parameters = zero_parameters_from_labels(observed, labels, k)
    codes = dense_center_codes(observed)
    original = model._regression_proposal

    def first_attempt_is_bad(params, resp, treatment, features, outcome,
                             center_codes, cfg, step_size):
        if step_size == cfg.step_size:                 # attempt 0: deliberately worse
            bad = params.clone()
            bad.gamma = float(params.gamma - 50.0)
            return bad
        return original(params, resp, treatment, features, outcome,
                        center_codes, cfg, step_size)

    monkeypatch.setattr(model, "_regression_proposal", first_attempt_is_bad)
    proposed, info, loglik = _hard_label_m_step_with_loglik(
        parameters, labels, observed.treatment, observed.features, observed.outcome,
        codes, config)

    assert info.backtracking_count >= 1, "no backtracking was triggered"
    assert not info.backtracking_failed
    assert _bits(loglik) == _bits(_reference_row_loglik(proposed, observed))
    # and differs from the array of the rejected candidate
    rejected = parameters.clone()
    rejected.gamma = float(parameters.gamma - 50.0)
    assert _bits(loglik) != _bits(_reference_row_loglik(rejected, observed))


def test_e3_no_loglik_when_backtracking_fails(monkeypatch):
    """When every backtracking step fails no array is returned, in particular not that of the last rejected attempt."""
    observed, labels, k = _fixture()
    config = _fit_config()
    parameters = zero_parameters_from_labels(observed, labels, k)
    codes = dense_center_codes(observed)
    responsibilities = np.eye(k, dtype=float)[labels]

    def always_bad(params, resp, treatment, features, outcome, center_codes,
                   cfg, step_size):
        bad = params.clone()
        bad.gamma = float(params.gamma - 50.0)
        return bad

    monkeypatch.setattr(model, "_regression_proposal", always_bad)
    _proposed, info, loglik = model._generalized_m_step_with_loglik(
        parameters, responsibilities, observed.treatment, observed.features,
        observed.outcome, codes, config)
    assert info.backtracking_failed
    assert loglik is None


def test_e3_refit_falls_back_when_no_loglik_available():
    """Failure path unchanged: raises, with status m_step_backtracking_failed."""
    observed, labels, k = _fixture()
    config = _fit_config(max_iter=5)
    initial = zero_parameters_from_labels(observed, labels, k)
    import bniseam_test.model as m
    original = m._regression_proposal
    try:
        m._regression_proposal = lambda params, *a, **kw: _worse(params)
        parameters, _trace, status = refit_fixed_labels(observed, initial, labels, config)
    finally:
        m._regression_proposal = original
    assert parameters is None
    assert status == "m_step_backtracking_failed"


def _worse(params):
    bad = params.clone()
    bad.gamma = float(params.gamma - 50.0)
    return bad


# --------------------------------------------------------------------------- #
# Public interface and error semantics
# --------------------------------------------------------------------------- #

def test_e3_public_m_steps_still_return_two_items():
    observed, labels, k = _fixture()
    config = _fit_config()
    parameters = zero_parameters_from_labels(observed, labels, k)
    responsibilities = np.eye(k, dtype=float)[labels]
    codes = dense_center_codes(observed)

    public = generalized_m_step(parameters, responsibilities, observed.treatment,
                                observed.features, observed.outcome, codes, config)
    assert isinstance(public, tuple) and len(public) == 2
    private = model._generalized_m_step_with_loglik(
        parameters, responsibilities, observed.treatment, observed.features,
        observed.outcome, codes, config)
    assert len(private) == 3
    assert np.array_equal(public[0].beta, private[0].beta)
    assert _bits(public[1].q_reg_new) == _bits(private[1].q_reg_new)

    public_hard = hard_label_m_step(parameters, labels, observed.treatment,
                                    observed.features, observed.outcome, codes, config)
    assert isinstance(public_hard, tuple) and len(public_hard) == 2


def test_e3_step_info_carries_no_big_array():
    """The n x K array does not enter StepInfo (which goes into every trace row and is serialized)."""
    observed, labels, k = _fixture()
    config = _fit_config()
    parameters = zero_parameters_from_labels(observed, labels, k)
    codes = dense_center_codes(observed)
    _proposed, info, _loglik = _hard_label_m_step_with_loglik(
        parameters, labels, observed.treatment, observed.features, observed.outcome,
        codes, config)
    for value in vars(info).values():
        assert not isinstance(value, np.ndarray), "StepInfo contains an array"


@pytest.mark.parametrize("mutate,expected", [
    ("bad_label_shape", "labels must contain one value per center"),
    ("label_out_of_range", "invalid hard labels"),
    ("nonpositive_pi", "classification weights must be positive"),
    ("wrong_loglik_shape", "row_loglik does not match the observed data"),
])
def test_e3_error_semantics_preserved(mutate, expected):
    observed, labels, k = _fixture()
    parameters = zero_parameters_from_labels(observed, labels, k)
    codes = dense_center_codes(observed)
    row_loglik = _reference_row_loglik(parameters, observed)
    if mutate == "bad_label_shape":
        labels = labels[:-1]
    elif mutate == "label_out_of_range":
        labels = labels.copy(); labels[0] = 9
    elif mutate == "nonpositive_pi":
        parameters = parameters.clone(); parameters.pi = np.array([1.0, 0.0])
    elif mutate == "wrong_loglik_shape":
        row_loglik = row_loglik[:-1]
    with pytest.raises((ValueError, FloatingPointError), match=expected):
        _classification_objective_from_loglik(observed, parameters, labels,
                                              row_loglik, codes)


def test_e3_nonfinite_loglik_still_raises():
    observed, labels, k = _fixture()
    parameters = zero_parameters_from_labels(observed, labels, k)
    codes = dense_center_codes(observed)
    row_loglik = _reference_row_loglik(parameters, observed).copy()
    row_loglik[0, 0] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite component log-likelihood"):
        _classification_objective_from_loglik(observed, parameters, labels,
                                              row_loglik, codes)


# --------------------------------------------------------------------------- #
# The four callers still work (E3 changes only the fixed-label path)
# --------------------------------------------------------------------------- #

def test_e3_fixed_label_refit_still_converges():
    """Uses the real tolerance; only completion is asserted, not a number of steps."""
    observed, labels, k = _fixture()
    parameters, trace, status = refit_fixed_labels(
        observed, zero_parameters_from_labels(observed, labels, k), labels,
        _fit_config(max_iter=30, tolerance=1e-5))
    assert status in {"converged", "max_iter_eligible"}
    assert parameters is not None and len(trace) > 1


def test_e3_cem_outer_loop_still_works():
    from bniseam_test.cem_optimizer import run_cem_candidate
    from bniseam_test.cem_types import CandidateSeed
    observed, labels, k = _fixture()
    result = run_cem_candidate(
        observed, CandidateSeed(candidate_id=0, method="fixed_start", labels=labels,
                                seed=None),
        _fit_config(max_iter=10), CEMConfig(k_components=2, max_outer_iter=2))
    assert result.status in {"converged", "max_outer_iter_eligible",
                             "terminal_refit_max_iter_uncertified",
                             "terminal_refit_failed"}


def test_e3_warmup_still_works():
    observed, labels, k = _fixture()
    parameters = zero_parameters_from_labels(observed, labels, k)
    responsibilities = np.eye(k, dtype=float)[labels]
    config = FitConfig(support_size=4, step_size=1.0, max_iter=5, warmup_steps=3)
    updated, trajectory = model.warmup_fixed_responsibilities(
        parameters, responsibilities, observed.treatment, observed.features,
        observed.outcome, dense_center_codes(observed), config)
    assert len(trajectory) == 3 and updated is not parameters


def test_e3_full_em_fit_still_works():
    observed, labels, k = _fixture()
    parameters = zero_parameters_from_labels(observed, labels, k)
    config = FitConfig(support_size=4, step_size=1.0, max_iter=5, warmup_steps=0)
    result = model.fit(parameters, observed.treatment, observed.features,
                       observed.outcome, dense_center_codes(observed),
                       int(observed.centers.size), config)
    assert len(result.trajectory) >= 2
    assert result.stop_reason in {"max_iter", "parameter_and_objective_tolerance",
                                  "backtracking_failed"}


def test_e3_linear_predictor_call_count_matches_the_reused_plan(monkeypatch):
    """Guard: the number of full passes over the data per iteration after E2 and E3.

    Per iteration: one gradient + one candidate q (two more per extra backtracking step); the classification
    objective reuses the array; q_reg_old only in the first iteration; plus the initial classification_objective
    before the loop.
    """
    observed, labels, k = _fixture()
    config = _fit_config(max_iter=20)
    initial = zero_parameters_from_labels(observed, labels, k)

    calls = {"n": 0}
    original = model.linear_predictor

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "linear_predictor", counted)
    import bniseam_test.cem_objectives as objectives
    monkeypatch.setattr(objectives, "linear_predictor", counted)

    steps: list = []
    original_step = model._hard_label_m_step_with_loglik

    def spy(*args, **kwargs):
        result, info, loglik = original_step(*args, **kwargs)
        steps.append(info)
        return result, info, loglik

    import bniseam_test.cem_optimizer as optimizer
    monkeypatch.setattr(optimizer, "_hard_label_m_step_with_loglik", spy)
    refit_fixed_labels(observed, initial, labels, config)

    n_steps = len(steps)
    backtracks = sum(int(info.backtracking_count) for info in steps)
    expected = 2 + 2 * n_steps + 2 * backtracks
    assert calls["n"] == expected, (
        f"linear_predictor called {calls['n']} times, {expected} expected after E2 and E3"
        f" ({n_steps} steps, {backtracks} extra backtracking steps)")
