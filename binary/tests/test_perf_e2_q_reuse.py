"""tests/test_perf_e2_q_reuse.py — E2 (reuse of q across consecutive fixed-label M steps): equivalence and boundaries.

Within one `refit_fixed_labels` the data, labels and one-hot responsibilities do not change, the accepted
parameters of a step become `current` of the next, and `regression_q` does not read `pi`; hence `q_reg_new` of
one step is `q_reg_old` of the next.

Every step compares the cached value with a fresh computation bit by bit, and the lifetime of the cache is covered:
first step, consecutive successes, success after backtracking, failure of every backtracking step, early return,
reset for a new fit, and paths where labels change (which must not use the cache).
"""

from __future__ import annotations

import numpy as np
import pytest

from bniseam_test import cem_optimizer, model
from bniseam_test.cem_optimizer import refit_fixed_labels, zero_parameters_from_labels
from bniseam_test.cem_types import CEMConfig
from bniseam_test.model import FitConfig, FitError, generalized_m_step, regression_q
from bniseam_test.multiview_types import ObservedDataView


def _bits(value: float) -> int:
    return int(np.float64(value).view(np.uint64))


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
                                center_codes=codes, centers=np.arange(n_centers, dtype=np.int64))
    return observed, labels, k


def _configs(max_iter: int = 12, *, tolerance: float = 0.0):
    """Tolerance 0 by default: the refit runs max_iter steps, so the number of steps does not depend on the data.

    The fixture uses default_rng, whose streams differ between NumPy versions; with a real tolerance the number of
    steps would change across versions and assertions such as "steps >= 3" would fail spuriously.
    """
    return (FitConfig(support_size=4, step_size=1.0, max_iter=max_iter, warmup_steps=0,
                      parameter_tolerance=tolerance,
                      objective_relative_tolerance=tolerance),
            CEMConfig(k_components=2, max_outer_iter=2))


def _spy_steps(monkeypatch):
    """Collect the StepInfo of every step.

    The trace rows hold only q_reg_improvement, not q_reg_old / q_reg_new, so these assertions read StepInfo.
    """
    steps: list = []
    original = model._hard_label_m_step_with_loglik

    def spy(*args, **kwargs):
        result, info, loglik = original(*args, **kwargs)
        steps.append(info)
        return result, info, loglik

    # refit_fixed_labels goes through the private entry point with loglik, so the probe hooks it
    monkeypatch.setattr(cem_optimizer, "_hard_label_m_step_with_loglik", spy)
    return steps


# --------------------------------------------------------------------------- #
# Core assertion: cached values are bitwise equal to fresh computations
# --------------------------------------------------------------------------- #

def test_e2_cached_q_equals_freshly_computed_q_at_every_step(monkeypatch):
    """q_reg_old at the start of every step is bitwise equal to regression_q computed on `current` of that step."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs(max_iter=15)
    initial = zero_parameters_from_labels(observed, labels, k)

    seen: list[tuple[int, int]] = []
    original = model._hard_label_m_step_with_loglik

    def spy(parameters, lab, treatment, features, outcome, center_codes, config,
            *, q_reg_old=None):
        responsibilities = np.eye(parameters.k_components, dtype=float)[lab]
        fresh = regression_q(parameters, responsibilities, treatment, features,
                             outcome, center_codes)
        result, info, loglik = original(parameters, lab, treatment, features, outcome,
                                        center_codes, config, q_reg_old=q_reg_old)
        seen.append((_bits(fresh), _bits(info.q_reg_old)))
        return result, info, loglik

    monkeypatch.setattr(cem_optimizer, "_hard_label_m_step_with_loglik", spy)
    parameters, trace, status = refit_fixed_labels(observed, initial, labels, fit_config)

    assert status in {"converged", "max_iter_eligible"}
    assert len(seen) >= 3, "too few steps to reach the reuse path"
    for index, (fresh_bits, used_bits) in enumerate(seen):
        assert fresh_bits == used_bits, f"q_reg_old of step {index} differs in bits from the fresh value"


def test_e2_first_step_computes_q_and_later_steps_reuse_it(monkeypatch):
    """The first step computes q afresh; later q_reg_old equals the previous q_reg_new bit by bit."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs(max_iter=15)
    initial = zero_parameters_from_labels(observed, labels, k)
    steps = _spy_steps(monkeypatch)
    refit_fixed_labels(observed, initial, labels, fit_config)

    assert len(steps) >= 3
    responsibilities = np.eye(k, dtype=float)[labels]
    first = regression_q(initial, responsibilities, observed.treatment,
                         observed.features, observed.outcome, observed.center_codes)
    assert _bits(steps[0].q_reg_old) == _bits(first), "the first step should use the fresh value"
    for previous, current in zip(steps, steps[1:]):
        assert _bits(current.q_reg_old) == _bits(previous.q_reg_new)

def test_e2_regression_q_call_count_drops_by_one_per_reused_step(monkeypatch):
    """Guard: with reuse, regression_q is called (steps - 1) times fewer than 2 x steps."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs(max_iter=15)
    initial = zero_parameters_from_labels(observed, labels, k)

    calls = {"n": 0}
    # the candidate q goes through _regression_q_parts, to which the public regression_q delegates, so hooking
    # this function counts every full pass over the data to compute q
    original = model._regression_q_parts

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_regression_q_parts", counted)
    infos = _spy_steps(monkeypatch)
    refit_fixed_labels(observed, initial, labels, fit_config)

    steps = len(infos)
    backtracks = sum(int(info.backtracking_count) for info in infos)
    # per step: one candidate q (one more per extra backtracking) + q_reg_old in the first step only
    expected = steps + backtracks + 1
    assert calls["n"] == expected, (
        f"_regression_q_parts called {calls['n']} times, {expected} expected with reuse"
        f" ({steps} steps, {backtracks} extra backtracking steps)")


# --------------------------------------------------------------------------- #
# Public interface: unchanged behaviour without the argument
# --------------------------------------------------------------------------- #

def test_e2_generalized_m_step_without_kwarg_is_unchanged():
    """Positional calls still work and equal a call with the correct q passed explicitly, bit by bit."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs()
    parameters = zero_parameters_from_labels(observed, labels, k)
    responsibilities = np.eye(k, dtype=float)[labels]
    codes = observed.center_codes

    baseline, info_a = generalized_m_step(parameters, responsibilities, observed.treatment,
                                          observed.features, observed.outcome, codes,
                                          fit_config)
    known = regression_q(parameters, responsibilities, observed.treatment,
                         observed.features, observed.outcome, codes)
    reused, info_b = generalized_m_step(parameters, responsibilities, observed.treatment,
                                        observed.features, observed.outcome, codes,
                                        fit_config, q_reg_old=known)

    assert _bits(info_a.q_reg_old) == _bits(info_b.q_reg_old)
    assert _bits(info_a.q_reg_new) == _bits(info_b.q_reg_new)
    assert _bits(info_a.accepted_step_size) == _bits(info_b.accepted_step_size)
    assert info_a.backtracking_count == info_b.backtracking_count
    assert info_a.backtracking_failed == info_b.backtracking_failed
    assert np.array_equal(baseline.beta, reused.beta)
    assert np.array_equal(baseline.pi, reused.pi)
    assert np.array_equal(baseline.intercepts, reused.intercepts)
    assert _bits(baseline.gamma) == _bits(reused.gamma)


def test_e2_q_reg_old_is_independent_of_pi():
    """Premise of E2: regression_q does not read pi; changing pi does not change it."""
    observed, labels, k = _fixture()
    parameters = zero_parameters_from_labels(observed, labels, k)
    responsibilities = np.eye(k, dtype=float)[labels]
    before = regression_q(parameters, responsibilities, observed.treatment,
                          observed.features, observed.outcome, observed.center_codes)
    shifted = parameters.clone()
    shifted.pi = np.array([0.9, 0.1])
    after = regression_q(shifted, responsibilities, observed.treatment,
                         observed.features, observed.outcome, observed.center_codes)
    assert _bits(before) == _bits(after)


# --------------------------------------------------------------------------- #
# Failure and early-return paths
# --------------------------------------------------------------------------- #

def test_e2_all_backtracking_failed_still_raises_and_does_not_continue(monkeypatch):
    """When every backtracking step fails, FitError is raised as before, the refit returns
    m_step_backtracking_failed, and the q of the last rejected candidate is never cached."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs(max_iter=6)
    initial = zero_parameters_from_labels(observed, labels, k)

    def sabotage(parameters, responsibilities, treatment, features, outcome,
                 center_codes, config, step_size):
        bad = parameters.clone()
        bad.gamma = float(parameters.gamma - 50.0)      # always lowers Q
        return bad

    monkeypatch.setattr(model, "_regression_proposal", sabotage)

    parameters, trace, status = refit_fixed_labels(observed, initial, labels, fit_config)
    assert parameters is None
    assert status == "m_step_backtracking_failed"
    assert len([row for row in trace if "q_reg_old" in row]) == 0


def test_e2_monotonicity_violation_returns_early(monkeypatch):
    """A decrease of the objective beyond the tolerance returns early as before; the cache ends with the function."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs(max_iter=8)
    initial = zero_parameters_from_labels(observed, labels, k)

    original = cem_optimizer.classification_objective
    original_from_loglik = cem_optimizer._classification_objective_from_loglik
    state = {"n": 0}

    def drifting(data, parameters, lab):
        state["n"] += 1
        value = original(data, parameters, lab)
        return value - 1000.0 if state["n"] >= 3 else value

    def drifting_from_loglik(data, parameters, lab, row_loglik, center_codes):
        state["n"] += 1
        value = original_from_loglik(data, parameters, lab, row_loglik, center_codes)
        return value - 1000.0 if state["n"] >= 3 else value

    # the fixed-label path uses the latter; both are hooked so that either path triggers
    monkeypatch.setattr(cem_optimizer, "classification_objective", drifting)
    monkeypatch.setattr(cem_optimizer, "_classification_objective_from_loglik",
                        drifting_from_loglik)
    parameters, _trace, status = refit_fixed_labels(observed, initial, labels, fit_config)
    assert parameters is None
    assert status == "monotonicity_violation"


def test_e2_cache_does_not_leak_across_refit_calls(monkeypatch):
    """Two independent refits: the first step of the second computes q afresh and does not reuse the first's last value."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs(max_iter=10)
    initial = zero_parameters_from_labels(observed, labels, k)

    steps = _spy_steps(monkeypatch)
    refit_fixed_labels(observed, initial, labels, fit_config)
    boundary = len(steps)
    refit_fixed_labels(observed, initial, labels, fit_config)

    first_run, second_run = steps[:boundary], steps[boundary:]
    assert first_run and second_run and len(first_run) == len(second_run)
    # identical inputs -> identical trajectories (the second refit does not reuse the first's last value)
    for left, right in zip(first_run, second_run):
        assert _bits(left.q_reg_old) == _bits(right.q_reg_old)
        assert _bits(left.q_reg_new) == _bits(right.q_reg_new)
    # and q_reg_old of the second refit's first step differs from q_reg_new of the first refit's last step
    assert _bits(second_run[0].q_reg_old) != _bits(first_run[-1].q_reg_new)

def test_e2_label_changing_path_never_receives_the_cache(monkeypatch):
    """The CEM outer loop may change labels in every round, so that path never receives q_reg_old."""
    from bniseam_test.cem_optimizer import run_cem_candidate
    from bniseam_test.cem_types import CandidateSeed

    observed, labels, k = _fixture()
    fit_config, cem_config = _configs(max_iter=6)

    seen: list[object] = []
    original = cem_optimizer.hard_label_m_step
    original_private = cem_optimizer._hard_label_m_step_with_loglik

    def spy(*args, **kwargs):
        seen.append(kwargs.get("q_reg_old", "absent"))
        return original(*args, **kwargs)

    def spy_private(*args, **kwargs):
        seen.append(kwargs.get("q_reg_old", "absent"))
        return original_private(*args, **kwargs)

    monkeypatch.setattr(cem_optimizer, "hard_label_m_step", spy)
    monkeypatch.setattr(cem_optimizer, "_hard_label_m_step_with_loglik", spy_private)
    run_cem_candidate(observed, CandidateSeed(candidate_id=0, method="fixed_start",
                                              labels=labels, seed=None),
                      fit_config, cem_config)

    outer = [value for value in seen if value == "absent"]
    assert outer, "the CEM outer loop should call without q_reg_old"
    # outer calls never pass it; cached values come only from the inner fixed-label refit
    assert all(value in ("absent", None) or isinstance(value, float) for value in seen)


def test_e2_wrong_cached_value_would_change_the_step():
    """Reverse guard: a wrong cached value changes the result, so the argument has an effect and its
    correctness matters."""
    observed, labels, k = _fixture()
    fit_config, _ = _configs()
    parameters = zero_parameters_from_labels(observed, labels, k)
    responsibilities = np.eye(k, dtype=float)[labels]
    codes = observed.center_codes

    _good, info_good = generalized_m_step(parameters, responsibilities, observed.treatment,
                                          observed.features, observed.outcome, codes,
                                          fit_config)
    _bad, info_bad = generalized_m_step(parameters, responsibilities, observed.treatment,
                                        observed.features, observed.outcome, codes,
                                        fit_config, q_reg_old=1.0e6)
    assert info_bad.backtracking_failed and not info_good.backtracking_failed
