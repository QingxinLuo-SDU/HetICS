"""tests/test_perf_equivalence.py — equivalence and boundaries of the optimizations E1 / E4 / D1.

The tests show that the optimizations leave the behaviour unchanged; they make no claim about speed.

E1: dense fast path of `cem_objectives.dense_center_codes`. The file holds a verbatim copy of the previous
    implementation, `_reference_dense_center_codes`, and asserts identical results and exception messages.
E4: `multiview_validation.XModelCache._fit_uncached` hoists `self._features[mask]` out of the closure, which
    requires that `x_ridge_objective_and_gradient` does not modify the feature matrix in place.
D1: `label_stage._cache_diagnostics` records every fit count, including the nested by_direction and the empty
    and failure paths.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bniseam_test.cem_objectives import dense_center_codes
from bniseam_test.multiview_types import ObservedDataView
from bniseam_test.multiview_validation import x_ridge_objective_and_gradient


# --------------------------------------------------------------------------- #
# E1
# --------------------------------------------------------------------------- #

def _reference_dense_center_codes(data: ObservedDataView) -> np.ndarray:
    """The previous implementation, copied verbatim, for differential checks."""
    observed = data
    centers = np.asarray(observed.centers)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers must be a non-empty one-dimensional array")
    mapping: dict[object, int] = {}
    for index, value in enumerate(centers):
        key = value.item() if isinstance(value, np.generic) else value
        if key in mapping:
            raise ValueError("centers must contain unique identifiers")
        mapping[key] = index
    result = np.empty(observed.n_samples, dtype=np.int64)
    for row, value in enumerate(np.asarray(observed.center_codes)):
        key = value.item() if isinstance(value, np.generic) else value
        try:
            result[row] = mapping[key]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "center_codes contains an identifier absent from centers") from error
    if np.unique(result).size != centers.size:
        raise ValueError("every center must have at least one observation")
    return result


def _view(codes, centers, n_features: int = 3) -> ObservedDataView:
    codes = np.asarray(codes)
    n = codes.size
    rng = np.random.default_rng(0)
    return ObservedDataView(
        features=rng.standard_normal((n, n_features)),
        treatment=np.zeros(n, dtype=float),
        outcome=np.zeros(n, dtype=float),
        center_codes=codes,
        centers=np.asarray(centers),
    )


def _assert_same(view: ObservedDataView) -> np.ndarray:
    """Both implementations give bitwise identical arrays, or raise the same exception type and message."""
    try:
        expected = _reference_dense_center_codes(view)
    except Exception as error:                                   # noqa: BLE001
        with pytest.raises(type(error), match=_escape(str(error))):
            dense_center_codes(view)
        return None
    actual = dense_center_codes(view)
    assert actual.dtype == expected.dtype == np.int64
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)
    return actual


def _escape(text: str) -> str:
    import re
    return re.escape(text)


DENSE_CASES = {
    "dense_int_centers": ([0, 0, 1, 1, 2, 2], [0, 1, 2]),
    "dense_single_center": ([0, 0, 0], [0]),
    "dense_unsorted_codes": ([2, 0, 1, 2, 0, 1], [0, 1, 2]),
    "dense_int32_codes": (np.array([0, 1, 2, 0], dtype=np.int32), np.array([0, 1, 2])),
    "dense_uint8_codes": (np.array([0, 1, 1, 0], dtype=np.uint8), np.array([0, 1])),
}

SLOW_CASES = {
    "non_dense_int_centers": ([10, 20, 30, 10], [10, 20, 30]),
    "negative_center_labels": ([-1, 0, -1, 0], [-1, 0]),
    "unsorted_center_list": ([2, 0, 1], [2, 0, 1]),
    "string_centers": (np.array(["a", "b", "a"]), np.array(["a", "b"])),
    "float_codes_matching_int_centers": (np.array([0.0, 1.0, 1.0]), np.array([0, 1])),
    "bool_codes": (np.array([False, True, True]), np.array([0, 1])),
}

FAILURE_CASES = {
    "code_out_of_range_dense": ([0, 1, 5], [0, 1, 2]),
    "negative_code_dense": (np.array([0, 1, -1]), np.array([0, 1, 2])),
    "center_without_observation_dense": ([0, 0, 1], [0, 1, 2]),
    "center_without_observation_slow": ([10, 10, 20], [10, 20, 30]),
    "duplicate_centers": ([0, 1], [0, 1, 1]),
    "code_absent_slow": ([10, 99], [10, 20]),
    "empty_centers": ([0, 1], np.array([], dtype=np.int64)),
}


@pytest.mark.parametrize("name", sorted(DENSE_CASES))
def test_e1_dense_fast_path_matches_reference(name):
    codes, centers = DENSE_CASES[name]
    assert _assert_same(_view(codes, centers)) is not None


@pytest.mark.parametrize("name", sorted(SLOW_CASES))
def test_e1_non_dense_inputs_still_use_original_path(name):
    codes, centers = SLOW_CASES[name]
    assert _assert_same(_view(codes, centers)) is not None


@pytest.mark.parametrize("name", sorted(FAILURE_CASES))
def test_e1_failures_match_reference(name):
    codes, centers = FAILURE_CASES[name]
    _assert_same(_view(codes, centers))


def test_e1_uint64_out_of_range_is_rejected_not_wrapped():
    """Range check before the int64 cast: otherwise 2**63 would wrap to a negative value."""
    codes = np.array([0, 1, 2**63], dtype=np.uint64)
    view = _view(codes, np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="absent from centers"):
        dense_center_codes(view)
    # the previous implementation also rejects it (the key is not in the dictionary)
    with pytest.raises(ValueError, match="absent from centers"):
        _reference_dense_center_codes(view)


def test_e1_result_is_an_independent_copy():
    """The result is a new array; changing it does not change the input."""
    codes = np.array([0, 1, 1, 0], dtype=np.int64)
    view = _view(codes, np.array([0, 1]))
    result = dense_center_codes(view)
    result[0] = 1
    assert view.center_codes[0] == 0
    assert np.shares_memory(result, view.center_codes) is False


def test_e1_fast_path_actually_taken():
    """Guard: if the fast path were not taken this test would still pass, so the call path is checked directly.

    A sentinel array that raises when an element of centers is accessed shows that dense input does not build
    the dictionary.
    """
    class Tripwire(np.ndarray):
        def __iter__(self):                                       # pragma: no cover
            raise AssertionError("the row-wise Python loop ran; the fast path was not taken")

    centers = np.arange(3).view(Tripwire)
    codes = np.array([0, 1, 2, 0], dtype=np.int64)
    view = _view(codes, centers)
    result = dense_center_codes(view)                 # must not iterate over centers
    assert np.array_equal(result, codes)


# --------------------------------------------------------------------------- #
# E4
# --------------------------------------------------------------------------- #

def test_e4_ridge_objective_does_not_mutate_features():
    """E4 hoists the masked copy out of the loop, which requires that the callee only reads the array."""
    rng = np.random.default_rng(7)
    features = rng.standard_normal((40, 5))
    snapshot = features.copy()
    treatment = (rng.random(40) < 0.5).astype(float)
    treatment[0], treatment[1] = 0.0, 1.0                 # both classes present
    theta = rng.standard_normal(6)
    x_ridge_objective_and_gradient(theta, features, treatment, 0.01)
    assert np.array_equal(features, snapshot)


def test_e4_repeated_evaluations_on_one_array_are_identical():
    """Repeated evaluation on the same array is bitwise identical, so reusing the copy is equivalent."""
    rng = np.random.default_rng(8)
    features = rng.standard_normal((40, 5))
    treatment = (rng.random(40) < 0.5).astype(float)
    treatment[0], treatment[1] = 0.0, 1.0
    theta = rng.standard_normal(6)
    first = x_ridge_objective_and_gradient(theta, features, treatment, 0.01)
    second = x_ridge_objective_and_gradient(theta, features, treatment, 0.01)
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])


def test_e4_sliced_copy_equals_repeated_slicing():
    """The hoisted copy is bitwise equal to the array sliced anew each time."""
    rng = np.random.default_rng(9)
    features = rng.standard_normal((60, 4))
    mask = np.zeros(60, dtype=bool)
    mask[::3] = True
    hoisted = features[mask]
    for _ in range(3):
        assert np.array_equal(hoisted, features[mask])
        assert hoisted.dtype == features[mask].dtype


# --------------------------------------------------------------------------- #
# D1
# --------------------------------------------------------------------------- #

def _pipeline_cache(**overrides):
    from bniseam_test.multiview_pipeline import CacheDiagnostics, DirectionCacheDiagnostics
    direction = DirectionCacheDiagnostics(
        unique_partition_count=12, refinement_fit_count=12, expensive_fit_count=12,
        pre_validation_fixed_fit_count=10, x_model_fit_count=19, x_cache_hits=15,
        f_model_fit_count=19, f_cache_hits=15)
    fields = dict(
        unique_partition_count=25, refinement_fit_count=25, expensive_fit_count=25,
        pre_validation_fixed_fit_count=22, final_fixed_refit_count=2,
        total_parameter_fit_count=49, by_direction={"A->B": direction})
    fields.update(overrides)
    return CacheDiagnostics(**fields)


def test_d1_exports_every_count_field():
    import label_stage as ls
    out = ls._cache_diagnostics(_pipeline_cache())
    for name in ls._CACHE_COUNT_FIELDS:
        assert name in out and isinstance(out[name], int)
    assert set(out) == set(ls._CACHE_COUNT_FIELDS) | {"by_direction"}


def test_d1_counts_are_not_double_counted():
    """Two invariants of the counts (expensive work is not counted twice)."""
    import label_stage as ls
    out = ls._cache_diagnostics(_pipeline_cache())
    assert out["expensive_fit_count"] == out["refinement_fit_count"]
    assert out["total_parameter_fit_count"] == (
        out["refinement_fit_count"]
        + out["pre_validation_fixed_fit_count"]
        + out["final_fixed_refit_count"])


def test_d1_by_direction_is_nested_plain_ints_and_json_serialisable():
    import label_stage as ls
    out = ls._cache_diagnostics(_pipeline_cache())
    assert isinstance(out["by_direction"], dict)
    inner = out["by_direction"]["A->B"]
    assert isinstance(inner, dict) and inner["x_model_fit_count"] == 19
    assert all(isinstance(v, int) for v in inner.values())
    assert json.loads(json.dumps(out)) == out


def test_d1_handles_empty_and_abstain_paths():
    """Abstaining or ineligible pipelines return _empty_cache(): all zeros, empty by_direction, no exception."""
    import label_stage as ls
    from bniseam_test.multiview_pipeline import _empty_cache
    out = ls._cache_diagnostics(_empty_cache())
    assert out["total_parameter_fit_count"] == 0
    assert out["by_direction"] == {}
    assert json.loads(json.dumps(out)) == out


def test_d1_missing_by_direction_does_not_raise():
    import label_stage as ls

    class Stub:
        unique_partition_count = 1
        refinement_fit_count = 1
        expensive_fit_count = 1
        pre_validation_fixed_fit_count = 0
        final_fixed_refit_count = 0
        total_parameter_fit_count = 1

    out = ls._cache_diagnostics(Stub())
    assert out["by_direction"] == {}
    assert out["total_parameter_fit_count"] == 1
