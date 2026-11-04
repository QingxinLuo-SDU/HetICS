"""Bernoulli-sEAM classification and observed objectives."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .model import Parameters, linear_predictor
from .multiview_types import ObservedDataView, _require_observed


def bernoulli_log_likelihood(
    outcome: np.ndarray,
    linear_predictor_values: np.ndarray,
) -> float:
    """Return a stable summed Bernoulli log likelihood from logits."""

    response = np.asarray(outcome, dtype=float)
    eta = np.asarray(linear_predictor_values, dtype=float)
    if response.shape != eta.shape:
        raise ValueError("outcome and linear predictor must have the same shape")
    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(eta)):
        raise ValueError("outcome and linear predictor must be finite")
    if np.any((response != 0.0) & (response != 1.0)):
        raise ValueError("outcome must be binary")
    value = float(np.sum(response * eta - np.logaddexp(0.0, eta)))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite Bernoulli log-likelihood")
    return value


def dense_center_codes(data: ObservedDataView) -> np.ndarray:
    """Map observed center identifiers to the order in ``data.centers``.

    E1 (perf): when the identifiers are already the dense integers ``0..E-1`` the
    mapping is the identity, so the per-row Python loop below recomputes a constant.
    That is the hot path -- ``_training_view`` hands the kernel exactly this shape and
    every fixed-label iteration re-derives it.  The fast path produces the identical
    int64 array and raises the identical errors; anything else falls through to the
    original loop unchanged.
    """

    observed = _require_observed(data)
    centers = np.asarray(observed.centers)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers must be a non-empty one-dimensional array")
    codes = np.asarray(observed.center_codes)
    if (
        np.issubdtype(centers.dtype, np.integer)
        and np.issubdtype(codes.dtype, np.integer)
        and np.array_equal(centers, np.arange(centers.size))
    ):
        # Range-check in the *source* dtype: casting an out-of-range uint64 to int64
        # first would wrap it to a negative value and hide the error.
        if codes.size and (codes.min() < 0 or codes.max() >= centers.size):
            raise ValueError("center_codes contains an identifier absent from centers")
        result = codes.astype(np.int64, copy=True)
        if np.count_nonzero(np.bincount(result, minlength=centers.size)) != centers.size:
            raise ValueError("every center must have at least one observation")
        return result
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
            raise ValueError("center_codes contains an identifier absent from centers") from error
    if np.unique(result).size != centers.size:
        raise ValueError("every center must have at least one observation")
    return result


def component_loglik_matrix(
    data: ObservedDataView,
    parameters: Parameters,
) -> np.ndarray:
    """Return each center's Bernoulli log likelihood for every component."""

    observed = _require_observed(data)
    eta = linear_predictor(parameters, observed.treatment, observed.features)
    row_loglik = observed.outcome[:, None] * eta - np.logaddexp(0.0, eta)
    center_codes = dense_center_codes(observed)
    n_centers = int(observed.centers.size)
    result = np.zeros((n_centers, parameters.k_components), dtype=float)
    np.add.at(result, center_codes, row_loglik)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite component log-likelihood")
    return result


def hard_score_matrix(
    data: ObservedDataView,
    parameters: Parameters,
) -> np.ndarray:
    """Return center/component scores for a hard classification objective."""

    if np.any(parameters.pi <= 0.0):
        raise FloatingPointError("classification weights must be positive")
    return component_loglik_matrix(data, parameters) + np.log(parameters.pi)[None, :]


def classification_objective(
    data: ObservedDataView,
    parameters: Parameters,
    labels: np.ndarray,
) -> float:
    """Return the complete-data objective selected by hard center labels."""

    observed = _require_observed(data)
    labels = np.asarray(labels, dtype=int)
    n_centers = int(observed.centers.size)
    if labels.shape != (n_centers,):
        raise ValueError("labels must contain one value per center")
    if np.any(labels < 0) or np.any(labels >= parameters.k_components):
        raise ValueError("invalid hard labels")
    scores = hard_score_matrix(observed, parameters)
    value = float(np.sum(scores[np.arange(n_centers), labels]))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite classification objective")
    return value


def _classification_objective_from_loglik(
    data: ObservedDataView,
    parameters: Parameters,
    labels: np.ndarray,
    row_loglik: np.ndarray,
    center_codes: np.ndarray,
) -> float:
    """E3 (perf): ``classification_objective`` fed an already-computed ``row_loglik``.

    Everything except the predictor/log-likelihood recomputation is the original code,
    in the original order, with the original checks -- the ``np.add.at`` scatter runs
    over the same rows in the same sequence, the ``log(pi)`` term uses the *updated*
    ``pi``, and the same entries are selected and summed.  The caller must pass the
    array produced from these very parameters; ``pi`` may differ because neither the
    predictor nor ``row_loglik`` reads it.
    """

    observed = _require_observed(data)
    labels = np.asarray(labels, dtype=int)
    n_centers = int(observed.centers.size)
    if labels.shape != (n_centers,):
        raise ValueError("labels must contain one value per center")
    if np.any(labels < 0) or np.any(labels >= parameters.k_components):
        raise ValueError("invalid hard labels")
    if np.any(parameters.pi <= 0.0):
        raise FloatingPointError("classification weights must be positive")
    if row_loglik.shape != (observed.n_samples, parameters.k_components):
        raise ValueError("row_loglik does not match the observed data")
    result = np.zeros((n_centers, parameters.k_components), dtype=float)
    np.add.at(result, center_codes, row_loglik)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite component log-likelihood")
    scores = result + np.log(parameters.pi)[None, :]
    value = float(np.sum(scores[np.arange(n_centers), labels]))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite classification objective")
    return value


def observed_mixture_objective(
    data: ObservedDataView,
    parameters: Parameters,
) -> float:
    """Return the observed mixture objective using stable log-sum-exp."""

    value = float(np.sum(logsumexp(hard_score_matrix(data, parameters), axis=1)))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite observed mixture objective")
    return value


def one_hot_responsibilities(
    labels: np.ndarray,
    k_components: int,
) -> np.ndarray:
    """Return exact, unsmoothed one-hot rows for hard labels."""

    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= k_components):
        raise ValueError("invalid hard labels")
    return np.eye(k_components, dtype=float)[labels]
