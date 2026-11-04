"""Truth-free center-level representations for the three frozen B2 views.

This module deliberately stops at representation construction.  Clustering,
fusion, candidate selection, and validation belong to later pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .multiview_types import ObservedDataView, _require_observed


_CLIP_LIMIT = 1.0e6
_FISHER_EPS = 1.0e-10


def _readonly_array(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ViewRepresentation:
    """One finite center-row-aligned representation and its audit record."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        names = tuple(self.feature_names)
        if matrix.ndim != 2:
            raise ValueError("view matrix must be two-dimensional")
        if matrix.shape[1] != len(names):
            raise ValueError("feature_names must align with view columns")
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError("view matrix must be finite")
        object.__setattr__(self, "matrix", _readonly_array(matrix, dtype=float))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class ViewBundle:
    """The outcome, treatment, and covariate views for one training half."""

    centers: np.ndarray
    outcome: ViewRepresentation
    treatment: ViewRepresentation
    covariate: ViewRepresentation

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers)
        if centers.ndim != 1:
            raise ValueError("centers must be one-dimensional")
        for representation in (self.outcome, self.treatment, self.covariate):
            if representation.matrix.shape[0] != centers.size:
                raise ValueError("every view row must align with centers")
        object.__setattr__(self, "centers", _readonly_array(centers))


@dataclass(frozen=True)
class _PreparedObserved:
    observed: ObservedDataView
    features: np.ndarray
    centers: np.ndarray
    per_center_counts: dict[str, int]
    feature_nonfinite_repairs: int
    feature_clips: int
    zero_variance_columns: int


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _center_key(value: Any) -> str:
    return str(_python_scalar(value))


def _validate_binary(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{name} must contain only finite binary values")
    return array


def _repair_and_clip_features(features: np.ndarray) -> tuple[np.ndarray, int, int]:
    repaired = np.array(features, dtype=float, copy=True)
    nonfinite = ~np.isfinite(repaired)
    repair_count = int(nonfinite.sum())
    if repair_count:
        for column in range(repaired.shape[1]):
            bad = nonfinite[:, column]
            if not np.any(bad):
                continue
            finite_values = repaired[~bad, column]
            replacement = float(np.median(finite_values)) if finite_values.size else 0.0
            repaired[bad, column] = replacement
    clip_mask = np.abs(repaired) > _CLIP_LIMIT
    clip_count = int(clip_mask.sum())
    if clip_count:
        np.clip(repaired, -_CLIP_LIMIT, _CLIP_LIMIT, out=repaired)
    return repaired, repair_count, clip_count


def _repair_and_clip_matrix(matrix: np.ndarray) -> tuple[np.ndarray, int, int]:
    repaired = np.array(matrix, dtype=float, copy=True)
    nonfinite = ~np.isfinite(repaired)
    repair_count = int(nonfinite.sum())
    if repair_count:
        repaired[nonfinite] = 0.0
    clip_mask = np.abs(repaired) > _CLIP_LIMIT
    clip_count = int(clip_mask.sum())
    if clip_count:
        np.clip(repaired, -_CLIP_LIMIT, _CLIP_LIMIT, out=repaired)
    return repaired, repair_count, clip_count


def _prepare(data: ObservedDataView) -> _PreparedObserved:
    observed = _require_observed(data)
    if observed.features.shape[1] < 1:
        raise ValueError("features must contain at least one column")
    _validate_binary(observed.treatment, "treatment")
    _validate_binary(observed.outcome, "outcome")

    centers = np.asarray(observed.centers)
    declared = [_python_scalar(value) for value in centers]
    observed_codes = [_python_scalar(value) for value in np.asarray(observed.center_codes)]
    if len(set(declared)) != len(declared) or set(declared) != set(observed_codes):
        raise ValueError("centers must uniquely and exactly match center_codes")
    if not declared:
        raise ValueError("centers must not be empty")

    features, repair_count, clip_count = _repair_and_clip_features(observed.features)
    global_variance = np.var(features, axis=0, ddof=0)
    per_center_counts = {
        _center_key(center): int(np.sum(observed.center_codes == center))
        for center in centers
    }
    if min(per_center_counts.values()) < 1:
        raise ValueError("every declared center must contain observations")
    return _PreparedObserved(
        observed=observed,
        features=features,
        centers=np.array(centers, copy=True),
        per_center_counts=per_center_counts,
        feature_nonfinite_repairs=repair_count,
        feature_clips=clip_count,
        zero_variance_columns=int(np.sum(global_variance <= _FISHER_EPS)),
    )


def _validate_components(k_components: int, n_centers: int) -> int:
    if isinstance(k_components, bool) or not isinstance(k_components, (int, np.integer)):
        raise TypeError("k_components must be an integer")
    value = int(k_components)
    if value < 1 or value >= n_centers:
        raise ValueError("k_components must be positive and smaller than n_centers")
    return value


def _base_diagnostics(prepared: _PreparedObserved, view: str) -> dict[str, Any]:
    return {
        "view": view,
        "input_row_count": int(prepared.observed.n_samples),
        "per_center_counts": dict(prepared.per_center_counts),
        "center_order": [_python_scalar(value) for value in prepared.centers],
        "zero_variance_column_count": int(prepared.zero_variance_columns),
        "input_zero_variance_column_count": int(prepared.zero_variance_columns),
        "clipping_count": int(prepared.feature_clips),
        "nonfinite_repair_count": int(prepared.feature_nonfinite_repairs),
        "input_feature_count": int(prepared.features.shape[1]),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1.0e-6, 1.0 - 1.0e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def _pooled_outcome_null(
    outcome: np.ndarray,
    treatment: np.ndarray,
    *,
    max_iter: int = 50,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(outcome.size, dtype=float), treatment])
    coefficients = np.array([_logit(float(outcome.mean())), 0.0], dtype=float)
    for _ in range(max_iter):
        eta = design @ coefficients
        probability = _sigmoid(eta)
        variance = np.maximum(probability * (1.0 - probability), _FISHER_EPS)
        score = design.T @ (outcome - probability)
        information = design.T @ (variance[:, None] * design)
        information.flat[:: information.shape[0] + 1] += 1.0e-8
        step = np.linalg.solve(information, score)
        old_loglik = float(np.sum(outcome * eta - np.logaddexp(0.0, eta)))
        scale = 1.0
        accepted = False
        while scale >= 2.0**-12:
            proposed = coefficients + scale * step
            proposed_eta = design @ proposed
            proposed_loglik = float(
                np.sum(outcome * proposed_eta - np.logaddexp(0.0, proposed_eta))
            )
            if proposed_loglik >= old_loglik - 1.0e-10:
                coefficients = proposed
                accepted = True
                break
            scale *= 0.5
        if not accepted or np.max(np.abs(scale * step)) <= tolerance:
            break
    return _sigmoid(design @ coefficients), design, coefficients


def _center_outcome_scores(prepared: _PreparedObserved) -> tuple[np.ndarray, np.ndarray]:
    outcome = prepared.observed.outcome.astype(float, copy=False)
    treatment = prepared.observed.treatment.astype(float, copy=False)
    probability, nuisance_design, coefficients = _pooled_outcome_null(outcome, treatment)
    variance = np.maximum(probability * (1.0 - probability), _FISHER_EPS)
    residual = outcome - probability
    nuisance_information = nuisance_design.T @ (variance[:, None] * nuisance_design)
    nuisance_information.flat[:: nuisance_information.shape[0] + 1] += 1.0e-8
    projection = np.linalg.solve(
        nuisance_information,
        nuisance_design.T @ (variance[:, None] * prepared.features),
    )
    efficient_features = prepared.features - nuisance_design @ projection

    rows: list[np.ndarray] = []
    for center in prepared.centers:
        mask = prepared.observed.center_codes == center
        feature_block = efficient_features[mask]
        feature_score = feature_block.T @ residual[mask]
        information = (feature_block * feature_block).T @ variance[mask]
        rows.append(feature_score / np.sqrt(np.maximum(information, _FISHER_EPS)))
    return np.vstack(rows), coefficients


def _center_treatment_scores(prepared: _PreparedObserved) -> tuple[np.ndarray, float]:
    treatment = prepared.observed.treatment.astype(float, copy=False)
    probability = float(np.clip(treatment.mean(), 1.0e-6, 1.0 - 1.0e-6))
    residual = treatment - probability
    variance = max(probability * (1.0 - probability), _FISHER_EPS)
    centered_features = prepared.features - prepared.features.mean(axis=0)
    rows: list[np.ndarray] = []
    for center in prepared.centers:
        mask = prepared.observed.center_codes == center
        feature_block = centered_features[mask]
        score = feature_block.T @ residual[mask]
        information = variance * np.sum(feature_block * feature_block, axis=0)
        rows.append(score / np.sqrt(np.maximum(information, _FISHER_EPS)))
    return np.vstack(rows), probability


def _pca_scores(matrix: np.ndarray, target_components: int) -> np.ndarray:
    centered = np.asarray(matrix, dtype=float) - np.mean(matrix, axis=0)
    rank = int(np.linalg.matrix_rank(centered))
    n_components = max(
        1,
        min(target_components, centered.shape[0] - 1, centered.shape[1], max(rank, 1)),
    )
    if rank == 0:
        return np.zeros((centered.shape[0], n_components), dtype=float)
    left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    for component in range(right.shape[0]):
        pivot = int(np.argmax(np.abs(right[component])))
        if right[component, pivot] < 0.0:
            left[:, component] *= -1.0
    return left[:, :n_components] * singular_values[:n_components]


def _robust_sparse_representation(
    score_matrix: np.ndarray,
    k_components: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(score_matrix, dtype=float)
    n_centers, p_total = matrix.shape
    heterogeneity = np.std(matrix, axis=0, ddof=1)
    median = float(np.median(heterogeneity))
    mad = float(np.median(np.abs(heterogeneity - median)))
    threshold = median + 3.0 * 1.4826 * mad
    width_limit = min(p_total, max(2 * k_components, int(np.ceil(np.sqrt(p_total)))))

    if p_total == 1:
        correlation = np.ones((1, 1), dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.nan_to_num(
                np.corrcoef(matrix, rowvar=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
    order = np.argsort(-heterogeneity, kind="mergesort")
    selected: list[int] = []

    def add_if_distinct(index: int) -> bool:
        if selected and float(np.max(np.abs(correlation[index, selected]))) >= 0.8:
            return False
        selected.append(int(index))
        return True

    for feature_index in order:
        if heterogeneity[feature_index] < threshold:
            break
        add_if_distinct(int(feature_index))
        if len(selected) >= width_limit:
            break
    required = min(k_components, p_total)
    if len(selected) < required:
        for feature_index in order:
            if int(feature_index) in selected:
                continue
            add_if_distinct(int(feature_index))
            if len(selected) >= required:
                break
    if not selected:
        selected = [int(order[0])]

    retained = matrix[:, selected]
    retained_mean = retained.mean(axis=0)
    retained_scale = retained.std(axis=0, ddof=0)
    retained_scale = np.where(retained_scale > _FISHER_EPS, retained_scale, 1.0)
    standardized = (retained - retained_mean) / retained_scale
    reduced = _pca_scores(standardized, 2 * k_components)
    reduced, repairs, clips = _repair_and_clip_matrix(reduced)
    return reduced, {
        "heterogeneity_threshold": float(threshold),
        "heterogeneity_median": median,
        "heterogeneity_mad": mad,
        "feature_heterogeneity": heterogeneity.astype(float).tolist(),
        "sparse_width_limit": int(width_limit),
        "selected_feature_indices": selected,
        "selected_feature_count": int(len(selected)),
        "pca_components": int(reduced.shape[1]),
        "score_zero_variance_column_count": int(np.sum(heterogeneity <= _FISHER_EPS)),
        "representation_nonfinite_repair_count": int(repairs),
        "representation_clipping_count": int(clips),
    }


def _degenerate_binary_representation(
    prepared: _PreparedObserved,
    *,
    view: str,
    reason: str,
    uses_outcome: bool,
    uses_treatment: bool,
) -> ViewRepresentation:
    """Return an explicit zero view when a binary score cannot be defined."""

    p_total = int(prepared.features.shape[1])
    diagnostics = _base_diagnostics(prepared, view)
    diagnostics.update(
        {
            "zero_variance_column_count": p_total,
            "score_zero_variance_column_count": p_total,
            "binary_variation": False,
            "view_eligible": False,
            "degenerate_reason": reason,
            "feature_heterogeneity": [0.0] * p_total,
            "heterogeneity_threshold": 0.0,
            "heterogeneity_median": 0.0,
            "heterogeneity_mad": 0.0,
            "sparse_width_limit": 0,
            "selected_feature_indices": [],
            "selected_feature_count": 0,
            "pca_components": 1,
            "representation_nonfinite_repair_count": 0,
            "representation_clipping_count": 0,
            "uses_outcome": uses_outcome,
            "uses_treatment": uses_treatment,
        }
    )
    return ViewRepresentation(
        matrix=np.zeros((prepared.centers.size, 1), dtype=float),
        feature_names=(f"{view}_score_degenerate",),
        diagnostics=diagnostics,
    )


def outcome_score_representation(
    data: ObservedDataView,
    *,
    k_components: int,
) -> ViewRepresentation:
    """Build the robust sparse efficient Bernoulli outcome-score embedding."""

    prepared = _prepare(data)
    k = _validate_components(k_components, prepared.centers.size)
    if np.unique(prepared.observed.outcome).size < 2:
        return _degenerate_binary_representation(
            prepared,
            view="outcome",
            reason="constant_outcome",
            uses_outcome=True,
            uses_treatment=True,
        )
    raw_scores, null_coefficients = _center_outcome_scores(prepared)
    raw_scores, score_repairs, score_clips = _repair_and_clip_matrix(raw_scores)
    matrix, sparse = _robust_sparse_representation(raw_scores, k)
    diagnostics = _base_diagnostics(prepared, "outcome")
    diagnostics.update(sparse)
    diagnostics["zero_variance_column_count"] = int(
        sparse["score_zero_variance_column_count"]
    )
    diagnostics["nonfinite_repair_count"] += int(score_repairs) + int(
        sparse["representation_nonfinite_repair_count"]
    )
    diagnostics["clipping_count"] += int(score_clips) + int(
        sparse["representation_clipping_count"]
    )
    diagnostics.update(
        {
            "pooled_null": "Y ~ 1 + X",
            "pooled_null_coefficients": null_coefficients.astype(float).tolist(),
            "uses_outcome": True,
            "uses_treatment": True,
            "binary_variation": True,
            "view_eligible": True,
            "degenerate_reason": None,
            "selection_tie_order": "descending heterogeneity, stable original-index order",
        }
    )
    names = tuple(f"outcome_score_pc_{index}" for index in range(matrix.shape[1]))
    return ViewRepresentation(matrix=matrix, feature_names=names, diagnostics=diagnostics)


def treatment_score_representation(
    data: ObservedDataView,
    *,
    k_components: int,
) -> ViewRepresentation:
    """Build the frozen robust sparse ``F``-``X`` Fisher-score embedding."""

    prepared = _prepare(data)
    k = _validate_components(k_components, prepared.centers.size)
    if np.unique(prepared.observed.treatment).size < 2:
        return _degenerate_binary_representation(
            prepared,
            view="treatment",
            reason="constant_treatment",
            uses_outcome=False,
            uses_treatment=True,
        )
    raw_scores, pooled_probability = _center_treatment_scores(prepared)
    raw_scores, score_repairs, score_clips = _repair_and_clip_matrix(raw_scores)
    matrix, sparse = _robust_sparse_representation(raw_scores, k)
    diagnostics = _base_diagnostics(prepared, "treatment")
    diagnostics.update(sparse)
    diagnostics["zero_variance_column_count"] = int(
        sparse["score_zero_variance_column_count"]
    )
    diagnostics["nonfinite_repair_count"] += int(score_repairs) + int(
        sparse["representation_nonfinite_repair_count"]
    )
    diagnostics["clipping_count"] += int(score_clips) + int(
        sparse["representation_clipping_count"]
    )
    diagnostics.update(
        {
            "pooled_treatment_probability": float(pooled_probability),
            "uses_outcome": False,
            "uses_treatment": True,
            "binary_variation": True,
            "view_eligible": True,
            "degenerate_reason": None,
            "selection_tie_order": "descending heterogeneity, stable original-index order",
        }
    )
    names = tuple(f"treatment_score_pc_{index}" for index in range(matrix.shape[1]))
    return ViewRepresentation(matrix=matrix, feature_names=names, diagnostics=diagnostics)


def covariate_moment_representation(
    data: ObservedDataView,
    *,
    eps: float = 1.0e-8,
) -> ViewRepresentation:
    """Build the frozen center mean/log-variance moment representation."""

    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    prepared = _prepare(data)
    global_mean = prepared.features.mean(axis=0)
    global_variance = prepared.features.var(axis=0, ddof=0)
    global_scale = np.maximum(np.sqrt(global_variance), eps)
    mean_rows: list[np.ndarray] = []
    variance_rows: list[np.ndarray] = []
    for center in prepared.centers:
        mask = prepared.observed.center_codes == center
        block = prepared.features[mask]
        mean_rows.append(
            np.sqrt(float(block.shape[0])) * (block.mean(axis=0) - global_mean) / global_scale
        )
        center_variance = block.var(axis=0, ddof=0)
        variance_rows.append(np.log((center_variance + eps) / (global_variance + eps)))
    matrix = np.concatenate([np.vstack(mean_rows), np.vstack(variance_rows)], axis=1)
    matrix, moment_repairs, moment_clips = _repair_and_clip_matrix(matrix)
    p_total = prepared.features.shape[1]
    diagnostics = _base_diagnostics(prepared, "covariate")
    diagnostics["nonfinite_repair_count"] += int(moment_repairs)
    diagnostics["clipping_count"] += int(moment_clips)
    diagnostics.update(
        {
            "variance_ddof": 0,
            "eps": float(eps),
            "global_feature_mean": global_mean.astype(float).tolist(),
            "global_feature_variance": global_variance.astype(float).tolist(),
            "feature_heterogeneity": np.std(
                matrix,
                axis=0,
                ddof=1 if matrix.shape[0] > 1 else 0,
            ).astype(float).tolist(),
            "selected_feature_indices": list(range(2 * p_total)),
            "selected_feature_count": int(2 * p_total),
            "source_feature_indices": list(range(p_total)),
            "selection_rule": "all mean and log-variance coordinates retained",
            "uses_outcome": False,
            "uses_treatment": False,
        }
    )
    names = tuple(f"F_mean_score_{index}" for index in range(p_total)) + tuple(
        f"F_log_variance_{index}" for index in range(p_total)
    )
    return ViewRepresentation(matrix=matrix, feature_names=names, diagnostics=diagnostics)


def build_view_bundle(
    data: ObservedDataView,
    *,
    k_components: int,
    eps: float = 1.0e-8,
) -> ViewBundle:
    """Build all three views from exactly the supplied observed training rows."""

    observed = _require_observed(data)
    outcome = outcome_score_representation(observed, k_components=k_components)
    treatment = treatment_score_representation(observed, k_components=k_components)
    covariate = covariate_moment_representation(observed, eps=eps)
    return ViewBundle(
        centers=observed.centers,
        outcome=outcome,
        treatment=treatment,
        covariate=covariate,
    )
