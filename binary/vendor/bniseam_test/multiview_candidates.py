"""Deterministic multiview center partitions for the frozen B2 label pool."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from .cem_assignment import solve_constrained_assignment
from .design import derive_algorithm_child_seed
from .multiview_types import ObservedDataView, _require_observed
from .multiview_views import ViewRepresentation, build_view_bundle


LOGICAL_CANDIDATE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("outcome__kmeans", "outcome", "kmeans"),
    ("outcome__ward", "outcome", "ward"),
    ("treatment__kmeans", "treatment", "kmeans"),
    ("covariate__kmeans", "covariate", "kmeans"),
    ("covariate__ward", "covariate", "ward"),
    ("fusion_xy__kmeans", "fusion_xy", "kmeans"),
    ("fusion_xy__ward", "fusion_xy", "ward"),
    ("fusion_xf__kmeans", "fusion_xf", "kmeans"),
    ("fusion_xf__ward", "fusion_xf", "ward"),
    ("fusion_yf__kmeans", "fusion_yf", "kmeans"),
    ("fusion_yf__ward", "fusion_yf", "ward"),
    ("fusion_fxy__kmeans", "fusion_fxy", "kmeans"),
    ("fusion_fxy__ward", "fusion_fxy", "ward"),
    ("balanced_random__01", "balanced_random", "balanced_random"),
    ("balanced_random__02", "balanced_random", "balanced_random"),
    ("balanced_random__03", "balanced_random", "balanced_random"),
    ("balanced_random__04", "balanced_random", "balanced_random"),
)

_FUSION_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "fusion_xy": ("outcome", "treatment"),
    "fusion_xf": ("treatment", "covariate"),
    "fusion_yf": ("outcome", "covariate"),
    "fusion_fxy": ("covariate", "treatment", "outcome"),
}


def _readonly(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _optional_array_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(left, right))


@dataclass(frozen=True)
class CandidatePoolConfig:
    """Frozen candidate construction controls."""

    k_components: int
    pi_cem: float
    eps: float = 1.0e-8
    correlation_threshold: float = 0.8
    kmeans_n_init: int = 100


@dataclass(frozen=True, eq=False)
class LogicalCandidate:
    """One stable logical pool entry, including duplicate provenance."""

    logical_id: str
    family: str
    algorithm: str
    eligible: bool
    canonical_id: str | None
    raw_labels: np.ndarray | None
    labels: np.ndarray | None
    cluster_sizes: tuple[int, ...]
    cost_matrix: np.ndarray | None
    ineligible_reason: str | None
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.raw_labels is not None:
            object.__setattr__(self, "raw_labels", _readonly(self.raw_labels, dtype=np.int64))
        if self.labels is not None:
            object.__setattr__(self, "labels", _readonly(self.labels, dtype=np.int64))
        if self.cost_matrix is not None:
            cost = np.asarray(self.cost_matrix, dtype=float)
            if cost.ndim != 2 or not np.all(np.isfinite(cost)):
                raise ValueError("candidate cost_matrix must be a finite matrix")
            object.__setattr__(self, "cost_matrix", _readonly(cost, dtype=float))
        object.__setattr__(self, "cluster_sizes", tuple(int(v) for v in self.cluster_sizes))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogicalCandidate):
            return NotImplemented
        return (
            self.logical_id == other.logical_id
            and self.family == other.family
            and self.algorithm == other.algorithm
            and self.eligible == other.eligible
            and self.canonical_id == other.canonical_id
            and _optional_array_equal(self.raw_labels, other.raw_labels)
            and _optional_array_equal(self.labels, other.labels)
            and self.cluster_sizes == other.cluster_sizes
            and _optional_array_equal(self.cost_matrix, other.cost_matrix)
            and self.ineligible_reason == other.ineligible_reason
            and self.diagnostics == other.diagnostics
        )


@dataclass(frozen=True, eq=False)
class CanonicalCandidate:
    """A label-name-invariant partition fitted at most once downstream."""

    canonical_id: str
    labels: np.ndarray
    cluster_sizes: tuple[int, ...]
    logical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _readonly(self.labels, dtype=np.int64))
        object.__setattr__(self, "cluster_sizes", tuple(int(v) for v in self.cluster_sizes))
        object.__setattr__(self, "logical_ids", tuple(self.logical_ids))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CanonicalCandidate):
            return NotImplemented
        return (
            self.canonical_id == other.canonical_id
            and np.array_equal(self.labels, other.labels)
            and self.cluster_sizes == other.cluster_sizes
            and self.logical_ids == other.logical_ids
        )


@dataclass(frozen=True, eq=False)
class CandidatePool:
    """All 17 logical records and their deduplicated eligible partitions."""

    centers: np.ndarray
    logical_candidates: tuple[LogicalCandidate, ...]
    canonical_candidates: tuple[CanonicalCandidate, ...]
    m_e: int
    status: str
    failure_reason: str | None
    representation_diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "centers", _readonly(self.centers))
        object.__setattr__(self, "logical_candidates", tuple(self.logical_candidates))
        object.__setattr__(self, "canonical_candidates", tuple(self.canonical_candidates))
        object.__setattr__(self, "representation_diagnostics", dict(self.representation_diagnostics))

    @property
    def fit_candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.canonical_id for candidate in self.canonical_candidates)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidatePool):
            return NotImplemented
        return (
            np.array_equal(self.centers, other.centers)
            and self.logical_candidates == other.logical_candidates
            and self.canonical_candidates == other.canonical_candidates
            and self.m_e == other.m_e
            and self.status == other.status
            and self.failure_reason == other.failure_reason
            and self.representation_diagnostics == other.representation_diagnostics
        )


@dataclass(frozen=True)
class _CandidateRepresentation:
    matrix: np.ndarray
    eligible: bool
    reason: str | None
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("candidate representation must be two-dimensional")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("candidate representation must be finite")
        object.__setattr__(self, "matrix", _readonly(matrix, dtype=float))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _validate_config(config: CandidatePoolConfig, n_centers: int) -> tuple[int, int]:
    if not isinstance(config, CandidatePoolConfig):
        raise TypeError("config must be a CandidatePoolConfig")
    if (
        isinstance(config.k_components, (bool, np.bool_))
        or not isinstance(config.k_components, Integral)
    ):
        raise TypeError("k_components must be an integer")
    k_components = int(config.k_components)
    if k_components < 1 or k_components >= n_centers:
        raise ValueError("k_components must be positive and smaller than n_centers")
    if (
        isinstance(config.pi_cem, (bool, np.bool_))
        or not isinstance(config.pi_cem, Real)
        or not np.isfinite(config.pi_cem)
        or config.pi_cem <= 0.0
    ):
        raise ValueError("pi_cem must be positive and finite")
    if not np.isfinite(config.eps) or config.eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    if (
        not np.isfinite(config.correlation_threshold)
        or config.correlation_threshold <= 0.0
        or config.correlation_threshold > 1.0
    ):
        raise ValueError("correlation_threshold must be in (0, 1]")
    if config.kmeans_n_init != 100:
        raise ValueError("the frozen KMeans contract requires n_init=100")
    m_e = max(2, int(math.ceil(n_centers * float(config.pi_cem))))
    return k_components, m_e


def _stable_seed(seed: int, stage: str) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    return derive_algorithm_child_seed(
        int(seed),
        stage="logical_candidate",
        logical_candidate_id=stage,
    )


def canonical_partition_key(labels: np.ndarray) -> tuple[int, ...]:
    """Return first-occurrence labels, invariant to numeric class names."""

    array = np.asarray(labels)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("labels must contain integers")
    mapping: dict[int, int] = {}
    result: list[int] = []
    for value in array.astype(np.int64, copy=False):
        numeric = int(value)
        if numeric not in mapping:
            mapping[numeric] = len(mapping)
        result.append(mapping[numeric])
    return tuple(result)


def _centroid_cost_matrix(
    representation: np.ndarray,
    raw_labels: np.ndarray,
    k_components: int,
) -> np.ndarray:
    matrix = np.asarray(representation, dtype=float)
    labels = np.asarray(raw_labels)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("representation must be a finite matrix")
    if labels.shape != (matrix.shape[0],):
        raise ValueError("raw_labels must contain one label per center")
    if (
        np.issubdtype(labels.dtype, np.bool_)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= k_components)
    ):
        raise ValueError("raw_labels must contain integer cluster indices")
    centroids: list[np.ndarray] = []
    for cluster in range(k_components):
        members = matrix[labels == cluster]
        if members.size == 0:
            raise ValueError("raw_labels must contain every cluster")
        centroids.append(members.mean(axis=0))
    difference = matrix[:, None, :] - np.vstack(centroids)[None, :, :]
    cost = np.sum(difference * difference, axis=2)
    if not np.all(np.isfinite(cost)):
        raise FloatingPointError("centroid distances are not finite")
    return cost


def repair_minimum_cluster_labels(
    representation: np.ndarray,
    raw_labels: np.ndarray,
    *,
    k_components: int,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair a partition by globally minimizing its raw-centroid costs."""

    cost = _centroid_cost_matrix(representation, raw_labels, k_components)
    repaired = solve_constrained_assignment(
        -cost,
        min_cluster_size,
        current_labels=np.asarray(raw_labels, dtype=np.int64),
    )
    return _readonly(repaired, dtype=np.int64), _readonly(cost, dtype=float)


def _deterministic_pca(
    matrix: np.ndarray,
    *,
    max_components: int,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrix, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    nonfinite_count = int(np.sum(~np.isfinite(centered)))
    if nonfinite_count:
        return np.zeros((values.shape[0], 0)), {
            "rank": 0,
            "input_dimension": int(values.shape[1]),
            "available_dimension": 0,
            "pca_components": 0,
            "nonfinite_count": nonfinite_count,
        }
    left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular_values > eps))
    input_dimension = int(values.shape[1])
    available = rank
    n_components = min(max_components, values.shape[0] - 1, available, rank)
    if n_components < 1:
        scores = np.zeros((values.shape[0], 0), dtype=float)
    else:
        for component in range(right.shape[0]):
            pivot = int(np.argmax(np.abs(right[component])))
            if right[component, pivot] < 0.0:
                left[:, component] *= -1.0
        scores = left[:, :n_components] * singular_values[:n_components]
    return scores, {
        "rank": rank,
        "input_dimension": input_dimension,
        "available_dimension": available,
        "pca_components": int(n_components),
        "singular_values": singular_values.astype(float).tolist(),
        "nonfinite_count": nonfinite_count,
    }


def _select_moment_block(
    matrix: np.ndarray,
    *,
    source_block: str,
    original_offset: int,
    k_components: int,
    correlation_threshold: float,
) -> tuple[list[int], dict[str, Any]]:
    values = np.asarray(matrix, dtype=float)
    p_total = values.shape[1]
    heterogeneity = np.std(values, axis=0, ddof=1 if values.shape[0] > 1 else 0)
    median = float(np.median(heterogeneity))
    mad = float(np.median(np.abs(heterogeneity - median)))
    threshold = median + 3.0 * 1.4826 * mad
    width_limit = min(p_total, max(2 * k_components, int(math.ceil(math.sqrt(p_total)))))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(
            np.corrcoef(values, rowvar=False) if p_total > 1 else np.ones((1, 1)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    order = np.argsort(-heterogeneity, kind="mergesort")
    selected: list[int] = []

    def add_distinct(index: int) -> bool:
        if selected and float(np.max(np.abs(correlation[index, selected]))) >= correlation_threshold:
            return False
        selected.append(index)
        return True

    for raw_index in order:
        index = int(raw_index)
        if heterogeneity[index] < threshold:
            break
        add_distinct(index)
        if len(selected) >= width_limit:
            break
    required = min(k_components, p_total, width_limit)
    for raw_index in order:
        if len(selected) >= required:
            break
        index = int(raw_index)
        if index not in selected:
            add_distinct(index)
    if not selected and p_total:
        selected = [int(order[0])]
    correlation_fill_exhausted = len(selected) < required
    return selected, {
        "source_block": source_block,
        "selected_local_indices": list(selected),
        "selected_original_indices": [original_offset + index for index in selected],
        "feature_heterogeneity": heterogeneity.astype(float).tolist(),
        "heterogeneity_median": median,
        "heterogeneity_mad": mad,
        "heterogeneity_threshold": float(threshold),
        "correlation_threshold": float(correlation_threshold),
        "width_limit": int(width_limit),
        "distinct_coordinate_count": int(len(selected)),
        "correlation_fill_exhausted": bool(correlation_fill_exhausted),
        "forced_correlated_fill_count": 0,
    }


def _preprocess_covariate(
    representation: ViewRepresentation,
    *,
    k_components: int,
    eps: float,
    correlation_threshold: float,
) -> _CandidateRepresentation:
    raw = np.asarray(representation.matrix, dtype=float)
    if raw.shape[1] % 2:
        raise ValueError("covariate moment matrix must contain paired M/V coordinates")
    p_total = raw.shape[1] // 2
    mean_indices, mean_diagnostics = _select_moment_block(
        raw[:, :p_total],
        source_block="M",
        original_offset=0,
        k_components=k_components,
        correlation_threshold=correlation_threshold,
    )
    variance_indices, variance_diagnostics = _select_moment_block(
        raw[:, p_total:],
        source_block="V",
        original_offset=p_total,
        k_components=k_components,
        correlation_threshold=correlation_threshold,
    )
    original_indices = mean_indices + [p_total + index for index in variance_indices]
    selected = raw[:, original_indices]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0, ddof=0)
    scale = np.where(scale > eps, scale, 1.0)
    standardized = (selected - mean) / scale
    reduced, pca_diagnostics = _deterministic_pca(
        standardized,
        max_components=2 * k_components,
        eps=eps,
    )
    rank = int(pca_diagnostics["rank"])
    eligible = rank > 0 and reduced.shape[1] > 0
    reason = None if eligible else "covariate_zero_rank"
    diagnostics = {
        "view": "covariate",
        "view_eligible": eligible,
        "degenerate_reason": reason,
        "selection_performed_separately": True,
        "mean_selection": mean_diagnostics,
        "variance_selection": variance_diagnostics,
        "selected_original_indices": original_indices,
        "selected_original_coordinate_names": [
            representation.feature_names[index] for index in original_indices
        ],
        "standardization_zero_scale_count": int(np.sum(selected.std(axis=0, ddof=0) <= eps)),
        **pca_diagnostics,
    }
    return _CandidateRepresentation(reduced, eligible, reason, diagnostics)


def _direct_representation(
    name: str,
    representation: ViewRepresentation,
    *,
    eps: float,
) -> _CandidateRepresentation:
    matrix = np.asarray(representation.matrix, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rank = int(np.sum(np.linalg.svd(centered, compute_uv=False) > eps))
    declared_eligible = bool(representation.diagnostics.get("view_eligible", True))
    eligible = declared_eligible and rank > 0
    reason = representation.diagnostics.get("degenerate_reason")
    if declared_eligible and rank == 0:
        reason = f"{name}_zero_rank"
    return _CandidateRepresentation(
        matrix,
        eligible,
        reason,
        {
            "view": name,
            "view_eligible": eligible,
            "degenerate_reason": reason,
            "rank": rank,
            "available_dimension": int(matrix.shape[1]),
            "pca_components": int(matrix.shape[1]),
            "nonfinite_count": 0,
            "source_view_diagnostics": dict(representation.diagnostics),
        },
    )


def _whiten_block(
    name: str,
    matrix: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any], str | None]:
    values = np.asarray(matrix, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    nonfinite_count = int(np.sum(~np.isfinite(centered)))
    pre_norm = float(np.linalg.norm(centered, ord="fro")) if not nonfinite_count else float("inf")
    if nonfinite_count:
        return np.zeros((values.shape[0], 0)), {
            "block": name,
            "rank": 0,
            "retained_dimensions": 0,
            "pre_frobenius_norm": pre_norm,
            "post_frobenius_norm": 0.0,
            "nonfinite_count": nonfinite_count,
        }, f"{name}_nonfinite_block"
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    keep = singular_values > eps
    rank = int(np.sum(keep))
    if rank == 0:
        return np.zeros((values.shape[0], 0)), {
            "block": name,
            "rank": 0,
            "retained_dimensions": 0,
            "pre_frobenius_norm": pre_norm,
            "post_frobenius_norm": 0.0,
            "singular_values": singular_values.astype(float).tolist(),
            "nonfinite_count": 0,
        }, f"{name}_zero_rank"
    retained_singular = singular_values[keep]
    whitened = left[:, keep] * (
        retained_singular / np.maximum(retained_singular, eps)
    )
    whitened_norm = float(np.linalg.norm(whitened, ord="fro"))
    whitened = whitened / max(whitened_norm, eps)
    return whitened, {
        "block": name,
        "rank": rank,
        "retained_dimensions": rank,
        "pre_frobenius_norm": pre_norm,
        "pre_scale_whitened_frobenius_norm": whitened_norm,
        "post_frobenius_norm": float(np.linalg.norm(whitened, ord="fro")),
        "singular_values": singular_values.astype(float).tolist(),
        "nonfinite_count": 0,
    }, None


def _fusion_representation(
    family: str,
    block_names: Sequence[str],
    representations: Mapping[str, _CandidateRepresentation],
    *,
    k_components: int,
    eps: float,
) -> _CandidateRepresentation:
    ineligible = [name for name in block_names if not representations[name].eligible]
    if ineligible:
        reason = "required_block_ineligible:" + ",".join(ineligible)
        return _CandidateRepresentation(
            np.zeros((next(iter(representations.values())).matrix.shape[0], 0)),
            False,
            reason,
            {
                "view": family,
                "view_eligible": False,
                "degenerate_reason": reason,
                "block_order": list(block_names),
                "ineligible_blocks": ineligible,
                "blocks": [],
                "rank": 0,
                "available_dimension": 0,
                "pca_components": 0,
                "nonfinite_count": 0,
            },
        )
    whitened_blocks: list[np.ndarray] = []
    block_diagnostics: list[dict[str, Any]] = []
    for name in block_names:
        whitened, diagnostics, reason = _whiten_block(
            name,
            representations[name].matrix,
            eps=eps,
        )
        block_diagnostics.append(diagnostics)
        if reason is not None:
            return _CandidateRepresentation(
                np.zeros((whitened.shape[0], 0)),
                False,
                reason,
                {
                    "view": family,
                    "view_eligible": False,
                    "degenerate_reason": reason,
                    "block_order": list(block_names),
                    "ineligible_blocks": [name],
                    "blocks": block_diagnostics,
                    "rank": 0,
                    "available_dimension": 0,
                    "pca_components": 0,
                    "nonfinite_count": int(diagnostics["nonfinite_count"]),
                },
            )
        whitened_blocks.append(whitened)
    concatenated = np.concatenate(whitened_blocks, axis=1)
    reduced, pca_diagnostics = _deterministic_pca(
        concatenated,
        max_components=2 * k_components,
        eps=eps,
    )
    eligible = pca_diagnostics["rank"] > 0 and reduced.shape[1] > 0
    reason = None if eligible else f"{family}_zero_rank"
    return _CandidateRepresentation(
        reduced,
        eligible,
        reason,
        {
            "view": family,
            "view_eligible": eligible,
            "degenerate_reason": reason,
            "block_order": list(block_names),
            "ineligible_blocks": [],
            "blocks": block_diagnostics,
            **pca_diagnostics,
        },
    )


def _ineligible_candidate(
    logical_id: str,
    family: str,
    algorithm: str,
    reason: str,
) -> LogicalCandidate:
    return LogicalCandidate(
        logical_id=logical_id,
        family=family,
        algorithm=algorithm,
        eligible=False,
        canonical_id=None,
        raw_labels=None,
        labels=None,
        cluster_sizes=(),
        cost_matrix=None,
        ineligible_reason=reason,
        diagnostics={
            "logical_id": logical_id,
            "family": family,
            "algorithm": algorithm,
            "eligible": False,
            "ineligible_reason": reason,
        },
    )


def _candidate_from_labels(
    *,
    logical_id: str,
    family: str,
    algorithm: str,
    raw_labels: np.ndarray,
    repaired_labels: np.ndarray,
    cost: np.ndarray,
    k_components: int,
    algorithm_diagnostics: Mapping[str, Any],
) -> LogicalCandidate:
    raw_sizes = np.bincount(raw_labels, minlength=k_components).astype(int)
    repaired_sizes = np.bincount(repaired_labels, minlength=k_components).astype(int)
    diagnostics = {
        "logical_id": logical_id,
        "family": family,
        "algorithm": algorithm,
        "eligible": True,
        "raw_labels": raw_labels.astype(int).tolist(),
        "repaired_labels": repaired_labels.astype(int).tolist(),
        "raw_cluster_sizes": raw_sizes.tolist(),
        "repaired_cluster_sizes": repaired_sizes.tolist(),
        "cost_shape": [int(cost.shape[0]), int(cost.shape[1])],
        "cost_min": float(np.min(cost)),
        "cost_max": float(np.max(cost)),
        **dict(algorithm_diagnostics),
    }
    return LogicalCandidate(
        logical_id=logical_id,
        family=family,
        algorithm=algorithm,
        eligible=True,
        canonical_id=None,
        raw_labels=raw_labels,
        labels=repaired_labels,
        cluster_sizes=tuple(repaired_sizes.tolist()),
        cost_matrix=cost,
        ineligible_reason=None,
        diagnostics=diagnostics,
    )


def _cluster_candidate(
    logical_id: str,
    family: str,
    algorithm: str,
    representation: np.ndarray,
    *,
    k_components: int,
    m_e: int,
    seed: int,
    n_init: int,
) -> LogicalCandidate:
    matrix = np.asarray(representation, dtype=float)
    if algorithm == "kmeans":
        random_state = _stable_seed(seed, logical_id)
        model = KMeans(
            n_clusters=k_components,
            n_init=n_init,
            random_state=random_state,
            algorithm="lloyd",
        )
        raw_labels = model.fit_predict(matrix).astype(np.int64)
        cost = np.square(model.transform(matrix))
        diagnostics = {
            "n_init": int(n_init),
            "random_state": int(random_state),
            "kmeans_algorithm": "lloyd",
        }
    elif algorithm == "ward":
        model = AgglomerativeClustering(
            n_clusters=k_components,
            metric="euclidean",
            linkage="ward",
        )
        raw_labels = model.fit_predict(matrix).astype(np.int64)
        cost = _centroid_cost_matrix(matrix, raw_labels, k_components)
        diagnostics = {"metric": "euclidean", "linkage": "ward"}
    else:
        raise ValueError(f"unsupported clustering algorithm: {algorithm}")
    repaired = solve_constrained_assignment(
        -cost,
        m_e,
        current_labels=raw_labels,
    )
    return _candidate_from_labels(
        logical_id=logical_id,
        family=family,
        algorithm=algorithm,
        raw_labels=raw_labels,
        repaired_labels=repaired,
        cost=cost,
        k_components=k_components,
        algorithm_diagnostics=diagnostics,
    )


def _random_candidate(
    logical_id: str,
    *,
    n_centers: int,
    k_components: int,
    m_e: int,
    seed: int,
) -> LogicalCandidate:
    rng = np.random.default_rng(_stable_seed(seed, logical_id))
    sizes = np.full(k_components, n_centers // k_components, dtype=int)
    remainder_clusters = rng.permutation(k_components)[: n_centers % k_components]
    sizes[remainder_clusters] += 1
    if np.any(sizes < m_e):
        return _ineligible_candidate(
            logical_id,
            "balanced_random",
            "balanced_random",
            "infeasible_balanced_random_sizes",
        )
    order = rng.permutation(n_centers)
    labels = np.empty(n_centers, dtype=np.int64)
    offset = 0
    for cluster, size in enumerate(sizes):
        labels[order[offset : offset + size]] = cluster
        offset += int(size)
    cost = np.ones((n_centers, k_components), dtype=float)
    cost[np.arange(n_centers), labels] = 0.0
    return _candidate_from_labels(
        logical_id=logical_id,
        family="balanced_random",
        algorithm="balanced_random",
        raw_labels=labels,
        repaired_labels=labels,
        cost=cost,
        k_components=k_components,
        algorithm_diagnostics={
            "random_state": int(_stable_seed(seed, logical_id)),
            "balanced_sizes": sizes.astype(int).tolist(),
        },
    )


def _deduplicate(
    logical_candidates: Sequence[LogicalCandidate],
    *,
    k_components: int,
) -> tuple[tuple[LogicalCandidate, ...], tuple[CanonicalCandidate, ...]]:
    key_to_id: dict[tuple[int, ...], str] = {}
    labels_by_id: dict[str, np.ndarray] = {}
    logical_ids_by_id: dict[str, list[str]] = {}
    updated: list[LogicalCandidate] = []
    for candidate in logical_candidates:
        if not candidate.eligible or candidate.labels is None:
            updated.append(candidate)
            continue
        key = canonical_partition_key(candidate.labels)
        canonical_id = key_to_id.get(key)
        if canonical_id is None:
            canonical_id = f"canonical_{len(key_to_id) + 1:03d}"
            key_to_id[key] = canonical_id
            labels_by_id[canonical_id] = np.asarray(key, dtype=np.int64)
            logical_ids_by_id[canonical_id] = []
        logical_ids_by_id[canonical_id].append(candidate.logical_id)
        updated.append(replace(candidate, canonical_id=canonical_id))
    canonical: list[CanonicalCandidate] = []
    for canonical_id in key_to_id.values():
        labels = labels_by_id[canonical_id]
        canonical.append(
            CanonicalCandidate(
                canonical_id=canonical_id,
                labels=labels,
                cluster_sizes=tuple(
                    np.bincount(labels, minlength=k_components).astype(int).tolist()
                ),
                logical_ids=tuple(logical_ids_by_id[canonical_id]),
            )
        )
    return tuple(updated), tuple(canonical)


def build_candidate_pool(
    data: ObservedDataView,
    *,
    config: CandidatePoolConfig,
    seed: int,
) -> CandidatePool:
    """Build the fixed 17-record logical pool from observed training rows only."""

    observed = _require_observed(data)
    bundle = build_view_bundle(
        observed,
        k_components=config.k_components,
        eps=config.eps,
    )
    k_components, m_e = _validate_config(config, bundle.centers.size)
    if k_components * m_e > bundle.centers.size:
        reason = "infeasible_min_cluster_capacity"
        logical = tuple(
            _ineligible_candidate(logical_id, family, algorithm, reason)
            for logical_id, family, algorithm in LOGICAL_CANDIDATE_SPECS
        )
        return CandidatePool(
            centers=bundle.centers,
            logical_candidates=logical,
            canonical_candidates=(),
            m_e=m_e,
            status="ineligible",
            failure_reason=reason,
            representation_diagnostics={
                "m_e_formula": "max(2, ceil(E*pi_cem))",
                "n_centers": int(bundle.centers.size),
                "k_components": k_components,
                "pi_cem": float(config.pi_cem),
                "m_e": m_e,
                "failure_reason": reason,
            },
        )

    representations: dict[str, _CandidateRepresentation] = {
        "outcome": _direct_representation("outcome", bundle.outcome, eps=config.eps),
        "treatment": _direct_representation("treatment", bundle.treatment, eps=config.eps),
        "covariate": _preprocess_covariate(
            bundle.covariate,
            k_components=k_components,
            eps=config.eps,
            correlation_threshold=config.correlation_threshold,
        ),
    }
    for family, block_names in _FUSION_BLOCKS.items():
        representations[family] = _fusion_representation(
            family,
            block_names,
            representations,
            k_components=k_components,
            eps=config.eps,
        )

    logical_candidates: list[LogicalCandidate] = []
    for logical_id, family, algorithm in LOGICAL_CANDIDATE_SPECS:
        if family == "balanced_random":
            logical_candidates.append(
                _random_candidate(
                    logical_id,
                    n_centers=bundle.centers.size,
                    k_components=k_components,
                    m_e=m_e,
                    seed=seed,
                )
            )
            continue
        representation = representations[family]
        if not representation.eligible:
            logical_candidates.append(
                _ineligible_candidate(
                    logical_id,
                    family,
                    algorithm,
                    representation.reason or "representation_ineligible",
                )
            )
            continue
        try:
            logical_candidates.append(
                _cluster_candidate(
                    logical_id,
                    family,
                    algorithm,
                    representation.matrix,
                    k_components=k_components,
                    m_e=m_e,
                    seed=seed,
                    n_init=config.kmeans_n_init,
                )
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            logical_candidates.append(
                _ineligible_candidate(
                    logical_id,
                    family,
                    algorithm,
                    f"clustering_failed:{type(error).__name__}:{error}",
                )
            )
    logical, canonical = _deduplicate(
        logical_candidates,
        k_components=k_components,
    )
    diagnostics = {
        family: dict(representation.diagnostics)
        for family, representation in representations.items()
    }
    diagnostics["pool"] = {
        "m_e_formula": "max(2, ceil(E*pi_cem))",
        "n_centers": int(bundle.centers.size),
        "k_components": k_components,
        "pi_cem": float(config.pi_cem),
        "m_e": m_e,
        "logical_candidate_count": len(logical),
        "eligible_logical_count": int(sum(candidate.eligible for candidate in logical)),
        "canonical_candidate_count": len(canonical),
        "fit_fanout": "canonical_only",
    }
    return CandidatePool(
        centers=bundle.centers,
        logical_candidates=logical,
        canonical_candidates=canonical,
        m_e=m_e,
        status="eligible" if canonical else "ineligible",
        failure_reason=None if canonical else "no_eligible_canonical_candidate",
        representation_diagnostics=diagnostics,
    )


__all__ = [
    "LOGICAL_CANDIDATE_SPECS",
    "CandidatePoolConfig",
    "LogicalCandidate",
    "CanonicalCandidate",
    "CandidatePool",
    "canonical_partition_key",
    "repair_minimum_cluster_labels",
    "build_candidate_pool",
]
