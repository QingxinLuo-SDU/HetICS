"""Exact CEM assignments subject to a global minimum cluster size."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from numbers import Integral, Real

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack


_BINARY_TOLERANCE = 1e-6
_MAX_EXACT_DP_ROWS = 512
_MAX_EXACT_DP_TRANSITIONS = 2_000_000


def _normalized_problem(
    scores: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    """Scale and row-center scores without changing assignment ordering."""

    row_minima = np.min(scores, axis=1, keepdims=True)
    row_maxima = np.max(scores, axis=1, keepdims=True)
    row_offsets = np.where(
        row_minima >= 0.0,
        row_minima,
        np.where(row_maxima <= 0.0, row_maxima, 0.0),
    )
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            centered_scores = scores - row_offsets
    except FloatingPointError as error:
        raise RuntimeError(
            "scores cannot be safely row-centered for MILP normalization"
        ) from error

    scale = float(np.max(np.abs(centered_scores)))
    if scale == 0.0:
        return centered_scores, 0.0

    normalized_scores = centered_scores / scale
    with np.errstate(over="ignore"):
        normalized_tolerance = float(np.divide(tolerance, scale))
    if not np.isfinite(normalized_tolerance):
        normalized_tolerance = 2.0 * scores.shape[0]
    return normalized_scores, normalized_tolerance


def _milp_solution(
    scores: np.ndarray,
    minimum: int,
    fixed: dict[int, int],
    score_floor: float | None,
) -> tuple[np.ndarray, float] | None:
    """Return a decoded global MILP solution, or ``None`` if infeasible."""

    n_centers, k_components = scores.shape
    n_variables = n_centers * k_components
    center_matrix = lil_matrix((n_centers, n_variables))
    class_matrix = lil_matrix((k_components, n_variables))
    for center in range(n_centers):
        for cluster in range(k_components):
            index = center * k_components + cluster
            center_matrix[center, index] = 1.0
            class_matrix[cluster, index] = 1.0

    matrices = [center_matrix.tocsr(), class_matrix.tocsr()]
    lower = [np.ones(n_centers), np.full(k_components, minimum)]
    upper = [np.ones(n_centers), np.full(k_components, np.inf)]
    for center, cluster in fixed.items():
        row = lil_matrix((1, n_variables))
        row[0, center * k_components + cluster] = 1.0
        matrices.append(row.tocsr())
        lower.append(np.ones(1))
        upper.append(np.ones(1))
    if score_floor is not None:
        matrices.append(scores.reshape(1, -1))
        lower.append(np.array([score_floor]))
        upper.append(np.array([np.inf]))

    result = milp(
        c=-scores.ravel(),
        integrality=np.ones(n_variables),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(
            vstack(matrices),
            np.concatenate(lower),
            np.concatenate(upper),
        ),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return None

    assignments = result.x.reshape(n_centers, k_components)
    labels = np.argmax(assignments, axis=1).astype(np.int64)
    selected = assignments[np.arange(n_centers), labels]
    if (
        np.any(selected < 1.0 - _BINARY_TOLERANCE)
        or np.any(np.abs(assignments.sum(axis=1) - 1.0) > _BINARY_TOLERANCE)
    ):
        raise RuntimeError("MILP returned a non-binary assignment")
    for center, cluster in fixed.items():
        if labels[center] != cluster:
            raise RuntimeError("MILP result violated a fixed assignment")

    counts = np.bincount(labels, minlength=k_components)
    if np.any(counts < minimum):
        raise RuntimeError("MILP result violated the minimum cluster size")
    value = float(np.sum(scores[np.arange(n_centers), labels]))
    if score_floor is not None and value < score_floor:
        return None
    return labels, value


def _solve_milp_proposal(
    scores: np.ndarray,
    minimum: int,
    current_labels: np.ndarray | None,
    tolerance: float,
) -> np.ndarray | None:
    """Build the fast normalized-MILP proposal required by the public path."""

    try:
        normalized_scores, normalized_tolerance = _normalized_problem(
            scores,
            tolerance,
        )
        optimum_solution = _milp_solution(
            normalized_scores,
            minimum,
            fixed={},
            score_floor=None,
        )
        if optimum_solution is None:
            return None
        _, normalized_optimum = optimum_solution
        normalized_score_floor = normalized_optimum - normalized_tolerance

        n_centers, k_components = scores.shape
        if current_labels is not None:
            current_counts = np.bincount(current_labels, minlength=k_components)
            current_value = float(
                np.sum(normalized_scores[np.arange(n_centers), current_labels])
            )
            if (
                np.all(current_counts >= minimum)
                and current_value >= normalized_score_floor
            ):
                return current_labels.copy()

        fixed: dict[int, int] = {}
        for center in range(n_centers):
            for cluster in range(k_components):
                candidate_fixed = {**fixed, center: cluster}
                candidate = _milp_solution(
                    normalized_scores,
                    minimum,
                    fixed=candidate_fixed,
                    score_floor=normalized_score_floor,
                )
                if candidate is not None and candidate[1] >= normalized_score_floor:
                    fixed[center] = cluster
                    break
            else:
                return None

        labels = np.fromiter(
            (fixed[center] for center in range(n_centers)),
            dtype=np.int64,
            count=n_centers,
        )
        counts = np.bincount(labels, minlength=k_components)
        if np.any(counts < minimum):
            return None
        value = float(np.sum(normalized_scores[np.arange(n_centers), labels]))
        if value < normalized_score_floor:
            return None
        return labels
    except (FloatingPointError, RuntimeError, ValueError):
        return None


def _solve_exact_assignment(
    scores: np.ndarray,
    minimum: int,
    current_labels: np.ndarray | None,
    tolerance: float,
) -> np.ndarray:
    """Solve and lexicographically certify the assignment using exact DP."""

    n_centers, k_components = scores.shape
    state_bound = (n_centers + 1) * (minimum + 1) ** k_components
    transition_bound = state_bound * k_components
    if (
        n_centers > _MAX_EXACT_DP_ROWS
        or transition_bound > _MAX_EXACT_DP_TRANSITIONS
    ):
        raise ValueError(
            "exact assignment DP state budget exceeded; "
            f"estimated transitions={transition_bound}"
        )

    exact_scores = tuple(
        tuple(Fraction(*float(value).as_integer_ratio()) for value in score_row)
        for score_row in scores
    )
    exact_tolerance = Fraction(*tolerance.as_integer_ratio())

    @lru_cache(maxsize=None)
    def best_remaining(
        row_index: int,
        capped_counts: tuple[int, ...],
    ) -> Fraction | None:
        remaining_rows = n_centers - row_index
        deficit = sum(minimum - count for count in capped_counts)
        if deficit > remaining_rows:
            return None
        if row_index == n_centers:
            return (
                Fraction(0)
                if all(count == minimum for count in capped_counts)
                else None
            )

        best: Fraction | None = None
        for cluster in range(k_components):
            next_counts_list = list(capped_counts)
            next_counts_list[cluster] = min(
                minimum,
                next_counts_list[cluster] + 1,
            )
            tail = best_remaining(row_index + 1, tuple(next_counts_list))
            if tail is None:
                continue
            candidate = exact_scores[row_index][cluster] + tail
            if best is None or candidate > best:
                best = candidate
        return best

    initial_counts = (0,) * k_components
    exact_optimum = best_remaining(0, initial_counts)
    if exact_optimum is None:
        raise RuntimeError("exact constrained assignment DP found no solution")
    score_floor = exact_optimum - exact_tolerance

    if current_labels is not None:
        current_counts = np.bincount(current_labels, minlength=k_components)
        current_score = sum(
            (
                exact_scores[center][int(cluster)]
                for center, cluster in enumerate(current_labels)
            ),
            start=Fraction(0),
        )
        if np.all(current_counts >= minimum) and current_score >= score_floor:
            return current_labels.copy()

    labels = np.empty(n_centers, dtype=np.int64)
    prefix_counts = initial_counts
    prefix_score = Fraction(0)
    for center in range(n_centers):
        for cluster in range(k_components):
            next_counts_list = list(prefix_counts)
            next_counts_list[cluster] = min(
                minimum,
                next_counts_list[cluster] + 1,
            )
            next_counts = tuple(next_counts_list)
            tail = best_remaining(center + 1, next_counts)
            if tail is None:
                continue
            candidate_score = prefix_score + exact_scores[center][cluster] + tail
            if candidate_score >= score_floor:
                labels[center] = cluster
                prefix_counts = next_counts
                prefix_score += exact_scores[center][cluster]
                break
        else:
            raise RuntimeError(
                "exact DP failed to construct a lexicographic "
                "tolerance-optimal assignment"
            )

    counts = np.bincount(labels, minlength=k_components)
    if np.any(counts < minimum):
        raise RuntimeError("returned assignment violates the minimum cluster size")
    exact_value = sum(
        (
            exact_scores[center][int(cluster)]
            for center, cluster in enumerate(labels)
        ),
        start=Fraction(0),
    )
    if exact_value < score_floor:
        raise RuntimeError(
            "returned assignment is not within tolerance of the exact optimum"
        )
    return labels


def solve_constrained_assignment(
    scores: np.ndarray,
    min_cluster_size: int,
    current_labels: np.ndarray | None = None,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Globally maximize scores under per-cluster lower bounds."""

    try:
        score_array = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be a finite two-dimensional matrix") from error
    if (
        score_array.ndim != 2
        or score_array.shape[0] == 0
        or score_array.shape[1] == 0
    ):
        raise ValueError("scores must be a two-dimensional non-empty matrix")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values")

    if (
        isinstance(min_cluster_size, (bool, np.bool_))
        or not isinstance(min_cluster_size, Integral)
        or min_cluster_size < 0
    ):
        raise ValueError("min_cluster_size must be a non-negative integer")
    minimum = int(min_cluster_size)
    n_centers, k_components = score_array.shape
    if k_components * minimum > n_centers:
        raise ValueError(
            "minimum cluster capacity is infeasible: "
            "number of clusters times min_cluster_size exceeds centers"
        )

    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, Real)
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be finite and non-negative")
    objective_tolerance = float(tolerance)

    labels_array: np.ndarray | None = None
    if current_labels is not None:
        labels_array = np.asarray(current_labels)
        if labels_array.ndim != 1 or labels_array.shape != (n_centers,):
            raise ValueError(
                "current_labels must be a one-dimensional array with one "
                "entry per center"
            )
        if (
            np.issubdtype(labels_array.dtype, np.bool_)
            or not np.issubdtype(labels_array.dtype, np.integer)
            or np.any(labels_array < 0)
            or np.any(labels_array >= k_components)
        ):
            raise ValueError(
                "current_labels must contain integer cluster indices in range"
            )
        labels_array = labels_array.astype(np.int64, copy=False)

    milp_proposal = _solve_milp_proposal(
        score_array,
        minimum,
        labels_array,
        objective_tolerance,
    )
    exact_labels = _solve_exact_assignment(
        score_array,
        minimum,
        labels_array,
        objective_tolerance,
    )
    if milp_proposal is not None and np.array_equal(milp_proposal, exact_labels):
        return milp_proposal
    return exact_labels
