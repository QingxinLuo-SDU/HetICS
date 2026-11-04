"""Robust multiview standardization, stability, guard, and selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def _readonly_labels(labels: np.ndarray) -> np.ndarray:
    values = np.array(labels, dtype=np.int64, copy=True)
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class RobustScoreTransform:
    center: float
    mad_scale: float
    iqr_scale: float
    scale: float
    finite_count: int

    def transform(self, value: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(value, dtype=float)
        transformed = np.clip((values - self.center) / self.scale, -5.0, 5.0)
        if values.ndim == 0:
            return float(transformed)
        return transformed


def fit_robust_score_transform(values: np.ndarray) -> RobustScoreTransform:
    """Fit the frozen median/MAD/IQR transform to finite entries exactly once."""

    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("robust score reference must contain at least one finite value")
    center = float(np.median(finite))
    mad_scale = float(1.4826 * np.median(np.abs(finite - center)))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    scale = float(max(mad_scale, iqr_scale, 1.0e-12))
    return RobustScoreTransform(center, mad_scale, iqr_scale, scale, int(finite.size))


@dataclass(frozen=True)
class RawViewScores:
    logical_id: str
    stage: str
    y: float
    x: float
    f: float
    status: str = "eligible"
    reason: str | None = None
    primary_y_risk: float | None = None
    primary_x_risk: float | None = None


@dataclass(frozen=True)
class StandardizedViewScores:
    logical_id: str
    stage: str
    z_y: float | None
    z_x: float | None
    z_f: float | None
    status: str
    reason: str | None


def standardize_multiview_scores(
    records: Sequence[RawViewScores],
) -> tuple[tuple[StandardizedViewScores, ...], dict[str, Any]]:
    """Fit one transform/view over the finite eligible pre/post union and reuse it."""

    frozen_records = tuple(records)
    reference = tuple(
        record
        for record in frozen_records
        if record.status == "eligible" and np.all(np.isfinite([record.y, record.x, record.f]))
    )
    if not reference:
        standardized = tuple(
            StandardizedViewScores(
                record.logical_id,
                record.stage,
                None,
                None,
                None,
                "ineligible",
                record.reason or "no_finite_reference_scores",
            )
            for record in frozen_records
        )
        return standardized, {
            "fit_count_by_view": {"y": 0, "x": 0, "f": 0},
            "finite_reference_count_by_view": {"y": 0, "x": 0, "f": 0},
            "transforms": {},
        }
    transforms = {
        "y": fit_robust_score_transform(np.array([record.y for record in reference])),
        "x": fit_robust_score_transform(np.array([record.x for record in reference])),
        "f": fit_robust_score_transform(np.array([record.f for record in reference])),
    }
    output: list[StandardizedViewScores] = []
    for record in frozen_records:
        finite = bool(np.all(np.isfinite([record.y, record.x, record.f])))
        if record.status != "eligible" or not finite:
            output.append(
                StandardizedViewScores(
                    record.logical_id,
                    record.stage,
                    None,
                    None,
                    None,
                    "ineligible",
                    record.reason or "nonfinite_raw_score",
                )
            )
            continue
        output.append(
            StandardizedViewScores(
                record.logical_id,
                record.stage,
                float(transforms["y"].transform(record.y)),
                float(transforms["x"].transform(record.x)),
                float(transforms["f"].transform(record.f)),
                "eligible",
                None,
            )
        )
    diagnostics = {
        "fit_count_by_view": {"y": 1, "x": 1, "f": 1},
        "finite_reference_count_by_view": {
            name: transform.finite_count for name, transform in transforms.items()
        },
        "transforms": {
            name: {
                "center": transform.center,
                "mad_scale": transform.mad_scale,
                "iqr_scale": transform.iqr_scale,
                "scale": transform.scale,
                "winsor_limits": [-5.0, 5.0],
            }
            for name, transform in transforms.items()
        },
    }
    return tuple(output), diagnostics


def _validated_partition(labels: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(labels)
    if (
        values.ndim != 1
        or values.size == 0
        or np.issubdtype(values.dtype, np.bool_)
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be a non-empty integer partition")
    return values.astype(np.int64, copy=True)


def adjusted_rand_stability(left: np.ndarray, right: np.ndarray) -> float:
    """Return ARI for diagnostics only; formal selection must not use it."""

    first = _validated_partition(left, name="left")
    second = _validated_partition(right, name="right")
    if first.shape != second.shape:
        raise ValueError("partitions must have the same number of centers")
    value = float(adjusted_rand_score(first, second))
    if not np.isfinite(value) or value < -1.0 - 1.0e-12 or value > 1.0 + 1.0e-12:
        raise FloatingPointError("adjusted Rand stability is outside its finite range")
    return float(np.clip(value, -1.0, 1.0))


def permutation_aligned_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Return the maximum center agreement after a global label permutation."""

    first = _validated_partition(left, name="left")
    second = _validated_partition(right, name="right")
    if first.shape != second.shape:
        raise ValueError("partitions must have the same number of centers")
    first_values, first_codes = np.unique(first, return_inverse=True)
    second_values, second_codes = np.unique(second, return_inverse=True)
    dimension = max(int(first_values.size), int(second_values.size))
    contingency = np.zeros((dimension, dimension), dtype=np.int64)
    np.add.at(contingency, (first_codes, second_codes), 1)
    rows, columns = linear_sum_assignment(-contingency)
    matched = int(contingency[rows, columns].sum())
    return float(matched / first.size)


def composite_score(
    z_y: float,
    z_x: float,
    z_f: float,
    stability_agreement: float,
) -> float:
    values = np.asarray([z_y, z_x, z_f, stability_agreement], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("composite inputs must be finite")
    if stability_agreement < -1.0e-12 or stability_agreement > 1.0 + 1.0e-12:
        raise ValueError("stability_agreement must lie in [0, 1]")
    agreement = float(np.clip(stability_agreement, 0.0, 1.0))
    return float((z_y + z_x + z_f) / 3.0 - (1.0 - agreement))


@dataclass(frozen=True)
class GuardDecision:
    eligible: bool
    status: str
    labels: np.ndarray | None
    chosen_stage: str | None
    score: float | None
    stability_agreement: float | None
    stability_ari: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels is not None:
            object.__setattr__(self, "labels", _readonly_labels(self.labels))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @classmethod
    def ineligible(
        cls,
        reason: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> "GuardDecision":
        detail = {} if diagnostics is None else dict(diagnostics)
        if "reason" not in detail:
            detail["reason"] = reason
        return cls(
            False,
            "ineligible_no_legal_stable_stage",
            None,
            None,
            None,
            None,
            None,
            detail,
        )

    @classmethod
    def retained(
        cls,
        labels: np.ndarray,
        *,
        stage: str,
        score: float,
        stability_agreement: float,
        stability_ari: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> "GuardDecision":
        return cls(
            True,
            "retained",
            labels,
            stage,
            float(score),
            float(stability_agreement),
            float(stability_ari),
            {} if diagnostics is None else diagnostics,
        )


def _minimum_cluster_legal(labels: np.ndarray, k_components: int, minimum: int) -> bool:
    if k_components < 1 or minimum < 1:
        raise ValueError("k_components and minimum_cluster_size must be positive")
    if np.any(labels < 0) or np.any(labels >= k_components):
        return False
    sizes = np.bincount(labels, minlength=k_components)
    return bool(sizes.shape == (k_components,) and np.all(sizes >= minimum))


def apply_refinement_guard(
    pre_labels: np.ndarray,
    post_labels: np.ndarray,
    *,
    companion_pre_labels: np.ndarray,
    companion_post_labels: np.ndarray,
    pre_score: float,
    post_score: float,
    k_components: int,
    minimum_cluster_size: int,
    pre_numerically_legal: bool = True,
    post_numerically_legal: bool = True,
    score_tolerance: float = 1.0e-8,
    stability_threshold: float = 0.80,
) -> GuardDecision:
    """Compatibility wrapper; the formal pipeline uses the atomic pair guard."""

    first, _ = apply_atomic_refinement_guard(
        pre_labels,
        post_labels,
        right_pre_labels=companion_pre_labels,
        right_post_labels=companion_post_labels,
        left_pre_score=pre_score,
        left_post_score=post_score,
        right_pre_score=pre_score,
        right_post_score=post_score,
        k_components=k_components,
        minimum_cluster_size=minimum_cluster_size,
        left_pre_numerically_legal=pre_numerically_legal,
        left_post_numerically_legal=post_numerically_legal,
        right_pre_numerically_legal=pre_numerically_legal,
        right_post_numerically_legal=post_numerically_legal,
        score_tolerance=score_tolerance,
        stability_threshold=stability_threshold,
    )
    return first


def apply_atomic_refinement_guard(
    left_pre_labels: np.ndarray,
    left_post_labels: np.ndarray,
    *,
    right_pre_labels: np.ndarray,
    right_post_labels: np.ndarray,
    left_pre_score: float,
    left_post_score: float,
    right_pre_score: float,
    right_post_score: float,
    k_components: int,
    minimum_cluster_size: int,
    left_pre_numerically_legal: bool = True,
    left_post_numerically_legal: bool = True,
    right_pre_numerically_legal: bool = True,
    right_post_numerically_legal: bool = True,
    score_tolerance: float = 1.0e-8,
    stability_threshold: float = 0.80,
) -> tuple[GuardDecision, GuardDecision]:
    """Choose post, pre, or rejection jointly for both split directions."""

    left_pre = _validated_partition(left_pre_labels, name="left_pre_labels")
    left_post = _validated_partition(left_post_labels, name="left_post_labels")
    right_pre = _validated_partition(right_pre_labels, name="right_pre_labels")
    right_post = _validated_partition(right_post_labels, name="right_post_labels")
    if not (left_pre.shape == left_post.shape == right_pre.shape == right_post.shape):
        raise ValueError("all guard partitions must align by center order")
    if score_tolerance < 0.0 or not np.isfinite(score_tolerance):
        raise ValueError("score_tolerance must be finite and nonnegative")
    if not np.isfinite(stability_threshold) or not 0.0 <= stability_threshold <= 1.0:
        raise ValueError("stability_threshold must lie in [0, 1]")

    pre_agreement = permutation_aligned_agreement(left_pre, right_pre)
    post_agreement = permutation_aligned_agreement(left_post, right_post)
    pre_ari = adjusted_rand_stability(left_pre, right_pre)
    post_ari = adjusted_rand_stability(left_post, right_post)

    def predicates(
        labels: np.ndarray,
        score: float,
        *,
        numerically_legal: bool,
        agreement: float,
        baseline_score: float | None,
    ) -> dict[str, bool]:
        result = {
            "numerical_legal": bool(numerically_legal and np.isfinite(score)),
            "minimum_cluster_legal": _minimum_cluster_legal(
                labels,
                k_components,
                minimum_cluster_size,
            ),
            "stable": bool(agreement >= stability_threshold),
        }
        if baseline_score is not None:
            result["score_noninferior"] = bool(
                np.isfinite(baseline_score)
                and np.isfinite(score)
                and score >= baseline_score - score_tolerance
            )
        return result

    post_by_direction = {
        "A->B": predicates(
            left_post,
            left_post_score,
            numerically_legal=left_post_numerically_legal,
            agreement=post_agreement,
            baseline_score=left_pre_score,
        ),
        "B->A": predicates(
            right_post,
            right_post_score,
            numerically_legal=right_post_numerically_legal,
            agreement=post_agreement,
            baseline_score=right_pre_score,
        ),
    }
    pre_by_direction = {
        "A->B": predicates(
            left_pre,
            left_pre_score,
            numerically_legal=left_pre_numerically_legal,
            agreement=pre_agreement,
            baseline_score=None,
        ),
        "B->A": predicates(
            right_pre,
            right_pre_score,
            numerically_legal=right_pre_numerically_legal,
            agreement=pre_agreement,
            baseline_score=None,
        ),
    }

    predicate_order = (
        "numerical_legal",
        "minimum_cluster_legal",
        "score_noninferior",
        "stable",
    )

    def failed(checks: Mapping[str, bool]) -> list[str]:
        return [name for name in predicate_order if name in checks and not checks[name]]

    failed_post = {
        "A->B": failed(post_by_direction["A->B"]),
        "B->A": failed(post_by_direction["B->A"]),
    }
    failed_pre = {
        "A->B": failed(pre_by_direction["A->B"]),
        "B->A": failed(pre_by_direction["B->A"]),
    }
    post_legal = not failed_post["A->B"] and not failed_post["B->A"]
    pre_legal = not failed_pre["A->B"] and not failed_pre["B->A"]
    chosen_stage = "post" if post_legal else "pre" if pre_legal else None
    diagnostics = {
        "atomic_pair_decision": chosen_stage,
        "post_predicates_by_direction": post_by_direction,
        "failed_post_predicates_by_direction": failed_post,
        "pre_predicates_by_direction": pre_by_direction,
        "failed_pre_predicates_by_direction": failed_pre,
        "post_predicates": post_by_direction["A->B"],
        "failed_post_predicates": failed_post["A->B"],
        "pre_predicates": pre_by_direction["A->B"],
        "failed_pre_predicates": failed_pre["A->B"],
        "post_stability_agreement": float(post_agreement),
        "pre_stability_agreement": float(pre_agreement),
        "post_stability_ari_diagnostic": float(post_ari),
        "pre_stability_ari_diagnostic": float(pre_ari),
        "post_stability_ari": float(post_ari),
        "pre_stability_ari": float(pre_ari),
        "stability_metric_used_for_selection": "permutation_aligned_agreement",
        "ari_used_for_selection": False,
        "score_tolerance": float(score_tolerance),
        "stability_threshold": float(stability_threshold),
    }
    if chosen_stage is None:
        reason = "neither paired post nor paired pre stage is jointly legal and stable"
        return (
            GuardDecision.ineligible(reason, diagnostics=diagnostics),
            GuardDecision.ineligible(reason, diagnostics=diagnostics),
        )

    agreement = post_agreement if chosen_stage == "post" else pre_agreement
    ari = post_ari if chosen_stage == "post" else pre_ari
    left_labels = left_post if chosen_stage == "post" else left_pre
    right_labels = right_post if chosen_stage == "post" else right_pre
    left_score = left_post_score if chosen_stage == "post" else left_pre_score
    right_score = right_post_score if chosen_stage == "post" else right_pre_score
    return (
        GuardDecision.retained(
            left_labels,
            stage=chosen_stage,
            score=left_score,
            stability_agreement=agreement,
            stability_ari=ari,
            diagnostics=diagnostics,
        ),
        GuardDecision.retained(
            right_labels,
            stage=chosen_stage,
            score=right_score,
            stability_agreement=agreement,
            stability_ari=ari,
            diagnostics=diagnostics,
        ),
    )


@dataclass(frozen=True)
class SelectionResult:
    status: str
    logical_id: str | None
    labels: np.ndarray | None
    score: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels is not None:
            object.__setattr__(self, "labels", _readonly_labels(self.labels))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def select_guarded_candidate(
    decisions: Mapping[str, GuardDecision],
    *,
    logical_order: Sequence[str],
) -> SelectionResult:
    """Select maximum score with exact ties resolved by the frozen logical order."""

    best_id: str | None = None
    best: GuardDecision | None = None
    considered: list[str] = []
    for logical_id in logical_order:
        decision = decisions.get(logical_id)
        if decision is None or not decision.eligible or decision.score is None:
            continue
        if not np.isfinite(decision.score):
            continue
        considered.append(logical_id)
        if best is None or decision.score > float(best.score):
            best_id = logical_id
            best = decision
    diagnostics = {
        "logical_order": list(logical_order),
        "eligible_in_order": considered,
        "tie_rule": "first_in_frozen_logical_order",
    }
    if best is None:
        return SelectionResult("abstain_no_stable_candidate", None, None, None, diagnostics)
    return SelectionResult("selected", best_id, best.labels, float(best.score), diagnostics)


__all__ = [
    "RobustScoreTransform",
    "RawViewScores",
    "StandardizedViewScores",
    "GuardDecision",
    "SelectionResult",
    "fit_robust_score_transform",
    "standardize_multiview_scores",
    "adjusted_rand_stability",
    "permutation_aligned_agreement",
    "composite_score",
    "apply_refinement_guard",
    "apply_atomic_refinement_guard",
    "select_guarded_candidate",
]
