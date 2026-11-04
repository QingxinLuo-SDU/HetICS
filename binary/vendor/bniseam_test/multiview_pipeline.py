"""Truth-free bidirectional multiview label recovery and label freezing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .cem_objectives import dense_center_codes
from .cem_optimizer import refine_bniseam_candidate, refit_bniseam_fixed_labels
from .cem_types import CEMConfig, RefinementResult
from .design import SEED_DERIVATION_VERSION, derive_algorithm_child_seed
from .feature_transform import FeatureTransform
from .model import FitConfig, Parameters
from .multiview_a2 import A2_MIN_VALIDATION_ROWS, preregistered_a2_calibration
from .multiview_candidates import (
    LOGICAL_CANDIDATE_SPECS,
    CandidatePool,
    CandidatePoolConfig,
    LogicalCandidate,
    build_candidate_pool,
    canonical_partition_key,
)
from .multiview_scoring import (
    GuardDecision,
    RawViewScores,
    apply_atomic_refinement_guard,
    composite_score,
    permutation_aligned_agreement,
    standardize_multiview_scores,
)
from .multiview_split import (HonestSplit, build_full_sample_split,
                              build_honest_split)
from .multiview_types import ObservedDataView, _require_observed, take_observed
from .multiview_validation import (
    XModelCache,
    fit_f_moment_model,
    score_f_moment_model,
    score_y_observed_mixture,
    score_y_partition_risk,
)


FORMAL_TRUTH_FREE_ARMS: tuple[str, ...] = (
    "conditional_bernoulli_bregman_cem",
    "outcome_score_robust_bregman",
    "treatment_score_robust_sparse_kmeans",
    "covariate_moment_robust_cluster",
    "fusion_xy",
    "fusion_xf",
    "fusion_yf",
    "fusion_fxy",
    "multiview_pool_outcome_selector",
    "multiview_pool_composite_guarded",
)
PRIMARY_ARM = "multiview_pool_composite_guarded"
_DIRECTIONS = (("A->B", "a", "b"), ("B->A", "b", "a"))
_LOGICAL_ORDER = tuple(spec[0] for spec in LOGICAL_CANDIDATE_SPECS)
_PRIMARY_FAMILIES = (
    "outcome",
    "treatment",
    "fusion_xy",
    "fusion_xf",
    "fusion_yf",
    "fusion_fxy",
    "balanced_random",
)
_PRIMARY_LOGICAL_ORDER = tuple(
    logical_id
    for logical_id, family, _ in LOGICAL_CANDIDATE_SPECS
    if family in _PRIMARY_FAMILIES
)


def _readonly(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _freeze_parameters(parameters: Parameters | None) -> Parameters | None:
    if parameters is None:
        return None
    frozen = parameters.clone()
    frozen.pi.setflags(write=False)
    frozen.intercepts.setflags(write=False)
    frozen.beta.setflags(write=False)
    return frozen


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Parameters):
        return {
            "pi": value.pi.tolist(),
            "gamma": float(value.gamma),
            "intercepts": value.intercepts.tolist(),
            "beta": value.beta.tolist(),
        }
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _deep_freeze(value: Any) -> Any:
    """Recursively detach public evidence and make every nested container immutable."""

    if isinstance(value, Parameters):
        return _freeze_parameters(value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return _deep_freeze(dict(values))


# Packaged addition: dispatch on an explicit index mode.
_SPLIT_BUILDERS = {
    "historical_half": build_honest_split,
    "full_sample": build_full_sample_split,
}


def _build_split(data: ObservedDataView, config: "PipelineConfig") -> HonestSplit:
    return _SPLIT_BUILDERS[config.split_mode](data, seed=config.split_seed)


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen controls for the complete truth-free label pipeline."""

    fit_config: FitConfig
    cem_config: CEMConfig
    seed_derivation_version: str = SEED_DERIVATION_VERSION
    dataset_seed_base: int = 1
    split_seed: int = 17
    candidate_seed: int = 29
    score_tolerance: float = 1.0e-8
    stability_threshold: float = 0.80
    f_eps: float = 1.0e-8
    f_scale_floor: float = 1.0e-8
    x_max_iter: int = 200
    epsilon_prob: float = 0.01
    # Packaged addition: index mode.
    # "historical_half" keeps the 50/25/25 split of the source package; with "full_sample" the label,
    # screening and effect stages each cover all rows and A/B still partition the label pool.
    split_mode: str = "historical_half"

    def __post_init__(self) -> None:
        if not isinstance(self.fit_config, FitConfig):
            raise TypeError("fit_config must be a FitConfig")
        if not isinstance(self.cem_config, CEMConfig):
            raise TypeError("cem_config must be a CEMConfig")
        if self.seed_derivation_version != SEED_DERIVATION_VERSION:
            raise ValueError("unsupported seed_derivation_version")
        for name in ("dataset_seed_base", "split_seed", "candidate_seed", "x_max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if self.dataset_seed_base <= 0 or self.split_seed < 0 or self.candidate_seed < 0:
            raise ValueError("dataset_seed_base must be positive and child seeds nonnegative")
        if self.x_max_iter < 1:
            raise ValueError("x_max_iter must be positive")
        if not np.isfinite(self.score_tolerance) or self.score_tolerance < 0.0:
            raise ValueError("score_tolerance must be finite and nonnegative")
        if not np.isfinite(self.stability_threshold) or not 0.0 <= self.stability_threshold <= 1.0:
            raise ValueError("stability_threshold must lie in [0, 1]")
        if not np.isfinite(self.f_eps) or self.f_eps <= 0.0:
            raise ValueError("f_eps must be positive and finite")
        if not np.isfinite(self.f_scale_floor) or self.f_scale_floor <= 0.0:
            raise ValueError("f_scale_floor must be positive and finite")
        if not np.isfinite(self.epsilon_prob) or not 0.0 < self.epsilon_prob <= 0.5:
            raise ValueError("epsilon_prob must lie in (0, 1/2]")
        if self.split_mode not in _SPLIT_BUILDERS:
            raise ValueError(
                f"split_mode must be one of {sorted(_SPLIT_BUILDERS)}")

    def child_seed(
        self,
        *,
        stage: str,
        direction: str = "dataset",
        logical_candidate_id: str = "dataset",
        start_index: int = 0,
    ) -> int:
        root = self.split_seed if stage == "split" else self.candidate_seed
        return derive_algorithm_child_seed(
            root,
            stage=stage,
            direction=direction,
            logical_candidate_id=logical_candidate_id,
            start_index=start_index,
        )


@dataclass(frozen=True)
class SplitManifest:
    split_seed: int
    split_digest: str
    row_counts: Mapping[str, int]
    row_digests: Mapping[str, str]
    center_order: tuple[Any, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_counts", _mapping(self.row_counts))
        object.__setattr__(self, "row_digests", _mapping(self.row_digests))
        object.__setattr__(self, "center_order", tuple(self.center_order))
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self.__dict__)


@dataclass(frozen=True)
class DirectionCandidateResult:
    logical_id: str
    family: str
    algorithm: str
    canonical_id: str | None
    status: str
    reason: str | None
    pre_labels: np.ndarray | None
    post_labels: np.ndarray | None
    raw_scores: Mapping[str, Mapping[str, float | None]]
    standardized_scores: Mapping[str, Mapping[str, float | None]]
    guard_status: str
    chosen_stage: str | None
    guarded_labels: np.ndarray | None
    guarded_composite_score: float | None
    stability_agreement: float | None
    stability_ari: float | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("pre_labels", "post_labels", "guarded_labels"):
            labels = getattr(self, name)
            if labels is not None:
                object.__setattr__(self, name, _readonly(labels, dtype=np.int64))
        object.__setattr__(
            self,
            "raw_scores",
            _mapping({stage: _mapping(scores) for stage, scores in self.raw_scores.items()}),
        )
        object.__setattr__(
            self,
            "standardized_scores",
            _mapping({stage: _mapping(scores) for stage, scores in self.standardized_scores.items()}),
        )
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self.__dict__)


@dataclass(frozen=True)
class DirectionCacheDiagnostics:
    unique_partition_count: int
    refinement_fit_count: int
    expensive_fit_count: int
    pre_validation_fixed_fit_count: int
    x_model_fit_count: int
    x_cache_hits: int
    f_model_fit_count: int
    f_cache_hits: int

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class DirectionResult:
    direction: str
    candidate_seed: int
    training_row_count: int
    validation_row_count: int
    training_row_digest: str
    validation_row_digest: str
    center_order: tuple[Any, ...]
    status: str
    failure_reason: str | None
    selected_f_coordinates: tuple[int, ...]
    candidates: Mapping[str, DirectionCandidateResult]
    score_transform_diagnostics: Mapping[str, Any]
    cache_diagnostics: DirectionCacheDiagnostics
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_order", tuple(self.center_order))
        object.__setattr__(self, "selected_f_coordinates", tuple(int(v) for v in self.selected_f_coordinates))
        object.__setattr__(self, "candidates", _mapping(self.candidates))
        object.__setattr__(self, "score_transform_diagnostics", _mapping(self.score_transform_diagnostics))
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self.__dict__)


@dataclass(frozen=True)
class ArmResult:
    arm: str
    status: str
    selection_rule: str
    criterion: str
    eligible_families: tuple[str, ...]
    selected_logical_id: str | None
    selected_direction: str | None
    selection_score: float | None
    bidirectional_score: float | None
    directional_scores: Mapping[str, float | None]
    frozen_labels: np.ndarray | None
    labels_frozen: bool
    post_freeze_reclassification_count: int | None
    fixed_refit_status: str | None
    fixed_refit_eligible: bool
    fixed_refit_converged: bool
    fixed_refit_trace: tuple[Mapping[str, Any], ...]
    parameters: Parameters | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "eligible_families", tuple(self.eligible_families))
        if self.frozen_labels is not None:
            object.__setattr__(self, "frozen_labels", _readonly(self.frozen_labels, dtype=np.int64))
        object.__setattr__(self, "directional_scores", _mapping(self.directional_scores))
        object.__setattr__(self, "fixed_refit_trace", tuple(_mapping(item) for item in self.fixed_refit_trace))
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self.__dict__)


@dataclass(frozen=True)
class CacheDiagnostics:
    unique_partition_count: int
    refinement_fit_count: int
    expensive_fit_count: int
    pre_validation_fixed_fit_count: int
    final_fixed_refit_count: int
    total_parameter_fit_count: int
    by_direction: Mapping[str, DirectionCacheDiagnostics]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_direction", _mapping(self.by_direction))

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self.__dict__)


@dataclass(frozen=True)
class PipelineResult:
    status: str
    split_manifest: SplitManifest
    directions: Mapping[str, DirectionResult]
    arms: Mapping[str, ArmResult]
    cache_diagnostics: CacheDiagnostics
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if tuple(self.arms) != FORMAL_TRUTH_FREE_ARMS:
            raise ValueError("pipeline must return exactly the ten formal arms in frozen order")
        object.__setattr__(self, "directions", _mapping(self.directions))
        object.__setattr__(self, "arms", _mapping(self.arms))
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    @property
    def primary(self) -> ArmResult:
        return self.arms[PRIMARY_ARM]

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self.__dict__)


@dataclass(frozen=True)
class OracleDiagnostic:
    status: str
    uses_truth: bool
    frozen_labels: np.ndarray
    labels_frozen: bool
    post_freeze_reclassification_count: int
    fixed_refit_status: str | None
    fixed_refit_eligible: bool
    fixed_refit_converged: bool
    parameters: Parameters | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frozen_labels", _readonly(self.frozen_labels, dtype=np.int64))
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self.__dict__)


@dataclass
class _StageWork:
    labels: np.ndarray
    raw: RawViewScores
    y_model_status: str
    y_model_eligible: bool
    y_model_failure_reason: str | None
    standardized: dict[str, float | None] = field(default_factory=dict)


@dataclass
class _CandidateWork:
    source: LogicalCandidate
    refinement: RefinementResult | None
    status: str
    reason: str | None
    stages: dict[str, _StageWork]
    refinement_failure_reason: str | None = None
    pre_fallback_attempted: bool = False
    pre_fallback_status: str | None = None
    minimum_cluster_size: int | None = None
    guard: GuardDecision | None = None


@dataclass
class _DirectionWork:
    name: str
    seed: int
    train_rows: np.ndarray
    validation_rows: np.ndarray
    train: ObservedDataView
    validation: ObservedDataView
    pool: CandidatePool
    transform: FeatureTransform
    selected_f_coordinates: tuple[int, ...]
    candidates: dict[str, _CandidateWork]
    score_diagnostics: dict[str, Any]
    cache_diagnostics: DirectionCacheDiagnostics
    diagnostics: dict[str, Any]


def _row_digest(rows: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(rows, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _take_preserving_centers(data: ObservedDataView, rows: np.ndarray) -> ObservedDataView:
    subset = take_observed(data, rows)
    present = {_python_scalar(value) for value in subset.center_codes}
    declared = np.array(
        [value for value in data.centers if _python_scalar(value) in present],
        dtype=np.asarray(data.centers).dtype,
    )
    return ObservedDataView(
        features=subset.features,
        treatment=subset.treatment,
        outcome=subset.outcome,
        center_codes=subset.center_codes,
        centers=declared,
    )


def _split_manifest(observed: ObservedDataView, split: HonestSplit, seed: int) -> SplitManifest:
    groups = {
        "label": split.label,
        "screening": split.screening,
        "effect": split.effect,
        "a": split.a,
        "b": split.b,
    }
    return SplitManifest(
        split_seed=int(seed),
        split_digest=str(split.diagnostics["digest_sha256"]),
        row_counts={name: int(rows.size) for name, rows in groups.items()},
        row_digests={name: _row_digest(rows) for name, rows in groups.items()},
        center_order=tuple(_python_scalar(value) for value in observed.centers),
        diagnostics=dict(split.diagnostics),
    )


def _empty_arm(name: str, status: str, *, reason: str | None = None) -> ArmResult:
    spec = _arm_spec(name)
    return ArmResult(
        arm=name,
        status=status,
        selection_rule=spec[2],
        criterion=spec[1],
        eligible_families=spec[0],
        selected_logical_id=None,
        selected_direction=None,
        selection_score=None,
        bidirectional_score=None,
        directional_scores={"A->B": None, "B->A": None},
        frozen_labels=None,
        labels_frozen=False,
        post_freeze_reclassification_count=None,
        fixed_refit_status=None,
        fixed_refit_eligible=False,
        fixed_refit_converged=False,
        fixed_refit_trace=(),
        parameters=None,
        diagnostics={} if reason is None else {"reason": reason},
    )


def _empty_cache() -> CacheDiagnostics:
    return CacheDiagnostics(0, 0, 0, 0, 0, 0, {})


def _ineligible_pipeline(
    status: str,
    manifest: SplitManifest,
    *,
    reason: str,
) -> PipelineResult:
    arms = {name: _empty_arm(name, status, reason=reason) for name in FORMAL_TRUTH_FREE_ARMS}
    return PipelineResult(
        status=status,
        split_manifest=manifest,
        directions={},
        arms=arms,
        cache_diagnostics=_empty_cache(),
        diagnostics={"reason": reason},
    )


def _same_transform(left: FeatureTransform, right: FeatureTransform) -> bool:
    return bool(np.array_equal(left.mean, right.mean) and np.array_equal(left.scale, right.scale))


def _fixed_refit_exception_result(
    labels: np.ndarray,
    transform: FeatureTransform,
    *,
    stage: str,
    error: Exception,
    minimum_cluster_size: int,
) -> RefinementResult:
    """Totalize an expected numerical/input refit exception in the multiview pipeline."""

    frozen = np.asarray(labels, dtype=np.int64).copy()
    status = f"{stage}_fixed_refit_exception:{type(error).__name__}"
    detail = f"{type(error).__name__}: {error}"
    return RefinementResult(
        parameters=None,
        pre_labels=frozen,
        post_labels=frozen.copy(),
        training_objective=None,
        status=status,
        eligible=False,
        converged=False,
        trace=[],
        minimum_cluster_size=int(minimum_cluster_size),
        failure_reason=detail,
        feature_transform=transform,
        diagnostics={
            "failure_kind": status,
            "failure": detail,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        },
    )


def _unavailable_stage_result(
    labels: np.ndarray,
    transform: FeatureTransform,
    *,
    status: str,
    reason: str,
    minimum_cluster_size: int,
) -> RefinementResult:
    frozen = np.asarray(labels, dtype=np.int64).copy()
    return RefinementResult(
        parameters=None,
        pre_labels=frozen,
        post_labels=frozen.copy(),
        training_objective=None,
        status=status,
        eligible=False,
        converged=False,
        trace=[],
        minimum_cluster_size=int(minimum_cluster_size),
        failure_reason=reason,
        feature_transform=transform,
        diagnostics={"failure_kind": status, "failure": reason},
    )


def _stage_raw_scores(
    logical_id: str,
    stage: str,
    labels: np.ndarray,
    parameters: Parameters | None,
    *,
    train: ObservedDataView,
    validation: ObservedDataView,
    transform: FeatureTransform,
    x_cache: XModelCache,
    f_cache: dict[tuple[int, ...], Any],
    selected_f_coordinates: tuple[int, ...],
    config: PipelineConfig,
    minimum_cluster_size: int,
    cache_counts: dict[str, int],
) -> RawViewScores:
    if parameters is None:
        return RawViewScores(logical_id, stage, np.nan, np.nan, np.nan, "ineligible", "missing_y_parameters")
    y_score = score_y_observed_mixture(validation, parameters, transform)
    y_primary = score_y_partition_risk(
        validation,
        labels,
        parameters,
        transform,
        epsilon_prob=config.epsilon_prob,
    )
    try:
        x_fit = x_cache.fit(labels, minimum_cluster_size=minimum_cluster_size)
        x_score = x_cache.score(validation, labels, x_fit.model)
        x_primary = x_cache.primary_risk(
            validation,
            labels,
            x_fit.model,
            epsilon_prob=config.epsilon_prob,
        )
    except (FloatingPointError, TypeError, ValueError) as error:
        return RawViewScores(logical_id, stage, np.nan, np.nan, np.nan, "ineligible", f"x_validation_failed:{type(error).__name__}:{error}")
    if not selected_f_coordinates:
        return RawViewScores(
            logical_id, stage, float(y_score.value) if y_score.eligible else np.nan,
            float(x_score.value) if x_score.eligible else np.nan, np.nan,
            "ineligible", "missing_explicit_f_coordinates",
            float(y_primary.value) if y_primary.eligible else None,
            float(x_primary.value) if x_primary.eligible else None,
        )
    partition = canonical_partition_key(labels)
    if partition in f_cache:
        cache_counts["f_hits"] += 1
        f_fit = f_cache[partition]
    else:
        cache_counts["f_fits"] += 1
        f_fit = fit_f_moment_model(
            train,
            labels,
            selected_coordinates=np.asarray(selected_f_coordinates, dtype=np.int64),
            eps=config.f_eps,
            scale_floor=config.f_scale_floor,
            minimum_cluster_size=minimum_cluster_size,
        )
        f_cache[partition] = f_fit
    f_score = score_f_moment_model(validation, labels, f_fit.model)
    scores = (y_score, x_score, f_score, y_primary, x_primary)
    if not all(score.eligible for score in scores):
        reasons = [score.reason for score in scores if not score.eligible]
        return RawViewScores(
            logical_id, stage, np.nan, np.nan, np.nan, "ineligible",
            ";".join(str(value) for value in reasons),
            float(y_primary.value) if y_primary.eligible else None,
            float(x_primary.value) if x_primary.eligible else None,
        )
    return RawViewScores(
        logical_id,
        stage,
        float(y_score.value),
        float(x_score.value),
        float(f_score.value),
        primary_y_risk=float(y_primary.value),
        primary_x_risk=float(x_primary.value),
    )


def _run_direction(
    name: str,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    observed: ObservedDataView,
    config: PipelineConfig,
) -> _DirectionWork:
    train = _take_preserving_centers(observed, train_rows)
    validation = _take_preserving_centers(observed, validation_rows)
    transform = FeatureTransform.fit(train.features)
    seed = config.child_seed(stage="candidate_pool", direction=name)
    pool = build_candidate_pool(
        train,
        config=CandidatePoolConfig(
            k_components=config.cem_config.k_components,
            pi_cem=config.cem_config.pi_cem,
        ),
        seed=seed,
    )
    selected_coordinates = tuple(
        int(value)
        for value in pool.representation_diagnostics.get("covariate", {}).get(
            "selected_original_indices", ()
        )
    )
    refinement_by_id: dict[str, RefinementResult | None] = {}
    refinement_failures: dict[str, str] = {}
    canonical_by_id = {candidate.canonical_id: candidate for candidate in pool.canonical_candidates}
    for canonical_id in pool.fit_candidate_ids:
        canonical = canonical_by_id[canonical_id]
        try:
            refinement = refine_bniseam_candidate(
                train,
                canonical.labels,
                fit_config=config.fit_config,
                cem_config=config.cem_config,
            )
            if not _same_transform(transform, refinement.feature_transform):
                refinement_failures[canonical_id] = "refinement_training_transform_mismatch"
                refinement_by_id[canonical_id] = None
            else:
                refinement_by_id[canonical_id] = refinement
        except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
            refinement_failures[canonical_id] = f"refinement_failed:{type(error).__name__}:{error}"
            refinement_by_id[canonical_id] = None

    x_cache = XModelCache(train, transform, max_iter=config.x_max_iter)
    f_cache: dict[tuple[int, ...], Any] = {}
    y_pre_cache: dict[tuple[int, ...], RefinementResult] = {}
    cache_counts = {"f_fits": 0, "f_hits": 0, "pre_y_fits": 0}
    work: dict[str, _CandidateWork] = {}
    raw_records: list[RawViewScores] = []

    def independently_fit_pre(labels: np.ndarray) -> RefinementResult:
        partition = canonical_partition_key(labels)
        if partition not in y_pre_cache:
            cache_counts["pre_y_fits"] += 1
            try:
                y_pre_cache[partition] = refit_bniseam_fixed_labels(
                    train,
                    labels,
                    fit_config=config.fit_config,
                    cem_config=config.cem_config,
                    feature_transform=transform,
                )
            except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
                y_pre_cache[partition] = _fixed_refit_exception_result(
                    labels,
                    transform,
                    stage="pre_validation",
                    error=error,
                    minimum_cluster_size=pool.m_e,
                )
        return y_pre_cache[partition]

    for logical in pool.logical_candidates:
        if not logical.eligible or logical.canonical_id is None or logical.labels is None:
            work[logical.logical_id] = _CandidateWork(
                logical,
                None,
                "ineligible_source",
                logical.ineligible_reason or "source_candidate_ineligible",
                {},
            )
            continue
        refinement = refinement_by_id.get(logical.canonical_id)
        refinement_usable = bool(
            refinement is not None
            and refinement.eligible
            and refinement.parameters is not None
        )
        if refinement_usable:
            pre_labels = np.asarray(refinement.pre_labels, dtype=np.int64)
            post_labels = np.asarray(refinement.post_labels, dtype=np.int64)
            pre_partition = canonical_partition_key(pre_labels)
            post_partition = canonical_partition_key(post_labels)
            pre_fit = (
                refinement
                if pre_partition == post_partition
                else independently_fit_pre(pre_labels)
            )
            post_fit = refinement
            refinement_failure_reason = None
            pre_fallback_attempted = pre_partition != post_partition
        else:
            pre_labels = np.asarray(logical.labels, dtype=np.int64)
            if refinement is not None and np.asarray(refinement.post_labels).shape == pre_labels.shape:
                post_labels = np.asarray(refinement.post_labels, dtype=np.int64)
                refinement_failure_reason = refinement.failure_reason or refinement.status
                post_status = f"refinement_ineligible:{refinement.status}"
            else:
                post_labels = pre_labels.copy()
                refinement_failure_reason = refinement_failures.get(
                    logical.canonical_id,
                    "missing_canonical_refinement",
                )
                post_status = "refinement_unavailable"
            pre_fit = independently_fit_pre(pre_labels)
            post_fit = _unavailable_stage_result(
                post_labels,
                transform,
                status=post_status,
                reason=refinement_failure_reason,
                minimum_cluster_size=pool.m_e,
            )
            pre_fallback_attempted = True
        stage_fits = {"pre": pre_fit, "post": post_fit}
        stages: dict[str, _StageWork] = {}
        for stage, labels in (("pre", pre_labels), ("post", post_labels)):
            stage_fit = stage_fits[stage]
            stage_parameters = stage_fit.parameters if stage_fit.eligible else None
            if stage == "post" and not stage_fit.eligible:
                raw = RawViewScores(
                    logical.logical_id,
                    stage,
                    np.nan,
                    np.nan,
                    np.nan,
                    status="ineligible",
                    reason=stage_fit.failure_reason or stage_fit.status,
                )
            else:
                raw = _stage_raw_scores(
                    logical.logical_id,
                    stage,
                    labels,
                    stage_parameters,
                    train=train,
                    validation=validation,
                    transform=transform,
                    x_cache=x_cache,
                    f_cache=f_cache,
                    selected_f_coordinates=selected_coordinates,
                    config=config,
                    minimum_cluster_size=pool.m_e,
                    cache_counts=cache_counts,
                )
            stages[stage] = _StageWork(
                np.array(labels, copy=True),
                raw,
                stage_fit.status,
                bool(stage_fit.eligible and stage_fit.parameters is not None),
                stage_fit.failure_reason,
            )
            raw_records.append(raw)
        work[logical.logical_id] = _CandidateWork(
            logical,
            refinement,
            "eligible_for_guard",
            None,
            stages,
            refinement_failure_reason=refinement_failure_reason,
            pre_fallback_attempted=pre_fallback_attempted,
            pre_fallback_status=(pre_fit.status if pre_fallback_attempted else None),
            minimum_cluster_size=pool.m_e,
        )

    standardized, score_diagnostics = standardize_multiview_scores(raw_records)
    for score in standardized:
        candidate = work[score.logical_id]
        stage = candidate.stages[score.stage]
        stage.standardized = {
            "y": score.z_y,
            "x": score.z_x,
            "f": score.z_f,
        }

    x_diagnostics = x_cache.diagnostics
    unique_count = len(pool.fit_candidate_ids)
    cache_diagnostics = DirectionCacheDiagnostics(
        unique_partition_count=unique_count,
        refinement_fit_count=unique_count,
        expensive_fit_count=unique_count,
        pre_validation_fixed_fit_count=cache_counts["pre_y_fits"],
        x_model_fit_count=int(x_diagnostics["fit_count"]),
        x_cache_hits=int(x_diagnostics["cache_hits"]),
        f_model_fit_count=cache_counts["f_fits"],
        f_cache_hits=cache_counts["f_hits"],
    )
    return _DirectionWork(
        name=name,
        seed=seed,
        train_rows=np.array(train_rows, copy=True),
        validation_rows=np.array(validation_rows, copy=True),
        train=train,
        validation=validation,
        pool=pool,
        transform=transform,
        selected_f_coordinates=selected_coordinates,
        candidates=work,
        score_diagnostics=score_diagnostics,
        cache_diagnostics=cache_diagnostics,
        diagnostics={
            "feature_transform_source": "direction_training_rows_only",
            "refinement_transform_equality_asserted": True,
            "candidate_pool_build_count": 1,
            "x_cache_instance_count": 1,
            "score_standardization_call_count": 1,
            "selected_f_coordinates_explicit": bool(selected_coordinates),
            "seed_namespace": {
                "version": config.seed_derivation_version,
                "dataset_seed_base": config.dataset_seed_base,
                "split_seed": config.split_seed,
                "candidate_seed": config.candidate_seed,
                "direction_candidate_seed": seed,
            },
            "cem_start_policy": {
                "mode": "fixed_candidate_only",
                "n_random_starts": config.cem_config.n_random_starts,
                "random_start_seeds": [],
                "fixed_start_seed": None,
            },
            "refinement_failures": refinement_failures,
            "fit_count_definition": {
                "expensive_fit_count": "classification_EM_refinement_calls_only",
                "pre_validation_fixed_fit_count": "extra_stage_specific_pre_Y_fixed_label_fits",
            },
        },
    )


def _guard_candidates(left: _DirectionWork, right: _DirectionWork, config: PipelineConfig) -> None:
    def candidate_can_enter_guard(candidate: _CandidateWork | None) -> bool:
        return bool(
            candidate is not None
            and candidate.status == "eligible_for_guard"
            and candidate.stages
        )

    def stage_primary_score(
        candidate: _CandidateWork,
        stage_name: str,
        stability: float,
    ) -> tuple[float, bool]:
        stage = candidate.stages[stage_name]
        legacy_values = tuple(stage.standardized.get(view) for view in ("y", "x", "f"))
        stage.standardized["composite"] = (
            composite_score(
                float(legacy_values[0]),
                float(legacy_values[1]),
                float(legacy_values[2]),
                stability,
            )
            if all(value is not None and np.isfinite(value) for value in legacy_values)
            else None
        )
        values = (stage.raw.primary_y_risk, stage.raw.primary_x_risk)
        legal = bool(
            stage.y_model_eligible
            and all(value is not None and np.isfinite(value) for value in values)
        )
        if not legal:
            stage.standardized["primary_y_risk"] = None
            stage.standardized["primary_x_risk"] = None
            stage.standardized["primary_risk"] = None
            stage.standardized["primary_score"] = None
            return float("nan"), False
        primary_y = float(values[0])
        primary_x = float(values[1])
        primary_risk = (primary_y + primary_x) / 2.0
        stage.standardized["primary_y_risk"] = primary_y
        stage.standardized["primary_x_risk"] = primary_x
        stage.standardized["primary_risk"] = primary_risk
        stage.standardized["primary_score"] = -primary_risk
        return -primary_risk, True

    for logical_id in _LOGICAL_ORDER:
        first = left.candidates.get(logical_id)
        second = right.candidates.get(logical_id)
        if not candidate_can_enter_guard(first) or not candidate_can_enter_guard(second):
            reason = "missing_or_ineligible_companion_direction"
            if first is not None:
                first.guard = GuardDecision.ineligible(reason)
            if second is not None:
                second.guard = GuardDecision.ineligible(reason)
            continue
        try:
            pre_stability = permutation_aligned_agreement(
                first.stages["pre"].labels,
                second.stages["pre"].labels,
            )
            post_stability = permutation_aligned_agreement(
                first.stages["post"].labels,
                second.stages["post"].labels,
            )
            first_pre_score, first_pre_legal = stage_primary_score(first, "pre", pre_stability)
            first_post_score, first_post_legal = stage_primary_score(first, "post", post_stability)
            second_pre_score, second_pre_legal = stage_primary_score(second, "pre", pre_stability)
            second_post_score, second_post_legal = stage_primary_score(second, "post", post_stability)
            first.guard, second.guard = apply_atomic_refinement_guard(
                first.stages["pre"].labels,
                first.stages["post"].labels,
                right_pre_labels=second.stages["pre"].labels,
                right_post_labels=second.stages["post"].labels,
                left_pre_score=first_pre_score,
                left_post_score=first_post_score,
                right_pre_score=second_pre_score,
                right_post_score=second_post_score,
                k_components=config.cem_config.k_components,
                minimum_cluster_size=config.cem_config.minimum_cluster_size_formula(
                    int(first.stages["pre"].labels.size)
                ),
                left_pre_numerically_legal=first_pre_legal,
                left_post_numerically_legal=first_post_legal,
                right_pre_numerically_legal=second_pre_legal,
                right_post_numerically_legal=second_post_legal,
                score_tolerance=config.score_tolerance,
                stability_threshold=config.stability_threshold,
            )
        except (FloatingPointError, TypeError, ValueError) as error:
            reason = f"guard_failed:{type(error).__name__}:{error}"
            first.guard = GuardDecision.ineligible(reason)
            second.guard = GuardDecision.ineligible(reason)


def _public_candidate(candidate: _CandidateWork) -> DirectionCandidateResult:
    raw = {
        stage: {
            "y": work.raw.y,
            "x": work.raw.x,
            "f": work.raw.f,
            "primary_y_risk": work.raw.primary_y_risk,
            "primary_x_risk": work.raw.primary_x_risk,
            "primary_risk": (
                None
                if work.raw.primary_y_risk is None or work.raw.primary_x_risk is None
                else (work.raw.primary_y_risk + work.raw.primary_x_risk) / 2.0
            ),
        }
        for stage, work in candidate.stages.items()
    }
    standardized = {stage: dict(work.standardized) for stage, work in candidate.stages.items()}
    guard = candidate.guard or GuardDecision.ineligible("guard_not_run")
    return DirectionCandidateResult(
        logical_id=candidate.source.logical_id,
        family=candidate.source.family,
        algorithm=candidate.source.algorithm,
        canonical_id=candidate.source.canonical_id,
        status=(
            "eligible_guarded"
            if guard.eligible
            else "ineligible_guard"
            if candidate.status == "eligible_for_guard"
            else candidate.status
        ),
        reason=None if guard.eligible else candidate.reason or str(guard.diagnostics.get("reason", guard.status)),
        pre_labels=(
            candidate.stages["pre"].labels if "pre" in candidate.stages else None
        ),
        post_labels=(
            candidate.stages["post"].labels if "post" in candidate.stages else None
        ),
        raw_scores=raw,
        standardized_scores=standardized,
        guard_status=guard.status,
        chosen_stage=guard.chosen_stage,
        guarded_labels=guard.labels,
        guarded_composite_score=guard.score,
        stability_agreement=guard.stability_agreement,
        stability_ari=guard.stability_ari,
        diagnostics={
            "source_candidate": candidate.source.diagnostics,
            "source_candidate_status": candidate.status,
            "refinement_status": None if candidate.refinement is None else candidate.refinement.status,
            "refinement_eligible": False if candidate.refinement is None else candidate.refinement.eligible,
            "refinement_converged": False if candidate.refinement is None else candidate.refinement.converged,
            "refinement_failure_reason": candidate.refinement_failure_reason,
            "pre_fallback_attempted": candidate.pre_fallback_attempted,
            "pre_fallback_status": candidate.pre_fallback_status,
            "minimum_cluster_size": candidate.minimum_cluster_size,
            "minimum_cluster_size_source": "formula_m_E",
            "stage_status": {
                stage: {
                    "y_model_status": work.y_model_status,
                    "y_model_eligible": work.y_model_eligible,
                    "y_model_failure_reason": work.y_model_failure_reason,
                    "raw_score_status": work.raw.status,
                    "raw_score_reason": work.raw.reason,
                }
                for stage, work in candidate.stages.items()
            },
            "guard": guard.diagnostics,
        },
    )


def _public_direction(work: _DirectionWork) -> DirectionResult:
    public_candidates = {
        logical_id: _public_candidate(work.candidates[logical_id])
        for logical_id in _LOGICAL_ORDER
        if logical_id in work.candidates
    }
    return DirectionResult(
        direction=work.name,
        candidate_seed=work.seed,
        training_row_count=int(work.train_rows.size),
        validation_row_count=int(work.validation_rows.size),
        training_row_digest=_row_digest(work.train_rows),
        validation_row_digest=_row_digest(work.validation_rows),
        center_order=tuple(_python_scalar(value) for value in work.train.centers),
        status=work.pool.status,
        failure_reason=work.pool.failure_reason,
        selected_f_coordinates=work.selected_f_coordinates,
        candidates=public_candidates,
        score_transform_diagnostics=work.score_diagnostics,
        cache_diagnostics=work.cache_diagnostics,
        diagnostics={
            **work.diagnostics,
            "logical_candidate_count": len(work.pool.logical_candidates),
            "canonical_candidate_count": len(work.pool.canonical_candidates),
            "pool_representation_diagnostics": work.pool.representation_diagnostics,
            "training_uses_validation_rows": False,
        },
    )


def _arm_spec(name: str) -> tuple[tuple[str, ...], str, str]:
    if name == "conditional_bernoulli_bregman_cem":
        return (
            ("outcome", "balanced_random"),
            "raw_y",
            "directional_heldout_observed_y_over_outcome_or_balanced_random",
        )
    if name == "outcome_score_robust_bregman":
        return (("outcome",), "raw_y", "directional_raw_y_within_outcome")
    if name == "treatment_score_robust_sparse_kmeans":
        return (("treatment",), "raw_x", "directional_raw_x_within_treatment")
    if name == "covariate_moment_robust_cluster":
        return (("covariate",), "raw_f", "directional_raw_f_within_covariate")
    if name in {"fusion_xy", "fusion_xf", "fusion_yf", "fusion_fxy"}:
        return ((name,), "composite", f"directional_composite_within_{name}")
    if name == "multiview_pool_outcome_selector":
        return (tuple(), "y", "directional_standardized_y_over_all_non_oracle_candidates")
    if name == "multiview_pool_composite_guarded":
        return (
            _PRIMARY_FAMILIES,
            "primary_risk",
            "directional_primary_risk_minimum_over_guarded_active_records",
        )
    raise KeyError(name)


def _criterion(candidate: _CandidateWork, metric: str) -> float | None:
    guard = candidate.guard
    if guard is None or not guard.eligible or guard.chosen_stage is None:
        return None
    stage = candidate.stages[guard.chosen_stage]
    if metric.startswith("raw_"):
        value = getattr(stage.raw, metric.removeprefix("raw_"), None)
    else:
        value = stage.standardized.get(metric)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


@dataclass(frozen=True)
class _ArmSelection:
    name: str
    families: tuple[str, ...]
    metric: str
    rule: str
    logical_id: str | None
    selection_score: float | None
    bidirectional_score: float | None
    directional_scores: Mapping[str, float | None]
    direction: str | None
    labels: np.ndarray | None
    considered: tuple[str, ...]


def _select_arm(name: str, directions: Mapping[str, _DirectionWork]) -> _ArmSelection:
    families, metric, rule = _arm_spec(name)
    minimize = metric == "primary_risk"
    best_id: str | None = None
    best_score: float | None = None
    best_direction: str | None = None
    best_scores: dict[str, float | None] = {"A->B": None, "B->A": None}
    considered: list[str] = []
    for logical_id in _LOGICAL_ORDER:
        left = directions["A->B"].candidates.get(logical_id)
        right = directions["B->A"].candidates.get(logical_id)
        if left is None or right is None:
            continue
        if families and (left.source.family not in families or right.source.family not in families):
            continue
        scores = {
            "A->B": _criterion(left, metric),
            "B->A": _criterion(right, metric),
        }
        for direction in ("A->B", "B->A"):
            score = scores[direction]
            if score is None:
                continue
            considered.append(f"{logical_id}|{direction}")
            if best_score is None or (score < best_score if minimize else score > best_score):
                best_id = logical_id
                best_score = score
                best_direction = direction
                best_scores = scores
    if best_id is None:
        return _ArmSelection(
            name,
            families,
            metric,
            rule,
            None,
            None,
            None,
            best_scores,
            None,
            None,
            tuple(considered),
        )
    finite_scores = [
        float(score) for score in best_scores.values() if score is not None
    ]
    bidirectional_score = (
        float(np.mean(finite_scores)) if len(finite_scores) == 2 else None
    )
    if best_direction is None:
        raise RuntimeError("directional selector lost its selected direction")
    chosen = directions[best_direction].candidates[best_id]
    labels = None if chosen.guard is None else chosen.guard.labels
    return _ArmSelection(
        name,
        families,
        metric,
        rule,
        best_id,
        best_score,
        bidirectional_score,
        best_scores,
        best_direction,
        None if labels is None else np.array(labels, copy=True),
        tuple(considered),
    )


def _arm_from_selection(
    selection: _ArmSelection,
    refit: RefinementResult | None,
) -> ArmResult:
    if selection.labels is None:
        return _empty_arm(selection.name, "abstain_no_stable_candidate", reason="no_legal_bidirectional_candidate")
    if refit is None:
        refit_status = "fixed_refit_missing_result"
        eligible = False
        converged = False
        trace: Sequence[Mapping[str, Any]] = ()
        parameters = None
        status = "fixed_refit_failed"
        reason = refit_status
        reclassification_count = None
    else:
        reclassification_count = int(
            np.sum(np.asarray(refit.post_labels, dtype=np.int64) != selection.labels)
        )
        refit_status = refit.status
        eligible = bool(refit.eligible and refit.parameters is not None)
        converged = bool(refit.converged)
        trace = tuple(refit.trace)
        parameters = refit.parameters if eligible else None
        if eligible:
            status = "complete" if converged else "complete_max_iter_eligible"
            reason = None
        else:
            status = "fixed_refit_failed"
            reason = refit.failure_reason or refit.status
    return ArmResult(
        arm=selection.name,
        status=status,
        selection_rule=selection.rule,
        criterion=selection.metric,
        eligible_families=selection.families,
        selected_logical_id=selection.logical_id,
        selected_direction=selection.direction,
        selection_score=selection.selection_score,
        bidirectional_score=selection.bidirectional_score,
        directional_scores=selection.directional_scores,
        frozen_labels=selection.labels,
        labels_frozen=True,
        post_freeze_reclassification_count=reclassification_count,
        fixed_refit_status=refit_status,
        fixed_refit_eligible=eligible,
        fixed_refit_converged=converged,
        fixed_refit_trace=tuple(trace),
        parameters=parameters,
        diagnostics={
            "candidate_tie_rule": "first_in_frozen_logical_order",
            "direction_tie_rule": "A->B_within_logical_candidate",
            "considered_directional_records": list(selection.considered),
            "selection_uses_bidirectional_mean": False,
            "selection_objective": (
                "minimize_equal_weight_primary_y_x_risk"
                if selection.metric == "primary_risk"
                else "maximize_score"
            ),
            "labels_copied_before_refit": True,
            "post_freeze_assignment_step": False,
            "fixed_refit_failure_reason": reason,
        },
    )


def run_label_pipeline(observed: ObservedDataView, config: PipelineConfig) -> PipelineResult:
    """Run the ten formal observed-data arms and freeze center labels."""

    data = _require_observed(observed)
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be a PipelineConfig")
    dense_center_codes(data)
    split = _build_split(data, config)
    manifest = _split_manifest(data, split, config.split_seed)
    expected_centers = tuple(_python_scalar(value) for value in data.centers)
    for rows in (split.a, split.b):
        subset = _take_preserving_centers(data, rows)
        if tuple(_python_scalar(value) for value in subset.centers) != expected_centers:
            return _ineligible_pipeline(
                "ineligible_center_missing_from_direction",
                manifest,
                reason="every center must have rows in both A and B",
            )
    label_fold = _take_preserving_centers(data, split.label)
    direction_work: dict[str, _DirectionWork] = {}
    for name, train_name, validation_name in _DIRECTIONS:
        direction_work[name] = _run_direction(
            name,
            np.asarray(getattr(split, train_name)),
            np.asarray(getattr(split, validation_name)),
            data,
            config,
        )
    _guard_candidates(direction_work["A->B"], direction_work["B->A"], config)
    raw_selections = {
        name: _select_arm(name, direction_work)
        for name in FORMAL_TRUTH_FREE_ARMS
    }
    selections = {
        name: (
            selection
            if selection.labels is None
            else replace(
                selection,
                labels=np.asarray(
                    canonical_partition_key(selection.labels),
                    dtype=np.int64,
                ),
            )
        )
        for name, selection in raw_selections.items()
    }

    full_transform = FeatureTransform.fit(label_fold.features)
    final_refit_cache: dict[tuple[int, ...], RefinementResult | None] = {}
    for selection in selections.values():
        if selection.labels is None:
            continue
        partition = canonical_partition_key(selection.labels)
        if partition not in final_refit_cache:
            try:
                result = refit_bniseam_fixed_labels(
                    label_fold,
                    np.array(selection.labels, copy=True),
                    fit_config=config.fit_config,
                    cem_config=config.cem_config,
                    feature_transform=full_transform,
                )
                if result is None:
                    raise RuntimeError("fixed-label refit returned no result")
            except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
                result = _fixed_refit_exception_result(
                    selection.labels,
                    full_transform,
                    stage="final",
                    error=error,
                    minimum_cluster_size=config.cem_config.minimum_cluster_size_formula(
                        int(label_fold.centers.size)
                    ),
                )
            if result is not None and not np.array_equal(result.post_labels, selection.labels):
                result = replace(
                    result,
                    parameters=None,
                    status="post_freeze_label_change_refused",
                    eligible=False,
                    converged=False,
                    failure_reason="fixed-label refit changed frozen labels",
                )
            final_refit_cache[partition] = result
    arms = {
        name: _arm_from_selection(
            selection,
            None if selection.labels is None else final_refit_cache[canonical_partition_key(selection.labels)],
        )
        for name, selection in selections.items()
    }
    by_direction = {name: work.cache_diagnostics for name, work in direction_work.items()}
    a2_calibration = preregistered_a2_calibration(epsilon_prob=config.epsilon_prob)
    realized_validation_rows = {
        name: int(work.validation_rows.size) for name, work in direction_work.items()
    }
    a2_calibration["realized_validation_rows"] = realized_validation_rows
    a2_calibration["minimum_validation_rows_satisfied"] = all(
        count >= A2_MIN_VALIDATION_ROWS for count in realized_validation_rows.values()
    )
    unique_count = sum(item.unique_partition_count for item in by_direction.values())
    refinement_count = sum(item.refinement_fit_count for item in by_direction.values())
    pre_count = sum(item.pre_validation_fixed_fit_count for item in by_direction.values())
    final_count = len(final_refit_cache)
    cache = CacheDiagnostics(
        unique_partition_count=unique_count,
        refinement_fit_count=refinement_count,
        expensive_fit_count=refinement_count,
        pre_validation_fixed_fit_count=pre_count,
        final_fixed_refit_count=final_count,
        total_parameter_fit_count=refinement_count + pre_count + final_count,
        by_direction=by_direction,
    )
    directions = {name: _public_direction(work) for name, work in direction_work.items()}
    status = "complete" if arms[PRIMARY_ARM].status.startswith("complete") else arms[PRIMARY_ARM].status
    return PipelineResult(
        status=status,
        split_manifest=manifest,
        directions=directions,
        arms=arms,
        cache_diagnostics=cache,
        diagnostics={
            "primary_arm": PRIMARY_ARM,
            "seed_namespace": {
                "version": config.seed_derivation_version,
                "dataset_seed_base": config.dataset_seed_base,
                "split_seed": config.split_seed,
                "candidate_seed": config.candidate_seed,
            },
            "honest_split_call_count": 1,
            "label_fold_row_count": int(split.label.size),
            "screening_rows_read": False,
            "effect_rows_read": False,
            "full_label_fold_transform_fit_count": 1,
            "final_refit_partition_reuse": True,
            "first_final_refit_event_index": (
                None if final_count == 0 else refinement_count + pre_count
            ),
            "post_freeze_refinement_calls": 0,
            "post_freeze_reclassification_calls": 0,
            "primary_risk": {
                "formula": "0.5 * primary_y_risk + 0.5 * primary_x_risk",
                "loss": "clipped_bernoulli_nll",
                "epsilon_prob": config.epsilon_prob,
                "coordinate": {"location": 0.0, "scale": 1.0},
                "coordinate_source": "preregistered_deterministic_identity",
                "row_weighting": "1/N_validation",
                "f_primary_weight": 0.0,
                "stability_primary_weight": 0.0,
                "a2_calibration": a2_calibration,
            },
        },
    )


def _validated_oracle_labels(labels: np.ndarray, n_centers: int, k_components: int) -> np.ndarray:
    values = np.asarray(labels)
    if (
        values.ndim != 1
        or values.shape != (n_centers,)
        or np.issubdtype(values.dtype, np.bool_)
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise ValueError("truth_labels must contain one integer label per declared center")
    values = values.astype(np.int64, copy=True)
    if not np.array_equal(np.unique(values), np.arange(k_components)):
        raise ValueError("truth_labels must use every configured class in 0,...,K-1")
    return values


def run_oracle_diagnostic(
    observed: ObservedDataView,
    truth_labels: np.ndarray,
    config: PipelineConfig,
) -> OracleDiagnostic:
    """Run an explicit truth-only fixed-label diagnostic outside the main pipeline."""

    data = _require_observed(observed)
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be a PipelineConfig")
    labels = _validated_oracle_labels(
        truth_labels,
        int(data.centers.size),
        config.cem_config.k_components,
    )
    split = _build_split(data, config)
    label_fold = _take_preserving_centers(data, split.label)
    transform = FeatureTransform.fit(label_fold.features)
    result = refit_bniseam_fixed_labels(
        label_fold,
        np.array(labels, copy=True),
        fit_config=config.fit_config,
        cem_config=config.cem_config,
        feature_transform=transform,
    )
    if result is None:
        return OracleDiagnostic(
            "diagnostic_no_refit_result",
            True,
            labels,
            True,
            0,
            None,
            False,
            False,
            None,
            {"boundary": "explicit_oracle_only", "entered_main_cache": False},
        )
    changed = int(np.sum(np.asarray(result.post_labels) != labels))
    if changed:
        status = "post_freeze_label_change_refused"
        parameters = None
        eligible = False
    else:
        status = "complete" if result.eligible else "fixed_refit_failed"
        parameters = result.parameters if result.eligible else None
        eligible = bool(result.eligible)
    return OracleDiagnostic(
        status,
        True,
        labels,
        True,
        changed,
        result.status,
        eligible,
        bool(result.converged),
        parameters,
        {
            "boundary": "explicit_oracle_only",
            "entered_main_cache": False,
            "fixed_refit_failure_reason": result.failure_reason,
        },
    )


__all__ = [
    "FORMAL_TRUTH_FREE_ARMS",
    "PRIMARY_ARM",
    "PipelineConfig",
    "SplitManifest",
    "DirectionCandidateResult",
    "DirectionCacheDiagnostics",
    "DirectionResult",
    "ArmResult",
    "CacheDiagnostics",
    "PipelineResult",
    "OracleDiagnostic",
    "run_label_pipeline",
    "run_oracle_diagnostic",
]
