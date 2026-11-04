"""Leakage-free held-out Y, X, and F validation for multiview labels."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .cem_objectives import dense_center_codes, observed_mixture_objective
from .feature_transform import FeatureTransform
from .model import Parameters, linear_predictor, sigmoid
from .multiview_candidates import canonical_partition_key
from .multiview_types import ObservedDataView, _require_observed


def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _json_number(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


@dataclass(frozen=True)
class ValidationScore:
    """One machine-readable held-out view score."""

    value: float | None
    status: str
    reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.status == "eligible" and self.value is not None


def _ineligible(reason: str, diagnostics: dict[str, Any] | None = None) -> ValidationScore:
    return ValidationScore(None, "ineligible", reason, {} if diagnostics is None else diagnostics)


def _validate_transform(transform: FeatureTransform, p: int) -> FeatureTransform:
    if not isinstance(transform, FeatureTransform):
        raise TypeError("feature_transform must be a FeatureTransform")
    if transform.mean.shape != (p,):
        raise ValueError("feature_transform dimension does not match features")
    return transform


def _validate_binary(values: np.ndarray, name: str) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any((array != 0.0) & (array != 1.0)):
        raise ValueError(f"{name} must contain finite binary values")


def _validate_epsilon_prob(epsilon_prob: float) -> float:
    epsilon = float(epsilon_prob)
    if not np.isfinite(epsilon) or not 0.0 < epsilon <= 0.5:
        raise ValueError("epsilon_prob must lie in (0, 1/2]")
    return epsilon


def clipped_bernoulli_nll(
    response: np.ndarray,
    probability: np.ndarray,
    *,
    epsilon_prob: float,
) -> np.ndarray:
    """Per-row Bernoulli NLL with the preregistered probability envelope."""

    outcomes = np.asarray(response, dtype=float)
    probabilities = np.asarray(probability, dtype=float)
    if outcomes.shape != probabilities.shape:
        raise ValueError("response and probability must have the same shape")
    _validate_binary(outcomes, "response")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probability must be finite")
    epsilon = _validate_epsilon_prob(epsilon_prob)
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    losses = -(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log1p(-clipped))
    if not np.all(np.isfinite(losses)):
        raise FloatingPointError("non-finite clipped Bernoulli loss")
    return losses


def _primary_risk_diagnostics(
    observed: ObservedDataView,
    *,
    epsilon_prob: float,
) -> dict[str, Any]:
    counts = {
        str(value.item() if isinstance(value, np.generic) else value): int(
            np.sum(observed.center_codes == value)
        )
        for value in observed.centers
    }
    denominator = float(observed.n_samples)
    return {
        "loss": "clipped_bernoulli_nll",
        "epsilon_prob": float(epsilon_prob),
        "loss_envelope": float(-math.log(epsilon_prob)),
        "normalization": "row_mean",
        "row_weight": float(1.0 / denominator),
        "center_weights": {key: value / denominator for key, value in counts.items()},
        "coordinate": {"location": 0.0, "scale": 1.0},
        "coordinate_source": "preregistered_deterministic_identity",
        "n_validation": int(observed.n_samples),
    }


def _canonical_labels(
    data: ObservedDataView,
    labels: np.ndarray,
    *,
    minimum_cluster_size: int = 1,
) -> np.ndarray:
    observed = _require_observed(data)
    raw = np.asarray(labels)
    if (
        raw.ndim != 1
        or raw.shape != (observed.centers.size,)
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("labels must contain one integer value per declared center")
    unique = np.unique(raw.astype(np.int64, copy=False))
    if unique.size == 0 or not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("labels must use the contiguous range 0,...,K-1")
    canonical = np.asarray(canonical_partition_key(raw), dtype=np.int64)
    sizes = np.bincount(canonical)
    if minimum_cluster_size < 1:
        raise ValueError("minimum_cluster_size must be positive")
    if sizes.size == 0 or np.any(sizes < minimum_cluster_size):
        raise ValueError("labels violate minimum cluster size")
    return canonical


def _transformed_observed(
    data: ObservedDataView,
    transform: FeatureTransform,
) -> ObservedDataView:
    observed = _require_observed(data)
    return ObservedDataView(
        features=transform.transform(observed.features),
        treatment=np.asarray(observed.treatment),
        outcome=np.asarray(observed.outcome),
        center_codes=np.asarray(observed.center_codes),
        centers=np.asarray(observed.centers),
    )


def score_y_observed_mixture(
    validation: ObservedDataView,
    parameters: Parameters,
    feature_transform: FeatureTransform,
) -> ValidationScore:
    """Score held-out outcomes by the center-level observed mixture likelihood."""

    observed = _require_observed(validation)
    transform = _validate_transform(feature_transform, observed.features.shape[1])
    diagnostics = {
        "normalization": "total_loglik/N_validation",
        "feature_transform_source": "training_fitted_reused",
        "n_validation": int(observed.n_samples),
        "n_validation_centers": int(observed.centers.size),
    }
    if observed.n_samples == 0 or observed.centers.size == 0:
        return _ineligible("empty_validation", diagnostics)
    if not isinstance(parameters, Parameters):
        raise TypeError("parameters must be Parameters fitted on training rows")
    try:
        parameters.validate()
    except ValueError as error:
        return _ineligible("invalid_parameters", {**diagnostics, "detail": str(error)})
    if np.any(parameters.pi <= 0.0) or not np.isclose(
        parameters.pi.sum(), 1.0, atol=1.0e-12, rtol=0.0
    ):
        return _ineligible("invalid_mixture_weights", diagnostics)
    try:
        _validate_binary(observed.outcome, "outcome")
        transformed = _transformed_observed(observed, transform)
        total = observed_mixture_objective(transformed, parameters)
        value = float(total / observed.n_samples)
    except (FloatingPointError, ValueError) as error:
        return _ineligible("y_scoring_failed", {**diagnostics, "detail": str(error)})
    if not np.isfinite(value):
        return _ineligible("nonfinite_y_score", diagnostics)
    return ValidationScore(value, "eligible", None, diagnostics)


def score_y_partition_risk(
    validation: ObservedDataView,
    labels: np.ndarray,
    parameters: Parameters,
    feature_transform: FeatureTransform,
    *,
    epsilon_prob: float,
) -> ValidationScore:
    """Primary Y risk for a frozen hard partition on held-out rows."""

    observed = _require_observed(validation)
    epsilon = _validate_epsilon_prob(epsilon_prob)
    if observed.n_samples == 0 or observed.centers.size == 0:
        return _ineligible("empty_validation")
    transform = _validate_transform(feature_transform, observed.features.shape[1])
    if not isinstance(parameters, Parameters):
        raise TypeError("parameters must be Parameters fitted on training rows")
    diagnostics = _primary_risk_diagnostics(observed, epsilon_prob=epsilon)
    diagnostics["feature_transform_source"] = "training_fitted_reused"
    try:
        parameters.validate()
        canonical = _canonical_labels(observed, labels)
        if parameters.k_components != int(np.unique(canonical).size):
            raise ValueError("partition class count does not match outcome predictor")
        _validate_binary(observed.outcome, "outcome")
        transformed = transform.transform(observed.features)
        eta = linear_predictor(parameters, observed.treatment, transformed)
        sample_classes = canonical[dense_center_codes(observed)]
        probabilities = sigmoid(eta[np.arange(observed.n_samples), sample_classes])
        losses = clipped_bernoulli_nll(
            observed.outcome,
            probabilities,
            epsilon_prob=epsilon,
        )
        value = float(np.mean(losses))
    except (FloatingPointError, ValueError) as error:
        return _ineligible("y_primary_risk_failed", {**diagnostics, "detail": str(error)})
    return ValidationScore(value, "eligible", None, diagnostics)


def x_ridge_objective_and_gradient(
    theta: np.ndarray,
    features: np.ndarray,
    treatment: np.ndarray,
    lambda_x: float,
) -> tuple[float, np.ndarray]:
    """Class-fit mean logistic NLL and analytic gradient with no intercept penalty."""

    coefficients = np.asarray(theta, dtype=float)
    matrix = np.asarray(features, dtype=float)
    response = np.asarray(treatment, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    if coefficients.shape != (matrix.shape[1] + 1,) or response.shape != (matrix.shape[0],):
        raise ValueError("ridge inputs have incompatible dimensions")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("ridge inputs must be finite")
    _validate_binary(response, "treatment")
    if not np.isfinite(lambda_x) or lambda_x < 0.0:
        raise ValueError("lambda_x must be finite and nonnegative")
    eta = coefficients[0] + matrix @ coefficients[1:]
    residual = sigmoid(eta) - response
    objective = float(
        np.mean(np.logaddexp(0.0, eta) - response * eta)
        + 0.5 * lambda_x * np.dot(coefficients[1:], coefficients[1:])
    )
    gradient = np.concatenate(
        [
            np.array([residual.mean()]),
            matrix.T @ residual / response.size + lambda_x * coefficients[1:],
        ]
    )
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite X ridge objective or gradient")
    return objective, gradient


@dataclass(frozen=True)
class XPartitionModel:
    context_digest: str
    partition_key: tuple[int, ...]
    centers: np.ndarray
    coefficients: np.ndarray
    lambda_x: float
    objective_normalization: str
    training_objectives: tuple[float, ...]
    optimizer_diagnostics: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition_key", tuple(int(v) for v in self.partition_key))
        object.__setattr__(self, "centers", _readonly(self.centers, dtype=None))
        object.__setattr__(self, "coefficients", _readonly(self.coefficients))
        object.__setattr__(
            self,
            "training_objectives",
            tuple(float(value) for value in self.training_objectives),
        )
        object.__setattr__(
            self,
            "optimizer_diagnostics",
            tuple(dict(value) for value in self.optimizer_diagnostics),
        )


@dataclass(frozen=True)
class XFitResult:
    model: XPartitionModel | None
    status: str
    reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _copy_transform(transform: FeatureTransform) -> FeatureTransform:
    return FeatureTransform(mean=np.array(transform.mean, copy=True), scale=np.array(transform.scale, copy=True))


def _copy_observed(data: ObservedDataView) -> ObservedDataView:
    return ObservedDataView(
        features=np.array(data.features, copy=True),
        treatment=np.array(data.treatment, copy=True),
        outcome=np.array(data.outcome, copy=True),
        center_codes=np.array(data.center_codes, copy=True),
        centers=np.array(data.centers, copy=True),
    )


def _context_digest(data: ObservedDataView, transform: FeatureTransform) -> str:
    digest = hashlib.sha256()
    for array in (
        data.features,
        data.treatment,
        data.center_codes,
        data.centers,
        transform.mean,
        transform.scale,
    ):
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


class XModelCache:
    """Direction-local canonical-partition cache for class-specific X models."""

    def __init__(
        self,
        training: ObservedDataView,
        feature_transform: FeatureTransform,
        *,
        max_iter: int = 200,
    ) -> None:
        observed = _require_observed(training)
        transform = _validate_transform(feature_transform, observed.features.shape[1])
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if observed.n_samples == 0:
            raise ValueError("training data must not be empty")
        _validate_binary(observed.treatment, "treatment")
        dense_center_codes(observed)
        self._training = _copy_observed(observed)
        self._transform = _copy_transform(transform)
        self._features = self._transform.transform(self._training.features)
        self._max_iter = int(max_iter)
        self._context_digest = _context_digest(self._training, self._transform)
        self._cache: dict[tuple[str, tuple[int, ...]], XFitResult] = {}
        self._hits = 0
        self._misses = 0
        self._fit_count = 0

    @property
    def diagnostics(self) -> dict[str, int]:
        return {
            "cache_hits": int(self._hits),
            "cache_misses": int(self._misses),
            "fit_count": int(self._fit_count),
        }

    def fit(self, labels: np.ndarray, *, minimum_cluster_size: int = 1) -> XFitResult:
        canonical = _canonical_labels(
            self._training,
            labels,
            minimum_cluster_size=minimum_cluster_size,
        )
        partition = tuple(int(v) for v in canonical)
        cache_key = (self._context_digest, partition)
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        self._misses += 1
        self._fit_count += 1
        result = self._fit_uncached(canonical)
        self._cache[cache_key] = result
        return result

    def _fit_uncached(self, canonical: np.ndarray) -> XFitResult:
        dense = dense_center_codes(self._training)
        sample_classes = canonical[dense]
        k_components = int(canonical.max()) + 1
        p = int(self._features.shape[1])
        lambda_x = float(math.sqrt(math.log(max(p, 2)) / self._training.n_samples))
        zero_variance = int(np.sum(np.std(self._training.features, axis=0, ddof=0) == 0.0))
        base_diagnostics: dict[str, Any] = {
            "lambda_x": lambda_x,
            "lambda_formula": "sqrt(log(max(p,2))/N_train)",
            "objective_normalization": "class_mean_nll_plus_lambda_l2_over_2",
            "intercept_penalized": False,
            "n_train": int(self._training.n_samples),
            "zero_variance_feature_count": zero_variance,
            "context_digest": self._context_digest,
        }
        coefficients: list[np.ndarray] = []
        objectives: list[float] = []
        optimizer_diagnostics: list[dict[str, Any]] = []
        for cluster in range(k_components):
            mask = sample_classes == cluster
            response = self._training.treatment[mask].astype(float, copy=False)
            if response.size == 0:
                return XFitResult(None, "ineligible", "empty_treatment_class", {**base_diagnostics, "failed_class": cluster})
            if np.unique(response).size < 2:
                return XFitResult(
                    None,
                    "ineligible",
                    "constant_treatment_within_class",
                    {**base_diagnostics, "failed_class": cluster, "class_n": int(response.size)},
                )
            probability = float(response.mean())
            start = np.zeros(p + 1, dtype=float)
            start[0] = math.log(probability / (1.0 - probability))

            # E4 (perf): hoist the boolean-mask copy out of the closure.  It used to be
            # rebuilt on every objective evaluation (~25 per class fit) even though the
            # rows never change; x_ridge_objective_and_gradient only reads it.  Bound to
            # the current class only, so it is released before the next iteration.
            class_features = self._features[mask]

            def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
                return x_ridge_objective_and_gradient(theta, class_features, response, lambda_x)

            optimized = minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self._max_iter, "ftol": 1.0e-12, "gtol": 1.0e-8},
            )
            detail = {
                "class": cluster,
                "class_n": int(response.size),
                "success": bool(optimized.success),
                "status": int(optimized.status),
                "message": str(optimized.message),
                "iterations": int(optimized.nit),
                "function_evaluations": int(optimized.nfev),
                "objective": _json_number(float(optimized.fun)),
            }
            optimizer_diagnostics.append(detail)
            if (
                not optimized.success
                or not np.all(np.isfinite(optimized.x))
                or not np.isfinite(optimized.fun)
            ):
                return XFitResult(
                    None,
                    "ineligible",
                    "x_optimizer_failed",
                    {**base_diagnostics, "failed_class": cluster, "optimizer": detail},
                )
            direct_objective, direct_gradient = objective(np.asarray(optimized.x))
            if not np.isclose(direct_objective, float(optimized.fun), atol=1.0e-10, rtol=1.0e-10):
                return XFitResult(
                    None,
                    "ineligible",
                    "x_objective_audit_failed",
                    {**base_diagnostics, "failed_class": cluster},
                )
            detail["direct_objective"] = float(direct_objective)
            detail["max_abs_gradient"] = float(np.max(np.abs(direct_gradient)))
            coefficients.append(np.asarray(optimized.x, dtype=float))
            objectives.append(float(direct_objective))
        model = XPartitionModel(
            context_digest=self._context_digest,
            partition_key=tuple(int(v) for v in canonical),
            centers=self._training.centers,
            coefficients=np.vstack(coefficients),
            lambda_x=lambda_x,
            objective_normalization="class_mean_nll_plus_lambda_l2_over_2",
            training_objectives=tuple(objectives),
            optimizer_diagnostics=tuple(optimizer_diagnostics),
        )
        return XFitResult(
            model,
            "eligible",
            None,
            {**base_diagnostics, "class_fits": optimizer_diagnostics},
        )

    def score(
        self,
        validation: ObservedDataView,
        labels: np.ndarray,
        model: XPartitionModel | None,
    ) -> ValidationScore:
        observed = _require_observed(validation)
        diagnostics = {
            "normalization": "total_loglik/N_validation",
            "feature_transform_source": "training_fitted_reused",
            "n_validation": int(observed.n_samples),
        }
        if model is None:
            return _ineligible("missing_x_model", diagnostics)
        if model.context_digest != self._context_digest:
            return _ineligible("x_model_context_mismatch", diagnostics)
        if observed.n_samples == 0:
            return _ineligible("empty_validation", diagnostics)
        if not np.array_equal(observed.centers, model.centers):
            return _ineligible("validation_center_order_mismatch", diagnostics)
        try:
            canonical = _canonical_labels(observed, labels)
            _validate_binary(observed.treatment, "treatment")
        except ValueError as error:
            return _ineligible("invalid_validation_labels_or_treatment", {**diagnostics, "detail": str(error)})
        if tuple(int(v) for v in canonical) != model.partition_key:
            return _ineligible("partition_mismatch", diagnostics)
        features = self._transform.transform(observed.features)
        dense = dense_center_codes(observed)
        sample_classes = canonical[dense]
        eta = np.empty(observed.n_samples, dtype=float)
        for cluster in range(model.coefficients.shape[0]):
            mask = sample_classes == cluster
            coefficient = model.coefficients[cluster]
            eta[mask] = coefficient[0] + features[mask] @ coefficient[1:]
        response = observed.treatment.astype(float, copy=False)
        total = float(np.sum(response * eta - np.logaddexp(0.0, eta)))
        value = total / observed.n_samples
        if not np.isfinite(value):
            return _ineligible("nonfinite_x_score", diagnostics)
        return ValidationScore(float(value), "eligible", None, diagnostics)

    def primary_risk(
        self,
        validation: ObservedDataView,
        labels: np.ndarray,
        model: XPartitionModel | None,
        *,
        epsilon_prob: float,
    ) -> ValidationScore:
        """Primary X risk for one frozen partition-specific treatment predictor."""

        observed = _require_observed(validation)
        epsilon = _validate_epsilon_prob(epsilon_prob)
        if observed.n_samples == 0:
            return _ineligible("empty_validation")
        diagnostics = _primary_risk_diagnostics(observed, epsilon_prob=epsilon)
        diagnostics["feature_transform_source"] = "training_fitted_reused"
        if model is None:
            return _ineligible("missing_x_model", diagnostics)
        if model.context_digest != self._context_digest:
            return _ineligible("x_model_context_mismatch", diagnostics)
        if not np.array_equal(observed.centers, model.centers):
            return _ineligible("validation_center_order_mismatch", diagnostics)
        try:
            canonical = _canonical_labels(observed, labels)
            _validate_binary(observed.treatment, "treatment")
            if tuple(int(v) for v in canonical) != model.partition_key:
                raise ValueError("partition_mismatch")
            features = self._transform.transform(observed.features)
            sample_classes = canonical[dense_center_codes(observed)]
            eta = np.empty(observed.n_samples, dtype=float)
            for cluster in range(model.coefficients.shape[0]):
                mask = sample_classes == cluster
                coefficient = model.coefficients[cluster]
                eta[mask] = coefficient[0] + features[mask] @ coefficient[1:]
            losses = clipped_bernoulli_nll(
                observed.treatment,
                sigmoid(eta),
                epsilon_prob=epsilon,
            )
            value = float(np.mean(losses))
        except (FloatingPointError, ValueError) as error:
            return _ineligible("x_primary_risk_failed", {**diagnostics, "detail": str(error)})
        return ValidationScore(value, "eligible", None, diagnostics)


@dataclass(frozen=True)
class FMomentModel:
    partition_key: tuple[int, ...]
    centers: np.ndarray
    global_feature_mean: np.ndarray
    global_feature_variance: np.ndarray
    global_feature_scale: np.ndarray
    selected_coordinates: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_scale: np.ndarray
    class_centroids: np.ndarray
    diagonal_scale: np.ndarray
    eps: float
    scale_floor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition_key", tuple(int(v) for v in self.partition_key))
        object.__setattr__(self, "centers", _readonly(self.centers, dtype=None))
        for name in (
            "global_feature_mean",
            "global_feature_variance",
            "global_feature_scale",
            "coordinate_mean",
            "coordinate_scale",
            "class_centroids",
            "diagonal_scale",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "selected_coordinates", _readonly(self.selected_coordinates, dtype=np.int64))


@dataclass(frozen=True)
class FFitResult:
    model: FMomentModel | None
    status: str
    reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _center_moments(
    data: ObservedDataView,
    global_mean: np.ndarray,
    global_variance: np.ndarray,
    global_scale: np.ndarray,
    eps: float,
) -> np.ndarray:
    observed = _require_observed(data)
    if observed.n_samples == 0 or observed.centers.size == 0:
        raise ValueError("data must contain validation rows for every center")
    if not np.all(np.isfinite(observed.features)):
        raise ValueError("features must be finite")
    dense = dense_center_codes(observed)
    rows: list[np.ndarray] = []
    for center_index in range(observed.centers.size):
        block = observed.features[dense == center_index]
        mean_score = np.sqrt(float(block.shape[0])) * (block.mean(axis=0) - global_mean) / global_scale
        variance_score = np.log((block.var(axis=0, ddof=0) + eps) / (global_variance + eps))
        rows.append(np.concatenate([mean_score, variance_score]))
    result = np.vstack(rows)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite covariate moment coordinates")
    return result


def fit_f_moment_model(
    training: ObservedDataView,
    labels: np.ndarray,
    *,
    selected_coordinates: np.ndarray | None = None,
    eps: float = 1.0e-8,
    scale_floor: float = 1.0e-8,
    minimum_cluster_size: int = 1,
) -> FFitResult:
    """Fit all F-moment coordinates, centroids, and scales on training rows only."""

    observed = _require_observed(training)
    if not np.isfinite(eps) or eps <= 0.0 or not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("eps and scale_floor must be positive and finite")
    if observed.n_samples == 0 or observed.features.shape[1] < 1:
        return FFitResult(None, "ineligible", "empty_training", {})
    try:
        canonical = _canonical_labels(observed, labels, minimum_cluster_size=minimum_cluster_size)
        if not np.all(np.isfinite(observed.features)):
            raise ValueError("features must be finite")
        dense_center_codes(observed)
    except ValueError as error:
        return FFitResult(None, "ineligible", "invalid_f_training_input", {"detail": str(error)})
    p = observed.features.shape[1]
    if selected_coordinates is None:
        selected = np.arange(2 * p, dtype=np.int64)
    else:
        raw_selected = np.asarray(selected_coordinates)
        if (
            raw_selected.ndim != 1
            or raw_selected.size == 0
            or np.issubdtype(raw_selected.dtype, np.bool_)
            or not np.issubdtype(raw_selected.dtype, np.integer)
            or np.any(raw_selected < 0)
            or np.any(raw_selected >= 2 * p)
            or np.unique(raw_selected).size != raw_selected.size
        ):
            return FFitResult(None, "ineligible", "invalid_selected_coordinates", {})
        selected = raw_selected.astype(np.int64, copy=True)
    global_mean = observed.features.mean(axis=0)
    global_variance = observed.features.var(axis=0, ddof=0)
    global_scale = np.maximum(np.sqrt(global_variance), eps)
    try:
        raw_moments = _center_moments(observed, global_mean, global_variance, global_scale, eps)[:, selected]
    except (FloatingPointError, ValueError) as error:
        return FFitResult(None, "ineligible", "f_moment_construction_failed", {"detail": str(error)})
    coordinate_mean = raw_moments.mean(axis=0)
    raw_coordinate_scale = raw_moments.std(axis=0, ddof=0)
    coordinate_scale = np.maximum(raw_coordinate_scale, scale_floor)
    standardized = (raw_moments - coordinate_mean) / coordinate_scale
    k_components = int(canonical.max()) + 1
    centroids = np.vstack([standardized[canonical == cluster].mean(axis=0) for cluster in range(k_components)])
    residual = standardized - centroids[canonical]
    raw_diagonal_scale = np.sqrt(np.mean(residual * residual, axis=0))
    diagonal_scale = np.maximum(raw_diagonal_scale, scale_floor)
    model = FMomentModel(
        partition_key=tuple(int(v) for v in canonical),
        centers=observed.centers,
        global_feature_mean=global_mean,
        global_feature_variance=global_variance,
        global_feature_scale=global_scale,
        selected_coordinates=selected,
        coordinate_mean=coordinate_mean,
        coordinate_scale=coordinate_scale,
        class_centroids=centroids,
        diagonal_scale=diagonal_scale,
        eps=float(eps),
        scale_floor=float(scale_floor),
    )
    diagnostics = {
        "transform_source": "training_only",
        "retained_dimension": int(selected.size),
        "selected_coordinates": selected.astype(int).tolist(),
        "variance_ddof": 0,
        "scale_floor": float(scale_floor),
        "coordinate_scale_floor_count": int(np.sum(raw_coordinate_scale < scale_floor)),
        "diagonal_scale_floor_count": int(np.sum(raw_diagonal_scale < scale_floor)),
        "global_feature_mean": global_mean.astype(float).tolist(),
        "global_feature_variance": global_variance.astype(float).tolist(),
        "coordinate_mean": coordinate_mean.astype(float).tolist(),
        "coordinate_scale": coordinate_scale.astype(float).tolist(),
        "class_centroids": centroids.astype(float).tolist(),
        "diagonal_scale": diagonal_scale.astype(float).tolist(),
    }
    return FFitResult(model, "eligible", None, diagnostics)


def score_f_moment_model(
    validation: ObservedDataView,
    labels: np.ndarray,
    model: FMomentModel | None,
) -> ValidationScore:
    """Score frozen validation-center moments by equal-center standardized distance."""

    observed = _require_observed(validation)
    diagnostics: dict[str, Any] = {
        "center_weighting": "equal",
        "transform_source": "training_fitted_reused",
        "n_validation_centers": int(observed.centers.size),
    }
    if model is None:
        return _ineligible("missing_f_model", diagnostics)
    dimension = int(model.selected_coordinates.size)
    diagnostics["retained_dimension"] = dimension
    if observed.n_samples == 0 or observed.centers.size == 0:
        return _ineligible("empty_validation", diagnostics)
    if dimension < 1:
        return _ineligible("no_retained_f_coordinates", diagnostics)
    if observed.features.shape[1] != model.global_feature_mean.size:
        return _ineligible("f_feature_dimension_mismatch", diagnostics)
    if not np.array_equal(observed.centers, model.centers):
        return _ineligible("validation_center_order_mismatch", diagnostics)
    if np.any(model.diagonal_scale <= 0.0) or not np.all(np.isfinite(model.diagonal_scale)):
        return _ineligible("invalid_f_diagonal_scale", diagnostics)
    try:
        canonical = _canonical_labels(observed, labels)
    except ValueError as error:
        return _ineligible("invalid_validation_labels", {**diagnostics, "detail": str(error)})
    if tuple(int(v) for v in canonical) != model.partition_key:
        return _ineligible("partition_mismatch", diagnostics)
    try:
        raw = _center_moments(
            observed,
            model.global_feature_mean,
            model.global_feature_variance,
            model.global_feature_scale,
            model.eps,
        )[:, model.selected_coordinates]
        standardized = (raw - model.coordinate_mean) / model.coordinate_scale
        residual = (standardized - model.class_centroids[canonical]) / model.diagonal_scale
        contributions = -np.sum(residual * residual, axis=1) / dimension
    except (FloatingPointError, ValueError) as error:
        return _ineligible("f_scoring_failed", {**diagnostics, "detail": str(error)})
    value = float(np.mean(contributions))
    if not np.isfinite(value) or not np.all(np.isfinite(contributions)):
        return _ineligible("nonfinite_f_score", diagnostics)
    diagnostics["per_center_contributions"] = contributions.astype(float).tolist()
    diagnostics["center_order"] = [value.item() if isinstance(value, np.generic) else value for value in observed.centers]
    return ValidationScore(value, "eligible", None, diagnostics)


__all__ = [
    "ValidationScore",
    "XPartitionModel",
    "XFitResult",
    "XModelCache",
    "FMomentModel",
    "FFitResult",
    "clipped_bernoulli_nll",
    "score_y_observed_mixture",
    "score_y_partition_risk",
    "x_ridge_objective_and_gradient",
    "fit_f_moment_model",
    "score_f_moment_model",
]
