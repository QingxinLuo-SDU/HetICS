"""Bernoulli-sEAM observed likelihood, E-step, and generalized-EM/IHT update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class FitError(RuntimeError):
    """Base class for machine-readable fitting failures."""


class EmptyEffectiveClassError(FitError):
    """Raised when an update has zero effective sample size."""


@dataclass(frozen=True)
class FitConfig:
    support_size: int
    step_size: float = 0.05
    backtracking_factor: float = 0.5
    max_backtracking: int = 12
    warmup_steps: int = 50
    max_iter: int = 500
    parameter_tolerance: float = 1e-5
    objective_relative_tolerance: float = 1e-7
    initialization_smoothing: float = 0.02
    q_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.support_size < 1:
            raise ValueError("support_size must be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must be in (0, 1)")
        if self.max_backtracking < 0 or self.max_iter < 1:
            raise ValueError("Invalid iteration budget")
        if not 0 <= self.initialization_smoothing < 1:
            raise ValueError("initialization_smoothing must be in [0, 1)")


@dataclass
class Parameters:
    pi: np.ndarray
    gamma: float
    intercepts: np.ndarray
    beta: np.ndarray

    def clone(self) -> "Parameters":
        return Parameters(
            pi=self.pi.copy(),
            gamma=float(self.gamma),
            intercepts=self.intercepts.copy(),
            beta=self.beta.copy(),
        )

    @property
    def k_components(self) -> int:
        return int(self.pi.size)

    def validate(self) -> None:
        if self.intercepts.shape != (self.k_components,):
            raise ValueError("Intercept dimension does not match pi")
        if self.beta.ndim != 2 or self.beta.shape[0] != self.k_components:
            raise ValueError("Beta dimension does not match pi")
        if np.any(self.pi < 0) or not np.isclose(
            self.pi.sum(), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("pi must be on the probability simplex")
        values = np.concatenate(
            [self.pi, [self.gamma], self.intercepts, self.beta.ravel()]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Parameters contain NaN or infinity")


@dataclass
class StepInfo:
    q_pi_old: float
    q_pi_new: float
    q_reg_old: float
    q_reg_new: float
    backtracking_count: int
    accepted_step_size: float
    backtracking_failed: bool


@dataclass
class FitResult:
    parameters: Parameters
    responsibilities: np.ndarray
    trajectory: list[dict[str, Any]]
    converged: bool
    stop_reason: str
    observed_decrease_steps: int
    backtracking_failures: int


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Evaluate expit stably without clipping probabilities or predictors."""

    value = np.asarray(value, dtype=float)
    output = np.empty_like(value)
    nonnegative = value >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exponential = np.exp(value[~nonnegative])
    output[~nonnegative] = exponential / (1.0 + exponential)
    return output


def linear_predictor(
    parameters: Parameters,
    treatment: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return (
        parameters.intercepts[None, :]
        + parameters.gamma * treatment[:, None]
        + features @ parameters.beta.T
    )

def e_step(
    parameters: Parameters,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    n_centers: int,
) -> tuple[float, np.ndarray]:
    """Return the observed log-likelihood and center-level responsibilities."""

    eta = linear_predictor(parameters, treatment, features)
    sample_loglik = outcome[:, None] * eta - np.logaddexp(0.0, eta)
    component_loglik = np.zeros((n_centers, parameters.k_components), dtype=float)
    for cluster in range(parameters.k_components):
        component_loglik[:, cluster] = np.bincount(
            center_codes,
            weights=sample_loglik[:, cluster],
            minlength=n_centers,
        )
    with np.errstate(divide="ignore"):
        logits = component_loglik + np.log(parameters.pi)[None, :]
    row_max = np.max(logits, axis=1, keepdims=True)
    if np.any(~np.isfinite(row_max)):
        raise FitError("Every component has zero probability for at least one center")
    shifted = np.exp(logits - row_max)
    denominator = shifted.sum(axis=1, keepdims=True)
    responsibilities = shifted / denominator
    observed = float(np.sum(row_max[:, 0] + np.log(denominator[:, 0])))
    if not np.isfinite(observed) or not np.all(np.isfinite(responsibilities)):
        raise FitError("Non-finite observed likelihood or responsibilities")
    return observed, responsibilities


def q_pi(parameters: Parameters, responsibilities: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            responsibilities > 0,
            responsibilities * np.log(parameters.pi)[None, :],
            0.0,
        )
    return float(np.sum(terms) / responsibilities.shape[0])


def _regression_q_parts(
    parameters: Parameters,
    responsibilities: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
) -> tuple[float, np.ndarray]:
    """E3 (perf): the original body, additionally handing back ``sample_loglik``.

    ``component_loglik_matrix`` recomputes exactly this array from exactly these
    parameters one call later (``pi`` is the only difference between them, and neither
    the predictor nor the per-row log-likelihood reads it).  Returning it lets the
    fixed-label refit aggregate the same numbers a second way instead of recomputing
    them.  The reduction below is untouched, so ``regression_q`` stays bit-for-bit
    what it was.
    """

    eta = linear_predictor(parameters, treatment, features)
    sample_loglik = outcome[:, None] * eta - np.logaddexp(0.0, eta)
    weights = responsibilities[center_codes]
    return float(np.sum(weights * sample_loglik) / outcome.size), sample_loglik


def regression_q(
    parameters: Parameters,
    responsibilities: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
) -> float:
    return _regression_q_parts(
        parameters, responsibilities, treatment, features, outcome, center_codes
    )[0]


def regression_gradient(
    parameters: Parameters,
    responsibilities: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute N-normalized common and class-block gradients."""

    eta = linear_predictor(parameters, treatment, features)
    residual = outcome[:, None] - sigmoid(eta)
    sample_weights = responsibilities[center_codes]
    weighted = sample_weights * residual
    n_samples = outcome.size
    gamma_gradient = float(np.sum(weighted * treatment[:, None]) / n_samples)
    intercept_gradient = weighted.sum(axis=0) / n_samples
    beta_gradient = weighted.T @ features / n_samples
    effective_sizes = sample_weights.sum(axis=0)
    return gamma_gradient, intercept_gradient, beta_gradient, effective_sizes


def hard_threshold(vector: np.ndarray, support_size: int) -> np.ndarray:
    """Keep the largest absolute coordinates with deterministic tie-breaking."""

    if support_size >= vector.size:
        return vector.copy()
    if support_size <= 0:
        return np.zeros_like(vector)
    indices = np.arange(vector.size)
    order = np.lexsort((indices, -np.abs(vector)))
    keep = order[:support_size]
    thresholded = np.zeros_like(vector)
    thresholded[keep] = vector[keep]
    return thresholded


def _regression_proposal(
    parameters: Parameters,
    responsibilities: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    config: FitConfig,
    step_size: float,
) -> Parameters:
    (
        gamma_gradient,
        intercept_gradient,
        beta_gradient,
        effective_sizes,
    ) = regression_gradient(
        parameters,
        responsibilities,
        treatment,
        features,
        outcome,
        center_codes,
    )
    if np.any(effective_sizes <= 0):
        raise EmptyEffectiveClassError(
            f"Empty effective class: {effective_sizes.tolist()}"
        )
    class_scaling = outcome.size / effective_sizes
    raw_beta = parameters.beta + step_size * class_scaling[:, None] * beta_gradient
    return Parameters(
        pi=parameters.pi.copy(),
        gamma=float(parameters.gamma + step_size * gamma_gradient),
        intercepts=(
            parameters.intercepts + step_size * class_scaling * intercept_gradient
        ),
        beta=np.vstack(
            [
                hard_threshold(raw_beta[cluster], config.support_size)
                for cluster in range(parameters.k_components)
            ]
        ),
    )


def generalized_m_step(
    parameters: Parameters,
    responsibilities: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    config: FitConfig,
    *,
    q_reg_old: float | None = None,
) -> tuple[Parameters, StepInfo]:
    """Public M-step: unchanged signature and unchanged ``(Parameters, StepInfo)``.

    The n x K array E3 reuses stays out of ``StepInfo`` on purpose -- that object is
    kept in every trajectory row and serialized into diagnostics.  Callers that want it
    use the private ``_generalized_m_step_with_loglik`` instead.
    """

    parameters, info, _loglik = _generalized_m_step_with_loglik(
        parameters, responsibilities, treatment, features, outcome, center_codes,
        config, q_reg_old=q_reg_old,
    )
    return parameters, info


def _generalized_m_step_with_loglik(
    parameters: Parameters,
    responsibilities: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    config: FitConfig,
    *,
    q_reg_old: float | None = None,
) -> tuple[Parameters, StepInfo, np.ndarray | None]:
    """Exact pi update plus one accepted preconditioned IHT regression step.

    Third element is the accepted candidate's per-row log-likelihood (E3), or ``None``
    when backtracking failed and no candidate was accepted.

    E2 (perf): ``q_reg_old`` may be supplied by a caller that already holds the value
    for *these* parameters and responsibilities -- in a fixed-label refit the accepted
    ``q_reg_new`` of step t is, by construction, the ``q_reg_old`` of step t+1
    (``regression_q`` never reads ``pi``, and the accepted candidate becomes ``current``).
    Default ``None`` keeps the original behaviour, and callers whose responsibilities or
    parameters can change between steps -- the CEM outer loop, warmup, the full EM --
    must not pass it.
    """

    q_pi_old = q_pi(parameters, responsibilities)
    new_pi = responsibilities.mean(axis=0)
    pi_parameters = parameters.clone()
    pi_parameters.pi = new_pi
    q_pi_new = q_pi(pi_parameters, responsibilities)
    if q_pi_new + config.q_tolerance < q_pi_old:
        raise FitError("Ordinary pi update lowered Q_pi")

    if q_reg_old is None:
        q_reg_old = regression_q(
            parameters,
            responsibilities,
            treatment,
            features,
            outcome,
            center_codes,
        )
    else:
        q_reg_old = float(q_reg_old)
    accepted: Parameters | None = None
    accepted_q = q_reg_old
    accepted_step = 0.0
    accepted_loglik: np.ndarray | None = None
    attempt = 0
    for attempt in range(config.max_backtracking + 1):
        candidate_step = config.step_size * config.backtracking_factor**attempt
        candidate = _regression_proposal(
            parameters,
            responsibilities,
            treatment,
            features,
            outcome,
            center_codes,
            config,
            candidate_step,
        )
        candidate_q, candidate_loglik = _regression_q_parts(
            candidate,
            responsibilities,
            treatment,
            features,
            outcome,
            center_codes,
        )
        if candidate_q + config.q_tolerance >= q_reg_old:
            accepted = candidate
            accepted_q = candidate_q
            accepted_step = candidate_step
            # E3: keep only the *accepted* candidate's array; a rejected attempt's
            # numbers must never be reused.
            accepted_loglik = candidate_loglik
            break

    failed = accepted is None
    if accepted is None:
        accepted = parameters.clone()
        accepted_loglik = None
    accepted.pi = new_pi
    accepted.validate()
    return accepted, StepInfo(
        q_pi_old=q_pi_old,
        q_pi_new=q_pi_new,
        q_reg_old=q_reg_old,
        q_reg_new=accepted_q,
        backtracking_count=attempt if not failed else config.max_backtracking + 1,
        accepted_step_size=accepted_step,
        backtracking_failed=failed,
    ), accepted_loglik


def hard_label_m_step(
    parameters: Parameters,
    labels: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    config: FitConfig,
    *,
    q_reg_old: float | None = None,
) -> tuple[Parameters, StepInfo]:
    """Run one fixed-label M-step and reject failed backtracking loudly.

    ``q_reg_old`` is the E2 pass-through; see ``generalized_m_step``.  Only a caller
    that holds the labels fixed across steps may supply it.
    """

    updated, info, _loglik = _hard_label_m_step_with_loglik(
        parameters, labels, treatment, features, outcome, center_codes, config,
        q_reg_old=q_reg_old,
    )
    return updated, info


def _hard_label_m_step_with_loglik(
    parameters: Parameters,
    labels: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    config: FitConfig,
    *,
    q_reg_old: float | None = None,
) -> tuple[Parameters, StepInfo, np.ndarray | None]:
    """``hard_label_m_step`` plus the accepted candidate's per-row log-likelihood (E3)."""

    responsibilities = np.eye(parameters.k_components, dtype=float)[labels]
    updated, info, loglik = _generalized_m_step_with_loglik(
        parameters,
        responsibilities,
        treatment,
        features,
        outcome,
        center_codes,
        config,
        q_reg_old=q_reg_old,
    )
    if info.backtracking_failed:
        raise FitError("hard-label M-step backtracking failed")
    return updated, info, loglik


def parameter_change(old: Parameters, new: Parameters) -> float:
    difference_sq = (
        np.sum((new.pi - old.pi) ** 2)
        + (new.gamma - old.gamma) ** 2
        + np.sum((new.intercepts - old.intercepts) ** 2)
        + np.sum((new.beta - old.beta) ** 2)
    )
    old_norm_sq = (
        np.sum(old.pi**2)
        + old.gamma**2
        + np.sum(old.intercepts**2)
        + np.sum(old.beta**2)
    )
    return float(math.sqrt(difference_sq) / (1.0 + math.sqrt(old_norm_sq)))


def responsibility_entropy(responsibilities: np.ndarray) -> float:
    k_components = responsibilities.shape[1]
    entropy = -np.sum(
        responsibilities * np.log(np.clip(responsibilities, 1e-15, 1.0)),
        axis=1,
    )
    return float(np.mean(entropy) / max(math.log(k_components), 1.0))


def warmup_fixed_responsibilities(
    parameters: Parameters,
    responsibilities: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    config: FitConfig,
) -> tuple[Parameters, list[dict[str, Any]]]:
    trajectory: list[dict[str, Any]] = []
    current = parameters
    for warmup_iteration in range(1, config.warmup_steps + 1):
        new, info = generalized_m_step(
            current,
            responsibilities,
            treatment,
            features,
            outcome,
            center_codes,
            config,
        )
        trajectory.append(
            {
                "warmup_iteration": warmup_iteration,
                "q_reg": info.q_reg_new,
                "q_pi": info.q_pi_new,
                "accepted_step_size": info.accepted_step_size,
                "backtracking_count": info.backtracking_count,
                "backtracking_failed": info.backtracking_failed,
                "parameter_change": parameter_change(current, new),
            }
        )
        current = new
    return current, trajectory


def fit(
    initial: Parameters,
    treatment: np.ndarray,
    features: np.ndarray,
    outcome: np.ndarray,
    center_codes: np.ndarray,
    n_centers: int,
    config: FitConfig,
) -> FitResult:
    """Run the frozen generalized-EM/IHT iterations."""

    initial.validate()
    current = initial
    observed, responsibilities = e_step(
        current,
        treatment,
        features,
        outcome,
        center_codes,
        n_centers,
    )
    trajectory: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "observed_loglik": observed,
            "observed_loglik_per_sample": observed / outcome.size,
            "parameter_change": None,
            "objective_relative_change": None,
            "responsibility_entropy": responsibility_entropy(responsibilities),
            "pi": current.pi.tolist(),
            "gamma": current.gamma,
            "support_sizes": np.count_nonzero(current.beta, axis=1).tolist(),
            "accepted_step_size": None,
            "backtracking_count": 0,
            "backtracking_failed": False,
            "q_pi_improvement": None,
            "q_reg_improvement": None,
        }
    ]
    observed_decreases = 0
    backtracking_failures = 0
    converged = False
    stop_reason = "max_iter"

    for iteration in range(1, config.max_iter + 1):
        new, info = generalized_m_step(
            current,
            responsibilities,
            treatment,
            features,
            outcome,
            center_codes,
            config,
        )
        new_observed, new_responsibilities = e_step(
            new,
            treatment,
            features,
            outcome,
            center_codes,
            n_centers,
        )
        change = parameter_change(current, new)
        objective_change = abs(new_observed - observed) / (1.0 + abs(observed))
        if new_observed + 1e-8 < observed:
            observed_decreases += 1
        if info.backtracking_failed:
            backtracking_failures += 1
        trajectory.append(
            {
                "iteration": iteration,
                "observed_loglik": new_observed,
                "observed_loglik_per_sample": new_observed / outcome.size,
                "parameter_change": change,
                "objective_relative_change": objective_change,
                "responsibility_entropy": responsibility_entropy(
                    new_responsibilities
                ),
                "pi": new.pi.tolist(),
                "gamma": new.gamma,
                "support_sizes": np.count_nonzero(new.beta, axis=1).tolist(),
                "accepted_step_size": info.accepted_step_size,
                "backtracking_count": info.backtracking_count,
                "backtracking_failed": info.backtracking_failed,
                "q_pi_improvement": info.q_pi_new - info.q_pi_old,
                "q_reg_improvement": info.q_reg_new - info.q_reg_old,
            }
        )
        current = new
        observed = new_observed
        responsibilities = new_responsibilities
        if (
            change <= config.parameter_tolerance
            and objective_change <= config.objective_relative_tolerance
        ):
            converged = True
            stop_reason = "parameter_and_objective_tolerance"
            break
        if info.backtracking_failed:
            stop_reason = "backtracking_failed"
            break

    return FitResult(
        parameters=current,
        responsibilities=responsibilities,
        trajectory=trajectory,
        converged=converged,
        stop_reason=stop_reason,
        observed_decrease_steps=observed_decreases,
        backtracking_failures=backtracking_failures,
    )
