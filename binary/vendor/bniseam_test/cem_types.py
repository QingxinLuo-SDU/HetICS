"""Configuration and machine-readable results for Bernoulli-sEAM CEM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .feature_transform import FeatureTransform
from .model import Parameters


ELIGIBLE_STATUSES = frozenset(
    {"converged", "max_outer_iter_eligible", "max_iter_eligible"}
)
HARD_FAILURE_STATUSES = frozenset(
    {
        "no_feasible_start",
        "assignment_infeasible",
        "nonfinite_objective",
        "m_step_backtracking_failed",
        "monotonicity_violation",
        "no_eligible_candidate",
        "full_refit_failed",
        "terminal_refit_failed",
        "terminal_refit_max_iter_uncertified",
        "terminal_assignment_failed",
        "invalid_input",
    }
)


@dataclass(frozen=True)
class CEMConfig:
    k_components: int
    pi_cem: float = 0.10
    split_fraction: float = 0.50
    max_outer_iter: int = 100
    objective_tolerance: float = 1e-7
    parameter_tolerance: float = 1e-5
    assignment_tolerance: float = 1e-10
    n_random_starts: int = 0

    def __post_init__(self) -> None:
        if self.k_components < 1:
            raise ValueError("k_components must be positive")
        if not 0.0 < self.pi_cem < 1.0 / self.k_components:
            raise ValueError("pi_cem must lie in (0, 1 / K)")
        if not 0.0 < self.split_fraction < 1.0:
            raise ValueError("split_fraction must lie in (0, 1)")
        if self.max_outer_iter < 1:
            raise ValueError("iteration budget must be positive")
        if isinstance(self.n_random_starts, (bool, np.bool_)) or not isinstance(
            self.n_random_starts,
            (int, np.integer),
        ):
            raise TypeError("n_random_starts must be an integer")
        if self.n_random_starts != 0:
            raise ValueError(
                "random starts are disabled in the frozen fixed-candidate CEM version"
            )
        if min(
            self.objective_tolerance,
            self.parameter_tolerance,
            self.assignment_tolerance,
        ) < 0.0:
            raise ValueError("tolerances must be nonnegative")

    def minimum_cluster_size_formula(self, n_centers: int) -> int:
        """Return the theoretical lower bound before feasibility checking."""

        return max(2, int(np.ceil(n_centers * self.pi_cem)))

    def minimum_cluster_size(self, n_centers: int) -> int:
        m_e = self.minimum_cluster_size_formula(n_centers)
        if self.k_components * m_e > n_centers:
            raise ValueError("K * m_E exceeds the number of centers")
        return m_e


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    digest: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CandidateSeed:
    candidate_id: int
    method: str
    labels: np.ndarray
    seed: int | None


@dataclass
class CEMCandidateResult:
    candidate_id: int
    method: str
    parameters: Parameters | None
    labels: np.ndarray | None
    status: str
    eligible: bool
    converged: bool
    training_objective: float | None
    validation_objective: float | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CEMRunResult:
    parameters: Parameters | None
    labels: np.ndarray | None
    selected_candidate_id: int | None
    status: str
    candidates: list[CEMCandidateResult]
    split: SplitIndices
    full_refit_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass
class RefinementResult:
    """Truth-free result of one fixed-start candidate refinement."""

    parameters: Parameters | None
    pre_labels: np.ndarray
    post_labels: np.ndarray
    training_objective: float | None
    status: str
    eligible: bool
    converged: bool
    trace: list[dict[str, Any]]
    minimum_cluster_size: int | None
    failure_reason: str | None
    feature_transform: FeatureTransform
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> np.ndarray:
        """Compatibility alias for the post-refinement center labels."""

        return self.post_labels
