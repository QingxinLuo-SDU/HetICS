"""v4_types.py — dataclasses and placeholders shared by the continuous pipeline.

status="ok" marks an available statistic; status="unavailable" marks a statistic that cannot be computed
(empty cluster, fold not runnable, ...) and carries an error_code instead of raising an exception.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SeedKey:
    """Identity of one dataset; v4_seeds.seeds_for_key() derives the integer seeds.

    seed_group_id defines common random numbers: the delta_spec points use the anchor group "A05".
    """

    beta_batch: str
    combo_id: str
    param_index: int
    data_index: int
    seed_group_id: str


@dataclass(frozen=True)
class SeedBundle:
    param_seed: int
    data_seed: int
    fold_seed: int
    algorithm_seed: int


@dataclass(frozen=True)
class ClusterPLRResult:
    """Two-fold whole-center PLR (Robinson) result for one cluster; ci_z and ci_t are both reported."""

    theta: float
    se: float
    variance: float
    ci_z: tuple
    ci_t: tuple
    center_scores: dict          # center -> score
    fold_by_center: dict         # center -> 0/1
    status: str
    error_code: str | None


@dataclass(frozen=True)
class MetaAnalysisResult:
    """REML combination of the cluster-specific estimates; intervals are exploratory (K = 3)."""

    method: str
    mu: float
    var_mu: float
    tau2: float
    weights: np.ndarray
    converged: bool
    boundary_tau2: bool
    objective: float
    intervals_exploratory: dict
    status: str
    error_code: str | None


@dataclass
class RealizationFit:
    """Result of one fit_once()."""

    data: dict
    centers: tuple
    center_labels: dict                      # center -> learned cluster k
    selected_sets: dict                      # (partition_mode, icp_rule, method) -> {k: set(feature)}
    plr_by_analysis: dict                    # (partition_mode, icp_rule, method) -> {k: ClusterPLRResult}
    diagnostics: dict = field(default_factory=dict)


def unavailable_plr(error_code: str,
                    fold_by_center: dict | None = None) -> ClusterPLRResult:
    nan = float("nan")
    return ClusterPLRResult(
        theta=nan, se=nan, variance=nan, ci_z=(nan, nan), ci_t=(nan, nan),
        center_scores={}, fold_by_center=dict(fold_by_center or {}),
        status=STATUS_UNAVAILABLE, error_code=error_code,
    )


def unavailable_meta(method: str, error_code: str, k: int = 0) -> MetaAnalysisResult:
    nan = float("nan")
    return MetaAnalysisResult(
        method=method, mu=nan, var_mu=nan, tau2=nan,
        weights=np.full(k, np.nan) if k else np.empty(0),
        converged=False, boundary_tau2=False, objective=nan,
        intervals_exploratory={}, status=STATUS_UNAVAILABLE, error_code=error_code,
    )
