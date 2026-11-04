"""Data types that keep observed simulation data separate from oracle truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ObservedDataView:
    """The only data surface accepted by public splitting and fitting code."""

    features: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    center_codes: np.ndarray
    centers: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        treatment = np.asarray(self.treatment)
        outcome = np.asarray(self.outcome)
        center_codes = np.asarray(self.center_codes)
        centers = np.asarray(self.centers)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        n_samples = features.shape[0]
        if any(values.ndim != 1 for values in (treatment, outcome, center_codes, centers)):
            raise ValueError("treatment, outcome, center_codes, and centers must be one-dimensional")
        if any(len(values) != n_samples for values in (treatment, outcome, center_codes)):
            raise ValueError("observed arrays must have the same number of samples")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "center_codes", center_codes)
        object.__setattr__(self, "centers", centers)

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class SimulationTruth:
    """Oracle-only simulation quantities; never accepted by fitting boundaries."""

    center_truth: np.ndarray
    sample_truth: np.ndarray
    center_to_cluster: Mapping[int, int]
    shared_causal: np.ndarray
    cluster_specific_causal: Mapping[int, np.ndarray]
    causal_by_cluster: Mapping[int, np.ndarray]
    spurious: np.ndarray
    noise: np.ndarray
    ate: float
    ate_by_cluster: np.ndarray
    n_by_cluster: np.ndarray
    ate_weighted: float
    beta_spec_x: Mapping[int, np.ndarray]
    beta_spec_x_effective: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class GeneratedData:
    """The explicit boundary between observable arrays and simulation truth."""

    observed: ObservedDataView
    truth: SimulationTruth


def _require_observed(data: object) -> ObservedDataView:
    if not isinstance(data, ObservedDataView):
        raise TypeError("expected ObservedDataView; simulation truth containers are forbidden")
    return data


def take_observed(data: ObservedDataView, indices: np.ndarray) -> ObservedDataView:
    """Return an observed-only row view without consulting any truth container."""

    observed = _require_observed(data)
    rows = np.asarray(indices)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("indices must be a one-dimensional integer array")
    rows = rows.astype(np.int64, copy=False)
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= observed.n_samples):
        raise IndexError("observed row index out of range")
    center_codes = observed.center_codes[rows]
    return ObservedDataView(
        features=observed.features[rows],
        treatment=observed.treatment[rows],
        outcome=observed.outcome[rows],
        center_codes=center_codes,
        centers=np.unique(center_codes),
    )
