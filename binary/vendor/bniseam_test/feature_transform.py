"""Training-fitted feature standardization for honest multiview analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureTransform:
    """Columnwise location and scale estimated from training features only."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("mean and scale must be one-dimensional and aligned")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("feature transform contains non-finite values")
        if np.any(scale <= 0.0):
            raise ValueError("feature scales must be positive")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "scale", scale.copy())

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureTransform":
        """Estimate population mean and scale from a training matrix."""

        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must contain only finite values")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the stored training moments without re-estimation."""

        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.size:
            raise ValueError("features must match the fitted feature dimension")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must contain only finite values")
        return (matrix - self.mean) / self.scale
