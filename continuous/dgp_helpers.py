"""dgp_helpers.py — elementwise causal functions and block covariance used by the latent-cluster DGP."""
import numpy as np


def _causal_func(X: np.ndarray, name: str) -> np.ndarray:
    """Elementwise causal function f."""
    if name == "tanh":
        return np.tanh(1.5 * X)
    if name == "sigmoid":
        return 2.0 / (1.0 + np.exp(-2.0 * X)) - 1.0
    if name == "relu":
        return np.maximum(X, 0.0) - 0.3989
    return X  # "linear"


def _block_cov(n: int, rho: float) -> np.ndarray:
    """n x n covariance with unit diagonal and off-diagonal rho."""
    S = rho * np.ones((n, n))
    np.fill_diagonal(S, 1.0)
    return S
