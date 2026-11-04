"""aipw_sandwich.py — two-arm AIPW estimates, the joint estimating equations, the full sandwich and size weighting.

Parameter order (alpha, beta0, beta1, theta):
    e  = expit(r'alpha)      propensity model
    m0 = expit(r'beta0)      outcome model of the X=0 arm
    m1 = expit(r'beta1)      outcome model of the X=1 arm
    phi = m1-m0 + X/e*(Y-m1) - (1-X)/(1-e)*(Y-m0)

Row-wise estimating equations:
    u_i = [ r*(X-e), (1-X)*r*(Y-m0), X*r*(Y-m1), phi-theta ]

A = -mean(du/dlambda); the meat is centered within centers; V = A^{-1} B A^{-T} / N.

A successful SE is labelled `candidate_fixed_center_sandwich`.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm

SE_STATUS_CANDIDATE = "candidate_fixed_center_sandwich"
_Z975 = float(norm.ppf(0.975))


# ── row scores ────────────────────────────────────────────────────────────────
def aipw_score(X, Y, e, m0, m1) -> np.ndarray:
    """AIPW score vector phi of length n."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    e = np.asarray(e, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    if not (X.shape == Y.shape == e.shape == m0.shape == m1.shape):
        raise ValueError("inputs of aipw_score must have the same shape")
    if X.ndim != 1:
        raise ValueError("aipw_score accepts one-dimensional inputs only")
    if np.any(e <= 0.0) or np.any(e >= 1.0):
        raise ValueError("propensity scores must lie in (0, 1)")
    return m1 - m0 + X / e * (Y - m1) - (1 - X) / (1 - e) * (Y - m0)


def _expit(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(np.clip(z, -700, None)) / (1.0 + np.exp(np.clip(z, -700, None))))


# ── joint estimating equations and analytic bread ─────────────────────────────
def equations_and_bread(R, X, Y, alpha, beta0, beta1, theta):
    """Returns (U, A).

    U is the n x (3q+1) matrix of row-wise estimating equations; A is (3q+1) x (3q+1) with
    A = -mean(du/dlambda), columns in the parameter order (alpha, beta0, beta1, theta).
    """
    R = np.asarray(R, dtype=float)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta0 = np.asarray(beta0, dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    theta = float(theta)
    if R.ndim != 2:
        raise ValueError("R must be a two-dimensional design matrix")
    n, q = R.shape
    if X.shape != (n,) or Y.shape != (n,):
        raise ValueError("X and Y must have as many rows as R")
    for name, vec in (("alpha", alpha), ("beta0", beta0), ("beta1", beta1)):
        if vec.shape != (q,):
            raise ValueError(f"{name} must have the {q} columns of the design matrix")
    if not np.all(np.isfinite(R)) or not np.isfinite(theta):
        raise ValueError("R and theta must be finite")

    e = _expit(R @ alpha)
    m0 = _expit(R @ beta0)
    m1 = _expit(R @ beta1)
    if np.any(e <= 0.0) or np.any(e >= 1.0):
        raise ValueError("propensity scores must lie in (0, 1)")

    phi = m1 - m0 + X / e * (Y - m1) - (1 - X) / (1 - e) * (Y - m0)

    U = np.empty((n, 3 * q + 1), dtype=float)
    U[:, 0:q] = R * (X - e)[:, None]
    U[:, q:2 * q] = R * ((1 - X) * (Y - m0))[:, None]
    U[:, 2 * q:3 * q] = R * (X * (Y - m1))[:, None]
    U[:, 3 * q] = phi - theta

    # own blocks of the three logistic equations: A_gg = mean(weight * r r')
    A = np.zeros((3 * q + 1, 3 * q + 1), dtype=float)
    A[0:q, 0:q] = (R * (e * (1 - e))[:, None]).T @ R / n
    A[q:2 * q, q:2 * q] = (R * ((1 - X) * m0 * (1 - m0))[:, None]).T @ R / n
    A[2 * q:3 * q, 2 * q:3 * q] = (R * (X * m1 * (1 - m1))[:, None]).T @ R / n

    # ATE row: minus the mean gradient of phi (all cross terms kept); last element 1
    g_alpha = -(R * (X * (Y - m1) * (1 - e) / e
                     + (1 - X) * (Y - m0) * e / (1 - e))[:, None])
    g_beta0 = R * (m0 * (1 - m0) * (-1.0 + (1 - X) / (1 - e)))[:, None]
    g_beta1 = R * (m1 * (1 - m1) * (1.0 - X / e))[:, None]
    A[3 * q, 0:q] = -g_alpha.mean(axis=0)
    A[3 * q, q:2 * q] = -g_beta0.mean(axis=0)
    A[3 * q, 2 * q:3 * q] = -g_beta1.mean(axis=0)
    A[3 * q, 3 * q] = 1.0
    return U, A


# ── meat centered within centers and the full sandwich ────────────────────────
def sandwich(U, A, env) -> np.ndarray:
    """V = A^{-1} B A^{-T} / N, with B the empirical second moment centered within centers."""
    U = np.asarray(U, dtype=float)
    A = np.asarray(A, dtype=float)
    env = np.asarray(env)
    if U.ndim != 2:
        raise ValueError("U must be a two-dimensional array")
    n, p = U.shape
    if A.shape != (p, p):
        raise ValueError(f"shape of A {A.shape} does not match the {p} columns of U")
    if env.shape != (n,):
        raise ValueError("env must have as many rows as U")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(A)):
        raise ValueError("U and A must be finite")

    centered = U.copy()
    for c in np.unique(env):
        mask = env == c
        centered[mask] -= centered[mask].mean(axis=0)
    B = centered.T @ centered / n

    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > 1.0 / np.finfo(float).eps:
        raise np.linalg.LinAlgError(f"bread matrix A is nearly singular (cond={cond:.3e}); SE not available")
    left = np.linalg.solve(A, B)
    V = np.linalg.solve(A, left.T).T / n
    if not np.all(np.isfinite(V)):
        raise ValueError("sandwich covariance has non-finite entries")
    asym = float(np.max(np.abs(V - V.T)))
    scale = max(float(np.max(np.abs(V))), 1e-300)
    if asym > 1e-8 * scale:
        raise ValueError(f"sandwich covariance is clearly asymmetric ({asym:.3e})")
    return (V + V.T) / 2


# ── size-weighted summary ─────────────────────────────────────────────────────
def aggregate(n, theta, V) -> dict[str, Any]:
    """a = n/sum(n); estimate a@theta; variance a@V@a; normal-quantile interval."""
    n = np.asarray(n)
    theta = np.asarray(theta, dtype=float)
    V = np.asarray(V, dtype=float)
    if n.ndim != 1 or theta.ndim != 1 or V.ndim != 2:
        raise ValueError("n and theta must be one-dimensional and V two-dimensional")
    k = n.shape[0]
    if theta.shape != (k,) or V.shape != (k, k):
        raise ValueError("n, theta and V must have the same number of clusters")
    if np.issubdtype(n.dtype, np.floating):
        if not np.all(np.isfinite(n)) or not np.all(n == np.round(n)):
            raise ValueError("n must be positive integers")
        n = np.round(n).astype(np.int64)
    if not np.issubdtype(n.dtype, np.integer) or np.any(n <= 0):
        raise ValueError("n must be positive integers")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(V)):
        raise ValueError("theta and V must be finite")
    asym = float(np.max(np.abs(V - V.T)))
    scale = max(float(np.max(np.abs(V))), 1e-300)
    if asym > 1e-8 * scale:
        raise ValueError("V must be symmetric")
    eig = np.linalg.eigvalsh((V + V.T) / 2)
    if float(eig.min()) < -1e-8 * scale:
        raise ValueError(f"V is clearly not positive semidefinite (smallest eigenvalue {float(eig.min()):.3e})")

    a = n.astype(float) / float(n.sum())
    estimate = float(a @ theta)
    variance = float(a @ V @ a)
    if variance < 0.0:
        if variance < -1e-12 * scale:
            raise ValueError("negative weighted variance")
        variance = 0.0
    se = float(np.sqrt(variance))
    return {"weights": a, "estimate": estimate, "se": se,
            "ci95": (estimate - _Z975 * se, estimate + _Z975 * se)}


# ── design matrix and per-cluster fits ────────────────────────────────────────
def build_design(F: np.ndarray, env_k: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    """R = [intercept, E_k - 1 center dummies of the cluster, selected columns of F], in a fixed order."""
    n_k = F.shape[0]
    centers = np.unique(env_k)
    cols = [np.ones(n_k)]
    for c in centers[1:]:
        cols.append((env_k == c).astype(float))
    for j in selected:
        cols.append(F[:, j].astype(float))
    return np.column_stack(cols)


def _check_selected(selected, p: int) -> list[int]:
    out = [int(j) for j in selected]
    if len(set(out)) != len(out):
        raise ValueError(f"selection has duplicate columns: {selected}")
    for j in out:
        if j < 0 or j >= p:
            raise ValueError(f"selected column {j} out of range (p={p})")
    return sorted(out)


def _irls_logistic(y: np.ndarray, R: np.ndarray, *, max_iter: int = 100,
                   tol: float = 1e-10):
    """IRLS (Newton-Raphson) for the unpenalized logistic MLE.

    The same estimator as statsmodels `GLM(..., family=Binomial()).fit()` (unpenalized maximum likelihood
    by IRLS); used only when statsmodels is unavailable and marked `fitter="irls_numpy"` in the diagnostics.

    No penalty, clipping or fallback; singularity or divergence raises, and the caller marks the cluster as
    not estimable.
    """
    y = np.asarray(y, dtype=float)
    R = np.asarray(R, dtype=float)
    n, q = R.shape
    beta = np.zeros(q, dtype=float)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = R @ beta
        mu = _expit(eta)
        w = mu * (1.0 - mu)
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(w)):
            raise RuntimeError("IRLS produced non-finite means or weights")
        gradient = R.T @ (y - mu)
        hessian = (R * w[:, None]).T @ R
        cond = np.linalg.cond(hessian)
        if not np.isfinite(cond) or cond > 1.0 / np.finfo(float).eps:
            raise RuntimeError(f"IRLS information matrix is nearly singular (cond={cond:.3e}), "
                               "usually complete separation or collinearity")
        step = np.linalg.solve(hessian, gradient)
        if not np.all(np.isfinite(step)):
            raise RuntimeError("IRLS step is not finite")
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            converged = True
            break
    if not np.all(np.isfinite(beta)):
        raise RuntimeError("IRLS coefficients are not finite")
    return beta, converged, n_iter


class _IRLSFit:
    """Minimal statsmodels-like interface for the IRLS result."""

    def __init__(self, params, converged, n_iter):
        self.params = params
        self.converged = converged
        self.fit_history = {"iteration": n_iter}

    def predict(self, R):
        return _expit(np.asarray(R, dtype=float) @ self.params)


SEPARATION_MARKERS = ("perfectseparation", "perfect separation", "separation detected")
_IMPORT_ERROR: str | None = None


def _statsmodels_module():
    """statsmodels.api, or None when it cannot be imported; the reason is kept in
    `statsmodels_import_error()` (a missing package and a broken installation are reported differently)."""
    global _IMPORT_ERROR
    try:
        import statsmodels.api as sm
    except Exception as error:                          # noqa: BLE001 any import failure means unavailable
        _IMPORT_ERROR = f"{type(error).__name__}: {error}"
        return None
    _IMPORT_ERROR = None
    return sm


def statsmodels_import_error() -> str | None:
    """Reason of the last failed import in `_statsmodels_module()`; None after a success."""
    return _IMPORT_ERROR


def is_separation_warning(text: str) -> bool:
    """Whether a warning or exception text indicates complete (quasi-complete) separation."""
    low = str(text).lower()
    return any(marker in low for marker in SEPARATION_MARKERS)


def _fit_one(name, y, R, sm):
    """Fit one unpenalized logistic model, capturing its warnings and detecting separation.

    Under separation the unpenalized MLE does not exist; the function raises and the caller marks the
    cluster as not estimable (no penalty, clipping or change of model).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")        # vendored modules may ignore warnings globally
        try:
            if sm is not None:
                fit = sm.GLM(np.asarray(y, dtype=float), R,
                             family=sm.families.Binomial()).fit()
            else:
                fit = _IRLSFit(*_irls_logistic(np.asarray(y, dtype=float), R))
        except Exception as error:                      # noqa: BLE001
            text = f"{type(error).__name__}: {error}"
            if is_separation_warning(text):
                raise RuntimeError(f"{name} model shows complete separation ({text}); "
                                   "the unpenalized MLE does not exist") from error
            raise RuntimeError(f"{name} logistic fit failed: {text}") from error
    messages = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    separated = [m for m in messages if is_separation_warning(m)]
    if separated:
        raise RuntimeError(f"{name} model shows complete separation ({separated[0]}); "
                           "the unpenalized MLE does not exist")
    if not bool(getattr(fit, "converged", True)):
        raise RuntimeError(f"{name} model did not converge")
    if not np.all(np.isfinite(np.asarray(fit.params, dtype=float))):
        raise RuntimeError(f"{name} model has non-finite coefficients")
    return fit, messages


def _fit_cluster(F_k, X_k, Y_k, env_k, selected, *, backend="statsmodels_glm"):
    """Fit the three logistic models; returns (R, e, m0, m1, params, diagnostics) or raises RuntimeError.

    The three design matrices R, R[X=0] and R[X=1] are checked for full column rank separately; checking R
    alone would let unidentified coefficients of one arm enter the counterfactual predictions.
    """
    sm = _statsmodels_module()
    if backend == "statsmodels_glm" and sm is None:
        raise RuntimeError("backend statsmodels_glm was requested but statsmodels is unavailable: "
                           f"{statsmodels_import_error()} (no automatic switch to IRLS)")
    if backend == "irls_numpy":
        sm = None
    elif backend != "statsmodels_glm":
        raise ValueError(f"unknown logistic backend: {backend}")

    R = build_design(F_k, env_k, selected)
    n_k, q = R.shape
    if not np.any(X_k == 1) or not np.any(X_k == 0):
        raise RuntimeError("one treatment arm is missing in the cluster; effect not estimable")
    if n_k <= q:
        raise RuntimeError(f"cluster sample size {n_k} is too small for {q} parameters")
    ranks = {}
    for name, sub in (("full", R), ("arm_x0", R[X_k == 0]), ("arm_x1", R[X_k == 1])):
        rank = int(np.linalg.matrix_rank(sub))
        ranks[name] = rank
        if sub.shape[0] <= q:
            raise RuntimeError(f"{name} design matrix has {sub.shape[0]} rows, too few for {q} parameters")
        if rank < q:
            raise RuntimeError(f"{name} design matrix is not of full column rank (rank={rank}<{q})")

    fitter = "statsmodels_glm" if sm is not None else "irls_numpy"
    fit_e, w_e = _fit_one("propensity", X_k, R, sm)
    fit_0, w_0 = _fit_one("outcome_x0", Y_k[X_k == 0], R[X_k == 0], sm)
    fit_1, w_1 = _fit_one("outcome_x1", Y_k[X_k == 1], R[X_k == 1], sm)
    warn_by_model = {"propensity": w_e, "outcome_x0": w_0, "outcome_x1": w_1}
    warn_messages = sorted(set(w_e) | set(w_0) | set(w_1))

    e = np.asarray(fit_e.predict(R), dtype=float)
    m0 = np.asarray(fit_0.predict(R), dtype=float)
    m1 = np.asarray(fit_1.predict(R), dtype=float)
    for name, pred in (("e", e), ("m0", m0), ("m1", m1)):
        if not np.all(np.isfinite(pred)):
            raise RuntimeError(f"{name} predictions are not finite")
    if np.any(e <= 0.0) or np.any(e >= 1.0):
        raise RuntimeError("propensity predictions reach 0 or 1 numerically (no default clipping)")

    params = (np.asarray(fit_e.params, dtype=float),
              np.asarray(fit_0.params, dtype=float),
              np.asarray(fit_1.params, dtype=float))
    diagnostics = {
        "n_k": int(n_k), "q": int(q),
        "n_treated": int(np.sum(X_k == 1)), "n_control": int(np.sum(X_k == 0)),
        "design_rank": ranks["full"],
        "design_rank_by_arm": ranks,
        "e_min": float(e.min()), "e_max": float(e.max()),
        "m0_min": float(m0.min()), "m0_max": float(m0.max()),
        "m1_min": float(m1.min()), "m1_max": float(m1.max()),
        "converged": {"propensity": bool(fit_e.converged),
                      "outcome_x0": bool(fit_0.converged),
                      "outcome_x1": bool(fit_1.converged)},
        "n_iterations": {"propensity": int(getattr(fit_e, "fit_history", {}).get("iteration", -1)),
                         "outcome_x0": int(getattr(fit_0, "fit_history", {}).get("iteration", -1)),
                         "outcome_x1": int(getattr(fit_1, "fit_history", {}).get("iteration", -1))},
        "fit_warnings": warn_messages,
        "fit_warnings_by_model": warn_by_model,
        "clipping_applied": False, "penalty": "none",
        "fitter": fitter,
    }
    diagnostics.update(_propensity_diagnostics(X_k, Y_k, e))
    return R, e, m0, m1, params, diagnostics


def _propensity_diagnostics(X_k, Y_k, e) -> dict:
    """Propensity-score quantiles and tails, and weights and ESS of the observed arms.

    The observed weights w1 = X/e and w0 = (1-X)/(1-e) are used, not the bound 1/min(e).
    """
    X_k = np.asarray(X_k).astype(int)
    Y_k = np.asarray(Y_k).astype(int)
    e = np.asarray(e, dtype=float)
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    out: dict[str, Any] = {}
    for arm, mask in (("treated", X_k == 1), ("control", X_k == 0)):
        sub = e[mask]
        out[f"ps_{arm}"] = {
            "n": int(mask.sum()),
            "min": (float(sub.min()) if sub.size else None),
            "max": (float(sub.max()) if sub.size else None),
            "quantiles": ({str(q): float(np.quantile(sub, q)) for q in qs}
                          if sub.size else None),
            "frac_lt_0.01": (float(np.mean(sub < 0.01)) if sub.size else None),
            "frac_gt_0.99": (float(np.mean(sub > 0.99)) if sub.size else None),
            "frac_lt_0.05": (float(np.mean(sub < 0.05)) if sub.size else None),
            "frac_gt_0.95": (float(np.mean(sub > 0.95)) if sub.size else None),
            "n_events": int(Y_k[mask].sum()),
            "n_nonevents": int(mask.sum() - Y_k[mask].sum()),
        }
    w1 = np.where(X_k == 1, 1.0 / e, 0.0)
    w0 = np.where(X_k == 0, 1.0 / (1.0 - e), 0.0)
    for name, w, mask in (("w1", w1, X_k == 1), ("w0", w0, X_k == 0)):
        obs = w[mask]
        ess = (float(obs.sum() ** 2 / np.sum(obs ** 2))
               if obs.size and np.sum(obs ** 2) > 0 else None)
        out[f"aipw_{name}"] = {
            "max": (float(obs.max()) if obs.size else None),
            "p95": (float(np.quantile(obs, 0.95)) if obs.size else None),
            "p99": (float(np.quantile(obs, 0.99)) if obs.size else None),
            "ess": ess,
            "ess_ratio": (ess / obs.size if ess is not None and obs.size else None),
        }
    out["ps_note"] = ("weights of the observed treatment arm, not 1/min(e); "
                      "p_x_given_latent of the generator is not the true PS given the observed F")
    return out


def fit_partition_aipw(F, X, Y, env, center_to_cluster, selected_by_cluster,
                       *, backend="statsmodels_glm") -> dict[str, Any]:
    """Two-arm AIPW and joint sandwich on a fixed partition and fixed selections."""
    from screening_bridge import cluster_members

    if center_to_cluster is None:
        return {"status": "abstain", "estimand": "full_sample_ate",
                "cluster_order": [], "n": [], "theta": [],
                "V": None, "weights": None, "estimate": None, "se": None,
                "ci95": None, "point_status": "unavailable",
                "se_status": "unavailable", "reason": "upstream_abstain",
                "errors_by_cluster": {}, "diagnostics": {}}

    F = np.asarray(F)
    X = np.asarray(X)
    Y = np.asarray(Y)
    env = np.asarray(env)
    if F.ndim != 2:
        raise ValueError("F must be a two-dimensional array")
    n_total, p = F.shape
    for name, arr in (("X", X), ("Y", Y), ("env", env)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if arr.shape[0] != n_total:
            raise ValueError(f"{name} has {arr.shape[0]} rows, F has {n_total}")
    if not np.all(np.isfinite(np.asarray(F, dtype=float))):
        raise ValueError("F has non-finite values")
    for name, arr in (("X", X), ("Y", Y)):
        if not np.all(np.isin(np.asarray(arr), (0, 1))):
            raise ValueError(f"{name} must take values 0/1 only")
    F = F.astype(float)
    X = X.astype(int)
    Y = Y.astype(int)

    members = cluster_members(env, center_to_cluster)
    cluster_order = sorted(members)
    if not isinstance(selected_by_cluster, Mapping):
        raise ValueError("selected_by_cluster must map clusters to selections")
    selected = {}
    for k in cluster_order:
        if k not in selected_by_cluster:
            raise ValueError(f"selected_by_cluster lacks cluster {k}")
        selected[k] = _check_selected(selected_by_cluster[k], p)

    n_by_cluster: list[int] = []
    theta_by_cluster: list[float] = []
    blocks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []   # (k, rows, U_k, A_k)
    errors: dict[int, str] = {}
    diagnostics: dict[int, Any] = {}

    for k in cluster_order:
        rows = np.flatnonzero(np.isin(env, members[k]))
        n_k = int(rows.size)
        n_by_cluster.append(n_k)
        if n_k == 0:
            errors[k] = "cluster has no observations"
            theta_by_cluster.append(float("nan"))
            continue
        try:
            R, e, m0, m1, params, diag = _fit_cluster(
                F[rows], X[rows], Y[rows], env[rows], selected[k], backend=backend)
            phi = aipw_score(X[rows].astype(float), Y[rows].astype(float), e, m0, m1)
            theta_k = float(np.mean(phi))
            U_k, A_k = equations_and_bread(R, X[rows].astype(float), Y[rows].astype(float),
                                           params[0], params[1], params[2], theta_k)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            errors[k] = f"{type(error).__name__}: {error}"
            theta_by_cluster.append(float("nan"))
            diagnostics[k] = {"status": "unavailable", "reason": errors[k],
                              "n_k": n_k, "selected": selected[k]}
            continue
        diag.update({"status": "complete", "selected": selected[k],
                     "theta_k": theta_k,
                     "phi_min": float(phi.min()), "phi_max": float(phi.max())})
        diagnostics[k] = diag
        theta_by_cluster.append(theta_k)
        blocks.append((k, rows, U_k, A_k))

    point_ok = not errors and len(blocks) == len(cluster_order)
    result: dict[str, Any] = {
        "status": "complete" if point_ok else "incomplete",
        "estimand": "full_sample_ate",
        "cluster_order": cluster_order,
        "n": n_by_cluster,
        "theta": theta_by_cluster,
        "V": None, "weights": None, "estimate": None, "se": None, "ci95": None,
        "point_status": "complete" if point_ok else "unavailable",
        "se_status": "unavailable",
        "errors_by_cluster": errors,
        "diagnostics": diagnostics,
    }
    if not point_ok:
        result["reason"] = "cluster_effect_unavailable"
        return result

    # ── global stacking: U is zero for other clusters; local mean A_k enters the global block times n_k/N ──
    widths = [U_k.shape[1] for _, _, U_k, _ in blocks]
    total = int(sum(widths))
    N = int(n_total)
    U_global = np.zeros((N, total), dtype=float)
    A_global = np.zeros((total, total), dtype=float)
    theta_positions: list[int] = []
    offset = 0
    for (k, rows, U_k, A_k), width in zip(blocks, widths):
        U_global[np.ix_(rows, np.arange(offset, offset + width))] = U_k
        A_global[offset:offset + width, offset:offset + width] = A_k * (rows.size / N)
        theta_positions.append(offset + width - 1)
        offset += width

    try:
        V_lambda = sandwich(U_global, A_global, env)
        V_theta = V_lambda[np.ix_(theta_positions, theta_positions)]
        summary = aggregate(np.asarray(n_by_cluster, dtype=np.int64),
                            np.asarray(theta_by_cluster, dtype=float), V_theta)
    except (np.linalg.LinAlgError, ValueError) as error:
        result["se_status"] = "unavailable"
        result["se_reason"] = f"{type(error).__name__}: {error}"
        result["weights"] = (np.asarray(n_by_cluster, float) / N).tolist()
        result["estimate"] = float(np.asarray(n_by_cluster, float) @
                                   np.asarray(theta_by_cluster, float) / N)
        return result

    result.update({
        "V": V_theta,
        "weights": summary["weights"].tolist(),
        "estimate": summary["estimate"],
        "se": summary["se"],
        "ci95": list(summary["ci95"]),
        "se_status": SE_STATUS_CANDIDATE,
    })
    if summary["se"] == 0.0:
        result["se_status"] = "unavailable"
        result["se_reason"] = "degenerate_zero_variance"
        result["ci95"] = None
    result["diagnostics"]["joint"] = {
        "parameter_count": total,
        "theta_positions": theta_positions,
        "n_total": N,
        "meat": "within_center_demeaned",
        "scaling": "mean_equations_divided_by_N_once",
    }
    return result


__all__ = ["aipw_score", "equations_and_bread", "sandwich", "aggregate",
           "build_design", "fit_partition_aipw", "SE_STATUS_CANDIDATE",
           "is_separation_warning", "statsmodels_import_error"]
