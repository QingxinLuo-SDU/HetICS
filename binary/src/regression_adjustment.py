"""regression_adjustment.py — regression-standardized ATE (g-computation) and its own joint sandwich.

In each frozen cluster one logistic regression of Y on the treatment X, all covariates selected by the
screening method and the center terms is fitted. For every individual of the cluster the risk is predicted
with X set to 1 and to 0, covariates and center unchanged; the mean difference over the cluster is the
cluster ATE, and the clusters are combined with weights proportional to their sizes.

    p1_i = expit(t_i(1)'b),  p0_i = expit(t_i(0)'b),  d_i = p1_i - p0_i
    theta_k = mean_k(d_i),   ATE = sum_k (n_k/N) theta_k

The coefficient of X is a conditional log-OR and `exp(b_X)` a conditional OR; neither is the risk-difference
ATE, and the SE of b_X is not used as the SE of the ATE.

    u_i = [t_i(Y_i-p_i), d_i-theta_k]
    J   = mean[p_i(1-p_i) t_i t_i']
    h   = mean[p1_i(1-p1_i) t_i(1) - p0_i(1-p0_i) t_i(0)]
    A   = [[J,0],[-h',1]]

Stacking, centering within centers, scaling by the global N and the size-weighted quadratic form follow the
AIPW rules with the RA matrices. The SE status is `candidate_ra_fixed_center_sandwich`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aipw_sandwich import (_check_selected, _expit, _fit_one, _statsmodels_module,
                           aggregate, sandwich, statsmodels_import_error)
from screening_bridge import cluster_members

SE_STATUS_RA = "candidate_ra_fixed_center_sandwich"


def build_ra_design(F_k, env_k, X_k, selected):
    """t = [intercept, E_k - 1 center dummies of the cluster, X, selected columns of F], in a fixed order.

    Returns (T, T1, T0): T is the design at the observed treatment; T1 and T0 set the X column to 1 and 0
    (covariates and center unchanged).
    """
    F_k = np.asarray(F_k, dtype=float)
    env_k = np.asarray(env_k)
    X_k = np.asarray(X_k, dtype=float)
    n_k = F_k.shape[0]
    centers = np.unique(env_k)
    base = [np.ones(n_k)]
    for c in centers[1:]:
        base.append((env_k == c).astype(float))
    tail = [F_k[:, j] for j in selected]
    T = np.column_stack(base + [X_k] + tail)
    x_col = len(base)
    T1 = T.copy()
    T1[:, x_col] = 1.0
    T0 = T.copy()
    T0[:, x_col] = 0.0
    return T, T1, T0, x_col


def ra_equations_and_bread(T, T1, T0, Y, b, theta):
    """Returns (U, A): U is the n x (q+1) matrix of row-wise equations and A = [[J,0],[-h',1]]."""
    T = np.asarray(T, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    T0 = np.asarray(T0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    b = np.asarray(b, dtype=float)
    n, q = T.shape
    if T1.shape != T.shape or T0.shape != T.shape:
        raise ValueError("counterfactual design matrices must have the shape of T")
    if Y.shape != (n,):
        raise ValueError("Y must have as many rows as T")
    if b.shape != (q,):
        raise ValueError(f"coefficients must have the {q} columns of the design matrix")

    p = _expit(T @ b)
    p1 = _expit(T1 @ b)
    p0 = _expit(T0 @ b)
    d = p1 - p0

    U = np.empty((n, q + 1), dtype=float)
    U[:, :q] = T * (Y - p)[:, None]
    U[:, q] = d - float(theta)

    A = np.zeros((q + 1, q + 1), dtype=float)
    A[:q, :q] = (T * (p * (1 - p))[:, None]).T @ T / n
    h = ((T1 * (p1 * (1 - p1))[:, None]).mean(axis=0)
         - (T0 * (p0 * (1 - p0))[:, None]).mean(axis=0))
    A[q, :q] = -h
    A[q, q] = 1.0
    return U, A


def _fit_ra_cluster(F_k, X_k, Y_k, env_k, selected, *, backend="statsmodels_glm"):
    """One logistic regression and the standardized ATE of one cluster; raises RuntimeError on failure."""
    sm = _statsmodels_module()
    if backend == "statsmodels_glm" and sm is None:
        raise RuntimeError("backend statsmodels_glm was requested but statsmodels is unavailable: "
                           f"{statsmodels_import_error()} (no automatic switch to IRLS)")
    if backend == "irls_numpy":
        sm = None
    elif backend != "statsmodels_glm":
        raise ValueError(f"unknown logistic backend: {backend}")

    T, T1, T0, x_col = build_ra_design(F_k, env_k, X_k, selected)
    n_k, q = T.shape
    if not np.any(X_k == 1) or not np.any(X_k == 0):
        raise RuntimeError("one treatment arm is missing in the cluster; the X column is degenerate")
    if n_k <= q:
        raise RuntimeError(f"cluster sample size {n_k} is too small for {q} parameters")
    rank = int(np.linalg.matrix_rank(T))
    if rank < q:
        raise RuntimeError(f"RA design matrix is not of full column rank (rank={rank}<{q})")

    fit, messages = _fit_one("ra_outcome", Y_k, T, sm)
    b = np.asarray(fit.params, dtype=float)
    p = _expit(T @ b)
    p1 = _expit(T1 @ b)
    p0 = _expit(T0 @ b)
    d = p1 - p0
    if not np.all(np.isfinite(d)):
        raise RuntimeError("counterfactual predictions are not finite")
    theta_k = float(np.mean(d))
    U, A = ra_equations_and_bread(T, T1, T0, Y_k, b, theta_k)
    diagnostics = {
        "n_k": int(n_k), "q": int(q), "design_rank": rank,
        "n_treated": int(np.sum(X_k == 1)), "n_control": int(np.sum(X_k == 0)),
        "x_column_index": int(x_col),
        "beta_x_conditional_log_or": float(b[x_col]),
        "p_mean": float(p.mean()), "p1_mean": float(p1.mean()), "p0_mean": float(p0.mean()),
        "d_min": float(d.min()), "d_max": float(d.max()),
        "converged": bool(getattr(fit, "converged", True)),
        "n_iterations": int(getattr(fit, "fit_history", {}).get("iteration", -1)),
        "fit_warnings": messages,
        "clipping_applied": False, "penalty": "none",
        "fitter": "statsmodels_glm" if sm is not None else "irls_numpy",
        "note": "beta_x is a conditional log-OR, not the ATE; the ATE is the standardized mean(p1-p0)",
    }
    # predictions mu_hat at the observed treatment, returned for PSM. They may be reused only with the same
    # design, selection, partition and row order; no SE or covariance is returned.
    outcome_fit = {
        "mu_hat": p,
        "design": "intercept+center_dummies+X+selected_F",
        "backend": diagnostics["fitter"],
        "fit_status": "complete",
        "n_parameters": int(q), "design_rank": int(rank),
        "converged": diagnostics["converged"],
        "fit_warnings": messages,
        "mu_min": float(p.min()), "mu_max": float(p.max()),
        "penalty": "none",
    }
    return U, A, theta_k, diagnostics, outcome_fit


def fit_partition_ra(F, X, Y, env, center_to_cluster, selected_by_cluster,
                     *, backend="statsmodels_glm") -> dict[str, Any]:
    """Regression-standardized ATE and its joint sandwich on a fixed partition and fixed selections."""
    if center_to_cluster is None:
        return {"estimator": "ra_standardized", "status": "abstain",
                "estimand": "full_sample_ate", "cluster_order": [], "n": [],
                "theta": [], "V": None, "weights": None, "estimate": None,
                "se": None, "ci95": None, "point_status": "unavailable",
                "se_status": "unavailable", "reason": "upstream_abstain",
                "errors_by_cluster": {}, "diagnostics": {}}

    F = np.asarray(F, dtype=float)
    X = np.asarray(X).astype(int)
    Y = np.asarray(Y).astype(int)
    env = np.asarray(env)
    n_total, p = F.shape
    if not isinstance(selected_by_cluster, Mapping):
        raise ValueError("selected_by_cluster must map clusters to selections")

    members = cluster_members(env, center_to_cluster)
    cluster_order = sorted(members)
    selected = {k: _check_selected(selected_by_cluster[k], p) for k in cluster_order}

    n_by_cluster: list[int] = []
    theta_by_cluster: list[float] = []
    blocks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    errors: dict[int, str] = {}
    diagnostics: dict[Any, Any] = {}
    outcome_fits: dict[int, dict] = {}

    for k in cluster_order:
        rows = np.flatnonzero(np.isin(env, members[k]))
        n_k = int(rows.size)
        n_by_cluster.append(n_k)
        if n_k == 0:
            errors[k] = "cluster has no observations"
            theta_by_cluster.append(float("nan"))
            continue
        try:
            U_k, A_k, theta_k, diag, fit_k = _fit_ra_cluster(
                F[rows], X[rows], Y[rows], env[rows], selected[k], backend=backend)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            errors[k] = f"{type(error).__name__}: {error}"
            theta_by_cluster.append(float("nan"))
            diagnostics[k] = {"status": "unavailable", "reason": errors[k],
                              "n_k": n_k, "selected": selected[k]}
            continue
        diag.update({"status": "complete", "selected": selected[k], "theta_k": theta_k})
        diagnostics[k] = diag
        fit_k["rows"] = rows
        fit_k["selected"] = list(selected[k])
        outcome_fits[k] = fit_k
        theta_by_cluster.append(theta_k)
        blocks.append((k, rows, U_k, A_k))

    point_ok = not errors and len(blocks) == len(cluster_order)
    result: dict[str, Any] = {
        "estimator": "ra_standardized",
        "estimand": "full_sample_ate",
        "status": "complete" if point_ok else "incomplete",
        "cluster_order": cluster_order, "n": n_by_cluster, "theta": theta_by_cluster,
        "V": None, "weights": None, "estimate": None, "se": None, "ci95": None,
        "point_status": "complete" if point_ok else "unavailable",
        "se_status": "unavailable",
        "errors_by_cluster": errors, "diagnostics": diagnostics,
        # for PSM within the same unit only; callers remove it (it holds arrays) before writing
        "outcome_fits": outcome_fits,
    }
    if not point_ok:
        result["reason"] = "cluster_effect_unavailable"
        return result

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
        result["se_reason"] = f"{type(error).__name__}: {error}"
        result["weights"] = (np.asarray(n_by_cluster, float) / N).tolist()
        result["estimate"] = float(np.asarray(n_by_cluster, float)
                                   @ np.asarray(theta_by_cluster, float) / N)
        return result

    result.update({
        "V": V_theta, "weights": summary["weights"].tolist(),
        "estimate": summary["estimate"], "se": summary["se"],
        "ci95": list(summary["ci95"]), "se_status": SE_STATUS_RA,
    })
    if summary["se"] == 0.0:
        result["se_status"] = "unavailable"
        result["se_reason"] = "degenerate_zero_variance"
        result["ci95"] = None
    result["diagnostics"]["joint"] = {
        "parameter_count": total, "theta_positions": theta_positions,
        "n_total": N, "meat": "within_center_demeaned",
        "scaling": "mean_equations_divided_by_N_once",
        "note": "joint matrices of RA itself, not the AIPW SE",
    }
    return result


__all__ = ["SE_STATUS_RA", "build_ra_design", "ra_equations_and_bread",
           "fit_partition_ra"]
