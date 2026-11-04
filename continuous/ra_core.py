"""ra_core.py — one continuous replicate: HetICS selection, cluster-specific effects and the comparators.

fit_once() learns the center clusters and the selected sets; this module then estimates the effect on the
same data with four estimators:
  robinson       whole-center PLR (two-fold whole-center cross-fitting, center-clustered sandwich, t_{M-1})
  ra             OLS Y ~ 1 + center indicators + X + W, coefficient of X; center-clustered CR1 SE, t_{M-1}
                 (main estimator)
  gps_ipw        normal linear model for X given (center, W), stabilized weights f(X) / f(X | W),
                 weighted Y ~ 1 + X; CR1 SE and t_{M-1}
  gps_ipw_trunc  as gps_ipw with weights truncated at the 1st and 99th percentiles
Arms estimated within each learned cluster:
  dual_icp (HetICS), xside_icp and yside_icp (single-side variants), oracle_min (target set S_k^*),
  dual_no_icp (dual-side candidates without invariance selection) and dual_global_icp (dual-side selection
  and invariance selection over all centers, estimation within learned clusters).
Arms estimated once on all centers:
  pooled_lasso_fp, pooled_pds_fp, pooled_oal_fp and hdps_fp (comparators applied as originally proposed) and
  oracle_safe_pooled (safe adjustment set S_safe).
The overall effect mu combines the cluster-specific estimates by REML with the diagonal matrix of their
squared standard errors; mu is unavailable when any cluster estimate is unavailable. In this DGP
theta_k = mu = gamma.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats

import step6x_config as cfg
import step6x_config_v4 as cfg4
from pooled_selection import select_pooled
from random_effects_gls import fit_reml_gls
from step6x_core_v4 import _PLRCache, _V4Cache, fit_method_cluster_plrs, fit_once
from v4_seeds import seeds_for_key
from whole_center_plr import Z_CRIT, fit_whole_center_plr

RA_VERSION = "cont-ra-v2"
MODE, RULE = cfg4.MAIN_PARTITION_MODE, cfg4.MAIN_ICP_RULE
ESTIMATORS = ("robinson", "ra", "gps_ipw", "gps_ipw_trunc")
CLUSTER_ARMS = tuple(cfg.METHODS) + ("dual_no_icp", "dual_global_icp")
POOLED_ARMS = ("pooled_lasso_fp", "pooled_pds_fp", "pooled_oal_fp", "hdps_fp", "oracle_safe_pooled")
TRUNC_Q = (0.01, 0.99)


def _peak_rss_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20   # Linux: KB
    except ImportError:
        return float("nan")


def _na(code):
    return dict(status="unavailable", error_code=code)


def _cr1_se(D, resid, bread, j, env, w=None):
    """Center-clustered CR1: bread (sum_c s_c s_c') bread x M/(M-1) x (n-1)/(n-p)."""
    n, p = D.shape
    s = D * (resid if w is None else resid * w)[:, None]
    cs = np.array([s[env == c].sum(axis=0) for c in np.unique(env)])
    m = cs.shape[0]
    v = (bread @ (cs.T @ cs) @ bread)[j, j] * m / (m - 1) * (n - 1) / (n - p)
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan")


def _block(theta, se, t_crit):
    if not (np.isfinite(theta) and np.isfinite(se)):
        return _na("theta_nonfinite")
    return dict(status="ok", theta=float(theta), se=float(se),
                ci_t=[float(theta - t_crit * se), float(theta + t_crit * se)],
                ci_z=[float(theta - Z_CRIT * se), float(theta + Z_CRIT * se)])


def _wls_slope(X, Y, w, env, t_crit):
    M = np.column_stack([np.ones(X.size), X])
    bread = np.linalg.inv((M * w[:, None]).T @ M)
    b = bread @ ((M * w[:, None]).T @ Y)
    return _block(b[1], _cr1_se(M, Y - M @ b, bread, 1, env, w=w), t_crit)


def extra_estimators(X, Y, W, env, gps: bool = True) -> dict:
    """RA and the two GPS-IPW estimators (gps=False: RA only); unavailable with fewer than two centers or a
    singular design."""
    m = np.unique(env).size
    if m < 2:
        return {e: _na("fold_not_runnable") for e in (ESTIMATORS[1:] if gps else ("ra",))}
    t_crit = float(stats.t.ppf(0.975, m - 1))
    dummies = (env[:, None] == np.unique(env)[None, 1:]).astype(float)
    base = np.column_stack([np.ones(X.size), dummies, W])
    out = {}
    try:
        D = np.column_stack([base[:, :1 + dummies.shape[1]], X, W])
        j = 1 + dummies.shape[1]
        bread = np.linalg.inv(D.T @ D)
        beta = bread @ (D.T @ Y)
        out["ra"] = _block(beta[j], _cr1_se(D, Y - D @ beta, bread, j, env), t_crit)
    except np.linalg.LinAlgError:
        out["ra"] = _na("design_singular")
    if not gps:
        return out
    try:
        g = np.linalg.solve(base.T @ base, base.T @ X)
        rx = X - base @ g
        s_c = np.sqrt(rx @ rx / (X.size - base.shape[1]))
        s_m = X.std(ddof=1)
        w = np.exp(-0.5 * ((X - X.mean()) / s_m) ** 2 + 0.5 * (rx / s_c) ** 2) * s_c / s_m
        out["gps_ipw"] = _wls_slope(X, Y, w / w.mean(), env, t_crit)
        lo, hi = np.quantile(w, TRUNC_Q)
        wt = np.clip(w, lo, hi)
        out["gps_ipw_trunc"] = _wls_slope(X, Y, wt / wt.mean(), env, t_crit)
        out["gps_ipw"]["max_weight_share"] = float(w.max() / w.sum())
        out["gps_ipw"]["r2_x_given_w"] = float(1 - rx.var() / X.var())
    except np.linalg.LinAlgError:
        out["gps_ipw"] = out["gps_ipw_trunc"] = _na("design_singular")
    return out


def _from_plr(res) -> dict:
    if res.status != "ok":
        return _na(res.error_code or "unavailable")
    return dict(status="ok", theta=float(res.theta), se=float(res.se), ci_t=[float(x) for x in res.ci_t],
                ci_z=[float(x) for x in res.ci_z])


def _score(block, gamma):
    if block.get("status") == "ok":
        for s in ("t", "z"):
            lo, hi = block[f"ci_{s}"]
            block[f"covered_{s}"] = bool(lo <= gamma <= hi)
            block[f"reject0_{s}"] = not (lo <= 0.0 <= hi)
    return block


def aggregate(blocks) -> dict:
    """REML with a diagonal covariance; unavailable when any cluster estimate is unavailable."""
    if len(blocks) != cfg.K_TRUE or any(b.get("status") != "ok" for b in blocks):
        return {"status": "unavailable", "error_code": "theta_vector_incomplete"}
    th = np.array([b["theta"] for b in blocks])
    var = np.array([b["se"] for b in blocks]) ** 2
    r = fit_reml_gls(th, np.diag(var))
    return {"status": "ok",
            "reml_gls": (dict(mu=float(r.mu), var_mu=float(r.var_mu), tau2=float(r.tau2))
                         if r.status == "ok" else _na(r.error_code))}


def _sel_counts(S, core, T) -> dict:
    S = set(S)
    return dict(n_selected=len(S), recall_core=(len(S & core) / len(core) if core else float("nan")),
                n_spur=len(S & T["spur"]), n_noise=len(S & T["noise"]))


def global_dual_selection(F, Y, X, env):
    """Dual-side knockoff candidates and Chow-ICP over all centers, with the centers as ICP environments."""
    centers = np.unique(env)
    n_env = int(centers.size)
    if n_env < 2:
        raise RuntimeError(f"global ICP needs at least two environments, found {n_env}")
    cache = _V4Cache()
    key = frozenset(int(c) for c in centers)
    S_x, S_y, _tk, _kc = cache.knockoff(key, F, Y, X, env, cfg.Q_KNOCK)
    cand = sorted(set(S_x) | set(S_y))
    out, _ = cache.icp(key, F, Y, env, cand, n_env)
    sel = sorted(int(j) for j in out[f"kept_{RULE}"])
    diag = dict(global_n_x=len(S_x), global_n_y=len(S_y), global_n_pre=len(cand),
                global_n_icp_env=n_env, global_icp_status=out["status"],
                global_icp_fallback=bool(out["icp_fallback_fired"]))
    return sel, diag


def fit_rep(combo, p: int, d: int, seed_key):
    t_rep = time.perf_counter()
    seeds = seeds_for_key(seed_key)
    full = fit_once(combo, seeds, modes=(MODE,), rules=(RULE,))
    t_fit = time.perf_counter() - t_rep
    data = full.data
    F = np.asarray(data["F"], dtype=float)
    X = np.asarray(data["X_treat"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    env = np.asarray(data["env"]).astype(int)
    gamma = float(combo["true_gamma"])
    dg = full.diagnostics
    members = dg["members_by_mode"][MODE]
    k2true = dg["k2true_by_mode"][MODE]
    T = dg["truth_index"]
    core_by_true = {int(k): set(v) for k, v in dg["A_core_by_k"].items()}

    # Selected sets: HetICS, single-side variants and oracle from fit_once; ablation and pooled sets here.
    sets = {m: {int(k): set(v) for k, v in full.selected_sets[(MODE, RULE, m)].items()} for m in cfg.METHODS}
    plr = {m: full.plr_by_analysis[(MODE, RULE, m)] for m in cfg.METHODS}
    t0 = time.perf_counter()
    pooled, _t, _diag = select_pooled(F, X, Y)
    t_pooled = time.perf_counter() - t0
    t0 = time.perf_counter()
    sel_g, diag_g = global_dual_selection(F, Y, X, env)
    t_global = time.perf_counter() - t0
    nonempty = {int(k): bool(mem) for k, mem in members.items()}
    sets["dual_no_icp"] = {int(k): set(v) for k, v in dg["pre_sets_by"][(MODE, "dual_icp")].items()}
    sets["dual_global_icp"] = {k: (set(sel_g) if ne else set()) for k, ne in nonempty.items()}
    cache = _PLRCache()
    for m in ("dual_no_icp", "dual_global_icp"):
        plr[m] = fit_method_cluster_plrs(data, members, sets[m], seeds.fold_seed, cache)

    # Estimation within learned clusters.
    t0 = time.perf_counter()
    arms = {}
    for m in CLUSTER_ARMS:
        clusters = []
        for k in sorted(int(x) for x in members):
            mem = members[k]
            S = sorted(int(j) for j in sets[m].get(k, ()))
            kt = int(k2true.get(k, -1))
            msk = np.isin(env, mem)
            est = {"robinson": _from_plr(plr[m][k])}
            if mem:
                est.update(extra_estimators(X[msk], Y[msk], F[msk][:, S], env[msk]))
            else:
                est.update({e: _na("empty_cluster") for e in ESTIMATORS[1:]})
            est = {e: _score(b, gamma) for e, b in est.items()}
            row = dict(k=k, k_true=kt, M_k=len(mem), n_k=int(msk.sum()), est=est,
                       **_sel_counts(S, core_by_true.get(kt, set()), T))
            if not m.startswith("oracle"):
                row["selected"] = S
            clusters.append(row)
        arms[m] = dict(scope="learned_cluster", clusters=clusters,
                       mu={e: aggregate([c["est"][e] for c in clusters]) for e in ESTIMATORS})

    # Estimation once on all centers.
    global_sets = {"pooled_lasso_fp": pooled["pooled_lasso"], "pooled_pds_fp": pooled["pooled_pds"],
                   "pooled_oal_fp": pooled["pooled_oal"], "hdps_fp": pooled["hdps"],
                   "oracle_safe_pooled": sorted(int(j) for j in data["A_safe"])}
    safe = set(int(j) for j in data["A_safe"])
    for m in POOLED_ARMS:
        S = sorted(int(j) for j in global_sets[m])
        est = {"robinson": _from_plr(fit_whole_center_plr(X, Y, F[:, S], env, seeds.fold_seed))}
        est.update(extra_estimators(X, Y, F[:, S], env))
        arms[m] = dict(scope="all_centers_pooled", est={e: _score(b, gamma) for e, b in est.items()},
                       **_sel_counts(S, safe, T))
        if not m.startswith("oracle"):
            arms[m]["selected"] = S
    t_est = time.perf_counter() - t0

    rec = dict(output_version=RA_VERSION, experiment=cfg4.EXPERIMENT, beta_tag=cfg.BETA_TAG,
               combo_id=combo["combo_id"], param_index=int(p), data_index=int(d),
               param_seed=int(seeds.param_seed), data_seed=int(seeds.data_seed), fold_seed=int(seeds.fold_seed),
               true_gamma=gamma, n=int(F.shape[0]), p=int(F.shape[1]), n_centers=int(np.unique(env).size),
               seam_ari=float(dg["seam_ari"]), global_dual=diag_g, arms=arms,
               t=dict(fit_once=round(t_fit, 2), pooled_selection=round(t_pooled, 2), global_dual=round(t_global, 2),
                      estimators=round(t_est, 2), total=round(time.perf_counter() - t_rep, 2)),
               peak_rss_gb=_peak_rss_gb())
    return rec


__all__ = ["RA_VERSION", "ESTIMATORS", "CLUSTER_ARMS", "POOLED_ARMS", "fit_rep", "extra_estimators", "aggregate",
           "global_dual_selection"]
