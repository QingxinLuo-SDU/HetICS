"""step6x_core_v4.py — one full-sample fit of the continuous HetICS selection pipeline.

fit_once() generates one dataset, learns the center clusters with sEAM, builds the exposure-side and
outcome-side knockoff candidates within each learned cluster, applies the within-cluster Chow-ICP and
fits the cluster-specific Robinson (whole-center PLR) estimates for every selected set. ra_core.py adds
the RA, GPS-IPW, ablation and all-center comparator estimates on the same data.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

import seam_support_methods as sm
import step6x_config as cfg
import step6x_config_v4 as cfg4
import step6x_methods as m6x
from latent_cluster_dgp_specx import generate_latent_cluster_dgp_specx
from latent_cluster_dgp_specx_u import generate_latent_cluster_dgp_specx_u
from plr_transfer import assignment_margins
from seam_clustering import run_seam_clustering
from v4_types import RealizationFit, unavailable_plr
from whole_center_plr import fit_whole_center_plr

K_TRUE = cfg.K_TRUE


def _final_loglik(trace):
    """Final value of the sEAM log-likelihood trace."""
    if trace is None:
        return float("nan")
    if np.isscalar(trace):
        return float(trace)
    a = np.asarray(trace, dtype=float).ravel()
    return float(a[-1]) if a.size else float("nan")


class _V4Cache:
    """Per-replicate cache of deterministic results keyed by center set (and candidate set)."""

    def __init__(self):
        self._knock, self._icp = {}, {}

    def knockoff(self, key, F_k, Y_k, X_k, env_k, q):
        if key in self._knock:
            return (*self._knock[key], True)
        out = m6x.knockoff_candidates(F_k, Y_k, X_k, env_k, q)
        self._knock[key] = out
        return (*out, False)

    def icp(self, key, F_k, Y_k, env_k, cand, n_centers_k):
        kk = (key, tuple(sorted(cand)))
        if kk in self._icp:
            return self._icp[kk], True
        out = sm.icp_detail(F_k, Y_k, env_k, cand, n_centers_k, alpha=cfg4.ALPHA_ICP)
        self._icp[kk] = out
        return out, False


def _members_by_mode(mode, data, centers, c2k_hat, K):
    """mode="seam": learned partition (used for estimation); mode="oracle": true partition (diagnostic)."""
    if mode == "seam":
        return {k: [int(e) for e in centers if c2k_hat[int(e)] == k] for k in range(K)}
    c2k_true = data["center_to_cluster"]
    return {k: [int(e) for e in centers if int(c2k_true[int(e)]) == k] for k in range(K)}


def run_seam(F, Y, env, combo, algorithm_seed):
    """Stage 0: learned center clusters."""
    return run_seam_clustering(F, Y, env, K=K_TRUE, sparsity=combo["seam_sparsity"],
                               max_iter=cfg.SEAM_MAX_ITER, tol=cfg.SEAM_TOL,
                               random_state=algorithm_seed, lipschitz_method="exact")


def fit_existing_selection_pipeline(data, combo, algorithm_seed,
                                    modes=("seam",), rules=("strict",),
                                    q_knock=cfg.Q_KNOCK) -> dict:
    """Stages 0-2 (clustering, candidates, invariance selection); no effect estimation."""
    K = K_TRUE
    F, Y, X, env = data["F"], data["Y"], data["X_treat"], data["env"]
    centers = np.unique(env)

    T = sm.truth_index(data)
    A_core_by_k = {int(k): set(v) for k, v in data["A_core_by_cluster"].items()}
    A_safe = set(data["A_safe"])

    t0 = time.perf_counter()
    seam = run_seam(F, Y, env, combo, algorithm_seed)
    t_seam = time.perf_counter() - t0

    cc = seam["center_cluster_hat"]
    c2k_hat = {int(e): int(cc[i]) for i, e in enumerate(centers)}
    # Assignment margin (top weight minus second weight) for every center.
    margins = assignment_margins(seam.get("weights"), centers)
    ari = float(adjusted_rand_score(data["true_cluster"], cc))
    est2true = sm.align_clusters_hungarian(data["true_cluster"], cc, K)
    sdiag = seam.get("diagnostics", {})
    n_iter = int(sdiag.get("n_iter", -1))

    cache = _V4Cache()
    members_by_mode, k2true_by_mode = {}, {}
    selected_sets, pre_sets_by = {}, {}
    cluster_rows, icp_rows = [], []
    t_select = {m: 0.0 for m in cfg.METHODS}

    for mode in modes:
        members = _members_by_mode(mode, data, centers, c2k_hat, K)
        members_by_mode[mode] = members
        # Quantities indexed by true cluster are mapped through the Hungarian alignment.
        k2true = dict(est2true) if mode == "seam" else {k: k for k in range(K)}
        k2true_by_mode[mode] = k2true

        pre_sets = {m: {} for m in cfg.METHODS}
        sel = {rule: {m: {} for m in cfg.METHODS} for rule in rules}

        for k in range(K):
            mem = members[k]
            if not mem:
                for m in cfg.METHODS:
                    pre_sets[m][k] = []
                    for rule in rules:
                        sel[rule][m][k] = []
                cluster_rows.append(dict(partition_mode=mode, seam_ari=round(ari, 4),
                                         k=k, k_true=k2true.get(k, k),
                                         n_centers_k=0, n_k=0, status="empty_cluster"))
                continue

            msk = np.isin(env, mem)
            F_k, Y_k, X_k, env_k = F[msk], Y[msk], X[msk], env[msk]
            key = frozenset(mem)

            S_x, S_y, tk, knock_cached = cache.knockoff(key, F_k, Y_k, X_k, env_k, q_knock)

            cluster_rows.append(dict(
                partition_mode=mode, seam_ari=round(ari, 4),
                k=k, k_true=k2true.get(k, k),
                n_centers_k=len(mem), n_k=int(msk.sum()),
                status="ok" if len(mem) > 1 else "single_center_cluster",
                size_x_pre=len(S_x), size_y_pre=len(S_y),
                knock_from_cache=knock_cached))

            cand = {
                "xside_icp":  set(S_x),
                "dual_icp":   set(S_x) | set(S_y),
                "yside_icp":  set(S_y),
                "oracle_min": set(A_core_by_k[k2true[k]]),
            }

            for m in cfg.METHODS:
                pre = sorted(cand[m])
                pre_sets[m][k] = pre
                t_select[m] += sum(tk[c] for c in m6x.KNOCK_COST[m])
                if m in cfg.ICP_METHODS:
                    out, icp_cached = cache.icp(key, F_k, Y_k, env_k, pre, len(mem))
                    for rule in rules:
                        sel[rule][m][k] = out[f"kept_{rule}"]
                    t_select[m] += out["t_icp"]
                    for d in out["detail"]:
                        icp_rows.append(dict(
                            partition_mode=mode, method=m, k=k, var=d["var"],
                            pv_raw=round(d["pv_raw"], 6), pv_bh=round(d["pv_bh"], 6),
                            kept=d["kept"], truth_type=sm.truth_type_of(d["var"], T),
                            in_own_spec=int(d["var"] in T["spec_by_k"].get(k2true[k], set())),
                            from_x=int(d["var"] in S_x), from_y=int(d["var"] in S_y),
                            icp_status=out["status"],
                            icp_fallback_fired=out["icp_fallback_fired"],
                            icp_from_cache=icp_cached))
                else:
                    for rule in rules:
                        sel[rule][m][k] = pre      # the oracle set is not screened

        for m in cfg.METHODS:
            pre_sets_by[(mode, m)] = pre_sets[m]
            for rule in rules:
                selected_sets[(mode, rule, m)] = {k: set(v) for k, v in sel[rule][m].items()}

    return dict(
        center_labels=c2k_hat, members_by_mode=members_by_mode,
        k2true_by_mode=k2true_by_mode,
        selected_sets=selected_sets, pre_sets_by=pre_sets_by,
        diagnostics=dict(
            seam_ari=ari, seam_converged=bool(sdiag.get("converged", False)),
            seam_n_iter=n_iter,
            seam_hit_max_iter=bool(n_iter >= cfg.SEAM_MAX_ITER),
            seam_loglik=_final_loglik(sdiag.get("log_likelihood")),
            **margins,
            t_seam=round(t_seam, 3), t_select=t_select,
            cluster_rows=cluster_rows, icp_rows=icp_rows,
            truth_index=T,
            A_core_by_k=A_core_by_k, A_safe=A_safe,
            members_by_mode=members_by_mode, k2true_by_mode=k2true_by_mode,
            pre_sets_by=pre_sets_by,
        ),
    )


class _PLRCache:
    """Whole-center PLR fits keyed by (center set, adjustment set, fold seed)."""

    def __init__(self):
        self._store = {}

    def get(self, data, mem, cols, fold_seed):
        key = (frozenset(mem), tuple(cols), fold_seed)
        if key in self._store:
            return self._store[key]
        env = np.asarray(data["env"])
        msk = np.isin(env, list(mem))
        res = fit_whole_center_plr(
            np.asarray(data["X_treat"])[msk], np.asarray(data["Y"])[msk],
            np.asarray(data["F"])[msk][:, list(cols)], env[msk], fold_seed)
        self._store[key] = res
        return res


def fit_method_cluster_plrs(data, members, selected_by_k, fold_seed,
                            cache: _PLRCache | None = None) -> dict:
    cache = cache or _PLRCache()
    out = {}
    for k, mem in members.items():
        if not mem:
            out[int(k)] = unavailable_plr("empty_cluster")
            continue
        cols = sorted(int(j) for j in selected_by_k.get(k, ()))
        out[int(k)] = cache.get(data, mem, cols, fold_seed)
    return out


def fit_once(combo, seed_bundle, modes=("seam",), rules=("strict",)) -> RealizationFit:
    """Generate one dataset, run the selection pipeline and fit cluster-specific PLRs for every arm."""
    data = generate_combo_data(combo, seed_bundle)
    selection = fit_existing_selection_pipeline(
        data, combo, algorithm_seed=seed_bundle.algorithm_seed,
        modes=modes, rules=rules)

    cache = _PLRCache()
    plr_by_analysis = {}
    for analysis_key, selected_by_k in selection["selected_sets"].items():
        mode = analysis_key[0]
        plr_by_analysis[analysis_key] = fit_method_cluster_plrs(
            data, selection["members_by_mode"][mode], selected_by_k,
            seed_bundle.fold_seed, cache)

    return RealizationFit(
        data=data,
        centers=tuple(sorted(int(e) for e in np.unique(data["env"]))),
        center_labels=selection["center_labels"],
        selected_sets=selection["selected_sets"],
        plr_by_analysis=plr_by_analysis,
        diagnostics=selection["diagnostics"],
    )


def generate_combo_data(combo, seed_bundle) -> dict:
    """Data generation: the specx DGP, or specx_u for points with an unmeasured confounder."""
    if combo.get("unmeasured_u"):
        data = generate_latent_cluster_dgp_specx_u(
            **combo["dgp_params"], **combo["unmeasured_u"],
            seed=seed_bundle.data_seed, param_seed=seed_bundle.param_seed,
        )
        assert data["scenario_tag"] == "specx_u", "U design point did not use specx_u"
    else:
        data = generate_latent_cluster_dgp_specx(
            **combo["dgp_params"], seed=seed_bundle.data_seed,
            param_seed=seed_bundle.param_seed,
        )
        assert data["scenario_tag"] == "specx", "design point did not use the specx DGP"
    return data
