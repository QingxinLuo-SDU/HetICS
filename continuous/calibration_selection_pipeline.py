"""calibration_selection_pipeline.py — selection step of the calibration experiment.

sEAM clustering once per dataset, one fixed-X knockoff construction per non-empty learned cluster with
exposure-side and outcome-side LCD selection, and one strict Chow-ICP on the dual-side candidates
S_x union S_y. The building blocks are those of the main pipeline.

Failures are returned as statuses rather than exceptions:
    empty_cluster                 the learned cluster has no center
    single_center_cluster         one center only, so the Chow test has no comparison
    cluster_unmapped              the Hungarian mapping assigned no true cluster
    knockoff_prerequisite_failed  n_k < 2p or a similar prerequisite failed
    empty_candidate               the knockoff step selected no covariate (estimation proceeds with an empty set)
    icp_no_pvalues                the ICP returned no p-value
    ok                            none of the above
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import adjusted_rand_score

import hypothesis_calibration_config as hcc
import step6x_config as cfg
from plr_transfer import assignment_margins
from seam_clustering import run_seam_clustering
from seam_support_methods import align_clusters_hungarian, icp_detail
from step6_methods import fixed_x_knockoff, lcd_select, prep

ICP_RULE = "strict"
ESTIMABLE_STATUSES = ("ok", "empty_candidate")
CHANNELS = ("dual",)


@dataclass(frozen=True)
class ChannelSelection:
    """Output of one candidate channel on one dataset, indexed by learned cluster."""

    channel: str
    selected_pre_icp: dict              # learned cluster k -> candidates before ICP
    selected_post_icp: dict             # learned cluster k -> set after strict ICP
    cluster_status: dict                # learned cluster k -> status
    a_core_recall: dict                 # learned cluster k -> recall of A_core of the matched true cluster
    icp_detail_rows: list = field(default_factory=list)

    @property
    def n_clusters_ok(self) -> int:
        return sum(1 for s in self.cluster_status.values() if s == "ok")

    @property
    def n_clusters_estimable(self) -> int:
        return sum(1 for s in self.cluster_status.values()
                   if s in ESTIMABLE_STATUSES)

    @property
    def status(self) -> str:
        return "ok" if self.n_clusters_estimable == int(cfg.K_TRUE) else "unavailable"

    @property
    def error_code(self):
        return None if self.status == "ok" else "cluster_selection_unavailable"


@dataclass(frozen=True)
class CalibrationSelectionResult:
    """Selection output on one dataset; the sEAM partition is shared by all channels."""

    center_labels: dict                 # center -> learned cluster k
    members: dict                       # learned cluster k -> [centers]
    k2true: dict                        # learned cluster k -> true cluster (Hungarian)
    channels: dict                      # channel -> ChannelSelection
    seam_ari: float
    label_exact: bool
    icp_rule: str
    alpha_icp: float
    diagnostics: dict = field(default_factory=dict)

    def channel(self, name: str) -> ChannelSelection:
        if name not in self.channels:
            raise KeyError(f"no channel {name!r}; available: {sorted(self.channels)}")
        return self.channels[name]


def _cluster_status_for(k: int, members, k2true: dict) -> str | None:
    """Status before selection; None means the cluster can be screened."""
    if not members:
        return "empty_cluster"
    if int(k) not in k2true:
        return "cluster_unmapped"
    if len(members) < 2:
        return "single_center_cluster"
    return None


def _label_exact(center_labels: dict, true_labels: dict, k2true: dict) -> bool:
    """Whether the learned labels equal the true labels after the Hungarian mapping."""
    common = sorted(set(center_labels) & set(true_labels))
    if not common or not k2true:
        return False
    return all(k2true.get(int(center_labels[c])) == int(true_labels[c])
               for c in common)


def _icp_status(raw: str) -> str:
    return {"ok": "ok", "empty_candidate": "empty_candidate",
            "no_pvalues": "icp_no_pvalues",
            "single_center_cluster": "single_center_cluster",
            }.get(raw, f"icp_{raw}")


def fit_calibration_selection(data: dict, point: dict, algorithm_seed: int,
                              channels=CHANNELS,
                              q_knock: float = cfg.Q_KNOCK,
                              alpha_icp: float | None = None
                              ) -> CalibrationSelectionResult:
    """sEAM once, one knockoff construction per non-empty cluster and one strict ICP per channel.

    alpha_icp defaults to hcc.ALPHA_ICP (1e-3).
    """
    channels = tuple(channels)
    unknown = [c for c in channels if c not in CHANNELS]
    if unknown:
        raise ValueError(f"unknown channels {unknown}; available: {list(CHANNELS)}")
    alpha = hcc.ALPHA_ICP if alpha_icp is None else float(alpha_icp)

    F = np.asarray(data["F"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float).ravel()
    X = np.asarray(data["X_treat"], dtype=float).ravel()
    env = np.asarray(data["env"]).astype(int).ravel()
    centers = np.unique(env)
    K = int(cfg.K_TRUE)

    t0 = time.perf_counter()
    seam = run_seam_clustering(F, Y, env, K=K, sparsity=point["seam_sparsity"],
                               max_iter=cfg.SEAM_MAX_ITER, tol=cfg.SEAM_TOL,
                               random_state=int(algorithm_seed),
                               lipschitz_method="exact")
    t_seam = time.perf_counter() - t0

    cc = np.asarray(seam["center_cluster_hat"]).ravel()
    center_labels = {int(e): int(cc[i]) for i, e in enumerate(centers)}
    members = {k: [int(e) for e in centers if center_labels[int(e)] == k]
               for k in range(K)}
    true_labels = {int(c): int(g) for c, g in data["center_to_cluster"].items()}
    k2true = align_clusters_hungarian(data["true_cluster"], cc, K)
    seam_ari = float(adjusted_rand_score(data["true_cluster"], cc))
    core_by_true = {int(k): set(int(j) for j in v)
                    for k, v in data["A_core_by_cluster"].items()}

    per = {c: dict(pre={}, post={}, status={}, recall={}, rows=[]) for c in channels}
    t_select = 0.0

    for k in range(K):
        mem = members[k]
        for c in channels:
            per[c]["pre"][k], per[c]["post"][k] = set(), set()
            per[c]["recall"][k] = float("nan")

        early = _cluster_status_for(k, mem, k2true)
        if early is not None:
            for c in channels:
                per[c]["status"][k] = early
            continue

        msk = np.isin(env, mem)
        F_k, Y_k, X_k, env_k = F[msk], Y[msk], X[msk], env[msk]

        t0 = time.perf_counter()
        try:
            Fp, Yp = prep(F_k, Y_k, env_k)
            _, Xp = prep(F_k, X_k, env_k)      # Fp is identical; only the response is needed
            Ft = fixed_x_knockoff(Fp)
        except (AssertionError, ValueError, np.linalg.LinAlgError):
            for c in channels:
                per[c]["status"][k] = "knockoff_prerequisite_failed"
            t_select += time.perf_counter() - t0
            continue
        S_y = set(int(j) for j in lcd_select(Fp, Ft, Yp, q=q_knock))
        S_x = set(int(j) for j in lcd_select(Fp, Ft, Xp, q=q_knock))
        t_select += time.perf_counter() - t0

        cand_by_channel = {"dual": S_x | S_y}
        core = core_by_true.get(int(k2true[int(k)]), set())

        for c in channels:
            cand = set(cand_by_channel[c])
            per[c]["pre"][k] = set(cand)
            out = icp_detail(F_k, Y_k, env_k, sorted(cand), len(mem), alpha=alpha)
            t_select += float(out.get("t_icp", 0.0))
            per[c]["post"][k] = set(int(j) for j in out["kept_strict"])
            per[c]["status"][k] = _icp_status(out.get("status", "ok"))
            for d in out.get("detail", []):
                per[c]["rows"].append(dict(
                    channel=c, k=int(k), k_true=int(k2true[int(k)]),
                    var=int(d["var"]), pv_raw=float(d["pv_raw"]),
                    pv_bh=float(d["pv_bh"]), kept=bool(d["kept"])))
            per[c]["recall"][k] = ((len(per[c]["post"][k] & core) / len(core))
                                   if core else float("nan"))

    sdiag = seam.get("diagnostics", {}) or {}
    n_iter = int(sdiag.get("n_iter", -1))
    diagnostics = dict(
        seam_n_iter=n_iter,
        seam_converged=bool(sdiag.get("converged", False)),
        seam_hit_max_iter=bool(n_iter >= cfg.SEAM_MAX_ITER),
        n_centers_total=int(centers.size),
        n_centers_by_cluster={k: len(v) for k, v in members.items()},
        n_rows_by_cluster={k: int(np.isin(env, v).sum()) for k, v in members.items()},
        t_seam=round(t_seam, 3), t_select=round(t_select, 3),
        **assignment_margins(seam.get("weights"), centers),
    )
    return CalibrationSelectionResult(
        center_labels=center_labels, members=members, k2true=dict(k2true),
        channels={c: ChannelSelection(
            channel=c, selected_pre_icp=per[c]["pre"],
            selected_post_icp=per[c]["post"], cluster_status=per[c]["status"],
            a_core_recall=per[c]["recall"], icp_detail_rows=per[c]["rows"])
            for c in channels},
        seam_ari=seam_ari,
        label_exact=_label_exact(center_labels, true_labels, k2true),
        icp_rule=ICP_RULE, alpha_icp=alpha, diagnostics=diagnostics,
    )
