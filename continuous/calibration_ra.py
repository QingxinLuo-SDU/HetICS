"""calibration_ra.py — one calibration replicate: HetICS and the oracle reference with four estimators.

  dual_icp          learned clusters and the dual-side set after invariance selection, aligned to the true
                    clusters by the Hungarian mapping
  oracle_reference  true clusters and A_core[k]
Estimators: Robinson, RA, GPS-IPW and truncated GPS-IPW (ra_core.extra_estimators) on the same members and
selected sets.
"""
from __future__ import annotations

import time

import numpy as np

import hypothesis_calibration_config as hcc
from calibration_selection_pipeline import fit_calibration_selection
from hypothesis_calibration_seeds import calibration_seeds
from latent_cluster_dgp_specx import generate_latent_cluster_dgp_specx
from plr_transfer import fit_reference_plr
from ra_core import ESTIMATORS, _from_plr, _na, _score, aggregate, extra_estimators
from step6x_core_v4 import fit_method_cluster_plrs
from v4_types import unavailable_plr

CAL_RA_VERSION = "cont-cal-ra-v3"
CAL_ESTIMATORS = ESTIMATORS
K_TRUE = hcc.K_TRUE


def _cluster(X, Y, F, env, members, S, rob, gamma, kt, k):
    msk = np.isin(env, members) if members else np.zeros(env.size, bool)
    S = sorted(int(j) for j in S)
    est = {"robinson": _from_plr(rob)}
    if members:
        est.update(extra_estimators(X[msk], Y[msk], F[msk][:, S], env[msk]))
    else:
        est.update({e: _na(rob.error_code or "empty_cluster") for e in ESTIMATORS[1:]})
    return dict(k_true=kt, k=k, M_k=len(members), n_k=int(msk.sum()), n_selected=len(S),
                est={e: _score(b, gamma) for e, b in est.items()})


def fit_rep(point: dict, p: int, d: int):
    t0 = time.perf_counter()
    seeds = calibration_seeds(point, p, d)
    data = generate_latent_cluster_dgp_specx(**point["dgp_params"], seed=seeds.data_seed,
                                             param_seed=seeds.param_seed)
    assert data["scenario_tag"] == "specx", "design point did not use the specx DGP"
    F = np.asarray(data["F"], dtype=float)
    X = np.asarray(data["X_treat"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    env = np.asarray(data["env"]).astype(int)
    gamma = float(point["gamma_true"])
    core = {int(k): set(int(j) for j in v) for k, v in data["A_core_by_cluster"].items()}
    true_members = {kt: sorted(int(c) for c, g in data["center_to_cluster"].items() if int(g) == kt)
                    for kt in range(K_TRUE)}

    selection = fit_calibration_selection(data, point, seeds.algorithm_seed,
                                          channels=[hcc.ARM_CHANNEL[a] for a in hcc.RANKED_ARMS],
                                          alpha_icp=point.get("alpha_icp", hcc.ALPHA_ICP))
    k_by_true = {int(kt): int(k) for k, kt in selection.k2true.items()}
    arms = {}
    for arm in hcc.RANKED_ARMS:
        ch = selection.channel(hcc.ARM_CHANNEL[arm])
        plr = fit_method_cluster_plrs(data, selection.members, ch.selected_post_icp, seeds.fold_seed)
        cl = []
        for kt in range(K_TRUE):
            if kt not in k_by_true:
                cl.append(_cluster(X, Y, F, env, [], set(), unavailable_plr("theta_vector_incomplete"),
                                   gamma, kt, -1))
                continue
            k = k_by_true[kt]
            cl.append(_cluster(X, Y, F, env, selection.members.get(k, []), ch.selected_post_icp.get(k, set()),
                               plr.get(k, unavailable_plr("empty_cluster")), gamma, kt, k))
        arms[arm] = cl
    ref = fit_reference_plr(data, seeds.fold_seed)
    arms[hcc.DIAGNOSTIC_ARM] = [_cluster(X, Y, F, env, true_members[kt], core.get(kt, set()),
                                         ref.get(kt, unavailable_plr("empty_cluster")), gamma, kt, kt)
                                for kt in range(K_TRUE)]
    out = dict(output_version=CAL_RA_VERSION, point_id=point["point_id"], param_index=int(p), data_index=int(d),
               gamma_true=gamma, centers_per_cluster=int(point["centers_per_cluster"]),
               n_centers=int(point["n_centers"]), n_per_center=int(point["dgp_params"]["n_per_center"]),
               cal_axis=point.get("cal_axis", "m"), seam_ari=float(selection.seam_ari),
               arms={a: dict(clusters=cl, mu={e: aggregate([c["est"][e] for c in cl]) for e in CAL_ESTIMATORS})
                     for a, cl in arms.items()},
               t_total=round(time.perf_counter() - t0, 2))
    return out


__all__ = ["CAL_RA_VERSION", "CAL_ESTIMATORS", "fit_rep"]
