"""screening_comparison.py — fixed-X knockoff screening with the within-cluster ICP on a fixed partition.

Each analysis cluster calls vendor step6_methods.knockoff_family_cluster once (fixed-X knockoffs,
LassoLarsIC with BIC, knockoff+ at q = 0.10, partial Chow ICP with BH). The exposure-side, outcome-side and
dual directions share one knockoff copy; the dual candidate set is the union of the two one-sided sets and
goes through one ICP.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from screening_bridge import cluster_members

FAMILIES = ("fixedx_bic",)
DIRECTIONS = ("xside_icp", "yside_icp", "dual_icp")


def screen_family(F, X, Y, env, center_to_cluster, family, *, m6, task=None, config=None):
    """Candidates and ICP selections of the three directions in every cluster, with per-cluster detail."""
    if center_to_cluster is None:
        return {"family": family, "status": "abstain",
                "directions": {d: {"status": "abstain", "selected_by_cluster": {},
                                   "pre_icp_by_cluster": {}, "errors_by_cluster": {},
                                   "elapsed_by_cluster": {},
                                   "reason": "upstream_abstain"} for d in DIRECTIONS},
                "detail_by_cluster": {}}
    if family not in FAMILIES:
        raise ValueError(f"Unknown screening family: {family}")

    F = np.asarray(F, dtype=float)
    X = np.asarray(X).astype(int)
    Y = np.asarray(Y).astype(int)
    env = np.asarray(env)
    members = cluster_members(env, center_to_cluster)

    result: dict[str, Any] = {
        "family": family, "status": "complete",
        "directions": {d: {"status": "complete", "selected_by_cluster": {},
                           "pre_icp_by_cluster": {}, "errors_by_cluster": {},
                           "elapsed_by_cluster": {}} for d in DIRECTIONS},
        "detail_by_cluster": {},
    }

    for k, mem in members.items():
        mask = np.isin(env, mem)
        t0 = time.perf_counter()
        try:
            fam, detail = m6.knockoff_family_cluster(F[mask], Y[mask].astype(float), X[mask].astype(float),
                                                     env[mask], len(mem))
        except Exception as error:                              # noqa: BLE001
            reason = f"{type(error).__name__}: {error}"
            for d in DIRECTIONS:
                result["directions"][d]["errors_by_cluster"][k] = reason
                result["directions"][d]["status"] = "failed"
            result["status"] = "failed"
            result["detail_by_cluster"][k] = {"status": "failed", "reason": reason}
            continue
        detail["elapsed_cluster"] = time.perf_counter() - t0
        result["detail_by_cluster"][k] = detail
        for name, key in zip(DIRECTIONS, ("x", "y", "dual")):
            sel, elapsed = fam[name]
            slot = result["directions"][name]
            slot["selected_by_cluster"][k] = sorted(int(j) for j in sel)
            slot["pre_icp_by_cluster"][k] = sorted(int(j) for j in detail["pre_icp"][key])
            slot["elapsed_by_cluster"][k] = float(elapsed)
    return result


__all__ = ["FAMILIES", "DIRECTIONS", "screen_family"]
