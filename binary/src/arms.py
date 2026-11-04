"""arms.py — arms of the formal binary units and the alignment of learned clusters to true clusters.

| method            | candidate / adjustment set                         | after |
|-------------------|----------------------------------------------------|-------|
| dual_icp          | fixed-X + BIC S_X ∪ S_Y                            | one ICP |
| xside_icp         | S_X                                                | one ICP |
| yside_icp         | S_Y                                                | one ICP |
| oracle_min        | Z ∪ shared ∪ spec of the aligned true cluster      | no ICP |
| oracle_reference  | true partition and true observed core              | no ICP; calibration rows only |

Recorded in assumptions_not_covered:
  * the fixed-X knockoff uses a linear BIC working score and claims no exact FDR control for a binary response;
  * the ICP is the partial Chow test with center intercepts and linear slopes and is not exactly calibrated
    for a binary outcome.
"""

from __future__ import annotations

import itertools

import numpy as np

FIXEDX_ARMS = ("xside_icp", "yside_icp", "dual_icp")
ORACLE_ARMS = ("oracle_min",)
REFERENCE_ARM = "oracle_reference"
ALL_ARMS = FIXEDX_ARMS + ORACLE_ARMS + (REFERENCE_ARM,)

ASSUMPTIONS_NOT_COVERED = (
    "fixed_x_knockoff_uses_linear_bic_score_on_binary_response",
    "official_icp_partial_chow_not_exact_under_binary_outcome",
)


def align_clusters_to_truth(analysis_members_rows: dict[int, np.ndarray],
                            true_cluster_sample: np.ndarray, K: int) -> dict:
    """One-to-one matching of learned to true clusters that maximizes the number of shared individuals.

    All injections are enumerated (K = 3 in this design), so the optimum is global; among tied optima the
    lexicographically smallest mapping vector is taken.
    """
    analysis_ids = sorted(analysis_members_rows)
    true_ids = list(range(int(K)))
    overlap = np.zeros((len(analysis_ids), len(true_ids)), dtype=np.int64)
    tcs = np.asarray(true_cluster_sample, int)
    for a, g in enumerate(analysis_ids):
        rows = np.asarray(analysis_members_rows[g], dtype=np.int64)
        if rows.size:
            counts = np.bincount(tcs[rows], minlength=len(true_ids))
            overlap[a, :] = counts[:len(true_ids)]

    best_value, best_perm, n_optimal = -1, None, 0
    for perm in itertools.permutations(true_ids, len(analysis_ids)):
        value = int(sum(overlap[a, perm[a]] for a in range(len(analysis_ids))))
        if value > best_value:
            best_value, best_perm, n_optimal = value, perm, 1
        elif value == best_value:
            n_optimal += 1
            if list(perm) < list(best_perm):
                best_perm = perm
    mapping = {analysis_ids[a]: int(best_perm[a]) for a in range(len(analysis_ids))}
    sizes = {g: int(np.asarray(analysis_members_rows[g]).size) for g in analysis_ids}
    purity = {}
    for a, g in enumerate(analysis_ids):
        n_g = sizes[g]
        purity[g] = float(overlap[a, mapping[g]] / n_g) if n_g else float("nan")
    return {
        "mapping": mapping,
        "overlap_table": {int(analysis_ids[a]): overlap[a].tolist()
                          for a in range(len(analysis_ids))},
        "total_overlap": int(best_value),
        "n_optimal_solutions": int(n_optimal),
        "tie_broken": bool(n_optimal > 1),
        "cluster_sizes": sizes,
        "purity": purity,
        "is_mixed_cluster": {g: bool(purity[g] < 1.0) for g in analysis_ids},
        "mixed_fraction": float(np.mean([1.0 - purity[g] for g in analysis_ids]))
        if analysis_ids else float("nan"),
        "rule": "global_one_to_one_max_individual_overlap; ties -> lexicographically "
                "smallest mapped true-cluster vector over ascending analysis clusters",
        "truth_use": "evaluation_and_oracle_diagnostics_only",
    }


__all__ = ["FIXEDX_ARMS", "ORACLE_ARMS", "REFERENCE_ARM", "ALL_ARMS",
           "ASSUMPTIONS_NOT_COVERED", "align_clusters_to_truth"]
