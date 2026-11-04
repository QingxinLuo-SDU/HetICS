"""_binary.py — HetICS-AIPW for a binary exposure and outcome."""
import numpy as np

from ._aipw import fit_partition_aipw
from ._screening import check_inputs, cluster_members, icp_select, knockoff_candidates

# The only arm statuses with eligible=True in `_arm_from_selection`; `complete_max_iter_eligible` counts as
# eligible.
ELIGIBLE_ARM_STATUSES = ("complete", "complete_max_iter_eligible")
SOLVER = dict(step_size=1.0, max_iter=300, warmup_steps=50, parameter_tolerance=0.001,
              objective_relative_tolerance=0.00001)


def _center_map(pipeline_result, observed):
    """Frozen labels of the primary arm as {center code: cluster}; (None, status) when they abstain or fail."""
    primary = pipeline_result.primary
    status = {"pipeline_status": str(pipeline_result.status), "primary_status": str(primary.status),
              "labels_frozen": bool(primary.labels_frozen),
              "fixed_refit_status": None if primary.fixed_refit_status is None else str(primary.fixed_refit_status),
              "fixed_refit_eligible": bool(primary.fixed_refit_eligible),
              "fixed_refit_converged": bool(primary.fixed_refit_converged)}
    labels = primary.frozen_labels
    if labels is None:
        status["reason"] = primary.diagnostics.get("fixed_refit_failure_reason", "no_frozen_labels")
        return None, status
    if str(primary.status) not in ELIGIBLE_ARM_STATUSES or not bool(primary.fixed_refit_eligible):
        status["reason"] = f"label arm not eligible: status={primary.status}"
        return None, status
    centers = np.asarray(observed.centers)
    return {int(c): int(k) for c, k in zip(centers, np.asarray(labels, dtype=np.int64))}, status


def fit_aipw(F, X, Y, centers, *, K, analysis_id="hetics", q=0.10, alpha=0.001, support_size=25, feature_names=None):
    """HetICS-AIPW: center labels, dual-side selection with invariance selection, and AIPW within clusters.

    F is the (n, p) array of baseline covariates (no intercept or center dummies); X and Y are 0/1 vectors of length
    n and centers holds the center of each row. K is the prespecified number of center clusters. analysis_id is
    any stable text; the seeds of the label algorithm derive from it. q and alpha are the knockoff+ level and the BH
    threshold of the invariance selection; support_size is the support of the label model.

    Returns a dict with "estimate", "se" and "ci95" (person-weighted average treatment effect on the risk-difference
    scale), "point_status" and "se_status", "clusters" (centers, candidate and selected covariates as column indices
    of F, cluster effect), "labels" (status of the label algorithm) and "effect" (the full AIPW result). Raises
    RuntimeError when the label algorithm abstains and ValueError when a cluster is too small for fixed-X knockoffs.
    """
    from ._labels import cem_types, design, model, multiview_pipeline
    from ._labels.multiview_types import ObservedDataView

    F, X, Y, env, labels, names = check_inputs(F, X, Y, centers, feature_names, binary=True)
    n, p = F.shape
    if not 1 <= K < labels.size:
        raise ValueError(f"K must be at least 1 and smaller than the number of centers; K={K}, {labels.size} centers")
    if not 1 <= support_size <= p:
        raise ValueError(f"support_size must be in [1, {p}]")
    base = int(design.derive_algorithm_seed_base(str(analysis_id), 1, 1))
    config = multiview_pipeline.PipelineConfig(
        fit_config=model.FitConfig(support_size=int(support_size), **SOLVER),
        cem_config=cem_types.CEMConfig(k_components=int(K), max_outer_iter=4),
        stability_threshold=0.80, dataset_seed_base=base,
        split_seed=int(design.derive_algorithm_child_seed(base, stage="split")),
        candidate_seed=int(design.derive_algorithm_child_seed(base, stage="candidate")),
        epsilon_prob=0.01, split_mode="full_sample")
    observed = ObservedDataView(F, X.astype(int), Y.astype(int), env, np.unique(env))
    center_map, label_status = _center_map(multiview_pipeline.run_label_pipeline(observed, config), observed)
    if center_map is None:
        raise RuntimeError(f"the label algorithm did not return an eligible partition: {label_status}")

    members = cluster_members(env, center_map)
    selected_by_cluster, rows = {}, []
    for k, mem in members.items():
        take = np.isin(env, mem)
        if take.sum() <= 2 * p + 1:
            raise ValueError(f"cluster {k} has {int(take.sum())} observations; fixed-X knockoffs need more than {2 * p + 1}")
        Fk, Xk, Yk, ek = F[take], X[take], Y[take], env[take]
        sx, sy = knockoff_candidates(Fk, Yk, Xk, ek, q)
        selected, detail = icp_select(Fk, Yk, ek, sx | sy, len(mem), alpha, fallback=True)
        selected_by_cluster[int(k)] = selected
        rows.append({"cluster": int(k), "centers": labels[np.asarray(mem)].tolist(), "n": int(take.sum()),
                     "candidates_exposure": sorted(sx), "candidates_outcome": sorted(sy), "selected": selected,
                     "selected_names": None if names is None else [names[j] for j in selected],
                     "invariance": detail, "single_center": len(mem) == 1})
    effect = fit_partition_aipw(F, X.astype(int), Y.astype(int), env, center_map, selected_by_cluster,
                                backend="statsmodels_glm")
    theta = dict(zip(effect.get("cluster_order") or [], effect.get("theta") or []))
    for row in rows:
        row["theta"] = theta.get(row["cluster"])
    return {"method": "HetICS-AIPW", "estimand": "risk difference", "estimate": effect.get("estimate"),
            "se": effect.get("se"), "ci95": effect.get("ci95"), "point_status": effect.get("point_status"),
            "se_status": effect.get("se_status"), "clusters": rows, "labels": label_status, "effect": effect,
            "settings": {"K": K, "analysis_id": str(analysis_id), "q": q, "alpha": alpha, "support_size": support_size}}
