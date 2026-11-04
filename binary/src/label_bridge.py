"""label_bridge.py — the label package (full-sample mode) on observed data of the latent-cluster DGP.

  * resolves the packaged `vendor/bniseam_test` (labels) and `vendor/step6_pkg` (screening and DGP);
  * generates one data set of the registered configuration and passes only the observed data to the
    label entry point;
  * calls `run_label_pipeline` with `split_mode="full_sample"` (primary risk, guard, AB/BA, freezing and
    refitting unchanged);
  * converts `pipeline.primary.frozen_labels` into center_to_cluster in the actual center order of the
    returned object; returns None with the status when the labels abstain or fail.

The truth (ate_do / ate_k / true_cluster) is returned separately for evaluation and is not an algorithm input.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
VENDOR_DIR = os.path.join(_PKG_ROOT, "vendor")
STEP6_DIR = os.path.join(VENDOR_DIR, "step6_pkg")

# Configuration: partition, combinations and seeds from config/b2_multiview_design.json; solver, sample size,
# guard threshold (GUARD_STABILITY_THRESHOLD) and epsilon_prob (EPSILON_PROB) as registered for the label
# package.
REGISTERED_UNIT: dict[str, Any] = {
    "combo_id": "B2_ANCHOR",
    "param_rep": 1,
    "data_rep": 1,
    "override_n_per_center": 8000,
    "design_config": "config/b2_multiview_design.json",
    "solver_config": "configs/b2_v3_r_nper8000_diagnostic.yaml",
    "solver": {
        "support_size": 25,
        "step_size": 1.0,
        "max_iter": 300,
        "warmup_steps": 50,
        "parameter_tolerance": 0.001,
        "objective_relative_tolerance": 0.00001,
    },
    "cem": {"k_components": 3, "max_outer_iter": 4},
    "guard_stability_threshold": 0.80,
    "epsilon_prob": 0.01,
    "split_mode": "full_sample",
}

# The only arm statuses with eligible=True in `_arm_from_selection`; `complete_max_iter_eligible` counts as
# eligible.
ELIGIBLE_ARM_STATUSES = ("complete", "complete_max_iter_eligible")


def _ensure_paths() -> None:
    for path in (VENDOR_DIR, STEP6_DIR):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def load_label_package():
    """Import the packaged label code; returns (design, pipeline_module, model, cem_types)."""
    _ensure_paths()
    from bniseam_test import cem_types, design, model, multiview_pipeline
    for mod in (design, multiview_pipeline):
        actual = os.path.abspath(getattr(mod, "__file__", "") or "")
        if not actual.startswith(os.path.abspath(VENDOR_DIR)):
            raise RuntimeError(f"{mod.__name__} comes from {actual}, not from the packaged copy")
    return design, multiview_pipeline, model, cem_types


def load_dgp():
    """Import the packaged latent-cluster DGP and the binary layer."""
    _ensure_paths()
    if "step6b_binary" in sys.modules:
        loaded = os.path.abspath(sys.modules["step6b_binary"].__file__ or "")
        if not loaded.startswith(os.path.abspath(STEP6_DIR)):
            raise RuntimeError(f"step6b_binary is already loaded from {loaded}")
    from latent_cluster_dgp_formal import generate_latent_cluster_dgp
    import step6b_binary as b6
    return generate_latent_cluster_dgp, b6


def registered_task(unit: dict[str, Any] | None = None):
    """The fixed unit chosen in advance from the registered design grid (seeds not chosen from results)."""
    import dataclasses

    unit = dict(REGISTERED_UNIT if unit is None else unit)
    design, _pipeline, _model, _cem = load_label_package()
    grid = [t for t in design.build_tasks()
            if t.combo_id == unit["combo_id"]
            and t.param_rep == int(unit["param_rep"])
            and t.data_rep == int(unit["data_rep"])]
    if not grid:
        raise ValueError(f"unit not in the registered grid: {unit}")
    task = grid[0]
    override = unit.get("override_n_per_center")
    if override is not None:
        combo = dataclasses.replace(
            task.combo,
            params={**dict(task.combo.params), "n_per_center": int(override)})
        task = dataclasses.replace(task, combo=combo)
    return task


def generate_observed(task):
    """One data set of the registered unit; returns (observed, truth_for_evaluation, meta)."""
    from bniseam_test.multiview_types import ObservedDataView

    generate_latent_cluster_dgp, b6 = load_dgp()
    params = dict(task.combo.params)
    sig_x_spec = float(params.pop("SIG_X_SPEC"))
    original = b6.SIG_X_SPEC
    b6.SIG_X_SPEC = sig_x_spec
    try:
        continuous = generate_latent_cluster_dgp(
            **params, seed=task.data_seed, param_seed=task.param_seed)
        binary = b6.make_binary(continuous, float(params["gamma_XY"]),
                                np.random.default_rng([task.data_seed, 1]),
                                np.random.default_rng([task.param_seed, 2]))
    finally:
        b6.SIG_X_SPEC = original

    center_codes = np.asarray(continuous["env"])
    observed = ObservedDataView(
        features=np.asarray(continuous["F"]),
        treatment=np.asarray(binary["X_bin"]),
        outcome=np.asarray(binary["Y_bin"]),
        center_codes=center_codes,
        centers=np.unique(center_codes),
    )
    truth = {                                     # evaluation only, not an algorithm input
        "ate_do": float(binary["ate_do"]),
        "ate_k": np.asarray(binary["ate_k"]).tolist(),
        "n_k": np.asarray(binary["n_k"]).tolist(),
        "ate_do_nk": float(binary["ate_do_nk"]),
        "true_cluster": np.asarray(continuous["true_cluster"]).tolist(),
    }
    meta = {
        "task_id": task.task_id,
        "combo_id": task.combo_id,
        "param_rep": int(task.param_rep), "data_rep": int(task.data_rep),
        "param_seed": int(task.param_seed), "data_seed": int(task.data_seed),
        "algorithm_seed_base": int(task.algorithm_seed_base),
        "split_seed": int(task.split_seed), "candidate_seed": int(task.candidate_seed),
        "SIG_X_SPEC": sig_x_spec,
        "dgp_params": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in params.items()},
        "n_rows": int(observed.n_samples),
        "n_centers": int(observed.centers.size),
        "prevalence_x": float(np.mean(binary["X_bin"])),
        "prevalence_y": float(np.mean(binary["Y_bin"])),
    }
    return observed, truth, meta


def full_sample_config(task, unit: dict[str, Any] | None = None):
    """PipelineConfig of the full-sample mode with the registered solver and guard parameters."""
    unit = dict(REGISTERED_UNIT if unit is None else unit)
    _design, pipeline, model, cem_types = load_label_package()
    solver = unit["solver"]
    return pipeline.PipelineConfig(
        fit_config=model.FitConfig(
            support_size=int(solver["support_size"]),
            step_size=float(solver["step_size"]),
            max_iter=int(solver["max_iter"]),
            warmup_steps=int(solver["warmup_steps"]),
            parameter_tolerance=float(solver["parameter_tolerance"]),
            objective_relative_tolerance=float(solver["objective_relative_tolerance"]),
        ),
        cem_config=cem_types.CEMConfig(
            k_components=int(unit["cem"]["k_components"]),
            max_outer_iter=int(unit["cem"]["max_outer_iter"]),
        ),
        stability_threshold=float(unit["guard_stability_threshold"]),
        dataset_seed_base=int(task.algorithm_seed_base),
        split_seed=int(task.split_seed),
        candidate_seed=int(task.candidate_seed),
        epsilon_prob=float(unit["epsilon_prob"]),
        split_mode=str(unit["split_mode"]),
    )


def center_map_from_pipeline(pipeline_result, observed) -> tuple[dict[int, int] | None, dict]:
    """Frozen labels of the primary arm as center_to_cluster; (None, status) when they abstain or fail.

    The mapping follows the actual order of `observed.centers`: `frozen_labels` has one label per center,
    indexed in the center order of `dense_center_codes`.
    """
    primary = pipeline_result.primary
    status = {
        "pipeline_status": str(pipeline_result.status),
        "primary_arm": str(primary.arm),
        "primary_status": str(primary.status),
        "labels_frozen": bool(primary.labels_frozen),
        "fixed_refit_status": (None if primary.fixed_refit_status is None
                               else str(primary.fixed_refit_status)),
        "fixed_refit_eligible": bool(primary.fixed_refit_eligible),
        "fixed_refit_converged": bool(primary.fixed_refit_converged),
        "post_freeze_reclassification_count": primary.post_freeze_reclassification_count,
        "selected_logical_id": primary.selected_logical_id,
        "selected_direction": primary.selected_direction,
    }
    labels = primary.frozen_labels
    if labels is not None:
        # kept for inspection when the arm failed, but not used as a mapping downstream
        status["frozen_labels_audit_only"] = np.asarray(labels,
                                                        dtype=np.int64).tolist()
    if labels is None:
        status["center_map_available"] = False
        status["reason"] = primary.diagnostics.get("fixed_refit_failure_reason",
                                                   "no_frozen_labels")
        return None, status

    # `_arm_from_selection` keeps frozen_labels when fixed_refit_failed, for inspection; such an arm must not
    # proceed to screening and estimation. Eligibility follows `complete_max_iter_eligible` (convergence is
    # not required in addition).
    if str(primary.status) not in ELIGIBLE_ARM_STATUSES or not bool(
            primary.fixed_refit_eligible):
        status["center_map_available"] = False
        status["reason"] = (
            f"label_arm_not_eligible: status={primary.status}, "
            f"fixed_refit_eligible={bool(primary.fixed_refit_eligible)}, "
            f"fixed_refit_status={primary.fixed_refit_status}")
        return None, status

    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(observed.centers)
    if labels.shape != (centers.size,):
        status["center_map_available"] = False
        status["reason"] = (f"frozen_labels has shape {labels.shape}, number of centers "
                            f"{centers.size}; inconsistent metadata")
        return None, status
    mapping = {int(c.item() if isinstance(c, np.generic) else c): int(k)
               for c, k in zip(centers, labels)}
    status["center_map_available"] = True
    status["center_order"] = [int(c.item() if isinstance(c, np.generic) else c)
                              for c in centers]
    status["frozen_labels"] = labels.tolist()
    status["cluster_sizes"] = {int(k): int(np.sum(labels == k))
                               for k in sorted(set(labels.tolist()))}
    return mapping, status


def split_index_report(observed, config) -> dict[str, Any]:
    """Index checks of the full-sample mode: coverage and the A/B exclusivity identities."""
    _design, pipeline, _model, _cem = load_label_package()
    split = pipeline._build_split(observed, config)
    n = observed.n_samples
    all_rows = np.arange(n, dtype=np.int64)
    a, b = np.asarray(split.a), np.asarray(split.b)
    return {
        "split_mode": config.split_mode,
        "n_rows": int(n),
        "label_rows": int(split.label.size),
        "screening_rows": int(split.screening.size),
        "effect_rows": int(split.effect.size),
        "a_rows": int(a.size), "b_rows": int(b.size),
        "label_covers_all_rows": bool(np.array_equal(np.asarray(split.label), all_rows)),
        "screening_covers_all_rows": bool(np.array_equal(np.asarray(split.screening), all_rows)),
        "effect_covers_all_rows": bool(np.array_equal(np.asarray(split.effect), all_rows)),
        "a_b_disjoint": bool(np.intersect1d(a, b, assume_unique=True).size == 0),
        "a_b_union_equals_label": bool(np.array_equal(np.union1d(a, b),
                                                      np.asarray(split.label))),
        "diagnostics": dict(split.diagnostics),
    }


__all__ = ["REGISTERED_UNIT", "ELIGIBLE_ARM_STATUSES", "STEP6_DIR", "VENDOR_DIR", "load_label_package",
           "load_dgp", "registered_task", "generate_observed", "full_sample_config",
           "center_map_from_pipeline", "split_index_report"]
