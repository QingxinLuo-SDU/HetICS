"""label_stage.py — center labels learned on all observations, and the support sets of the label model.

The label algorithm is `bniseam_test.multiview_pipeline.run_label_pipeline` with `split_mode="full_sample"`
(AB/BA, guard, freezing and refitting unchanged); eligible arms are `complete` and
`complete_max_iter_eligible`.

Support sets: `primary.parameters.beta` has shape (K, p) and is hard-thresholded per cluster to
support_size=25, with column indices of the original F, so
    S_label,g = np.nonzero(beta[g])[0]
The rows of beta are the values of `frozen_labels`; nothing is reordered or adjusted by the truth.

Failures:
  * eligible labels but `parameters is None` or a wrong shape -> support_unavailable (the partition is used);
  * ineligible labels -> center_map=None; no estimate that needs the partition is available, and
    frozen_labels is kept for inspection only.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

import label_bridge as lb


@dataclasses.dataclass(frozen=True)
class AlgorithmTask:
    """The label pipeline needs three integer seeds."""
    algorithm_seed_base: int
    split_seed: int
    candidate_seed: int


def algorithm_task(point_id: str, param_rep: int, data_rep: int) -> AlgorithmTask:
    import design_table as dt
    design = dt._design_module()
    base = dt.algorithm_seed_base(point_id, param_rep, data_rep)
    return AlgorithmTask(
        algorithm_seed_base=base,
        split_seed=int(design.derive_algorithm_child_seed(base, stage="split")),
        candidate_seed=int(design.derive_algorithm_child_seed(base, stage="candidate")),
    )


LABEL_UNIT: dict[str, Any] = {
    "solver": {"support_size": 25, "step_size": 1.0, "max_iter": 300,
               "warmup_steps": 50, "parameter_tolerance": 0.001,
               "objective_relative_tolerance": 0.00001},
    "cem": {"k_components": 3, "max_outer_iter": 4},
    "guard_stability_threshold": 0.80,
    "epsilon_prob": 0.01,
    "split_mode": "full_sample",
}


def learn_labels(observed, task: AlgorithmTask, *, k_components: int = 3) -> dict:
    """One run of the full-sample label pipeline; returns center_map, status and support sets."""
    _design, pipeline, _model, _cem = lb.load_label_package()
    unit = {**LABEL_UNIT, "cem": {**LABEL_UNIT["cem"], "k_components": int(k_components)}}
    config = lb.full_sample_config(task, unit)
    out: dict[str, Any] = {"label_config": {
        "split_mode": config.split_mode,
        "support_size": int(unit["solver"]["support_size"]),
        "max_iter": int(unit["solver"]["max_iter"]),
        "k_components": int(unit["cem"]["k_components"]),
        "guard_stability_threshold": float(unit["guard_stability_threshold"]),
        "epsilon_prob": float(unit["epsilon_prob"]),
        "algorithm_seed_base": int(task.algorithm_seed_base),
        "split_seed": int(task.split_seed),
        "candidate_seed": int(task.candidate_seed),
    }}
    t0 = time.perf_counter()
    try:
        result = pipeline.run_label_pipeline(observed, config)
    except Exception as error:                                   # noqa: BLE001
        out.update(center_map=None, support_by_cluster=None,
                   label_status={"pipeline_status": "exception",
                                 "reason": f"{type(error).__name__}: {error}"},
                   support_status="label_unavailable",
                   elapsed=time.perf_counter() - t0)
        return out
    elapsed = time.perf_counter() - t0
    center_map, status = lb.center_map_from_pipeline(result, observed)
    out.update(center_map=center_map, label_status=status, elapsed=elapsed)

    primary = result.primary
    status["cache_diagnostics"] = _cache_diagnostics(result.cache_diagnostics)
    if center_map is None:
        out["support_by_cluster"] = None
        out["support_status"] = "label_unavailable"
        return out

    support, support_status = export_support(primary, observed)
    out["support_by_cluster"] = support
    out["support_status"] = support_status
    return out


_CACHE_COUNT_FIELDS = (
    "unique_partition_count",
    "refinement_fit_count",
    "expensive_fit_count",
    "pre_validation_fixed_fit_count",
    "final_fixed_refit_count",
    "total_parameter_fit_count",
)


def _cache_diagnostics(cache) -> dict[str, Any]:
    """All counts of `CacheDiagnostics`.

    The counts do not add up:
      * `expensive_fit_count == refinement_fit_count` (two names of the same work);
      * `total_parameter_fit_count = refinement + pre_validation + final`.
    `by_direction` is a nested diagnostic object per direction; it is an empty dict when missing.
    """
    out: dict[str, Any] = {}
    for name in _CACHE_COUNT_FIELDS:
        value = getattr(cache, name, None)
        out[name] = None if value is None else int(value)
    directions: dict[str, dict[str, int]] = {}
    for name, item in dict(getattr(cache, "by_direction", None) or {}).items():
        to_dict = getattr(item, "to_dict", None)
        raw = to_dict() if callable(to_dict) else dict(item)
        directions[str(name)] = {str(key): int(value) for key, value in raw.items()}
    out["by_direction"] = directions
    return out


def export_support(primary, observed) -> tuple[dict[int, list[int]] | None, str]:
    """S_label,g = nonzero original F columns of row g of the refitted parameter beta."""
    params = getattr(primary, "parameters", None)
    if params is None:
        return None, "support_unavailable_no_parameters"
    beta = np.asarray(getattr(params, "beta", None), dtype=float)
    if beta.ndim != 2:
        return None, "support_unavailable_bad_beta_shape"
    p = int(np.asarray(observed.features).shape[1])
    if beta.shape[1] != p:
        return None, (f"support_unavailable_beta_columns_{beta.shape[1]}_vs_F_{p}")
    labels = np.asarray(primary.frozen_labels, dtype=np.int64)
    present = sorted(set(int(v) for v in labels.tolist()))
    if max(present) >= beta.shape[0]:
        return None, "support_unavailable_label_row_mismatch"
    support = {int(g): sorted(int(j) for j in np.flatnonzero(beta[g])) for g in present}
    return support, "complete"


__all__ = ["AlgorithmTask", "algorithm_task", "LABEL_UNIT", "learn_labels",
           "export_support"]
