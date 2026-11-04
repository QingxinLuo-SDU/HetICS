"""run_formal.py — runner of the formal binary simulations.

One base group (structure x param_rep x data_rep) is one parallel task: the continuous base data are
generated once, every point sharing the structure is run, and the data are released.

Steps of one point:
  1. binary treatment and outcome (with or without U) and the evaluation truth;
  2. center labels learned on all observations, giving a frozen partition;
  3. candidate and adjustment sets of the row's execution_methods, each computed once;
  4. RA, AIPW and PSM per method and analysis cluster (PSM reuses the RA outcome fits);
  5. evaluation against the truth of the members each method actually targets, per cluster and overall;
  6. atomic write of the unit record.

The truth enters only the evaluation; U is not available to any arm, including the oracle.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
import zlib
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import arms                                                      # noqa: E402
import design_table as dt                                        # noqa: E402
import evaluation as ev                                          # noqa: E402
import formal_dgp                                                # noqa: E402
import label_stage                                               # noqa: E402
from aipw_sandwich import fit_partition_aipw                     # noqa: E402
from psm_ate import fit_partition_psm                            # noqa: E402
from regression_adjustment import fit_partition_ra               # noqa: E402
from screening_bridge import cluster_members, load_screening     # noqa: E402

OUTPUT_VERSION = dt.IMPLEMENTATION_VERSION
ESTIMATOR_ORDER = ("ra_standardized", "aipw", "psm_caliper")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {(str(k) if not isinstance(k, str) else k): to_jsonable(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def unit_path(out_dir: str, row: dict, param_rep: int, data_rep: int) -> str:
    key = dt.unit_key(row, param_rep, data_rep)
    return os.path.join(out_dir, "units", row["point_id"], f"{key}.json")


def write_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False)
        # flush before renaming, so that a power loss cannot leave an empty JSON behind
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def unit_is_complete(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (record.get("unit_complete") is True
            and record.get("output_version") == OUTPUT_VERSION)


def write_shard_markers(out_dir: str, rows: list[dict], param_reps, data_reps,
                        planned_data_reps: int) -> dict:
    """A shard is point_id x param_rep over the planned data_reps.

    complete is written only when every planned data_rep is stored and complete.
    """
    marker_dir = os.path.join(out_dir, "shard_complete")
    os.makedirs(marker_dir, exist_ok=True)
    written, partial = 0, 0
    for row in rows:
        for param_rep in param_reps:
            keys = [dt.unit_key(row, param_rep, d) for d in range(1, planned_data_reps + 1)]
            paths = [os.path.join(out_dir, "units", row["point_id"], f"{k}.json")
                     for k in keys]
            if all(unit_is_complete(p) for p in paths):
                write_atomic(os.path.join(marker_dir,
                                          f"{row['point_id']}__p{param_rep:03d}.json"),
                             {"point_id": row["point_id"], "param_rep": param_rep,
                              "data_reps": list(range(1, planned_data_reps + 1)),
                              "output_version": OUTPUT_VERSION,
                              "unit_complete": True})
                written += 1
            else:
                partial += 1
    return {"shard_complete_written": written, "shard_incomplete": partial,
            "planned_data_reps": planned_data_reps}


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def run_estimators(F, X, Y, env, center_map, selected_by_cluster, *, backend):
    """RA, AIPW and PSM; PSM reuses only the RA outcome fits of the same design, selection, partition and rows."""
    out: dict[str, dict] = {}
    t0 = time.perf_counter()
    ra = fit_partition_ra(F, X, Y, env, center_map, selected_by_cluster, backend=backend)
    ra_elapsed = time.perf_counter() - t0
    outcome_fits = ra.pop("outcome_fits", {})
    ra["elapsed"] = ra_elapsed
    out["ra_standardized"] = ra

    t0 = time.perf_counter()
    aipw = fit_partition_aipw(F, X, Y, env, center_map, selected_by_cluster,
                              backend=backend)
    aipw["elapsed"] = time.perf_counter() - t0
    aipw["estimator"] = "aipw"
    out["aipw"] = aipw

    t0 = time.perf_counter()
    psm = fit_partition_psm(F, X, Y, env, center_map, selected_by_cluster,
                            backend=backend, outcome_fits=outcome_fits)
    psm["elapsed"] = time.perf_counter() - t0
    psm["outcome_model_shared_with_ra"] = bool(outcome_fits)
    out["psm_caliper"] = psm
    return out


def matched_rows_by_cluster(psm: dict) -> dict[int, np.ndarray]:
    detail = psm.get("matching_detail") or {}
    out: dict[int, np.ndarray] = {}
    for k, block in detail.items():
        row_ids = np.asarray(block["row_id"], dtype=np.int64)
        matched = np.asarray(block["matched"], dtype=bool)
        out[int(k)] = row_ids[matched]
    return out


# ---------------------------------------------------------------------------
# One analysis unit
# ---------------------------------------------------------------------------
def run_unit(continuous, row: dict, group: dict, structures: dict, m6, b6,
             config: dict, u_vec, u_meta) -> tuple[dict, dict]:
    point_id = row["point_id"]
    param_rep, data_rep = int(group["param_rep"]), int(group["data_rep"])
    backend = config["logistic_backend"]
    wanted = set(dt.methods_of(row))
    timings: dict[str, float] = {}
    record: dict[str, Any] = {
        "output_version": OUTPUT_VERSION,
        "design_revision": row["design_revision"],
        "unit_key": dt.unit_key(row, param_rep, data_rep),
        "point_id": point_id, "scenario": row["scenario"],
        "tier": row["tier"], "axis_membership": row["axis_membership"],
        "report_panels": dt.panels_of(row),
        "calibration_member": row["calibration_member"] == "true",
        "param_rep": param_rep, "data_rep": data_rep,
        "gamma_logor": row["gamma_logor"],
        "structure_id": group["structure_id"],
        "execution_methods": sorted(wanted),
        "estimators": list(ESTIMATOR_ORDER),
        "assumptions_not_covered": list(arms.ASSUMPTIONS_NOT_COVERED),
        "unit_complete": False,
        "errors": [],
    }

    t_unit = time.perf_counter()
    t0 = time.perf_counter()
    observed, truth, unit_meta = formal_dgp.binary_unit(
        continuous, row, group, b6, u=u_vec)
    timings["binary_transform"] = time.perf_counter() - t0
    record["unit_meta"] = unit_meta
    record["latent_confounder_diagnostics"] = (u_meta if row["has_unmeasured_U"] == "true"
                                               else None)

    F = np.asarray(observed.features, dtype=float)
    X = np.asarray(observed.treatment).astype(int)
    Y = np.asarray(observed.outcome).astype(int)
    env = np.asarray(observed.center_codes)
    p = int(F.shape[1])
    record["observed_shape"] = {"n_rows": int(F.shape[0]), "p": p,
                                "n_centers": int(np.unique(env).size)}

    # --- labels ---
    t0 = time.perf_counter()
    task = label_stage.algorithm_task(point_id, param_rep, data_rep)
    label = label_stage.learn_labels(observed, task,
                                     k_components=int(row["K"]))
    timings["label"] = time.perf_counter() - t0
    center_map = label["center_map"]
    record["label_status"] = label["label_status"]
    record["label_config"] = label["label_config"]
    record["support_status"] = label["support_status"]
    record["support_by_cluster"] = label["support_by_cluster"]

    members = cluster_members(env, center_map) if center_map is not None else {}
    members_rows = {int(g): np.flatnonzero(np.isin(env, mem))
                    for g, mem in members.items()}
    record["cluster_diagnostics"] = {
        "center_map": (None if center_map is None
                       else {str(k): int(v) for k, v in center_map.items()}),
        "cluster_sizes": {str(g): int(r.size) for g, r in members_rows.items()},
        "n_analysis_clusters": len(members_rows),
        "single_center_clusters": [int(g) for g, mem in members.items()
                                   if len(mem) <= 1],
    }
    if center_map is not None:
        from sklearn.metrics import adjusted_rand_score
        centers = [int(c) for c in np.unique(env)]
        true_by_center = np.asarray(truth["true_cluster"])
        if true_by_center.size == len(centers):
            record["cluster_diagnostics"]["ari"] = float(adjusted_rand_score(
                true_by_center, [center_map[c] for c in centers]))

    # --- truth targets ---
    truth_layer = {"true_cluster_targets": ev.true_cluster_targets(truth),
                   "ate_do": truth["ate_do"], "ate_k": truth["ate_k"],
                   "n_k": truth["n_k"], "gamma_logor": truth["gamma_logor"],
                   "observed_core_by_cluster": {str(k): v for k, v in
                                                truth["observed_core_by_cluster"].items()},
                   "a_safe": truth["a_safe"],
                   "full_causal_core_note": truth["full_causal_core_note"]}
    record["truth"] = truth_layer

    alignment = None
    if members_rows:
        alignment = arms.align_clusters_to_truth(
            members_rows, truth["true_cluster_sample"], int(row["K"]))
        record["cluster_alignment"] = alignment

    # --- candidate and adjustment sets ---
    selections: dict[str, dict] = {}
    t0 = time.perf_counter()
    need_fixedx = bool(wanted & set(arms.FIXEDX_ARMS))
    fixedx = None
    if need_fixedx and center_map is not None:
        import screening_comparison as sc
        try:
            fixedx = sc.screen_family(F, X, Y, env, center_map, "fixedx_bic",
                                      m6=m6, task=task, config=config)
        except Exception as error:                               # noqa: BLE001
            record["errors"].append(f"fixedx_family: {type(error).__name__}: {error}")
    timings["fixedx_family"] = time.perf_counter() - t0

    for name in arms.FIXEDX_ARMS:
        if name not in wanted:
            continue
        if fixedx is None:
            selections[name] = {"status": "unavailable",
                                "reason": ("upstream_abstain" if center_map is None
                                           else "fixedx_family_failed"),
                                "selected_by_cluster": None}
            continue
        block = fixedx["directions"][name]
        selections[name] = {
            "status": block["status"],
            "selected_by_cluster": {int(k): v for k, v in
                                    block["selected_by_cluster"].items()},
            "pre_icp_by_cluster": {int(k): v for k, v in
                                   block["pre_icp_by_cluster"].items()},
            "errors_by_cluster": block.get("errors_by_cluster", {}),
            "candidate_definition": {"xside_icp": "fixed_x_bic_S_X",
                                     "yside_icp": "fixed_x_bic_S_Y",
                                     "dual_icp": "fixed_x_bic_S_X_union_S_Y"}[name],
            "icp": "official_once",
        }

    # oracle arm
    for name in arms.ORACLE_ARMS:
        if name not in wanted:
            continue
        selections[name] = _oracle_selection(name, truth, members_rows, alignment)

    # oracle_reference (true partition and true observed core)
    reference_map = None
    if arms.REFERENCE_ARM in wanted:
        reference_map, ref_sel = _reference_selection(truth, env)
        selections[arms.REFERENCE_ARM] = ref_sel

    record["selections"] = {k: {kk: vv for kk, vv in v.items()
                                if kk != "selected_by_cluster"}
                            for k, v in selections.items()}
    record["selected_by_cluster"] = {
        k: (None if v.get("selected_by_cluster") is None
            else {str(g): list(s) for g, s in v["selected_by_cluster"].items()})
        for k, v in selections.items()}

    # --- selection metrics ---
    record["feature_selection"] = evaluate_selections(selections, truth, p, alignment)

    # --- estimation ---
    effects, estimator_diag, matching_arrays, est_timings, est_errors = estimate_selections(
        F, X, Y, env, truth, selections, center_map, reference_map, members_rows,
        backend=backend, save_matching=config.get("save_matching_detail") == "full")
    timings.update(est_timings)
    record["errors"].extend(est_errors)
    record["effects"] = effects
    record["estimator_diagnostics"] = estimator_diag
    timings["unit_total"] = time.perf_counter() - t_unit
    record["timings"] = timings
    record["unit_complete"] = True
    return record, matching_arrays


def evaluate_selections(selections: dict, truth: dict, p: int, alignment) -> dict:
    """Recall and false inclusion by covariate family, per arm and analysis cluster."""
    feature_eval: dict[str, Any] = {}
    for name, block in selections.items():
        sel_by_cluster = block.get("selected_by_cluster")
        if sel_by_cluster is None:
            continue
        partition_align = (alignment if name != arms.REFERENCE_ARM else None)
        per_cluster = {}
        for g, sel in sel_by_cluster.items():
            if name == arms.REFERENCE_ARM:
                own = int(g)                       # true partition: the cluster label is the true cluster
            elif partition_align is not None:
                own = partition_align["mapping"].get(int(g))
            else:
                own = None
            per_cluster[str(g)] = ev.selection_metrics(sel, truth, p, own)
        feature_eval[name] = per_cluster
    return feature_eval


def estimate_selections(F, X, Y, env, truth, selections, center_map, reference_map,
                        members_rows, *, backend, save_matching=False):
    """RA, AIPW and PSM per arm, evaluated against the members each arm actually targets."""
    timings: dict[str, float] = {}
    errors: list[str] = []
    effects: dict[str, Any] = {}
    estimator_diag: dict[str, Any] = {}
    matching_arrays: dict[str, dict] = {}
    full_targets = (ev.cluster_targets(truth, members_rows) if members_rows
                    else {"by_cluster": {}, "sizes": {}, "overall": truth["ate_do"],
                          "scope": "analysis_cluster_full_members"})
    for name, block in selections.items():
        sel_by_cluster = block.get("selected_by_cluster")
        use_map = reference_map if name == arms.REFERENCE_ARM else center_map
        if sel_by_cluster is None or use_map is None:
            effects[name] = {est: [{"estimator": est, "stage": "overall",
                                    "analysis_cluster": -1,
                                    "point_status": "unavailable",
                                    "se_status": "unavailable",
                                    "reason": block.get("reason", "upstream_abstain"),
                                    "ci_valid": False, "cover_true": None,
                                    "cover_zero": None, "reject_zero": None}]
                             for est in ESTIMATOR_ORDER}
            continue
        t0 = time.perf_counter()
        try:
            results = run_estimators(F, X, Y, env, use_map, sel_by_cluster,
                                     backend=backend)
        except Exception as error:                               # noqa: BLE001
            errors.append(f"estimators[{name}]: {type(error).__name__}: {error}")
            effects[name] = {est: [{"estimator": est, "stage": "overall",
                                    "analysis_cluster": -1,
                                    "point_status": "unavailable",
                                    "se_status": "unavailable",
                                    "reason": f"{type(error).__name__}: {error}",
                                    "ci_valid": False, "cover_true": None,
                                    "cover_zero": None, "reject_zero": None}]
                             for est in ESTIMATOR_ORDER}
            continue
        timings[f"estimators::{name}"] = time.perf_counter() - t0

        arm_members = (members_rows if name != arms.REFERENCE_ARM
                       else {int(g): np.flatnonzero(np.isin(env, mem))
                             for g, mem in cluster_members(env, use_map).items()})
        arm_full = ev.cluster_targets(truth, arm_members)
        rows_out: dict[str, Any] = {}
        for est in ESTIMATOR_ORDER:
            res = results[est]
            if est == "psm_caliper":
                retained = matched_rows_by_cluster(res)
                targets = ev.retained_targets(truth, retained, res.get("weights"),
                                              res.get("cluster_order"))
                different = not bool(res.get("full_ate_comparable"))
            else:
                targets = arm_full
                different = False
            rows_out[est] = ev.effect_rows(res, est, truth, targets,
                                           full_targets=arm_full,
                                           target_different=different)
        effects[name] = rows_out
        estimator_diag[name] = {
            est: {"status": results[est].get("status"),
                  "point_status": results[est].get("point_status"),
                  "se_status": results[est].get("se_status"),
                  "estimand": results[est].get("estimand"),
                  "elapsed": results[est].get("elapsed"),
                  "errors_by_cluster": results[est].get("errors_by_cluster"),
                  "se_reasons_by_cluster": results[est].get("se_reasons_by_cluster"),
                  "diagnostics": {str(k): v for k, v in
                                  (results[est].get("diagnostics") or {}).items()},
                  # full covariance (K x K) kept so that a'Va can be recomputed from the record
                  "V": results[est].get("V"),
                  "cluster_order": results[est].get("cluster_order"),
                  "weights": results[est].get("weights")}
            for est in ESTIMATOR_ORDER}
        if save_matching:
            matching_arrays[name] = results["psm_caliper"].get("matching_detail") or {}
    return effects, estimator_diag, matching_arrays, timings, errors


def _oracle_selection(name, truth, members_rows, alignment) -> dict:
    if not members_rows:
        return {"status": "unavailable", "reason": "upstream_abstain",
                "selected_by_cluster": None}
    base = sorted(set(truth["idx_Z"]) | set(truth["true_shared_causal"]))
    out: dict[int, list[int]] = {}
    for g in members_rows:
        true_k = alignment["mapping"].get(int(g)) if alignment else None
        spec = truth["true_cluster_specific_causal"].get(int(true_k), []) \
            if true_k is not None else []
        out[int(g)] = sorted(set(base) | set(spec))
    return {"status": "complete", "selected_by_cluster": out,
            "is_oracle": True, "icp": "none",
            "adjustment_role": "aligned_own_cluster_core",
            "alignment_rule": alignment["rule"] if alignment else None,
            "caveat": ("no guarantee of sufficient adjustment under a wrong partition; with U this is an "
                       "observed-covariate oracle"),
            "partition_role": "learned_partition"}


def _reference_selection(truth, env) -> tuple[dict, dict]:
    centers = [int(c) for c in np.unique(env)]
    true_by_center = np.asarray(truth["true_cluster"]).tolist()
    if len(true_by_center) != len(centers):
        return None, {"status": "unavailable",
                      "reason": f"true_cluster has length {len(true_by_center)} != number of centers "
                                f"{len(centers)}", "selected_by_cluster": None}
    mapping = {c: int(k) for c, k in zip(centers, true_by_center)}
    core = truth["observed_core_by_cluster"]
    selected = {int(k): list(v) for k, v in core.items()}
    present = sorted(set(mapping.values()))
    missing = [k for k in present if k not in selected]
    if missing:
        return None, {"status": "unavailable",
                      "reason": f"no core for true clusters {missing}", "selected_by_cluster": None}
    return mapping, {"status": "complete",
                     "selected_by_cluster": {k: selected[k] for k in present},
                     "is_oracle": True, "icp": "none",
                     "partition_role": "true_partition",
                     "adjustment_role": "true_observed_core",
                     "caveat": "removes label and selection errors only; no guarantee of a correct "
                               "working model or 95% coverage; not used with U"}


# ---------------------------------------------------------------------------
# Base groups
# ---------------------------------------------------------------------------
def run_base_group(payload) -> dict:
    group, out_dir, config_path, stage = payload
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    config = load_config(config_path)
    rows = {r["point_id"]: r for r in formal_rows(stage)}
    structures = dt.build_structures(list(rows.values()))
    param_reps = config["param_reps"]
    data_reps = config["data_reps"]
    summary = {"base_group_key": group["base_group_key"], "units": {},
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    todo = [rows[pid] for pid in group["point_ids"]
            if not unit_is_complete(unit_path(out_dir, rows[pid],
                                              group["param_rep"], group["data_rep"]))]
    if not todo:
        summary["status"] = "already_complete"
        return summary

    t0 = time.perf_counter()
    try:
        m6 = load_screening(formal_dgp.STEP6_DIR)
        # The BH threshold of the ICP comes from the configuration (1e-3). It is a module constant of the
        # vendored code, so every worker sets it after its own load_screening.
        _alpha_icp = float(config["icp_alpha"])
        if not (0.0 < _alpha_icp <= 0.05):
            raise RuntimeError("icp_alpha=%r is outside (0, 0.05]" % _alpha_icp)
        m6.ALPHA_ICP = _alpha_icp
        _gen, b6 = formal_dgp.load_sources()
        continuous, base_meta = formal_dgp.generate_base(group, structures)
    except Exception as error:                                   # noqa: BLE001
        summary["status"] = "base_failed"
        summary["reason"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        return summary
    summary["base_elapsed"] = time.perf_counter() - t0
    summary["base_meta"] = base_meta

    u_vec = u_meta = None
    if any(rows[pid]["has_unmeasured_U"] == "true" for pid in group["point_ids"]):
        rho_block = float(structures[group["structure_id"]]["params"]["rho_block"])
        u_vec, _S, u_meta = formal_dgp.unmeasured_u_for_base(continuous, group, rho_block)

    for row in todo:
        key = dt.unit_key(row, group["param_rep"], group["data_rep"])
        path = unit_path(out_dir, row, group["param_rep"], group["data_rep"])
        try:
            record, matching = run_unit(continuous, row, group, structures, m6, b6,
                                        config, u_vec, u_meta)
            record["base_meta"] = base_meta
            if matching:
                mpath = os.path.join(out_dir, "matching", row["point_id"],
                                     f"{key}.npz")
                _save_matching(mpath, matching)
                record["matching_detail_path"] = os.path.relpath(mpath, out_dir)
            write_atomic(path, record)
            summary["units"][key] = {"status": "complete",
                                     "elapsed": record["timings"]["unit_total"],
                                     "errors": record["errors"]}
        except Exception as error:                               # noqa: BLE001
            failure = {"output_version": OUTPUT_VERSION, "unit_key": key,
                       "point_id": row["point_id"], "unit_complete": False,
                       "status": "failed",
                       "reason": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()}
            write_atomic(path, failure)
            summary["units"][key] = {"status": "failed", "reason": failure["reason"]}
    del continuous, u_vec
    gc.collect()
    summary["status"] = "done"
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["elapsed"] = time.perf_counter() - t0
    return summary


def _save_matching(path: str, matching: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for method, by_cluster in matching.items():
        for k, block in (by_cluster or {}).items():
            tag = f"{method}__k{k}"
            payload[f"{tag}__row_id"] = np.asarray(block["row_id"], dtype=np.int32)
            payload[f"{tag}__matched"] = np.asarray(block["matched"], dtype=bool)
            payload[f"{tag}__match_row_id"] = np.asarray(
                [-1 if v is None else int(v) for v in block["match_row_id"]],
                dtype=np.int32)
            payload[f"{tag}__lp"] = np.asarray(block["lp"], dtype=np.float32)
            payload[f"{tag}__reuse_K"] = np.asarray(block["reuse_count_K"], dtype=np.int32)
            payload[f"{tag}__c_weight"] = np.asarray(block["c_weight"], dtype=np.float32)
            payload[f"{tag}__treatment"] = np.asarray(block["treatment"], dtype=np.int8)
            payload[f"{tag}__center"] = np.asarray(block["center"], dtype=np.int32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def effective_workers(requested: int, n_tasks: int) -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(int(requested), int(n_tasks), cpu))


def formal_rows(stage: str = "r2") -> list[dict]:
    """Use the original verified design tables and canonical structure seeds."""
    if stage == "r2":
        return dt.load_rows()
    if stage != "r1":
        raise ValueError(f"Unknown stage: {stage}")
    from frozen_rerun import STAGES, stage_rows
    rows = stage_rows()
    return [rows[pid] for pid in STAGES["r1"]["points"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Runner of the formal binary simulations")
    parser.add_argument("--stage", choices=("r1", "r2"), default="r2")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--points", default=None,
                        help="comma-separated subset of point_id")
    parser.add_argument("--param-reps", default=None, help="e.g. 1-10 or 1,2,3")
    parser.add_argument("--data-reps", default=None)
    parser.add_argument("--static-only", action="store_true",
                        help="run the design checks only, without the DGP")
    parser.add_argument("--verify-complete", action="store_true",
                        help="count completed units without running; exit code 5 if any is missing")
    args = parser.parse_args(argv)

    args.config = args.config or os.path.join(dt.CONFIG_DIR, "formal_v5.json" if args.stage == "r1" else "formal_r2.json")
    config = load_config(args.config)
    expected_revision = dt.ROUND1_DESIGN_REVISION if args.stage == "r1" else dt.DESIGN_REVISION
    if config["design_revision"] != expected_revision:
        parser.error("Configuration design_revision does not match --stage.")
    rows = formal_rows(args.stage)
    checks = dt.static_checks(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    write_atomic(os.path.join(args.out_dir, "static_checks.json"), checks)
    if checks["status"] != "ok":
        print("Design checks failed:", checks["problems"], file=sys.stderr)
        return 2
    print(f"[static] {checks['n_rows']} rows, {checks['n_structures']} base structures, "
          f"{checks['planned_estimator_attempts_total']:,} planned attempts: passed")
    if args.static_only:
        return 0

    param_reps = _parse_reps(args.param_reps) or config["param_reps"]
    data_reps = _parse_reps(args.data_reps) or config["data_reps"]
    # Structure identities and seeds always come from the full table; --points only selects the points to
    # run, so a point gets the same seeds in a subset run and in a full run.
    groups = dt.build_base_groups(rows, list(param_reps), list(data_reps))
    if args.points:
        keep = {p.strip() for p in args.points.split(",") if p.strip()}
        unknown = keep - {r["point_id"] for r in rows}
        if unknown:
            print(f"--points contains unknown point_id: {sorted(unknown)}", file=sys.stderr)
            return 2
        rows = [r for r in rows if r["point_id"] in keep]
        filtered = []
        for group in groups:
            points = [p for p in group["point_ids"] if p in keep]
            if points:
                filtered.append({**group, "point_ids": points})
        groups = filtered
        if not groups:
            print("--points matches no row", file=sys.stderr)
            return 2
    if args.verify_complete:
        return verify_complete(args.out_dir, rows, list(param_reps), list(data_reps))

    pending = dt.unresolved_points(rows)
    if pending:
        print(f"[fatal] gamma of these points is not frozen yet ({dt.GAMMA_SMALL_FROZEN_PATH} is missing): "
              f"{pending}. Run src/calibrate_gamma_small.py first or remove them from --points",
              file=sys.stderr)
        return 7
    # Fixed shuffle by crc32 of base_group_key, so that heavy structures do not run at the same time.
    # The order does not depend on completion; resumed runs skip completed units.
    groups.sort(key=lambda g: zlib.crc32(g["base_group_key"].encode("utf-8")))
    workers = effective_workers(args.workers or config["requested_workers"], len(groups))
    print(f"[plan] {len(rows)} rows x param{list(param_reps)} x data{list(data_reps)}"
          f" -> {len(groups)} base groups, workers={workers}")

    from record_environment import main as record_env
    try:
        record_env([os.path.join(args.out_dir, "environment.json")])
    except SystemExit:
        pass
    except Exception as error:                                   # noqa: BLE001
        print(f"[warn] environment record failed: {error}", file=sys.stderr)

    manifest = {"output_version": OUTPUT_VERSION,
                "design_revision": expected_revision, "stage": args.stage,
                "run_table_sha256": dt.sha256_of(os.path.join(dt.CONFIG_DIR, "formal_run_table_v5_main500.csv")
                                                 if args.stage == "r1" else dt.RUN_TABLE_PATH),
                "gamma_small_frozen": dt.read_gamma_small() if args.stage == "r2" else None,
                "config": config, "n_rows": len(rows), "n_groups": len(groups),
                "param_reps": list(param_reps), "data_reps": list(data_reps),
                "workers": workers,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_atomic(os.path.join(args.out_dir, "manifest.json"), manifest)

    payloads = [(g, args.out_dir, args.config, args.stage) for g in groups]
    t0 = time.perf_counter()
    summaries = []
    if workers == 1:
        for payload in payloads:
            summaries.append(run_base_group(payload))
            print(f"[done] {summaries[-1]['base_group_key']} "
                  f"{summaries[-1].get('status')}", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for summary in pool.imap_unordered(run_base_group, payloads):
                summaries.append(summary)
                print(f"[done] {summary['base_group_key']} {summary.get('status')}",
                      flush=True)
    elapsed = time.perf_counter() - t0

    markers = write_shard_markers(args.out_dir, rows, list(param_reps),
                                  list(data_reps), int(config["planned_data_reps"]))
    n_complete = sum(1 for s in summaries for v in s.get("units", {}).values()
                     if v["status"] == "complete")
    n_failed = sum(1 for s in summaries for v in s.get("units", {}).values()
                   if v["status"] == "failed")
    base_failed = sum(summary.get("status") == "base_failed" for summary in summaries)
    manifest["base_groups_failed"] = base_failed
    manifest.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "elapsed_seconds": elapsed,
                     "wall_clock_seconds": elapsed,
                     "cpu_seconds_upper_bound": elapsed * workers,
                     "cpu_time_note": "cpu_seconds_upper_bound = wall clock x workers, "
                                      "an upper bound, not measured usage",
                     "units_complete": n_complete, "units_failed": n_failed,
                     "group_summaries": summaries, "shard_markers": markers})
    write_atomic(os.path.join(args.out_dir, "manifest.json"), manifest)
    print(f"[end] {n_complete} units complete, {n_failed} failed, "
          f"wall clock {elapsed:.1f}s, {markers}")
    return 0 if n_failed == 0 and base_failed == 0 else 1


def verify_complete(out_dir: str, rows: list[dict], param_reps: list[int],
                    data_reps: list[int]) -> int:
    """Completeness check: counts units with unit_complete=True (failure records are also .json files)."""
    total_missing, total_failed, report = 0, 0, {}
    for row in rows:
        done = failed = 0
        for p in param_reps:
            for d in data_reps:
                path = unit_path(out_dir, row, p, d)
                if unit_is_complete(path):
                    done += 1
                elif os.path.isfile(path):
                    failed += 1
        expect = len(param_reps) * len(data_reps)
        report[row["point_id"]] = {"complete": done, "failed_or_corrupt": failed,
                                   "expected": expect}
        total_missing += expect - done
        total_failed += failed
        print(f"[verify] {row['point_id']:5s} {done:4d}/{expect}"
              + (f"  (failed or corrupt: {failed})" if failed else ""), flush=True)
    write_atomic(os.path.join(out_dir, "verify_complete.json"),
                 {"points": report, "missing": total_missing, "failed_or_corrupt": total_failed,
                  "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if total_missing:
        print(f"[verify] incomplete: {total_missing} units missing ({total_failed} failed or corrupt)",
              flush=True)
        return 5
    print("[verify] all units complete", flush=True)
    return 0


def _parse_reps(text: str | None):
    if not text:
        return None
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(v) for v in text.split(",") if v.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
