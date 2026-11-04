"""summarize_formal.py — long tables and reports of a completed run directory.

Denominators:
  * `n_planned_stage` = planned replications of the current run; replications not executed yet are not
    counted as failures;
  * `n_attempted`, `n_valid_ci` and `n_failed` are reported separately, with the availability rates;
  * the MCSE uses the actual parameter blocks: data replications are summarized within a block first, then
    across blocks.

The two coverages are never merged; gamma=0 reports the type I error, gamma != 0 the power = 1 - coverage_zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np

import design_table as dt

EFFECT_KEYS = ("estimator", "analysis_cluster", "stage", "target_scope",
               "estimate", "se", "truth_target", "error", "abs_error",
               "ci_valid", "ci_low", "ci_high", "ci_width", "cover_true",
               "cover_zero", "reject_zero", "invalid_reason", "point_status",
               "se_status", "estimand", "target_different", "raw_gap_to_full",
               "target_shift", "n_target")


def load_units(out_dir: str) -> list[dict]:
    root = os.path.join(out_dir, "units")
    records = []
    if not os.path.isdir(root):
        return records
    for point_dir in sorted(os.listdir(root)):
        folder = os.path.join(root, point_dir)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    records.append(json.load(handle))
            except (OSError, json.JSONDecodeError) as error:
                # empty or truncated JSON (power loss): skipped and named; a resumed run recomputes it
                print(f"[summarize] skipping corrupt unit file {path}: {error}", flush=True)
    return records


def write_csv(path: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _flat(row.get(k)) for k in fields})
    return len(rows)


def _flat(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _identity(record: dict) -> dict:
    return {"design_revision": record.get("design_revision"),
            "output_version": record.get("output_version"),
            "scenario": record.get("scenario"),
            "point_id": record.get("point_id"),
            "tier": record.get("tier"),
            "report_panels": "|".join(record.get("report_panels") or []),
            "param_rep": record.get("param_rep"),
            "data_rep": record.get("data_rep"),
            "structure_id": record.get("structure_id"),
            "unit_key": record.get("unit_key")}


def build_tables(records: list[dict]) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {k: [] for k in (
        "unit_status", "errors", "label_status", "cluster_diagnostics",
        "truth_effects", "feature_selection", "effect_raw",
        "probability_diagnostics", "selection_raw")}
    for rec in records:
        ident = _identity(rec)
        tables["unit_status"].append({
            **ident, "unit_complete": rec.get("unit_complete"),
            "status": rec.get("status", "complete" if rec.get("unit_complete") else "failed"),
            "reason": rec.get("reason"),
            "n_rows": (rec.get("observed_shape") or {}).get("n_rows"),
            "p": (rec.get("observed_shape") or {}).get("p"),
            "n_centers": (rec.get("observed_shape") or {}).get("n_centers"),
            "prevalence_x": (rec.get("unit_meta") or {}).get("prevalence_x"),
            "prevalence_y": (rec.get("unit_meta") or {}).get("prevalence_y"),
            **{f"t_{k}": v for k, v in (rec.get("timings") or {}).items()
               if "::" not in k},
        })
        for err in rec.get("errors") or []:
            tables["errors"].append({**ident, "error": err})
        if not rec.get("unit_complete"):
            if rec.get("reason"):
                tables["errors"].append({**ident, "error": rec["reason"],
                                         "fatal": True})
            continue

        label = rec.get("label_status") or {}
        tables["label_status"].append({
            **ident, "primary_status": label.get("primary_status"),
            "center_map_available": label.get("center_map_available"),
            "labels_frozen": label.get("labels_frozen"),
            "fixed_refit_status": label.get("fixed_refit_status"),
            "fixed_refit_eligible": label.get("fixed_refit_eligible"),
            "fixed_refit_converged": label.get("fixed_refit_converged"),
            "reason": label.get("reason"),
            "support_status": rec.get("support_status"),
            "selected_direction": label.get("selected_direction"),
            "selected_logical_id": label.get("selected_logical_id"),
            "label_seconds": (rec.get("timings") or {}).get("label"),
        })
        cd = rec.get("cluster_diagnostics") or {}
        align = rec.get("cluster_alignment") or {}
        tables["cluster_diagnostics"].append({
            **ident, "n_analysis_clusters": cd.get("n_analysis_clusters"),
            "cluster_sizes": cd.get("cluster_sizes"), "ari": cd.get("ari"),
            "single_center_clusters": cd.get("single_center_clusters"),
            "alignment_mapping": align.get("mapping"),
            "alignment_tie_broken": align.get("tie_broken"),
            "alignment_n_optimal": align.get("n_optimal_solutions"),
            "overlap_table": align.get("overlap_table"),
            "purity": align.get("purity"),
            "mixed_fraction": align.get("mixed_fraction"),
        })
        truth = rec.get("truth") or {}
        tables["truth_effects"].append({
            **ident, "gamma_logor": truth.get("gamma_logor"),
            "ate_do": truth.get("ate_do"), "ate_k": truth.get("ate_k"),
            "n_k": truth.get("n_k"),
            "true_cluster_targets": (truth.get("true_cluster_targets") or {}).get(
                "by_true_cluster"),
            "observed_core_sizes": {k: len(v) for k, v in
                                    (truth.get("observed_core_by_cluster") or {}).items()},
            "a_safe_size": len(truth.get("a_safe") or []),
            "full_causal_core_note": truth.get("full_causal_core_note"),
            "u_diagnostics": rec.get("latent_confounder_diagnostics"),
        })

        for method, per_cluster in (rec.get("feature_selection") or {}).items():
            for g, metrics in per_cluster.items():
                tables["feature_selection"].append(
                    {**ident, "method": method, "analysis_cluster": g, **metrics})
        for method, by_cluster in (rec.get("selected_by_cluster") or {}).items():
            if by_cluster is None:
                tables["selection_raw"].append(
                    {**ident, "method": method, "analysis_cluster": None,
                     "selected": None,
                     "reason": (rec.get("selections") or {}).get(method, {}).get("reason")})
                continue
            for g, cols in by_cluster.items():
                block = (rec.get("selections") or {}).get(method, {})
                pre_by = block.get("pre_icp_by_cluster") or {}
                pre = pre_by.get(int(g), pre_by.get(str(g)))     # keys are strings after a JSON round trip
                tables["selection_raw"].append(
                    {**ident, "method": method, "analysis_cluster": g,
                     "candidate_definition": block.get("candidate_definition"),
                     "pre_icp": pre, "selected": cols,
                     "n_pre_icp": (None if pre is None else len(pre)),
                     "n_selected": len(cols)})

        for method, by_est in (rec.get("effects") or {}).items():
            for est, rows in by_est.items():
                for row in rows:
                    tables["effect_raw"].append(
                        {**ident, "method": method,
                         "gamma_logor": truth.get("gamma_logor"),
                         **{k: row.get(k) for k in EFFECT_KEYS},
                         "reason": row.get("reason")})
        for method, by_est in (rec.get("estimator_diagnostics") or {}).items():
            for est, block in by_est.items():
                for g, diag in (block.get("diagnostics") or {}).items():
                    if not isinstance(diag, dict):
                        continue
                    keep = {k: v for k, v in diag.items()
                            if k in ("n_k", "q", "design_rank", "n_treated",
                                     "n_control", "e_min", "e_max", "ps_treated",
                                     "ps_control", "aipw_w1", "aipw_w0",
                                     "n_matched", "retention_rate",
                                     "retention_treated", "retention_control",
                                     "cross_center_match_rate", "caliper_k",
                                     "smd_max_before", "smd_max_after",
                                     "max_abs_c", "sum_c_squared", "max_reuse_K",
                                     "V_model", "se_model", "max_variance_share",
                                     "weight_identity_I_plus_K_holds",
                                     "mu_hat_min", "mu_hat_max", "status")}
                    if keep:
                        tables["probability_diagnostics"].append(
                            {**ident, "method": method, "estimator": est,
                             "analysis_cluster": g, **keep})
    return tables


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _mean(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _block_mcse(blocks: list[float]) -> float | None:
    vals = [v for v in blocks if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def summarize(effect_rows: list[dict], rows_by_point: dict[str, dict],
              planned_reps: int) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in effect_rows:
        key = (row["point_id"], row["method"], row["estimator"], row["stage"],
               str(row["analysis_cluster"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, rows in sorted(grouped.items()):
        point_id, method, estimator, stage, cluster = key
        design_row = rows_by_point.get(point_id, {})
        # gamma in effect in the unit: the S points hold a sentinel in the run table; the frozen value is in the unit
        gammas = {r.get("gamma_logor") for r in rows if r.get("gamma_logor") is not None}
        if len(gammas) > 1:
            raise RuntimeError(f"{point_id}: units disagree on gamma: {sorted(gammas)}")
        if gammas:
            gamma = float(gammas.pop())
        else:                                    # records without per-row gamma: use the design row
            g = str(design_row.get("gamma_logor", "") or "")
            gamma = float(g) if g and g != dt.GAMMA_SMALL_SENTINEL else float("nan")
        errors = [r["error"] for r in rows if r.get("error") is not None]
        valid = [r for r in rows if r.get("ci_valid")]
        n_attempted = len(rows)
        n_point = sum(1 for r in rows if r.get("estimate") is not None)
        n_valid = len(valid)

        by_block: dict[int, list[dict]] = {}
        for r in rows:
            by_block.setdefault(int(r["param_rep"]), []).append(r)
        block_bias = [_mean([x.get("error") for x in blk]) for blk in by_block.values()]
        block_mse = [_mean([(x["error"] ** 2) for x in blk
                            if x.get("error") is not None])
                     for blk in by_block.values()]

        bias = _mean(errors)
        rmse = (math.sqrt(_mean([e ** 2 for e in errors]))
                if errors else None)
        cover_true = _mean([1.0 if r["cover_true"] else 0.0 for r in valid
                            if r["cover_true"] is not None])
        cover_zero = _mean([1.0 if r["cover_zero"] else 0.0 for r in valid
                            if r["cover_zero"] is not None])
        reject = _mean([1.0 if r["reject_zero"] else 0.0 for r in valid
                        if r["reject_zero"] is not None])
        se_vals = [r["se"] for r in rows if r.get("se") is not None]
        rmse_mcse = None
        if rmse and rmse > 0:
            block_mse_clean = [v for v in block_mse if v is not None]
            if len(block_mse_clean) >= 2:
                mse_mcse = float(np.std(block_mse_clean, ddof=1)
                                 / math.sqrt(len(block_mse_clean)))
                rmse_mcse = mse_mcse / (2 * rmse)

        record = {
            "point_id": point_id, "scenario": design_row.get("scenario"),
            "tier": design_row.get("tier"),
            "report_panels": design_row.get("report_panels"),
            "gamma_logor": gamma, "method": method, "estimator": estimator,
            "stage": stage, "analysis_cluster": cluster,
            "target_scope": rows[0].get("target_scope"),
            "target_different": any(bool(r.get("target_different")) for r in rows),
            "n_planned_stage": planned_reps, "n_attempted": n_attempted,
            "n_point_available": n_point, "n_valid_ci": n_valid,
            "n_failed": n_attempted - n_point,
            "point_availability": (n_point / n_attempted) if n_attempted else None,
            "ci_availability": (n_valid / n_attempted) if n_attempted else None,
            "bias": bias, "rmse": rmse,
            "estimate_sd": (float(np.std([r["estimate"] for r in rows
                                          if r.get("estimate") is not None], ddof=1))
                            if n_point > 1 else None),
            "error_sd": (float(np.std(errors, ddof=1)) if len(errors) > 1 else None),
            "mean_model_se": _mean(se_vals),
            "rms_model_se": (math.sqrt(_mean([s ** 2 for s in se_vals]))
                             if se_vals else None),
            "coverage_true": cover_true, "coverage_zero": cover_zero,
            "reject_zero_rate": reject,
            "n_param_blocks": len(by_block),
            "bias_mcse": _block_mcse(block_bias), "rmse_mcse": rmse_mcse,
            "mcse_basis": f"{len(by_block)} param blocks",
        }
        if gamma == 0.0:
            record["type1_error"] = reject
            record["power"] = None
            record["test_note"] = ("gamma=0: type I error = mean(reject_zero), "
                                   "= 1 - coverage_true on the same valid denominator")
        else:
            record["type1_error"] = None
            record["power"] = reject
            record["test_note"] = ("power = mean(reject_zero) = 1 - coverage_zero, "
                                   "not 1 - coverage_true")
        out.append(record)
    return out


def paired_diffs(effect_rows: list[dict]) -> list[dict]:
    """Paired error differences of the methods for the same (point, param, data, estimator, stage, cluster)."""
    index: dict[tuple, dict[str, dict]] = {}
    for row in effect_rows:
        key = (row["point_id"], row["param_rep"], row["data_rep"],
               row["estimator"], row["stage"], str(row["analysis_cluster"]))
        index.setdefault(key, {})[row["method"]] = row
    out: list[dict] = []
    for key, by_method in index.items():
        base = by_method.get("dual_icp")
        if base is None or base.get("error") is None:
            continue
        for method, row in by_method.items():
            if method == "dual_icp" or row.get("error") is None:
                continue
            same_target = (row.get("target_scope") == base.get("target_scope")
                           and not row.get("target_different")
                           and not base.get("target_different"))
            out.append({
                "point_id": key[0], "param_rep": key[1], "data_rep": key[2],
                "estimator": key[3], "stage": key[4], "analysis_cluster": key[5],
                "method": method, "reference": "dual_icp",
                "error_diff": row["error"] - base["error"],
                "abs_error_diff": abs(row["error"]) - abs(base["error"]),
                "comparable_same_target": same_target,
                "comparability_note": (None if same_target else
                                       "target-different: not ranked as the same target"),
            })
    return out


def anomaly_report(summary_rows: list[dict]) -> list[dict]:
    """Points to inspect in a short run: evidence only, no tuning conclusions."""
    flags: list[dict] = []
    for row in summary_rows:
        if row["stage"] != "overall":
            continue
        reasons = []
        if row["point_availability"] is not None and row["point_availability"] < 1.0:
            reasons.append(f"point-estimate availability {row['point_availability']:.2f}")
        if row["ci_availability"] is not None and row["ci_availability"] < 1.0:
            reasons.append(f"interval availability {row['ci_availability']:.2f}")
        if (row["coverage_true"] is not None and row["n_valid_ci"] >= 5
                and row["coverage_true"] < 0.80):
            reasons.append(f"coverage_true {row['coverage_true']:.2f}")
        if (row["mean_model_se"] and row["error_sd"]
                and row["mean_model_se"] > 0
                and not 0.5 <= row["error_sd"] / row["mean_model_se"] <= 2.0):
            reasons.append(f"error SD / mean SE {row['error_sd'] / row['mean_model_se']:.2f}")
        if row["gamma_logor"] == 0.0 and row["type1_error"] is not None \
                and row["n_valid_ci"] >= 5 and row["type1_error"] > 0.30:
            reasons.append(f"rejection rate at zero {row['type1_error']:.2f}")
        if reasons:
            flags.append({**{k: row[k] for k in (
                "point_id", "scenario", "tier", "gamma_logor", "method",
                "estimator", "n_planned_stage", "n_attempted", "n_valid_ci",
                "bias", "rmse", "coverage_true", "coverage_zero",
                "reject_zero_rate", "mean_model_se", "error_sd")},
                "flags": "; ".join(reasons),
                "caveat": "few replications per point; rejection rates vary strongly"})
    return flags


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize formal simulation results")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--planned-reps", type=int, default=10,
                        help="planned replications per point of the current run")
    args = parser.parse_args(argv)

    records = load_units(args.out_dir)
    if not records:
        print("no unit results found", flush=True)
        return 2
    tables_dir = args.tables_dir or os.path.join(args.out_dir, "tables")
    tables = build_tables(records)
    rows_by_point = {r["point_id"]: r for r in dt.load_rows()}
    summary_rows = summarize(tables["effect_raw"], rows_by_point, args.planned_reps)
    tables["summary"] = summary_rows
    tables["paired_diff"] = paired_diffs(tables["effect_raw"])
    tables["anomaly_report"] = anomaly_report(summary_rows)

    counts = {}
    for name, rows in tables.items():
        counts[name] = write_csv(os.path.join(tables_dir, f"{name}.csv"), rows)

    timings: dict[str, list[float]] = {}
    sizes = []
    for rec in records:
        for stage, seconds in (rec.get("timings") or {}).items():
            timings.setdefault(stage.split("::")[0], []).append(float(seconds))
    units_root = os.path.join(args.out_dir, "units")
    for root, _dirs, files in os.walk(units_root):
        for name in files:
            sizes.append(os.path.getsize(os.path.join(root, name)))
    match_bytes = 0
    match_root = os.path.join(args.out_dir, "matching")
    for root, _dirs, files in os.walk(match_root):
        for name in files:
            match_bytes += os.path.getsize(os.path.join(root, name))

    resource = {
        "n_units": len(records),
        "n_units_complete": sum(1 for r in records if r.get("unit_complete")),
        "stage_seconds": {k: {"n": len(v), "median": float(np.median(v)),
                              "mean": float(np.mean(v)), "max": float(np.max(v)),
                              "total": float(np.sum(v))}
                          for k, v in sorted(timings.items())},
        "unit_json_bytes": {"total": int(sum(sizes)),
                            "median": int(np.median(sizes)) if sizes else 0},
        "matching_npz_bytes_total": int(match_bytes),
        "per_unit_bytes_estimate": int((sum(sizes) + match_bytes) / max(len(records), 1)),
        "note": "stage times are accumulated within processes; wall clock and concurrency are in manifest.json; "
                "the numbers come from the actual execution",
    }
    with open(os.path.join(tables_dir, "resource_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(resource, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(tables_dir, "table_counts.json"), "w",
              encoding="utf-8") as handle:
        json.dump(counts, handle, ensure_ascii=False, indent=2)
    print(f"[summarize] table rows {counts}")
    print(f"[summarize] resource report written to {tables_dir}/resource_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
