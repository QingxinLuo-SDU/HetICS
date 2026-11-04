"""run_pooled.py — pooled comparators for binary outcomes: selection and estimation pool all centers.

Each unit is regenerated from the design row and the recorded seeds (frozen_rerun):
  1. data and truth regenerated with the original seeds (the difference in ate_do is recorded);
  2. the five selections of pooled_arms (cache first, recomputed with --allow-recompute);
  3. RA, AIPW and PSM of each selection on all centers.
Units whose labels abstained are estimated as usual (label_abstain = true; paired comparisons drop them).

--gate needs the original reference units and caches (not distributed): for one unit per point the dual
screening is replayed on the learned partition and the stored dual_icp AIPW is re-estimated (the same
comparison as the ablation gate), and the pooled lasso selection is recomputed and compared with the cache.
--precheck checks cache coverage and abstention against the label table.

Usage: python -B -u src/run_pooled.py --stage r1 --out-dir runs_pooled/r1 --points P001,P003 --workers 4 --allow-recompute
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ablation_arms as aa                                       # noqa: E402
import design_table as dt                                        # noqa: E402
import formal_dgp                                                # noqa: E402
import label_stage                                               # noqa: E402
import pooled_arms as pa                                         # noqa: E402
import pooled_selection as ps                                    # noqa: E402
import screening_comparison as sc                                # noqa: E402
from frozen_rerun import (ATE_TOL, LABEL_DIR, STAGES, _peak_rss_gb, _truthy,  # noqa: E402
                          build_groups, frozen_center_map, load_labels, stage_rows)
from run_ablation import _load_m6, gate_compare, load_cache      # noqa: E402
from run_formal import _parse_reps, unit_path, write_atomic      # noqa: E402
from screening_bridge import cluster_members                     # noqa: E402

POOLED_VERSION = "binary-pooled-v3-20261104"
POOLED_DIR = os.path.join(dt.CONFIG_DIR, "pooled")
POOLED_STAGES = {"r1": STAGES["r1"], "r2": STAGES["r2"]}


def load_sel_cache(stage: str, *, allow_missing: bool = False) -> dict[str, dict]:
    out: dict[str, dict] = {}
    path = os.path.join(POOLED_DIR, f"sel_cache_{stage}.csv.gz")
    if not os.path.isfile(path):
        if allow_missing:
            return out
        raise FileNotFoundError(f"Selection cache not found: {path}. Use --allow-recompute for a new run; --gate requires original reference files.")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            if row["unit_key"] in out:
                raise RuntimeError(f"Duplicate unit in the selection cache: {row['unit_key']}")
            out[row["unit_key"]] = row
    return out


def pooled_unit_complete(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as h:
            rec = json.load(h)
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("unit_complete") is True and rec.get("output_version") == POOLED_VERSION


def pooled_unit(continuous, row, group, lab, sel_row, abl_row, b6, m6, u_vec, u_meta, *,
                backend: str, gate: bool, allow_recompute: bool) -> dict:
    t_unit = time.perf_counter()
    timings: dict[str, float] = {}
    observed, truth, unit_meta = formal_dgp.binary_unit(continuous, row, group, b6, u=u_vec)
    F = np.asarray(observed.features, dtype=float)
    X = np.asarray(observed.treatment).astype(int)
    Y = np.asarray(observed.outcome).astype(int)
    env = np.asarray(observed.center_codes)
    p = int(F.shape[1])
    ate_diff = float(truth["ate_do"]) - float(lab["ate_do"])
    checks = {"ate_do_diff": ate_diff, "n_rows_match": int(F.shape[0]) == int(lab["n_rows"]),
              "regeneration_equal": abs(ate_diff) <= ATE_TOL and int(F.shape[0]) == int(lab["n_rows"])}
    center_map = frozen_center_map(lab)
    abstain = center_map is None
    task = label_stage.algorithm_task(row["point_id"], int(group["param_rep"]), int(group["data_rep"]))

    selections, sel_t = pa.pooled_selections(F, X, Y, env, truth, sel_row, b6,
                                             allow_recompute=allow_recompute or abstain)
    timings.update(sel_t)
    effects, est_diag, est_errors, est_t = pa.estimate_pooled(F, X, Y, env, truth, selections, backend=backend)
    timings.update(est_t)

    replay = None
    if gate:
        # (1) replay of the dual screening on the learned partition and re-estimation of the stored final sets
        if not abstain:
            members = cluster_members(env, center_map)
            members_rows = {int(g): np.flatnonzero(np.isin(env, mem)) for g, mem in members.items()}
            t0 = time.perf_counter()
            fam = sc.screen_family(F, X, Y, env, center_map, "fixedx_bic", m6=m6, task=task, config={})
            timings["gate_replay_cluster_family"] = time.perf_counter() - t0
            d = fam["directions"]["dual_icp"]
            replay = {"status": d["status"],
                      "pre_icp_by_cluster": {str(k): v for k, v in d["pre_icp_by_cluster"].items()},
                      "selected_by_cluster": {str(k): v for k, v in d["selected_by_cluster"].items()}}
            dual = {"dual_icp": aa.cached_selection("dual_icp", abl_row, members_rows)}
            eff, _d, _e, _t = aa.estimate_aipw(F, X, Y, env, truth, dual, center_map, members_rows, backend=backend)
            effects["dual_icp"] = eff["dual_icp"]
        # (2) pooled lasso recomputed and compared with the cache covariate by covariate
        t0 = time.perf_counter()
        blk = ps.pooled_lasso(F, Y, b6)
        timings["gate_recompute_pooled_lasso"] = time.perf_counter() - t0
        checks["gate_pooled_lasso_recomputed"] = (sorted(int(j) for j in blk["selected"])
                                                  if blk.get("status") == "complete" else None)
    timings["unit_total"] = time.perf_counter() - t_unit
    return {
        "output_version": POOLED_VERSION, "design_revision": row["design_revision"],
        "unit_key": dt.unit_key(row, group["param_rep"], group["data_rep"]),
        "point_id": row["point_id"], "scenario": row["scenario"], "tier": row["tier"],
        "param_rep": int(group["param_rep"]), "data_rep": int(group["data_rep"]),
        "gamma_logor": row["gamma_logor"], "structure_id": group["structure_id"],
        "execution_methods": list(pa.POOLED_ARMS), "estimators": ["ra_standardized", "aipw", "psm_caliper"],
        "icp_alpha_used": float(m6.ALPHA_ICP), "label_abstain": abstain,
        "unit_complete": True, "errors": est_errors, "unit_meta": unit_meta,
        "latent_confounder_diagnostics": u_meta if row["has_unmeasured_U"] == "true" else None,
        "observed_shape": {"n_rows": int(F.shape[0]), "p": p, "n_centers": int(np.unique(env).size)},
        "truth": {"ate_do": truth["ate_do"], "ate_k": truth["ate_k"], "n_k": truth["n_k"],
                  "gamma_logor": truth["gamma_logor"],
                  "observed_core_by_cluster": {str(k): v for k, v in truth["observed_core_by_cluster"].items()}},
        "selections": selections, "feature_selection": pa.feature_eval(selections, truth, p),
        "effects": effects, "estimator_diagnostics": est_diag,
        "regeneration_checks": checks, "gate_replay": replay, "timings": timings,
        "resource": {"worker_peak_rss_gb": _peak_rss_gb()},
    }


def run_group(payload) -> dict:
    group, structures, rows, out_dir, sel_cache, abl_cache, backend, gate, allow_recompute = payload
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    summary = {"base_group_key": group["base_group_key"], "units": {}}
    todo = [pid for pid in group["point_ids"]
            if gate or not pooled_unit_complete(unit_path(out_dir, rows[pid], group["param_rep"], group["data_rep"]))]
    if not todo:
        summary["status"] = "already_complete"
        return summary
    t0 = time.perf_counter()
    try:
        m6 = _load_m6()
        _gen, b6 = formal_dgp.load_sources()
        continuous, base_meta = formal_dgp.generate_base(group, structures)
        u_vec = u_meta = None
        if any(rows[pid]["has_unmeasured_U"] == "true" for pid in todo):
            rho_block = float(structures[group["structure_id"]]["params"]["rho_block"])
            u_vec, _s, u_meta = formal_dgp.unmeasured_u_for_base(continuous, group, rho_block)
    except Exception as error:                                   # noqa: BLE001
        summary.update(status="base_failed", reason=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc())
        return summary
    for pid in todo:
        row, lab = rows[pid], group["labels"][pid]
        key = dt.unit_key(row, group["param_rep"], group["data_rep"])
        path = unit_path(out_dir, row, group["param_rep"], group["data_rep"])
        try:
            rec = pooled_unit(continuous, row, group, lab, sel_cache.get(key, {}), abl_cache.get(key, {}), b6, m6,
                              u_vec if row["has_unmeasured_U"] == "true" else None, u_meta,
                              backend=backend, gate=gate, allow_recompute=allow_recompute)
            rec["base_meta"] = base_meta
            write_atomic(path, rec)
            summary["units"][key] = {"status": "complete", "elapsed": rec["timings"]["unit_total"]}
        except Exception as error:                               # noqa: BLE001
            write_atomic(path, {"output_version": POOLED_VERSION, "unit_key": key, "point_id": pid,
                                "unit_complete": False, "status": "failed",
                                "reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            summary["units"][key] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
    del continuous, u_vec
    gc.collect()
    summary.update(status="done", elapsed=time.perf_counter() - t0)
    return summary


def gate_check(rec: dict, abl_row: dict, sel_row: dict) -> dict:
    lasso_cached = pa._cached(sel_row, "pooled_lasso_fp")
    lasso_new = (rec.get("regeneration_checks") or {}).get("gate_pooled_lasso_recomputed")
    if rec.get("label_abstain"):
        out = {"unit_key": rec["unit_key"], "point_id": rec["point_id"], "problems": [],
               "timings": rec["timings"], "unit_seconds": rec["timings"].get("unit_total"),
               "worker_peak_rss_gb": (rec.get("resource") or {}).get("worker_peak_rss_gb")}
    else:
        out = gate_compare(rec, abl_row)
    if lasso_cached is not None and lasso_new != lasso_cached:
        out["problems"].append(f"pooled_lasso recomputed {None if lasso_new is None else len(lasso_new)} covariates, "
                               f"cache {len(lasso_cached)}")
    overall = lambda m: next((r for r in (rec["effects"].get(m) or {}).get("aipw", [])  # noqa: E731
                              if r.get("stage") == "overall"), {})
    out["pooled_arms_without_estimate"] = [m for m in pa.POOLED_ARMS if overall(m).get("estimate") is None]
    out["pass"] = not out["problems"]
    return out


def verify_complete(out_dir, rows, points, param_reps, data_reps) -> int:
    report, missing = {}, 0
    for pid in points:
        done = failed = 0
        for p in param_reps:
            for d in data_reps:
                path = unit_path(out_dir, rows[pid], p, d)
                if pooled_unit_complete(path):
                    done += 1
                elif os.path.isfile(path):
                    failed += 1
        expect = len(param_reps) * len(data_reps)
        report[pid] = {"complete": done, "failed_or_corrupt": failed, "expected": expect}
        missing += expect - done
        print(f"[verify] {pid:5s} {done:4d}/{expect}" + (f"  (failed or corrupt: {failed})" if failed else ""), flush=True)
    write_atomic(os.path.join(out_dir, "verify_complete.json"),
                 {"points": report, "missing": missing, "output_version": POOLED_VERSION,
                  "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("[verify] all units complete" if not missing else f"[verify] {missing} units missing", flush=True)
    return 0 if not missing else 5


def precheck(stage, rows, labels, points, param_reps, data_reps, sel_cache, *, allow_recompute=False) -> int:
    groups, _ = build_groups(rows, labels, points, param_reps, data_reps)
    problems, n_units = [], 0
    for g in groups:
        for pid, lab in g["labels"].items():
            n_units += 1
            frozen_center_map(lab)
            key = lab["unit_key"]
            c = sel_cache.get(key)
            if c is None:
                if not allow_recompute:
                    problems.append(f"{key}: missing from the selection cache")
                continue
            if _truthy(c["center_map_available"]) != _truthy(lab["center_map_available"]):
                problems.append(f"{key}: abstention differs between cache and label table")
            elif _truthy(lab["center_map_available"]):
                miss = [m for m in pa.CACHE_COLUMN.values() if not c.get(m)]
                if miss and not allow_recompute:
                    problems.append(f"{key}: lacks {miss}")
    print(f"[precheck] stage {stage}: {n_units} units; {len(problems)} cache problems"
          + (f" (e.g. {problems[:3]})" if problems else ""), flush=True)
    if allow_recompute:
        print("[precheck] Missing selections will be recomputed; this is not reference gate validation.", flush=True)
    return 0 if not problems else 6


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pooled comparators for binary outcomes")
    ap.add_argument("--stage", required=True, choices=sorted(POOLED_STAGES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--points", default=None, help="default: every point of the stage")
    ap.add_argument("--param-reps", default="1-500")
    ap.add_argument("--data-reps", default="1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--backend", default="statsmodels_glm")
    ap.add_argument("--verify-complete", action="store_true")
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--allow-recompute", action="store_true",
                    help="compute selections missing from the cache (several minutes per unit at the paper design)")
    args = ap.parse_args(argv)

    rows = stage_rows()
    stage_points = list(POOLED_STAGES[args.stage]["points"])
    points = [p.strip() for p in args.points.split(",") if p.strip()] if args.points else stage_points
    if set(points) - set(stage_points):
        print(f"[fatal] points outside stage {args.stage}: {sorted(set(points) - set(stage_points))}", file=sys.stderr)
        return 2
    param_reps, data_reps = _parse_reps(args.param_reps), _parse_reps(args.data_reps)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.verify_complete:
        return verify_complete(args.out_dir, rows, points, param_reps, data_reps)

    labels = load_labels([os.path.join(LABEL_DIR, f"labels_{args.stage}.csv.gz")])
    cache_path = os.path.join(POOLED_DIR, f"sel_cache_{args.stage}.csv.gz")
    sel_cache = load_sel_cache(args.stage, allow_missing=args.allow_recompute and not args.gate)
    if args.precheck:
        return precheck(args.stage, rows, labels, points, param_reps, data_reps, sel_cache,
                        allow_recompute=args.allow_recompute and not args.gate)

    gate = bool(args.gate)
    abl_cache: dict[str, dict] = {}
    if gate:
        abl_cache = load_cache(args.stage)
        with gzip.open(os.path.join(LABEL_DIR, f"gate_{args.stage}.json.gz"), "rt", encoding="utf-8") as h:
            refs = {blk["unit_key"]: blk for pid, blk in json.load(h).items() if pid in points}
        groups, structures = [], None
        for key, ref in sorted(refs.items()):
            g, structures = build_groups(rows, labels, [ref["point_id"]], [int(ref["param_rep"])],
                                         [int(ref["data_rep"])])
            groups.extend(g)
        out_dir = os.path.join(args.out_dir, "gate")
    else:
        groups, structures = build_groups(rows, labels, points, param_reps, data_reps)
        out_dir = args.out_dir
    print(f"[plan] stage {args.stage}: {len(points)} points -> {len(groups)} base groups; arms {list(pa.POOLED_ARMS)}; "
          f"workers={args.workers}{' (gate)' if gate else ''}", flush=True)
    manifest = {"output_version": POOLED_VERSION, "stage": args.stage, "gate": gate, "points": points,
                "param_reps": args.param_reps, "data_reps": args.data_reps, "workers": args.workers,
                "icp_alpha": aa.ICP_ALPHA, "arms": list(pa.POOLED_ARMS), "estimators": ["ra_standardized", "aipw", "psm_caliper"],
                "allow_recompute": bool(args.allow_recompute),
                "sel_cache_sha256": dt.sha256_of(cache_path) if os.path.isfile(cache_path) else None,
                "labels_sha256": dt.sha256_of(os.path.join(LABEL_DIR, f"labels_{args.stage}.csv.gz")),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_atomic(os.path.join(out_dir, "manifest.json"), manifest)
    payloads = []
    for g in groups:
        keys = [lab["unit_key"] for lab in g["labels"].values()]
        payloads.append((g, structures, rows, out_dir, {k: sel_cache[k] for k in keys if k in sel_cache},
                         {k: abl_cache[k] for k in keys if k in abl_cache}, args.backend, gate,
                         bool(args.allow_recompute)))
    summaries, t0 = [], time.perf_counter()
    workers = max(1, min(args.workers, len(payloads)))
    if workers == 1:
        for pl in payloads:
            summaries.append(run_group(pl))
            print(f"[done] {summaries[-1]['base_group_key']} {summaries[-1].get('status')}", flush=True)
    else:
        # ProcessPoolExecutor; exit with code 9 when a worker is killed (executor shutdown can hang, gh-94777)
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        try:
            for fut in as_completed([ex.submit(run_group, pl) for pl in payloads]):
                s = fut.result()
                summaries.append(s)
                print(f"[done] {s['base_group_key']} {s.get('status')}", flush=True)
        except BrokenProcessPool as error:
            print(f"[fatal] a worker was killed (usually out of memory): {error}. Completed units are on disk; "
                  "rerun with fewer workers to resume", flush=True)
            manifest.update(broken_pool=True, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            write_atomic(os.path.join(out_dir, "manifest.json"), manifest)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(9)
        ex.shutdown(wait=True)
    n_ok = sum(1 for s in summaries for v in s.get("units", {}).values() if v["status"] == "complete")
    n_bad = sum(1 for s in summaries for v in s.get("units", {}).values() if v["status"] != "complete")
    base_failed = sum(summary.get("status") == "base_failed" for summary in summaries)
    manifest["base_groups_failed"] = base_failed
    manifest.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    wall_seconds=time.perf_counter() - t0, units_complete=n_ok, units_failed=n_bad,
                    group_summaries=summaries)
    write_atomic(os.path.join(out_dir, "manifest.json"), manifest)
    print(f"[end] {n_ok} complete, {n_bad} failed", flush=True)

    if gate:
        results = []
        for key, ref in sorted(refs.items()):
            path = unit_path(out_dir, rows[ref["point_id"]], int(ref["param_rep"]), int(ref["data_rep"]))
            try:
                with open(path, encoding="utf-8") as h:
                    rec = json.load(h)
                res = (gate_check(rec, abl_cache.get(key, {}), sel_cache.get(key, {})) if rec.get("unit_complete")
                       else {"unit_key": key, "point_id": ref["point_id"], "pass": False, "problems": [rec.get("reason")]})
            except (OSError, json.JSONDecodeError) as error:
                res = {"unit_key": key, "point_id": ref["point_id"], "pass": False, "problems": [f"no result: {error}"]}
            results.append(res)
            print(f"[gate] {key}: {'pass' if res['pass'] else 'FAIL ' + '; '.join(map(str, res['problems']))[:300]}"
                  + (f"  {res.get('unit_seconds', 0):.0f}s  peak {res.get('worker_peak_rss_gb')} GB"
                     if res.get("unit_seconds") else ""), flush=True)
        n_pass = sum(r["pass"] for r in results)
        ok = n_pass == len(results) and len(results) >= min(20, len(points))
        write_atomic(os.path.join(out_dir, "gate_report.json"),
                     {"pass": ok, "n_units": len(results), "n_pass": n_pass,
                      "rule": "dual replay (identical candidates and final sets; AIPW estimate and SE within tolerance) and pooled lasso recomputed = cache",
                      "results": results})
        print(f"[gate] {'pass' if ok else 'FAIL'}: {n_pass}/{len(results)} units", flush=True)
        return 0 if ok else 8
    return 0 if n_bad == 0 and base_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
