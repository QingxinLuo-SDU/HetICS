"""run_ablation.py — HetICS ablation for binary outcomes: dual_no_icp and dual_global_icp, with dual_icp as reference.

Each unit is regenerated from the design row and the recorded seeds (frozen_rerun):
  1. data and truth regenerated with the original seeds (the difference in ate_do is recorded);
  2. frozen labels give the original learned partition (units whose labels abstained are not estimable);
  3. the three selections (ablation_arms), selection metrics per cluster, and RA, AIPW and PSM.

--gate needs the original reference units (gate_<stage>.json.gz, not distributed): for one unit per point the
dual screening is replayed on the learned partition; the candidates and final sets per cluster must equal the
stored ones and the AIPW estimates and SEs of dual_icp must agree within |Δ| <= 1e-12 + 1e-9·|original|.

Usage: python -B -u src/run_ablation.py --stage r1 --out-dir runs_ablation/r1 --points P001,P003 --workers 4 --allow-recompute
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
import arms                                                      # noqa: E402
import design_table as dt                                        # noqa: E402
import evaluation as ev                                          # noqa: E402
import formal_dgp                                                # noqa: E402
import label_stage                                               # noqa: E402
import screening_comparison as sc                                # noqa: E402
from run_formal import ESTIMATOR_ORDER, _parse_reps, evaluate_selections, unit_path, write_atomic  # noqa: E402
from frozen_rerun import (ATE_TOL, GATE_ABS, GATE_REL, LABEL_DIR, STAGES, _peak_rss_gb,  # noqa: E402
                          _truthy, build_groups, frozen_center_map, load_labels, stage_rows)
from screening_bridge import cluster_members, load_screening     # noqa: E402

ABLATION_VERSION = "binary-ablation-v6-20261028"
ABL_DIR = os.path.join(dt.CONFIG_DIR, "ablation")
CONFIG_ALPHA_FILE = os.path.join(dt.CONFIG_DIR, "formal_r2.json")
ABL_STAGES = {"r1": STAGES["r1"], "r2": STAGES["r2"]}


def ablation_rows() -> dict[str, dict]:
    """The 74 rows with frozen labels (round 1 and round 2)."""
    return stage_rows()


def load_cache(stage: str, *, allow_missing: bool = False) -> dict[str, dict]:
    out: dict[str, dict] = {}
    path = os.path.join(ABL_DIR, f"cache_{stage}.csv.gz")
    if not os.path.isfile(path):
        if allow_missing:
            return out
        raise FileNotFoundError(f"Selection cache not found: {path}. Use --allow-recompute for a new run; --gate requires original reference files.")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            if row["unit_key"] in out:
                raise RuntimeError(f"Duplicate unit in the cache: {row['unit_key']}")
            out[row["unit_key"]] = row
    return out


def ablation_unit_complete(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as h:
            rec = json.load(h)
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("unit_complete") is True and rec.get("output_version") == ABLATION_VERSION


def _load_m6():
    m6 = load_screening(formal_dgp.STEP6_DIR)
    with open(CONFIG_ALPHA_FILE, encoding="utf-8") as h:
        alpha = float(json.load(h)["icp_alpha"])
    if alpha != aa.ICP_ALPHA:
        raise RuntimeError(f"Configured icp_alpha={alpha} differs from {aa.ICP_ALPHA}")
    m6.ALPHA_ICP = alpha                          # set in every worker
    return m6


def ablation_unit(continuous, row, group, lab, cache_row, b6, m6, u_vec, u_meta, *,
                  backend: str, gate: bool, allow_recompute: bool = False) -> dict:
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
    members = cluster_members(env, center_map) if center_map is not None else {}
    members_rows = {int(g): np.flatnonzero(np.isin(env, mem)) for g, mem in members.items()}
    alignment = (arms.align_clusters_to_truth(members_rows, truth["true_cluster_sample"], int(row["K"]))
                 if members_rows else None)
    task = label_stage.algorithm_task(row["point_id"], int(group["param_rep"]), int(group["data_rep"]))

    selections: dict[str, dict] = {}
    if center_map is None:
        for name in aa.ABLATION_ARMS:
            selections[name] = {"status": "unavailable", "reason": "upstream_abstain",
                                "selected_by_cluster": None}
    else:
        recomputed = not cache_row
        if recomputed:
            if gate or not allow_recompute:
                raise ValueError("Missing original selections: enable --allow-recompute for a new run; gate validation requires original references.")
            t0 = time.perf_counter()
            fam = sc.screen_family(F, X, Y, env, center_map, "fixedx_bic", m6=m6, task=task, config={})
            timings["recompute_cluster_family"] = time.perf_counter() - t0
            d = fam["directions"]["dual_icp"]
            cache_row = {"dual_status": d["status"],
                         "dual_pre": json.dumps(d["pre_icp_by_cluster"]),
                         "dual_sel": json.dumps(d["selected_by_cluster"])}
        for name in ("dual_icp", "dual_no_icp"):
            selections[name] = aa.cached_selection(name, cache_row, members_rows)
            if recomputed:
                selections[name]["source"] = "recomputed_original_screen_family"
                if name == "dual_icp":
                    selections[name]["icp"] = "official_once_recomputed_final"
        t0 = time.perf_counter()
        selections["dual_global_icp"] = aa.global_selection(F, X, Y, env, members_rows, m6=m6, task=task)
        timings["global_screen"] = time.perf_counter() - t0
    replay = None
    if gate and center_map is not None:              # gate: replay the dual screening on the learned partition
        t0 = time.perf_counter()
        fam = sc.screen_family(F, X, Y, env, center_map, "fixedx_bic", m6=m6, task=task, config={})
        timings["replay_cluster_family"] = time.perf_counter() - t0
        d = fam["directions"]["dual_icp"]
        replay = {"status": d["status"],
                  "pre_icp_by_cluster": {str(k): v for k, v in d["pre_icp_by_cluster"].items()},
                  "selected_by_cluster": {str(k): v for k, v in d["selected_by_cluster"].items()}}

    feature_eval = evaluate_selections(selections, truth, p, alignment)
    effects, est_diag, est_errors, est_t = aa.estimate_all(F, X, Y, env, truth, selections, center_map,
                                                           members_rows, backend=backend)
    timings.update(est_t)
    timings["unit_total"] = time.perf_counter() - t_unit
    return {
        "output_version": ABLATION_VERSION, "design_revision": row["design_revision"],
        "unit_key": dt.unit_key(row, group["param_rep"], group["data_rep"]),
        "point_id": row["point_id"], "scenario": row["scenario"], "tier": row["tier"],
        "param_rep": int(group["param_rep"]), "data_rep": int(group["data_rep"]),
        "gamma_logor": row["gamma_logor"], "structure_id": group["structure_id"],
        "execution_methods": list(aa.ABLATION_ARMS), "estimators": list(ESTIMATOR_ORDER),
        "icp_alpha_used": float(m6.ALPHA_ICP),
        "unit_complete": True, "errors": est_errors, "unit_meta": unit_meta,
        "latent_confounder_diagnostics": u_meta if row["has_unmeasured_U"] == "true" else None,
        "observed_shape": {"n_rows": int(F.shape[0]), "p": p, "n_centers": int(np.unique(env).size)},
        "label_status": {"source": "frozen_labels_from_returned_unit", "source_machine": lab["source_machine"],
                         "center_map_available": center_map is not None},
        "cluster_diagnostics": {"center_map": (None if center_map is None else
                                               {str(k): int(v) for k, v in center_map.items()}),
                                "cluster_sizes": {str(g): int(r.size) for g, r in members_rows.items()}},
        "truth": {"true_cluster_targets": ev.true_cluster_targets(truth), "ate_do": truth["ate_do"],
                  "ate_k": truth["ate_k"], "n_k": truth["n_k"], "gamma_logor": truth["gamma_logor"],
                  "observed_core_by_cluster": {str(k): v for k, v in truth["observed_core_by_cluster"].items()}},
        "cluster_alignment": alignment,
        "selections": {k: {kk: vv for kk, vv in v.items() if kk != "selected_by_cluster"}
                       for k, v in selections.items()},
        "selected_by_cluster": {k: (None if v.get("selected_by_cluster") is None else
                                    {str(g): list(s) for g, s in v["selected_by_cluster"].items()})
                                for k, v in selections.items()},
        "feature_selection": feature_eval, "effects": effects, "estimator_diagnostics": est_diag,
        "regeneration_checks": checks, "gate_replay": replay, "timings": timings,
        "resource": {"worker_peak_rss_gb": _peak_rss_gb()},
    }


def run_group(payload) -> dict:
    group, structures, rows, out_dir, cache, backend, gate, allow_recompute = payload
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    summary = {"base_group_key": group["base_group_key"], "units": {}}
    todo = [pid for pid in group["point_ids"]
            if gate or not ablation_unit_complete(unit_path(out_dir, rows[pid], group["param_rep"],
                                                            group["data_rep"]))]
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
            rec = ablation_unit(continuous, row, group, lab, cache.get(key, {}), b6, m6,
                                u_vec if row["has_unmeasured_U"] == "true" else None, u_meta,
                                backend=backend, gate=gate, allow_recompute=allow_recompute)
            rec["base_meta"] = base_meta
            write_atomic(path, rec)
            summary["units"][key] = {"status": "complete", "elapsed": rec["timings"]["unit_total"]}
        except Exception as error:                               # noqa: BLE001
            write_atomic(path, {"output_version": ABLATION_VERSION, "unit_key": key, "point_id": pid,
                                "unit_complete": False, "status": "failed",
                                "reason": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc()})
            summary["units"][key] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
    del continuous, u_vec
    gc.collect()
    summary.update(status="done", elapsed=time.perf_counter() - t0)
    return summary


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= GATE_ABS + GATE_REL * abs(float(b))


def gate_compare(rec: dict, cache_row: dict) -> dict:
    """Replay of the dual screening: candidates and final sets per cluster, and the dual_icp AIPW estimate and SE."""
    out = {"unit_key": rec["unit_key"], "point_id": rec["point_id"], "problems": [], "max_abs_diff": 0.0}
    rp = rec.get("gate_replay") or {}
    norm = lambda d: {str(k): (None if v is None else sorted(int(j) for j in v)) for k, v in (d or {}).items()}  # noqa: E731
    if norm(rp.get("pre_icp_by_cluster")) != norm(json.loads(cache_row["dual_pre"])):
        out["problems"].append("replayed candidates before the ICP differ from the stored unit")
    if norm(rp.get("selected_by_cluster")) != norm(json.loads(cache_row["dual_sel"])):
        out["problems"].append("replayed final sets differ from the stored unit")
    got = {(r["stage"], str(r["analysis_cluster"])): r for r in rec["effects"]["dual_icp"]["aipw"]}
    want = {(r["stage"], str(r["analysis_cluster"])): r
            for r in json.loads(cache_row["orig_aipw"])["dual_icp"]}
    if set(got) != set(want):
        out["problems"].append(f"AIPW rows differ: {sorted(got)} vs {sorted(want)}")
    for k, w in want.items():
        g = got.get(k) or {}
        for field in ("estimate", "se"):
            a, b = g.get(field), w.get(field)
            if a is not None and b is not None:
                out["max_abs_diff"] = max(out["max_abs_diff"], abs(float(a) - float(b)))
            if not _close(a, b):
                out["problems"].append(f"aipw {k} {field}: {a} vs {b}")
    gi = (rec["selections"].get("dual_global_icp") or {})
    out["global_n_icp_environments"] = gi.get("n_icp_environments")
    out["unit_seconds"] = rec["timings"].get("unit_total")
    out["timings"] = rec["timings"]
    out["worker_peak_rss_gb"] = (rec.get("resource") or {}).get("worker_peak_rss_gb")
    out["pass"] = not out["problems"]
    return out


def verify_complete(out_dir, rows, points, param_reps, data_reps) -> int:
    report, missing = {}, 0
    for pid in points:
        done = failed = 0
        for p in param_reps:
            for d in data_reps:
                path = unit_path(out_dir, rows[pid], p, d)
                if ablation_unit_complete(path):
                    done += 1
                elif os.path.isfile(path):
                    failed += 1
        expect = len(param_reps) * len(data_reps)
        report[pid] = {"complete": done, "failed_or_corrupt": failed, "expected": expect}
        missing += expect - done
        print(f"[verify] {pid:5s} {done:4d}/{expect}" + (f"  (failed or corrupt: {failed})" if failed else ""), flush=True)
    write_atomic(os.path.join(out_dir, "verify_complete.json"),
                 {"points": report, "missing": missing, "output_version": ABLATION_VERSION,
                  "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("[verify] all units complete" if not missing else f"[verify] {missing} units missing", flush=True)
    return 0 if not missing else 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HetICS ablation for binary outcomes")
    ap.add_argument("--stage", required=True, choices=sorted(ABL_STAGES))
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
                    help="Recompute missing selections with the original screening pipeline; not a substitute for --gate references.")
    args = ap.parse_args(argv)

    rows = ablation_rows()
    stage_points = list(ABL_STAGES[args.stage]["points"])
    points = [p.strip() for p in args.points.split(",") if p.strip()] if args.points else stage_points
    if set(points) - set(stage_points):
        print(f"[fatal] points outside stage {args.stage}: {sorted(set(points) - set(stage_points))}", file=sys.stderr)
        return 2
    param_reps, data_reps = _parse_reps(args.param_reps), _parse_reps(args.data_reps)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.verify_complete:
        return verify_complete(args.out_dir, rows, points, param_reps, data_reps)

    labels = load_labels([os.path.join(LABEL_DIR, f"labels_{args.stage}.csv.gz")])
    cache_path = os.path.join(ABL_DIR, f"cache_{args.stage}.csv.gz")
    cache = load_cache(args.stage, allow_missing=args.allow_recompute and not args.gate)
    if args.precheck:
        groups, _ = build_groups(rows, labels, points, param_reps, data_reps)
        problems = []
        for g in groups:
            for pid, lab in g["labels"].items():
                frozen_center_map(lab)
                key = lab["unit_key"]
                c = cache.get(key)
                if c is None and not (args.allow_recompute and not args.gate):
                    problems.append(f"{key}: missing from the cache")
                elif c is None:
                    continue  # Label and seed checks ran in build_groups; selections will be recomputed.
                elif _truthy(c["center_map_available"]) != _truthy(lab["center_map_available"]):
                    problems.append(f"{key}: abstention differs between cache and label table")
                elif _truthy(lab["center_map_available"]) and c["dual_pre"] in ("null", ""):
                    problems.append(f"{key}: cache lacks the original dual candidates")
        n_units = sum(len(g["point_ids"]) for g in groups)
        print(f"[precheck] stage {args.stage}: {n_units} units; {len(problems)} cache problems"
              + (f" (e.g. {problems[:3]})" if problems else ""), flush=True)
        if args.allow_recompute and not args.gate:
            print("[precheck] Missing selections will be recomputed; this is not reference gate validation.", flush=True)
        return 0 if not problems else 6

    gate = bool(args.gate)
    if gate:
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
    need = {lab["unit_key"] for g in groups for lab in g["labels"].values()}
    cache = {k: v for k, v in cache.items() if k in need}         # each worker carries only its cache rows
    print(f"[plan] stage {args.stage}: {len(points)} points -> {len(groups)} base groups; arms {list(aa.ABLATION_ARMS)}; "
          f"workers={args.workers}{' (gate)' if gate else ''}", flush=True)
    manifest = {"output_version": ABLATION_VERSION, "stage": args.stage, "gate": gate, "points": points,
                "param_reps": args.param_reps, "data_reps": args.data_reps, "workers": args.workers,
                "icp_alpha": aa.ICP_ALPHA, "arms": list(aa.ABLATION_ARMS), "estimators": list(ESTIMATOR_ORDER),
                "allow_recompute": bool(args.allow_recompute),
                "cache_sha256": dt.sha256_of(cache_path) if os.path.isfile(cache_path) else None,
                "labels_sha256": dt.sha256_of(os.path.join(LABEL_DIR, f"labels_{args.stage}.csv.gz")),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_atomic(os.path.join(out_dir, "manifest.json"), manifest)
    payloads = [(g, structures, rows, out_dir, {lab["unit_key"]: cache[lab["unit_key"]]
                                                for lab in g["labels"].values() if lab["unit_key"] in cache},
                 args.backend, gate, bool(args.allow_recompute)) for g in groups]
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
                res = (gate_compare(rec, cache[key]) if rec.get("unit_complete") else
                       {"unit_key": key, "point_id": ref["point_id"], "pass": False, "problems": [rec.get("reason")]})
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
                      "tolerance": {"abs": GATE_ABS, "rel": GATE_REL,
                                    "rule": "dual replay: identical candidates and final sets per cluster; dual_icp AIPW estimate and SE |Δ|<=abs+rel*|original|"},
                      "results": results})
        print(f"[gate] {'pass' if ok else 'FAIL'}: {n_pass}/{len(results)} units reproduce the original dual", flush=True)
        return 0 if ok else 8
    return 0 if n_bad == 0 and base_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
