"""extract_ablation_cache.py — values reused by the ablation, streamed from the original unit JSON files.

One row per unit (config/ablation/cache_<stage>.csv.gz):
  * the dual candidates before the ICP per cluster `pre_icp_by_cluster` (used as is by dual_no_icp) and the final
    sets `selected_by_cluster` (re-estimated by dual_icp on the regenerated data, for per-unit checks and pairing);
  * the original AIPW effect rows of dual_icp / xside_icp / yside_icp (cluster and overall);
  * stage times of the original unit (fixedx_family, estimators::dual_icp, AIPW).
The inputs are the same unit archives as for extract_frozen_labels.py.

Usage:
  python -I tools/extract_ablation_cache.py --stage r1 --points P001-P040 --out config/ablation <archives>
  python -I tools/extract_ablation_cache.py --stage r2 --points P041-P054,U001-U020 --out config/ablation <archives>
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import statistics
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_frozen_labels import MEMBER_RE, _tar_streams, parse_points  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PAIR_METHODS = ("dual_icp", "xside_icp", "yside_icp")
ROW_KEYS = ("stage", "analysis_cluster", "estimate", "se", "truth_target", "point_status",
            "se_status", "ci_valid", "cover_true", "reject_zero", "error", "n_target")
FIELDS = ["unit_key", "point_id", "param_rep", "data_rep", "source_machine", "output_version",
          "unit_complete", "center_map_available", "n_rows", "n_centers",
          "dual_status", "dual_pre", "dual_sel", "dual_errors", "orig_aipw",
          "t_fixedx_family", "t_est_dual", "t_aipw_dual"]


def _row(rec: dict, machine: str) -> dict:
    sel = ((rec.get("selections") or {}).get("dual_icp") or {})
    eff = rec.get("effects") or {}
    aipw = {m: [{k: r.get(k) for k in ROW_KEYS} for r in ((eff.get(m) or {}).get("aipw") or [])]
            for m in PAIR_METHODS}
    tm = rec.get("timings") or {}
    diag = (((rec.get("estimator_diagnostics") or {}).get("dual_icp") or {}).get("aipw") or {})
    shape = rec.get("observed_shape") or {}
    return {
        "unit_key": rec["unit_key"], "point_id": rec["point_id"],
        "param_rep": rec["param_rep"], "data_rep": rec["data_rep"], "source_machine": machine,
        "output_version": rec.get("output_version"), "unit_complete": rec.get("unit_complete"),
        "center_map_available": (rec.get("label_status") or {}).get("center_map_available"),
        "n_rows": shape.get("n_rows"), "n_centers": shape.get("n_centers"),
        "dual_status": sel.get("status"),
        "dual_pre": json.dumps(sel.get("pre_icp_by_cluster")),
        "dual_sel": json.dumps((rec.get("selected_by_cluster") or {}).get("dual_icp")),
        "dual_errors": json.dumps(sel.get("errors_by_cluster") or {}),
        "orig_aipw": json.dumps(aipw),
        "t_fixedx_family": tm.get("fixedx_family"), "t_est_dual": tm.get("estimators::dual_icp"),
        "t_aipw_dual": diag.get("elapsed"),
    }


def scan_file(args) -> dict:
    path, points = args
    rows, seen, bad = [], {}, []
    for label, stream in _tar_streams(path):
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                m = MEMBER_RE.search(member.name)
                if not member.isfile() or not m or m.group("point") not in points:
                    continue
                try:
                    rec = json.loads(tar.extractfile(member).read())
                except json.JSONDecodeError as error:
                    bad.append(f"{label}:{member.name}: {error}")
                    continue
                if rec.get("unit_key") != m.group("key"):
                    bad.append(f"{label}:{member.name}: unit_key of the file name and content differ")
                    continue
                rows.append(_row(rec, m.group("machine")))
                seen.setdefault(rec["unit_key"], []).append(label)
    return {"rows": rows, "seen": seen, "bad": bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    files = sorted({f for pat in args.inputs for f in (glob.glob(pat) or [pat])})
    points = parse_points(args.points)
    with ProcessPoolExecutor(max_workers=min(len(files), 3)) as pool:
        parts = list(pool.map(scan_file, [(f, points) for f in files]))
    rows = sorted((r for p in parts for r in p["rows"]),
                  key=lambda r: (r["point_id"], int(r["param_rep"]), int(r["data_rep"])))
    seen: dict[str, list] = {}
    for p in parts:
        for k, v in p["seen"].items():
            seen.setdefault(k, []).extend(v)
    per_point: dict[str, int] = {}
    for r in rows:
        per_point[r["point_id"]] = per_point.get(r["point_id"], 0) + 1
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    short = {p: n for p, n in per_point.items() if n != args.reps}
    missing = sorted(points - set(per_point))
    not_complete_dual = [r["unit_key"] for r in rows
                         if r["center_map_available"] and r["dual_status"] != "complete"]
    no_pre = [r["unit_key"] for r in rows if r["center_map_available"] and r["dual_pre"] in ("null", "{}")]

    def stat(field):
        out = {}
        for pid in sorted(per_point):
            v = [float(r[field]) for r in rows if r["point_id"] == pid and r[field] is not None]
            out[pid] = {"median": round(statistics.median(v), 1), "mean": round(statistics.fmean(v), 1),
                        "n": len(v), "machine": sorted({r["source_machine"] for r in rows
                                                        if r["point_id"] == pid})} if v else None
        return out

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"cache_{args.stage}.csv.gz")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    report = {"stage": args.stage, "inputs": files, "n_units": len(rows),
              "expected_units": len(points) * args.reps, "per_point": per_point,
              "duplicates": dup, "missing_points": missing, "points_not_at_reps": short,
              "abstained_units": [r["unit_key"] for r in rows if not r["center_map_available"]],
              "dual_not_complete_units": not_complete_dual, "dual_pre_missing_units": no_pre,
              "bad_members": [b for p in parts for b in p["bad"]],
              "output_versions": sorted({r["output_version"] for r in rows}),
              "timing_original_s": {"fixedx_family": stat("t_fixedx_family"),
                                    "estimators_dual_all3": stat("t_est_dual"),
                                    "aipw_dual": stat("t_aipw_dual")}}
    with open(os.path.join(args.out, f"cache_{args.stage}_report.json"), "w", encoding="utf-8") as h:
        json.dump(report, h, ensure_ascii=False, indent=1)
    ok = (not dup and not missing and not short and not report["bad_members"]
          and len(rows) == report["expected_units"])
    print(f"[cache] {args.stage}: {len(rows)}/{report['expected_units']} units; {len(dup)} duplicates; missing points {missing}; "
          f"short {short}; {len(report['abstained_units'])} abstaining; {len(not_complete_dual)} dual not complete; "
          f"{len(no_pre)} without candidates -> {path}  {'complete' if ok else 'INCOMPLETE'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
