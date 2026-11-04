"""extract_pooled_cache.py — selection cache of the pooled comparators, config/pooled/sel_cache_<stage>.csv.gz.

The pooled comparators select one set from all centers, so their selections can be taken from stored runs
instead of being recomputed:
  pooled_lasso                ← pooled_lasso_selected of the frozen label table labels_<stage>.csv.gz
  pooled_pds / pooled_oal / hdps ← merged per-cluster selection tables (columns unit_key, method,
                                analysis_cluster, selected); the set must be the same in every cluster of a unit
Units whose labels abstained have no stored selections; the runner recomputes them.

Usage (from the binary directory):
  python -I tools/extract_pooled_cache.py --stage r1 --selection <selection_raw.csv> [<selection_raw.csv> ...]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ("pooled_pds", "pooled_oal", "hdps")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _norm(text: str) -> str:
    return json.dumps(sorted(int(j) for j in json.loads(text)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("r1", "r2"))
    ap.add_argument("--selection", nargs="*", default=[], help="merged per-cluster selection tables")
    a = ap.parse_args(argv)

    rows: dict[str, dict] = {}
    with gzip.open(os.path.join(HERE, "config", "labels", f"labels_{a.stage}.csv.gz"), "rt",
                   encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            ok = r["pooled_lasso_status"] == "complete" and r["pooled_lasso_selected"] not in ("", "null")
            rows[r["unit_key"]] = {"unit_key": r["unit_key"], "point_id": r["point_id"],
                                   "center_map_available": r["center_map_available"],
                                   "pooled_lasso": _norm(r["pooled_lasso_selected"]) if ok else "",
                                   **{m: "" for m in METHODS}}
    conflicts, foreign = [], 0
    for path in a.selection:
        with open(path, encoding="utf-8-sig", newline="") as h:
            for r in csv.DictReader(h):
                m = r["method"]
                if m not in METHODS or r["selected"] in ("", "null", "None"):
                    continue
                row = rows.get(r["unit_key"])
                if row is None:
                    foreign += 1
                    continue
                sel = _norm(r["selected"])
                if row[m] and row[m] != sel:
                    conflicts.append(f"{r['unit_key']} {m} cluster {r['analysis_cluster']}")
                row[m] = sel
    if conflicts:
        print(f"[fatal] a pooled selection differs between clusters of a unit: {conflicts[:5]}", file=sys.stderr)
        return 3

    out_dir = os.path.join(HERE, "config", "pooled")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"sel_cache_{a.stage}.csv.gz")
    fields = ["unit_key", "point_id", "center_map_available", "pooled_lasso", *METHODS]
    with gzip.open(out, "wt", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    abstain = {k for k, r in rows.items() if r["center_map_available"].lower() not in ("true", "1")}
    miss = {m: sorted(k for k, r in rows.items() if not r[m] and k not in abstain)
            for m in ("pooled_lasso", *METHODS)}
    report = {"stage": a.stage, "n_units": len(rows), "n_abstain": len(abstain),
              "selection_inputs": a.selection, "rows_not_in_labels": foreign,
              "missing_non_abstain": {m: len(v) for m, v in miss.items()},
              "missing_examples": {m: v[:5] for m, v in miss.items()}}
    with open(os.path.join(out_dir, f"sel_cache_{a.stage}_report.json"), "w", encoding="utf-8") as h:
        json.dump(report, h, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
