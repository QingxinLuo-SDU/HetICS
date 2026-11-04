"""merge_tables.py (round 2) — merge of the round-2 shards: gates before merging, tables concatenated by point, checks after.

Round-2 specifics:
  - design_revision = r2_cal3_sgamma_sens_sigy1_500_20261016;
  - the points of a shard come from results/points_<id>.txt, or from runs/<id>/verify_complete.json when a shard
    has no points file; when both exist they must agree;
  - gate g: every shard with S points has gamma_small_frozen.json with status=frozen, the same version and the same
    gamma; it replaces the @gamma_small sentinel of the S rows of the run table.

Concatenation is exact: every table written on a shard holds per-unit rows, and summary / paired_diff /
anomaly_report are grouped by point. When every point lies on exactly one shard (gate a), concatenating the
shard tables equals running summarize_formal on all units, with nothing to deduplicate or add across shards.

Gates (any failure stops the merge and writes no merged tables):
  a  points without duplicates or gaps: the points_<id>.txt files are disjoint and cover the 59 points of the
     run table; exactly 59 x 500 units, every unit of a shard belongs to a point of its points file, 500 per point
  b  unit_meta.sig_y_applied == 1.0 in every unit
  c  design_revision == r2_cal3_sgamma_sens_sigy1_500_20261016 in every unit and every table row
Checks after merging: CSV rows per shard and table = table_counts.json = rows recomputed from the units;
          the units of unit_status = the units of the shard archive; every summary denominator is 500.

Usage: python -I merge_tables.py --res-dir <extracted> --scan <scan output> --run-table <csv> --out <dir> --machines <ids>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import sys

import pandas as pd

MACHINES = ()  # Set explicitly by --machines; identities are not part of the design.
EXPECTED_REVISION = "r2_cal3_sgamma_sens_sigy1_500_20261016"
N_REPS = 500
PKG = "binary"   # package directory inside res_<ID>; set by --package
GAMMA_SENTINEL = "@gamma_small"
DERIVED = {"summary", "paired_diff", "anomaly_report"}
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)


def _machine_root(res_dir, m):
    return os.path.join(res_dir, f"res_{m}", PKG)


def _results(res_dir, m, name):
    return os.path.join(_machine_root(res_dir, m), "results", name)


def load_points(res_dir, m) -> tuple[set, set]:
    """(points of the points file, points of verify_complete.json); shards without a points file use the latter for both."""
    with open(os.path.join(_machine_root(res_dir, m), "runs", m, "verify_complete.json"),
              encoding="utf-8") as f:
        vc = set(json.load(f)["points"])
    path = _results(res_dir, m, f"points_{m}.txt")
    if not os.path.exists(path):
        return vc, vc
    with open(path, encoding="utf-8") as f:
        return set(f.read().split()), vc


def gate_gamma(args, run_table) -> tuple[list[str], dict, float | None]:
    """Gate g: shards with S points each calibrated the small gamma once (deterministically); the frozen values must agree."""
    fails: list[str] = []
    frozen = {}
    for m in MACHINES:
        path = _results(args.res_dir, m, "gamma_small_frozen.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                g = json.load(f)
            frozen[m] = {k: g.get(k) for k in ("status", "gamma_logor", "design_revision",
                                                "run_table_sha256", "frozen_utc")}
    s_points = set(run_table.loc[run_table["gamma_logor"] == GAMMA_SENTINEL, "point_id"])
    for m in MACHINES:
        if load_points(args.res_dir, m)[0] & s_points and m not in frozen:
            fails.append(f"[g] {m} has S points but no gamma_small_frozen.json")
    bad = {m: v for m, v in frozen.items()
           if v["status"] != "frozen" or v["design_revision"] != EXPECTED_REVISION}
    if bad:
        fails.append(f"[g] frozen files with a wrong status or version: {bad}")
    values = {v["gamma_logor"] for v in frozen.values()}
    if len(values) != 1:
        fails.append(f"[g] the shards froze different small gamma values: { {m: v['gamma_logor'] for m, v in frozen.items()} }")
    gamma = float(next(iter(values))) if len(values) == 1 else None
    return fails, {"frozen_by_machine": frozen, "gamma_small": gamma, "s_points": sorted(s_points)}, gamma


def gate_units(args, run_table, gamma) -> tuple[list[str], dict, pd.DataFrame]:
    fails: list[str] = []
    info: dict = {}
    points_all = set(run_table["point_id"])
    pts = {}
    for m in MACHINES:
        pts[m], vc = load_points(args.res_dir, m)
        if pts[m] != vc:
            fails.append(f"[a] {m}: points file and verify_complete.json disagree: "
                         f"file extra {sorted(pts[m] - vc)}, verify extra {sorted(vc - pts[m])}")
    info["points_by_machine"] = {m: sorted(v) for m, v in pts.items()}
    seen = collections.Counter(p for v in pts.values() for p in v)
    dup = sorted(p for p, c in seen.items() if c > 1)
    if dup:
        fails.append(f"[a] points repeated across points files: {dup}")
    if set(seen) != points_all:
        fails.append(f"[a] union of the points files != run table: missing {sorted(points_all - set(seen))}, "
                     f"extra {sorted(set(seen) - points_all)}")

    idx = pd.read_csv(os.path.join(args.scan, "units_index.csv"), dtype={"machine": str})
    info["n_unit_files"] = int(len(idx))
    bad_json = idx[idx["json_error"].notna()]
    if len(bad_json):
        fails.append(f"[a] {len(bad_json)} unit JSON files cannot be parsed (empty or truncated): "
                     f"{bad_json['member'].head(10).tolist()}")
    idx = idx[idx["json_error"].isna()].copy()
    idx["param_rep"] = idx["param_rep"].astype(int)
    mism = idx[(idx["dir_point"] != idx["point_id"]) | (idx["file_stem"] != idx["unit_key"])]
    if len(mism):
        fails.append(f"[a] {len(mism)} units whose file path disagrees with the content: {mism['member'].head(5).tolist()}")
    if set(idx["data_rep"].astype(int)) != {1}:
        fails.append(f"[a] data_rep is not always 1: {sorted(set(idx['data_rep']))}")

    keys = idx.groupby(["point_id", "param_rep"]).size()
    dups = keys[keys > 1]
    if len(dups):
        where = idx.set_index(["point_id", "param_rep"]).loc[dups.index, "machine"]
        fails.append(f"[a] {len(dups)} (point, param_rep) pairs repeated, e.g. "
                     f"{where.reset_index().head(10).to_dict('records')}")
    expected = {(p, r) for p in points_all for r in range(1, N_REPS + 1)}
    got = set(keys.index)
    if expected - got:
        miss = sorted(expected - got)
        fails.append(f"[a] {len(miss)} units missing, e.g. {miss[:10]}")
    if got - expected:
        extra = sorted(got - expected)
        fails.append(f"[a] {len(extra)} units outside the design, e.g. {extra[:10]}")

    per = idx.groupby(["machine", "point_id"]).size()
    info["units_by_machine"] = idx.groupby("machine").size().to_dict()
    for m in MACHINES:
        on_m = per.loc[m] if m in per.index.get_level_values(0) else pd.Series(dtype=int)
        stray = sorted(set(on_m.index) - pts[m])
        if stray:
            fails.append(f"[a] the archive of {m} contains points outside points_{m}.txt: "
                         f"{ {p: int(on_m[p]) for p in stray} }")
        short = {p: int(on_m.get(p, 0)) for p in pts[m] if on_m.get(p, 0) != N_REPS}
        if short:
            fails.append(f"[a] points of {m} without {N_REPS} units: {short}")

    sig = pd.to_numeric(idx["sig_y_applied"], errors="coerce")
    bad_sig = idx[sig != 1.0]
    info["sig_y_applied_values"] = {str(k): int(v) for k, v in
                                    idx["sig_y_applied"].value_counts(dropna=False).items()}
    if len(bad_sig):
        fails.append(f"[b] {len(bad_sig)} units with sig_y_applied != 1.0 (or missing), "
                     f"e.g. {bad_sig[['machine', 'unit_key', 'sig_y_applied']].head(10).to_dict('records')}")

    info["design_revision_values"] = idx["design_revision"].value_counts(dropna=False).to_dict()
    bad_rev = idx[idx["design_revision"] != EXPECTED_REVISION]
    if len(bad_rev):
        fails.append(f"[c] {len(bad_rev)} units with design_revision != {EXPECTED_REVISION}: "
                     f"{info['design_revision_values']}")
    info["output_version_values"] = idx["output_version"].value_counts(dropna=False).to_dict()
    if idx["output_version"].nunique(dropna=False) != 1:
        fails.append(f"[c] output_version is not unique: {info['output_version_values']}")

    # further conditions that invalidate the results
    if not idx["unit_complete"].astype(bool).all():
        fails.append(f"[x] {int((~idx['unit_complete'].astype(bool)).sum())} units with unit_complete=False")
    arms_design = run_table.set_index("point_id")["execution_methods"].map(
        lambda s: "|".join(sorted(s.split("|"))))
    arm_bad = idx[idx["methods"] != idx["point_id"].map(arms_design)]
    if len(arm_bad):
        fails.append(f"[x] {len(arm_bad)} units whose arms differ from the run table, e.g. "
                     f"{arm_bad[['unit_key', 'methods']].head(3).to_dict('records')}")
    gam_rt = run_table.set_index("point_id")["gamma_logor"].replace(GAMMA_SENTINEL, gamma).astype(float)
    gam = pd.to_numeric(idx["gamma_logor_meta"]) - idx["point_id"].map(gam_rt)
    if (gam.abs() > 1e-12).any():
        fails.append(f"[x] {int((gam.abs() > 1e-12).sum())} units whose gamma differs from the run table (S rows with the frozen gamma)")
    info["n_errors_total"] = int(idx["n_errors"].sum())
    return fails, info, idx


def merge(args, idx) -> tuple[list[str], dict]:
    fails: list[str] = []
    info: dict = {"tables": {}}
    with open(os.path.join(args.scan, "expected_table_counts.json"), encoding="utf-8") as f:
        expected = json.load(f)
    counts_json = {}
    for m in MACHINES:
        with open(os.path.join(_machine_root(args.res_dir, m), "runs", m, "tables",
                               "table_counts.json"), encoding="utf-8") as f:
            counts_json[m] = json.load(f)
    names = sorted(set().union(*[set(c) for c in counts_json.values()]))
    pts = {m: load_points(args.res_dir, m)[0] for m in MACHINES}
    units_by_m = {m: set(idx.loc[idx["machine"] == m, "unit_key"]) for m in MACHINES}

    tmp = os.path.join(args.out, "tables.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for name in names:
        srcs = {m: os.path.join(_machine_root(args.res_dir, m), "runs", m, "tables", f"{name}.csv")
                for m in MACHINES}
        header: list[str] = []
        for m, p in srcs.items():
            if os.path.exists(p) and os.path.getsize(p) > 0:
                with open(p, encoding="utf-8-sig", newline="") as f:
                    for col in next(csv.reader(f)):
                        if col not in header:
                            header.append(col)
        per_m = {}
        if not header:                                  # e.g. errors: 0 rows on every shard, no file
            for m in MACHINES:
                per_m[m] = 0
                exp = expected[m].get(name, 0)
                if counts_json[m].get(name, 0) != 0 or exp != 0:
                    fails.append(f"[t] {name}@{m}: no file, but table_counts={counts_json[m].get(name)}"
                                 f", recomputed from units={exp}")
            info["tables"][name] = {"rows": 0, "per_machine": per_m}
            continue
        with open(os.path.join(tmp, f"{name}.csv"), "w", encoding="utf-8-sig", newline="") as out:
            w = csv.DictWriter(out, fieldnames=header)
            w.writeheader()
            for m, p in srcs.items():
                n = 0
                points, revs, ukeys = set(), set(), set()
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    with open(p, encoding="utf-8-sig", newline="") as f:
                        for row in csv.DictReader(f):
                            w.writerow(row)
                            n += 1
                            points.add(row.get("point_id"))
                            if "design_revision" in row:
                                revs.add(row["design_revision"])
                            if name == "unit_status":
                                ukeys.add(row["unit_key"])
                per_m[m] = n
                # derived tables (by point) are not built per unit; they are recomputed from effect_raw below
                exp = counts_json[m].get(name, 0) if name in DERIVED else expected[m].get(name, 0)
                if not (n == counts_json[m].get(name, 0) == exp):
                    fails.append(f"[t] {name}@{m}: CSV {n} rows / table_counts {counts_json[m].get(name)}"
                                 f" / recomputed from units {exp} disagree")
                stray = points - pts[m]
                if stray:
                    fails.append(f"[t] {name}@{m}: contains points of other shards {sorted(stray)}")
                if revs - {EXPECTED_REVISION}:
                    fails.append(f"[c] {name}@{m}: other design_revision values {sorted(revs - {EXPECTED_REVISION})}")
                if name == "unit_status" and ukeys != units_by_m[m]:
                    fails.append(f"[t] unit_status@{m}: units differ from the shard archive"
                                 f" (table extra {len(ukeys - units_by_m[m])}, archive extra {len(units_by_m[m] - ukeys)})")
        info["tables"][name] = {"rows": sum(per_m.values()), "per_machine": per_m}
        print(f"[merge] {name}: {sum(per_m.values())} rows", flush=True)

    summ = pd.read_csv(os.path.join(tmp, "summary.csv"))
    if set(summ["n_planned_stage"]) != {N_REPS}:
        fails.append(f"[t] summary.n_planned_stage is not always {N_REPS}: {sorted(set(summ['n_planned_stage']))}")
    ov = summ[summ["stage"] == "overall"]
    if set(ov["n_attempted"]) != {N_REPS}:
        bad = ov[ov["n_attempted"] != N_REPS][["point_id", "method", "estimator", "n_attempted"]]
        info["overall_rows_not_500"] = bad.to_dict("records")
    keys = ["point_id", "method", "estimator", "stage", "analysis_cluster"]
    if summ.duplicated(keys).any():
        fails.append("[t] summary has repeated (point, arm, estimator, stage, cluster) rows")

    # derived tables: bias / coverage and paired_diff rows recomputed from the merged effect_raw must equal the shard versions
    eff = pd.read_csv(os.path.join(tmp, "effect_raw.csv"),
                      usecols=["point_id", "param_rep", "data_rep", "method", "estimator",
                               "stage", "analysis_cluster", "error", "ci_valid", "cover_true"],
                      dtype={"analysis_cluster": str, "ci_valid": str, "cover_true": str})
    eff["cov"] = (eff["cover_true"] == "True").astype(float).where(
        (eff["ci_valid"] == "True") & eff["cover_true"].notna())
    re = eff.groupby(keys).agg(bias_re=("error", "mean"), cov_re=("cov", "mean")).reset_index()
    summ["analysis_cluster"] = summ["analysis_cluster"].astype(str)
    j = summ.merge(re, on=keys, how="outer", indicator=True)
    d_bias = (j["bias"] - j["bias_re"]).abs().max()
    d_cov = (j["coverage_true"] - j["cov_re"]).abs().max()
    info["summary_recompute"] = {"unmatched_groups": int((j["_merge"] != "both").sum()),
                                 "max_abs_diff_bias": float(d_bias),
                                 "max_abs_diff_coverage_true": float(d_cov)}
    if (j["_merge"] != "both").any() or not (d_bias < 1e-12 and d_cov < 1e-12):
        fails.append(f"[t] summary differs from the effect_raw recomputation: {info['summary_recompute']}")
    e = eff[eff["error"].notna()]
    pk = ["point_id", "param_rep", "data_rep", "estimator", "stage", "analysis_cluster"]
    n_by = e.groupby(pk).size().rename("n").reset_index()
    dual = e.loc[e["method"] == "dual_icp", pk].drop_duplicates()
    exp_pd = int((n_by.merge(dual, on=pk)["n"] - 1).sum())
    info["paired_diff_recompute"] = {"expected_rows": exp_pd,
                                     "merged_rows": info["tables"]["paired_diff"]["rows"]}
    if exp_pd != info["tables"]["paired_diff"]["rows"]:
        fails.append(f"[t] paired_diff rows differ from the effect_raw recomputation: {info['paired_diff_recompute']}")
    del eff, e

    if not fails:
        final = os.path.join(args.out, "tables")
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
    return fails, info


def main(argv=None) -> int:
    global MACHINES, PKG
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--run-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--machines", required=True, nargs="+", help="Shard IDs used in the input archives and res_<ID> folders")
    ap.add_argument("--package", default=PKG, help="package directory inside res_<ID>")
    args = ap.parse_args(argv)
    if len(set(args.machines)) != len(args.machines) or any(not m or not all(c.isalnum() or c in "-_" for c in m) for m in args.machines):
        ap.error("Shard IDs must be unique and contain only letters, digits, hyphens or underscores.")
    MACHINES = tuple(args.machines)
    PKG = args.package
    os.makedirs(args.out, exist_ok=True)
    run_table = pd.read_csv(args.run_table, dtype={"point_id": str})

    report = {"expected_design_revision": EXPECTED_REVISION, "n_reps": N_REPS}
    fails, report["gate_gamma"], gamma = gate_gamma(args, run_table)
    info: dict = {}
    if gamma is not None:
        fails_u, info, idx = gate_units(args, run_table, gamma)
        fails += fails_u
    report["gates_units"] = info
    if fails:
        report["status"] = "FAIL_BEFORE_MERGE"
    else:
        fails2, info2 = merge(args, idx)
        report["merge"] = info2
        fails += fails2
        report["status"] = "FAIL_AFTER_MERGE" if fails2 else "PASS"
    report["failures"] = fails
    report["manifests"] = {}
    for m in MACHINES:
        with open(os.path.join(_machine_root(args.res_dir, m), "runs", m, "manifest.json"),
                  encoding="utf-8") as f:
            man = json.load(f)
        report["manifests"][m] = {k: man.get(k) for k in (
            "output_version", "design_revision", "run_table_sha256", "gamma_small_frozen",
            "workers", "started_utc",
            "finished_utc", "wall_clock_seconds", "cpu_seconds_upper_bound",
            "units_complete", "units_failed")}
    with open(os.path.join(args.out, "verification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"[verify] {report['status']}")
    for line in fails:
        print("  x", line)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
