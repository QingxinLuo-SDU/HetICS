"""merge_ablation.py — summary tables of the binary ablation from unit records (streamed, nothing extracted).

  python -I merge/merge_ablation.py --source ablation --inputs <run dirs, .tar.gz or .zip> ... --out <dir>
  python -I merge/merge_ablation.py --source formal   --inputs <formal run dirs or archives> ... --out <dir>

--source ablation reads run_ablation.py units (dual_icp, dual_no_icp, dual_global_icp); --source formal reads
run_formal.py units for the one-sided arms (dual_icp, xside_icp, yside_icp). Only the 74 points with frozen
labels (P001-P054, U001-U020) are summarized; other points are skipped and counted.
Output (prefix xy_ for --source formal):
  points.csv     design points and the shard that returned them
  est.csv        point x arm x estimator x level: bias, empirical_se (SD of the errors), mse = bias² + empirical_se²,
                 coverage, type1_error (gamma = 0) or power, n_valid
  sel.csv        point x arm: minimum core recall, safe-set precision, spurious and noise inclusion rates on the
                 union of the clusters, mean adjustment set size per cluster
  paired.csv     point x arm: differences from dual_icp in the same unit (overall ATE, AIPW): squared error,
                 absolute error and coverage, and precision and spurious inclusion, each with its MCSE
  merge_qa.json  units per input, versions, per-point counts, covariate layout checks
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import design_table as dt                     # noqa: E402
from frozen_rerun import STAGES, stage_rows   # noqa: E402
from merge_pooled import UNIT_RE, iter_units  # noqa: E402

SOURCES = {
    "ablation": {"arms": ("dual_icp", "dual_no_icp", "dual_global_icp"),
                 "versions": {"binary-ablation-v6-20261028"}, "prefix": ""},
    "formal": {"arms": ("dual_icp", "xside_icp", "yside_icp"),
               "versions": {STAGES["r1"]["source_output_version"], dt.IMPLEMENTATION_VERSION}, "prefix": "xy_"},
}
ESTIMATORS = ("aipw", "ra_standardized", "psm_caliper")


class Moments:
    __slots__ = ("n", "s1", "s2", "nci", "cov", "rej")

    def __init__(self):
        self.n = self.nci = self.cov = self.rej = 0
        self.s1 = self.s2 = 0.0

    def add(self, row):
        if row["point_status"] != "complete" or row.get("error") is None:
            return
        e = float(row["error"])
        self.n += 1
        self.s1 += e
        self.s2 += e * e
        if row.get("ci_valid"):
            self.nci += 1
            self.cov += bool(row["cover_true"])
            self.rej += bool(row["reject_zero"])


def _sd(s1, s2, k):
    return math.sqrt(max(s2 / k - (s1 / k) ** 2, 0.0) * k / (k - 1)) if k > 1 else math.nan


def _sel_pair(q) -> dict:
    """Mean and MCSE of the paired selection differences (arm - dual_icp)."""
    if not q:
        return dict(d_precision=math.nan, d_precision_mcse=math.nan, d_spurious_recall=math.nan,
                    d_spurious_recall_mcse=math.nan)
    n1, s1, ss1, n2, s2, ss2 = q
    return dict(d_precision=s1 / n1 if n1 else math.nan,
                d_precision_mcse=_sd(s1, ss1, n1) / math.sqrt(n1) if n1 > 1 else math.nan,
                d_spurious_recall=s2 / n2 if n2 else math.nan,
                d_spurious_recall_mcse=_sd(s2, ss2, n2) / math.sqrt(n2) if n2 > 1 else math.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-reps", type=int, default=500, help="units expected per point")
    a = ap.parse_args(argv)
    src = SOURCES[a.source]
    arms, prefix = src["arms"], src["prefix"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pts = stage_rows()
    layout = {p: (int(r["n_Z"]) + int(r["n_causal_shared"]) + int(r["K"]) * int(r["n_causal_specific"]),
                  int(r["n_spurious"]), int(r["n_noise"])) for p, r in pts.items()}
    mom = defaultdict(Moments)                                         # (point, arm, est, level) -> moments
    sel = defaultdict(lambda: {"n": 0, "core": [], "prec": [], "spur": [], "noise": [], "size": []})
    pair = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])   # n, Σd², Σ(d²)², Σ|d|, Σ(|d|)², nci, Σdcov, Σdcov²
    spair = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])            # n_prec, Σdprec, Σdprec², n, Σdspur, Σdspur²
    owner, seen, qa, bad = {}, set(), {"inputs": {}, "skipped_points": 0}, []
    for path in a.inputs:
        n_units, versions = 0, defaultdict(int)
        print(f"[merge] reading {path}", flush=True)
        for name, raw in iter_units(Path(path)):
            machine = UNIT_RE.search(name).group(1)
            d = json.loads(raw.decode("utf-8"))
            pid, key = d["point_id"], d["unit_key"]
            if pid not in pts:
                qa["skipped_points"] += 1
                continue
            n_units += 1
            versions[d.get("output_version")] += 1
            if key in seen:
                raise SystemExit(f"[merge] unit {key} appears twice")
            seen.add(key)
            if owner.setdefault(pid, machine) != machine:
                raise SystemExit(f"[merge] point {pid} is on {owner[pid]} and {machine}")
            if d.get("unit_complete") is not True or d.get("output_version") not in src["versions"]:
                bad.append(key)
                continue
            unit_sel = {}
            for arm in arms:
                for est in ESTIMATORS:
                    for row in d["effects"][arm][est]:
                        mom[(pid, arm, est, row["stage"])].add(row)
                # selection on the union of the clusters; units where the arm is not estimable are skipped
                if d["selections"][arm].get("status") != "complete":
                    continue
                fs = d["feature_selection"][arm]
                n_safe, n_spur, n_noise = layout[pid]
                for k, f in fs.items():
                    got = (f["n_Z"] + f["n_shared"] + f["n_spec_own"] + f["n_spec_foreign"], f["n_spur"], f["n_noise"])
                    if got != layout[pid]:
                        raise SystemExit(f"[merge] {key} {arm} cluster {k}: covariate layout {got} != design {layout[pid]}")
                union = set().union(*(set(v) for v in d["selected_by_cluster"][arm].values()))
                s = sel[(pid, arm)]
                s["n"] += 1
                cores = [f["observed_core_recall"] for f in fs.values() if f.get("observed_core_recall") is not None]
                if cores:
                    s["core"].append(min(cores))
                prec = sum(j < n_safe for j in union) / len(union) if union else None
                if prec is not None:
                    s["prec"].append(prec)
                s["spur"].append(sum(n_safe <= j < n_safe + n_spur for j in union) / n_spur)
                unit_sel[arm] = (prec, s["spur"][-1])
                s["noise"].append(sum(j >= n_safe + n_spur for j in union) / n_noise)
                s["size"].append(sum(len(v) for v in d["selected_by_cluster"][arm].values()) / len(fs))
            for arm in arms[1:]:                       # paired selection differences, arm - dual_icp
                if arm in unit_sel and "dual_icp" in unit_sel:
                    (pa, sa), (pr, sr) = unit_sel[arm], unit_sel["dual_icp"]
                    q = spair[(pid, arm)]
                    if pa is not None and pr is not None:
                        q[0] += 1; q[1] += pa - pr; q[2] += (pa - pr) ** 2
                    q[3] += 1; q[4] += sa - sr; q[5] += (sa - sr) ** 2
            ov = {arm: next(r for r in d["effects"][arm]["aipw"] if r["stage"] == "overall") for arm in arms}
            ref = ov["dual_icp"]
            for arm in arms[1:]:                       # paired estimation differences, overall ATE, AIPW
                r = ov[arm]
                if ref["point_status"] != "complete" or r["point_status"] != "complete":
                    continue
                p = pair[(pid, arm)]
                ds = r["error"] ** 2 - ref["error"] ** 2
                da = abs(r["error"]) - abs(ref["error"])
                p[0] += 1; p[1] += ds; p[2] += ds * ds; p[3] += da; p[4] += da * da
                if ref.get("ci_valid") and r.get("ci_valid"):
                    dc = int(bool(r["cover_true"])) - int(bool(ref["cover_true"]))
                    p[5] += 1; p[6] += dc; p[7] += dc * dc
        qa["inputs"][path] = {"units": n_units, "versions": dict(versions)}
    if bad:
        raise SystemExit(f"[merge] {len(bad)} units incomplete or of another output version, e.g. {bad[:5]}")
    if not seen:
        raise SystemExit("[merge] no unit was read; check --inputs")

    per_point = defaultdict(int)
    for k in seen:
        per_point[k.split("__")[0]] += 1
    wrong = {p: n for p, n in per_point.items() if n != a.n_reps}

    est_rows = []
    for (pid, arm, est, level), m in sorted(mom.items()):
        if m.n < 2:
            continue
        bias = m.s1 / m.n
        esd = math.sqrt(max(m.n / (m.n - 1) * (m.s2 / m.n - bias ** 2), 0.0))
        rate = lambda x: x / m.nci if m.nci else math.nan  # noqa: E731
        gamma0 = float(pts[pid]["gamma_logor"]) == 0.0
        est_rows.append({"point_id": pid, "method": arm, "estimator": est, "level": level,
                         "n_valid": m.n, "n_valid_ci": m.nci, "bias": bias, "empirical_se": esd,
                         "mse": bias ** 2 + esd ** 2, "coverage": rate(m.cov),
                         "type1_error": rate(m.rej) if gamma0 else math.nan,
                         "power": math.nan if gamma0 else rate(m.rej), "machine": owner[pid]})
    mean = lambda xs: sum(xs) / len(xs) if xs else math.nan  # noqa: E731
    sel_rows = [{"point_id": pid, "method": arm, "n_units": s["n"], "core_recall": mean(s["core"]),
                 "precision": mean(s["prec"]), "spurious_recall": mean(s["spur"]), "noise_rate": mean(s["noise"]),
                 "adjustment_set_size": mean(s["size"]), "machine": owner[pid]}
                for (pid, arm), s in sorted(sel.items())]
    pair_rows = [{"point_id": pid, "method": arm, "n_pairs": n,
                  "d_sq_error": s / n, "d_sq_error_mcse": _sd(s, ss, n) / math.sqrt(n),
                  "d_abs_error": ab / n, "d_abs_error_mcse": _sd(ab, aa, n) / math.sqrt(n),
                  "n_pairs_ci": nc, "d_coverage": c / nc if nc else math.nan,
                  "d_coverage_mcse": _sd(c, cc, nc) / math.sqrt(nc) if nc > 1 else math.nan,
                  **_sel_pair(spair.get((pid, arm))), "machine": owner[pid]}
                 for (pid, arm), (n, s, ss, ab, aa, nc, c, cc) in sorted(pair.items())]
    point_rows = [{"point_id": pid, "tier": r["tier"], "scan_parameter": r["scan_parameter"],
                   "scan_value": r["scan_value"], "gamma_logor": r["gamma_logor"],
                   "returned": pid in per_point, "n_units": per_point.get(pid, 0), "machine": owner.get(pid, "")}
                  for pid, r in sorted(pts.items())]
    qa.update(source=a.source, units=len(seen), points_returned=len(per_point),
              points_missing=sorted(set(pts) - set(per_point)), points_not_n_reps=wrong,
              owner=dict(sorted(owner.items())), est_rows=len(est_rows), sel_rows=len(sel_rows),
              pair_rows=len(pair_rows))
    for name, rows in (("points", point_rows), ("est", est_rows), ("sel", sel_rows), ("paired", pair_rows)):
        if not rows:
            continue
        with open(out / f"{prefix}{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / f"{prefix}merge_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[merge] {len(seen)} units, {len(per_point)}/{len(pts)} points"
          + (f"; {len(wrong)} points without {a.n_reps} units" if wrong else "") + f" -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
