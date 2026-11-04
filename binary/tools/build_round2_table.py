"""build_round2_table.py — round-2 run table (59 points) derived from the v4 master table, then frozen.

Round 2 = calibration at gamma=0 (C001-C012) + calibration at a small gamma (S01-S13, new)
        + delta sensitivity (P041-P047) + correlation sensitivity (P048-P054) + all 20 U points.

Fields changed relative to the v4 rows (provenance test in tests/test_round2_design.py):
  * every row: SIG_Y 2->1, n_reps 1000->500, param_reps 100->500, data_reps 10->1, design_revision,
    planned_estimator_attempts.

Each S row (one per calibration structure) copies the gamma=0 row of its structure: S01 <- P004 (anchor),
S02..S13 <- C001..C012. gamma_logor holds the sentinel `@gamma_small`, replaced in design_table.load_rows() by
the value frozen by calibrate_gamma_small.py.

Usage (from the package root): python -B tools/build_round2_table.py
"""
from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, "config", "formal_run_table_v4.csv")
OUT = os.path.join(ROOT, "config", "formal_run_table_r2.csv")

REVISION = "r2_cal3_sgamma_sens_sigy1_500_20261016"
SENTINEL = "@gamma_small"
CAL_ARMS = "dual_icp|oracle_reference"
REPS = 500

# S point <- template row (gamma=0 point of the same structure); axis names follow the v4 calibration axes
S_TEMPLATES = [("S01", "P004", "CALS_E02|CALS_F04")] + [
    (f"S{i + 2:02d}", f"C{i + 1:03d}", None) for i in range(12)]


def main() -> int:
    with open(V4, encoding="utf-8-sig", newline="") as handle:
        v4 = list(csv.DictReader(handle))
    fields = list(v4[0])
    by_id = {r["point_id"]: r for r in v4}

    def common(row: dict) -> dict:
        row = dict(row)
        row.update(SIG_Y="1", n_reps=str(REPS), param_reps=str(REPS), data_reps="1",
                   design_revision=REVISION)
        return row

    def attempts(row: dict) -> dict:
        row["planned_estimator_attempts"] = str(
            int(row["n_methods"]) * int(row["n_estimators"]) * REPS)
        return row

    out = []
    for i in range(1, 13):
        row = common(by_id[f"C{i:03d}"])
        row.update(execution_methods=CAL_ARMS, n_methods="2")
        out.append(attempts(row))
    for sid, template, axis in S_TEMPLATES:
        row = common(by_id[template])
        row.update(
            point_id=sid,
            axis_membership=axis or by_id[template]["axis_membership"].replace("CAL_", "CALS_"),
            tier="calibration_small_gamma", scan_parameter="gamma_logor",
            scan_value=SENTINEL, gamma_logor=SENTINEL, status="design_frozen_not_run",
            calibration_member="true", report_panels="calibration",
            execution_methods=CAL_ARMS, n_methods="2",
            pair_reference_point_id=template)
        out.append(attempts(row))
    for i in list(range(41, 55)):
        out.append(attempts(common(by_id[f"P{i:03d}"])))
    for i in range(1, 21):
        out.append(attempts(common(by_id[f"U{i:03d}"])))

    # the csv module writes CRLF line endings; the sha256 is checked on the bytes, so *.csv must not be normalized
    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {OUT}: {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
