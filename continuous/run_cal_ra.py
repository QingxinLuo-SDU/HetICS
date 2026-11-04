"""run_cal_ra.py — calibration experiment (RA and the other estimators): run, verify and combine.

SCENA_CAL_AXIS selects the design (read at import):
  m (default)  centers axis: 14 points (3, 4, 5, 6, 8, 10, 12 centers per cluster x gamma in {0, 0.035})
  n            sample-size axis: 12 centers, 5,000-12,000 individuals per center x gamma in {0, 0.035}
               (12 points; the 10,000 level is the centers-axis point M04)
  p            power along the sample-size axis: 12 centers, 7 sample sizes x gamma = 0.055
Each point has 10 parameter seeds x 100 data seeds. SCENA_CAL_DATA=lo-hi restricts data_index to [lo, hi).
Tasks (point_id, p, d) are split into 1000 crc32 buckets; each replicate is one gzip JSON file
<out>/<point>/p<pp>_d<ddd>.json.gz. SCENA_BETA_ANCHOR must be "2.0,3.0" (the seed namespace of the calibration
design; the coefficient band itself is (1.0, 1.25), see hypothesis_calibration_config).

  python run_cal_ra.py run     --out runs_cal_ra --buckets 0-1000 --workers 12
  python run_cal_ra.py verify  --out runs_cal_ra --buckets 0-1000
  python run_cal_ra.py combine --runs <dir> [<dir> ...] --dest combined_cal_ra

Exit codes as in run_ra_cont.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

from run_ra_cont import N_BUCKETS, _parse_range, _pool, read_rep, write_atomic

_POINTS: dict | None = None
CAL_AXIS = os.environ.get("SCENA_CAL_AXIS", "m")
CAL_DATA = os.environ.get("SCENA_CAL_DATA", "")


def _points() -> dict:
    global _POINTS
    if _POINTS is None:
        import hypothesis_calibration_config as hcc
        if CAL_AXIS not in hcc.CAL_AXES:
            raise SystemExit(f"SCENA_CAL_AXIS must be one of {hcc.CAL_AXES}, got {CAL_AXIS!r}")
        build = {"n": hcc.build_n_axis_points, "p": hcc.build_p_axis_points}.get(CAL_AXIS, hcc.build_calibration_points)
        _POINTS = {p["point_id"]: p for p in build()}
    return _POINTS


def bucket(pid: str, p: int, d: int) -> int:
    return zlib.crc32(f"cal:{pid}:{p}:{d}".encode()) % N_BUCKETS


def all_tasks(lo: int = 0, hi: int = N_BUCKETS):
    import hypothesis_calibration_config as hcc
    n_data = hcc.N_DATA_SEEDS_PER_PARAM
    d_lo, d_hi = (int(x) for x in CAL_DATA.split("-")) if CAL_DATA else (0, n_data)
    if not 0 <= d_lo < d_hi <= n_data:
        raise SystemExit(f"SCENA_CAL_DATA={CAL_DATA} is invalid (expected lo-hi within 0..{n_data})")
    out = [(pid, p, d) for pid in _points() for p in range(hcc.N_PARAM_SEEDS) for d in range(d_lo, d_hi)]
    return [t for t in out if lo <= bucket(*t) < hi]


def rep_path(out_root, pid, p, d) -> Path:
    return Path(out_root) / pid / f"p{p:02d}_d{d:03d}.json.gz"


def rep_complete(path: Path) -> bool:
    import hypothesis_calibration_config as hcc
    from calibration_ra import CAL_RA_VERSION
    try:
        rec = read_rep(path)
    except (OSError, ValueError, EOFError):
        return False
    return rec.get("output_version") == CAL_RA_VERSION and set(rec.get("arms", {})) == set(hcc.ARMS)


def run_task(task):
    pid, p, d, out_root = task
    from calibration_ra import fit_rep
    path = rep_path(out_root, pid, p, d)
    t0 = time.perf_counter()
    try:
        write_atomic(path, fit_rep(_points()[pid], p, d))
        return pid, p, d, "ok", time.perf_counter() - t0
    except Exception as error:                                   # noqa: BLE001
        fail = path.with_name(f"p{p:02d}_d{d:03d}.failed.json")
        fail.parent.mkdir(parents=True, exist_ok=True)
        fail.write_text(json.dumps({"point_id": pid, "param_index": p, "data_index": d,
                                    "reason": f"{type(error).__name__}: {error}",
                                    "traceback": traceback.format_exc()}, ensure_ascii=False), encoding="utf-8")
        return pid, p, d, f"failed: {type(error).__name__}: {error}", time.perf_counter() - t0


def cmd_run(a) -> int:
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool
    lo, hi = _parse_range(a.buckets)
    tasks = all_tasks(lo, hi)
    todo = [t for t in tasks if not rep_complete(rep_path(a.out, *t))]
    print(f"[run] calibration axis {CAL_AXIS}: buckets {lo}-{hi}, data {CAL_DATA or 'all'}, {len(tasks)} replicates, "
          f"{len(todo)} to run, workers={a.workers}", flush=True)
    if not todo:
        return 0
    n_bad, t0 = 0, time.perf_counter()
    ex = _pool(a.workers)
    try:
        futs = [ex.submit(run_task, (*t, a.out)) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            pid, p, d, status, secs = fut.result()
            if status != "ok":
                n_bad += 1
                print(f"[failed] {pid} p{p} d{d}: {status}", flush=True)
            if i % 50 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"[progress] {i}/{len(todo)}  {i / el * 3600:.0f} per hour  about {el / i * (len(todo) - i) / 3600:.1f} h left"
                      f"  (last {secs:.0f} s)  {time.strftime('%F %T')}", flush=True)
    except BrokenProcessPool as error:
        print(f"[fatal] a worker was killed (usually out of memory): {error}. Rerun with fewer workers to resume", flush=True)
        sys.stdout.flush()
        os._exit(9)
    ex.shutdown(wait=True)
    return 0 if n_bad == 0 else 1


def cmd_verify(a) -> int:
    lo, hi = _parse_range(a.buckets)
    tasks = all_tasks(lo, hi)
    missing = [t for t in tasks if not rep_complete(rep_path(a.out, *t))]
    print(f"[verify] calibration axis {CAL_AXIS}, buckets {lo}-{hi}, data {CAL_DATA or 'all'}: expected {len(tasks)}, "
          f"missing {len(missing)}", flush=True)
    return 0 if not missing else 5


def cmd_combine(a) -> int:
    import numpy as np
    import pandas as pd
    import hypothesis_calibration_config as hcc
    rows, seen = [], set()
    for root in a.runs:
        for f in sorted(Path(root).rglob("p*_d*.json.gz")):
            rec = read_rep(f)
            key = (rec["point_id"], rec["param_index"], rec["data_index"])
            if key in seen:
                continue
            seen.add(key)
            for arm, blk in rec["arms"].items():
                for c in blk["clusters"]:
                    for e, b in c["est"].items():
                        rows.append(dict(point_id=key[0], param_index=key[1], data_index=key[2],
                                         gamma_true=rec["gamma_true"], centers_per_cluster=rec["centers_per_cluster"],
                                         n_per_center=rec.get("n_per_center", hcc.N_PER_CENTER),
                                         arm=arm, k_true=c["k_true"], estimator=e, status=b.get("status"),
                                         theta_hat=b.get("theta"), se=b.get("se"),
                                         reject0_t=b.get("reject0_t"), covered_t=b.get("covered_t"),
                                         reject0_z=b.get("reject0_z"), covered_z=b.get("covered_z")))
    raw = pd.DataFrame(rows)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    raw.to_csv(dest / "raw_cal_ra.csv.gz", index=False)
    df = raw.copy()
    for c in ("reject0_t", "covered_t", "reject0_z", "covered_z"):
        df[c] = df[c].map({True: 1.0, False: 0.0})
    df["err"] = df.theta_hat.astype(float) - df.gamma_true
    keys = ["point_id", "gamma_true", "centers_per_cluster", "n_per_center", "arm", "estimator"]
    cols = ["err", "se", "reject0_t", "covered_t", "reject0_z", "covered_z"]
    per = df.groupby(keys + ["param_index"])[cols].mean()
    g = per.groupby(level=keys)
    out = g.mean().rename(columns={"err": "bias", "se": "mean_model_se", "reject0_t": "reject_rate_t",
                                   "covered_t": "coverage_t", "reject0_z": "reject_rate_z", "covered_z": "coverage_z"})
    out["reject_rate_t_mc_se"] = g.std(ddof=1)["reject0_t"] / np.sqrt(g.size())
    out["empirical_sd"] = df.groupby(keys).theta_hat.std(ddof=1)
    out["n_ok"] = df[df.status == "ok"].groupby(keys).size()
    out = out.reset_index()
    out["test"] = np.where(out.gamma_true == 0, "type1_error", "power")
    out.to_csv(dest / "summary_cal_ra.csv", index=False)
    n = raw.drop_duplicates(["point_id", "param_index", "data_index"]).groupby("point_id").size()
    m_ids = {p["point_id"] for p in hcc.build_calibration_points()}
    n_ids = {p["point_id"] for p in hcc.build_n_axis_points()}
    p_ids = {p["point_id"] for p in hcc.build_p_axis_points()}
    present = set(n.index)
    axes = {"centers axis": m_ids <= present, "sample-size axis": n_ids <= present, "sample-size power": p_ids <= present}
    print(f"[combine] {len(raw)} rows; replicates per point {n.to_dict()}; complete designs "
          f"{[k for k, v in axes.items() if v]} -> {dest}", flush=True)
    whole = all(s <= present or not s & present for s in (m_ids, n_ids, p_ids)) and present <= m_ids | n_ids | p_ids
    return 0 if (n == hcc.N_PARAM_SEEDS * hcc.N_DATA_SEEDS_PER_PARAM).all() and whole else 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibration experiment of the continuous branch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="runs_cal_ra")
        s.add_argument("--buckets", default=f"0-{N_BUCKETS}")
        if name == "run":
            s.add_argument("--workers", type=int, required=True)
    c = sub.add_parser("combine")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--dest", default="combined_cal_ra")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "verify": cmd_verify, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
