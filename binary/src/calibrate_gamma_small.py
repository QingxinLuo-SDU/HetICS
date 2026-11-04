"""calibrate_gamma_small.py — calibration of the small gamma of the calibration panel (chosen once and frozen).

With gamma = +1 the effect is about 48 times its SE and the power is always 1, so that half of the panel
carries no information. A small gamma with effect/SE of about 2-3 (power of a two-sided 5% test of about
0.5-0.85) is chosen instead.

This calibrates a design parameter (the size of the effect put into the DGP) and touches no critical
value. The calibration data do not overlap the formal runs (param_rep 9001-9040 vs 1-500; the derived seeds
are checked one by one).

Steps:
  1. if a frozen file exists, exit 0 without choosing again;
  2. 40 param_reps on the anchor structure (template of S01, 12 centers x 8000); each rep generates the
     continuous data once and, for every gamma of the grid, the binary data, oracle_reference (true
     partition and true core) and AIPW. The binary layer uses the same uniforms for every gamma, so δ(γ) is
     a smooth curve;
  3. δ(γ) = median_r(ate_do_r / se_r); the grid value in [2, 3] closest to 2.5 is chosen (ties: the smaller);
  4. results/gamma_small_frozen.json (grid table, rule, seeds, environment and code hashes) and a per-rep
     CSV are written; run_formal replaces the sentinel gamma of the S rows with the frozen value.

On failure (no eligible gamma, too few valid reps, δ not monotone) nothing is frozen, *_failed.json is
written and the exit code is 7; the S points then do not run and the other points are unaffected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import design_table as dt                                        # noqa: E402

EXIT_NO_ELIGIBLE = 7
Z975 = 1.959963984540054


# ---------------------------------------------------------------------------
# Pure functions: summary and choice (tested directly)
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def grid_table(records: list[dict], grid: list[str]) -> list[dict]:
    """Per-gamma summary: δ (median of truth/SE), mean z, observed rejection rate, predicted power."""
    out = []
    for g in grid:
        rows = [r for r in records if r["gamma_logor"] == g]
        ok = [r for r in rows if r["ok"]]
        ratio = [r["ate_do"] / r["se"] for r in ok]
        z = [r["estimate"] / r["se"] for r in ok]
        delta = float(np.median(ratio)) if ok else float("nan")
        out.append({
            "gamma_logor": g, "n_reps": len(rows), "n_valid": len(ok),
            "delta_median": delta,
            "delta_q10": float(np.quantile(ratio, 0.1)) if ok else float("nan"),
            "delta_q90": float(np.quantile(ratio, 0.9)) if ok else float("nan"),
            "z_mean": float(np.mean(z)) if ok else float("nan"),
            "reject_rate": float(np.mean([abs(v) > Z975 for v in z])) if ok else float("nan"),
            "power_predicted": (_norm_cdf(delta - Z975) + _norm_cdf(-delta - Z975)
                                if ok else float("nan")),
            "ate_do_median": float(np.median([r["ate_do"] for r in ok])) if ok else float("nan"),
            "se_median": float(np.median([r["se"] for r in ok])) if ok else float("nan"),
        })
    return out


def select_gamma(table: list[dict], *, target: float, band: tuple[float, float],
                 min_valid_fraction: float) -> tuple[dict | None, list[str]]:
    """Choose a grid value by the fixed rule; returns (chosen row or None, reasons)."""
    reasons = []
    low = [t["gamma_logor"] for t in table
           if t["n_valid"] < min_valid_fraction * max(t["n_reps"], 1)]
    if low:
        reasons.append(f"gamma values with a valid fraction below {min_valid_fraction}: {low}")
    deltas = [t["delta_median"] for t in table]
    if any(not math.isfinite(d) for d in deltas) or any(
            b < a for a, b in zip(deltas, deltas[1:])):
        reasons.append(f"δ is not finite and nondecreasing in gamma: {[round(d, 3) for d in deltas]}")
    if reasons:
        return None, reasons
    eligible = [t for t in table if band[0] <= t["delta_median"] <= band[1]]
    if not eligible:
        return None, [f"no gamma has δ in {list(band)}: "
                      f"{[(t['gamma_logor'], round(t['delta_median'], 3)) for t in table]}"]
    best = min(eligible, key=lambda t: (abs(t["delta_median"] - target),
                                        float(t["gamma_logor"])))
    return best, []


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def calibration_groups(rows: list[dict], cal: dict) -> tuple[list[dict], dict]:
    template = cal["template_point"]
    structures = dt.build_structures(rows)
    sid = dt.structure_of(rows)[template]
    groups = []
    for rep in cal["param_reps"]:
        seeds = dt.base_seeds(sid, rep, cal["data_rep"])
        groups.append({
            "base_group_key": dt.base_group_key(sid, rep, cal["data_rep"]),
            "structure_id": sid, "param_rep": int(rep), "data_rep": int(cal["data_rep"]),
            "param_seed": seeds["param_seed"], "data_seed": seeds["data_seed"],
            "u_epsilon_seed": dt.u_epsilon_seed(sid, rep, cal["data_rep"]),
            "sig_x_spec": structures[sid]["sig_x_spec"], "point_ids": [template]})
    return groups, {sid: structures[sid]}


def seed_disjointness(rows: list[dict], cal: dict, formal_param_reps: list[int],
                      formal_data_reps: list[int]) -> dict:
    """Calibration seeds against the formal runs (all 13 calibration structures x 1..500); they must not overlap."""
    overlap_reps = sorted(set(cal["param_reps"]) & set(formal_param_reps))
    point_to_structure = dt.structure_of(rows)
    structs = sorted({point_to_structure[r["point_id"]] for r in rows
                      if r["tier"] in (dt.GAMMA_SMALL_TIER, "calibration_added_null")})
    groups, _ = calibration_groups(rows, cal)
    cal_seeds = {g["param_seed"] for g in groups} | {g["data_seed"] for g in groups}
    formal_seeds = set()
    for sid in structs:
        for rep in formal_param_reps:
            for d in formal_data_reps:
                s = dt.base_seeds(sid, rep, d)
                formal_seeds |= {s["param_seed"], s["data_seed"]}
    record = {"calibration_param_reps": f"{min(cal['param_reps'])}-{max(cal['param_reps'])}",
              "formal_param_reps": f"{min(formal_param_reps)}-{max(formal_param_reps)}",
              "overlapping_param_reps": overlap_reps,
              "n_calibration_seeds": len(cal_seeds),
              "n_formal_seeds_checked": len(formal_seeds),
              "n_formal_structures_checked": len(structs),
              "overlapping_seeds": len(cal_seeds & formal_seeds)}
    if overlap_reps or record["overlapping_seeds"]:
        raise RuntimeError(f"calibration seeds overlap the formal runs: {record}")
    return record


# ---------------------------------------------------------------------------
# One param_rep (worker)
# ---------------------------------------------------------------------------
def calibrate_rep(payload) -> list[dict]:
    group, structures, row, grid, backend = payload
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    import formal_dgp
    from aipw_sandwich import fit_partition_aipw
    from run_formal import _reference_selection

    _gen, b6 = formal_dgp.load_sources()
    continuous, _meta = formal_dgp.generate_base(group, structures)
    out = []
    for g in grid:
        rec = {"gamma_logor": g, "param_rep": group["param_rep"],
               "param_seed": group["param_seed"], "data_seed": group["data_seed"],
               "ok": False}
        try:
            observed, truth, _m = formal_dgp.binary_unit(
                continuous, {**row, "gamma_logor": g}, group, b6)
            env = np.asarray(observed.center_codes)
            mapping, sel = _reference_selection(truth, env)
            if mapping is None:
                raise RuntimeError(sel.get("reason"))
            res = fit_partition_aipw(
                np.asarray(observed.features, float), np.asarray(observed.treatment).astype(int),
                np.asarray(observed.outcome).astype(int), env, mapping,
                sel["selected_by_cluster"], backend=backend)
            est, se = res.get("estimate"), res.get("se")
            rec.update(ate_do=float(truth["ate_do"]),
                       estimate=None if est is None else float(est),
                       se=None if se is None else float(se),
                       se_status=res.get("se_status"))
            rec["ok"] = bool(est is not None and se is not None and np.isfinite(est)
                             and np.isfinite(se) and se > 0)
        except Exception as error:                              # noqa: BLE001
            rec["error"] = f"{type(error).__name__}: {error}"
        out.append(rec)
    del continuous
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_json(path: str, payload: dict) -> None:
    from run_formal import write_atomic
    write_atomic(path, payload)


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("calibrate_gamma_small.py", "formal_dgp.py", "binary_with_u.py",
                 "aipw_sandwich.py", "design_table.py"):
        with open(os.path.join(_HERE, name), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def _environment() -> dict:
    import scipy
    import statsmodels
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "platform": platform.platform(), "hostname": socket.gethostname(),
            "machine": os.environ.get("MACHINE", "")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibration of the small gamma (chosen once and frozen)")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(_HERE), "config", "formal_r2.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frozen", default=dt.GAMMA_SMALL_FROZEN_PATH)
    args = parser.parse_args(argv)

    if os.path.isfile(args.frozen):
        frozen = dt.read_gamma_small(args.frozen)
        print(f"[calib] already frozen: gamma_logor={frozen['gamma_logor_str']}"
              f" ({frozen['frozen_utc']}, {frozen.get('environment', {}).get('machine', 'not published')}); not choosing again",
              flush=True)
        return 0

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    cal = config["gamma_small_calibration"]
    rows = dt.load_rows(frozen_path=args.frozen)
    row = next(r for r in rows if r["point_id"] == cal["template_point"])
    disjoint = seed_disjointness(rows, cal, config["param_reps"], config["data_reps"])
    print(f"[calib] seeds do not overlap: {disjoint}", flush=True)

    groups, structures = calibration_groups(rows, cal)
    payloads = [(g, structures, row, cal["grid"], config["logistic_backend"])
                for g in groups]
    workers = max(1, min(int(args.workers), len(payloads), os.cpu_count() or 1))
    print(f"[calib] {len(groups)} param_reps x {len(cal['grid'])} gamma values, "
          f"arm={cal['arm']} estimator={cal['estimator']} workers={workers}", flush=True)
    t0 = time.perf_counter()
    records: list[dict] = []
    if workers == 1:
        for p in payloads:
            records.extend(calibrate_rep(p))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for i, recs in enumerate(pool.imap_unordered(calibrate_rep, payloads), 1):
                records.extend(recs)
                print(f"[calib] {i}/{len(payloads)} param_reps done", flush=True)
    elapsed = time.perf_counter() - t0
    records.sort(key=lambda r: (r["param_rep"], float(r["gamma_logor"])))

    results_dir = os.path.dirname(os.path.abspath(args.frozen))
    os.makedirs(results_dir, exist_ok=True)
    reps_csv = os.path.join(results_dir, "gamma_small_calibration_reps.csv")
    fields = ["gamma_logor", "param_rep", "param_seed", "data_seed", "ok",
              "ate_do", "estimate", "se", "se_status", "error"]
    with open(reps_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    table = grid_table(records, cal["grid"])
    best, reasons = select_gamma(table, target=float(cal["target_ratio"]),
                                 band=tuple(cal["band"]),
                                 min_valid_fraction=float(cal["min_valid_fraction"]))
    for t in table:
        print(f"[calib]  γ={t['gamma_logor']}  δ={t['delta_median']:.3f}  "
              f"predicted power={t['power_predicted']:.3f}  rejection rate={t['reject_rate']:.3f}  "
              f"valid {t['n_valid']}/{t['n_reps']}", flush=True)
    record = {
        "parameter": "gamma_logor", "applies_to_tier": dt.GAMMA_SMALL_TIER,
        "applies_to_points": [r["point_id"] for r in rows if r["tier"] == dt.GAMMA_SMALL_TIER],
        "design_revision": dt.DESIGN_REVISION,
        "output_version": dt.IMPLEMENTATION_VERSION,
        "run_table_sha256": dt.sha256_of(dt.RUN_TABLE_PATH),
        "calibration_code_sha256": _code_sha256(),
        "calibration_spec": cal,
        "seed_disjointness": disjoint,
        "structure_id": groups[0]["structure_id"],
        "grid_table": table,
        "per_rep_csv": os.path.basename(reps_csv),
        "elapsed_seconds": elapsed,
        "environment": _environment(),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if best is None:
        record.update(status="failed", reasons=reasons)
        _write_json(os.path.join(results_dir, "gamma_small_calibration_failed.json"), record)
        print("[calib] no eligible gamma; nothing frozen and the S points do not run. Reasons:", file=sys.stderr)
        for line in reasons:
            print("   ", line, file=sys.stderr)
        return EXIT_NO_ELIGIBLE

    record.update(
        status="frozen", gamma_logor=float(best["gamma_logor"]),
        gamma_logor_str=best["gamma_logor"], selected_row=best,
        parameter_settings=[{
            "scenario_name": dt.GAMMA_SMALL_TIER, "tier": dt.GAMMA_SMALL_TIER,
            "parameter_name": "gamma_logor", "parameter_value": best["gamma_logor"],
            "parameter_status": "tuned",
            "setting_meaning": "true treatment effect (log-OR) of the power half of the calibration panel",
            "why_it_matters": (f"at gamma=+1 effect/SE is about 48 and the power is 1; this value gives "
                               f"δ={best['delta_median']:.3f} and predicted power="
                               f"{best['power_predicted']:.3f} at the anchor, so the power curves over the "
                               "numbers of centers and sample sizes have a slope")}])
    _write_json(args.frozen, record)
    print(f"[calib] frozen gamma_logor={best['gamma_logor']}"
          f" (δ={best['delta_median']:.3f}, predicted power={best['power_predicted']:.3f})"
          f" -> {args.frozen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
