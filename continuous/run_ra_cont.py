"""run_ra_cont.py — continuous simulation: run, verify and combine.

A task is (combo, param_index, data_index); tasks are split into 1000 crc32 buckets and each machine runs a
bucket range (--buckets lo-hi). Each replicate is one gzip JSON file
runs_ra/<experiment>/<combo>/p<pp>_d<ddd>.json.gz, written atomically; completed replicates are skipped.

  python run_ra_cont.py run     --out runs_ra --buckets 0-417 --workers 30
  python run_ra_cont.py verify  --out runs_ra --buckets 0-417
  python run_ra_cont.py combine --runs <dir> [<dir> ...] --dest combined_ra

The experiment of run/verify is set by SCENA_EXPERIMENT (grid / u07); SCENA_BETA_ANCHOR must be "1.0,1.25".
Exit codes: 0 success; 1 failed replicates; 5 incomplete; 9 a worker was killed (usually out of memory; rerun
with fewer workers to resume).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

N_BUCKETS = 1000
_COMBOS: dict | None = None


def _combos() -> dict:
    global _COMBOS
    if _COMBOS is None:
        import step6x_config as cfg
        import step6x_config_v4 as cfg4
        if cfg.BETA_TAG != "b1-125":
            raise RuntimeError(f"SCENA_BETA_ANCHOR gives {cfg.BETA_TAG}; expected b1-125 (1.0,1.25)")
        _COMBOS = {c["combo_id"]: c for c in cfg4.build_combos_v4(cfg.BETA_TAG)}
    return _COMBOS


def bucket(combo_id: str, p: int, d: int) -> int:
    return zlib.crc32(f"{combo_id}:{p}:{d}".encode()) % N_BUCKETS


def all_tasks(lo: int = 0, hi: int = N_BUCKETS, combos: str | None = None) -> list[tuple[str, int, int]]:
    """combos: comma-separated prefixes of design points (e.g. "A,B,C"); None = all."""
    import step6x_config_v4 as cfg4
    keep = tuple(x.strip() for x in combos.split(",") if x.strip()) if combos else ()
    out = [(cid, p, d) for cid in _combos() if not keep or cid.startswith(keep)
           for p in range(cfg4.N_PARAM_SEEDS) for d in range(cfg4.N_DATA_SEEDS_PER_PARAM)]
    return [t for t in out if lo <= bucket(*t) < hi]


def rep_path(out_root: str, exp: str, cid: str, p: int, d: int) -> Path:
    return Path(out_root) / exp / cid / f"p{p:02d}_d{d:03d}.json.gz"


def read_rep(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as h:
        return json.load(h)


def rep_complete(path: Path) -> bool:
    from ra_core import CLUSTER_ARMS, POOLED_ARMS, RA_VERSION
    try:
        rec = read_rep(path)
    except (OSError, ValueError, EOFError):
        return False
    return rec.get("output_version") == RA_VERSION and set(rec.get("arms", {})) == set(CLUSTER_ARMS + POOLED_ARMS)


def write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as h:
        json.dump(obj, h, ensure_ascii=False, allow_nan=True)
    os.replace(tmp, path)


def run_task(task) -> tuple:
    cid, p, d, out_root = task
    import step6x_config_v4 as cfg4
    from ra_core import fit_rep
    from v4_seeds import seed_key_for
    path = rep_path(out_root, cfg4.EXPERIMENT, cid, p, d)
    t0 = time.perf_counter()
    try:
        combo = _combos()[cid]
        rec = fit_rep(combo, p, d, seed_key_for(combo, p, d))
        write_atomic(path, rec)
        return cid, p, d, "ok", time.perf_counter() - t0, rec.get("peak_rss_gb")
    except Exception as error:                                   # noqa: BLE001
        fail = path.with_name(f"p{p:02d}_d{d:03d}.failed.json")
        fail.parent.mkdir(parents=True, exist_ok=True)
        fail.write_text(json.dumps({"combo_id": cid, "param_index": p, "data_index": d,
                                    "reason": f"{type(error).__name__}: {error}",
                                    "traceback": traceback.format_exc()}, ensure_ascii=False), encoding="utf-8")
        return cid, p, d, f"failed: {type(error).__name__}: {error}", time.perf_counter() - t0, None


def _pool(workers: int):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))


def _parse_range(text: str) -> tuple[int, int]:
    lo, hi = (int(x) for x in text.split("-"))
    if not (0 <= lo < hi <= N_BUCKETS):
        raise SystemExit(f"--buckets {text} is invalid (expected lo-hi within 0..{N_BUCKETS})")
    return lo, hi


def cmd_run(a) -> int:
    import step6x_config_v4 as cfg4
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool
    exp = cfg4.EXPERIMENT
    lo, hi = _parse_range(a.buckets)
    tasks = all_tasks(lo, hi, a.combos)
    todo = [t for t in tasks if not rep_complete(rep_path(a.out, exp, *t))]
    print(f"[run] {exp} (points {a.combos or 'all'}): buckets {lo}-{hi}, {len(tasks)} replicates, {len(todo)} to run, "
          f"workers={a.workers}", flush=True)
    if not todo:
        return 0
    n_ok = n_bad = 0
    t0 = time.perf_counter()
    ex = _pool(a.workers)
    try:
        futs = [ex.submit(run_task, (*t, a.out)) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            cid, p, d, status, secs, rss = fut.result()
            if status == "ok":
                n_ok += 1
            else:
                n_bad += 1
                print(f"[failed] {cid} p{p} d{d}: {status}", flush=True)
            if i % 100 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                eta = el / i * (len(todo) - i) / 3600
                print(f"[progress] {exp} {i}/{len(todo)}  {i / el * 3600:.0f} per hour  about {eta:.1f} h left"
                      f"  (last {secs:.0f} s, peak {rss} GB)  {time.strftime('%F %T')}", flush=True)
    except BrokenProcessPool as error:
        print(f"[fatal] a worker was killed (usually out of memory): {error}. Completed replicates are on disk; "
              "rerun with fewer workers to resume",
              flush=True)
        sys.stdout.flush()
        os._exit(9)                       # executor shutdown can hang after a killed worker (CPython gh-94777)
    ex.shutdown(wait=True)
    print(f"[run] {exp} finished: {n_ok} succeeded, {n_bad} failed", flush=True)
    return 0 if n_bad == 0 else 1


def cmd_verify(a) -> int:
    import step6x_config_v4 as cfg4
    exp = cfg4.EXPERIMENT
    lo, hi = _parse_range(a.buckets)
    tasks = all_tasks(lo, hi, a.combos)
    missing = [t for t in tasks if not rep_complete(rep_path(a.out, exp, *t))]
    by_combo: dict[str, int] = {}
    for cid, _p, _d in missing:
        by_combo[cid] = by_combo.get(cid, 0) + 1
    print(f"[verify] {exp} buckets {lo}-{hi}: expected {len(tasks)}, missing {len(missing)}"
          + (f" (by point: {dict(sorted(by_combo.items()))})" if missing else ""), flush=True)
    return 0 if not missing else 5


# Combination of the replicate files.
def _est_row(b):
    ci = b.get("ci_t") or [math.nan, math.nan]
    return dict(status=b.get("status"), theta_hat=b.get("theta"), se=b.get("se"), ci_t_lo=ci[0], ci_t_hi=ci[1],
                covered_t=b.get("covered_t"), reject0_t=b.get("reject0_t"))


# Selection metrics
#   recall      each non-empty learned cluster aligned to a true cluster k: |S_k_hat & S_k*| / |S_k*|, minimum over
#               clusters; missing when a non-empty cluster is unmapped. All-center arms: the single set is compared
#               with every S_k* and the minimum is taken.
#   precision   S_hat = union of the cluster-specific sets; |S_hat & S_safe| / |S_hat|; missing when S_hat is empty
#               (recorded in empty_selection)
#   spurious / noise selection rate   |S_hat & class| / size of the class
#   Oracle arms store no selection. Summaries average within parameter seeds and then equally across them.
SEL_METRICS = ("recall_min", "all_core", "precision", "spur_rate", "noise_rate", "n_union", "empty_selection")


def truth_sets(p: int, K: int = 3) -> dict:
    """Covariate roles from the fixed column layout of the DGP: Z, shared, spec_0..spec_{K-1}, spur, noise.

    Block sizes follow _p_scaled_params (20/25/5/50 at p = 500).
    """
    import run_scenarioA_main as off
    pr, _ = off._p_scaled_params(int(p), {"K": K})
    nz, nsh, nsp, nsu = pr["n_Z"], pr["n_causal_shared"], pr["n_causal_specific"], pr["n_spurious"]
    base = set(range(nz + nsh))
    spec = {k: set(range(nz + nsh + k * nsp, nz + nsh + (k + 1) * nsp)) for k in range(K)}
    lo = nz + nsh + K * nsp
    core = {k: base | spec[k] for k in range(K)}
    return dict(core=core, safe=set().union(*core.values()), spur=set(range(lo, lo + nsu)),
                noise=set(range(lo + nsu, int(p))))


def selection_row(arm: dict, truth: dict) -> dict | None:
    """Selection metrics of one arm in one replicate (None for oracle arms); checks the stored counts."""
    nan = float("nan")
    core, safe = truth["core"], truth["safe"]
    if arm["scope"] == "all_centers_pooled":
        if "selected" not in arm:
            return None
        union = set(arm["selected"])
        assert abs(arm["recall_core"] - len(union & safe) / len(safe)) < 1e-12 and \
            (arm["n_spur"], arm["n_noise"]) == (len(union & truth["spur"]), len(union & truth["noise"])), \
            "truth sets disagree with the stored counts (all-center arm)"
        per, n_unmapped = [len(union & c) / len(c) for c in core.values()], 0
    else:
        live = [c for c in arm["clusters"] if c["M_k"] > 0]
        if any("selected" not in c for c in live):
            return None
        n_unmapped = sum(c["k_true"] < 0 for c in live)
        per = []
        for c in live:
            S = set(c["selected"])
            assert (c["n_spur"], c["n_noise"]) == (len(S & truth["spur"]), len(S & truth["noise"])), \
                "truth sets disagree with the stored spurious / noise counts"
            if c["k_true"] >= 0:
                r = len(S & core[c["k_true"]]) / len(core[c["k_true"]])
                assert abs(r - c["recall_core"]) < 1e-12, "truth sets disagree with the stored cluster recall"
                per.append(r)
        union = set().union(*(set(c["selected"]) for c in live))
    usable = bool(per) and n_unmapped == 0
    rmin = min(per) if usable else nan
    return dict(recall_min=rmin, all_core=(float(rmin == 1.0) if usable else nan),
                precision=(len(union & safe) / len(union) if union else nan),
                spur_rate=len(union & truth["spur"]) / len(truth["spur"]),
                noise_rate=(len(union & truth["noise"]) / len(truth["noise"]) if truth["noise"] else nan),
                n_union=float(len(union)), empty_selection=float(not union), n_clusters_unmapped=n_unmapped)


def summarize_selection(sel):
    keys = ["experiment", "combo_id", "method", "scope"]
    out = _two_level(sel, keys, list(SEL_METRICS))
    cnt = sel.groupby(keys).agg(n_reps=("recall_min", "size"), n_recall_defined=("recall_min", "count"),
                                n_precision_defined=("precision", "count"), true_gamma=("true_gamma", "first"),
                                p=("p", "first"))
    return out.join(cnt).reset_index()


def load_raw(run_dirs, with_selection: bool = False, version: str | None = None):
    """Long tables theta (cluster-specific and all-center estimates) and mu (overall effect of the cluster arms);
    with_selection adds the per-replicate selection metrics."""
    import pandas as pd
    th, mu, sel, seen, dup, old = [], [], [], set(), 0, 0
    truth_by_p: dict[int, dict] = {}
    for root in run_dirs:
        for f in sorted(Path(root).rglob("p*_d*.json.gz")):
            rec = read_rep(f)
            if version and rec.get("output_version") != version:
                old += 1
                continue
            key = (rec["experiment"], rec["combo_id"], rec["param_index"], rec["data_index"])
            if key in seen:            # the same replicate run on two machines; results are identical
                dup += 1
                continue
            seen.add(key)
            base = dict(experiment=key[0], combo_id=key[1], param_index=key[2], data_index=key[3],
                        true_gamma=rec["true_gamma"], p=rec["p"], n_centers=rec["n_centers"])
            if with_selection:
                truth = truth_by_p.setdefault(int(rec["p"]), truth_sets(rec["p"]))
                for m, arm in rec["arms"].items():
                    r = selection_row(arm, truth)
                    if r is not None:
                        sel.append(dict(base, method=m, scope=arm["scope"], **r))
            for m, arm in rec["arms"].items():
                units = arm["clusters"] if "clusters" in arm else [dict(arm, k=-1, k_true=-1)]
                for c in units:
                    for e, b in c["est"].items():
                        th.append(dict(base, method=m, scope=arm["scope"], k=c["k"], k_true=c.get("k_true"),
                                       estimator=e, n_selected=c.get("n_selected"), recall_core=c.get("recall_core"),
                                       n_spur=c.get("n_spur"), n_noise=c.get("n_noise"), **_est_row(b)))
                for e, agg in arm.get("mu", {}).items():
                    r = agg.get("reml_gls", {}) if agg.get("status") == "ok" else {}
                    mu.append(dict(base, method=m, estimator=e, status=agg.get("status"),
                                   mu_reml=r.get("mu"), var_mu_reml=r.get("var_mu"), tau2=r.get("tau2")))
    if dup:
        print(f"[combine] skipped {dup} duplicate replicates", flush=True)
    if old:
        print(f"[combine] skipped {old} replicates with output_version other than {version}", flush=True)
    if with_selection:
        return pd.DataFrame(th), pd.DataFrame(mu), pd.DataFrame(sel)
    return pd.DataFrame(th), pd.DataFrame(mu)


def _two_level(df, keys, value_cols, sq_col=None):
    """Average within parameter seeds, then equally across them; MC SE = SD across parameter seeds / sqrt(n)."""
    import numpy as np
    per = df.groupby(keys + ["param_index"])[value_cols].mean()
    g = per.groupby(level=keys)
    mean, sd, n = g.mean(), g.std(ddof=1), g.size()
    out = mean.copy()
    for c in value_cols:
        out[f"{c}_mc_se"] = sd[c] / np.sqrt(n)
    out["n_param_seeds"] = n
    return out


def summarize_theta(th):
    import numpy as np
    df = th.copy()
    ok = df.status == "ok"
    df["err"] = np.where(ok, df.theta_hat.astype(float) - df.true_gamma, np.nan)
    df["sq_err"] = df.err ** 2
    for c in ("covered_t", "reject0_t"):
        df[c] = df[c].map({True: 1.0, False: 0.0})
    cols = ["err", "sq_err", "se", "covered_t", "reject0_t", "n_selected", "recall_core", "n_spur", "n_noise"]
    df[cols] = df[cols].astype(float)
    keys = ["experiment", "combo_id", "method", "scope", "estimator"]
    out = _two_level(df, keys, cols)
    out["rmse"] = np.sqrt(out["sq_err"])
    out["empirical_sd"] = np.sqrt(np.maximum(out["sq_err"] - out["err"] ** 2, 0.0))
    out["sd_over_se"] = out["empirical_sd"] / out["se"]
    out = out.rename(columns={"err": "bias", "se": "mean_model_se", "covered_t": "coverage_t", "reject0_t": "reject0"})
    cnt = df.groupby(keys).agg(n_rows=("status", "size"), n_ok=("status", lambda s: int((s == "ok").sum())),
                               true_gamma=("true_gamma", "first"))
    return out.join(cnt).reset_index()


def summarize_mu(mu):
    import numpy as np
    df = mu.copy()
    df["err"] = df.mu_reml.astype(float) - df.true_gamma
    df["sq_err"] = df.err ** 2
    keys = ["experiment", "combo_id", "method", "estimator"]
    out = _two_level(df, keys, ["err", "sq_err"])
    out["rmse"] = np.sqrt(out["sq_err"])
    cnt = df.groupby(keys).agg(n_reps=("status", "size"), n_ok=("status", lambda s: int((s == "ok").sum())),
                               true_gamma=("true_gamma", "first"))
    return out.rename(columns={"err": "bias"}).join(cnt).reset_index()


def point_dirs(run_dirs) -> dict:
    """Collect the <experiment>/<point> directories of all run roots by (experiment, point)."""
    out: dict = {}
    for root in run_dirs:
        for d in sorted(Path(root).glob("*/*")):
            if d.is_dir():
                out.setdefault((d.parent.name, d.name), []).append(d)
    return out


def _combine_point(dirs):
    from ra_core import RA_VERSION
    th, mu, sel = load_raw(dirs, with_selection=True, version=RA_VERSION)
    if th.empty:
        return None
    reps = th.drop_duplicates(["experiment", "combo_id", "param_index", "data_index"]).groupby(
        ["experiment", "combo_id"]).size().rename("n_reps").reset_index()
    return th, mu, sel, summarize_theta(th), summarize_mu(mu), summarize_selection(sel), reps


def cmd_combine(a) -> int:
    # Read and summarize point by point (every summary key contains the point; < 1 GB per process).
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    raw = {n: dest / f"raw_{n}.csv.gz" for n in ("theta", "mu", "selection")}
    for f in raw.values():
        f.unlink(missing_ok=True)
    parts: dict = {"theta": [], "mu": [], "selection": [], "reps": []}
    n_th = n_mu = 0
    jobs = [dirs for _key, dirs in sorted(point_dirs(a.runs).items())]
    workers = getattr(a, "workers", 1) or 1
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for res in (ex.map(_combine_point, jobs) if ex else map(_combine_point, jobs)):
            if res is None:
                continue
            th, mu, sel, st, sm_, ss, reps = res
            for n, df in (("theta", th), ("mu", mu), ("selection", sel)):
                df.to_csv(raw[n], mode="a", header=not raw[n].exists(), index=False)
            for n, df in (("theta", st), ("mu", sm_), ("selection", ss), ("reps", reps)):
                parts[n].append(df)
            n_th, n_mu = n_th + len(th), n_mu + len(mu)
            print(f"[combine] {reps.experiment.iloc[0]} {reps.combo_id.iloc[0]}: {int(reps.n_reps.iloc[0])} replicates", flush=True)
    finally:
        if ex:
            ex.shutdown()
    pd.concat(parts["theta"]).to_csv(dest / "summary_theta.csv", index=False)
    pd.concat(parts["mu"]).to_csv(dest / "summary_mu.csv", index=False)
    pd.concat(parts["selection"]).to_csv(dest / "summary_selection.csv", index=False)
    reps = pd.concat(parts["reps"])
    reps["complete"] = reps.n_reps == 1000
    reps.to_csv(dest / "counts_check.csv", index=False)
    n_bad = int((~reps.complete).sum())
    print(f"[combine] theta {n_th} rows, mu {n_mu} rows; {len(reps)} points; {n_bad} points with fewer than 1000 replicates -> {dest}",
          flush=True)
    return 0 if n_bad == 0 else 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Continuous simulation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="runs_ra")
        s.add_argument("--buckets", default=f"0-{N_BUCKETS}")
        s.add_argument("--combos", default=None, help="only design points with these prefixes (comma-separated, e.g. A,B,C)")
        if name == "run":
            s.add_argument("--workers", type=int, required=True)
    c = sub.add_parser("combine")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--dest", default="combined_ra")
    c.add_argument("--workers", type=int, default=1, help="parallel points (< 1 GB per process)")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "verify": cmd_verify, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
